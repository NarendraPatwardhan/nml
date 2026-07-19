# NVFP4 decode performance refactor

Status: current 2.5-fold decode refactor in progress; accepted baseline is
59.611 tokens/s and the whole-model gate is at least 149 tokens/s

This document records the measured cause of GPT-OSS 20B's unacceptable decode
performance on an NVIDIA A40 and defines the corrective architecture. It
replaces earlier hypotheses about product orchestration with node-level CUDA
evidence. Correctness of the current model execution is established; its
performance is not accepted.

The governing principle is simple: compact weights are useful on
pre-Blackwell GPUs only when their decode path remains a bandwidth-oriented
contraction. Expanding a four-bit code through expensive scalar arithmetic, or
recomputing model nonlinearities for every output tile, defeats the reason for
retaining compact storage.

The implementation now follows the architecture below: gate/up owns the one
activation, down consumes only the activated intermediate, E2M1/E4M3FN decode
is exact integer/bitcast arithmetic with one scale load per representation
block, and every `M = 1` ordinary or routed projection selects a compact GEMV
rather than a dead-row matrix tile. The same expert boundary is used by dense
and NVFP4 lowering. Deterministic CPU oracles, generated-TTIR contracts, IR
contracts, artifact-materialization contracts, the complete CPU suite,
CUDA-remote suite, CUDA binary closure, and CUDA package suite all pass
remotely through BuildBuddy. The accepted invocations are recorded in
[`TASKS.md`](./TASKS.md). That was source/CPU/compiler evidence rather than
NVIDIA execution evidence; the subsequent complete A40 loop is recorded below.

The complete A40 loop has now executed the corrected image successfully. The
unprofiled product run sustained 59.611 steady device tokens/second; the
mandatory combined GDB and node-level Nsight Systems run sustained 55.475
steady device tokens/second under profiler overhead. The refactor delivered a
real 6.6-fold gain over the original approximately 9-token/second baseline,
but it has not met the 100-token/second acceptance gate.

## CURRENT: bandwidth-oriented decode and composed execution

The current objective is no longer the historical 100-token/second gate. NML
must deliver at least a 2.5-fold improvement over the accepted unprofiled
59.611-token/second result: at least 149 tokens/second, or no more than 6.72
milliseconds per steady decoded token. The stretch objective is 200
tokens/second, or 5 milliseconds per token.

This target is credible. GPT-OSS touches approximately 2.03 GB of physically
stored active compact weights per token. The accepted result therefore
delivers only about 121 GB/s of useful weight traffic. A 149-token/second
result requires about 302 GB/s: approximately 43% of the A40's published 696
GB/s peak bandwidth. The A40 establishes the magnitude of the defect; it does
not define the implementation. All corrective architecture must apply by
capability to SM75, SM80--SM89, and SM90 rather than recognizing individual
device names.

The post-refactor trace makes the remaining attribution unambiguous:

| Recurring decode work | Cost per token | Share of GPU kernel time |
| --- | ---: | ---: |
| Routed gate/up compact GEMV | 4.19 ms | 23.5% |
| Routed down compact GEMV | 4.01 ms | 22.5% |
| Ordinary Q/K/V/O/head compact GEMVs | 7.12 ms | 39.9% |
| All compact projections | **15.32 ms** | **85.9%** |
| Paged attention and segment reduction | 0.21 ms | about 1.2% |

Attention, KV-cache maintenance, checkpoint validation, token transfer, and
sampling are not primary defects. The implementation must address the compact
projection microarchitecture and execution composition together. Faster
kernels alone eventually expose the historical 26 product-level device
submissions per token; fewer submissions alone cannot overcome kernels
occupying nearly the entire device timeline.

### C.1 Compact `M = 1` is a distinct execution regime

`M` is the number of activation rows participating in a matrix contraction.
A single-sequence decode step normally has `M = 1`: one new token supplies one
activation row, so a projection is a matrix-vector operation. Prefill and
batched decode have `M > 1`: several activation rows reuse the same weights,
so matrix or tensor-core implementations can amortize weight traffic.

This distinction is semantic, not merely a tile-size preference. `M = 1` is
primarily a compressed-memory-bandwidth problem. It needs enough independent
output owners, coalesced packed-weight loads, register-only decode, and a
single reduction per output. Larger `M` may instead favor Triton matrix tiles
and tensor cores. NML must therefore select the compact decode family
explicitly and must never silently route a supported `M = 1` workload through
a generic matrix fallback.

### C.2 Framework correction: a source-owned pre-Blackwell decode family

Replace the latency-sensitive `M = 1` Triton GEMVs with one framework-level
semantic compact-linear operation backed by private architecture families:

- SM75;
- SM80--SM89; and
- SM90.

Native Blackwell NVFP4 will later implement the same semantic operation. Device
names do not participate in planning or cache identities. Unsupported
capability and shape combinations fail during planning rather than taking an
unannounced slow path.

Ordinary weights are physically N-major. Their decode kernel must assign an
output row, or a small fixed group of output rows, to a warp; lanes traverse
contiguous K, decode E2M1 values in registers, reuse each E4M3 block scale,
reduce within the warp, and write the completed output. This exposes one
natural unit of parallelism per output row rather than one program per
64-output tile. It is especially important for the 512-output K/V projections,
which currently launch only eight Triton programs.

Expert weights are physically K-major and require the complementary mapping:
lanes own consecutive output columns while traversing K. The direct decode
path consumes the already-known four expert IDs and routing weights without a
generic sorted-assignment schedule. Gate/up applies paired bias and clamped
SwiGLU once. Down applies routing weights and reduces the four selected expert
outputs before its final store, eliminating the `[4, hidden]` temporary and
the following StableHLO reduction.

Triton remains an intentional matrix path for prefill, continuous batches,
and other `M > 1` contractions. The two paths implement different execution
regimes of the same semantic operation; neither is a fallback for the other.

### C.3 Role- and capability-aware planning

The private compact-kernel plan must include the complete specialization
identity:

- SM family and architectural resource limits;
- SM count where it determines available parallelism;
- physical representation and logical `M`, `N`, and `K`;
- projection role: Q, K/V, O, head, expert gate/up, or expert down;
- warp, output-owner, and reduction geometry; and
- fused epilogue identity.

A small finite set of reviewed kernel variants replaces the current universal
`BLOCK_N = 64`, `BLOCK_K = 128`, four-warp policy. The selected identity is
part of compilation and cache keys. Public tensor APIs remain independent of
CUDA geometry.

The first kernels consume the existing source representation directly. NML
must not expand compact weights persistently to BF16. A versioned prepared
representation may be introduced only if hardware counters show that the
source representation prevents efficient transactions after the output-owner
kernels are in place. Such a representation must be owned by the framework,
accounted as resident prepared bytes, keyed by representation recipe and
kernel ABI, streamed during materialization, and replace rather than duplicate
the source device allocation.

### C.4 Product correction: repeated decode segments

Before this tranche, GPT-OSS submitted embedding, 24 layer executions, and
head execution separately for every decoded token: 26 product-level device
executions. The trace attributed approximately 25 recurring graph launches per
token after its own warmup and capture accounting, plus nearly 50,000
graph-node parameter updates over the profiled run. These costs overlap slow
kernels, but become a hard floor after the compact family is corrected.

Decode must compile a bounded repeated segment that follows the model's
full/sliding layer schedule. A four- or six-layer segment executable is reused
with distinct parameter bundles for each compatible segment. This preserves
bounded compilation and ZML's compile-reuse principle while allowing XLA to
capture the operations inside each segment into one command buffer. The
accepted steady decode path may submit no more than six device executions per
token. A full 24-layer executable is permitted only if it satisfies the same
compilation, reuse, and diagnostic contracts; it is not required by the
architecture.

The boundary remains strict: the GPT-OSS product owns model topology, layer
schedule, and segment construction; NML owns graphs, executable compilation,
arguments, buffers, and device execution. No GPT-OSS topology enters the
runtime or kernel crates.

### C.5 High-value semantic fusion

Once the compact kernels and composed execution are established, backend
patterns should remove recurrent traffic and low-occupancy boundaries:

- fuse Q/K/V compact projection so the activation is read once and the small
  K/V projections share a sufficiently large launch;
- fuse RMS normalization with the consuming Q/K/V projection;
- fuse router projection, softmax, and top-four selection into the direct
  expert schedule;
- fuse routed down projection with expert weighting and reduction;
- fuse final RMS normalization, compact LM head, and streaming top-k so the
  full logits tensor is not materialized; and
- combine residual and normalization boundaries where graph dependencies
  permit it.

These are framework/backend lowering patterns over ordinary semantic NML
operations. GPT-OSS continues to express the model rather than choosing CUDA
kernels. Attention is deliberately excluded from this performance phase: the
trace assigns it only about 0.21 milliseconds per token.

### C.6 Steady-state runtime preparation

After an executable's shapes, parameters, outputs, and aliasing have been
validated, runtime may build a private prepared invocation containing stable
raw argument arrays, parameter-slot mappings, and output bindings. Decode then
updates only changing state or cache buffers. The token result uses persistent
pinned scalar staging and event-based availability. This removes repeated
vector construction and validation without widening the public API or adding
a second eager scheduler.

Global NVFP4 scale is folded into the block scale once per block. Artifact and
representation validation happens before device execution; hot kernels trust
the validated materialization receipt. Invalid artifacts remain hard errors,
but validation arithmetic is not repeated for every decoded weight.

### C.7 Acceptance budget and evidence

| Steady-state component | Maximum budget |
| --- | ---: |
| All compact projections, including head and experts | 5.2 ms/token |
| Attention and KV-cache work | 0.3 ms/token |
| Norms, routing, residuals, and sampling | 0.4 ms/token |
| Submission gaps and device idle time | 0.5 ms/token |
| Token synchronization and download | 0.2 ms/token |
| **Total** | **6.6 ms/token, at least 151 tokens/second** |

The compact family must sustain at least 50--60% of the device's sustainable
memory bandwidth on representative decode projections, rather than meeting an
A40-specific launch configuration. Correctness is established against the
reference lowering. Whole-model evidence is required on available SM8x and
SM90 hardware; SM75 may use representative kernel-level numerical and
performance evidence where the model does not fit.

The next paid GPU evidence loop must retain GDB and Nsight Systems and add
Nsight Compute counters for ordinary Q, K/V, and head projections plus expert
gate/up and down. Required measurements include DRAM throughput, sectors per
request, occupancy, register use, and issue-stall reasons. The current source
inspection strongly suggests under-parallelization and inefficient memory
transactions, but that specific mechanism remains an inference until those
hardware counters are collected.

The implementation order is binding: compact output-owner kernels and direct
top-four MoE first; role-aware planning with them; repeated decode segments;
then semantic fusions and prepared invocation cleanup. The 2.5-fold gate is a
whole-model result and no individual microbenchmark substitutes for it.

### C.8 Current implementation state

The source tree now implements the first complete performance tranche:

- ordinary compact decode is an output-owner CUDA GEMV family for SM75,
  SM80--SM89, and SM90; compiler planning selects four or eight warps from
  output geometry and SM count, and the selected target name participates in
  the executable cache identity;
- Q/K/V is one semantic parallel-linear group. Single-row compact CUDA lowers
  it to one launch and one shared activation stream, while `M > 1` preserves
  three matrix-oriented Triton contractions;
- single-row MoE consumes direct route IDs. The retained-router boundary fuses
  dense router projection, activation-dtype softmax rounding, deterministic
  top-four selection, and renormalization before direct gate/up and down;
- direct down owns route weighting and reduction, so no `[4, hidden]`
  temporary or following StableHLO reduction remains;
- compact LM-head decode streams exact top-64 candidates. The full vocabulary
  logits tensor and general sort remain the portable semantic definition but
  are dead on this CUDA path; XLA owns both bounded merge workspaces;
- GPT-OSS decode compiles six-layer alternating-attention segments and submits
  embedding, four segments, and head: six device executions per token rather
  than 26; and
- runtime argument completeness is O(1), immutable component contracts are
  validated at binding, and token download reuses one host staging slice.

Focused CUDA-configured BuildBuddy gates pass in invocations
`4246944a-2f1a-480e-839a-1cf42b60a638`,
`02c32417-0343-46a3-903e-444cacb2cc38`,
`226115b1-270f-4734-a023-4e64266ed110`,
`b0e16c89-04f0-42da-b286-372f9b2c53bb`, and
`20a6d461-fd93-4637-afaa-05893c0d420f`. The final focused review gate, including
representative-slot rebinding and expert-sharded decode, passes in
`1c25610d-bb61-4c20-b274-b9b0575d4695`. The complete GPU-independent CUDA suite
passes in `a9853052-3b1c-464c-ba1b-e109b608282b`, the full CUDA binary closure
in `ebf5521e-5dd7-4d4e-80be-e788f40a3dcf`, and package/OCI structure contracts
in `e1ddc567-0b9a-4c7d-aedf-a60125c6382e`. These prove source, Rust, MLIR, all
declared CUDA architecture compilation, executable closure, and image
structure; they are not numerical GPU execution or throughput evidence.

RMS-normalization fusion and a lower-level prepared PJRT invocation remain
open. RMS normalization is intentionally not folded into every output-owner
block before counters justify it: doing so can reread the activation and norm
weight once per projection block and increase L2 traffic relative to one small
materialized normalized vector. The next paid profile decides between a
single-producer normalized staging kernel and a genuinely fused contraction.
The whole-model 149-token/second gate also remains open until the immutable
image passes the mandatory GDB, Nsight Systems, numerical generation, and
Nsight Compute evidence loop.

## 0. Post-refactor A40 evidence

The corrected immutable product image is:

```text
ghcr.io/narendrapatwardhan/nml@
sha256:69e805cd5128e9b8d8c7dbe8caf9ae092f985ef6cc19966ebf5c2b345b6b85a0
```

The combined debugger and profiler report is retained outside Git at:

```text
references/runpod/reports/
  20260719T121922Z-aiuvl369ogh26v-69e805cd5128-diagnostic/
```

Nsight Systems launched GDB, which launched the product entrypoint. GDB
reported a normal inferior exit; the product generated 128 sensible tokens;
Nsight produced a 5,520,338-byte `.nsys-rep`, its SQLite export, and all four
required CUDA summaries. The A40 was compute capability 8.6 with 46,068 MiB
reported memory and driver 580.159.04. The paid Pod was terminated only after
the complete attempt directory was validated locally.

The node-level post-refactor attribution is:

| Component | Trace share | Total over 127 decode executions | Approximate cost per token |
| --- | ---: | ---: | ---: |
| Grouped gate/up compact GEMV | 23.5% | 532.0 ms | 4.19 ms |
| Grouped down compact GEMV | 22.5% | 509.1 ms | 4.01 ms |
| Ordinary compact GEMVs | 39.9% | 903.9 ms | 7.12 ms |
| Paged-attention kernels | about 1.2% | 27.1 ms | 0.21 ms |
| Prefill compact matrix kernels | about 7.7% | one execution per layer | not decode recurring |

The recurring compact GEMV floor is therefore about 15.31 ms/token, down from
the original 105.5 ms/token. Gate/up and down are now balanced, and grouped
down no longer repeats SwiGLU for every output tile: the intended semantic
refactor is visible in hardware behavior. The remaining dominant defect is the
microarchitecture shared by the M=1 compact GEMV family, not attention.

The product reports 2,295.674 ms of device decode time for 127 executions, or
18.08 ms/token. The approximately 2.76 ms/token above the compact-GEMV floor
includes attention, graph work, and execution boundaries. Nsight records 3,200
`cuGraphLaunch` calls averaging about 105 microseconds of host API time—roughly
25 graph launches and 2.62 ms of submission time per token. Consequently, a
twofold GEMV improvement alone approaches but does not robustly cross the
100-token/second gate. The next implementation must combine a substantially
more bandwidth-efficient packed GEMV with token-level execution composition;
neither workstream substitutes for the other.

## 1. Evidence identity

The accepted correctness run and the profiler run used the immutable image:

```text
ghcr.io/narendrapatwardhan/nml@
sha256:91ea7415e50b820867c527c1d1c9db1df05bd0e81bdeddd29af8e806d0bdc042
```

The profiled source identity was:

```text
commit: 3a7ea04e419ae42d995510084022d269ff6b047c
dirty source fingerprint:
d42b1d32aba4301b85abe54fb8673e446203557458371ae482b2e882059f1d83
```

The execution device was a single NVIDIA A40, compute capability 8.6, with
CUDA driver/runtime 13.1. The request used:

```text
prompt tokens:       68
generated tokens:   128
prefill capacity:   256
cache capacity:     512
resident weights:   11,777,751,752 bytes
KV-cache storage:   25,165,824 bytes
```

The complete node-level Nsight Systems report is retained outside Git in:

```text
references/runpod/reports/
  20260719T082602Z-mnva1mt0kk4ec9-91ea7415e50b-nsys/
```

The directory contains the immutable attempt description, product output,
profiler log, `.nsys-rep`, exported SQLite database, CUDA kernel summary,
CUDA API summary, and kernel-launch summary. The paid profiling Pod was
terminated after the report was collected.

## 2. Conclusion

Nine tokens per second is not an A40 hardware limit, an attention problem, or
a long-context spill. NML's current NVFP4 kernels consume essentially the
entire decode budget.

The node-level trace assigns GPU kernel time as follows:

| Component | Trace share | Total over 128 model executions | Approximate cost per execution |
| --- | ---: | ---: | ---: |
| Grouped down projection | 61.9% | 8,876.2 ms | 69.3 ms |
| Grouped gate/up projection | 21.3% | 3,054.5 ms | 23.9 ms |
| Ordinary NVFP4 linears | 15.8% | 2,262.7 ms | 17.7 ms |
| Paged attention kernels | about 0.2% | 26.9 ms | about 0.2 ms |
| All remaining GPU work | less than 1% | remainder | less than 1 ms |

The totals include one prefill execution, so medians are more representative
of steady decode. Their result is the same:

```text
grouped down:       2.856 ms/layer * 24 = 68.5 ms/token
grouped gate/up:    0.932 ms/layer * 24 = 22.4 ms/token
ordinary linears:                         14.6 ms/token
                                                    --------
NVFP4 median floor:                      105.5 ms/token
```

That floor predicts approximately 9.5 tokens/second before minor graph work,
which agrees with the product measurement. The trace therefore explains the
observed throughput rather than merely correlating with it.

The model touches approximately 2 GB of compact active weight data per token.
At nine tokens/second, effective useful weight bandwidth is only about
18 GB/s, roughly 2.6% of the A40's approximately 696 GB/s peak bandwidth.
Reaching 100 tokens/second would require roughly 200 GB/s, or about 29% of
peak. That is an aggressive but credible target for purpose-built packed
decode kernels; the current result is not.

## 3. Primary defect: SwiGLU is fused at the wrong boundary

The grouped gate/up projection writes interleaved gate and up channels. The
grouped down kernel then loads those channels and applies GPT-OSS clamped
residual SwiGLU inside its K reduction loop.

The down projection has 2,880 outputs and uses a 64-column output tile. It
therefore has 45 output tiles. Each output tile independently reloads the same
gate/up activation and recalculates the same SwiGLU values. Activation work
that belongs once per routed intermediate element is repeated about 45 times
per routed expert.

This misplaced fusion also creates extreme register pressure. `ptxas` reports
the following for `nvfp4_grouped_down`:

```text
720 bytes spill stores
688 bytes spill loads
```

This explains the otherwise impossible relationship between the two expert
projections: down reads about half as many compact weights as gate/up, yet a
steady down launch takes roughly three times as long.

### Required replacement

Clamped residual SwiGLU belongs in the gate/up epilogue:

```text
hidden
  -> grouped compact gate/up contraction
  -> add paired gate/up biases
  -> apply clamped residual SwiGLU once per pair
  -> store [assignments, intermediate] activated values
  -> grouped compact down contraction
  -> apply routing weights and down bias
```

The gate/up kernel already owns both paired channels. Its output tile must pair
the interleaved gate/up accumulators, apply the exact activation once, and
write only the activated intermediate width. The down kernel must accept that
ordinary activated tensor and contain no activation transcendental.

NML's SM75 CUDA adapter already implements this boundary: its gate/up kernel
accumulates paired channels, adds their biases, applies clamped residual
SwiGLU once, and stores `[assignments, intermediate]`; its down kernel consumes
that activated tensor. The SM8x Triton path diverged from the established
design. The refactor therefore converges the Triton lowering on the proven
SM75 semantic boundary rather than inventing a third expert interface.

This is not an optional fusion experiment. It restores the model's natural
semantic boundary, halves the gate/up intermediate storage, removes repeated
work, and eliminates the register-spilling down-kernel composition.

## 4. Second defect: quantized values are decoded with transcendental math

The profiled E2M1 decoder derived every unpacked weight using dynamic floating
point arithmetic and `exp2`. E2M1 has exactly sixteen bit patterns. It does not
require a transcendental function.

The profiled E4M3FN scale decoder also derived scale values through `exp2`.
Worse, the one-per-16 scale is loaded into a full `[block_k, block_n]` weight
tile, so the same scale is loaded and decoded repeatedly for each of its
sixteen associated weight lanes.

Every dense and grouped NVFP4 contraction pays this cost. The expert kernels
alone process billions of active logical weights per token, turning a compact
bandwidth workload into an instruction-heavy conversion workload.

### Required replacement

- Add typed integer bitcast support to the private Triton builder.
- Decode normal E4M3FN values by constructing the exact IEEE/BF16 exponent and
  mantissa bits; handle the small subnormal domain through a fixed exact map.
- Decode E2M1 through an exact sixteen-entry mapping or equivalent integer bit
  construction. No `exp2`, logarithm, division, or general exponentiation may
  occur in weight decoding.
- Load and decode one block scale for sixteen values, then broadcast it within
  the register tile.
- Preserve the independent scalar codec as the numerical oracle.
- Add TTIR/generated-code contracts that reject `math.exp2` in E2M1 and E4M3
  weight decoding. The real SwiGLU approximation may still use its declared
  exponential in the gate/up epilogue.

## 5. Third defect: decode uses a generic matrix tile instead of compact GEMV

Autoregressive decode is `M=1`. The retained Triton linear family promotes it
to `block_m=16` so it can feed an ordinary BF16 tensor-core dot. Fifteen rows
are masked, but the contraction still performs the tile's arithmetic and
carries the associated register state.

That can be a reasonable correctness path or a prefill path. It is not the
final decode design. The enormous 201,088-row output head and the repeated
ordinary projections make the dense compact path alone cost approximately
15--18 ms/token.

### Required replacement

NML needs finite, source-owned compact GEMV families for decode:

- ordinary `[N, K]` projection, including the vocabulary head;
- routed gate/up projection with paired epilogue activation; and
- routed down projection over already-activated intermediates.

Each decode kernel must:

- map warps to useful output rows/columns without fifteen dead logical rows;
- vector-load aligned packed payload and block scales;
- unpack and scale only in registers;
- reuse each decoded scale across its complete 16-value block;
- accumulate in F32 and preserve the declared BF16 output semantics;
- consume only selected experts; and
- expose no full dequantized intermediate.

The existing tensor-core family remains available for prefill shapes where M
is large enough to reuse decoded weights. Decode and prefill must not be forced
through one geometry merely to reduce the number of internal kernel types.

If the source artifact layout prevents aligned/coalesced decode loads, the
loader may create one immutable, versioned, device-prepared layout after
upload. Preparation must be representation-aware, included in identity and
accounting, verified against the scalar oracle, and release the obsolete
source device buffers. Per-token repacking and persistent BF16 expansion are
forbidden.

## 6. Product orchestration is real but secondary

The CUDA API trace records 3,200 `cuGraphLaunch` calls over 128 complete model
executions: exactly 25 graph launches per execution. They correspond to the 24
layer executables and the head; compact embedding is launched separately.

This is a future latency floor, but it is not the present 12x--25x loss. GPU
kernel execution already accounts for the measured approximately 110 ms/token.
Public llama.cpp measurements also retain hundreds of tokens/second with their
fusion and graph optimization disabled, showing that orchestration changes
alone explain a modest fraction rather than this collapse.

After the compact kernels approach their bandwidth target, compile/capture one
token-level execution containing embedding, all layers, cache updates, head,
and sampling state. It should reuse framework model construction and parameter
bindings; it must not introduce a second eager scheduler or move product model
semantics into the runtime.

The implementation order matters. Combining the present kernels into one
larger executable would make a 9-token/second path slightly less bad while
hiding the actual defect.

## 7. What is not responsible

### Attention

Paged attention and its segment reduction together consume about 0.2% of GPU
kernel time at this request shape. Flash/paged attention work cannot recover
the missing order of magnitude.

### Context length or host spill

The profiled request used cache capacity 512, 25 MB of KV-cache storage, and a
fully GPU-resident 11.8 GB checkpoint. The roughly 9-token/second public failure
mode at 128k context is caused by system-memory/PCIe spill. It is unrelated to
this run.

### Sampling, sorting, or token download

Sorting and sampling kernels are individually measured in microseconds. Token
downloads consumed less than one millisecond per token and do not explain the
GPU-resident delay.

### Artifact validation

Full launch-time rehashing consumed approximately 225 seconds. It was a serious
startup lifecycle defect, but it was outside steady decode. The artifact
materializer now authenticates the pinned manifest, hashes every payload once,
makes the materialization read-only, and atomically issues a receipt containing
the exact filesystem identity of each verified file. Normal launch hashes only
the small manifest and compares the receipt with cheap current metadata.
Missing or stale proof hard-fails instead of triggering an implicit full scan.
This removes checkpoint-size-dependent revalidation from startup; it does not
change steady tokens/second.

## 8. Verification ladder

The refactor is accepted in ascending scope. Compile and CPU execution remain
remote through BuildBuddy according to the repository workflow. GPU execution
uses the appropriate real device.

### 8.1 CPU semantic contracts

- Exhaust every E2M1 code and relevant E4M3FN scale class.
- Compare the new integer/lookup decode to the independent scalar codec.
- Compare gate/up activation and down output against the existing analytic
  reference for fixed and randomized bounded shapes.
- Verify padding, nibble order, scale sharing, bias order, route weighting, and
  BF16/F32 tolerance.

### 8.2 Compiler and semantic acceptance before GPU rental

- Compile both the SM75 CUDA adapter and SM80+ Triton path through BuildBuddy.
- Reparse and verify every generated TTIR module through `KernelSpec`.
- Reject `math.exp2` in quant decoding, reject `tt.dot` in `M = 1` kernels,
  require F32 reductions, and require exactly one activation in gate/up and no
  activation in down.
- Compare fixed and deterministic randomized CPU shapes with the independent
  scalar representation oracle, including odd widths, empty experts, uneven
  routes, bias order, route weighting, nibble order, and scale sharing.
- Compile the complete CUDA product and device-contract binaries without
  claiming that a BuildBuddy worker executed them on NVIDIA hardware.

The local SM75 adapter is a different implementation from the A40 Triton path.
Running a reduced SM75 block would neither predict the Triton speedup nor be a
better acceptance unit than the complete A40 model. It is therefore not part
of this paid A40 performance loop. Its source remains compile-gated and its
next runtime change requires separately authorized real-device evidence.

### 8.3 End-to-end A40 acceptance

- Run the unchanged immutable OCI image and exact verified checkpoint.
- Run the complete deterministic compact-operation corpus through actual SM86
  dispatch before the full model, in the same Pod and image.
- Separate validation, compilation, upload, prefill, first decode, and steady
  decode.
- Profile M=1 ordinary, gate/up, and down nodes within the complete model;
  confirm no quant-decode transcendental, no grouped-down spill, and useful
  compact bandwidth rather than relying on host timing.
- Compare representative prefill nodes so the retained matrix family has not
  silently regressed.
- Require numerically sensible Harmony output under the declared sampling
  configuration.
- Require at least 100 steady tokens/second at short context before further
  serving features are prioritized.
- Retain the complete profiler report and exact image/source identity.

The 100-token/second threshold is a milestone gate, not a claim that it is the
final ceiling. Once it is met, token-level execution capture and further
layout/kernel tuning should pursue the device's truthful bandwidth limit.

## 9. Implementation sequence

1. [implemented] Refactor the semantic expert boundary so gate/up produces the activated
   intermediate and down consumes it.
2. [implemented] Implement exact non-transcendental E2M1/E4M3FN kernel decoding and one-scale-
   per-block reuse.
3. [implemented] Add CPU numerical and SM75/SM8x compiler contracts around the
   new expert boundary; real SM8x evidence is the complete A40 loop.
4. [implemented] Add dedicated M=1 compact GEMV families for ordinary and routed projection.
5. [not indicated] Introduce a versioned prepared device layout only where profiling proves it
   is required.
6. [executed; gate unmet] Rebuild the immutable product image and require the
   end-to-end A40 gate. Correctness passed at 59.611 unprofiled steady device
   tokens/second; the required 100 tokens/second did not.
7. [post-gate] Compose/capture token-level execution after the kernels no longer dominate.
8. [implemented] Replace launch-time full hashing with trusted artifact
   materialization and a bounded immutable receipt.

Every step must retain one semantic graph surface and capability-selected
private lowerings. There will be no slow generic fallback presented as
acceleration, no benchmark-only backend override, and no duplicate product
scheduler introduced to work around a framework boundary.

## 10. Definition of done

This refactor is complete only when all of the following are true:

- clamped residual SwiGLU is evaluated once per routed intermediate element;
- grouped down contains no SwiGLU exponential and no material register spill;
- E2M1/E4M3FN weight decoding contains no transcendental arithmetic;
- each block scale is decoded once and reused for its 16 values;
- M=1 uses a purpose-built compact decode family;
- CPU and rented SM8x results match the independent oracle; the modified SM75
  adapter is never counted as runtime evidence until it is separately run;
- the exact product image sustains at least 100 tokens/second on A40 at short
  context;
- the profiler shows attention and orchestration as measured secondary costs,
  rather than assumptions; and
- all measurements retain source, image, model, device, phase, and report
  identity.

## 11. External comparison

The cited RunAIHome compilation reports 111--225 tokens/second on consumer
NVIDIA GPUs at short context, while explaining that approximately 9
tokens/second is a 128k-context host-spill failure mode:

- <https://runaihome.com/blog/gpt-oss-20b-local-ai-hardware-guide-2026/>

More directly, llama.cpp reports GPT-OSS 20B MXFP4 at 232 tokens/second on an
RTX 4090 even with fusion and graph optimization disabled, and approximately
272 tokens/second with them enabled:

- <https://github.com/ggml-org/llama.cpp/discussions/17621>

Those implementations use different storage recipes and hardware, so their
numbers are comparison points rather than NML acceptance evidence. They do,
however, rule out the claim that four-bit GPT-OSS inherently runs near nine
tokens/second on pre-Blackwell NVIDIA devices.
