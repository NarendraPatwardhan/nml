# NVFP4 decode performance refactor

Status: accepted architecture restored manually, accepted by the complete
BuildBuddy CUDA gates, and reconfirmed by a fresh A40 baseline

This document records the measured causes of GPT-OSS 20B decode performance
on an NVIDIA A40 and the accepted corrective architecture. It replaces
hypotheses with whole-model CUDA evidence. Numerical correctness is
established; performance remains below the product target.

The governing principle is simple: compact weights are useful on
pre-Blackwell GPUs only when their decode path remains a bandwidth-oriented
contraction. Expanding a four-bit code through expensive scalar arithmetic, or
recomputing model nonlinearities for every output tile, defeats the reason for
retaining compact storage.

The accepted architecture puts clamped SwiGLU in gate/up, gives down only the
activated intermediate, performs exact register-local E2M1/E4M3FN decode with
block-scale reuse, and separates compact M=1 GEMV from matrix-shaped prefill.
Its immutable A40 run sustained 59.611 steady device tokens/s unprofiled and
55.475 under the mandatory combined GDB/Nsight harness, a real 6.6-fold
improvement over the original approximately 9-token/s baseline. BuildBuddy
invocations and the durable A40 evidence are recorded in
[`TASKS.md`](./TASKS.md) and below.

## Failed composed-decode/direct-kernel experiment

The follow-on output-owner/direct-kernel approach is rejected. Immutable image
`sha256:c10b80d8dd511e2e7b24e914a73deba64ccab37f150ea74f04afa3bce24fe6c3`
from source commit `355863cb1c88b454fcbeac949d64f20db13b16f9` completed
a coherent 320-token GPT-OSS generation on an A40 under the combined
Nsight-Systems-over-GDB harness, but sustained only 5.601 steady device
tokens/s and 5.583 decode-loop tokens/s. That is a 9.9-fold regression from
the preceding profiled baseline.

| Failed kernel | Total GPU time | Share | Mean launch |
| --- | ---: | ---: | ---: |
| Direct expert down | 41.283 s | 73.4% | 5.392 ms |
| Direct expert gate/up | 10.519 s | 18.7% | 1.374 ms |
| Streaming linear top-64 | 2.741 s | 4.9% | 8.566 ms |

Direct expert gate/up and down consumed 92.1% of GPU kernel time; including
streaming head top-k accounts for 97%. Attention and orchestration were not
the cause. The durable report is
`references/runpod/reports/20260719T152709Z-lm4xqsqg7we5ym-c10b80d8dd51-diagnostic`.

Compile and structural contracts did not validate this performance claim.
Publishing the complete tranche before its first A40 measurement allowed bad
kernel assumptions to compound. Output-owner ordinary GEMV, direct top-four
expert kernels, streaming head top-k, six-layer decode segments, and their
sole-purpose semantic APIs have therefore been manually removed together.
They are not retained as dormant alternatives or hidden fallbacks. The
BuildBuddy-only publication procedure and combined GDB/Nsight evidence policy
remain because they are independent infrastructure.

The restored source passed the complete remote CUDA contract suite in
BuildBuddy invocation `61f0b083-80e5-40a7-8d91-bb4dfd80c4a6`, package and
image contracts in `7287d013-ed87-44cc-a42c-9897fb1d1e1d`, and the full CUDA
binary plus serving-image closure in `f2a46248-ea91-44c2-8866-3b3d833c0219`.
These results prove restoration and construction only. The subsequently
published image resolved to the accepted immutable digest
`sha256:69e805cd5128e9b8d8c7dbe8caf9ae092f985ef6cc19966ebf5c2b345b6b85a0`.
A fresh run of that digest from restored source commit
`6f8dd0b222721a3ecd0a501e035192cd2b400ef4` then completed normally on A40
under the combined GDB/Nsight harness. It generated 128 coherent tokens and
sustained 57.248 steady device tokens/s, 56.398 overall device-decode tokens/s,
and 54.504 decode-loop tokens/s. The complete validated report is
`references/runpod/reports/20260719T155954Z-fdmcvpur8oks3p-69e805cd5128-diagnostic`.
The paid Pod `fdmcvpur8oks3p` was terminated and deletion confirmed after the
report was collected. This fresh result is consistent with the prior profiled
55.475-token/s measurement and conclusively excludes the rejected
5.601-token/s architecture from the restored tree.

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
