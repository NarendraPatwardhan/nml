# NML product contract

Status: current architecture, measured progress, and ordered product roadmap

This is the single product document for NML's GPT-OSS 20B NVFP4 inference
server. It replaces the former `NVFP4.md`, `REQUESTS.md`, and `NEXT.md`.
[`SYSTEM.md`](./SYSTEM.md) governs framework architecture and
[`TASKS.md`](./TASKS.md) is the executable checklist. This file explains what
the product is, what has actually been proved, and which major capabilities
come next.

Git history is the implementation archive. This document deliberately omits
superseded experiments unless they explain a current design decision or a
measured result.

## 1. Product objective

NML will be a production-shaped, OpenAI-compatible inference server for one
exact GPT-OSS 20B NVFP4 artifact. It must:

- retain the checkpoint in compact NVFP4 form;
- run efficiently on pre-Blackwell NVIDIA GPUs, beginning with the A40;
- retain a truthful future native-Blackwell path for the same logical recipe;
- serve multiple independent clients through continuous batching and chunked
  prefill;
- own one process-wide paged K/V arena;
- preserve fast interactive batch-one decode while increasing useful
  aggregate throughput under concurrency;
- stream text and strict Harmony tool calls without executing user tools;
- add automatic prefix caching, real tensor parallelism, and lossless DFlash
  speculative decoding in that order; and
- report end-to-end behavior separately from kernel/device diagnostics.

The product is not complete because a model can generate one answer or because
a kernel benchmark is fast. Completion requires a long-running bounded server,
correct request lifecycle and cache ownership, stable performance under real
arrival patterns, and permanent evidence from the hardware path being claimed.

## 2. Current status at a glance

The latest measured serving implementation is commit
`9f75e94c8eb4f3d3551c914ea7f33187cbe6d598`. Its immutable serving image is:

```text
ghcr.io/narendrapatwardhan/nml@sha256:00aaed758f68c51c679426805147f094271982e3e1563bb3580e187fed02172d
```

The A40 server-load report
`20260723T131209Z-wvlc4ygb9ajxck-00aaed758f68-server-load` succeeded with the
106-token prompt, 320 generated tokens, two measured repetitions, and
concurrency `C={1,2,4,8}`. The report includes the complete Nsight Systems
capture and SQLite export.

A later convergence experiment at source
`652cd2589e445a8adf015b4be639098099e43809`, image
`sha256:552b27eb1ccc2a1493b80ac76c13ea1550f522a4ac060bad1b18087181c85911`,
was measured and rejected. It does not supersede the accepted result above.

The migration control remains source
`fb415a8dadd51a0053b9be314faa836e2b274721`, image
`sha256:3c81704ea85512df7ff76de83ea21f403ef592dc50c23c5ae20e8d70c1e7f3ff`,
and the same 106+320 workload:

| Legacy diagnostic boundary | Accepted result |
| --- | ---: |
| Steady device | 156.062 TPS |
| Device decode | 155.644 TPS |
| Complete decode loop | **151.324 TPS** |

This remains historical evidence for the old diagnostic route, not a license
to preserve duplicate orchestration indefinitely.

### 2.1 Current measured A40 result

| Concurrency | Selected steady decode family | End-to-end output TPS | Decode-engine row TPS | Maximum p95 TTFT |
| ---: | --- | ---: | ---: | ---: |
| C1 | B1 | 135.741-136.851 | **150.08** | 0.242 s |
| C2 | B2, apart from boundary B1 steps | 119.907-120.053 | 130.35 | 0.501 s |
| C4 | B4, apart from one boundary B1 step | 166.955-167.105 | 181.58 | 0.704 s |
| C8 | B8, apart from one boundary B1 step | 205.226-205.687 | 223.85 | 1.140 s |

`Decode-engine row TPS` is derived from the engine's measured decoded-row sum
divided by measured decode-batch seconds. `End-to-end output TPS` includes
prompt work, first-token latency, decode, server control, and client-visible
completion. Neither number substitutes for the other.

The immediate batch-one objective was to recover the former approximately
150-TPS decode loop without creating a private B1 engine. That objective is
now met by the generic serving lane. Full C1 request throughput remains about
136 TPS, so another roughly 10% of request-level cost must be removed before
claiming 150 end-to-end TPS.

C8 now produces 1.51x C1 aggregate output throughput. This proves useful
concurrency scaling for the control workload, but it is not final production
acceptance: C2 remains slower in aggregate than C1, per-request latency grows
substantially, and the present matrix does not cover mixed prompt lengths,
arrival rates, or long-prompt interference.

### 2.2 Improvement from the preceding server image

| Concurrency | Previous aggregate TPS | Current aggregate TPS | Change |
| ---: | ---: | ---: | ---: |
| C1 | 129.846 | 136.296 | +5.0% |
| C2 | 47.334 | 119.980 | +153.5% |
| C4 | 78.427 | 167.030 | +113.0% |
| C8 | 121.459 | 205.457 | +69.2% |

The change came from fixing generic small-batch execution rather than adding a
special server route:

- B1/B2/B4/B8 sparse expert decode uses one compact selected-route GEMV block
  per route instead of a mostly empty 16-row grouped matrix tile;
- B2/B4/B8 Q, K, and V projections use one fused, weight-tile-major,
  row-minor Triton launch;
- inactive route slots retain expert ID `-1`, allowing the kernel to skip them
  without per-layer scan/scatter compaction; and
- B16 and larger remain on the grouped matrix family, where multiple rows per
  expert can provide real reuse.

The principal gate/up, down, and QKV kernel time per layer changed
approximately as follows:

| Family | Previous combined time | Current combined time |
| --- | ---: | ---: |
| B1 | 174 us | 162 us |
| B2 | 1,458 us | 269 us |
| B4 | 1,702 us | 521 us |
| B8 | 2,228 us | 1,043 us |

This is the relevant proof that the small-batch gain is kernel alpha rather
than benchmark or scheduler manipulation.

### 2.3 Rejected generalized small-M experiment

The rejected `552b27eb1ccc` A40 run produced approximately
106.1/122.8/134.6/139.8 aggregate output TPS at C1/C2/C4/C8. Relative to the
accepted control, C1 fell 22.5%, C4 fell 19.5%, and C8 fell 32.0%; C2 improved
only 2.3%. The complete trace makes the cause unambiguous:

- measured decode-device time increased from 76.25 to 101.35 seconds;
- ordinary linear families added approximately 19.65 GPU-seconds;
- QKV added approximately 6.83 GPU-seconds; and
- those two changes explain the complete approximately 25.10-second increase.

One generalized small-M GEMV helper caused both failures. It broadened the
proven scalar B1 contraction from rank-one TTIR to a leading-M tensor and sent
dense B2/B4/B8 projections—including the 201,088-row LM head—through a
row-wise algorithm. The LM-head GEMV alone consumed 28.84 GPU-seconds. The M2
QKV body consumed 13.14 GPU-seconds and total QKV time rose from 7.53 to 14.37
seconds. Selected-route expert gate/up and down time remained essentially
flat, so sparse expert GEMV is retained.

## 3. Terms and product metrics

- **C (concurrency)** is the number of client requests currently in flight.
- **B (batch family)** is the smallest compiled GPU row capacity that can hold
  the scheduled rows.
- **Q (query family)** is a compiled token capacity per row for prefill or
  verification.
- **Active rows/tokens** are useful positions inside the static `B*Q`
  rectangle.
- **TTFT** is time to first visible token.
- **TPOT** is time per output token after the first token.
- **Aggregate output TPS** is total visible output tokens divided by wall time
  for the workload.
- **Decode-engine row TPS** is decoded active rows divided by time recorded for
  decode submissions. It is a diagnostic of the decode machinery, not the
  complete request result.

For example, three decoding clients select B4 and leave one inactive row. A
106-token prompt should use the smallest retained Q family that can represent
the scheduled chunk. Inactive rows and padded query positions must do no cache
writes and no expert weight work.

```text
client arrivals
      |
      v
bounded admission and continuous scheduler
      |
      +-- decode rows first
      |
      `-- aged chunked-prefill work from the remaining token budget
                         |
                         v
              smallest compiled B/Q family
                         |
                         v
       embedding -> 24 layers -> head/sampling
                         |
                         v
              process-wide paged K/V arena
```

Performance claims always name the measurement boundary. A fast decode-only
number cannot conceal slow prefill or host orchestration, and a high aggregate
number cannot conceal unacceptable individual TPOT.

## 4. Governing product architecture

There is one serving architecture across every retained family:

```text
Tokio/Axum control plane
        |
        | bounded commands and per-request events
        v
one dedicated engine OS thread
        |
        +-- admission/reservations
        +-- continuous scheduler
        +-- compiled family registry
        +-- resident target parameters
        +-- global page manager and device K/V arena
        `-- request-local Harmony/sampling state
```

Tokio owns sockets, timers, cancellation, signals, and bounded channels. The
engine thread constructs and destroys all PJRT/XLA objects and is the sole
owner of model execution and cache mutation. A Tokio worker never compiles or
executes a model graph.

Static graph specialization is allowed. It must not create another scheduler,
cache owner, request lifecycle, or B1-only engine. Membership changes select a
new smallest family at a visible-token boundary and continue over the same
process-wide page arena.

The hot stable-decode transaction is:

```text
device-resident token/RNG/position/length/page metadata
        |
        v
embedding -> nine queued layer pairs -> suffix -> head
        |                                      |
        |                                      +-- compact token/RNG D2H
        |                                      |
        `--------------------------------------`-- donated next batch slab
                                                       |
                                                       v
                                             next prefix queued
```

Stable membership has zero ordinary H2D transfers and one compact
`B*20-byte` D2H result per visible step. Host re-entry is required only for a
command, cancellation, deadline, shutdown, backpressure, terminal row,
physical-page extension, or membership change.

## 5. NVFP4 representation contract

### 5.1 Recipe, not scalar dtype

NVFP4 is a quantization recipe:

```text
x = x_e2m1 * s_block * s_global
```

- `x_e2m1` is a packed four-bit E2M1 payload value;
- `s_block` is an E4M3 block scale shared by 16 consecutive values; and
- `s_global` is the artifact's global factor.

Two payload values occupy one byte. The compact lower bound is 0.5625 bytes
per logical weight before padding, alignment, metadata, and global factors.

NVFP4 is not added to the ordinary `DType` enum. It is one logical
`Parameter` backed by validated payload, scale, and global-factor components.
PJRT sees ordinary physical buffers; only representation-aware loading,
sharding, and lowering may interpret them.

MXFP4, GGUF Q4_K_M, generic W4A16, and W8A8 are different recipes. They are
not compatibility aliases and are not prerequisites for useful pre-Blackwell
NVFP4 execution.

### 5.2 Canonical artifact identity

The selected GPT-OSS artifact fixes:

- tensor names and logical shapes;
- output-major, K-contiguous recipe-v2 contraction layout;
- nibble order and E2M1 behavior;
- block size and block axis;
- E4M3 variant and scale direction;
- global-factor algebra;
- source transpose and expert axis order;
- edge padding and alignment; and
- conversion and immutable materialization identity.

Ordinary projections use logical `[N,K]`; expert gate/up uses `[E,2I,K]`;
expert down uses `[E,H,I]`. Source expert tensors are transposed during
materialization. Runtime repacking, a second persistent prepared-weight copy,
recipe-v1 compatibility, and persistent BF16 expansion are not part of the
product.

Unknown manifests, ambiguous component extents, unsafe shard boundaries, or
unsupported device/layout combinations fail with a named diagnostic. There is
no heuristic full-model dequantization fallback.

### 5.3 Parameter and loading boundary

The logical/runtime split is:

```text
Parameter
  logical shape, representation and stable component identities

LoadedParameter
  resident physical component buffers for that Parameter

Tensor
  ordinary executable graph value with one ordinary dtype
```

One structural `ParameterTree` traversal loads both dense one-component
parameters and NVFP4 multi-component parameters. Executable manifests flatten
and validate the physical bindings. Quantized parameters do not travel through
a separate model stack.

Physical sharding must co-shard payload and scale spans from logical tensor
partitions. A shard boundary cannot split a 16-value scale block unless the
artifact explicitly defines correct padding and reconstruction.

### 5.4 Hardware dispatch

| Device | Required execution identity |
| --- | --- |
| CPU | Exact compact-weight implementation; no persistent dense copy |
| SM75 | Fused compact W4A16 emulation path |
| SM8x, including A40 | Fused Triton W4A16 emulation path |
| SM90 | Fused compact path; FP8-assisted alternatives only after evidence |
| SM100+ | Native block-scaled NVFP4 where supported; named fallback otherwise |

Only the SM100+ path may be described as native NVFP4. Pre-Blackwell paths
unpack and scale tiles inside the contraction kernel, retaining the compact
resident-memory and bandwidth benefit.

The current pre-Blackwell vertical includes compact embedding/output
projection, ordinary projection, fused QKV, and grouped/sparse expert
gate/up/down execution. The remaining NVFP4-specific work is:

- optimized and measured CPU x86-64 plus retained AArch64 evidence;
- current-source real SM75 execution evidence;
- native SM100 lowering and instruction proof;
- complete independent end-to-end artifact-oracle comparison;
- full phase-separated memory reporting; and
- physical model-specific TP=2/4 CUDA sharding evidence.

### 5.5 Hardware evidence

The representation and pre-Blackwell lowering are not compile-only:

- the full compact contract image
  `sha256:17040fd252bac543bb3b02e9abc253d309d05a7b64cf6ee7b8c6cc8b64c426b4`
  passed on an RTX A6000 (SM86) and an H100 80GB HBM3 (SM90a);
- GPT-OSS-sized projection and grouped-MoE phase measurements were retained
  from image
  `sha256:4f4619f040bd8c59549b90e5b3606c930bc063389aee4e280a25853c15fdf0ff`;
- the current A40 (SM86) product image and report prove complete target-model
  execution, generic continuous batching, and the small-batch Triton paths;
- CPU codecs and compact operations provide the independent semantic
  implementation; and
- current-source SM75 plus native SM100 execution remain unproved on their
  respective physical devices.

Evidence from an older image establishes the path at that source revision; it
does not silently promote later source. Every future device claim pins its own
commit, image, artifact, and report.

## 6. Implemented inference-server foundation

The current tree provides:

- Tokio/Axum/Tower OpenAI chat serving with bounded admission;
- streaming and non-streaming responses, cancellation, readiness, metrics,
  structured errors, and graceful shutdown;
- one dedicated engine owner for PJRT, parameters, executables, pages, and
  scheduling;
- compile-before-residency component executables and reusable bindings;
- a finite parity profile with `B={1,2,4,8}` and
  `Q={16,128,256}`;
- continuous decode-first batching and chunked prefill;
- dynamic row repacking into the smallest retained family;
- one global 16-token paged target K/V arena with arbitrary page tables;
- reservations, page checkpoints, tentative append, commit, rollback,
  truncate, replay, cancellation release, and exact host accounting;
- paired Triton K/V append on CUDA, resolving each physical page once;
- batched explicit-state top-k/temperature/top-p/min-p sampling;
- inactive-row preservation for cache, positions, RNG, and output sentinels;
- mask-aware MoE with inactive expert ID `-1`;
- compact serving slabs and donated stable-batch state;
- nine-layer-pair generic lookahead;
- direct stable continuation without ordinary scheduler re-entry;
- fused compact QKV and selected-route expert GEMV through B8;
- process-lifetime result workspaces per family;
- package-private GPT-OSS Harmony rendering and strict incremental parsing;
  and
- structured tool declarations, tool-call output, and tool-result history
  without server-side tool execution.

The current tree does not yet provide:

- automatic prefix lookup/refcounts/eviction;
- physical GPT-OSS TP=2/4 execution on CUDA;
- a DFlash artifact, draft graph, verifier, draft cache, or scheduler policy;
- the final production B/Q envelope;
- complete load/failure acceptance; or
- deletion of the remaining diagnostic generation route.

## 7. What worked

The successful performance path was cumulative:

1. Recipe-v2 output-major/K-contiguous compact weights fixed the fundamental
   pre-Blackwell memory access pattern.
2. Direct expert GEMV avoided materializing dense weights and matched sparse
   GPT-OSS decode geometry.
3. Fused QKV removed three repeated compact-weight projection launches at B1.
4. Correct layer lookahead kept GPU work queued while the host handled a
   visible token.
5. The serving ABI collapsed fourteen H2D and three D2H operations into one
   initial slab upload and one compact result download.
6. Device slab donation removed steady H2D entirely during stable membership.
7. Paired K/V append removed decomposed mask/index/scatter launch clusters.
8. Fixed sparse masked schedules removed per-layer scan/scatter rebuilding.
9. O(1) page commit and metadata rebuild only on membership changes reduced
   host work and parameter rebinding.
10. Small-batch fused QKV and selected-route expert GEMV restored efficient
    B2/B4/B8 geometry.

The result is one generic batching data plane with legacy B1 decode parity and
large small-batch gains.

## 8. What did not work

The following are rejected or unpromoted:

- Recipe-v3 variants that fell to roughly 60-70 TPS.
- Two-pass split-K without atomics.
- Narrow tile rebalancing that reduced occupancy or starved SMs.
- Whole-transformer StableHLO monolith execution.
- A private single-sequence serving lane with state export/import.
- Per-token scheduler re-entry and batch reconstruction.
- Treating fourteen small transfers as unavoidable model work.
- Running B2/B4/B8 through prefill-shaped matrix tiles.
- Splitting Q/K/V for small decode batches.
- Per-layer sparse-route compaction when the route schedule is already fixed.
- Claiming improvement from compilation without immutable device execution.
- Using decode-only throughput as the end-to-end product score.
- Using future prefix, TP, or DFlash gains to conceal current machinery cost.

Single-kernel atomic split-K remains a possible research direction, but the
current evidence does not justify expanding TTIR atomic support before
trace-correlated bottlenecks with lower implementation risk.

## 9. Current efficiency phase

The current phase is not DFlash, prefix caching, or tensor parallelism. It is
to finish the target-model server machinery and converge on one route.

### 9.1 Achieved

- Generic B1 decode-engine parity: approximately 150.1 TPS.
- C8 aggregate output throughput: approximately 205.5 TPS.
- C8/C1 aggregate scaling: approximately 1.51x.
- Correct steady selection of B1/B2/B4/B8 decode families.
- Major B2/B4/B8 kernel-time reduction.
- Successful immutable A40 execution with complete Nsight recovery.

### 9.2 Remaining measured inefficiencies

1. **C1 request overhead:** decode-engine throughput is approximately
   150.1 TPS while complete request throughput is approximately 136.3 TPS.
   Prefill/TTFT accounts for about 0.21-0.24 seconds on the 106-token prompt;
   visible-token and request-boundary costs account for the rest.
2. **C2 crossover:** B2 decode-engine throughput is approximately 130.4 row
   TPS, below B1's 150.1. B2 improved radically, but batching two rows still
   does not amortize enough work to beat one B1 row in aggregate.
3. **Sublinear B8:** B8 raises useful throughput to approximately 223.9 decode
   rows/s, but per-row latency is materially worse and the kernel times still
   grow nearly linearly from B4 to B8.
4. **Prefill-family proof:** Q128 is compiled, but permanent runtime evidence
   must explicitly report the selected Q family and prove padded positions
   skip routed expert weight work. Admission logs currently expose the profile
   prompt limit, not the actual scheduled query family.
5. **Production profile:** B and Q limits were deliberately reduced to control
   compile cost during parity work. They must be expanded after route
   convergence, then warmed before readiness.

### 9.3 Implemented convergence set, pending A40 measurement

The trace-directed set below is implemented and covered by permanent
construction/ownership contracts. None of its expected performance effect is
claimed until it runs as one immutable image on A40.

1. Reservations now assign every private physical page at admission and upload
   the complete lifetime-stable table. Tentative/committed lengths alone
   control visibility. Stable decode has no physical-growth transition or
   page-boundary lookahead transfer path.
2. An idle engine opens a 50-200 microsecond prefill-formation window derived
   from `max_prefill_wait`; a B1-only profile disables it and active decode can
   never enter it.
3. Dense ordinary projections have separate algorithmic families: exact M1
   uses the proven rank-one GEMV, while every M greater than one uses the
   tensor-core matrix family and reuses decoded weights across real rows. On
   the A40, permanent lowering contracts pin the LM head to M1/N32/K256 for B1
   and M16/N64/K128 for B2/B4/B8.
4. Fused QKV keeps one scalar GEMV CTA per real row and projection tile. Rows
   remain the minor grid dimension for L2 locality, but the contraction body
   is never widened to M2. This restores the accepted B1 body and the
   previously measured B2/B4/B8 fused-QKV path.
5. Sampling performs one batch-wide top-k/filter pipeline over `[B,V]`; only
   the independent per-row RNG transition remains scalarized. Inactive rows
   preserve state and the padding sentinel.
6. Grouped expert prefill selects bounded N64/K128 or N128/K64 families from
   scheduled route-block capacity, projection width, SM count, and architecture
   capabilities. Compiler contracts pin the production Q16/Q128/Q256 choices.
7. Decode uses one reusable four-layer executable across six layer groups,
   reducing recurring transformer submissions from twelve pair calls to six
   group calls while preserving sliding/full order and one-token lookahead.
8. Triton function names include the selected M/N/K tile, and finite metrics
   expose idle formation and stable decode metadata rebinds. The next profile
   can therefore prove plan reachability and absence of page-boundary rebinds
   directly.

The corrected dense-linear and QKV family split has passed affected IR,
Triton, server, image-structure, CUDA-generation, and serving-image BuildBuddy
gates. A new immutable A40 run is still required before claiming that the
accepted 150-TPS decode-engine result and higher-batch throughput are restored.

After A40 validation:

9. Make the diagnostic `generate` command use the generic serving engine, then
   delete the remaining request-local execution route. Static graph
   specialization remains; duplicate orchestration and cache ownership do not.
10. Run a compact mixed-load A40 acceptance matrix only after focused and full
   CPU/CUDA BuildBuddy gates and the server image build prove every intended
   path constructs successfully.
11. Expand the production B/Q envelope and warm retained hot families before
   readiness.

### 9.4 Tuning policy

NML does not dispatch on marketing GPU names and does not runtime-autotune an
unbounded set of graphs. Lowering receives normalized CUDA facts: compute
capability and reported core count. It combines them with static workload
geometry—M/N/K, projection role, route density, batch/query family, and whether
the phase is latency- or throughput-sensitive—to select from a small,
reviewable set of Triton configurations.

Memory capacity is a separate concern. Discovered or declared memory, the
resident compact model footprint, K/V page geometry, the safety reserve, and
request token budgets determine the frozen page count and admission envelope.
Changing memory capacity must not silently change numerical kernel semantics,
compile an unbounded family, or resize the arena after readiness.

## 10. Realistic product scenarios

### 10.1 Interactive cold request

```text
prompt -> chunked prefill -> first token -> stable B1 decode -> completion
```

Report TTFT, p50/p95 TPOT, request latency, output TPS, prefill family, and
decode family. The current 106+320 control remains the regression anchor.

### 10.2 Concurrent conversations

```text
active rows:      1       3       8       7       4       2
selected family: B1      B4      B8      B8      B4      B2
```

Joining or leaving requests cause one visible-token-boundary replan. Survivors
continue over the same global cache pages in the new smallest family.

### 10.3 Long prompt arriving during decode

```text
decode -> decode -> prefill chunk -> decode -> decode -> prefill chunk
```

Decode has priority, but aged prefill receives bounded progress. A 90K-token
prompt is a sequence of page-backed chunks, not one enormous contiguous graph
or K/V allocation. Acceptance reports both long-request TTFT and unrelated
interactive TPOT.

### 10.4 Slow or disconnected client

Bounded response channels prevent a slow reader from blocking unrelated
device work. Backpressure, cancellation, deadline, or disconnect removes the
row at a visible-token boundary and releases reservations/pages exactly once.

### 10.5 Shared-prefix traffic

Until Milestone 5 lands, shared prefixes still execute target prefill
independently. Correct page indirection is necessary but is not itself a
prefix cache.

## 11. Ordered remaining milestones

Later features cannot be credited toward the current efficiency phase.

### Milestone 4: finish the OpenAI tool surface

Current support renders validated function definitions into Harmony, parses a
generated tool call, returns `finish_reason=tool_calls`, and accepts a
subsequent tool result. Remaining work:

- round-trip protocol tests from schema through generated call and client
  result;
- explicit proof that the server never invokes subprocesses or network tools;
- complete streaming/non-streaming equality and malformed-history coverage;
  and
- documented unsupported tool-choice behavior until constrained decoding
  exists.

NML transports tool calls. It never executes them.

### Milestone 5: automatic prefix caching

Add `server/prefix.rs` with:

- versioned SHA-256 chained hashes over complete 16-token pages;
- namespace identity covering target manifest/recipe, tokenizer, model
  configuration, adapter state, and optional tenant salt;
- exact token IDs and parent-hash verification on lookup;
- longest-contiguous-prefix lookup;
- immutable sealed-page descriptors;
- separate live references and zero-reference cache ownership;
- intrusive LRU/free queues;
- eviction only for zero-reference sealed pages;
- duplicate concurrent producer handling;
- private partial tail pages; and
- hit/miss/token/refcount/eviction metrics without prompt content in labels.

Full-prefix hits replay only the final token needed to reestablish next-token
semantics and mask cache writes. Acceptance compares clean and hit outputs for
greedy and fixed-seed sampling, stresses cancellation/eviction races, and
proves repeated long prompts reduce TTFT.

### Milestone 6: GPT-OSS tensor parallelism

Support homogeneous TP=2 and TP=4:

- validate exact device count, backend, compute capability, topology, and
  collective support;
- attach product-owned partition metadata before lowering;
- vocabulary-shard embedding and LM head;
- column-shard Q/K/V and expert gate/up;
- row-shard attention output and expert down;
- keep router, norms, positions, page tables, and sampling state replicated;
- all-reduce hidden states where row-sharded projections require it;
- compute global top-k candidates and deterministic sampling across vocabulary
  shards;
- co-shard E2M1 payload and E4M3 scales directly from checkpoint spans;
- validate every 16-value scale-block boundary;
- shard K/V pages over KV heads while retaining one logical page namespace;
  and
- prove TP=1/2/4 numerical identity, collectives, per-device memory, and real
  CUDA execution.

TP=8 remains deferred until the expert-down K=360 shard has an explicit
alignment/padding contract. Full-model replication is not tensor parallelism.

### Milestone 7: lossless DFlash speculation

Pin `z-lab/gpt-oss-20b-DFlash` revision
`d53f6551543204c859e8bbaaddbd15d11b447af9` as a separate immutable artifact.
NML reimplements the graph in Rust/NML; Python `trust_remote_code` is not a
runtime dependency.

The planned contract is:

- eight Qwen3-style BF16 draft layers;
- target hidden taps after layers `[1,6,11,16,21]`;
- learned `5*2880 -> 2880` projection and RMS normalization;
- mask token ID `200000`;
- one authoritative pending token plus seven parallel proposals;
- one causal eight-token target verification graph;
- exact accepted-prefix selection and successor sampling state;
- tentative target and draft page transactions with commit/rollback;
- target-plus-draft prefix identity;
- bounded draft/verify batch families; and
- an `auto` policy with explicit reason codes for disabled cases.

Greedy output must match speculation-off exactly. Fixed-seed stochastic output
and RNG progression must also match. Promotion requires measured A40 benefit
after draft memory, feature taps, verification, rollback, batching, and server
costs—not an H200 result copied from the paper.

### Milestone 8: production hardening

- add `/v1/completions` through the same engine path;
- finish overload, disconnect, deadline, and shutdown tests;
- add fixed-arrival-rate and mixed-length load generation;
- run hours-long page/RSS/device-memory stability tests;
- define supported OpenAI fields and exact rejection behavior;
- add optional bearer authentication and tenant-derived prefix salts;
- expand and warm the production family profile;
- publish deployment and operator documentation; and
- retain final BuildBuddy, A40, and multi-GPU evidence.

## 12. Verification and promotion

Routine gates use BuildBuddy only:

```text
bb test //products/serve:serve_contract_test --config=buildbuddy --config=cpu
bb test <affected targets> --config=buildbuddy --config=cuda
bb build //products/serve:serve_image --config=buildbuddy --config=cuda
```

Exact affected targets remain in `TASKS.md`. A successful remote CUDA compile
does not prove GPU execution.

Paid A40 cycles occur only after:

1. affected CPU, IR, Triton, serve, and full CUDA construction gates pass;
2. the intended family/dispatch is proved reachable from code and permanent
   contracts;
3. one immutable image digest and source commit are pinned;
4. the run script covers the needed matrix without compiling irrelevant
   families; and
5. collection retrieves and validates result JSON, server log, metrics,
   Nsight report, SQLite export, and profiler summaries before pod termination.

Every performance report pins source, image, model receipt, device, server
profile, workload, warmup, repetitions, and measurement boundary.

The current next acceptance matrix should remain compact:

- the exact 106+320 C1 regression;
- C2/C4/C8 steady decode on the same prompt/output;
- one mixed short/long prompt scenario;
- one cancellation/backpressure scenario; and
- explicit B/Q family, page, transfer, TTFT, TPOT, throughput, and GPU busy
  metrics.

The matrix expands to production lengths and arrival distributions after
route convergence and profile expansion. Testing is evidence for product
behavior, not an end in itself.

## 13. Explicit non-goals and rejected shortcuts

- No new quantization recipe to compensate for inefficient serving machinery.
- No persistent BF16 expansion of NVFP4 weights.
- No runtime model compilation after readiness.
- No whole-transformer monolith.
- No per-request PJRT owner or per-request K/V arena.
- No unbounded command, response, or admission queue.
- No B1-only scheduler, cache owner, or request lifecycle.
- No page table that affects reads but not writes.
- No prefix caching of mutable partial pages.
- No cache key based only on raw text or request JSON.
- No server-side arbitrary tool execution.
- No Python remote-code execution in the product image.
- No speculative token made visible before target verification.
- No full-model replication labeled tensor parallelism.
- No benchmark promotion from a cold run, device-only time, or unpinned
  artifacts.
- No future feature used to hide a present target-model efficiency regression.

The direction is one compact NVFP4 model representation, one generic serving
engine, and a small set of architecture-appropriate Triton kernels. The
current A40 result proves that this design can preserve 150-TPS batch-one
decode while scaling useful small-batch work; the next work is to close the
remaining request-level and B2 efficiency gaps, remove the duplicate
diagnostic route, and then move to prefix caching.

## 14. Primary technical references

- [NVIDIA Transformer Engine NVFP4 format and layouts](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- [NVIDIA nvmath NVFP4 matmul requirements](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/linalg/generated/nvmath.linalg.advanced.Matmul-class.html)
- [NVIDIA NVFP4 and MXFP4 comparison](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [CUDA FP4 intrinsics](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP4.html)
- [cuDNN block-scaling semantics](https://docs.nvidia.com/deeplearning/cudnn/v1.15.0/operations/BlockScaling.html)
- [vLLM chunked prefill and parallelism](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [vLLM prefix-cache design](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [DFlash GPT-OSS artifact](https://huggingface.co/z-lab/gpt-oss-20b-DFlash)
- [DFlash paper](https://arxiv.org/abs/2602.06036)

These references inform implementation. They do not replace NML's artifact,
ownership, numerical, or real-hardware acceptance contracts.
