# NML work plan

Status: executable product tracker

[`SYSTEM.md`](./SYSTEM.md) is the governing architecture.
[`NVFP4.md`](./NVFP4.md) is the representation and kernel contract. This file
contains only durable evidence, unfinished product work, and ordered exits. Git
history—not a growing list of superseded tasks—is the implementation archive.

## Acceptance language

- `[x]` means the implementation exists and its applicable permanent contract
  passed in the venue that can truthfully execute it.
- `[ ]` means work or evidence remains.
- `DEFERRED` is an explicit ordering decision, not an implied capability.
- Remote compilation is not GPU execution. Device discovery is not kernel
  execution. A real CUDA path passes only when normal capability dispatch
  launches it on suitable hardware and its output satisfies an independent
  numerical contract.
- Local compilation and tests are never run by automation without explicit
  owner permission. Routine compile and CPU gates use BuildBuddy with both
  `--config=buildbuddy` and the truthful backend configuration.

## Established substrate

- [x] Rust shapes, semantic axes, layouts, ordinary dtypes, complex values,
  typed tensor programs, StableHLO lowering, Shardy annotations, XLA
  compilation, and CPU/CUDA PJRT ownership.
- [x] Persistent buffers, named executable arguments/results, donation,
  output aliasing, SafeTensors indexing, structural parameter declarations,
  tied storage, bounded component streaming, and exact memory accounting.
- [x] Algebra, comparisons, casts, unary math, activations, reshape/transpose,
  indexing, gather/scatter, reductions, normalization, sorting, sampling,
  convolution, pooling, resize, FFT/IFFT, and explicit-state random programs.
- [x] Ordinary and paged attention, RoPE/YaRN, causal and sliding masks,
  learned sinks, cache update/truncation/replay, FA2, FA3, and Triton paged
  paths.
- [x] Portable routed MoE, grouped expert execution, four-device CPU Shardy
  execution, and grouped Triton CUDA kernels.
- [x] IREE tokenizer ownership with ordinary-text special-token exclusion and
  incremental UTF-8 decoding.
- [x] Real suitable-device substrate evidence: FA2/Triton paths passed on SM86;
  FA3/Triton paths passed on SM90. Both ran numerical attention and grouped-MoE
  contracts through ordinary dispatch. Multi-GPU CUDA remains unproven.
- [x] Builder-authored Triton function ABIs are immutable inputs to typed
  StableHLO calls. Structural contracts reject ABI drift before XLA, including
  the split-K paged-attention plus learned-sink cross-product.

## Current milestone: componentized GPT-OSS 20B NVFP4 generation

Exit: one immutable GPT-OSS 20B NVFP4 artifact loads once; a Harmony prompt
executes embedding, all 24 layers, and the head through compact CPU/CUDA paths;
repeated decode produces independently checked text on a suitable
non-Blackwell GPU without a monolithic transformer compiler module or a
persistent dense weight expansion.

### Artifact, representation, and kernels

- [x] Select and hash one exact GPT-OSS 20B artifact with conversion provenance,
  configuration, tokenizer, tensor inventory, physical component inventory,
  and recipe identity.
- [x] Move the full 11.8 GB content hash to canonical artifact ingestion. The
  materializer authenticates the pinned manifest, hashes every declared file,
  makes the result read-only, and atomically issues an exact filesystem-identity
  receipt. Product startup hashes only the bounded manifest and hard-fails a
  missing or stale receipt; it never silently repeats the payload scan.
- [x] Define NVFP4 recipe v2: output-major/K-contiguous packed E2M1 payload, E4M3FN block scales, F32
  global scale, block geometry, padding, logical shape, and component sharding.
- [x] Replace dense-only weight handling with one closed `Parameter` /
  `LoadedParameter` abstraction. Dense is one component; NVFP4 is three
  components. Ordinary tensor dtypes remain unrelated to storage recipes.
- [x] Stream physical components transactionally with bounded staging and
  source/resident/prepared/staging accounting. No implicit persistent BF16
  expansion is admitted.
- [x] Implement compact CPU embedding, linear, and routed clamped-SwiGLU
  semantics with exact decoding fixtures.
- [x] Implement fused SM75 CUDA custom calls and SM80+ Triton compact embedding,
  projection, and routed expert paths. Unsupported capability/geometry is a
  hard error; optimized dispatch does not fall back to a generic model path.
- [ ] Run the unchanged complete compact operation contract on the next rented
  non-Blackwell acceptance GPU after the current source digest is published.
- [ ] DEFERRED: implement and prove native SM100+ block-scaled execution. Until
  generated code proves native instructions, Blackwell execution remains
  labeled fused emulation.

### Product-owned model and protocol

- [x] Validate the exact 24-layer GPT-OSS configuration, alternating
  full/sliding attention schedule, GQA geometry, YaRN, learned sinks, expert
  geometry, normalization, vocabulary, and untied output projection.
- [x] Declare 411 logical parameters over all 703 compact physical components
  without guessed aliases or transposes.
- [x] Implement package-private Harmony rendering and incremental parsing over
  `o200k_harmony`, including roles, channels, tool calls/results, UTF-8
  fragments, terminal tokens, malformed-stream poisoning, and strict ordinary
  text encoding. NML returns tool calls and never executes them.
- [x] Keep artifact identity, GPT-OSS architecture, checkpoint schema, Harmony,
  shape-family policy, and request lifecycle under `products/serve`.
  Framework crates contain only model-independent operations and ownership.
- [x] Delete the prior model product, compatibility API, tests, Bazel targets,
  and governing-policy references. NML has one selected serving model.

### Reusable component execution

- [x] Replace monolithic full-transformer compilation with bounded reusable
  executables: embedding, sliding-attention layer, full-attention layer, and
  final head for prefill; embedding, alternating sliding/full layer pair, and
  final head for decode.
- [x] Add representation-aware executable parameter slots. A loaded layer may
  bind to a representative compiled layer only when shape, representation,
  component roles/storage, platform, sharding, and executable contracts agree.
- [x] Add asynchronous `enqueue` plus explicit `wait`; keep synchronous `call`
  as the convenience boundary. PJRT readiness dependencies chain component
  outputs without one host synchronization per layer.
- [x] Compose prefill through 24 layer invocations and decode through 12 layer-
  pair invocations while donating hidden state and every K/V pair. Share one
  request-owned I32 identity page table instead of copying model policy into
  the generic runtime cache owner.
- [x] Use finite power-of-two prefill buckets and page-aligned power-of-two cache
  buckets. Validate, normalize, and deduplicate every configured profile;
  compile the complete plan while the checkpoint is metadata-only, then upload
  parameters. Requests select the smallest fitting resident profile and never
  compile. Executables and parameters persist across requests; parser,
  positions, page metadata, and K/V storage remain request-local.
- [x] Pin structural contracts for component input counts, parameter-component
  counts, phase-specific state, donation aliases, representative-layer scope,
  and bucket rejection. Pin a real CPU two-executable dependency chain with a
  differently named parameter bound through a reusable slot.
- [x] The focused product contract and runtime chain pass through BuildBuddy
  CPU execution.
- [x] Build the complete CUDA product binaries and GPU-independent CUDA
  contracts from the final source tree through BuildBuddy.
- [x] Run the full immutable checkpoint on a suitable non-Blackwell CUDA
  device. The initial A40 baseline generated 320 tokens through all 24 layers
  and reported 7.7 steady decode tokens/s; that run exposed the sparse-MoE and
  attention-page performance defects below rather than closing performance.
- [ ] Pin an independent generation fixture and require it in the distinct
  acceptance target. The readable-generation target may not masquerade as
  independent acceptance.
- [ ] Publish and run one immutable product/device-contract OCI digest carrying
  the accepted source revision and exact runtime closure.

### Full-model performance correction

- [x] Move clamped residual SwiGLU to the gate/up epilogue for both dense and
  NVFP4 expert lowering. Gate/up now writes only
  `[assignments, intermediate]`; down consumes that activated tensor and owns
  no activation transcendental. Non-local partitions write zero only for live
  assignments, while inactive capacity blocks perform no stores or weight
  work.
- [x] Replace dynamic E2M1/E4M3FN exponentiation with exact integer/bitcast
  decoding. Load each block scale once for its complete 16-value
  representation block and structurally reject quant-decode `math.exp2`.
- [x] Add dedicated compact `M = 1` GEMV families for ordinary linears,
  paired gate/up plus activation, and routed down on SM75 and SM80+. Decode
  kernels use F32 reductions and no dead-row `tt.dot`; prefill retains the
  tensor-core matrix family.
- [x] Extend the independent CPU oracle across deterministic randomized expert
  shapes, odd widths, empty experts, uneven routes, bias/route order, and
  one-token decode. Generated-TTIR and IR contracts pin the same boundary.

- [x] Make MoE schedule capacity proportional to the number of experts that can
  actually be non-empty, and use ZML's direct sparse assignment crossover for
  decode-shaped route sets.
- [x] Carry an explicit active-block scalar through dense/NVFP4 and
  expert-parallel grouped lowering. Inactive and non-local Triton programs now
  branch before every weight address, scale decode, and contraction; SM75
  custom calls return before fragment or weight work.
- [x] Select the finite decode/small/large grouped-NVFP4 tile families from
  token geometry and named CUDA capabilities instead of retaining the generic
  32x32 contraction tile for GPT-OSS decode.
- [x] Replace the product's 256-token cache pages with 16-token pages and
  independently cap framework decode attention tiles at 64 tokens. This
  removes the geometry that produced roughly 12 KiB of register spills in each
  A40 split-K attention producer.
- [x] Resolve PJRT loaded-executable output arity once at compilation, retain
  executable input indices and output names, and stop repeating immutable
  metadata work at every component enqueue.
- [x] Replace the product's implicit greedy head with request-owned,
  explicit-state top-k/temperature/top-p/min-p sampling. Greedy remains the
  explicit `top_k = 1` mode.
- [x] Report non-synchronizing host submission time separately for embedding,
  sliding layers, full layers, and the head in prefill, first decode, and
  steady decode phases.
- [x] Pass the complete BuildBuddy CPU and CUDA build/test envelopes for the
  final corrected source tree. Focused artifact, CPU-NVFP4, TTIR, IR, and
  product contracts pass in invocation
  `4ed47218-5dcf-43d3-b86d-b433d23f3166`; the complete CPU suite passes in
  `55b69599-5499-49f9-8ac0-c894bc2d835a`, the GPU-independent CUDA suite in
  `613a8006-71a4-4f58-8bcf-55fc7f7d6f53`, the complete CUDA binary closure in
  `b5552060-f990-4070-9af6-4da37d7bcba9`, and CUDA packaging in
  `259a3cb4-8d7b-4c50-a769-8c43ac8e97ba`. The corrected CUDA serve OCI image
  constructs in `64b7b6b6-9ced-4360-93be-9d8d6ac11436`. None is presented as
  NVIDIA runtime evidence.
- [x] Capture post-refactor A40 hardware evidence for image digest
  `69e805cd5128...` with one mandatory Nsight-Systems-over-GDB stochastic
  128-token run. GDB observed a normal inferior exit, Nsight exported its full
  node trace and four CUDA summaries, and the product sustained 55.475 steady
  device tokens/s under profiling versus 59.611 without profiling. The report
  directory under `references/runpod/reports` is
  `20260719T121922Z-aiuvl369ogh26v-69e805cd5128-diagnostic`; this is diagnostic
  evidence, not the final gate, because the published image records the
  equivalent pre-commit dirty source identity rather than the accepted commit.
- [x] Reject and manually remove the composed-decode/direct-kernel experiment.
  Immutable image `c10b80d8dd51...` completed a coherent 320-token A40 run
  under the combined GDB/Nsight harness, but sustained only 5.601 steady
  device tokens/s versus the accepted profiled baseline of 55.475. Direct
  expert gate/up and down consumed 92.1% of GPU kernel time; adding streaming
  head top-k raised the three rejected kernels to 97%. The durable report is
  `references/runpod/reports/20260719T152709Z-lm4xqsqg7we5ym-c10b80d8dd51-diagnostic`.
  Output-owner ordinary GEMV, direct top-four expert kernels, streaming head
  top-k, six-layer decode segments, and their sole-purpose semantic APIs are
  not retained as dormant paths. BuildBuddy publication and the combined
  diagnostic harness remain independent accepted infrastructure. The restored
  source passes the full remote CUDA suite in
  `61f0b083-80e5-40a7-8d91-bb4dfd80c4a6`, package/image contracts in
  `7287d013-ed87-44cc-a42c-9897fb1d1e1d`, and the CUDA binary plus serving-image
  closure in `f2a46248-ea91-44c2-8866-3b3d833c0219`.
- [x] Publish the manually restored source and rerun the whole GPT-OSS A40
  baseline through the combined GDB/Nsight harness. BuildBuddy publication
  invocation `9d38ff9d-c361-425a-83e6-28ec433db6e4` resolved restored commit
  `6f8dd0b222721a3ecd0a501e035192cd2b400ef4` to immutable digest
  `69e805cd5128...`. The fresh 128-token run sustained 57.248 steady device
  tokens/s, 56.398 device-decode tokens/s, and 54.504 decode-loop tokens/s,
  generated coherent text, and exited normally. The validated report is
  `references/runpod/reports/20260719T155954Z-fdmcvpur8oks3p-69e805cd5128-diagnostic`;
  Pod `fdmcvpur8oks3p` was terminated and deletion confirmed.
- [x] Replace recipe v1 with one recipe-v2 output-major/K-contiguous layout for
  ordinary, gate/up, and down contractions. Conversion transposes expert source
  tensors before quantization; CPU, SM75, Triton matrix, and Triton decode
  implementations consume the same packed components. No compatibility or
  persistent prepared representation remains. Focused IR, Triton, and compact
  CPU contracts pass remotely, including odd reduction widths, in BuildBuddy
  invocations `4d31e920-c3cd-41f3-b9ee-7babad1d1842`,
  `c04134f4-38ef-4e10-989b-8e79fe20c5a9`, and
  `8dd33c22-6a36-43a4-bb76-ca89d0bb945c`.
- [x] Replace decode-shaped Triton compact projections with finite rowwise GEMV
  schedules that vector-load contiguous K blocks, decode one scale per 16
  values, accumulate in F32, and retain the gate/up activation epilogue.
  Matrix-shaped prefill retains its tensor-core schedule over the same recipe.
- [x] Convert and publish the exact GPT-OSS artifact as recipe v2 and reject
  every v1 identity. The public artifact is pinned at revision
  `704c34282b2d84cc6a4e5ce7de14b6f6fc1286e9`; its converter-produced manifest
  hashes to `3c36a89cbc0f908b3e782550fe32f3b6890ef3f857232d11710bc8e0dbcea71d`.
  Conversion remained CPU-only even though a price-capped RunPod worker hosted
  the process after BuildBuddy capacity failures; no GPU conversion semantics
  or local model transformation were introduced. The complete common/CUDA
  package contract suite passes in BuildBuddy invocation
  `2436149e-5211-4036-9de2-451a61661407`, and the CUDA binary plus serving OCI
  closure builds in invocation `ac508b59-a6f4-46d7-b61e-733b577e8192`.
- [x] Reuse one baked argument owner per request/component across decode
  iterations. Persistent parameter components remain bound while only donated
  hidden/cache state and request-local scalar inputs are replaced; executable
  output arity and names remain compile-time metadata.
- [x] Compile bounded two-layer decode executables for the model's real
  alternating attention schedule, reducing recurring PJRT graph submissions
  from 24 layer calls to 12 without restoring the rejected six-layer/direct-
  kernel experiment. Prefill retains single-layer components. The structural
  product contract and complete CUDA product binaries pass in BuildBuddy
  invocations `73b53056-cf33-4626-8f60-7e19206c3df5` and
  `8587ad5e-ff8a-440c-8c2b-7e34c3edccc0`; runtime promotion still awaits the
  recipe-v2 whole-model measurement.
- [x] Keep sampling and top-k device-resident and reduce token readback to the
  selected scalar plus readiness boundary; do not materialize vocabulary
  logits on the host.
- [ ] Publish the first complete recipe-v2 image and measure it on A40 through
  the mandatory combined GDB/Nsight harness before composing any further
  projection or orchestration experiment. Promotion requires correct text,
  numerical contracts, durable trace evidence, and at least 143.12 steady
  device tokens/s, a 2.5-fold improvement over the restored 57.248-token/s
  baseline. Image digest `d4da39627c61...` from source commit `feaf370` ran
  320 tokens normally on an A40 and generated coherent text under the combined
  harness. It reached 100.083 steady device tokens/s, 99.777 overall device
  decode tokens/s, and 92.660 decode-loop tokens/s. The durable report is
  `references/runpod/reports/20260720T074402Z-ylcav28r6vy6kf-d4da39627c61-diagnostic`;
  Pod `ylcav28r6vy6kf` was terminated and deletion confirmed. The runtime path
  is real and 1.75-fold faster than the restored baseline, but this task remains
  open because it misses the accepted 143.12-token/s performance gate.

## Next milestone: continuous batching and shared paged state

Exit: at least two GPT-OSS requests share one physical decode invocation and
one server-owned page arena while matching independent sequential execution.

- [ ] Generalize model inputs, positions, logits selection, sampling, and result
  demultiplexing from batch one to fixed-capacity active slots.
- [ ] Create one generation-stamped physical K/V page arena with checked memory
  accounting, transactional leases, copy-on-write partial pages, and one
  idempotent reclamation path.
- [ ] Add a bounded scheduler for chunked prefill and decode with explicit
  token/sequence budgets, starvation bounds, slot refill, and no recompilation.
- [ ] Make completion, cancellation, disconnect, deadline, execution failure,
  and shutdown reclaim request state exactly once.
- [ ] Add prefix caching over immutable complete pages keyed by exact tokens,
  model, protocol, RoPE, representation recipe, executable family, and
  topology. Admission and eviction share the same page budget.
- [ ] Compare batched and sequential tokens, cache contents, sampling state,
  cancellation, and page accounting.

## Later milestone: bounded serving control plane

Exit: a real loopback server handles concurrent bounded streaming and
non-streaming GPT-OSS requests with deterministic lifecycle behavior.

- [ ] Add one Tokio runtime. One dedicated engine owner holds PJRT state and
  communicates through bounded command, token/event, and completion channels;
  PJRT work never blocks Tokio workers opportunistically.
- [ ] Add bounded Axum/Tower chat-completions and Responses-style routes with
  readiness, liveness, model identity, streaming, strict request limits,
  cancellation, deadlines, signals, and graceful shutdown.
- [ ] Integrate Harmony tool-call output without executing tools.
- [ ] Cover admission races, queue saturation, page exhaustion, disconnects,
  partial streams, engine/listener failure, repeated lifecycle, and zero-owner
  shutdown.
- [ ] DEFERRED: bounded-cardinality Prometheus metrics and structured tracing.
  Correct serving capability precedes observability surface.

## Later milestone: distributed GPT-OSS execution

- [ ] Define tensor/expert-parallel Shardy placement for embeddings, attention,
  routers, experts, output projection, and required reductions without hidden
  whole-weight all-gathers.
- [ ] Compare four-device CPU execution with the single-device oracle.
- [ ] Run the unchanged paged/batched contract on homogeneous multi-GPU CUDA
  and retain collective, memory, correctness, and scaling evidence.

## Deferred independent verticals

- [ ] DEFERRED: native Blackwell NVFP4 after the non-Blackwell product closes.
- [ ] DEFERRED: W4A16, with a separately selected artifact and complete packing,
  scale, zero-point, compute, accumulation, checkpoint, and kernel contract.
- [ ] DEFERRED: W8A8, with an independently selected integer/FP8 activation and
  weight recipe.
- [ ] DEFERRED: NVFP4 KV-cache storage; compact weights do not imply compact KV.
- [ ] DEFERRED: speculative decoding until a GPT-OSS-compatible draft artifact
  or public algorithm is selected and end-to-end benefit can be measured.
- [ ] DEFERRED: explicitly authored analytic backward/training programs.

## Capability ledger

- [x] Primitive algebra, comparisons, selection, casts, bit operations, unary
  math, activations, complex values, and FFT/IFFT.
- [x] Reshape, transpose, concatenation, slicing, dynamic update, gather,
  scatter, embedding lookup, and layout-aware compiled graphs.
- [x] Reductions, softmax, RMSNorm, LayerNorm, L2 normalization, log-sum-exp,
  argmax, sorting, top-k, greedy selection, and stochastic sampling.
- [x] Matrix contraction, linear layers, convolution, pooling, and resize.
- [x] Explicit-state random generation.
- [x] Ordinary and paged attention, RoPE/YaRN, masks, sliding windows, learned
  sinks, and persistent KV update/truncate/rollback/replay.
- [x] Capability-dispatched FA2, FA3, Triton paged attention, portable MoE, and
  grouped Triton MoE with real suitable-device numerical evidence.
- [x] Shardy CPU meshes, tiled parameters/activations, collectives, and expert
  sharding.
- [x] Dense and NVFP4 parameter representations with compact CPU, SM75, and
  SM80+ operation paths.
- [x] GPT-OSS configuration, complete checkpoint declaration, Harmony, and
  reusable component graph/execution architecture.
- [x] Full-checkpoint GPT-OSS text generation executes on real CUDA hardware;
  the distinct independent-fixture acceptance gate remains open above.
- [ ] Continuous batching, shared page arena, and prefix caching.
- [ ] Real multi-GPU CUDA Shardy execution and collectives.
- [ ] Bounded network serving lifecycle.
- [ ] Dedicated optimized attention and compact-model performance evidence.
