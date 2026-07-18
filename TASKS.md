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
- [x] Define NVFP4 recipe v1: packed E2M1 payload, E4M3FN block scales, F32
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

- [x] Replace monolithic full-transformer compilation with four bounded
  executables per shape family: embedding, reusable sliding-attention layer,
  reusable full-attention layer, and final head.
- [x] Add representation-aware executable parameter slots. A loaded layer may
  bind to a representative compiled layer only when shape, representation,
  component roles/storage, platform, sharding, and executable contracts agree.
- [x] Add asynchronous `enqueue` plus explicit `wait`; keep synchronous `call`
  as the convenience boundary. PJRT readiness dependencies chain component
  outputs without one host synchronization per layer.
- [x] Compose prefill and decode through the 24 layer invocations while donating
  hidden state and every K/V pair. Share one request-owned I32 identity page
  table instead of copying model policy into the generic runtime cache owner.
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
- [ ] Run the full immutable checkpoint on a suitable non-Blackwell CUDA device.
  Record artifact bytes, host peak, compact resident bytes, cache/workspace,
  compilation per component family, first execution, prefill, steady decode,
  selected kernels, generated token IDs, Harmony events, and confirmed Pod
  cleanup.
- [ ] Pin an independent generation fixture and require it in the distinct
  acceptance target. The readable-generation target may not masquerade as
  independent acceptance.
- [ ] Publish and run one immutable product/device-contract OCI digest carrying
  the accepted source revision and exact runtime closure.

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
- [ ] Full-checkpoint GPT-OSS text generation accepted on real CUDA hardware.
- [ ] Continuous batching, shared page arena, and prefix caching.
- [ ] Real multi-GPU CUDA Shardy execution and collectives.
- [ ] Bounded network serving lifecycle.
- [ ] Dedicated optimized attention and compact-model performance evidence.
