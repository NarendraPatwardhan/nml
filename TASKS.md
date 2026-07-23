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

## Current serving foundation

- [x] Retain one exact GPT-OSS 20B NVFP4 target artifact, recipe-v2 compact
  parameter representation, immutable materialization receipt, IREE
  `o200k_harmony` tokenizer, and package-private
  `openai-harmony-gpt-oss-v1` protocol owner.
- [x] Retain reusable compile-before-residency component executables,
  asynchronous PJRT dependency chaining, explicit-state dynamic sampling,
  16-token paged-attention reads, cache rollback/replay primitives, and Shardy
  mesh/physical shard loading substrate.
- [x] Retain the accepted A40 batch-1 kernel path: fused compact QKV,
  recipe-v2 expert gate/up and down, exact-tail/codebook decoder, and bounded
  five-layer-pair lookahead.
- [x] Pin the server performance control to source
  `fb415a8dadd51a0053b9be314faa836e2b274721`, image
  `sha256:3c81704ea85512df7ff76de83ea21f403ef592dc50c23c5ae20e8d70c1e7f3ff`,
  and the checked-in A40 report: 156.062 steady-device, 155.644 device-decode,
  and **151.324 decode-loop tokens/s** for the 106+320 token workload.
- [x] Record the serving ownership decision in `SYSTEM.md`: Tokio owns async
  control, one dedicated engine owner retains PJRT/model/cache state, and
  `products/serve` owns scheduling, global pages, streaming, tools, and
  metrics.
- [x] Preserve the exact control workload as a named server regression profile
  so every serving milestone can compare batch-1 decode-loop throughput against
  151.324 tokens/s rather than an unpinned historical number.

## Milestone 0: contracts and step-wise model execution

Exit: the current single-request generator is expressed through request-local
prepare/prefill/decode/finalize steps, owned protocol sessions can coexist, and
the adapter produces exactly the current tokens/events without adding HTTP or
global page sharing yet.

### 0.1 Freeze public/internal contracts

- [x] Add `RequestId`, `SequenceId`, `FinishReason`, `CancelReason`,
  `PreparedInferenceRequest`, and stable engine error categories under
  `products/serve`; keep these independent of HTTP response types.
- [x] Define exact request limits: maximum body bytes, prompt tokens, completion
  tokens, context tokens, queue capacity, active sequence count, per-request
  event capacity, and deadline representation. Reject overflow before cache or
  device mutation.
- [x] Define one explicit `ServerProfile` containing batch buckets, prefill
  query buckets, maximum model length, batched-token budget, prefill chunk,
  cache memory/safety budget, TP degree, and speculation policy. Validate and
  deduplicate it at startup.
- [x] Add pure state-transition contracts proving terminal finalization is
  idempotent and releases response/cache/admission ownership exactly once.

### 0.2 Own tokenizer/decoder sessions safely

- [x] Refactor `crates/nml-tokenizer/src/lib.rs` so `Tokenizer` and every
  `Decoder` retain one audited shared immutable tokenizer owner instead of a
  borrow tied to a stack-local protocol value.
- [x] Preserve destruction order: the IREE tokenizer allocation outlives all
  encoders/decoders and is freed exactly once after the last owner.
- [x] Decide concurrency from evidence: prove independent IREE
  encoder/decoder use is thread-safe with a permanent contract, or serialize
  calls behind the narrow shared owner. Do not add unchecked `Send`/`Sync`.
- [x] Make `HarmonyParser` an owned per-request state with no self-reference,
  leaked allocation, or lifetime borrowed from `Generator`.
- [x] Keep ordinary-text special-token exclusion, incremental UTF-8 behavior,
  malformed-stream poisoning, terminal handling, and tool-JSON withholding
  byte-identical to the current fixtures.
- [x] Add a permanent many-session test that interleaves at least 32 Harmony
  parsers and proves their decoder state cannot cross requests.

### 0.3 Split model execution into steps

- [x] Replace `ResidentModel::generate` as the engine primitive with explicit
  prepare, bounded prefill, one-step decode, and finalize methods in
  `products/serve/src/gpt_oss/execution.rs`.
- [x] Move request tokens, profile selection, position, sampling state, parser
  independence, K/V ownership, generated count, and stop state into one
  request state object rather than local variables in a complete generation
  loop.
- [x] Keep model definition -> compilation -> parameter residency ordering and
  make every step select only precompiled executables.
- [x] Retain a blocking `Generator::generate` diagnostic adapter as the pinned
  151-TPS migration control; it is temporary until the generic serving lane
  proves parity and is then deleted by the Milestone 3 convergence task.
- [x] Preserve the batch-1 five-pair lookahead and its “never past visible
  budget”/terminal-discard invariants in the step driver.
- [ ] Add fixed-seed equivalence tests for normal return, tool call, early stop,
  maximum-token truncation, zero-new-token request, and stochastic sampling.
- [x] Run the affected CPU and CUDA compile/contracts through BuildBuddy; do
  not begin Tokio work until step equivalence is green.

## Milestone 1: Tokio/Axum serving shell

Exit: a long-running binary accepts OpenAI-shaped chat requests, streams or
assembles results, applies overload/cancellation/shutdown boundaries, and sends
all PJRT work to one dedicated engine thread. Execution may still be
single-request/serialized until Milestone 3.

### 1.1 Engine thread and bounded channels

- [x] Add `products/serve/src/server.rs` and `server/engine.rs` with bounded
  `EngineCommand` and per-request `EngineEvent` Tokio channels.
- [x] Construct `Platform`, target definition, compiled plan, resident
  parameters, and request execution state on one named OS thread. Send only a
  startup result and lightweight `EngineHandle` back to Tokio.
- [ ] Prove no XLA compile, PJRT enqueue/wait/download, parameter upload, or
  buffer destruction runs on a Tokio worker.
- [x] Add bounded command draining per engine iteration so a request flood
  cannot starve active inference.
- [ ] Use `try_send` for request events. On a full/disconnected response queue,
  cancel the request with a stable reason instead of blocking the engine.
- [x] Add cancellation tokens plus explicit cancel commands; make disconnect,
  deadline, server shutdown, and slow-reader cancellation converge on the same
  idempotent terminal path.
- [x] Add startup/readiness one-shot and graceful shutdown command. Join the
  engine owner so PJRT/plugin destruction order is deterministic.

### 1.2 HTTP lifecycle and operations

- [x] Add pinned Axum, Tower, Tokio, `tokio-util`, tracing, and
  `prometheus-client` dependencies from the existing Bzlmod crate universe to
  `products/serve/BUILD.bazel`; do not introduce a Cargo dependency graph.
- [x] Convert `main.rs` to a Tokio entry point with a `serve` command and
  explicit bind/model/backend/profile/cache/parallel/shutdown options.
- [x] Keep a named `generate` diagnostic command only if the current RunPod
  acceptance harness requires it, and drive it through the same executor.
- [x] Implement `GET /healthz`, `GET /readyz`, `GET /metrics`, and
  `GET /v1/models` with truthful distinctions between process liveness,
  engine readiness, and model identity.
- [x] Apply request body limit, load shedding, bounded concurrent preparation,
  and admission timeout through Tower. Do not use a blanket request timeout
  that can abandon live device work.
- [x] Bind readiness behavior explicitly: socket may bind while startup occurs,
  but inference returns 503 until the engine reports resident/warmed state.
- [x] On SIGINT/SIGTERM stop admission, fail queued requests, allow active work
  until the grace deadline, cancel the remainder, join the engine, and exit
  nonzero if shutdown integrity fails.

### 1.3 Initial OpenAI chat boundary

- [x] Add strict request/response/SSE Serde types in
  `products/serve/src/api/openai.rs`; reject unknown/unsupported semantics
  rather than passing arbitrary JSON to GPT-OSS.
- [x] Implement `POST /v1/chat/completions` for one choice, text-only messages,
  max tokens, temperature/top-p/top-k/min-p, seed, stream, and usage.
- [x] Render structured messages through Harmony before engine admission; do
  not accept caller-supplied raw protocol tokens.
- [x] Translate one raw token stream through an owned Harmony parser into final
  content/reasoning/terminal events.
- [x] Implement exact SSE role/content/finish/usage deltas and `[DONE]`; build
  non-stream responses from the same internal event sequence.
- [x] Attach a drop guard so client disconnect cancels queued, prefilling, or
  decoding work and eventually returns every request resource.
- [x] Return OpenAI-shaped 400/429/503/internal errors with stable internal
  codes; never expose artifact paths, prompt content, or raw device errors.
- [ ] Add protocol-level tests for streaming/non-streaming equality, disconnect,
  queue full, not ready, deadline, invalid model, and unsupported fields.

### 1.4 Initial observability

- [x] Add bounded-label counters/gauges/histograms for received/admitted/
  terminal requests, queue depths, active state, TTFT, TPOT, end-to-end latency,
  engine iteration, and errors.
- [x] Add JSON tracing with request ID, state, phase, batch family, timings, and
  error code. Keep text, token IDs, tool arguments, and salts out of logs.
- [x] Add an engine snapshot command for readiness/metrics that cannot mutate
  scheduling state or wait on a client.
- [x] Build and structure-test the OCI image through BuildBuddy and update the
  image entrypoint/structure fixture for the server command.

## Milestone 2: process-wide paged KV arena

Exit: request-local K/V allocations are gone; all target layers use arbitrary
physical page IDs for both writes and reads; reservations, rollback,
cancellation, and reclamation pass exact accounting.

### 2.1 Correct page-aware writes

- [x] Add a generic batched page-update operation in `crates/nml-ir` accepting
  physical cache, updates, block tables, start positions, query lengths, and
  active rows.
- [x] Author the portable StableHLO scatter/update semantics across page
  boundaries with donation/output aliasing and no dense logical-cache copy.
- [x] Validate every used physical page ID and allow `-1` only in inactive
  trailing block-table slots.
- [x] Prove padded queries/inactive rows do not write any cache element.
- [x] Support a per-query write mask so an exact full-prefix hit can replay the
  final cached prompt token to reconstruct logits without appending duplicate
  K/V.
- [x] Replace GPT-OSS's identity-only dense
  `reshape + dynamic_update_slice` K/V write with the new operation in every
  prefill/decode layer graph.
- [x] Add a contract using a non-monotonic page table and boundary-crossing
  query chunk; compare full output/cache bytes to an independent dense oracle.
- [x] Replace the trace-correlated decomposed CUDA K/V scatters with one paired
  generic Triton append that resolves each active row's page once, writes K and
  V, skips inactive rows, and aliases both donated cache buffers.
- [ ] Confirm on A40/Nsight that each layer emits the paired append and that the
  prior decomposed mask/index/scatter kernel cluster is gone.

### 2.2 Allocate global target storage

- [x] Add `server/cache.rs` with a preallocated descriptor pool, free queue,
  request block tables, reservation credits, and exact target cache accounting.
- [x] Freeze a whole-page count before graph compilation from an explicit cache
  budget (or conservative platform/declaration accounting) and include it in
  every serving family; never resize/recompile after parameter residency.
- [ ] After parameter residency, recheck actual remaining memory against the
  frozen cache budget and safety margin; fail readiness instead of silently
  shrinking or recompiling.
- [x] Allocate one global K and V arena buffer per target layer with shape
  `[physical_pages, 16, local_kv_heads, 64]`; never allocate per-request K/V.
- [x] Use one physical page ID namespace across all 24 target layers and one
  logical block table per sequence.
- [x] Model page states `Free`, private partial, and immutable sealed; only a
  complete committed page may be sealed.
- [x] Track host committed/tentative lengths separately so stale speculative or
  rolled-back bytes remain invisible.
- [x] Upload tokens, page tables, lengths, row masks, sampling controls, and RNG
  state as typed contiguous sections of one validated U8 slab per scheduled
  batch, with exactly one product-level H2D transfer.
- [ ] Measure the compact slab's upload bytes and time separately on A40 and
  retain the correlated profiler summary outside the source snapshot.

### 2.3 Reservations and lifecycle

- [x] Reserve page credits at admission for prompt misses plus worst-case
  remaining output, accounting for the private tail, so an admitted request
  cannot encounter a mid-generation cache OOM.
- [x] Allocate physical pages lazily from credits; queued requests wait when
  capacity cannot guarantee finish.
- [x] Implement cache checkpoints, tentative allocation, commit, rollback,
  truncate, replay, cancellation release, and reverse-order terminal release.
- [x] Ensure one terminal path decrements every page refcount and reservation
  exactly once, including device error and full response queue.
- [x] Add pure randomized state-machine tests that compare the page manager to
  a simple reference model under allocate/append/seal/share/checkpoint/
  rollback/cancel sequences.
- [ ] Add engine integration tests with concurrent requests using permuted
  pages and prove free/private/referenced totals return to the startup snapshot.
- [ ] Run a real A40 batch-1 regression before enabling batching. Page
  indirection/global allocation must preserve at least 150 decode-loop TPS.

## Milestone 3: continuous batching and chunked prefill

Exit: one iteration advances a dynamic set of requests, batch membership may
change every token, long prefills are chunked around decode work, and real A40
concurrency increases aggregate throughput without breaking the batch-1
control.

### 3.1 Compile finite serving families

- [x] Extend GPT-OSS `ShapeFamily` with batch capacity, query capacity,
  logical-page capacity, physical-page count, and parallel config.
- [x] Compile the first A40 batch buckets `1,2,4,8,16,32`, decode query size 1,
  and prefill query buckets `16,64,128,256` from one validated
  `ServerProfile`.
- [x] Log and report the exact family count/compile time; reject duplicate or
  combinatorially unbounded profile input.
- [ ] Continue compiling all families before parameter residency and warm each
  hot family before readiness.

### 3.2 Batched graph semantics

- [x] Generalize embedding, layers, paired decode, head, page update, and paged
  attention inputs to `[B,Q,...]`, per-row positions/lengths, and active masks.
- [x] Flatten `B*Q` only for routing/linear operations that require token-major
  input and restore axes before cache/attention/output.
- [x] Extend generic explicit-state sampling to `[B,2]` RNG states plus per-row
  top-k/temperature/top-p/min-p and active rows.
- [x] Make inactive rows preserve cache, RNG, positions, and output sentinel
  exactly.
- [x] Return one compact token/state result buffer per batch and scatter it to
  request states after one engine-thread download.
- [x] Add fixed-seed contracts proving request A is invariant when request B is
  inserted, removed, cancelled, or assigned another padded slot.

### 3.3 Batched compact CUDA paths

- [ ] Audit actual lowering for every retained small-M family; reject any path
  that expands persistent NVFP4 weights to BF16.
- [ ] Preserve the accepted M=1 fused-QKV/GEMV/expert/head dispatch unchanged.
- [ ] Add or tune Triton compact QKV/projection kernels for small
  `M=B*Q` so weights can be reused across active rows rather than issuing B
  isolated M=1 launches.
- [x] Flatten the active `[B,Q]` mask into routed MoE, encode inactive routes
  with expert ID `-1`, exclude them from the assignment schedule, touch no
  inactive expert weights, and return exact zero for inactive tokens.
- [x] Preserve sparse masked B1/B2 routing with one scan and two compact
  scatters so a runtime mask does not force decode through the full per-expert
  scheduler.
- [ ] Confirm on A40/Nsight that Q128 serves the 106-token control and padded
  prompt/batch positions no longer launch expert blocks.
- [ ] Add batch-aware vocabulary projection/sampling and retain the global
  top-64 candidate contract.
- [ ] Benchmark each batch bucket against issuing the same active rows
  separately; scheduler must not select a regressing family until corrected.

### 3.4 Scheduler implementation

- [x] Add `server/scheduler.rs` with FIFO arrival, explicit waiting/prefill/
  decode queues, dynamic slots, aging, and immutable `BatchPlan` output.
- [x] Drain a bounded number of commands/cancellations at the start of each
  iteration.
- [x] Admit only requests whose reservation credits guarantee completion.
- [x] Schedule eligible decode first up to sequence/token budgets.
- [x] Spend remaining token budget on oldest prompt chunks; cap each request at
  256 prefill tokens per first acceptance profile.
- [x] Reserve at least one prefill chunk after `max_prefill_wait` so sustained
  decode cannot starve admission.
- [x] Execute decode and prefill as separate submissions in the same scheduler
  iteration; do not block all decode behind one long prompt.
- [x] Repack surviving/new sequences into the smallest next batch family after
  every result; request identity must not equal batch slot.
- [x] Replace per-token batch reconstruction with one generic stable
  continuation lane for every retained B family. Keep token, RNG, position,
  sequence length, page-table metadata, executable arguments, and the accepted
  five-pair prefix resident while membership is stable.
- [x] Make the serving head donate the next batch slab, advancing token, RNG,
  position, and sequence length on device with zero steady H2D and one compact
  B*20-byte D2H.
- [x] Queue the next embedding and five layer pairs immediately after the head
  submission, overlapping the compact result download and host token handling
  with useful GPU work.
- [x] Continue stable B1-B32 batches without scheduler re-entry until a command,
  cancellation, deadline, shutdown, backpressure, terminal row, page-table
  change, or membership change requires a visible-token-boundary replan.
- [x] Reuse process-lifetime compiled family bindings and one compact
  result-download workspace per family.
- [x] Delete `SingleSequenceDecodeLane`, its RNG export/import transition, and
  its private B1 serving lifecycle; B1 is now the smallest generic family.
- [ ] Publish the refactored server image and prove at least 150 decode-loop
  tokens/s on the exact 106+320 A40 control before removing the diagnostic
  performance route.
- [ ] After A40 parity, drive `generate` through the generic serving lane and delete
  `ProductSession`/request-local prefill plus the remaining non-serving route;
  retain static family specialization as graph policy, not a second engine.

### 3.5 Deterministic and load acceptance

- [x] Add a device-free virtual-time scheduler test for mixed arrivals,
  sustained decode, aged prefill, queue saturation, page pressure, cancellation
  in every state, and slow-reader event saturation.
- [ ] Compare batched to independent output for greedy and stochastic requests,
  mixed sampling settings, chunked/unchunked prompts, and stop/tool tokens.
- [ ] Add a hermetic load client target supporting fixed concurrency,
  fixed arrival rate, streaming/non-streaming, deterministic disconnect, and
  prompt/output mixes.
- [ ] Publish one immutable image and run five warm A40 repetitions for
  `128/128`, `1K/128`, and `4K/256` at concurrency `1,2,4,8,16,32`.
- [ ] Report TTFT, TPOT, end-to-end latency, request/prompt/output throughput,
  queue time, batch histogram, GPU busy time, and page utilization.
- [ ] Promote only if concurrency-8 aggregate output throughput is at least
  1.5x concurrency-1, p95 TPOT is below 2.5x concurrency-1, all output
  contracts pass, and the exact single-stream control remains at least 150
  decode-loop tokens/s.

## Milestone 4: complete OpenAI tool calling

Exit: clients can submit function schemas, receive one strict Harmony tool
call through stream/non-stream responses, execute it themselves, and submit a
validated tool result in the next conversation. NML performs no tool action.

- [x] Add structured OpenAI `tools` definitions with type `function`, validated
  name/description/JSON Schema parameters, and explicit size/count limits.
- [x] Implement `tool_choice=none` by omitting definitions and
  `tool_choice=auto` by rendering the audited Harmony developer tool section.
- [x] Reject `required` and named forced choice until a real constrained
  decoder can enforce them; do not approximate enforcement with a prompt and
  claim API compatibility.
- [x] Validate assistant `tool_calls` history and subsequent `role=tool`
  messages: stable call ID, matching name, correct order, exactly one result,
  valid raw JSON arguments, and no result-before-call.
- [x] Map history to `Message::ToolCall`/`Message::ToolResult` and prove exact
  byte/token rendering with the current Harmony fixtures.
- [x] Generate stable `call_<request-id>_0` IDs without using the ID in prompt
  tokens or prefix identity.
- [x] Emit a complete SSE tool-call delta only after the Harmony parser closes
  valid JSON with `<|call|>`; never stream partial/invalid arguments.
- [x] Map terminal reason to OpenAI `tool_calls`, include name/raw arguments in
  non-stream output, and stop generation without invoking anything.
- [ ] Add round-trip tests for tool schema -> generated call -> client result ->
  follow-up completion, malformed model calls, truncated JSON, disconnect,
  cancellation, and repeated-prefix history.
- [ ] Add an explicit negative contract proving no subprocess, network call,
  registry dispatch, or function callback occurs when a tool call is returned.

## Milestone 5: automatic prefix caching

Exit: exact complete prompt/history pages are shared through chained hashes,
live requests pin refcounts, zero-reference sealed pages are LRU-evictable, and
repeated-prefix TTFT improves without cross-namespace reuse.

### 5.1 Hash/index contract

- [ ] Add `server/prefix.rs` with versioned SHA-256 chained block hashes over
  parent digest, exact 16 token IDs, namespace digest, and request cache salt.
- [ ] Build the namespace digest from target manifest/recipe, tokenizer,
  Harmony protocol, model/cache/RoPE semantics, KV dtype/page size, and
  a model/cache semantic fingerprint that excludes batch slots and padding.
- [ ] Retain exact token IDs and parent hash in each descriptor and compare
  after lookup; never rely on digest equality alone to reuse unrelated state.
- [ ] Index only full committed sealed pages. Partial tail pages remain private
  and absent from the hash index.
- [ ] Add optional/required cache-salt policy. Different salts must have zero
  shared page IDs and distinct hit timing/accounting paths.

### 5.2 Lookup, seal, refcount, eviction

- [ ] Walk the longest contiguous matching prompt chain, increment every live
  refcount, and move the prefill cursor past exact hit tokens.
- [ ] If the whole prompt is hit, replay only its final token with cache writes
  disabled so attention sees existing K/V and the head obtains valid logits;
  otherwise cap reuse before the final token. Never fabricate logits from KV.
- [ ] Seal and index a private page only after its sixteenth token is committed.
  Permit generated-history pages so later chat turns can reuse them.
- [ ] Keep zero-reference sealed pages in the index and an intrusive LRU/free
  queue; index membership must not pin memory.
- [ ] Evict only zero-reference sealed pages, remove the hash mapping before
  reuse, clear token/hash state, and preserve live references.
- [ ] Handle duplicate concurrently produced hashes without rewriting an
  in-flight append-only block table; collapse duplicates after release.
- [ ] Add metrics for lookup blocks/tokens, hits/misses, live/cached refs,
  duplicate blocks, evictions, and saved prefill tokens.

### 5.3 Prefix acceptance

- [ ] Prove identical, one-block-divergent, partial-tail, different-model,
  different-protocol-version, and different-salt lookup behavior.
- [ ] Compare clean versus prefix-hit outputs for greedy and stochastic
  sampling and tool histories.
- [ ] Stress concurrent hit/cancel/evict/allocation ordering and return exact
  page accounting.
- [ ] Run repeated 4K-token prompts on A40; second execution must skip at least
  99% of complete-page target prefill tokens. Report actual TTFT improvement.
- [ ] Keep prompt tokens, salts, and hash material out of logs/metric labels.

## Milestone 6: GPT-OSS tensor parallelism

Exit: one target model is physically and computationally sharded across real
2x/4x homogeneous CUDA meshes with correct collectives, global sampling,
sharded KV, and measured per-device memory.

### 6.1 Product parallel plan

- [ ] Add `gpt_oss/parallel.rs` and validated `ParallelConfig` supporting TP
  degrees 1, 2, and 4 on one homogeneous mesh axis.
- [ ] Require exact device count, same backend/compute capability, compatible
  memory, and one PJRT client spanning the mesh.
- [ ] Attach product-owned partition metadata before graph lowering; setting
  only execution count is not sufficient.
- [ ] Replicate norms, positions, page tables, routing metadata, and sampling
  controls.
- [ ] Vocabulary-shard embedding with local lookup plus hidden all-reduce.
- [ ] Column-shard Q/K/V over heads and keep Q/K/V activations plus KV cache
  head-local.
- [ ] Row-shard attention output and all-reduce hidden.
- [ ] Keep router replicated; column-shard expert gate/up intermediate and
  row-shard expert down with hidden all-reduce.
- [ ] Reshape logical gate/up output as `[2, intermediate]` and shard the
  intermediate axis so each device owns corresponding gate and up channels;
  reject a flat contiguous split that separates the halves.
- [ ] Vocabulary-shard LM head and implement exact global sampling from the
  union of local top-64 `(logit, global_id)` candidates.
- [ ] Preserve deterministic tie ordering and one replicated per-request RNG
  successor state.

### 6.2 Physical NVFP4 shards

- [ ] Propagate partitions consistently through payload, E4M3 scales, global
  scale, bias, logical shape, and custom-call local ABI.
- [ ] Read checkpoint spans directly into their owning device shard; prove no
  full packed target tensor is uploaded/retained per GPU.
- [ ] Validate every K shard boundary against 16-value scale blocks and every
  N shard against complete output rows.
- [ ] Load each gate/up shard from its two corresponding output-row spans into
  one local compact component without retaining a full packed copy.
- [ ] Accept TP=2 expert-down K=1,440 and TP=4 K=720 as aligned shapes with exact
  accounting.
- [ ] Mark TP=8 `DEFERRED` until the K=360 expert shard has explicit declared
  padding/masking compatible with recipe-v2 scales.
- [ ] Make batch/prefill/decode families include parallel config and compile
  before sharded parameter residency.

### 6.3 TP cache/server integration and evidence

- [ ] Keep one host scheduler/page owner; one logical page ID maps to the
  corresponding local-head page on every device.
- [ ] Replicate metadata and shard K/V buffers over KV heads; verify target KV
  bytes per device scale with TP degree.
- [ ] Exercise continuous batching, prefix hits, cancellation, streaming, and
  tools under TP without adding one server loop per device.
- [ ] Add CPU multi-device numerical placement tests as a permanent compiler
  oracle, without calling them CUDA evidence.
- [ ] Run real 2x and 4x homogeneous CUDA contracts that compare TP=1/2/4
  greedy tokens, stochastic numerical behavior, shards, collectives, and page
  accounting.
- [ ] Capture XLA/Nsight collective evidence and exact per-device parameter/KV/
  temporary bytes.
- [ ] Benchmark throughput, TTFT, TPOT, collective time, topology, and memory
  separately; do not promise a PCIe latency speedup from correctness alone.

## Milestone 7: DFlash speculative decoding

Exit: the pinned 0.8B BF16 DFlash drafter runs inside the same scheduler/cache
system, target verification commits only an accepted prefix with exact RNG and
KV rollback, and A40 evidence decides whether `auto` may enable it by default.

### 7.1 Pin and declare the draft artifact

- [ ] Pin `z-lab/gpt-oss-20b-DFlash` revision
  `d53f6551543204c859e8bbaaddbd15d11b447af9` with exact file/tensor hashes,
  inventory, sizes, config, model card/license, and source provenance.
- [ ] Add immutable materialization/receipt validation separate from the target
  artifact and keep draft weights outside OCI layers.
- [ ] Validate 784,767,104 BF16 parameters; 8 layers; hidden 2,880;
  intermediate 7,680; Q/KV heads 64/8; head dim 64; block size 8; mask ID
  200000; feature taps `[1,6,11,16,21]`; YaRN/context values.
- [ ] Declare exact draft attention/norm/MLP/feature-projection parameters and
  prove embedding/LM head are intentionally reused from target residency.
- [ ] Compile every selected target-tap/project/draft/verify family before
  loading target or draft weights.
- [ ] Reject Python `trust_remote_code`; retain pinned `dflash.py`/`utils.py` as
  reference provenance only.

### 7.2 Target feature and draft graph

- [ ] Add DFlash target variants that expose hidden taps after layers
  1,6,11,16,21, including auxiliary output after the first layer of pairs
  containing layers 6 and 16.
- [ ] Keep tap buffers bounded to the current `[B,Q,2880]` chunk; concatenate
  five taps, apply learned `14400 -> 2880` projection and hidden RMS norm, then
  release taps.
- [ ] During target prefill, immediately ingest each projected chunk into the
  eight-layer draft context K/V cache so no full-prompt tap history remains.
- [ ] Implement eight Qwen3-style draft layers with non-causal attention,
  target-context plus current noise K/V, exact RoPE, Q/K normalization,
  residuals, SiLU MLP, final norm, and target LM head.
- [ ] Form `[pending_token, MASK x7]`, generate seven greedy proposals in one
  draft forward, and crop tentative draft noise K/V while retaining context.
- [ ] Compare bounded projected features, draft hidden states, logits, and
  proposals to the pinned reference implementation.

### 7.3 Target verifier and transactional state

- [ ] Add causal target `Phase::Verify` for eight tokens with feature taps and
  page-aware tentative K/V writes.
- [ ] Store one pending authoritative target token that is sampled but not yet
  emitted or written to target KV.
- [ ] Compare seven proposals against target posterior samples, find accepted
  prefix `L`, emit/commit pending plus L drafts, and retain posterior L as the
  next pending token.
- [ ] Return/select successor sampling state after exactly L+1 target samples;
  unused verifier positions cannot advance visible RNG state.
- [ ] Roll target cache from eight tentative writes to L+1 committed writes and
  keep stale bytes hidden by logical length.
- [ ] Force acceptance lengths 0 through 7 in permanent tests and verify token,
  sampling, target pages, draft pages, reservations, and feature lifetimes.
- [ ] If fewer than eight visible tokens remain, emit pending and switch to
  ordinary decode; never launch past the request token budget.
- [ ] If a Harmony return/call token occurs at pending or any accepted position,
  cut the visible suffix, rollback through the terminal token, and release
  before prefix sealing.
- [ ] Cover cancellation/device error during feature projection, draft,
  verification, result download, and commit.

### 7.4 Draft arena, prefix bundles, batching, TP

- [ ] Allocate/account a separate eight-layer draft context KV arena at 16,384
  bytes/token/device for TP=1 and shard it over KV heads under TP.
- [ ] Tie target and draft logical page ownership transactionally while keeping
  separate physical IDs and memory budgets.
- [ ] Extend the prefix namespace with DFlash artifact/graph identity and admit
  a speculative prefix hit only when both target and draft KV pages exist.
- [ ] Never call a target-only prefix hit DFlash-ready; prefill missing draft
  context or use ordinary target mode explicitly.
- [ ] Compile bounded draft/verify batch families and add scheduler queues for
  requests ready to draft versus verify.
- [ ] Implement `auto` reason codes for short tail, memory pressure,
  incompatible feature, saturated ordinary batch, and low measured acceptance.
- [ ] Make draft execution failure fail the affected request rather than
  silently changing decoding mode.
- [ ] Complete TP=1 acceptance first; run TP=2/4 only after TP=1 performance
  promotes.

### 7.5 DFlash correctness and A40 promotion

- [ ] Compare speculation off/on exact greedy tokens across a broad prompt set.
- [ ] Compare fixed-seed stochastic visible tokens and sampling-state
  progression across mismatches, cancellation, terminal tokens, and tails.
- [ ] Exercise prefix hit/miss, page pressure, batched membership changes, and
  tool calls with speculation.
- [ ] Run Math500, GSM8K, HumanEval, and MT-Bench workloads at concurrency
  `1,4,8,16,32` on the same immutable A40 image with speculation off/on.
- [ ] Report accepted-length distribution, proposed/accepted/rejected tokens,
  draft/verify/rollback time, target calls per visible token, TTFT, TPOT,
  aggregate TPS, cache memory, and auto-disable reasons.
- [ ] Promote DFlash default `auto` only if mean committed tokens per target
  verify is at least 3.5, concurrency-1 end-to-end speedup is at least 1.25x,
  and every lossless/output/cache contract passes.
- [ ] If functional DFlash misses the threshold, keep it opt-in and record the
  measured A40 bottleneck; H200 model-card numbers are not promotion evidence.

## Milestone 8: production hardening and final serving acceptance

Exit: the complete server withstands overload, disconnects, shutdown, prefix
pressure, multi-GPU failure, and long mixed load with truthful operational
metrics and reproducible deployment artifacts.

- [ ] Add `POST /v1/completions` using the same preparation/engine/event path;
  do not fork model execution.
- [ ] Add optional bearer authentication/tenant-to-cache-salt derivation only
  if deployment scope requires it; keep unauthenticated single-tenant mode
  explicit.
- [ ] Add hours-long mixed load with arrival bursts, long prefills, streaming
  clients, tool calls, repeated prefixes, cancellations, and speculation auto.
- [ ] Prove bounded RSS/device memory, stable page totals, stable queue depth
  after load recedes, and no request/event/task leaks.
- [ ] Exercise SIGTERM during startup, idle, prefill, decode, draft, verify, and
  TP collective; verify readiness, client terminal behavior, engine join, and
  exit status.
- [ ] Define and publish supported API fields, exact rejection behavior,
  default server profile, cache isolation mode, TP degrees, and DFlash default.
- [ ] Update `README.md`, `SYSTEM.md`, OCI labels/structure, and deployment
  examples only after the corresponding behavior passes.
- [ ] Produce final BuildBuddy CPU/CUDA/package evidence plus single-A40 and
  2x/4x CUDA reports with exact source/image/model/profile/topology pins.
- [ ] Confirm no non-master branch remains part of the release workflow and
  all durable serving work is reachable from `master`.

## Required verification commands

Use only after the owner requests execution for the coherent milestone.

- CPU product contract:
  `bb test //products/serve:serve_contract_test --config=buildbuddy --config=cpu`
- Affected framework contracts: exact `bb test` targets with
  `--config=buildbuddy --config=cpu`.
- CUDA compilation/contracts: exact `bb test` targets with
  `--config=buildbuddy --config=cuda`.
- Server image:
  `bb build //products/serve:serve_image --config=buildbuddy --config=cuda`.
- Image structure: the existing `serve_image_structure_test` target through
  BuildBuddy/CUDA configuration.
- Real GPU execution: publish the immutable image, run the checked-in remote
  controller on the named hardware, collect complete reports, and terminate
  the paid resource only after artifacts are durable.

## Dependency order

| Milestone | Hard prerequisites | Unblocks |
| --- | --- | --- |
| 0 Step API | current accepted generator | Tokio engine, multi-request state |
| 1 Tokio shell | owned sessions and steps | external clients, cancellation, metrics |
| 2 Global pages | page-aware cache writes | continuous batching, prefix, speculation |
| 3 Continuous batching | global pages and batched sampling | scalable server, DFlash batching |
| 4 Tool calling | HTTP + owned Harmony sessions | production agent conversations |
| 5 Prefix caching | sealed global page ownership | prefill elimination, DFlash bundles |
| 6 Tensor parallel | batched model and global cache | multi-GPU capacity/execution |
| 7 DFlash | batching, rollback, prefix/page transactions | lossless target-call reduction |
| 8 Hardening | all selected features accepted | release-quality inference server |

Work may prepare a later pure schema/reference contract early, but no feature
may claim execution before its hard prerequisites pass. In particular,
non-identity cache writes precede batching, sealed refcounts precede prefix
reuse, and transactional rollback precedes DFlash.
