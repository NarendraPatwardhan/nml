# NML inference server implementation plan

Status: implementation-grade roadmap for converting `products/serve` from one
blocking generation call into a production-shaped GPT-OSS inference server.

This document replaces the completed A40 kernel-optimization investigation.
The accepted single-request implementation is the performance control for the
server work; it is not discarded. [`SYSTEM.md`](./SYSTEM.md) remains the
governing architecture and [`TASKS.md`](./TASKS.md) is the executable checklist.

## 1. Objective and definition of done

Build one long-running Rust inference server around the selected GPT-OSS 20B
NVFP4 product with:

- Tokio/Axum/Tower request handling and OpenAI-compatible chat streaming;
- continuous batching with chunked prefill and decode-first scheduling;
- one process-wide paged KV arena instead of one cache allocation per request;
- automatic prefix caching over immutable full KV pages;
- real Shardy/XLA tensor parallelism across homogeneous NVIDIA GPUs;
- complete GPT-OSS Harmony tool-call input/output behavior without executing
  user tools inside NML; and
- lossless speculative decoding with the pinned
  [`z-lab/gpt-oss-20b-DFlash`](https://huggingface.co/z-lab/gpt-oss-20b-DFlash)
  drafter.

The server is complete only when all of the following are true:

1. Multiple clients can submit, stream, cancel, and complete requests through
   the HTTP boundary while one dedicated engine owner retains every PJRT/XLA
   object.
2. Every engine iteration may advance a different set of active requests; a
   finished or cancelled request leaves the next iteration without rebuilding
   or recompiling the model.
3. KV memory is allocated once, divided into 16-token physical pages, assigned
   through arbitrary per-sequence block tables, and reclaimed without copying
   unaffected K/V data.
4. Prefix hits skip target-model prefill for every complete matching page and
   never permit one namespace to observe another namespace's cached content.
5. Tensor-parallel execution loads physical parameter shards, runs one SPMD
   executable over the declared mesh, performs the required collectives, and
   produces the same model result as the one-device path. Replicating the full
   model on every GPU is not tensor parallelism.
6. OpenAI chat requests can declare function tools, generated tool calls are
   returned with a `tool_calls` finish reason, and subsequent tool results can
   be rendered back into the exact Harmony history. NML never invokes the tool.
7. DFlash drafts seven tokens in parallel, the target verifies an eight-token
   block, only the accepted prefix is committed, and the visible distribution
   and sampling-state progression match ordinary target decoding.
8. The immutable A40 single-stream control remains at least 150 decode-loop
   tokens/s, or any regression is isolated, explained by a named server cost,
   and repaired before promotion.
9. Concurrency, cache reuse, tensor parallelism, and speculation are each
   accepted by permanent contracts in a venue that really executes the
   claimed hardware path. Compilation alone is never reported as execution.

## 2. Governing constraints

These are architecture constraints, not optional implementation preferences.

- `products/serve` owns admission, scheduling, batching, global page leases,
  transport, cancellation, protocol adaptation, and serving metrics.
- GPT-OSS owns its exact artifact, Harmony token semantics, checkpoint schema,
  graph families, target hidden-state taps, and model-specific sharding.
- Framework crates gain only reusable mechanisms such as batched sampling,
  page-aware cache update, or safe tokenizer ownership. They must not learn
  request IDs, HTTP schemas, GPT-OSS layer numbers, DFlash artifact names, or
  scheduling policy.
- Tokio owns sockets, timers, signals, cancellation, and bounded channels.
  XLA compilation and PJRT execution never run on Tokio worker threads.
- One dedicated OS thread constructs and destroys `Platform`, loaded
  parameters, executables, cache buffers, and the scheduler. The async side
  holds only a bounded command sender and per-request event receivers.
- All executable families are compiled before target or draft parameters
  become resident. A request never invokes XLA compilation.
- Static shapes remain finite and explicit. Runtime batching selects the
  smallest precompiled batch/query family and masks inactive rows; it does not
  create an unbounded family cross-product.
- The current recipe-v2 NVFP4 representation, fused QKV, expert kernels,
  five-pair single-stream lookahead, and compact model artifact remain the
  control. Serving work does not reopen the quantization format.
- Page allocation policy cannot change attention semantics or CUDA tile width.
  GPT-OSS pages remain 16 tokens; attention compute tiles remain independently
  bounded.
- Prefix caching shares only complete immutable pages. A partial tail page is
  private to one sequence and can be overwritten after rollback.
- Tool calling means protocol rendering, parsing, and transport. It does not
  authorize arbitrary server-side tool execution.
- DFlash custom Python is a readable reference, not a runtime dependency.
  NML pins the artifact and reimplements its exact inference graph in Rust/NML.
- No local compile, test, benchmark, format, image build, or model execution is
  part of this plan. Routine gates use `bb` with `--config=buildbuddy` and the
  truthful CPU/CUDA configuration; real CUDA evidence consumes a published
  immutable image on the appropriate GPU venue.

## 3. Baseline that must survive

The server starts from commit `fb415a8dadd51a0053b9be314faa836e2b274721`
and image digest
`sha256:3c81704ea85512df7ff76de83ea21f403ef592dc50c23c5ae20e8d70c1e7f3ff`.
The exact A40 report is retained as operator evidence outside the source
snapshot; raw profiler reports and remote-control scripts are not product
source and are not committed under `references/`.

| Control metric | Accepted value |
| --- | ---: |
| Prompt tokens | 106 |
| Generated tokens | 320 |
| Steady-device throughput | 156.062 tokens/s |
| Device-decode throughput | 155.644 tokens/s |
| Complete decode-loop throughput | **151.324 tokens/s** |
| Decode device time | 2,049.544 ms |
| Token download time | 58.517 ms |
| Complete decode loop | 2,108.061 ms |

The current implementation now provides:

- compile-before-residency and reusable embedding/layer/head executables;
- finite prefill and cache profiles;
- 16-token paged-attention reads through an I32 page table;
- persistent donated K/V buffers with cache truncate/rollback/replay
  primitives in the runtime;
- asynchronous PJRT enqueue/dependency chaining;
- exact explicit-state top-k/temperature/top-p/min-p sampling;
- Shardy mesh, physical shard loading, and CPU multi-device evidence;
- package-private Harmony conversation rendering and strict incremental output
  parsing, including tool calls and tool results; and
- A40-accepted Triton fused QKV and compact expert decode kernels;
- a Tokio/Axum OpenAI chat server with bounded admission, streaming,
  cancellation, shutdown, readiness, and Prometheus metrics;
- one global paged K/V arena with arbitrary page-aware reads/writes,
  reservations, rollback, reclamation, and exact host accounting;
- continuous decode-first batching with chunked prefill, dynamic row repacking,
  provisional `B={1,2,4,8}`/`Q={16,128,256}` families, per-row sampling, and
  inactive-row preservation;
- mask-aware routed MoE that excludes inactive prompt positions and batch rows
  from the expert assignment schedule;
- a paired generic Triton paged K/V append whose donated results alias the
  process-wide cache buffers on CUDA;
- one generic stable decode lane for every retained batch family, with a
  donated device-resident batch slab and five-layer-pair lookahead;
- direct stable-batch continuation until membership, page tables,
  cancellation, deadline, backpressure, or shutdown requires replanning;
- reusable process-lifetime executable bindings and per-family compact result
  workspaces;
- complete Harmony tool schema/history/output handling without server-side tool
  execution; and
- a serving-only compact control/result ABI: every token, page-table entry,
  length, row mask, sampling scalar, and RNG word occupies a typed contiguous
  section of one U8 input slab, and token plus RNG state return in one
  20-byte-per-row result buffer.

The current implementation does **not** yet provide:

- prefix ownership/refcounts/eviction;
- model-specific GPT-OSS tensor partitions on CUDA;
- a DFlash artifact, graph, cache, verifier, or scheduler.

The identity-only write trap has been removed: every serving layer performs a
page-aware update indexed by physical page and page offset. Prefix sharing must
still wait for immutable sealed-page identity, reference ownership, and
eviction policy; correct page indirection alone is not a prefix cache.

### 3.1 Continuous-batching A40 evidence and current repair

The first immutable server image accepted 45/45 requests, reclaimed all 2,389
cache pages, and produced real decode families B1/B2/B4/B8. Aggregate output
throughput scaled from 27.98 tokens/s at concurrency 1 to 72.98 tokens/s at
concurrency 8 (2.61x), proving scheduler membership and compact-kernel dispatch
worked. It did not satisfy the absolute performance objective.

Nsight correlated the gap to the serving control path rather than NVFP4
compute correctness:

- every B1 decode layer launched the same accepted compact gate/up, down, QKV,
  and linear GEMV kernels as the legacy path;
- the legacy device path remained 157.21 tokens/s on the same image;
- each server step uploaded fourteen small buffers, downloaded token/RNG/
  position separately, and repeatedly rebound their addresses through every
  layer-pair graph;
- small-M B2/B4/B8 matrix kernels also remain slower than their target
  aggregate throughput warrants; and
- compiling the full initial family cross-product delayed readiness by roughly
  thirteen minutes.

The first repair is implemented in the serving graph ABI without changing the
legacy single-request ABI or NVFP4 representation. Tokens, positions, lengths,
masks, page tables, sampling scalars, and explicit RNG state are arranged as
typed contiguous sections of one U8 host slab, uploaded once, and sliced and
bitcast inside each serving graph. Deterministic position readback is removed;
token plus two U64 RNG words are bitcast into one 20-byte-per-row result. The
hot serving step therefore moves from 14 H2D plus 3 D2H transfers to exactly
1 H2D plus 1 D2H transfer at the product ABI. CPU and CUDA BuildBuddy contracts
pass.

The correlated A40 follow-up reached 117.113-118.553 end-to-end output
tokens/s for the 106+320 workload, or roughly 132-136 decode tokens/s after
TTFT. Nsight showed that compact NVFP4 compute did not regress: useful kernels
took 6.615-6.645 ms/token versus 6.722 ms/token in the accepted single-request
trace. The remaining loss was a 0.522-0.621 ms/token GPU hole at the exact
boundary between the five serving-lookahead pairs and the following seven
pairs. The synchronous serving transaction returned through output decoding,
page accounting, event delivery, scheduler requeue/replan, batch reconstruction,
and a fresh slab upload before submitting that suffix.

The B1-specific continuation architecture has now been removed from the
server. The current repair is one generic stable-batch data plane:

- the serving head returns the compact token/RNG result and a donated next
  batch slab;
- that slab advances token, RNG, position, and sequence length on device;
- stable membership for any retained B family therefore performs zero steady
  H2D transfers and one compact B*20-byte D2H;
- after submitting the result download, the engine immediately queues the next
  embedding and five layer pairs from the donated slab, overlapping visible
  token handling with useful GPU work;
- the engine continues the same batch without scheduler re-entry while all
  rows survive and no command, cancellation, deadline, shutdown, backpressure,
  page-table change, or membership change requires replanning;
- a membership change occurs at a visible-token boundary and selects the next
  smallest generic family over the same process-wide K/V arena;
- compiled family arguments remain process-resident and every family owns a
  reusable compact result-download workspace; and
- the old `SingleSequenceDecodeLane` and its RNG export/import transition have
  been deleted rather than retained as a second serving route.

Two trace-correlated compute repairs are included in the same current phase:

- inactive `[B,Q]` positions now enter routed MoE as masked assignments, receive
  expert ID `-1`, create no expert schedule entries, and return exact zero; and
- sparse masked B1/B2 decode compacts valid routes with one scan and two small
  scatters, preserving one grouped block per selected route instead of
  regressing to the full per-expert scheduler; and
- K and V page writes lower together to one generic Triton custom call on
  CUDA, resolving the physical page once and aliasing both donated cache
  buffers.

The prefill family set also includes Q128, routing the 106-token control prompt
to 128 positions rather than 256. This complements mask-aware MoE by reducing
padding in dense QKV/projection operations.

BuildBuddy contracts prove the portable semantics, Triton construction,
serving graph aliasing, scheduler selection, and complete CUDA build. They are
not runtime evidence. The immediate promotion gate is an immutable A40 run
covering the 106+320 C1 end-to-end control and the provisional B1-B8 concurrency
matrix. Nsight must confirm the compact append, masked expert work, zero steady
H2D contract, and removal of the recurring orchestration hole. No later
roadmap capability may be credited toward this gate.

The first image containing the paired append, digest
`sha256:5dda558dd3c016cff514f4c72726648b6f383a0e582bc9ebe0dfe9475541b211`,
failed readiness before executing a request. Its A40 log identified an invalid
LLVM `i8 -> <1 x i1>` bitcast for the append's Bool mask pointers. XLA stores
StableHLO predicates as one byte per element at a Triton custom-call boundary;
the TTIR ABI incorrectly declared I1 storage pointers. The kernel now declares
I8 storage pointers, converts loaded 0/1 bytes to register predicates, and the
typed custom-call boundary explicitly permits this Bool-storage ABI. This adds
no mask-conversion graph or CUDA launch. BuildBuddy Triton, IR, serve, and full
CUDA construction gates pass; a replacement A40 image remains the runtime
proof.

## 4. External reference facts and what they do not prove

The DFlash artifact is pinned for planning at Hugging Face revision
`d53f6551543204c859e8bbaaddbd15d11b447af9`. Its current model card and custom
code establish this exact contract:

- 784,767,104 BF16 parameters, approximately 1.57 GB of payload;
- eight Qwen3-style full-attention draft layers;
- hidden size 2,880, intermediate size 7,680, 64 query heads, eight KV heads,
  and head dimension 64;
- target feature taps after GPT-OSS target layers `[1, 6, 11, 16, 21]`;
- concatenation of the five target features followed by a learned
  `5*2880 -> 2880` projection and RMS normalization;
- mask token ID `200000`;
- block size eight: one already-authoritative token plus seven drafted tokens;
- non-causal draft attention over cached target-context features and the
  current mask/noise block; and
- reuse of the target token embedding and target LM head.

The authors report mean accepted lengths of 4.2-5.1 and 1.7-2.2x end-to-end
speedups from concurrency 1 through 32, but those results used BF16 target
execution on one H200. They justify implementation and an A40 experiment; they
do not prove an A40 NVFP4 speedup. NML promotion uses its own target artifact,
sampler, batching policy, and A40 measurements.

Relevant primary references:

- [DFlash GPT-OSS model card](https://huggingface.co/z-lab/gpt-oss-20b-DFlash)
- [DFlash paper](https://arxiv.org/abs/2602.06036)
- [DFlash reference implementation](https://github.com/z-lab/dflash)
- [vLLM chunked-prefill and parallelism documentation](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [vLLM prefix-cache design](https://docs.vllm.ai/en/stable/design/prefix_caching/)

Ideas taken from these references are re-expressed through NML's ownership,
artifact, tokenizer, XLA, PJRT, and permanent-test contracts. Their API or
runtime architecture is not copied wholesale.

## 5. End-state architecture

```text
HTTP client
    |
    v
Axum route -> request validation -> Harmony render/tokenize
    |                                  |
    | bounded EngineCommand::Submit    | owned incremental parser
    v                                  v
Tokio control plane <----------- bounded token/event stream
    |
    | bounded command queue + cancellation tokens
    v
dedicated engine OS thread
    |
    +-- admission queue and request state machine
    +-- continuous-batch scheduler
    +-- global target KV page arena
    +-- optional DFlash KV page arena
    +-- prefix hash/refcount/LRU index
    +-- resident target and draft executables/parameters
    +-- one PJRT client spanning the selected TP mesh
    |
    v
batched prefill / decode / verify / draft executable submissions
    |
    v
one compact token-result download per scheduled batch
```

### 5.1 Ownership table

| Owner | Long-lived state | Forbidden state |
| --- | --- | --- |
| Axum handler | validated request, request ID, response assembler/SSE writer, cancellation guard | PJRT buffers, executable handles, scheduler/page state |
| Protocol preparation | structured Harmony conversation, exact token IDs, owned response parser | model buffers, HTTP socket ownership |
| Tokio server state | bounded command sender, readiness state, metrics handles, shutdown token | direct XLA/PJRT calls |
| Engine thread | platform, target/draft model, executables, scheduler, request token state, page arenas, prefix index | socket writes or unbounded waits on clients |
| GPT-OSS executor | model-specific batched graphs, parameter binding, feature taps, stop-token semantics, TP partition plan | HTTP status codes or queue policy |
| Framework crates | tensors, page-aware update semantics, batched sampling, sharding, buffers, executable calls | GPT-OSS/DFlash names or request scheduling |

### 5.2 Bounded communication

Use Tokio bounded channels only:

```rust
enum EngineCommand {
    Submit {
        request: PreparedInferenceRequest,
        events: mpsc::Sender<EngineEvent>,
        cancellation: CancellationToken,
        admitted: oneshot::Sender<Result<RequestId, AdmissionError>>,
    },
    Cancel { request_id: RequestId, reason: CancelReason },
    Snapshot { reply: oneshot::Sender<EngineSnapshot> },
    Shutdown { deadline: Instant },
}

enum EngineEvent {
    Token { token_id: u32, index: usize, timing: TokenTiming },
    Finished { reason: FinishReason, usage: Usage },
    Failed { code: EngineErrorCode, message: String },
}
```

Starting capacities are explicit configuration, not hidden constants:

- engine command queue: 1,024;
- per-request event queue: 64;
- admitted/queued requests: 1,024;
- maximum active sequences: 32 for the first A40 profile;
- maximum batched tokens per scheduler iteration: 4,096;
- maximum prefill chunk: 256 tokens per request per iteration.

These are initial acceptance values. A benchmark may change them only together
with a checked-in profile and before/after evidence. Queue saturation returns
HTTP 429; it never grows an unbounded allocation.

The engine uses `try_send` for response events. A full per-request stream queue
marks the request `client_backpressure`, cancels it, and releases its pages;
the GPU owner never blocks behind a slow or disconnected socket.

### 5.3 Startup and shutdown

1. Parse and validate all server/model/parallel/cache profiles.
2. Spawn the engine thread.
3. On that thread, construct the platform and topology, validate both artifact
   manifests, declare all target/draft parameters, freeze the configured
   physical-page count from the explicit cache budget and declared memory
   accounting, compile every selected family with that count, load parameters
   once, validate remaining memory, allocate cache arenas, and warm one
   execution of each hot family. Cache sizing never triggers a post-residency
   recompile.
4. Send a startup result through a one-shot channel.
5. Bind the public socket only after startup succeeds, or bind but keep
   `/readyz` false and reject inference until the result arrives. Pick one
   behavior and test it; the preferred behavior is bind early, readiness false.
6. On SIGTERM/SIGINT, stop admission, fail queued requests, let active requests
   run until the configured grace deadline, then cancel the remainder.
7. Join the engine thread so PJRT buffers, executables, parameters, platform,
   and plugin state are destroyed in their valid order.

## 6. Request and scheduler data model

### 6.1 Identifiers and immutable request input

- `RequestId`: monotonic process-unique 128-bit/display-safe ID.
- `SequenceId`: engine-private ID; one request initially owns one sequence.
- `PreparedInferenceRequest`:
  - exact rendered token IDs;
  - maximum new tokens;
  - per-request explicit sampling parameters and seed;
  - stop-token set fixed by GPT-OSS Harmony;
  - optional deadline;
  - optional prefix-cache salt;
  - speculation policy (`auto`, `disabled`, later `required` only for tests);
  - prompt byte/token accounting and API metadata that does not enter graphs.

Sampling state is part of the sequence, not global engine state. Batching must
not change one request's random stream when another request arrives, leaves,
or changes batch slot.

### 6.2 Request state machine

```text
Received
  -> Queued
  -> Admitted
  -> PrefixMatched
  -> Prefilling <-> QueuedForPrefill
  -> Decoding <-> QueuedForDecode
  -> ToolCall | Completed | LengthLimited

Any nonterminal state
  -> Cancelled | DeadlineExceeded | Failed
```

`RequestState` retains:

- prompt tokens and prefill cursor;
- generated token IDs and remaining visible budget;
- logical sequence length and independently tracked tentative length;
- sampling state;
- target block table and private tail-page state;
- optional DFlash block table, pending authoritative token, accepted-feature
  buffers, and rolling acceptance statistics;
- admission reservation credits;
- arrival, admission, first-scheduled, first-token, and last-token timestamps;
- cancellation token and response event sender.

Terminal transition is idempotent and centralized. It must:

1. emit at most one terminal event;
2. release target and draft page references exactly once;
3. return unused reservation credits;
4. remove the request from every scheduler queue/slot; and
5. update counters before dropping the event sender.

### 6.3 Batch plan

The scheduler produces an immutable host plan before touching device state:

```rust
struct BatchPlan {
    phase: BatchPhase,
    family: BatchFamily,
    slots: Vec<ScheduledSequence>,
    token_count: usize,
    page_allocations: Vec<PageMutation>,
    rollback_points: Vec<CacheCheckpoint>,
}

enum BatchPhase {
    Decode,
    Prefill,
    Draft,
    Verify,
}
```

The plan validates all pages, lengths, positions, token budgets, and output
capacity before any mutation. Applying it is transactional at the host metadata
level: if device submission fails, affected requests fail and all uncommitted
leases are returned from their checkpoints.

### 6.4 Scheduling policy

One engine iteration:

1. Drain at most a bounded number of commands so request floods cannot starve
   already-running generations.
2. Observe cancellation/deadline flags and finalize those requests.
3. Evict zero-reference cached pages if admission needs reservation credits.
4. Admit queued requests in FIFO order when their prompt plus worst-case output
   page reservation can finish without mid-generation OOM.
5. Schedule every eligible ordinary decode request first, up to
   `max_num_sequences` and the per-iteration token budget.
6. Schedule DFlash draft/verify work according to its separate cost and
   acceptance policy; do not mix a draft graph with a target decode graph.
7. Spend the remaining token budget on oldest prefills. Chunk any prefill that
   does not fit; do not defer the whole request.
8. If the oldest prefill has waited beyond `max_prefill_wait`, reserve at least
   one prefill chunk even under sustained decode load.
9. Select the smallest compiled family whose batch/query capacities cover the
   plan, pad unused rows/tokens, and mark them inactive.
10. Execute decode before prefill for latency. Prefill and decode may occur in
    the same scheduler iteration as separate executable submissions; they do
    not need to be one mixed-shape graph to qualify as continuous batching.
11. Download one compact result buffer for the batch, scatter token/state
    results to requests, parse stop conditions, and commit/release metadata.
12. Immediately form the next iteration; no request owns the engine loop.

Decode-first must not mean prefill starvation. Permanent scheduler simulation
tests cover sustained decode arrivals, an aged long prefill, cancellations,
and page pressure with deterministic virtual time.

## 7. Phase A: step-wise engine and Tokio serving shell

This phase changes ownership without yet claiming continuous batching.

### 7.1 Refactor generation into steps

Replace `ResidentModel::generate` as the engine primitive with:

- `prepare`: select an existing profile and initialize request state;
- `prefill_step`: consume one bounded prompt chunk and return the first token
  only when the full prompt is complete;
- `decode_step`: consume the prior device token and advance one visible token;
- `finish`: truncate/finalize protocol state and release cache leases.

Retain a blocking single-request adapter for permanent equivalence tests, but
implement it by driving the same step API. There must not be a second model
execution path.

### 7.2 Make Harmony sessions independently owned

The current `HarmonyParser<'tokenizer>` borrows its tokenizer, which prevents
many live parsers from being stored or moved independently. Refactor the
generic tokenizer owner so a decoder retains shared ownership of the immutable
tokenizer allocation. Then:

- `Tokenizer` is cheaply cloneable through one audited shared inner owner;
- `Decoder` owns that shared handle rather than a borrow;
- the underlying tokenizer is freed after the final encoder/decoder;
- concurrent encoder/decoder use is enabled only if the IREE bridge contract
  and a permanent concurrency test prove it safe; otherwise calls are
  serialized behind the narrow tokenizer owner; and
- `HarmonyParser` becomes an owned request object with no self-reference or
  leaked lifetime.

Render/tokenize on a bounded CPU preparation path before engine admission.
Response tasks own the parser and translate raw token events into HTTP events.
The engine remains token/protocol-ID aware only where GPT-OSS stop semantics
require it.

### 7.3 Introduce the async server

Replace the one-shot default binary with a `serve` command. Keep a clearly
named diagnostic `generate` command only if the existing RunPod acceptance
harness still needs it; both commands must use the same executor.

Initial routes:

- `GET /healthz`: process is alive; no GPU claim.
- `GET /readyz`: target model, configured executables, cache arena, and optional
  drafter are resident and the engine accepts commands.
- `GET /metrics`: Prometheus text format.
- `GET /v1/models`: selected served model identity and capabilities.
- `POST /v1/chat/completions`: streaming and non-streaming chat completion.
- `POST /v1/completions`: text-prompt compatibility after chat is stable.

Use Tower body-size, concurrency, timeout, and load-shed layers. Request body
and admission timeout are bounded; generation duration is governed by request
deadline/cancellation rather than an HTTP middleware that can orphan device
work.

SSE requirements:

- exact `text/event-stream` framing;
- one stable completion ID and created timestamp;
- first role delta, content/reasoning/tool deltas, one finish delta, usage when
  requested, and terminal `[DONE]`;
- disconnect drops a guard that cancels the engine request;
- a non-stream response is assembled from the same internal events; and
- no prompt, generated content, tool arguments, or cache salt enters logs by
  default.

Phase exit: two simultaneous HTTP clients can queue, stream, cancel, and
complete through the dedicated engine owner, even if execution is temporarily
serialized one request at a time. The old CLI and new HTTP path produce the
same fixed-seed tokens/events.

## 8. Phase B: global paged KV arena

Continuous batching and prefix caching depend on this phase.

### 8.1 Physical layout

Allocate one target K buffer and one target V buffer per GPT-OSS layer:

```text
[physical_pages, 16, local_kv_heads, 64] BF16
```

All 24 layers use the same physical page ID namespace. Logical page `j` for a
request maps to physical page `block_table[j]` in every layer's K/V arena.
One host `PageDescriptor` therefore owns the corresponding page across all 48
target buffers.

Target KV bytes per token at TP=1 are exact:

```text
24 layers * 2 (K,V) * 8 KV heads * 64 * 2 BF16 bytes = 49,152 bytes/token
49,152 * 16 = 786,432 bytes/physical page
```

At tensor parallel degree `P`, KV heads are sharded and target cache bytes per
device are divided by `P`. Page metadata and block tables are replicated.

At startup:

1. Require an explicit cache byte budget, or derive a conservative budget from
   the platform's pre-compile memory report, declared target/draft resident
   bytes, and an explicit compiler/temporary safety margin.
2. Convert that budget into a whole physical-page count **before graph
   compilation** and freeze the count into every serving family.
3. Compile all families while parameters and cache buffers are absent, then
   load parameters once.
4. Recheck actual remaining memory against the already frozen cache budget and
   safety margin. If it does not fit, fail readiness; do not shrink the arena
   and compile another family after residency.
5. Require enough pages for at least one maximum configured prefill chunk plus
   one output page; otherwise fail readiness with exact accounting.
6. Allocate every per-layer arena buffer once. Requests allocate metadata and
   page leases, never K/V tensors.

### 8.2 Page-aware write operation

Add a model-independent page update operation with semantics:

```text
cache[page_table[b, position[b] / page_size],
      position[b] % page_size,
      :, :] = update[b, :, :]
```

The batched form accepts:

- physical cache `[P, S, Hkv, D]`;
- updates `[B, Q, Hkv, D]`;
- block table `[B, L]`;
- starting positions `[B]`;
- valid query lengths `[B]`;
- active rows `[B]`; and
- an optional per-query write mask for query-only replay of an already cached
  final prompt token.

It validates every used page ID before launch, skips padded/inactive tokens,
supports a query chunk crossing page boundaries, donates and aliases cache
storage, and has a portable StableHLO implementation. Optimized Triton update
is added only if profiling shows the portable scatter is material.

Replace the dense `reshape + dynamic_update_slice` in GPT-OSS with this
operation. Keep paged-attention reads unchanged. Permanent tests use
permuted/non-contiguous page tables so the old identity-only behavior cannot
pass accidentally.

### 8.3 Host page manager

Preallocate one descriptor per physical page:

```rust
struct PageDescriptor {
    physical_id: u32,
    ref_count: u32,
    state: PageState,
    valid_tokens: u8,
    block_hash: Option<BlockHash>,
    token_ids: [u32; 16],
    lru_links: LruLinks,
}

enum PageState {
    Free,
    Private { sequence: SequenceId },
    Sealed,
}
```

Invariants:

- a free page has refcount zero and no request block-table reference;
- a private page has exactly one owning sequence and may be partial;
- only a complete page can become sealed/shareable;
- a sealed page is immutable; appending allocates the next page;
- `ref_count` equals live block-table references, not cache-index membership;
- a sealed zero-reference page remains cacheable and evictable;
- allocation may evict only sealed zero-reference pages;
- rollback never changes bytes in an earlier sealed page;
- cancellation releases pages in reverse logical order and cannot double-free;
- host logical/tentative lengths, not stale device bytes, define visibility.

### 8.4 Admission reservations

Do not fail a request halfway through its declared output because another
request consumed the last page. The first implementation reserves page credits
for:

- prompt pages not satisfied by prefix hits; plus
- the request's worst-case remaining output pages, accounting for its current
  partial private tail.

Credits guarantee finish but physical pages are assigned lazily. Zero-reference
cached pages count as evictable capacity. Requests that cannot reserve wait in
the admission queue. Later preemption/recompute may improve utilization, but
it is not a prerequisite for a correct first server and must not be faked by
allowing mid-request OOM.

### 8.5 Device metadata

For each batch, construct:

- compact tokens;
- positions and sequence lengths;
- active/query-length masks; and
- a block-table matrix padded with `-1` only after the last used logical page.

Upload metadata once per batch, replicated over a TP mesh. Select the smallest
batch family and retain one configured logical-page capacity rather than
compiling every possible current length. Measure metadata upload bytes/time.
If it exceeds 2% of steady decode time, retain stable device block-table rows
per request and update only changed entries through a small scatter executable.

Phase exit: multiple request block tables use deliberately permuted physical
pages through every layer, produce the dense-cache oracle result, survive
truncate/replay/cancel, and return all pages to the initial accounting state.

## 9. Phase C: continuous batching

### 9.1 Static batch families

Extend `ShapeFamily` into a serving family containing:

- phase (`prefill`, `decode`, later `verify`/`draft`);
- batch capacity;
- query capacity;
- configured logical-page capacity;
- process-wide physical-page count; and
- tensor-parallel configuration.

Parity A40 family set:

- batch capacities: `1, 2, 4, 8`;
- decode query capacity: `1`;
- prefill query capacities: `16, 128, 256`;
- one operator-selected maximum model length/page-table width; and
- one exact physical-page count derived at startup.

This reduces serving compilation from 30 to 16 families while retaining the
106-token Q128 control, B2-B8 continuous batching, and Q256 chunked prefill.
After the generic B1 path recovers at least 150 end-to-end tokens/s, expand the
same family mechanism to the production envelope. The family count remains
bounded, auditable, and logged; do not compile a batch x every prompt bucket x
every sequence-length bucket cross-product.

### 9.2 Batched model shapes

Refactor every target component:

- tokens: `[B, Q]`;
- hidden: `[B, Q, 2880]`;
- positions: `[B]` start plus an internal query iota;
- query lengths: `[B]`;
- logical sequence lengths: `[B]`;
- active rows: `[B]`;
- page tables: `[B, logical_pages]`;
- sampling states: `[B, 2]`;
- per-row top-k, temperature, top-p, and min-p: `[B]`; and
- output tokens/states: `[B]` / `[B, 2]`.

Flatten only where the semantic operation requires `[B*Q, ...]`, and restore
batch/query axes afterward. Inactive rows must not route experts, mutate KV,
advance RNG, or emit tokens.

Extend generic sampling so each row has independent dynamic controls and RNG.
The selected token for request A must be invariant when request B is inserted,
removed, cancelled, or assigned another slot.

### 9.3 Batched compact kernels

Correct portable execution comes first, but CUDA promotion requires retained
compact weights throughout:

- batch 1 continues to use the accepted M=1 fused-QKV/GEMV/expert path;
- small `M=B*Q` batches use a Triton family that shares weight loads across
  rows where profitable instead of launching B independent one-row kernels;
- routed MoE flattens active tokens, builds one assignment schedule, and
  launches only selected expert blocks;
- padded rows produce no assignments and no weight traffic;
- head projection computes only active rows and preserves the global top-64
  sampling contract; and
- kernel selection is based on exact M/geometry/capability, never a silent
  dense BF16 expansion.

Benchmark each retained B family. A family that is slower than issuing its
active requests separately must not be selected until corrected.

### 9.4 Dynamic membership and fairness

After every batch result:

- remove terminal/cancelled rows;
- keep nonterminal request state independent of its prior slot;
- admit and insert new requests immediately;
- select the next smallest family for the new cardinality; and
- preserve FIFO within equal phase/priority plus aging across phases.

No CUDA graph, executable arguments object, or cache table may capture a
request's slot as permanent identity.

### 9.5 Preserve low-latency execution in the generic path

Stable membership at any batch size retains the complete batch slab and the
bounded five-layer-pair prefix on device. The serving head donates the updated
slab, the next prefix is enqueued while the compact result downloads, and the
engine bypasses remove/requeue/replan until useful work or a lifecycle event
changes membership.

B1 is therefore the smallest instance of the ordinary mechanism:

- zero steady H2D and one compact result D2H;
- the same process-wide physical cache and paired page append as B2-B32;
- the same token/RNG/position advancement in the donated device slab;
- the same visible-token-boundary checks for command arrival, cancellation,
  deadline, shutdown, and output backpressure; and
- the same family re-selection when a request joins or leaves.

There is no RNG export/import bridge, private B1 scheduler, or second cache
owner. The remaining promotion gate is the real A40 measurement: do not call
the regression recovered until the exact C1 end-to-end control reaches at
least 150 tokens/s and the concurrency matrix proves larger stable batches
retain useful aggregate throughput.

### 9.6 Continuous-batching acceptance

Permanent deterministic contracts:

- batched outputs equal independent single-request outputs for greedy and
  seeded stochastic sampling;
- batch membership changes every step without changing surviving sequences;
- chunked and unchunked prefill produce the same first-token distribution;
- inactive slots never change cache/RNG/accounting;
- cancellation before prefill, during prefill, during decode, and while an
  event queue is full releases all state;
- one long prefill cannot starve active decode, and sustained decode cannot
  starve an aged prefill;
- page reservations prevent mid-request OOM; and
- no request invokes compilation.

Real A40 promotion workload, five warm repetitions per point:

- prompt/output mixes: `128/128`, `1K/128`, and `4K/256`;
- parity concurrency: `1, 2, 4, 8`;
- report request throughput, prompt tokens/s, output tokens/s, TTFT, TPOT,
  end-to-end latency, batch-size histogram, GPU busy time, queue time, and page
  utilization;
- concurrency-8 aggregate output throughput must be at least 1.5x the same
  image's concurrency-1 rate without correctness loss;
- p95 TPOT at concurrency 8 must remain below 2.5x concurrency-1 TPOT; and
- the exact single-stream control must remain at least 150 decode-loop tokens/s.

## 10. Phase D: OpenAI chat and tool calling

### 10.1 Public request schema

Implement strict Serde types rather than passing arbitrary JSON into Harmony.
Initial supported fields:

- `model` (must select the resident GPT-OSS identity/alias);
- `messages` with `system`, `developer`, `user`, `assistant`, and `tool` roles;
- text content only; multimodal content is rejected explicitly;
- `max_tokens` / `max_completion_tokens` with one normalized internal field;
- `temperature`, `top_p`, NML's bounded `top_k` and `min_p` extension;
- `seed`, `stream`, and `stream_options.include_usage`;
- function `tools` with name, optional description, and JSON Schema parameters;
- `tool_choice` initially `none` and `auto`;
- optional request deadline and prefix `cache_salt` extension; and
- `speculative` extension (`auto` or `disabled`).

Reject unsupported fields whose semantics would otherwise be silently wrong:
parallel choices (`n > 1`), logprobs until implemented, multimodal parts,
response-format grammars, and forced/named tool choice until a real constrained
tool decoder exists.

### 10.2 Harmony mapping

- OpenAI system/developer messages map to exact `SystemContent` and
  `DeveloperContent`.
- Function schemas pass through the existing audited JSON-Schema-to-TypeScript
  renderer.
- `tool_choice=none` omits tool definitions; `auto` includes them.
- Assistant text/history maps to the correct Harmony channel records.
- An assistant `tool_calls` entry maps to `Message::ToolCall`.
- A subsequent `role=tool` message must reference a prior call ID and maps to
  `Message::ToolResult` with the matching function name.
- Malformed call IDs, duplicate results, result-before-call, invalid names,
  non-JSON arguments, and unsupported parallel calls fail before tokenization.

GPT-OSS currently terminates one completion at one `<|call|>`, so expose one
tool call per choice. Generate a stable server call ID such as
`call_<request-id>_0`; preserve the function name/recipient separately.

### 10.3 Output behavior

- Final-channel text becomes ordinary assistant `content`.
- Analysis may be exposed through a documented `reasoning` extension; it must
  never be merged silently into final content.
- A complete parsed tool call becomes `message.tool_calls[0]` with type
  `function`, name, and exact raw JSON arguments.
- Because the Harmony parser deliberately withholds incomplete tool JSON, an
  SSE response emits the tool arguments as one complete delta when `<|call|>`
  closes it. Partial invalid JSON is never exposed.
- Finish reasons map to `stop`, `length`, `tool_calls`, `cancelled`, or an
  OpenAI-shaped error before stream completion.
- NML returns the call and stops. No registry, subprocess, HTTP callback, or
  in-process function execution is added.

Tool acceptance includes official Harmony byte fixtures, OpenAI JSON/SSE
fixtures, multi-turn call/result history, prefix hits over identical tool
schemas, early disconnect, malformed model output, and complete absence of a
tool execution side effect.

## 11. Phase E: automatic prefix caching

Prefix caching begins only after the global page manager is correct.

### 11.1 Cache key

Compute a chained SHA-256 for each complete 16-token block:

```text
block_hash = SHA256(
    version ||
    parent_block_hash ||
    exact_16_u32_token_ids ||
    cache_namespace_digest ||
    request_cache_salt
)
```

The namespace digest includes:

- exact target artifact manifest and recipe identity;
- tokenizer file identity and `openai-harmony-gpt-oss-v1`;
- model/attention/RoPE/cache semantic version;
- KV dtype and page size;
- target model/cache semantic fingerprint, explicitly excluding batch slot,
  batch-capacity padding, and other scheduler-only choices; and
- when applicable, exact DFlash artifact/graph identity.

Sampling parameters are not part of target prompt KV identity. Exact rendered
token IDs already include system instructions, tool schemas, assistant/tool
history, dates, and caller content.

Keep the exact block token IDs and parent hash in the descriptor and compare
them after digest lookup. A hash collision is an error/second candidate, never
permission to reuse unrelated KV.

### 11.2 Lookup and allocation

At request preparation:

1. Hash complete prompt blocks from the root.
2. Walk the longest contiguous chain present in the prefix index.
3. Increment each matched page refcount and remove it from the evictable free
   queue while in use.
4. Set the prefill cursor to matched_tokens.
5. Allocate a private page for the unmatched suffix/tail only when scheduled.
6. If the entire prompt is matched, replay only the final prompt token through
   the target with its cache-write mask disabled. Its existing cached K/V
   remains visible to attention, the query reconstructs the final hidden/logits,
   and no duplicate KV entry is appended. If query-only replay is unavailable,
   cap the reusable prefix so at least one prompt token is computed normally;
   never claim a full hit without a valid first-token logit.

When a private page becomes full and all of its tokens are committed, seal it,
assign its chained hash, and insert it into the index. Generated-history pages
may also be sealed because a later chat turn can contain the exact assistant
prefix. Partial pages are never indexed or shared.

### 11.3 Refcounts and LRU eviction

- Index membership does not pin a page.
- Releasing the final live reference appends a sealed page to the LRU tail.
- Allocation first consumes truly free pages, then evicts the LRU head among
  sealed zero-reference pages.
- Eviction removes the hash mapping and clears the descriptor before reuse.
- Reverse logical release order makes late, highly specific blocks older
  eviction candidates before widely shared early blocks.
- Duplicate hashes may temporarily name more than one physical page if both
  were produced concurrently; retain one canonical candidate after requests
  release rather than rewriting an in-flight append-only block table.

### 11.4 Isolation

`cache_salt` participates in the root hash. Requests with different salts
cannot share pages or infer hits through timing. Server configuration may:

- require a salt in multi-tenant mode;
- derive a salt from an authenticated tenant identity when auth is later
  introduced; or
- use one documented global namespace for trusted single-tenant deployment.

Never log salts or block token contents. Metrics expose counts only.

### 11.5 Prefix acceptance

- Second execution of an identical prompt reports exact full-page hit tokens
  and executes only its residual tail/first-token work.
- Prompts differing in one block share only the common parent chain.
- Same tokens under a different model/protocol/salt miss completely.
- Prefix reuse produces the same output as clean prefill for greedy and seeded
  stochastic requests.
- Cancellation/refcount races cannot evict a page still referenced by another
  request.
- Under forced pressure, LRU eviction returns exact page accounting and never
  changes a live request.
- A 4K-token repeated-prefix A40 workload must reduce second-request prefill
  model tokens by at least 99% of complete-page tokens and materially improve
  TTFT; report the exact speedup rather than assigning a guessed threshold.

## 12. Phase F: tensor-parallel GPT-OSS sharding

### 12.1 Supported topology

Start with one homogeneous single-node tensor mesh axis and TP degrees
`1, 2, 4`. GPT-OSS has 64 query heads and eight KV heads, so these degrees
partition attention evenly. TP=8 is deferred until NVFP4 row-shard padding is
explicitly implemented: the 2,880-wide expert intermediate would produce a
360-element K shard, which is not aligned to the recipe's 16-value scale
blocks.

All selected devices must report compatible backend, compute capability, and
memory. One PJRT client/executable spans the mesh. Do not spawn one independent
model server per GPU and call it tensor parallelism.

### 12.2 Partition plan

Use one product-owned `TP_AXIS` and attach partitions to logical parameter and
activation shapes before lowering:

| Component | Tensor-parallel placement |
| --- | --- |
| RMSNorm, positions, page tables, routing logits/IDs | replicated |
| Token embedding vocabulary rows | vocabulary-sharded local lookup, then hidden all-reduce |
| Q/K/V weights and biases | column-sharded over query/KV heads |
| Q/K/V activations and KV cache | local head shards |
| Attention output projection | row-sharded input, hidden all-reduce |
| Router projection | replicated |
| Expert gate/up | column-sharded over intermediate channels |
| Expert activation | local intermediate shard |
| Expert down | row-sharded intermediate input, hidden all-reduce |
| LM head | vocabulary-row sharded |
| Sampling | local top-64 candidates, all-gather candidates, global top-k/filter/sample, replicated token/state |

All 32 experts remain present as tensor-sharded experts in this milestone.
Expert parallelism/all-to-all is a different feature and is not substituted for
the requested tensor parallelism.

GPT-OSS stores gate and up as two contiguous intermediate halves. Treat the
logical output as `[gate_or_up=2, intermediate]` and shard the intermediate
axis, so every device receives matching gate/up channels. A naive contiguous
split of the flattened `2*intermediate` axis would place gate on some devices
and up on others and is invalid.

### 12.3 NVFP4 physical sharding

For every structured parameter component:

- payload, local E4M3 scale, global scale, bias, and logical shape use one
  consistent shard contract;
- checkpoint byte spans are read directly into the owning device shard;
- no full packed tensor is uploaded to every device and sliced afterward;
- N/output-axis shards preserve output-major row boundaries;
- K/input-axis shards begin/end on 16-value scale-block boundaries;
- a gate/up shard reads the two corresponding non-contiguous output-row spans
  from the checkpoint and presents one local `[2*intermediate_local, K]`
  component without constructing a full-device copy;
- TP=2 and TP=4 expert down shards are respectively 1,440 and 720 values and
  therefore aligned;
- any required padding is declared, zero-filled, excluded from logical output,
  and included in exact resident-byte accounting; and
- custom-call ABI validation uses local component shapes while semantic output
  retains global Shardy placement.

### 12.4 Collectives and sampling

Let Shardy/XLA insert collectives from explicit placement where it can prove
the correct result. Add an explicit model-independent collective only if the
semantic graph cannot express the needed global top-k candidate exchange.

Global top-k correctness does not require gathering the full 201,088 logits:
the global top 64 is contained in the union of each shard's local top 64.
Gather candidate `(logit, global_token_id)` pairs, perform stable global
ordering/filtering, sample once from the request's replicated RNG state, and
broadcast the selected token/state. Tie behavior must be deterministic and
match TP=1.

### 12.5 Cache and scheduler interaction

- The scheduler and page manager remain one host owner, not one copy per GPU.
- One logical page ID identifies corresponding local-head pages on every
  device.
- Page tables, sequence lengths, sampling controls, and active masks are
  replicated.
- KV physical storage is sharded over KV heads, reducing per-device cache
  bytes approximately by TP degree.
- Batch membership and prefix hash identity are common across the mesh.
- A collective/device failure fails the whole batch and all affected requests;
  partial shard progress is never exposed.

### 12.6 Tensor-parallel acceptance

On real 2x and 4x homogeneous CUDA hosts:

- exact topology and device assignment are reported;
- physical parameter inventory proves shards rather than replicas;
- per-device resident parameter and KV bytes agree with declared partitions;
- TP=1/2/4 greedy token streams match exactly on fixed fixtures;
- stochastic distributions/logits satisfy the declared numerical contract and
  fixed seeds are reproducible;
- paged allocation, prefix hits, cancellation, streaming, and tools work under
  TP;
- Nsight/XLA evidence shows required collectives execute;
- throughput, TTFT, TPOT, collective time, link topology, and memory headroom
  are reported separately; and
- no speedup is promised on PCIe-only topology merely because memory sharding
  is correct.

## 13. Phase G: DFlash speculative decoding

This phase starts only after ordinary continuous batching, global pages,
prefix caching, and TP=1 are stable. It extends those owners; it does not add a
second server loop.

### 13.1 Artifact and checkpoint

Create a separate immutable artifact contract pinned to revision
`d53f6551543204c859e8bbaaddbd15d11b447af9`:

- authenticate `config.json`, `model.safetensors`, license/model card, exact
  tensor inventory, sizes, hashes, and repository revision;
- materialize outside the OCI image and issue the same read-only filesystem
  identity receipt as the target artifact;
- validate exact architecture values listed in section 4;
- declare the eight draft layers, final norm, target-feature projection and
  hidden norm;
- prove whether the checkpoint intentionally omits token embedding and LM head
  and bind those operations to the resident target weights;
- retain BF16 drafter weights as BF16 initially; do not invent a draft
  quantization project; and
- reject `trust_remote_code`; `dflash.py` and `utils.py` remain provenance
  references for the Rust/NML implementation.

Compilation still precedes both target and draft parameter residency. Startup
memory accounting reports target parameters, draft parameters, target KV,
draft KV, temporary verification buffers, and safety margin separately.

### 13.2 Exact algorithm state

Each speculative sequence owns:

- one `pending_token` sampled authoritatively by the target but not yet written
  to target KV or emitted;
- target committed and tentative lengths;
- target page table;
- eight-layer draft context KV page table/length;
- projected target context features for only the current verification result;
- target and draft cache checkpoints;
- proposed/accepted counters and rolling acceptance EMA; and
- the same target sampling state used by ordinary decoding.

The pending-token formulation matches the reference:

1. Target prefill samples the first pending token while target KV contains the
   prompt only.
2. The draft input block is `[pending, MASK x 7]`.
3. DFlash predicts positions 1-7 in one non-causal forward pass.
4. Target verifies `[pending, draft_1, ..., draft_7]` causally in one
   eight-token forward pass.
5. Target sampling yields posterior tokens for every verifier position.
6. Let `L` be the longest prefix where `draft_i == posterior_{i-1}`.
7. Emit/commit `pending` plus `L` accepted drafts.
8. Retain `posterior_L` as the next pending authoritative token.
9. Roll target KV back from eight tentative writes to `L+1` committed writes.

If all seven drafts match, commit eight tokens and retain the posterior after
the eighth as the next pending token. A pending stop token terminates before a
draft launch.

### 13.3 Target feature taps

DFlash needs target hidden states after zero-based GPT-OSS layers
`1, 6, 11, 16, 21`.

Add DFlash-specific target component variants that return those taps without
changing ordinary output:

- taps after the second layer of a pair are ordinary pair-boundary outputs;
- taps after the first layer of a pair (`6` and `16`) are retained as explicit
  auxiliary outputs before the second layer overwrites/donates hidden state;
- prefill/verify operate in bounded chunks so at most five
  `[B,Q,2880]` tap buffers are live;
- one projector concatenates the five taps, applies the learned
  `14400 -> 2880` weight and hidden RMS norm, then releases taps; and
- no full-prompt five-layer hidden history is retained.

During chunked target prefill, immediately convert projected features into
draft context K/V for all eight draft layers. The DFlash layer computes K/V for
target context directly from the projected target feature; context tokens do
not need to traverse the draft attention/MLP as queries.

### 13.4 Draft graph

For each of eight draft layers:

1. RMS-normalize the current eight noise/mask hidden states.
2. Compute queries from noise hidden.
3. Compute K/V for newly accepted target context features and for the current
   noise block.
4. Append new context and noise K/V tentatively to that layer's paged cache.
5. Apply YaRN RoPE at the exact context/noise positions.
6. Run non-causal GQA over previous context, newly appended context, and all
   eight current noise positions.
7. Apply output projection, residual, post-attention norm, Qwen3 SiLU MLP, and
   residual.
8. After all layers, apply final norm and the target vocabulary head to the
   last seven positions.
9. Select greedy draft candidates; target sampling remains authoritative.
10. Crop draft logical cache length to retain context K/V and discard noise
    K/V. Stale bytes remain invisible and are overwritten next iteration.

An optimized hybrid paged-plus-ephemeral attention may avoid physically
writing the noise K/V later, but only after the exact paged implementation is
accepted and profiling proves the writes matter.

### 13.5 Verification graph and lossless sampling

Add `Phase::Verify` with query width eight and the ordinary causal target math,
including feature taps. The verifier returns enough data to:

- sample one target posterior per position with the request's exact top-k,
  temperature, top-p, and min-p controls;
- retain the successor RNG state after each position;
- select the state after exactly the committed `L+1` visible tokens; and
- keep later speculative draws from advancing externally visible state.

The verifier may compute all eight posterior samples in parallel/sequential
graph logic, but state selection must make it equivalent to `L+1` ordinary
target samples. The target distribution—not the greedy draft distribution—is
authoritative.

Do not speculate past the visible budget. When fewer than eight visible tokens
remain, emit the pending token and switch to ordinary decode for the tail
rather than launching a full block beyond `max_tokens`.

If a Harmony return/call token appears inside the accepted block:

- expose tokens only through the first terminal token;
- rollback target logical length after that token;
- discard later accepted/tentative features and sampling states; and
- finish/release the request before any suffix can enter prefix caching.

### 13.6 DFlash page and prefix ownership

Draft context KV uses the same 16-token logical token blocks but eight draft
layers. Track it as a separate physical arena and account exact bytes:

```text
8 layers * 2 * 8 KV heads * 64 * 2 BF16 bytes = 16,384 bytes/token at TP=1
```

A DFlash prefix hit is valid only when a block bundle contains both target KV
and matching draft context KV under the DFlash namespace digest. A target-only
cached page cannot initialize DFlash because the selected target hidden
features are absent. For an `auto` speculative request:

- use a complete target+draft bundle hit when present;
- otherwise prefill the suffix while populating both arenas; or
- deliberately run ordinary target decoding when memory/policy disables draft
  cache, without labeling a target-only hit as DFlash-ready.

Target and draft pages have tied logical ownership but separate physical IDs
and memory budgets. Bundle refcounts and rollback are transactional.

### 13.7 DFlash batching and auto policy

Compile draft/verify batch capacities `1, 2, 4, 8, 16, 32` only where startup
compile/memory budgets allow. Draft batches contain requests ready to propose;
verify batches contain requests with complete proposals. Ordinary decode and
speculative phases share fairness/admission but use separate executables.

Start with a static enabled policy for correctness. Then `auto` may disable
speculation when:

- the request has fewer than eight visible tokens remaining;
- draft cache reservation cannot finish safely;
- active target decode batching is already more efficient than draft+verify;
- rolling accepted length falls below a measured break-even threshold; or
- a configured model/request feature is incompatible.

Every auto decision increments a reason-labeled metric. Never silently fall
back because a draft execution failed; execution failure fails affected
requests.

### 13.8 DFlash acceptance and promotion

Correctness gates:

- greedy fixed prompts produce exactly the ordinary target tokens;
- stochastic fixed-seed runs preserve target sampling-state progression and
  visible tokens across mismatch positions;
- forced accept lengths 0-7 exercise every commit/rollback path;
- terminal tokens at pending and every accepted position expose no suffix;
- max-token tails, cancellation during draft, cancellation during verify,
  prefix hit/miss, page pressure, and TP=1 all return exact accounting;
- target/draft cache pages never become visible before commit; and
- official DFlash reference fixtures agree on projected features, draft hidden
  output, proposals, and acceptance for bounded BF16 tensors.

A40 performance gate, compared on the same image with speculation off/on:

- workloads: Math500, GSM8K, HumanEval, and MT-Bench prompts using the same
  reasoning effort as the model card where possible;
- concurrency: `1, 4, 8, 16, 32`;
- report mean acceptance length, proposed/accepted/rejected tokens, draft time,
  verify time, rollback time, target calls/token, TTFT, TPOT, aggregate TPS,
  cache bytes, and auto-disable reasons;
- initial promotion requires mean committed tokens per target verify at least
  3.5 and at least 1.25x end-to-end speedup at concurrency 1 without degrading
  correctness;
- if functional DFlash misses the performance threshold, keep it opt-in and
  record the measured bottleneck; do not enable it by default based on H200
  claims; and
- repeat TP=2/4 only after TP=1 promotes, with complete memory/collective
  accounting.

## 14. Observability and operational behavior

### 14.1 Metrics

Prometheus families, all bounded-label:

- requests received/admitted/completed/cancelled/failed/rejected;
- request/engine-event queue depth and saturation;
- active/queued/prefilling/decoding/speculating sequences;
- TTFT, TPOT, end-to-end latency, queue and tokenization histograms;
- scheduler iterations, scheduled tokens, batch size/query size/phase;
- prefill/decode/draft/verify execution and submission time;
- target/draft physical pages free/private/sealed/referenced/evicted;
- reservation credits and admission waits;
- prefix lookup blocks/tokens/hits/misses/duplicates/evictions;
- speculative proposed/accepted/rejected tokens and accepted-length histogram;
- per-family compile/warmup time and selected-family counts;
- per-device parameter/cache/temporary bytes and TP degree; and
- terminal reason/error code counts.

Do not use request IDs, model text, function names, token IDs, salts, or user
identities as metric labels.

### 14.2 Structured tracing

Trace request ID, phase, batch family, slot count, page mutations, queue delay,
and engine error codes. Content logging is off by construction. A diagnostic
operator flag may log token IDs only in an explicitly non-production mode.

### 14.3 Errors

Define stable internal error categories and OpenAI-shaped HTTP mappings:

- invalid request/model/tool history -> 400;
- unsupported request feature -> 400 with explicit field;
- queue/admission overload -> 429 with retry guidance;
- deadline/cancel -> client-visible terminal/cancel where transport allows;
- engine unavailable/not ready -> 503;
- artifact/compile/startup failure -> readiness false and process failure;
- device execution/cache invariant failure -> affected requests fail, engine
  becomes unhealthy if state integrity cannot be proven.

Never catch an optimized-lowering/device error and retry a different semantic
path inside a live request.

## 15. Proposed source layout

Use the smallest modules that preserve ownership boundaries:

```text
products/serve/src/
  main.rs                       CLI, Tokio startup, signals
  lib.rs                        server-facing public configuration/error
  api.rs                        Axum router, shared state, errors
  api/openai.rs                 strict request/response/SSE schemas
  server.rs                     EngineHandle and lifecycle
  server/engine.rs              dedicated thread loop and commands/events
  server/scheduler.rs           request state machine and batch planning
  server/cache.rs               target/draft page pools and reservations
  server/prefix.rs              chained hashes, index, refcounts, LRU
  server/metrics.rs             Prometheus registry and snapshots
  gpt_oss.rs                    product assembly
  gpt_oss/execution.rs          resident target executor and bindings
  gpt_oss/graph.rs              batched target graph families
  gpt_oss/parallel.rs           TP partition plan
  gpt_oss/protocol.rs           Harmony renderer/parser
  gpt_oss/dflash.rs             DFlash product owner
  gpt_oss/dflash/config.rs      exact pinned config validation
  gpt_oss/dflash/checkpoint.rs  draft parameter declarations
  gpt_oss/dflash/graph.rs       feature/project/draft/verify graphs
  gpt_oss/dflash/execution.rs   pending-token and cache transitions
```

Do not create all files before their milestone. Split an existing file only
when the new owner exists and the permanent target lists the new source.

Likely reusable framework changes:

- `crates/nml-tokenizer`: shared tokenizer/owned decoder lifecycle;
- `crates/nml-ir`: page-aware cache update and per-row batched sampling;
- `crates/nml-runtime`: only buffer/cache metadata primitives genuinely shared
  by other products; server LRU/admission remains under `products/serve`;
- `crates/nml-sharding`: no GPT-OSS policy, only any missing generic placement
  validation; and
- Triton crate: small-M batched compact kernels only after semantic shapes and
  portable contracts exist.

## 16. Verification matrix

### 16.1 BuildBuddy gates after each coherent milestone

- CPU contracts:
  `bb test //products/serve:serve_contract_test --config=buildbuddy --config=cpu`
- affected framework contracts with the same BuildBuddy/CPU configuration;
- CUDA compilation/contracts:
  `bb test <affected targets> --config=buildbuddy --config=cuda`
- server image:
  `bb build //products/serve:serve_image --config=buildbuddy --config=cuda`
- image structure contract through its existing BuildBuddy target.

Exact target lists are recorded in `TASKS.md` as modules land. No local
substitute is treated as evidence.

### 16.2 Real hardware gates

- single A40: batch-1 regression, continuous batching, prefix, tools, DFlash;
- 2x and 4x homogeneous CUDA host: TP correctness, shards, collectives,
  memory, and throughput;
- suitable SM90 later: portability/performance comparison, not a prerequisite
  for the A40-serving milestone;
- CPU multi-device remains the compile/placement oracle but cannot prove CUDA
  collectives.

Every paid GPU report pins source SHA, dirty-tree state, model/draft artifact
revisions, image digest, compiler/runtime versions, topology, server profile,
request corpus, and warmup/repetition policy. Cold compile/graph setup is
reported separately from warm serving.

### 16.3 Load and failure testing

Add a hermetic protocol/load client target that can drive:

- fixed concurrency and arrival-rate modes;
- streaming and non-streaming requests;
- disconnects at deterministic token indices;
- slow readers to saturate event queues;
- repeated/shared/divergent prefixes;
- mixed prompt/output lengths and sampling settings;
- tool-call round trips; and
- speculation on/off with exact seed comparison.

Failure injection is host-level and deterministic: page-pool exhaustion,
queue saturation, cancellation at each state transition, malformed protocol
output, and engine shutdown. Do not fake GPU execution, but do unit-test the
pure scheduler/page state machine without a device.

## 17. Ordered milestone exits

The implementation order is strict:

1. **Contracts and step API**: owned protocol sessions, step-wise executor,
   immutable single-request equivalence and A40 baseline retained.
2. **Tokio serving shell**: bounded control plane, dedicated engine thread,
   health/readiness/metrics/chat streaming/cancellation.
3. **Global paged arena**: arbitrary page-aware writes, process-wide storage,
   reservations, rollback and exact reclamation.
4. **Continuous batching**: batched shapes/sampling/kernels, chunked prefill,
   dynamic membership, fairness, concurrency evidence.
5. **Complete OpenAI tool surface**: structured chat histories, auto/none tool
   choice, SSE tool calls, tool-result round trips, no execution.
6. **Prefix caching**: chained full-page hashes, salt isolation, refcounts/LRU,
   TTFT evidence.
7. **Tensor parallelism**: TP=2/4 parameter/activation/cache sharding, global
   sampling, CUDA collective and memory proof.
8. **DFlash**: pinned artifact, feature taps, draft/verify, lossless rollback,
   target+draft prefix bundles, A40 promotion.
9. **Production hardening**: overload/shutdown/load tests, SLO/profile tuning,
   final docs and deployment examples.

No later milestone can be used to hide an earlier missing invariant. In
particular:

- continuous batching cannot precede page-aware cache writes;
- prefix caching cannot precede sealed-page ownership/refcounts;
- DFlash cannot precede target rollback and global page transactions;
- DFlash prefix reuse cannot use target-only pages;
- TP support cannot be inferred from CPU compilation; and
- HTTP concurrency cannot be claimed while one blocking generation call owns
  the engine until completion.

## 18. Explicit non-goals and rejected shortcuts

- No new quantization recipe or persistent BF16 expansion.
- No whole-transformer StableHLO monolith; the measured monolith regressed.
- No one-Tokio-task-per-request PJRT execution.
- No unbounded command, response, or admission queue.
- No page table that affects reads but not writes.
- No per-request K/V arena after the global arena milestone.
- No prefix caching of partial pages.
- No cache key based only on text strings, request JSON, or an unsafe fast hash
  in multi-tenant mode.
- No server-side arbitrary tool execution.
- No Python `trust_remote_code` in the product image.
- No DFlash speed claim copied from H200 results.
- No speculative suffix made visible before target verification/commit.
- No full-model replication labeled tensor parallelism.
- No TP=8 until 16-value NVFP4 shard alignment/padding is explicit.
- No runtime compilation when a new batch cardinality arrives.
- No benchmark promotion from one cold run, steady-device time alone, or an
  unpinned image/model/report.

This plan deliberately builds the serving control plane around the model and
kernel path that already achieved the A40 objective. The largest new gains
should come from useful concurrent work, prefix elimination, multi-GPU memory
headroom, and fewer target passes through DFlash—not from reopening the proven
NVFP4 representation.
