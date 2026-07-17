# NML work plan

Status: current implementation tracker

[`SYSTEM.md`](./SYSTEM.md) defines architecture and stable decisions. This file
contains only current capability evidence and work that still changes the
product. Git history is the archive for completed implementation details; this
file is not a chronological diary.

## Acceptance language

- `[x]` means the capability is implemented and its applicable permanent
  contracts have passed.
- `[ ]` means executable product work remains.
- `DEFERRED` means the work is deliberately outside the current sequence. It is
  retained here so it cannot silently become an implied capability.
- Building a kernel proves that it compiles. Reading device capability proves
  that dispatch can inspect the device. Neither proves that the kernel launched
  or produced correct values.
- A CUDA path is accepted when normal product dispatch selects it on suitable
  hardware, launches it through XLA/PJRT, and compares its output with an
  independent numerical reference. Tests do not expose a backend override just
  to manufacture coverage.

## Completed substrate

The first seven implementation milestones are complete. Their detailed task
history remains in Git; the resulting product capabilities are:

- [x] Rust types, shapes, layouts, semantic axes, and compact tensor-program API.
- [x] Owned MLIR construction, StableHLO serialization, Shardy annotations, XLA
  compilation, and distinct CPU/CUDA PJRT loaders over common safe ownership.
- [x] Typed host slices, persistent device buffers, donation/aliasing, named
  executable arguments, checkpoint declarations, SafeTensors loading, and
  structural model traversal.
- [x] Primitive algebra, unary math, nonlinear activations, structural
  operations, reductions, normalization, indexing, sorting, sampling,
  convolution, pooling, resize, FFT/IFFT, and explicit-state random generation.
- [x] Shardy-native logical meshes, four-device CPU execution, tiled placement,
  explicit collectives, and expert-sharded CPU MoE.
- [x] Ordinary attention, RoPE, masks, sliding windows, persistent dense/paged
  KV state, and portable blockwise paged attention.
- [x] CUDA custom-call lifecycle, upstream FA2/FA3 adapters, Triton paged
  attention, and grouped Triton expert projections.
- [x] IREE tokenization and end-to-end Qwen3-0.6B BF16 generation on CPU and the
  local SM75 CUDA product path.

## Real CUDA path evidence

The current remote artifact is the public device-contract image at:

```text
ghcr.io/narendrapatwardhan/nml@sha256:f128c4581b4dd4e8d5df0974bf20cb62b260381dcae12d34e8a0ba23787371fe
```

It was built from source commit
`bd62d67d5d1197fda0b18097d5c3ed70eadeaeeb` with a recorded dirty source bit.
That identity matters: later source changes require a new digest and new
evidence.

### Executed contract set

The unchanged six-contract image ran on an RTX A6000 (SM86) and an H100 80GB
HBM3 (SM90). Every contract passed, and both Pods were terminated with cleanup
confirmed.

| Permanent contract | SM86 | SM90 | What the success establishes |
| --- | --- | --- | --- |
| CUDA runtime | passed | passed | Packaged CUDA/PJRT runtime loads and owns the real device. |
| Checkpoint linear | passed | passed | Real SafeTensors parameters upload once and execute repeatedly. |
| Attention | passed | passed | Portable semantics plus the capability-selected Flash/Triton paths launch and match an independent dense reference. |
| Neural operations | passed | passed | The general operation set and grouped Triton MoE execute numerically in F32, F16, and BF16. |
| Execution performance | passed | passed | Linear+SiLU, convolution/pooling, and grouped Triton MoE pass phase-separated regression ceilings. |
| Flash capability policy | passed | passed | Device discovery works and the Flash implementation for the *other* architecture rejects before dereferencing dummy launch inputs. This is policy evidence, not supported-kernel execution evidence. |

Run records:

- [x] SM86: lease `f76a42f3-27b3-4ffa-8eb7-ac27b66bff8f`, driver
  `570.195.03`, all six contracts in 37.576 seconds.
- [x] SM90: lease `2a37f867-fb63-4c15-a3a6-fc8ba83b5f28`, driver
  `580.126.09`, all six contracts in 35.371 seconds.

### Which optimized paths actually ran

The attention contract uses ordinary public attention shapes. The detected
compute capability drives the private lowering; there is no test-only backend
selector.

| Path | Real execution evidence |
| --- | --- |
| FA2 ordinary attention | [x] SM86 launched F16 causal/sliding-window and BF16 noncausal/custom-scale attention and matched the host reference. |
| FA2 paged attention | [x] SM86 launched F16 and BF16, page size 256, prefill and single-token decode, and matched the host reference. |
| FA3 ordinary attention | [x] SM90 launched the same F16/BF16 ordinary attention cases and matched the host reference. |
| FA3 paged attention | [x] SM90 launched F16 and BF16 page sizes 16 and 256 for prefill and single-token decode and matched the host reference. |
| Triton paged 2D | [x] SM86 used page-16 F16/BF16/F32 prefill; SM90 used the F32 case. Outputs matched the same host reference. |
| Triton paged split-K | [x] SM86 used page-16 F16/BF16/F32 single-token decode; SM90 used the F32 case. Outputs matched the same host reference. |
| Triton grouped MoE | [x] SM86 and SM90 launched SwiGLU, GELU, and ReLU expert paths in F32, F16, and BF16 and matched the portable host calculation. |

The SM86 run is the acceptance evidence for the FA2 dispatch class
(SM80-SM89). The SM90 run is the acceptance evidence for FA3. NML does not
require one rented card for every marketing SKU or minor compute capability
when the code selects the same implementation path.

### What has not run on suitable rented CUDA hardware

This is the complete current hardware debt; it must not be inflated into a
claim that FA2, FA3, or Triton are compile-only.

- [ ] Dedicated attention performance workloads have not measured FA2, FA3,
  Triton 2D, or Triton split-K latency/throughput across representative prefill,
  decode, sequence-length, and page geometries. Their numerical paths are real;
  their tuning quality is not yet established.
- [ ] No homogeneous multi-GPU CUDA run has exercised Shardy partitioning,
  cross-device collectives, tensor parallelism, or expert parallelism. The
  current rented evidence is single-device. A single-device all-reduce did run,
  but it is an identity operation and is not distributed evidence.
- [ ] The Qwen production image and real checkpoint have not run on the rented
  FA2/FA3 hosts. Qwen has CPU and local SM75 end-to-end evidence; the rented
  image was the substrate contract image, not the serving image.
- [ ] Linux AArch64 CUDA packaging and execution have not run on a native
  DGX-Spark-class host.
- [ ] A failure originating *after launch* inside a supported FA2, FA3, Triton
  attention, or Triton MoE kernel has not occurred in a permanent product
  contract, so end-to-end supported-kernel failure propagation has not been
  observed on those devices. Incompatible FA2/FA3 capability rejection did run.
  We will not add an artificial crashing kernel merely to check this box.
- [ ] The corrected `f128...371fe` image has not also been run on the local SM75
  host. Earlier local SM75 contracts and the Qwen production image ran
  successfully, but they are different artifact evidence.

GPT-OSS, continuous batching, prefix caching, tool calling, and quantization
have not run because those product capabilities are not implemented yet. They
are implementation work below, not missing validation of an existing kernel.

## Current milestone: deployment closure with Qwen retained

This milestone ends when one immutable Linux CUDA artifact can be built by
BuildBuddy, published once, and executed unchanged through the same permanent
interface on local and RunPod GPUs. Qwen remains the regression model while
the execution envelope is completed.

### OCI construction and publication

- [x] Use `rules_img` as the only OCI construction graph over digest-pinned
  distroless bases. Do not add `rules_oci` as a parallel graph.
- [x] Build separate production-serving and device-contract images over the
  same CUDA/PJRT runtime contract. Model weights stay outside both images.
- [x] Build and structure-test the native Linux x86-64 images through
  BuildBuddy.
- [x] Publish public images to `ghcr.io/narendrapatwardhan/nml` through the OCI
  Registry API. GitHub CLI is not part of publication or administration.
- [ ] Make exact digest references mandatory in local and RunPod acceptance
  commands. Mutable tags may exist only for discovery and must resolve to the
  recorded digest before execution.
- [ ] Add immutable source-revision labels during trusted publication. Select a
  non-root runtime user only after local NVIDIA and RunPod device access prove
  the least privilege that works in both venues.
- [ ] Prove a clean local machine can pull and execute the BuildBuddy-built
  image without downloading the LLVM/XLA/CUDA build cache.
- [ ] Extend the BuildBuddy workflow to build CPU contracts, GPU-independent
  CUDA contracts, exact CUDA binaries, image structure contracts, and OCI
  images in their truthful venues. Hosted workers never execute device tests.
- [ ] `DEFERRED`: build the Linux AArch64 image on a native Linux AArch64 CUDA
  venue and combine the native manifests into one index.

### Local and RunPod execution

- [x] The in-image Rust runner owns a fixed manifest of permanent contracts,
  serial execution, deadlines, bounded logs, structured hardware identity,
  immutable results, and child cleanup. It never invokes Bazel or a shell.
- [x] The Bazel-built RunPod controller uses GraphQL for Pod placement/status,
  ports, and termination; REST is limited to optional template management.
- [x] Lease state is atomic and external to the repository. Success, failure,
  timeout, interruption, and controller exceptions all enter the same
  termination path; an unconfirmed termination remains a visible possibly
  billable orphan.
- [x] Execute one digest unchanged on real FA2 and FA3 dispatch classes and
  retain structured terminal results: the SM86 and SM90 runs above.
- [ ] Add one repository-owned local executor that accepts an exact digest,
  uses Docker or Podman with the NVIDIA runtime's `--gpus all` contract, mounts
  only declared inputs/results, and removes the container after completion.
- [ ] Run the corrected contract digest through that local executor on SM75 and
  compare the structured results with direct Bazel device execution.
- [ ] Finish runner lifecycle evidence when real permanent events are available:
  contract failure, deadline, client disconnect, and shutdown during execution.
  Unsupported selection, malformed request, repeated-run rejection, immutable
  result retrieval, and normal cleanup already pass. Do not create disposable
  probes or deliberately crashing GPU binaries.
- [ ] Require named RunPod secret references before remote model download or
  private artifact access. Raw credentials never enter Pod configuration,
  Bazel inputs, logs, or lease records.
- [ ] Add an optional persistent model-cache/network-volume identity only when
  remote model execution needs it. Every mount must be revalidated against the
  exact model manifest; filenames are not artifact identity.

## Next milestone: GPT-OSS 20B BF16

Qwen remains a permanent regression model. GPT-OSS becomes the default only
after one exact BF16 artifact is selected and the complete model passes.

### Select and pin one artifact

- [x] Audit trustworthy BF16 distributions by immutable revision, actual
  SafeTensors inventory, tensor shapes/dtypes, tokenizer/Harmony files,
  conversion provenance, and reproducibility.
- [x] Record the audit result: `unsloth/gpt-oss-20b-BF16` revision
  `cc89b3e7fd423253264883a80a4fa5abc619649f` is structurally viable and contains
  41,829,514,368 bytes of BF16 tensors, but Unsloth states that it was
  up-converted from the official MXFP4 payload. FriendliAI and CrusoeAI mirror
  the same shards. `lmsys/gpt-oss-20b-bf16` is rejected because most parameters
  are actually F8_E5M2. The z-lab DFlash repository is a Qwen draft model, not a
  GPT-OSS base checkpoint.
- [ ] Make the owner decision whether the documented Unsloth up-conversion is
  acceptable as the BF16 product artifact. Do not describe it as original
  pre-quantization BF16 weights.
- [ ] Pin the selected revision and a checked manifest containing every required
  file, size, hash, role, tensor name, shape, and dtype. Mismatch must fail
  before graph construction or device allocation.

### Implement the checkpoint and model

- [ ] Parse and validate the exact configuration: architecture, layer/expert
  counts, head geometry, attention schedule, context/RoPE parameters,
  normalization, activation, tokenizer identity, and output-weight policy.
- [ ] Declare exact embedding, attention, router, expert, normalization,
  attention-sink, and output tensors. Do not guess aliases, alternate names, or
  transposes.
- [ ] Preserve BF16 in host storage, device storage, and ordinary contractions;
  report checkpoint, persistent device, executable, cache, and workspace memory
  before upload.
- [ ] Build the private GPT-OSS block graph from existing RMSNorm, GQA,
  RoPE/YaRN, dense/sliding attention, top-k MoE, grouped expert, residual, and
  Shardy primitives.
- [ ] Add learned attention-sink denominator bias to portable ordinary and
  paged online softmax. Optimized backends may be used only where their ABI can
  preserve the exact semantics; otherwise dispatch must choose the portable
  path truthfully.
- [ ] Implement the exact clamped/residual SwiGLU composite and alternating
  full-attention/128-token-window schedule, including boundary and long-position
  contracts.
- [ ] Validate `o200k_harmony` tokenization through the existing IREE tokenizer
  boundary and implement versioned Harmony rendering/incremental parsing for
  roles, analysis/final channels, tool calls, and tool results.
- [ ] Compare selected block/expert outputs with an independent trustworthy
  implementation using declared tolerances.
- [ ] Execute the complete pinned GPT-OSS 20B BF16 artifact on capable rented
  CUDA hardware. Compare fixed prompt tokens, intermediate values, channel
  structure, and greedy continuation with the independent oracle. A reduced
  model or isolated block does not satisfy end-to-end acceptance.

## Following milestone: serving product

Serving stays above the compact `nml` facade. One dedicated engine owner holds
PJRT state; Tokio owns network and orchestration, never opportunistic PJRT work.

### Engine and protocol

- [x] Establish a private model-neutral engine boundary and retain Qwen through
  it. The current compatibility path truthfully supports batch capacity one.
- [ ] Add one Tokio runtime with bounded command/completion/token channels,
  cancellation tokens, deadlines, signals, and transactional startup/shutdown.
- [ ] Add bounded Axum/Tower chat-completions and Responses-style HTTP routes,
  streaming, liveness, readiness, and model identity endpoints.
- [ ] Enforce prompt/output/message/tool/schema/queue/concurrency limits before
  unbounded allocation or device work.
- [ ] Implement Harmony validation, rendering, incremental UTF-8/channel/tool
  parsing, and deterministic malformed-output handling. The server returns tool
  calls; it does not execute user tools.

### Batching and cache ownership

- [ ] Remove batch-one assumptions from model inputs, positions, cache state,
  logits selection, sampling, and result demultiplexing.
- [ ] Compile a finite startup-declared family of prefill buckets and
  fixed-capacity decode executables while sharing one parameter allocation.
- [ ] Create one server-owned physical K/V page arena with checked accounting,
  generation-stamped leases, and one idempotent reclamation path.
- [ ] Execute at least two independent requests in one physical decode call and
  compare output, RNG, and cache state with independent sequential execution.
- [ ] Schedule chunked prefill and decode under explicit per-tick token/sequence
  budgets with starvation bounds and slot refill without recompilation.
- [ ] Make cancellation correct while queued, page-waiting, in-flight,
  token-ready, streaming, and disconnected.
- [ ] Add prefix caching over exact token/model/protocol/RoPE/representation/
  topology identity. Share only immutable complete pages; partial extension is
  copy-on-write and eviction shares the admission page budget.

### Distributed serving, observability, and resilience

- [ ] Define the GPT-OSS tensor-parallel plan for embeddings/output, attention,
  routers, experts, and reductions through Shardy.
- [ ] Compare the four-device CPU topology with the single-device oracle, then
  run the unchanged paged/batched contract on homogeneous multi-GPU CUDA.
- [ ] Expose bounded-cardinality Prometheus metrics and structured tracing for
  admission, queueing, batches, pages, prefix reuse, compilation, execution,
  streaming, cancellation, errors, and shutdown. User content and secrets never
  become labels.
- [ ] Cover admission races, disconnects, deadlines, page exhaustion,
  engine/listener failure, signals, repeated lifecycle, and partial streams.
- [ ] Run a real loopback server contract with concurrent streaming and
  non-streaming requests, physical batching, page/prefix reuse, tools,
  cancellation, metrics, and graceful zero-owner shutdown.
- [ ] Record phase-separated startup, memory, TTFT, inter-token latency,
  throughput, batching scale, prefix-reuse, and shutdown baselines.

### Speculative decoding

- [ ] Select an exact draft artifact or public algorithm before implementing a
  producer. DFlash is not assumed to apply to GPT-OSS.
- [ ] Define model-independent proposal, verification, cache rollback, and RNG
  semantics over the existing engine.
- [ ] Prove greedy equivalence or the declared stochastic distribution and
  retain the path only if measured target invocations or latency improve after
  draft cost and memory are included.

## Deferred quantization milestone

This work begins only after the BF16 GPT-OSS serving path is complete.

- [ ] `DEFERRED`: audit trustworthy GPT-OSS 20B NVFP4 artifacts by immutable
  revision, actual packing/scales, conversion provenance, hardware assumptions,
  and oracle outputs; select exactly one.
- [ ] `DEFERRED`: implement that artifact's packed checkpoint storage, scale
  semantics, layout transforms, accumulation dtype, capability gates, portable
  dequantized reference, and selected CUDA execution path.
- [ ] `DEFERRED`: run the complete NVFP4 model through generation, paged serving,
  continuous batching, tensor parallelism, prefix caching, and Harmony; compare
  it with both the artifact oracle and NML's BF16 baseline.
- [ ] `DEFERRED`: measure total checkpoint/resident/workspace memory, upload,
  compilation, TTFT, latency, throughput, and output deltas before accepting the
  representation.
- [ ] `DEFERRED`: W4A16 and W8A8 remain separate future verticals. An NVFP4
  decision does not imply their packing, kernels, or checkpoint formats.

## Capability ledger

This ledger tracks usable product families, not individual opcodes. A completed
family has durable applicable CPU/CUDA numerical and ownership coverage; a
pending product family is not implied by the primitives beneath it.

- [x] Arithmetic, comparisons, selection, casts, bit operations, unary math,
  activations, and complex/FFT operations.
- [x] Reshape, transpose, concatenation, slicing, dynamic update, gather,
  scatter, embeddings, and layout-aware compiled graphs.
- [x] Reductions, softmax, RMSNorm, LayerNorm, L2 normalization, log-sum-exp,
  argmax, and related composites.
- [x] Matrix contraction, linear layers, convolution, pooling, and spatial
  resize.
- [x] Explicit-state random generation, stable/unstable sorting, argsort, top-k,
  greedy selection, and stochastic sampling.
- [x] Ordinary and blockwise paged attention, RoPE, masks, sliding windows, and
  persistent KV update/truncate/rollback/replay.
- [x] Capability-dispatched FA2, FA3, Triton paged attention, and grouped Triton
  MoE with real suitable-device numerical execution.
- [x] Portable MoE routing/expert execution and four-device CPU expert sharding.
- [x] IREE tokenization and Qwen3-0.6B BF16 generation.
- [ ] Immutable OCI execution fully closed across BuildBuddy, local NVIDIA, and
  RunPod, including digest-only interfaces and the remaining clean-consumer
  proof.
- [ ] GPT-OSS 20B BF16 generation with exact checkpoint, sink, clamped/residual
  SwiGLU, alternating windows, YaRN, tokenizer, and Harmony semantics.
- [ ] Continuous batching, server-owned paged KV arena, prefix caching, bounded
  streaming/cancellation, tools, and metrics.
- [ ] Real multi-GPU CUDA Shardy execution and collectives.
- [ ] Dedicated optimized attention performance and tuning evidence.
- [ ] `DEFERRED`: NVFP4, W4A16, and W8A8 complete quantized execution verticals.
- [ ] `DEFERRED`: explicitly authored analytic backward/training graphs.
