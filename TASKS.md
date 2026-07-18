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

## Current milestone: GPT-OSS 20B NVFP4

[`NVFP4.md`](./NVFP4.md) is the detailed architecture and acceptance contract.
This is the first priority. Existing APIs are not compatibility constraints:
dense-only checkpoint, traversal, binding, and bufferization assumptions are
replaced when they prevent the single coherent parameter system described
there.

### A. Select and freeze one artifact

- [x] Audit trustworthy GPT-OSS 20B NVFP4 artifacts by immutable revision,
  license, source/conversion provenance, actual SafeTensors inventory,
  configuration, tokenizer/Harmony assets, and independent runtime support.
- [x] Select exactly one artifact. Record every file hash and every tensor's
  physical dtype, shape, byte extent, role, logical mapping, and transpose.
- [x] Specify E2M1 nibble order, E4M3 variant, block/global scale algebra,
  block axis, padding, swizzle, 1D/2D scaling, and higher-precision exceptions.
- [x] Freeze permanent decoded-value and projection fixtures from widely
  separated rows of the immutable published artifact. A bounded-range
  extractor verifies publication/inventory identity and records original
  compact bytes, decoded F32 hashes and samples, and independent F64 results.
- [ ] Pin independent representative layer and generation fixtures against the
  exact same revision. Never substitute self-generated weights or relabel
  MXFP4 evidence.

### B. Replace the dense-only parameter/storage model

- [x] Separate bounded physical checkpoint records and component storage specs
  from logical parameter shapes; dense storage now flows through that boundary.
- [x] Add private validated packed-E2M1x2 and E4M3-bit storage encodings without
  adding `DType::NvFp4` or public general FP8 arithmetic.
- [x] Introduce one logical `Parameter` and runtime `LoadedParameter` boundary;
  dense parameters use the same one-component representation and binding path.
- [x] Add the closed NVFP4 representation whose parameter owns payload, local
  scales, global factor, and exact artifact-selected metadata.
- [x] Replace `TensorStore`'s one-record-to-one-`Tensor` assumption and the
  dense-only `NmlStruct`/`Bufferized` leaf mapping. Keep one structural traversal
  for dense and quantized parameters and delete superseded compatibility paths.
- [x] Keep ordinary `Tensor`, `Shape`, `Slice`, and `Buffer` invariants simple.
  Flatten parameter components only at the private lowering/executable-binding
  boundary, with deterministic names and checked representation identity.
- [x] Replace logical-shape checkpoint upload with direct physical-component
  streaming and retain bounded CPU/CUDA transfer accounting.
- [x] Make multi-component loading transactional and account exact source,
  resident, prepared, and bounded staging bytes. Source-layout execution owns
  only compact components and cannot create a persistent BF16/FP16 expansion.
- [ ] Add versioned one-time layout preparation when the first backend actually
  requires a distinct prepared representation; account it separately and
  release superseded source buffers only after verified transactional commit.
- [x] Derive physical payload/scale shards from logical Shardy ranges. Validate
  block/tile alignment, padding, expert-axis slicing, and component co-sharding
  before allocation.
- [x] Port the permanent dense Qwen regression model and all CPU product
  contracts to the new one-component parameter system; the old path is gone.
- [x] Re-run the migrated parameter loading/binding and CUDA device contracts
  on the local SM75 acceptance GPU. The six permanent CUDA contracts passed in
  BuildBuddy invocation `c76dfa96-28d1-452d-afab-8d49911d3c19`; this is
  substrate evidence, not a claim that the Qwen product reran in this change.

### C. Establish exact and performant CPU execution

- [x] Implement exhaustive E2M1 and artifact-selected E4M3 decoding, scale
  algebra, padding validation, and checked physical extent calculations.
- [x] Implement bounded portable CPU embedding, conventional linear,
  input-major grouped projection, and exact GPT-OSS routed expert semantics over
  compact components. Compare them with an independently decoded F64 oracle,
  including odd widths and uneven/empty expert routing.
- [x] Add semantic parameter-aware embedding, linear, and grouped expert
  operations. Reject NVFP4 parameters from operations without defined
  quantized semantics before MLIR construction. The shared dense/compact API,
  StableHLO typed-FFI boundary, registered CPU handlers, and F16/BF16 CPU PJRT
  execution contracts cover all three operations. Supported SM80+ CUDA targets
  lower the same semantics to fused compact-weight Triton kernels. SM75 linear,
  embedding, and schedule-driven routed experts lower to dedicated typed CUDA
  custom calls and pass the unchanged complete numerical contract locally.
- [x] Implement a blockwise CPU oracle that expands only bounded tiles, then an
  optimized x86-64 path with vectorized unpack/scale and cache-aware
  contraction. Preserve a portable/AArch64-capable implementation. Runtime
  dispatch selects an AVX2 register-only nibble unpack, scale application,
  dot, and AXPY path; the allocation-free scalar block implementation remains
  the exact fallback on AArch64 and non-AVX2 x86.
- [ ] Permanently compare embedding, decode GEMV, prefill GEMM, gate/up/down
  projections, uneven/empty grouped experts, epilogues, and logical shards
  against an independently decoded F32/F64 oracle.
- [ ] Record x86-64 CPU memory and performance by phase; a deliberately slow
  reference alone does not close the CPU product target.

### D. Implement fused pre-Blackwell CUDA execution

- [x] Replace scattered compute-capability integer checks with one private
  named capability value used by semantic lowering and structured diagnostics.
- [x] Extend the typed Rust TTIR builder only with the pinned Triton surface
  needed for E2M1/scale handling, packed operations, and `tt.dot_scaled`.
  Continue to reparse and verify complete TTIR before StableHLO embedding.
- [x] Implement SM8x fused compact W4A16 embedding and decode/prefill linear
  kernels. E2M1 payload and E4M3 block/global scales are decoded inside each
  tile, the source representation is never persistently expanded, contractions
  accumulate in F32, and optional bias is added inside the compact projection
  epilogue on both CPU FFI and CUDA rather than launching a second graph op.
- [x] Implement the SM90 lowering through the same fused compact W4A16 Triton
  contract with Hopper-aware launch geometry. This is deliberately labeled
  emulation through ordinary tensor-core dot lowering, not native Blackwell
  block-scaled FP4 execution.
- [x] Implement SM75 linear and embedding CUDA custom calls. Projection decodes
  packed E2M1 and E4M3FN scales into one F16 WMMA tile, performs Turing
  half-precision contraction with F32 accumulation, fuses optional bias, and
  explicitly converts BF16 tiles as required by the governing design. Embedding
  decodes only selected rows. Neither path enters the unproven SM75 Triton
  pipeline or creates a persistent dense weight.
- [x] Implement the SM75 fused routed-expert CUDA custom call with exact
  clamped/residual SwiGLU, uneven/empty routing, and no per-expert host loop.
- [x] Implement quantized grouped gate/up/down expert projections with uneven
  routing, empty experts, exact GPT-OSS activation semantics, and expert-axis
  sharding. Compact components stay inside grouped Triton calls; no per-expert
  host loop or persistent dense conversion is used.
- [x] Execute the permanent numerical contract on local SM75, rented SM8x, and
  rented SM90 through normal capability dispatch. The complete FP16/BF16
  linear, embedding, and routed-expert contract passed locally on SM75 in
  BuildBuddy invocation `e1b483ce-a933-4c19-a387-87d3ea7e5b26`. The unchanged
  OCI contract at
  `ghcr.io/narendrapatwardhan/nml@sha256:17040fd252bac543bb3b02e9abc253d309d05a7b64cf6ee7b8c6cc8b64c426b4`
  then passed on an RTX A6000 (SM86, driver `550.127.08`) in lease
  `1c30d217-cd03-4a61-bd8c-2d5ce10d1eb3` and an H100 80GB HBM3 (SM90a,
  driver `580.126.09`) in lease `09de1c2c-6304-44f5-bb03-81d9d64cf4c1`.
  Both Pods reached authenticated readiness, returned the permanent runner's
  structured success result, and had termination confirmed.
- [x] Execute phase-separated NVFP4 performance contracts on local SM75,
  rented SM8x, and rented SM90. The contract uses GPT-OSS 20B dimensions:
  width/intermediate 2880, vocabulary 201088, 32 experts, and top-4 routing.
  Host preparation is excluded from upload time, and compact payload/scales
  remain compact throughout every measured CUDA path. The immutable remote
  image was
  `ghcr.io/narendrapatwardhan/nml@sha256:4f4619f040bd8c59549b90e5b3606c930bc063389aee4e280a25853c15fdf0ff`.
  It passed on an RTX A6000 (SM86, driver `550.54.15`) in lease
  `e9413e85-7205-46df-b790-0b557986627c` and an H100 80GB HBM3 (SM90a,
  driver `580.126.09`) in lease
  `d3122c7b-c398-49b4-b68a-3ebd5f819695`; both Pods terminated with cleanup
  confirmed. Local SM75 evidence is BuildBuddy invocation
  `c38202f9-02be-4cb9-a9b3-61f2f7fd9dc2`.

  | Device/workload | compile ms | upload ms | first ms | steady ms | download ms |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | SM75 embedding, 128 tokens | 11.838 | 546.207 | 334.700 | 1.710 | 178.576 |
  | SM75 decode, M=1 | 12.730 | 1.958 | 1.748 | 1.607 | 1.264 |
  | SM75 prefill, M=128 | 11.354 | 2.955 | 4.766 | 4.785 | 199.251 |
  | SM75 grouped MoE, 16 tokens | 2085.209 | 102.849 | 60.376 | 53.540 | 19.209 |
  | SM86 embedding, 128 tokens | 134.856 | 290.125 | 0.450 | 0.113 | 99.918 |
  | SM86 decode, M=1 | 205.951 | 4.038 | 0.490 | 0.253 | 1.068 |
  | SM86 prefill, M=128 | 235.905 | 1.699 | 0.761 | 0.548 | 112.507 |
  | SM86 grouped MoE, 16 tokens | 1837.742 | 60.560 | 7.264 | 5.246 | 12.648 |
  | SM90 embedding, 128 tokens | 181.832 | 213.848 | 0.416 | 0.107 | 114.512 |
  | SM90 decode, M=1 | 267.667 | 1.950 | 0.477 | 0.238 | 1.245 |
  | SM90 prefill, M=128 | 278.127 | 2.279 | 0.604 | 0.287 | 156.511 |
  | SM90 grouped MoE, 16 tokens | 1859.218 | 83.080 | 5.987 | 3.187 | 13.606 |

  These are single-run acceptance baselines, not a broad benchmark study.
  Numerical-contract or whole-process wall time is not kernel-performance
  evidence; comparisons use the named phase and identical workload shape.

### E. Implement native Blackwell execution

- [ ] Expose typed `tt.dot_scaled` E2M1/E4M3 construction and prove the pinned
  XLA/Triton custom-call pipeline compiles a real SM100 kernel.
- [ ] Implement versioned source-to-native payload/scale packing and swizzling,
  including native M/N/K/alignment checks and sampled boundary verification.
- [ ] If native block-scaled MMA requires two FP4 operands, implement explicit
  transient 1D activation quantization, scale/global-factor computation,
  rounding, workspace accounting, and an independent CPU oracle. An upcasted
  weight feeding BF16 MMA remains labeled emulation even on Blackwell.
- [ ] Implement native ordinary and grouped projection paths. Unsupported
  native geometries may use the named fused emulation path, never a hidden
  persistent dense conversion.
- [ ] Run the unchanged operation contracts on a real SM100+ GPU, compare with
  the CPU representation oracle, and retain generated-code or profiler evidence
  that native block-scaled instructions executed.

### F. Complete the GPT-OSS 20B model vertical

- [x] Parse and validate the exact selected configuration: layer/expert counts,
  head geometry, attention schedule, context/YaRN parameters, normalization,
  activation, and output-weight policy. Nearby variants and unknown fields are
  rejected by the package-private GPT-OSS model target.
- [x] Declare every embedding, attention, router, expert, normalization,
  attention-sink, and output parameter from the checked representation
  manifest. The structured 411-parameter tree matches all 703 physical
  components without aliases, guessed transposes, or persistent expansion.
- [x] Build the package-private, batch-one model graph from exact RMSNorm, GQA,
  learned-sink attention, YaRN, alternating full/sliding attention, top-k
  routing, quantized grouped experts, residuals, and the compact output
  projection. The donated contiguous cache is exposed through a one-page
  identity view and the semantic paged operation retains capability-selected
  Triton/FA2/FA3 dispatch. Its permanent graph contract consumes all 703
  physical parameter components and produces `[1, 201088]` last-token logits.
- [x] Define the finite prefill/decode graph family and keep greedy sampling on
  device. Prefill derives positions internally; decode accepts one scalar
  position; both return one I32 token plus 48 donated cache buffers without a
  second parameter owner.
- [ ] Integrate that graph family and the declared Shardy placement with the
  model-neutral engine after the exact Harmony protocol owner is available.
- [x] Add learned attention-sink denominator bias to ordinary and paged
  online-softmax semantics. FA2 and FA3 remain selected and consume their F32
  log-normalizer through an exact StableHLO correction epilogue. Triton 2D
  initializes its online state from the sink, while split-K adds the sink once
  in global reduction. Permanent lowering contracts prohibit a sink-induced
  portable path, and CPU contracts compare both cache representations in F32,
  F16, and BF16.
- [x] Wire exact interleaved, clamped/residual SwiGLU and alternating
  full/128-token-window attention into the model graph. YaRN's frequency
  interpolation and attention amplitude have permanent CPU numerical coverage;
  every retained optimized attention backend preserves learned-sink semantics.
- [ ] Validate `o200k_harmony` through the IREE tokenizer boundary and implement
  versioned Harmony rendering/incremental parsing for roles, channels, tool
  calls, and tool results.
- [ ] Compare decoded parameters, representative layer/expert outputs, router
  choices, final logits, tokens, and fixed prompts with the independent
  implementation of the exact same artifact.
- [ ] Execute the complete model without hidden dense weights on capable CUDA
  hardware and record checkpoint, host, resident, workspace, compile, first-run,
  prefill, and decode costs separately.

### G. Close sharding and product evidence

- [ ] Define GPT-OSS logical tensor/expert-parallel placement over quantized
  component shards and prove there is no hidden whole-weight all-gather.
- [ ] Compare the established four-device CPU topology with the single-device
  oracle, then run homogeneous multi-GPU CUDA Shardy and collectives.
- [ ] Make artifact/recipe/prepared-layout identity part of executable,
  prepared-weight, result, and future prefix-cache keys.
- [ ] Publish and run one immutable NVFP4 product/device-contract image on the
  applicable local and rented venues, retaining structured dispatch, memory,
  correctness, and performance evidence.

## Deferred milestone: deployment closure with Qwen retained

All unchecked work in this section is deferred until the NVFP4 vertical needs
it directly or reaches its acceptance boundary. Completed OCI/RunPod machinery
remains the execution substrate for rented NVFP4 hardware.

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
- [ ] Move routine publication onto a BuildBuddy remote runner so completed
  OCI layers move directly from the colocated cache to GHCR. Store only
  `GHCR_USERNAME` and a least-scope `GHCR_TOKEN` as encrypted BuildBuddy
  organization secrets and inject exactly those names with `env-secrets`.
  Create an owner-only runner-local registry-auth file or credential-helper
  bridge for `rules_img`, remove it after the publisher exits, disable automatic
  retry for the mutating publish command, and independently resolve the
  resulting public tag to its immutable registry digest. Local `bb run`
  publication through Docker's credential store remains recovery-only.
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

## Supporting artifact evidence and deferred BF16 product vertical

The completed BF16 audit is retained because a dense artifact may be useful as
an independent oracle or a declared source for deterministic NVFP4 conversion.
All unchecked BF16 product work is deferred; it does not precede NVFP4.

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

### Deferred standalone BF16 checkpoint and model

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
- [x] Add learned attention-sink denominator bias to ordinary and paged
  online softmax. Portable StableHLO, Triton 2D/split-K, FA2, and FA3 now
  preserve the exact semantics without changing backend selection.
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

## Deferred milestone: serving product

Serving stays above the compact `nml` facade. One dedicated engine owner holds
PJRT state; Tokio owns network and orchestration, never opportunistic PJRT work.
Unchecked work here resumes after the NVFP4 model vertical unless a bounded
piece is required to prove its end-to-end execution.

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

## Deferred independent quantization verticals

- [ ] `DEFERRED`: select an exact W4A16 artifact/workload and independently
  define signedness, grouping, scale, zero-point, packing, transpose, compute,
  accumulation, checkpoint, and kernel contracts.
- [ ] `DEFERRED`: select an exact W8A8 artifact/workload and independently
  define integer/FP8 values, static/dynamic activation quantization, scale
  granularity, accumulation, requantization, checkpoint, and kernel contracts.
- [ ] `DEFERRED`: NVFP4 KV-cache quantization is a separate representation and
  attention-kernel vertical; weight NVFP4 does not imply it.

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
- [ ] GPT-OSS 20B NVFP4 generation with exact representation, compact CPU and
  capability-dispatched CUDA execution, sink, clamped/residual SwiGLU,
  alternating windows, YaRN, tokenizer, Harmony, and grouped expert semantics.
- [ ] Continuous batching, server-owned paged KV arena, prefix caching, bounded
  streaming/cancellation, tools, and metrics.
- [ ] Real multi-GPU CUDA Shardy execution and collectives.
- [ ] Dedicated optimized attention performance and tuning evidence.
- [ ] `DEFERRED`: standalone BF16 GPT-OSS product execution after any BF16
  artifact has been deliberately selected rather than merely used as an oracle.
- [ ] `DEFERRED`: W4A16 and W8A8 complete independent execution verticals.
- [ ] `DEFERRED`: explicitly authored analytic backward/training graphs.
