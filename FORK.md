# NML: source-guided reimplementation charter

Status: working document for owner review

ZML reference snapshot: `references/zml` at commit
`ed2bd190e8dd1e47bda2819015f443874b68243a` (2026-07-10)

This document records what exists in the reference snapshot, where its
complexity comes from, and what NML would gain or lose by retaining, narrowing,
replacing, or omitting each part. Except for decisions explicitly made in this
document, it does **not** decide what NML will cut.

## 1. Decision vocabulary

- `DECIDED`: stated by the project owner and treated as a constraint.
- `UNDECIDED`: requires an explicit owner decision.
- `DEFERRED`: intentionally not decided yet; this is not the same as removal.
- `OUT`: explicitly excluded by the owner.
- `DEFAULT-ZML`: no contrary NML decision or established NML pattern exists,
  so ZML's architecture and behavior govern the rewrite.

No subsystem should disappear merely because it is listed as a complexity
axis. Complexity is evidence for a decision, not the decision itself.

## 2. Decisions already made

| ID | State | Decision |
| --- | --- | --- |
| D-001 | `DECIDED` | NML is a manual, source-guided reimplementation. ZML may be read continuously, but NML product/runtime/model code is written for NML's requirements rather than copied wholesale. D-010 explicitly permits deliberate reuse and adaptation of build/dependency integration. |
| D-002 | `DECIDED` | CPU and NVIDIA CUDA are the only accelerator targets. AMD ROCm, Google TPU, AWS Neuron/Trainium, Intel oneAPI, and Apple Metal are not NML accelerator targets. |
| D-003 | `DECIDED` | NML's canonical ordinary scalar set is bool; signed and unsigned 8/16/32/64-bit integers; FP16, BF16, FP32, FP64; and C64/C128. Complex types are retained because FFT/IFFT and complex-form operations such as RoPE need them. The low-bit storage and compute types required by D-004 are separate quantization contracts rather than additions inferred from this ordinary set. |
| D-004 | `DECIDED` | W4A16, W8A8, and NVFP4 are first-class NML goals. A dtype name alone does not count as quantization support. |
| D-005 | `DECIDED` | Upstream project-service and personal-editor configuration is not ported: ZML's `.github/`, `.nvim.lua`, `.vscode/`, `.zed/`, and similar repository-personalization files are reference material only. This does not prohibit NML from later creating its own CI or editor-neutral config. |
| D-006 | `DECIDED` | NML is an acceleration substrate for experiments, not an attempt to reproduce LLMD or a hosted serving product. |
| D-007 | `DECIDED` | NML will not add a general autograd engine merely to support LoRA experiments. An analytic backward pass may be expressed as another explicitly authored compiled graph. |
| D-008 | `DECIDED` | Rust is NML's core implementation language. Rust owns the safe host runtime, graph frontend, shapes/types, quantization contracts, model-facing APIs, and orchestration. Unsafe foreign interfaces are isolated behind narrow internal boundaries. |
| D-009 | `DECIDED` | Bazel/Bzlmod is NML's build foundation and Rust is integrated with `rules_rust`. Cargo may still describe/import Rust dependencies where useful, but it is not a second top-level build orchestrator. |
| D-010 | `DECIDED` | NML will lift, retain, and adapt as much useful ZML Bazel logic for PJRT plugins and external native libraries as practical. Re-deriving working dependency builds is not a clean-rewrite goal. Reused infrastructure must remain auditable and carry appropriate provenance. |
| D-011 | `DECIDED` | There is no prototype, spike, proof-of-concept, or throwaway implementation or verification phase. Development starts in the intended product architecture. Verification consists of durable unit, integration, end-to-end, numerical, compatibility, and performance tests attached to product capabilities. |
| D-012 | `DECIDED` | Supported hosts are Linux x86-64, Linux ARM64/AArch64, and macOS ARM64/AArch64 (Apple Silicon). Windows and Intel macOS are unsupported. DGX Spark is a required Linux ARM64/CUDA-class host. macOS is CPU-only because Metal is outside NML's accelerator scope. |
| D-013 | `DECIDED` | CPU is both the correctness/reference target and a performance target. CPU paths are not permitted to remain intentionally slow reference-only implementations. |
| D-014 | `DECIDED` | Retain all portions of ZML's CPU and CUDA PJRT loaders and plugin packaging. Adapt every Zig-dependent part to Rust while preserving loader, packaging, compatibility, and hermetic-runtime behavior. Extend CPU PJRT packaging to select Linux ARM64; the pinned artifact release contains that build, but the reference snapshot does not declare or select it. |
| D-015 | `DECIDED` | XLA is NML's compiler, as it is in ZML. NML follows ZML's MLIR/StableHLO-to-XLA-to-PJRT architecture unless a later explicit decision changes a specific layer. |
| D-016 | `DECIDED` | NML supports the complete CUDA capability range supported by the retained ZML CUDA/XLA/PJRT stack; NML does not intentionally narrow that range. Hardware outside the supported range receives a hard, diagnostic error rather than silent fallback. Feature-specific requirements such as native NVFP4 remain explicit capability checks. |
| D-017 | `DECIDED` | `rules_rust` uses the latest available stable Rust compiler, never nightly. Bazel pins the exact stable release for hermeticity and updates that pin when the project updates its toolchain. At this document revision the latest stable release is Rust 1.97.0 (2026-07-09). |
| D-018 | `DECIDED` | When no explicit NML decision or established NML pattern answers a question, follow ZML. Do not create architectural questions merely because a rewrite makes alternatives theoretically possible. Departures from ZML require a clear NML requirement or recorded decision. |
| D-019 | `DECIDED` | ZML's development, release, and repository-support surface listed in section 5.21 is out. NML does not inherit Nix/devenv, remote BuildBuddy configuration, cross-compilation platforms, OCI/tar packaging, release/install scripts, ZLS tooling, buildifier/formatting helpers, logos/terminal assets, or contribution/documentation structure unless the owner explicitly requests a specific addition. Required core Bazel platform constraints and CPU/CUDA dependency/plugin packaging under D-009, D-010, D-012, and D-014 are not release-surface inheritance. |
| D-020 | `DECIDED` | Following ZML, the Bazel platform layer owns independent `cpu` and `cuda` backend build settings in addition to host OS/architecture platforms. CPU defaults on, CUDA defaults off, and enabling CUDA does not disable CPU. These settings select the corresponding PJRT loaders, dependency/package closures, profiling hooks, and backend code as those targets are implemented. |
| D-021 | `DECIDED` | Local Bazel invocations run from the NML repository root and reuse `../nml-bazel-cache` as their output-user-root. The startup option must precede the Bazel command: `bazel --output_user_root=../nml-bazel-cache <command> ...`. Keeping this directory beside rather than inside the repository satisfies Bazel's workspace-containment rule and preserves expensive hermetic XLA/LLVM/Rust actions across builds. |
| D-022 | `DECIDED` | PJRT GPU custom-call registration is a foundational NML capability, not optional future surface. Retain the pinned XLA GPU extension ABI, including untyped and typed calls plus instantiate, prepare, initialize, and execute lifecycle handlers. Rust validates extension discovery and status handling; handler registration remains explicitly unsafe because PJRT exposes handler addresses as untyped `void*` values. |
| D-023 | `DECIDED` | Follow ZML's treatment of MLIR `index`: it is a compiler-internal MLIR type, not a runtime `DataType`, tensor element type, host-storage type, or PJRT buffer type. The MLIR layer still supports index constants and signed/unsigned casts wherever compiler APIs, layouts, loops, memrefs, or dialect operations require them. |
| D-024 | `DECIDED` | NML's initial compiler graph pins OpenXLA/XLA commit `41370d1124c74d7b93a207136a636d8c631cbed9`, matching the ZML reference integration. PJRT headers, XLA schemas, LLVM/MLIR, StableHLO, and Shardy are resolved through this graph rather than independent drifting source pins. |
| D-025 | `DECIDED` | NML does not consume ZML's source forks of LLVM, Zig, XLA, `rules_ml_toolchain`, or other external projects. Compiler sources come from the original OpenXLA repository and its upstream-pinned dependency graph; Rust/Bazel dependencies come from their upstream registries; NVIDIA and system runtime packages come from their original distributors. Two deliberate ZML-derived inputs remain acceptable under D-010 and D-014: NML carries the audited `cuda-root-path-local-defines.patch` locally against upstream OpenXLA, and SHA-256-pinned CPU/CUDA plugin binaries are downloaded from `zml/pjrt-artifacts`. The latter is a ZML-hosted packaging dependency, not a source-fork dependency. Any future dependency on ZML-hosted forked source requires a new explicit decision. |
| D-026 | `DECIDED` | After the first typed CPU/CUDA execution milestone, NML implements the parameter/buffer/checkpoint substrate and then ports ZML's CPU/CUDA-relevant attention architecture before beginning W4A16. This includes portable attention semantics, CUDA FlashAttention, paged KV-cache behavior, and the Triton machinery required by ZML's default CUDA paged-attention path. Quantization follows these layers so packed weights, custom kernels, model parameters, and persistent device memory use the final ownership and dispatch architecture. |
| D-027 | `DECIDED` | BuildBuddy was explicitly added after D-019. Repository configuration exposes an opt-in `--config=buildbuddy` remote cache/results profile without storing credentials; the authenticated `bb` CLI supplies the user's key. Remote Bazel builds that assemble the complete CUDA runtime request at least 20 GB free runner disk with `--runner_exec_properties=EstimatedFreeDiskBytes=20GB`. This changes build placement only, not the hermetic runtime graph or acceptance requirements. |

An item is `UNDECIDED` only where this document identifies a deliberate NML
departure that still needs an exact contract. Otherwise D-018 applies the
`DEFAULT-ZML` rule until NML establishes a contrary decision or pattern.

## 3. What “clean room” means here

The intended engineering discipline is:

1. Keep ZML in `references/zml` as read-only reference material.
2. Identify a behavior or capability NML actually needs.
3. Write an NML-facing requirement and acceptance test before porting the idea.
4. Consult ZML, public specifications, vendor documentation, and independent
   implementations to understand the problem.
5. Implement the product design, following ZML unless an explicit NML decision
   or established NML pattern requires a departure.
6. Record the reference commit and relevant reference paths in the change.
7. Verify behavior with independent test vectors or framework oracles where
   practical.

This is a source-guided clean reimplementation process, not a formal legal
two-team clean-room arrangement: the same developers may read ZML and implement
NML. The label must not be used as a substitute for provenance or license
review. ZML's checked-in license is Apache-2.0 (`references/zml/LICENSE`).

Build and dependency infrastructure is a deliberate exception to the
implementation rewrite rule. NML may directly adapt useful Bazel rules,
repository extensions, PJRT plugin packaging, external library declarations,
patch application, and hermetic toolchain logic from ZML. Such reuse should be
traceable to the reference commit and retain any applicable notices. This
exception does not make `references/zml` itself an NML build dependency.

The reference tree must never become:

- an NML build input;
- a source of mechanically renamed product/runtime/model implementation files;
- a place from which comments or tests are copied without deliberate review;
- an implicit compatibility specification that forces unused behavior into
  NML.

## 4. Snapshot of ZML today

The checked-out snapshot is active rather than dormant: its final week includes
changes to oneAPI, CUDA PJRT artifacts, Hugging Face VFS reads, tokenization,
Triton attention, Metal, profiling dependencies, and `zml-smi`.

Static inventory of tracked source/build files in this snapshot:

| Area | Tracked files | Approx. source/build lines | Observation |
| --- | ---: | ---: | --- |
| `zml/` | 72 | 39,325 | Tensor frontend, operations, compilation, buffers, loading, attention, MoE, tokenization, profiling, and sharding are exposed through one main library. |
| `mlir/` | 23 | 9,806 | Zig wrappers for MLIR plus StableHLO, Shardy, TTIR, Mosaic TPU, and general dialects. |
| `examples/` | 32 | 9,694 | The LLM example alone is about 8,556 lines and contains four model variants. |
| `bin/` | 95 | 8,617 | Almost entirely the independent `zml-smi` monitoring/TUI program. |
| `kernels/` | 14 tracked files, 11 source/build files | 5,963 | Two kernel-construction stacks: Triton and Mosaic TPU. |
| `platforms/` | 60 | 5,497 | Seven accelerator plugins plus packaging and hermetic runtime logic. |
| `stdx/` | 16 | 2,972 | General Zig utilities used across the project. |
| `pjrt/` | 3 | 2,326 | Broad PJRT C API and FFI wrapper. |
| `tools/` | 22 tracked files, 18 source/build files | 1,449 | Profilers, Hugging Face helper, buildifier, ZLS, workspace status, and other utilities. |
| `third_party/` | 62 | 1,454 local wrapper/patch lines | Dependency declarations and patches; fetched dependencies are much larger than these checked-in lines. |
| `bazel/`, `ffi/`, `upb/`, root build files | — | about 1,963 | Build rules, C interop, protobuf support, and workspace configuration. |

Across the snapshot there are 459 tracked files, including 223 Zig files and
roughly 89,066 lines of Zig/C/C++/Python/shell/Bazel/Starlark source and build
logic. These figures describe review surface, not runtime size.

### 4.1 Architectural path

The central ZML path is:

```text
Zig model function using Tensor/Shape
        |
        v
MLIR construction (func + StableHLO + Shardy, with custom calls)
        |
        v
XLA compilation through a PJRT plugin
        |
        v
PJRT LoadedExecutable
        |
        v
Buffer arguments/results across one or more devices
```

Important surrounding paths are:

- safetensors/model metadata -> `TensorStore` -> parallel buffer loading;
- logical tensor tags -> partition specs -> physical topology -> Shardy
  annotations;
- generic StableHLO ops plus custom attention/MoE kernels;
- platform-specific PJRT plugins and packaged runtime libraries;
- tokenizer, VFS, chat/session, sampling, and model implementations in the LLM
  example.

The core conceptual split is strong and worth evaluating on its merits:

- `Shape`: tensor metadata;
- `Slice`: shaped host bytes;
- `Buffer`: shaped accelerator allocation(s);
- `Tensor`: symbolic value during graph construction;
- `Exe`: compiled executable over buffers.

See `references/zml/docs/learn/concepts.md` and
`references/zml/zml/{shape,slice,buffer,tensor,module,exe}.zig`.

### 4.2 What the public repository is and is not

The root README calls ZML a production inference stack, but this snapshot's
public inference product surface is primarily an example CLI plus reusable
acceleration components. The repository contains model loading, tokenization,
prefill/decode sessions, KV caches, sampling, attention backends, and model
implementations. It does not contain a general production request scheduler,
OpenAI-compatible server, continuous batching service, or the closed LLMD
product.

That distinction matters when deciding whether a component belongs to an
acceleration substrate, an inference library layered on it, an example, or a
separate serving product.

## 5. Confirmed complexity axes

### 5.1 Accelerator and platform breadth

ZML's platform enum contains CPU, CUDA, ROCm, TPU, Neuron, oneAPI, and Metal
(`platforms/platforms.zig:12`). Each target affects more than a loader:

- Bazel flags and repository extensions;
- PJRT plugin artifacts and dynamic loading;
- OS/architecture constraints;
- device discovery and memory-kind behavior;
- physical mesh construction;
- attention and MoE backend selection;
- custom kernels and dialects;
- profiler integration;
- CI and packaging.

This axis is decided:

- hosts are Linux x86-64, Linux ARM64/AArch64, and macOS ARM64/AArch64;
  Windows and Intel macOS are out, and macOS is CPU-only;
- DGX Spark is a Linux ARM64 target (NVIDIA documents its CPU as ARM64-based);
- CPU is both the correctness/reference backend and a performance backend;
- all ZML CPU/CUDA PJRT loader and plugin-packaging behavior is retained, with
  Zig-dependent implementation adapted to Rust;
- NML adds Linux ARM64 CPU PJRT packaging: ZML's pinned artifact release
  contains the Linux ARM64 build, but ZML currently declares only Linux x86-64
  and macOS x86-64/ARM64 CPU repositories, while its CUDA
  packaging already includes Linux ARM64. NML retains only the macOS ARM64
  artifact from that macOS pair;
- XLA remains the compiler behind PJRT;
- NML supports the full CUDA capability range supported by the retained ZML
  CUDA/XLA/PJRT stack rather than choosing a smaller range;
- unsupported GPUs and feature-specific unsupported capabilities produce hard,
  diagnostic errors rather than silent fallback.

DGX Spark architecture reference:
[NVIDIA DGX Spark system overview](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html).

### 5.2 Dtype breadth versus quantization breadth

ZML declares 28 scalar `DataType` cases: bool; thirteen floating-point formats
including FP4, eight FP8 variants, BF16, FP16, FP32, and FP64; six signed
integer widths; six unsigned widths; and two complex types
(`zml/dtype.zig:8`). This broad enum fans out into:

- Zig host representations and conversions (`dtype.zig`, `floats.zig`);
- MLIR type conversion (`mlirx.zig`);
- PJRT buffer types (`pjrtx.zig`, `pjrt/pjrt.zig`);
- constants, comparisons, formatting, slicing, and tests;
- StableHLO operation validity;
- kernel-DSL type support;
- file-format import support.

The layers do not support the same matrix:

- `DataType` declares 28 cases.
- `mlirx.Type.fromDType` explicitly panics for `f8e8m0`.
- the safetensors importer recognizes 14 file dtypes and only one FP8 spelling
  (`zml/safetensors.zig:753`);
- the Triton builder exposes 11 dtypes and has no 4-bit integer or FP4 kernel
  type (`kernels/triton/dtype.zig`);
- various operations accept still smaller subsets.

This is evidence that “the type exists” is not a useful support claim.

The ordinary scalar set is now decided by D-003. Support is still claimed per
layer rather than inferred from enum membership, and MLIR `index` remains a
separate compiler-only type under D-023:

| Type | Host storage | File import | CPU graph | CUDA graph | Custom kernels | Public API |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 | required | required | required | required | where used | required |
| FP16 | required | required | required | required | where used | required |
| BF16 | required | required | required | required | where used | required |
| Bool; I8/I16/I32/I64; U8/U16/U32/U64 | required | as formats require | required | required | as needed | required |
| FP64 | required | required | required | required | where used | required |
| C64/C128 | required | required | required | required | where used | required |
| MLIR index | none: not a storage type | none | compiler internals only | compiler internals only | none | not a `DataType` |
| 2-bit/general FP8 | outside the ordinary set unless a quantization contract requires one | — | — | — | — | — |

### 5.3 Quantization is a representation and kernel contract

ZML already contains names that resemble the desired features, but not the
desired end-to-end support:

- `DataType` includes `i4`, `u4`, and `f4e2m1`.
- `Shape.byteSize()` is `dtype.sizeOf() * element_count`, where `sizeOf()` is
  Zig `@sizeOf`, so the ordinary shape/storage abstraction does not itself
  specify packed nibbles (`shape.zig:501`, `dtype.zig:146`).
- safetensors import has no 4-bit case in this snapshot.
- the Triton DSL has no 4-bit dtype.
- MoE options name FP8 W8A8, INT8 W8A8, INT8 W8A16, and INT4 W4A16, but
  `validateOptions` rejects all of them (`zml/moe/triton.zig:22` and `:641`).
- that same MoE path requires BF16 activations and BF16 or FP8 weights
  (`zml/moe/triton.zig:652`).
- the fused MoE kernel describes itself as BF16/no-quant/no-bias and rejects
  its quant flags (`zml/moe/triton_kernels/triton_kernels.zig:168`).

NML therefore needs an explicit decision on whether these are separate types:

```text
logical tensor element type
compute/accumulation type
physical packed storage type
quantization scheme
scale and zero-point tensors
packing order and alignment
kernel capability
checkpoint encoding
```

Combining them into one `DType` is one option; separating them is another. The
choice is `UNDECIDED`, but it must be conscious.

#### W4A16 questions

“W4A16” only states weight and activation widths. It does not uniquely specify:

- signed/symmetric versus asymmetric values;
- AWQ, GPTQ, or another calibration/layout convention;
- group size and grouping axis;
- scale and zero-point dtype;
- nibble packing order;
- transposed/prepacked CUDA layouts;
- whether weights are unpacked, dequantized, or consumed directly by the
  kernel;
- FP16 versus BF16 activation/output variants;
- CPU reference semantics and CUDA capability requirements.

Each supported checkpoint convention needs an importer into one declared NML
in-memory contract, or it must remain a distinct contract.

#### W8A8 questions

“W8A8” also leaves choices open:

- INT8 or FP8 values;
- static or dynamic activation quantization;
- per-tensor, per-token, per-channel, or grouped scales;
- symmetric versus asymmetric quantization;
- INT32 or floating-point accumulation;
- output and requantization dtype;
- SmoothQuant or other preprocessing assumptions;
- weight-only persistence versus quantized activations between operators.

The first NML W8A8 recipe is `UNDECIDED`.

#### NVFP4 questions

NVFP4 is not merely a generic E2M1 scalar. NVIDIA's current description uses:

- an E2M1 4-bit value;
- a local E4M3 scale shared by 16 consecutive elements;
- a global FP32 scale;
- optional 2D weight scaling in addition to 1D scaling.

Primary references:

- [NVIDIA Transformer Engine NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.16/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- [NVIDIA cuDNN frontend block-scaling documentation](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.23.0/operations/BlockScaling.html)
- [TensorRT-LLM quantization overview](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md)

NML must still decide:

- which checkpoint encodings it imports;
- 1D versus 2D weight scaling support;
- the canonical packed layout and transpose semantics;
- whether activation quantization is offline, runtime, or both;
- native Blackwell-only execution versus any fallback;
- which layer types receive NVFP4 paths;
- whether NVFP4 KV cache is in scope separately from NVFP4 GEMM.

### 5.4 Build and dependency system

ZML uses Bazel/Bzlmod, a pinned Zig 0.16 toolchain, a bootstrapped LLVM
toolchain, XLA/StableHLO/Shardy/Triton sources, PJRT plugin archives, platform
runtime packages, Python toolchains, apt repositories, OCI images, tar rules,
SWIG, protobuf/upb, and numerous patched third-party repositories.
`MODULE.bazel` has 27 direct `bazel_dep` declarations and many extension/repo
registrations. `build.zig` is empty; it is not a working alternate build.

This breadth comes from several independent requirements. Bazel/Bzlmod and
`rules_rust` are already selected by D-009, and D-010 establishes a strong
preference for retaining working external build integration. The remaining
scope is decided per concern:

| Concern | What ZML gets from it | NML decision |
| --- | --- | --- |
| Hermetic Rust compiler | Reproducible compiler version and cross builds | `DECIDED`: latest stable through `rules_rust`, pinned exactly; currently 1.97.0; nightly prohibited |
| XLA/LLVM build | Compiler backend and MLIR APIs | `DECIDED`: retain/follow ZML and adapt Zig-facing edges to Rust |
| PJRT plugin packaging | CPU/CUDA runtimes arrive with expected shared libraries | `DECIDED`: retain all ZML CPU/CUDA logic, adapt Zig-dependent parts, and add Linux ARM64 CPU packaging |
| CUDA redistribution packaging | Hermetic CUDA/cuDNN/NCCL/runtime sandbox | `DECIDED`: retain all ZML CPU/CUDA packaging behavior and adapt as required |
| Remote execution/cache | Makes very large builds practical in CI | `DECIDED`: explicitly added by D-027 as an opt-in BuildBuddy profile; credentials remain outside the repository |
| Python toolchains | Hugging Face helper and profiler tooling | `DEFAULT-ZML` for retained features; excluded-platform tooling is out |
| OCI/tar/deploy rules | Cross-build and deployment artifacts | `OUT` by D-019 unless explicitly requested |
| Nix/devenv | Contributor environment convenience | `OUT` by D-019 unless explicitly requested |

The task is therefore not to reconsider Bazel, but to prune and adapt its graph
without breaking hermetic PJRT, XLA/LLVM, CUDA, or other selected external
builds. Cargo must not invoke an opaque nested Bazel build from `build.rs`;
Bazel remains the owner of cross-language linking and native dependency
construction.

Every local command in this document and future implementation work uses the
shared relative cache explicitly from the repository root:

```text
bazel --output_user_root=../nml-bazel-cache build //...
bazel --output_user_root=../nml-bazel-cache test //...
```

`--output_user_root` is a Bazel startup option, so it appears before `build`,
`test`, `query`, or any other command. The relative spelling is normative; it
keeps the workspace relocatable while ensuring all NML builds reuse one sibling
cache.

### 5.5 Graph-construction model

ZML runs an ordinary Zig function while a thread-local `CompilationContext` is
active. Tensor methods append MLIR operations to the current block. Zig
reflection flattens tensor-bearing structs into function arguments and rebuilds
buffer-bearing result structs. See `zml/module.zig`, `zml/meta.zig`, and
`zml/mem.zig`.

Benefits:

- model code looks like direct Zig;
- arbitrary nested structs can describe parameters/state;
- shape and tag checks happen while building the graph;
- input/output donation can be inferred from returned tensors.

Costs or constraints:

- graph construction depends on hidden thread-local state;
- many errors panic or use `catch unreachable` rather than returning structured
  diagnostics;
- compilation is currently forced through a single-threaded `std.Io` context;
- reflection and tensor identity rules are part of the effective ABI;
- dynamic control flow must be expressed through graph operations, not ordinary
  data-dependent Zig branching.

D-018 applies: NML retains this model semantically. Zig reflection and
thread-local machinery are adapted to Rust traits, derives, and a safely scoped
compilation context; changing the graph-construction model requires an explicit
NML decision.

### 5.6 Shape, tags, layout, and placement

ZML shapes have a maximum rank of eight and combine:

- dimensions;
- dtype;
- optional semantic string tags;
- a partition spec for every axis.

Tagged axes make model code readable and prevent many transpose/dot mistakes.
They are used pervasively by model implementations. Partition metadata flowing
inside the same shape object lets operations preserve distribution intent.

The combined design also couples local shape manipulation to distributed
placement. NML retains ZML's semantic tags, required/optional tag behavior,
rank limit, layout ownership, placement organization, and static/dynamic shape
semantics by default. Their Rust representation may differ, but their behavior
does not change merely because the implementation language does. A concrete
workload or recorded decision must justify any departure.

### 5.7 Operation breadth

`zml/tensor.zig` is about 4,617 lines, `zml/ops.zig` about 2,347, `zml/nn.zig`
about 1,892, and `zml/shape.zig` about 1,744. They cover far more than matrix
multiplication: indexing, gathers/scatters, reductions, convolution, FFT,
randomness, sampling, control flow, collectives, custom calls, dynamic updates,
and extensive shape/tag transformations.

This breadth enables unusual modalities and analytic backward graphs. It also
multiplies dtype, shape, sharding, and backend test combinations.

NML ports ZML's CPU/CUDA-relevant operation semantics by default. Code whose
only purpose is an excluded accelerator is omitted, while new quantization
operations extend rather than redefine the inherited surface. Operation-level
cuts require explicit decisions; the existence of a smaller theoretical API is
not itself a reason to create one.

### 5.8 Generic kernels and kernel DSLs

ZML has:

- a shared kernel DSL layer;
- a Zig Triton/TTIR builder of about 2,245 lines;
- a Mosaic TPU builder and pipeline;
- attention and MoE kernels written on those builders;
- external CUDA FlashAttention custom calls.

CPU/CUDA removes the Mosaic TPU builder and pipeline as product requirements.
NML otherwise retains ZML's shared kernel layer, Triton/TTIR builder, and
external CUDA FlashAttention mechanisms, adapting their Zig portions to Rust.
Quantized GEMMs extend this architecture with whichever XLA, Triton, cuBLASLt,
CUTLASS, or CUDA mechanisms their recorded contracts require. The kernel
architecture is not redesigned without an explicit decision.

For every mechanism considered, record:

- supported dtypes and quant schemes;
- hardware capability checks;
- compilation/JIT/AOT behavior;
- binary distribution implications;
- tuning and cache requirements;
- correctness oracle;
- fallback behavior.

### 5.9 Attention breadth

Standard attention has six backends in one tagged union: vanilla StableHLO,
networked `attnd`, Neuron NKI, CUDA FlashAttention 2, CUDA FlashAttention 3, and
Metal FlashAttention (`zml/attention/attention.zig:10`). Paged attention has a
separate five-backend union spanning CUDA FlashAttention, Triton, Mosaic TPU,
and Metal (`zml/attention/paged_attention.zig:13`). Metadata, parameters,
allocation, and KV layouts vary by backend.

NML retains ZML's CPU/CUDA-relevant portable attention, CUDA FlashAttention,
Triton paged-attention, prefill/decode specialization, KV layouts, attention
modes, batching semantics, and custom-call behavior. Neuron, TPU, and Metal
implementations are out as targets. D-016 governs capability checks and hard
errors. New modality, speculative-decoding, and quantized-KV requirements may
extend these behaviors through explicit decisions.

### 5.10 MoE breadth

ZML has Triton, Mosaic TPU, and Metal MoE backends plus topology-aware expert
sharding. The current Triton implementation contains launch heuristics and
partial FP8 activation quantization, but its principal fused kernel still
rejects the advertised quant modes.

NML retains ZML's CPU/CUDA-relevant MoE semantics, routing primitives,
topology-aware expert sharding, and Triton backend behavior. Mosaic TPU and
Metal implementations are out as targets. Quantized MoE gaps are filled under
D-004 rather than used as a reason to redesign or defer the inherited MoE
surface.

### 5.11 Device count, sharding, and topology

`zml/Sharding.zig` is about 2,345 lines. It models:

- logical meshes and intent;
- physical axes such as link, torus dimensions, and bus;
- point-to-point, ring, mesh, tree, and isolated geometry;
- automatic CPU/GPU/TPU/Neuron topology construction;
- logical-to-physical binding and folding;
- Shardy and legacy GSPMD attributes;
- replicated and partitioned host/device buffers;
- placement and host slicing.

This is one of ZML's largest independent complexity axes. It enables
multi-device workloads and model/expert/data partitioning, which may be
important for large models even with quantization. It also contains topology
concepts that exist specifically for excluded accelerators.

NML retains ZML's multi-device, topology, Shardy/GSPMD, collective, executable,
and sharded-buffer behavior for CPU and CUDA. Topology cases that exist solely
for excluded accelerators are omitted as targets. Quantized packed layouts must
preserve correct sharding and loading without accidental repacking or
dequantized persistent copies.

### 5.12 Memory ownership, loading, and mutation

ZML supports:

- host `Slice` views, including non-contiguous strides;
- sharded `Buffer` objects over PJRT buffers;
- default, device, pinned-host, and unpinned-host memory kinds;
- buffer donation/reuse and output aliasing;
- uninitialized buffers and device-pointer access;
- parallel/chunked safetensors loading and DMA;
- reflection-based `Bufferized(Model)` structures.

This layer is relevant to all three named quant formats: a system can execute a
quantized kernel while still wasting memory if it retains a persistent
dequantized copy.

NML retains ZML's ownership semantics, `Slice`/`Buffer` split, memory kinds,
donation and aliasing behavior, parallel loading, and reflection-based
`Bufferized` behavior, expressed through Rust ownership and RAII. Quantized
formats add a strict requirement that packed weights remain packed in persistent
host and device storage. Other changes require an explicit decision.

### 5.13 Model and checkpoint formats

ZML centers its loader around safetensors repositories and Hugging Face-style
`config.json`. Its LLM example detects Llama, LFM2.5, Qwen3.5 dense, and
Qwen3.5 MoE model types. Quantized checkpoint ecosystems add separate config
schemas, auxiliary scale/zero-point tensors, and packed layouts.

NML retains ZML's safetensors, Hugging Face configuration, model-repository,
tied-weight, alias, and loader semantics by default. No additional container is
implicitly added. The accepted quantization encodings and metadata remain
explicit decisions because W4A16, W8A8, and NVFP4 are deliberate departures;
LoRA adapter overlays are an explicit addition required by D-007's use cases.

### 5.14 Filesystem and remote IO

ZML's LLM example registers local file, Hugging Face, S3, and GCS backends into
a custom `std.Io` virtual filesystem (`examples/llm/main.zig:55`). The VFS
implements virtual handles and routes normal directory/file operations across
schemes. S3 includes SigV4 credentials, listing, ranged parallel reads, retries,
and endpoint/region discovery; GCS and Hugging Face add their own behavior.

This makes remote repositories look local to the loader. It also places cloud
auth, HTTP semantics, retries, concurrency, and virtual handle correctness in
the acceleration repository.

NML retains ZML's local, Hugging Face, S3, and GCS VFS behavior and transparent
repository semantics by default, adapting `std.Io` interfaces to Rust. A later
replacement with explicit download/cache adapters would require a recorded
decision.

### 5.15 Tokenization and preprocessing

ZML presents one tokenizer union over:

- an IREE tokenizer runtime for Hugging Face `tokenizer.json`;
- SentencePiece through C++ and SWIG;
- a roughly 1,243-line homemade tokenizer.

Together the tokenizer subtree is about 2,616 source/build lines before fetched
IREE, SentencePiece, SWIG, and protobuf code. Tokenization is needed by the LLM
example but not by tensor compilation itself. Other modalities have their own
pre/postprocessing needs.

NML retains ZML's tokenizer formats, behavior, and package boundary by default,
including the IREE, SentencePiece, and native implementations, with bindings
adapted to Rust. Other modalities extend the preprocessing surface when their
product requirements are implemented; they do not force a speculative boundary
redesign now.

### 5.16 Profiling and observability

ZML includes two profiling layers:

- host `TraceMe` scopes bridged to NVTX/ROCTx/macOS signposts;
- PJRT profiler sessions producing XSpace protobuf, merged host traces, and a
  streamed Perfetto JSON conversion.

This pulls profiler-specific C++, protobuf/upb targets, XLA/TSL libraries, and
the `xspace_to_perfetto` tool into the main ZML target. External wrappers also
exist for Nsight Systems, rocprofv3, XProf, and Neuron profiling.

NML retains ZML's CPU/CUDA/macOS-relevant wall-clock utilities, TraceMe/NVTX
annotations, PJRT profiling, XSpace/Perfetto export, Nsight integration, and
benchmark behavior. ROCm-, Neuron-, TPU-, and Metal-accelerator-specific pieces
are out as product targets. CPU and CUDA performance claims require durable
profiling and benchmark evidence.

### 5.17 `zml-smi` and operational tooling

`bin/zml-smi` is a substantial independent application with collectors,
process/device information, CSV/JSON/Prometheus outputs, API client/server,
TUI widgets, images, release/install scripts, and support for NVIDIA, AMD,
Intel, TPU, Neuron, Linux, and macOS. It accounts for 95 tracked files and
about 8,617 source/build lines.

NML retains its supported-platform operational behavior by default, adapting
the application to Rust and removing collectors for excluded accelerators.
Removing the monitoring product entirely would require an explicit decision.

### 5.18 Examples, models, and user-facing inference

The examples include MNIST, IO, benchmarking, sharding, and a large LLM CLI.
The LLM example owns:

- model detection and model-specific Zig implementations;
- compilation and buffer-loading lifecycle;
- contiguous KV caches and sessions;
- prefill/decode execution;
- sampling;
- chat loop and terminal UI;
- tokenizer loading;
- remote repository resolution;
- optional profiling.

NML retains CPU/CUDA-relevant example, model, session, and inference behavior by
default and rewrites it in Rust. Examples become durable executable product and
end-to-end verification targets; they are not throwaway demonstrations or
throwaway checks. Moving a capability across a package boundary requires a
concrete reason, not a rewrite-wide reclassification exercise.

### 5.19 Testing and support claims

The snapshot has roughly 116 Zig test blocks. Tests are mostly colocated with
implementation. Current CI builds CPU, CUDA, ROCm, TPU, and Neuron variants;
it runs tests for CPU, CUDA, and ROCm, while TPU and Neuron are build-only in
that workflow. oneAPI and Metal are not in the checked-in CI matrix.

NML needs an explicit definition of “supported.” Candidate evidence dimensions
are:

- build coverage;
- unit/shape tests;
- CPU numerical reference;
- CUDA numerical comparison;
- loader-to-kernel end-to-end test;
- real model quality check;
- measured resident memory;
- latency/throughput benchmark;
- hardware capability coverage;
- failure behavior for unsupported combinations.

ZML's tests and observable behavior are the baseline. NML adds durable CPU and
CUDA build, numerical, loader-to-kernel, real-model, memory, performance,
capability, and failure-behavior gates appropriate to each supported feature.
Exact tolerances for the new quantization recipes remain explicit decisions,
but a quant mode cannot be marked supported based only on parsing, enum
plumbing, or compilation.

CUDA package integrity and CUDA device execution are separate permanent
contracts. Package integrity runs hermetically and remains cacheable. Device
execution runs locally without test-result caching because the physical GPU,
driver, and `/dev/nvidiactl` are host state outside Bazel's declared file
inputs; sandboxing or reusing a result from different hardware would make the
support claim invalid. CUDA compiler and runtime artifacts remain normally
cached.

### 5.20 Error model and API stability

ZML frequently uses assertions, panics, `catch unreachable`, compile errors,
and ordinary error unions. Some APIs are public because the umbrella module
re-exports them, even where comments describe limitations or TODOs.

NML follows ZML's error categories, diagnostics, and API boundaries by default,
expressed idiomatically through Rust results, assertions, and narrowly justified
panics. D-016 adds one firm exception: unsupported GPU capability is a hard,
diagnostic error, never a silent fallback. Other changes to checkpoint failure,
IR diagnostics, public package stability, or compatibility require explicit
decisions.

### 5.21 Development, release, and repository surface

Beyond the already excluded upstream `.github` and editor files, ZML contains:

- Nix/devenv files;
- remote BuildBuddy execution configuration;
- cross-compilation platforms;
- OCI and tar packaging;
- release/install scripts;
- ZLS completion tooling;
- buildifier and formatting helpers;
- logos and terminal assets;
- contribution and documentation structure.

Everything listed above is `OUT` under D-019 and is not inherited or recreated
unless the owner explicitly requests that specific surface. This exclusion does
not remove the core Bazel platform constraints needed to build on supported
hosts or the CPU/CUDA dependency, PJRT loader, and plugin-packaging logic
required by D-009, D-010, D-012, and D-014.

### 5.22 Rust as the core language

Rust is suitable for every host-side responsibility Zig performs in the ZML
reference, but the rewrite must recognize that Zig is more than FFI glue. In
ZML, Zig currently provides:

- raw PJRT, MLIR, upb/protobuf, POSIX, and platform-library interop;
- ownership wrappers for clients, devices, memories, buffers, executables, and
  asynchronous events;
- the embedded tensor graph frontend that emits MLIR/StableHLO;
- compile-time reflection over nested model structures;
- shape, dtype, sharding, loading, model, session, and orchestration code;
- builders for custom kernel IR.

Rust replaces the first two roles directly and strengthens the ownership
boundary. Raw handles and C layouts can live in a narrow unsafe crate, while
RAII wrappers own destruction and encode client/device/buffer relationships.
PJRT's C API is specifically intended as a framework-to-plugin boundary, and
MLIR's C API is intended to be wrapped by higher-level languages. Neither
requires Zig as the consumer language.

The PJRT loader topology follows ZML rather than introducing a generic backend
abstraction: one common C-API/object layer is shared by separate CPU and CUDA
platform loaders. CPU and CUDA each own their plugin artifact closure and
platform-specific initialization. CUDA compatibility-driver selection and
hermetic libraries remain in the CUDA layer, while the common PJRT layer owns
PJRT's typed extension-chain traversal and GPU custom-call registration ABI,
as ZML's common PJRT wrapper does. NML's departures at this boundary are
strictly Rust adaptations: Bazel-owned bindgen replaces Zig `translate-c`,
borrows/RAII encode PJRT object lifetimes, and C status values become
structured Rust errors. As in ZML, a loaded PJRT library remains resident for
the process lifetime because PJRT defines plugin initialization but no
symmetric plugin shutdown operation.

Rust does not remove XLA, MLIR, StableHLO, PJRT, CUDA, or kernel complexity. It
changes who owns the frontend and how safely NML expresses the host runtime.
CUDA kernel performance remains determined by the selected CUDA, cuBLASLt,
CUTLASS, Triton, XLA, or other kernel implementation rather than by whether
the host caller is Rust or Zig.

The main non-mechanical adaptation is Zig compile-time reflection. ZML can
recursively discover `Tensor` fields in arbitrary values and synthesize a
structurally equivalent `Bufferized(T)` type. NML preserves that capability in
Rust through explicit traits and derive/procedural macros, for example:

```rust
#[derive(NmlStruct)]
struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}
```

The derive generates traversal, argument flattening, buffer counterparts,
result reconstruction, and checkpoint-field mapping as auditable Rust code.
The default contract follows ZML's behavior; Rust syntax and generated type
names are adaptations rather than reasons to redesign the model lifecycle.

The graph frontend likewise follows ZML's scoped thread-local
`CompilationContext` model by default. Rust may implement activation with a
scoped guard and safe wrapper around thread-local state, but a different graph
construction architecture requires a later explicit NML decision.

The intended responsibility boundaries are:

```text
nml-sys
  Raw generated or hand-audited PJRT, MLIR, XLA FFI, and CUDA bindings.
  Unsafe code is contained here.

nml-runtime
  Safe Client, Device, Memory, Buffer, Executable, Event, and plugin wrappers.

nml-ir
  Shape, Tensor, Graph, operations, IR emission, compilation, and diagnostics.

nml-derive
  Procedural derives for model traversal, buffer structures, and checkpoint
  mapping where the final API calls for generation.

nml-quant
  Packed storage, quantization recipes, scale metadata, checkpoint import,
  capability checks, and quantized operation contracts.

nml-kernels
  Custom-call ABI, kernel registration, dispatch, and external CUDA/native
  kernel integration.

nml-models or product-level consumers
  Model and workload implementations layered on the substrate.
```

These are architectural responsibility boundaries, not necessarily the final
crate count or names.

Rust's key advantages for NML are:

- ownership and destruction of FFI resources;
- explicit safe/unsafe separation;
- structured error propagation across checkpoint/runtime/compiler layers;
- thread-safe multi-model and multi-session orchestration;
- procedural macros for auditable generated model plumbing;
- strong enums and structs for separating logical dtype, packed storage,
  quantization recipe, scale tensors, accumulation type, and kernel capability;
- a mature general systems ecosystem without requiring the external compiler
  stack to be rewritten.

The costs are:

- Zig `comptime` APIs cannot be translated literally;
- procedural macros introduce generated-code and diagnostics design work;
- Rust wrappers must track the evolving MLIR C API and any experimental XLA
  custom-call ABI just as Zig wrappers do;
- cross-language Bazel linking, bindgen, and native library packaging must be
  kept coherent;
- CUDA kernels generally remain in external native/kernel languages.

D-009 resolves the build integration: Bazel is the top-level graph and
`rules_rust` supplies Rust libraries, binaries, tests, and procedural macros.
Where raw bindings are generated, Bazel-owned bindgen rules or checked-in
audited bindings may be used; the choice is per interface. Rust crates from the
Cargo ecosystem may be imported into Bazel, but Cargo does not own the native
PJRT/XLA/CUDA build.

D-011 resolves the development posture. There is no feasibility gate for Rust
and no preliminary matmul, plugin-load, custom-call, or model-derive exercise
whose result is thrown away. Those capabilities are implemented directly in
their final product layers and verified by durable tests that remain part of
the repository.

Primary interface/build references:

- [OpenXLA PJRT uniform device API](https://openxla.org/xla/pjrt)
- [MLIR C API design and ownership model](https://mlir.llvm.org/docs/CAPI/)
- [OpenXLA custom-call and XLA FFI documentation](https://openxla.org/xla/custom_call)
- [Bazel `rules_rust` documentation](https://bazelbuild.github.io/rules_rust/)
- [Rust foreign-function interface guidance](https://doc.rust-lang.org/nomicon/ffi.html)
- [Rust 1.97.0 stable release announcement](https://blog.rust-lang.org/2026/07/09/Rust-1.97.0/)

## 6. Requirements imposed by the intended experiments

These workloads do not decide implementations, but they prevent apparently
safe cuts from being evaluated only against one autoregressive LLM.

### 6.1 Multiple modalities

Potential substrate requirements:

- convolution and image/video layout operations;
- FFT or signal operations for audio;
- variable ranks and semantic axes;
- cross-attention and bidirectional attention;
- dynamic or multiple input shapes;
- explicit pre/postprocessing boundary;
- checkpoint naming/config flexibility.

For each modality, NML should add a concrete acceptance workload before using
it to justify broad operator coverage.

### 6.2 Speculative decoding

Potential substrate requirements:

- multiple compiled models/executables in one process;
- independent or shared tokenizer/vocabulary contracts;
- fast target verification over token blocks;
- KV cache checkpoint, fork, rollback, truncate, or replay semantics;
- deterministic sampling state;
- efficient variable-length decode graphs;
- optional model/state placement across devices.

A serving scheduler is not automatically required by speculative decoding; the
state primitives and orchestration boundary must be decided separately.

### 6.3 LoRA with an explicit analytic backward graph

Potential substrate requirements:

- trainable FP16/BF16/FP32 adapter buffers over frozen or quantized base
  weights;
- user-authored forward and backward compiled graphs;
- reductions and outer products needed by chosen LoRA layers;
- mutable optimizer state and parameter updates;
- buffer donation/aliasing without corrupting saved activations;
- deterministic RNG if dropout is used;
- mixed precision and accumulation rules;
- checkpointing only adapter and optimizer state;
- optional fused base-plus-adapter linear kernels.

This does not require general autograd, but it may require operations and
memory semantics that pure stateless inference does not.

### 6.4 Ordered product milestones after the first execution path

The first milestone established and validated this complete path on CPU and a
real CUDA device:

```text
typed Rust graph
  -> owned and verified StableHLO/MLIR
  -> XLA compile options and portable artifact
  -> PJRT compilation, transfer, and execution
  -> CPU/CUDA numerical results
```

The next milestones are ordered by dependency, not by ease of demonstration.
Each milestone is complete only when its durable CPU/CUDA, ownership, failure,
numerical, and where applicable memory/performance contracts pass. A parser,
enum, emitted operation, registered symbol, or compiled kernel by itself does
not complete a milestone.

#### Milestone 2: parameter, buffer, and checkpoint substrate

Path:

```text
typed and aligned host storage
  -> Slice/view shape, stride, layout, offset, and endian contracts
  -> persistent device Buffer and memory-kind ownership
  -> named executable parameters and reusable activation bindings
  -> Rust structural traversal/derive support
  -> safetensors field, alias, and tied-weight loading
  -> repeated FP16/BF16 parameterized execution
```

This milestone preserves ZML's conceptual `Tensor`/`Slice`/`Buffer`/`Exe` and
`Bufferized(T)` lifecycle in Rust. Ordinary product APIs stop exposing raw byte
vectors as the tensor abstraction. Parameters are uploaded once, retained on
the selected device, and reused across executions; transfers, donation,
aliasing, and destruction remain explicit. Checkpoint loading must not create
an untracked persistent converted copy.

Acceptance includes loading an FP16 and BF16 linear layer with optional bias
from safetensors, compiling it once, executing multiple inputs on CPU and CUDA,
checking numerical results against independent host math, and proving that
parameter storage is neither reloaded nor duplicated between invocations.

#### Milestone 3: portable attention semantics and KV state

Path:

```text
required tensor operations
  -> Q/K/V projection and semantic head axes
  -> RoPE, causal/sliding masks, and scaled dot-product attention
  -> MHA, GQA, and MQA
  -> prefill and decode graph forms
  -> persistent KV allocation and updates
  -> paged-cache/page-table semantics
```

The operation layer grows through the behavior required by the retained ZML
attention paths: elementwise arithmetic, broadcast, reshape, transpose,
conversion, gather/slice/update, concatenation, reductions, normalization, and
softmax. CPU portable attention is both the correctness oracle and a
performance implementation under D-013; the same StableHLO path also remains
available to CUDA/XLA.

Acceptance covers causal and non-causal attention, sliding windows, multiple
query-to-KV head ratios, prefill, single-token and multi-token decode, and KV
updates across successive executions. Results and cache contents are compared
against independent reference math, including boundary page and sequence
lengths. The cache lifecycle must already support the later checkpoint,
rollback, truncate, and replay work needed by speculative decoding.

#### Milestone 4: CUDA FlashAttention and Triton paged attention

ZML's ordinary CUDA attention selects FlashAttention 3 for compute capability
9.0 and FlashAttention 2 otherwise. Its default CUDA paged-attention backend is
Triton. Retaining the CPU/CUDA attention architecture therefore requires two
kernel integration paths rather than treating Triton as unrelated future
tooling.

FlashAttention path:

```text
upstream FlashAttention source and audited Bazel packaging
  -> Rust loader and tensor/parameter ABI
  -> PJRT GPU custom-call lifecycle registration
  -> FA2/FA3 capability dispatch and hard failures
  -> ordinary plus paged prefill/decode execution
```

D-025 still applies: ZML's build and loader logic is reference material, but a
ZML-hosted FlashAttention source fork is not silently introduced. Required
changes are either obtained from the original upstream project or carried as
audited local patches. FA2 is validated on supported local hardware. FA3 is not
called validated until its permanent contract runs on real compatible
hardware; compilation alone is insufficient.

Triton path:

```text
pinned XLA Triton/TTIR sources
  -> narrow TTIR C bindings
  -> safe Rust Builder, Value, dtype, argument, and control-flow API
  -> isolated throwaway MLIR context for TTIR emission
  -> typed Kernel input/output/config/launch contract
  -> TTIR-bearing StableHLO custom call
  -> XLA lowering and CUDA execution
  -> unified 2D/3D paged-attention kernels and segment reduction
```

The Rust Triton layer preserves the useful structure of ZML's
`kernels/triton`, `mlir/dialects/ttir`, `zml/kernel.zig`,
`zml/attention/triton_attention.zig`, and
`zml/attention/triton_kernels/unified_attention.zig`. Rust traits, generics, or
derive code may replace Zig `comptime`, but the result retains typed named
arguments, explicit output shapes and aliases, verified TTIR, launch grids,
warp/stage configuration, and deterministic failure behavior. The oneAPI
kernel specialization is outside D-002; this does not remove the shared Triton
behavior used by CUDA.

Acceptance compares portable CPU attention, CUDA FlashAttention, and CUDA
Triton results for the configurations each backend supports. It exercises
mixed prefill/decode batches, page-table boundaries, sliding windows, GQA/MQA,
in-place KV updates, output aliasing, and repeated execution without cache
reallocation. Kernel capability mismatches are hard diagnostic errors.

#### Milestone 5: W4A16 as the first quantized execution vertical

W4A16 begins only after Milestones 2 through 4 establish model parameter
ownership, checkpoint loading, custom-kernel integration, attention state, and
CPU/CUDA dispatch.

Path:

```text
explicit W4A16 recipe and accepted checkpoint encodings
  -> canonical packed nibble, scale, zero-point, group, and transpose contract
  -> packed host Slice and persistent packed device Buffer
  -> direct checkpoint import without persistent dequantization
  -> independent CPU reference operator
  -> CUDA kernel through the established custom-call/Triton substrate
  -> model linear layers and attention projections
  -> numerical, memory-footprint, capability, and performance acceptance
```

The exact W4A16 recipe remains an explicit owner decision under section 5.3;
milestone ordering does not choose AWQ, GPTQ, group size, scale dtype, packing
order, or accumulation policy implicitly. Support is not claimed unless a real
checkpoint remains packed through loading and persistent device storage, CPU
and CUDA results meet the declared tolerance, unsupported hardware fails
clearly, and the measured memory footprint excludes a hidden full-precision
weight copy.

W8A8 and NVFP4 build on the same quantization, checkpoint, buffer, and kernel
boundaries. Their relative implementation order and exact recipes remain
separate owner decisions; neither is inferred merely from completing W4A16.

## 7. Subsystem decision ledger

This table is the place for the owner to make cuts. It intentionally starts
undecided except where earlier decisions force an answer.

| Subsystem | Reference paths | Value if retained | Consequence if omitted or replaced | State / owner decision |
| --- | --- | --- | --- | --- |
| Rust core language | NML; ZML Zig code is behavioral/design reference | Safe host runtime, graph frontend, quantization contracts, orchestration, and strong FFI boundaries | Reversing it would change the product architecture | `DECIDED` by D-008 |
| `rules_rust` integration | NML Bazel graph; official `rules_rust` | Hermetic Rust toolchain, libraries, tests, and procedural macros inside Bazel | A second build orchestrator would fragment native dependency ownership | `DECIDED` by D-009 |
| Zig `Shape`/`Tensor`/`Slice`/`Buffer`/`Exe` conceptual split | `zml/{shape,tensor,slice,buffer,exe}.zig` | Clear symbolic/host/device/compiled lifecycle | NML needs another lifecycle and ownership model | `DEFAULT-ZML`: preserve concepts in Rust |
| Semantic axis tags | `zml/shape.zig`, tensor/nn ops | Readable model code and shape safety | Less frontend machinery; more positional-axis risk | `DEFAULT-ZML` |
| Reflection-based `Bufferized(T)` and argument flattening | `zml/{mem,meta,module,exe}.zig` | Model structs flow through compile/load/run APIs | More explicit schemas or generated bindings required | `DEFAULT-ZML`: preserve behavior with Rust traits/derives |
| StableHLO frontend | `zml/tensor.zig`, `zml/ops.zig`, `mlir/dialects/stablehlo` | Broad optimized portable ops | NML needs another graph IR/codegen path | `DECIDED` by D-015 |
| XLA compiler | `third_party/xla`, `zml/module.zig` | CPU/CUDA optimization and established compiler | Replacing it would contradict D-015 | `DECIDED` by D-015 |
| PJRT runtime/plugin interface | `pjrt/`, `platforms/`, `zml/platform.zig` | Common device/buffer/executable API | Omitting it would contradict the selected runtime/build direction | `DECIDED`: Rust safe wrapper over PJRT; lift/adapt ZML integration by D-010 |
| Bazel/Bzlmod | root build files, `bazel/`, `MODULE.bazel` | Hermetic Rust/native dependency graph | Another top-level build would fragment ownership | `DECIDED` by D-009; `DEFAULT-ZML` where not overridden |
| External native build logic | `MODULE.bazel`, `bazel/`, `platforms/{cpu,cuda}`, `third_party/` | Known working PJRT/XLA/LLVM/CUDA acquisition, patching, and linking | Reimplementation adds risk without product differentiation | `DECIDED`: lift/adapt as much as useful by D-010 |
| Supported hosts | `platforms/BUILD.bazel`, CPU/CUDA packaging | Linux x86-64, Linux ARM64, macOS ARM64 | Windows, Intel macOS, and Metal remain outside scope | `DECIDED` by D-012; macOS CPU-only |
| CPU platform | `platforms/cpu` | Correctness/reference and performance backend | Cannot omit | `DECIDED` by D-013/D-014/D-020; `//platforms:cpu` defaults on; add Linux ARM64 CPU artifact |
| CUDA platform | `platforms/cuda` | Required accelerated backend | Cannot omit | `DECIDED` by D-014/D-016/D-020; `//platforms:cuda` defaults off independently; full retained ZML capability range, hard error otherwise |
| ROCm/TPU/Neuron/oneAPI/Metal platform support | corresponding `platforms/*` | No value to declared NML accelerator scope except as design reference | Removes their packaging, branching, kernels, tests, and topology cases | `OUT` as NML targets; reference code remains readable |
| General multi-device sharding | `zml/Sharding.zig` | Large-model and distributed experiment capability | Single-device simplicity; later distributed design required if needed | `DEFAULT-ZML` |
| Shardy/GSPMD integration | `zml/Sharding.zig`, `zml/module.zig` | Compiler-assisted partitioning | Explicit collectives or no distribution | `DEFAULT-ZML` |
| Ordinary dtype matrix | `zml/{dtype,floats,mlirx,pjrtx}.zig` | Broad format interop | Smaller branch/test matrix | `DECIDED`: D-003 canonical set and D-023 compiler-only index |
| W4A16 | partial names only in current ZML | Required memory-efficient inference path | Violates D-004 | `DECIDED: required`; exact recipe `UNDECIDED` |
| W8A8 | partial names/FP8 pieces in current ZML | Required quantized inference path | Violates D-004 | `DECIDED: required`; exact recipe `UNDECIDED` |
| NVFP4 | scalar plumbing but no end-to-end path | Required Blackwell-class quant path | Violates D-004 | `DECIDED: required`; exact scope `UNDECIDED` |
| Generic custom-call boundary | `zml/ops.zig`, `zml/kernel.zig` | Escape hatch for quant/attention kernels | Compiler graph limited to built-in ops | `DEFAULT-ZML` |
| Triton kernel builder | `kernels/triton`, `mlir/dialects/ttir` | In-language custom kernel authoring | Another CUDA kernel mechanism is needed for custom fast paths | `DEFAULT-ZML`: adapt Zig implementation to Rust |
| External CUDA FlashAttention | `platforms/cuda/flashattn`, `third_party/flashattn` | Optimized attention | Baseline or different optimized attention needed | `DEFAULT-ZML` within D-016 capability/error policy |
| Portable attention | `zml/attention/attention.zig`, `zml/nn.zig` | CPU oracle/fallback and non-causal variants | Attention becomes custom-kernel-only | `DEFAULT-ZML` plus CPU performance requirement |
| Paged attention/KV paging | `zml/attention/paged_attention.zig` | Serving/long-context memory management | Simpler contiguous state; weaker batching/long-context behavior | `DEFAULT-ZML` |
| Networked `attnd` | `zml/attention/attnd.zig` | Remote attention experiment | Removes networking from attention core | `DEFAULT-ZML` |
| MoE primitives/backend | `zml/moe`, Qwen MoE example | MoE model experiments | Dense-only until a later implementation | `DEFAULT-ZML` for CPU/CUDA-relevant behavior |
| Safetensors | `zml/safetensors.zig` | Common model container and metadata | Another importer/container is required | `DEFAULT-ZML` |
| `TensorStore` and reflection loader | `zml/io.zig` | Maps checkpoint names to model structs and loads in parallel | Model loaders become explicit/per-model | `DEFAULT-ZML` |
| Local file IO | `zml/io/vfs/file.zig` | Basic checkpoint loading | Another file abstraction required | `DEFAULT-ZML` |
| Hugging Face remote IO | `zml/io/vfs/hf.zig`, `tools/hf` | Direct hub model use | Users pre-download or use an external adapter | `DEFAULT-ZML` |
| S3/GCS/general VFS | `zml/io/vfs*` | Transparent remote repositories | Smaller core; explicit external data staging | `DEFAULT-ZML` |
| Tokenizers | `zml/tokenizer` | End-to-end text examples | Tokenization stays outside substrate | `DEFAULT-ZML` |
| Sampling helpers | `zml/nn.zig`, LLM sessions | End-to-end generation | Caller owns sampling graphs/host logic | `DEFAULT-ZML` |
| Model implementations | `examples/llm/models` | Acceptance workloads and reusable examples | Smaller tree; fewer real-model proofs | `DEFAULT-ZML` per CPU/CUDA-relevant model behavior |
| Full PJRT/XSpace profiler | `zml/profiling`, `tools/xspace_to_perfetto`, `upb` | Integrated CPU/device traces | External/simple profiling only | `DEFAULT-ZML` for CPU/CUDA/macOS-relevant behavior |
| NVTX annotations | `zml/profiling/tracer.zig` and C++ bridge | Nsight correlation | Profiling has fewer semantic regions | `DEFAULT-ZML` |
| `zml-smi` | `bin/zml-smi` | Hardware/process monitoring and metrics | Rely on vendor/system tools | `DEFAULT-ZML` restricted to supported platforms |
| `stdx` utility library | `stdx/` | Reusable flags/meta/format/debug/containers | Port only local utilities or use Rust ecosystem | `DEFAULT-ZML` semantics with Rust adaptation |
| Development/release/repository support surface | Nix/devenv, cross-compilation, OCI/tar, release/install, ZLS, buildifier/formatting, assets, contribution/docs structure | Contributor, release, and deployment conveniences | Build product capabilities directly; add individual surfaces only on request | `OUT` by D-019 unless explicitly requested; BuildBuddy is the sole explicit exception under D-027 |
| Prototype or throwaway phase | none; product tests live with their subsystem | No durable product value | Product development and durable verification begin immediately | `OUT` by D-011 |
| Upstream CI/editor/repository personalization | `.github`, `.nvim.lua`, `.vscode`, `.zed`, etc. | Upstream contributor workflow only | Add an NML-owned surface only when explicitly requested | `OUT` by D-005/D-019 |
| General autograd | not present in ZML | Conventional training API | Analytic graphs remain explicit | `OUT` by D-007 unless owner reverses decision |
| LLMD/production serving clone | not in public snapshot | Would be a separate product | NML remains an acceleration substrate | `OUT` by D-006 |

## 8. Dependency relationships to respect while deciding

Some decisions cannot be evaluated independently:

```text
XLA/PJRT
  -> MLIR/StableHLO wrappers
  -> protobuf/upb compile options
  -> CPU/CUDA plugin packaging
  -> likely favors retaining a capable native build system

quantized checkpoint support
  -> file metadata/importer
  -> packed storage and byte-stride semantics
  -> buffer loading without persistent dequantization
  -> CPU reference math
  -> CUDA capability and kernel
  -> model-level numerical and memory proof

multi-device support
  -> device topology/mesh decision
  -> shape or tensor placement annotations
  -> sharded buffer loading
  -> compiler partitioning or explicit collectives
  -> quantized packing/sharding rules

speculative decoding
  -> model/session composition
  -> mutable KV state
  -> checkpoint/rollback semantics
  -> deterministic token verification/sampling

analytic LoRA backward
  -> mutable parameters and optimizer state
  -> saved activation policy
  -> mixed-precision accumulation
  -> explicit backward ops
  -> adapter checkpoint format
```

An apparent cut at the top of one chain may require a replacement for every
downstream responsibility.

## 9. Definition of end-to-end quantization support

Every supported quantization recipe must cover the whole path:

1. Parse a declared checkpoint format and reject incompatible metadata.
2. Preserve packed weights in host storage with documented byte layout.
3. Transfer/store packed weights on the target without a persistent full-size
   FP16/BF16 duplicate.
4. Compile or select a kernel only on supported hardware.
5. Execute a correct CPU implementation and enforce CPU performance gates;
   feature-specific CUDA-only behavior requires an explicit recorded exception.
6. Execute the real CUDA kernel.
7. Compare layer output against a high-precision oracle with a declared metric
   and tolerance.
8. Run at least one real model end to end.
9. Measure actual device-resident memory, compile time, load time, prefill, and
   decode performance.
10. Produce a clear error for unsupported model, layout, dtype, or GPU cases.

All ten items are mandatory for a support claim. Exact models, numerical
tolerances, and benchmark thresholds are part of each quantization recipe's
recorded product contract. The checklist prevents enum-only support.

## 10. Remaining explicit departures from ZML

Rust, Bazel, `rules_rust`, PJRT integration, and the product-development posture
are settled. D-018 supplies the answer for ordinary architectural ambiguity:
follow ZML. The remaining decisions exist because NML has deliberately departed
from ZML's dtype and quantization scope:

1. What exact ordinary dtype list is public and end-to-end supported?
2. What is the first canonical W4A16 recipe and checkpoint encoding?
3. What is the first canonical W8A8 recipe?
4. What exact NVFP4 scope is first: weights/activations, GEMM only, MoE, KV
   cache, 1D/2D scaling, and which CUDA capability?

Other departures may be added only when a concrete NML requirement or an
established NML pattern justifies them; they are not standing questions.

## 11. Template for recording each future decision

```markdown
### D-NNN: Short title

- State: DECIDED | DEFAULT-ZML | DEFERRED | OUT
- Date:
- Owner:
- Requirement/workload:
- Chosen option:
- Alternatives considered:
- Why:
- Dependencies affected:
- Acceptance evidence:
- Reference snapshot and paths consulted:
- Revisit trigger:
```

## 12. Reference evidence map

Use these paths when revisiting decisions. They are evidence, not files to copy.

| Topic | Primary reference paths |
| --- | --- |
| Project claim and supported hardware | `references/zml/README.md` |
| Model lifecycle and core concepts | `references/zml/docs/learn/concepts.md` |
| Platform list and loading | `references/zml/platforms/platforms.zig`, `references/zml/zml/platform.zig` |
| Platform build flags | `references/zml/platforms/BUILD.bazel` |
| CPU/CUDA packaging and host artifacts | `references/zml/MODULE.bazel`, `references/zml/platforms/{cpu,cuda}`, `references/zml/platforms/cpu/cpu.bzl`, `references/zml/platforms/BUILD.bazel` |
| Dtypes and host float representations | `references/zml/zml/{dtype,floats}.zig` |
| MLIR/PJRT dtype conversion | `references/zml/zml/{mlirx,pjrtx}.zig`, `references/zml/pjrt/pjrt.zig` |
| Shape/tags/partition specs | `references/zml/zml/shape.zig` |
| Tensor/ops/nn surface | `references/zml/zml/{tensor,ops,nn}.zig` |
| Compilation path | `references/zml/zml/module.zig` |
| Execution and buffers | `references/zml/zml/{exe,buffer,slice,mem,meta}.zig` |
| Sharding/topology | `references/zml/zml/Sharding.zig` |
| MLIR wrappers | `references/zml/mlir` |
| Kernel builders | `references/zml/kernels/{common,triton,mosaic_tpu}` |
| Attention variants | `references/zml/zml/attention` |
| MoE and partial quant work | `references/zml/zml/moe` |
| Safetensors/model loading | `references/zml/zml/{safetensors,io}.zig` |
| VFS/remote storage | `references/zml/zml/io/vfs*` |
| Tokenizers | `references/zml/zml/tokenizer` |
| Profiling | `references/zml/zml/profiling`, `references/zml/tools/xspace_to_perfetto` |
| LLM inference/session examples | `references/zml/examples/llm` |
| Monitoring application | `references/zml/bin/zml-smi` |
| Dependency graph | `references/zml/MODULE.bazel`, `references/zml/third_party` |
| CI coverage | `references/zml/.github/workflows/ci.yaml` |
| Coding/repository guidance | `references/zml/AGENTS.md` |
| Rust/PJRT/MLIR/build interface guidance | Official PJRT, MLIR C API, XLA FFI, Rust FFI, and `rules_rust` links in section 5.22 |

## 13. North-star statement

NML is a Rust product built with Bazel and `rules_rust`, pinned hermetically to
the latest stable Rust release and never nightly. It uses StableHLO and XLA as
its compiler path and PJRT through a safe Rust runtime boundary. It deliberately
retains and adapts ZML's CPU/CUDA loaders, plugin packaging, and proven Bazel
logic for external native libraries. Development begins in the intended product
architecture; there is no prototype, spike, or throwaway verification phase.

Supported hosts are Linux x86-64, Linux ARM64/AArch64, and macOS ARM64/AArch64;
Windows and Intel macOS are out, and macOS is CPU-only. CPU is both the
correctness oracle and a performance backend. NVIDIA CUDA retains the complete
capability range exposed by the inherited ZML CUDA/XLA/PJRT stack, with hard
diagnostic errors for GPUs outside that range.

NML exists to make unusual model execution experiments possible on CPU and
NVIDIA CUDA. Ordinary FP32/FP16/BF16-class execution and real W4A16, W8A8, and
NVFP4 paths are complete loader-to-kernel capabilities, not enum claims. The
substrate remains general enough to express multiple modalities, speculative
decoding, and explicit analytic LoRA backward graphs without turning into a
serving product or a general autograd framework.

D-018 is the standing rule for every subsystem: follow ZML unless a clear NML
decision or established NML pattern says otherwise. The ledger records those
departures; it is not a catalog of questions created for their own sake.
