# NML system architecture

Status: authoritative system document

Last architectural review: 2026-07-16

NML is a Rust acceleration substrate for CPU and NVIDIA CUDA. It lets a product
describe tensor programs and persistent state in Rust, lowers those programs
through StableHLO and Shardy, compiles them with XLA, and executes them through
PJRT. The system is deliberately small at its public boundary while retaining
the compiler, ownership, sharding, checkpoint, and kernel machinery needed for
real model products.

This document describes NML as it exists and the invariants its future work
must preserve. [`TASKS.md`](./TASKS.md) is the implementation and capability
ledger. It records completed milestones, acceptance evidence, and deferred
hardware runs; it is not a second architecture specification.

## 1. Product boundary

NML owns:

- typed tensor-program construction;
- semantic shapes, layouts, and logical placement;
- StableHLO, Shardy, XLA, and PJRT integration;
- safe ownership of host tensors, device buffers, executables, and events;
- checkpoint declarations and persistent parameter loading;
- portable CPU/CUDA implementations and selected optimized CUDA kernels;
- explicit persistent state such as KV caches;
- capability dispatch and diagnostic failure on unsupported hardware;
- durable numerical, ownership, failure, and performance contracts.

NML is not a hosted serving product, a clone of LLMD, or a general eager tensor
runtime. Scheduling, request admission, continuous batching, network APIs, and
deployment policy belong to products built on NML. NML also does not contain a
general autograd engine. Training experiments may provide explicitly authored
forward and backward compiled programs over the same buffer/runtime substrate.

There is no prototype, spike, probe, smoke-test, or throwaway implementation
phase. New capabilities are added in their intended product architecture and
are accepted through permanent tests proportional to their claims. A parser,
enum case, emitted operation, registered symbol, or compiled kernel is not by
itself a supported capability.

## 2. System shape

The main execution path is:

```text
Rust model and tensor program
        |
        v
typed NML graph: Tensor + Shape + placement + state aliases
        |
        v
MLIR module: StableHLO + Shardy + selected custom calls
        |
        v
versioned StableHLO portable artifact and XLA compile options
        |
        v
CPU or CUDA PJRT plugin
        |
        v
PJRT LoadedExecutable over persistent and donated Buffers
```

The system separates five concepts that must not collapse into one another:

- `Tensor` is a symbolic value used while constructing a compiled program.
- `Shape` describes dtype, dimensions, semantic axes, layout, and partitions.
- `Slice` is shaped host storage or a shaped view over host storage.
- `Buffer` owns one or more device allocations and their placement.
- `Exe` owns a compiled executable and its argument/result contract.

`Bufferized<T>` and structural traversal connect Rust model structures to this
lifecycle. They do not turn checkpoint parser records, transfer guards,
individual PJRT shards, launch records, or MLIR objects into public concepts.

### 2.1 Package responsibilities

| Package | Responsibility |
| --- | --- |
| `crates/nml` | Compact public facade and product-facing composition. |
| `crates/nml-types` | Dtypes, bounded shapes, semantic axes, layouts, and partition metadata. |
| `crates/nml-tensor` | Typed/aligned host tensor storage and views. |
| `crates/nml-ir` | Symbolic tensor programs, validation, StableHLO/Shardy lowering, attention, and portable MoE. |
| `crates/nml-derive` | Auditable Rust structural traversal generated for model values. |
| `crates/nml-checkpoint` | SafeTensors discovery, declarations, aliases, tied weights, and loading. |
| `crates/nml-sharding` | Logical meshes, tensor placement, and Shardy-facing contracts. |
| `crates/nml-runtime` | Platforms, buffers, executables, argument binding, result ownership, and cache state. |
| `crates/nml-compiler` | StableHLO version negotiation and compilation orchestration. |
| `crates/nml-xla*` | XLA compile-options ownership and narrow raw bindings. |
| `crates/nml-mlir*` | Safe MLIR ownership and narrow raw bindings. |
| `crates/nml-pjrt*` | Common PJRT ownership plus distinct CPU and CUDA plugin loaders. |
| `crates/nml-platform` | Host/backend discovery and product platform assembly. |
| `crates/nml-kernel-triton` | Private typed TTIR construction and optimized CUDA kernels. |
| `crates/nml-kernel-flash-attention` | FlashAttention custom-call adapters and lifecycle registration. |
| `crates/nml-tokenizer*` | Safe IREE tokenizer ownership and its narrow C ABI. |
| `products/serve` | Qwen model execution and the serving control plane built above the substrate. |

Raw PJRT, MLIR, XLA, CUDA, and tokenizer ABIs remain internal. Unsafe code is
kept at narrow FFI and device-pointer boundaries, while safe owners encode
destruction order and retain the shared library/API state on which their
objects depend.

## 3. Language and public API

Rust is the core language. It owns graph construction, model structures,
checkpoint mapping, runtime orchestration, capability selection, and errors.
Native C, C++, CUDA, MLIR, TTIR, and vendor APIs remain where their ecosystems
require them; Rust does not attempt to rewrite XLA or performance kernels for
the sake of language uniformity.

The public root surface stays comparable in magnitude to ZML's useful core.
Its principal concepts are `DataType`, `Shape`, `Tensor`, `Slice`, `Buffer`,
`Exe`, `Bufferized<T>`, `Memory`, `Platform`, and `Sharding`. Backend launch
records, custom-call ABIs, MLIR owners, PJRT handles, checkpoint plans, and
kernel selectors are not root-level product types.

Rust traits and procedural derives replace the structural role of Zig
reflection. A derive may flatten tensor-bearing fields, construct bufferized
counterparts, map checkpoint names, and rebuild results, but the generated
behavior remains explicit and inspectable. Ordinary graph operations live on
the tensor store rather than creating a public class hierarchy for every
operation or backend.

Errors cross public boundaries as structured Rust results. Panics are reserved
for violated internal invariants that callers cannot cause. Unsupported device,
dtype, layout, topology, checkpoint, and kernel combinations fail before an
invalid launch whenever the information is available.

## 4. Types, shapes, and graph construction

### 4.1 Ordinary scalar types

The ordinary tensor element set is:

```text
bool
i8, i16, i32, i64
u8, u16, u32, u64
f16, bf16, f32, f64
c64, c128
```

`C64` means two 32-bit floating-point components; `C128` means two 64-bit
components. Complex values are retained for FFT/IFFT and complex-form model
operations. MLIR `index` is compiler-internal: it is not a `DataType`, host
storage type, checkpoint dtype, PJRT buffer type, or public tensor element.

Enum membership does not imply universal operation support. Each operation
states and validates its applicable dtype classes. Quantized storage is not
smuggled into this ordinary set as a misleading scalar name.

### 4.2 Quantized representations

Quantization separates:

```text
logical element and result dtype
compute and accumulation dtype
physical packed storage
quantization recipe
scale and zero-point tensors
packing order, alignment, grouping, and transpose rules
checkpoint encoding
kernel and hardware capability
```

W4A16, W8A8, and NVFP4 are product goals, but none is currently a supported
execution vertical. Their implementation remains deferred until explicitly
scheduled. Section 14 defines the acceptance boundary.

### 4.3 Shape invariants

Shapes have a maximum rank of eight. They carry checked dimensions, dtype,
optional semantic `AxisTag` values, physical layout, and partition metadata.
All element counts, byte sizes, offsets, strides, reshapes, and transposes use
checked arithmetic. Operations preserve tags and partitions when semantics are
preserved and require explicit remapping when they are not.

Semantic axes are part of model correctness. Attention head axes, feature axes,
batch axes, mesh consumption, and checkpoint layout must not be inferred from
incidental integer positions when a tagged contract exists.

### 4.4 Scoped graph construction

An ordinary Rust function executes while a scoped compilation context is
active. Tensor operations append verified IR operations rather than performing
eager numerical work. Data-dependent control flow is expressed with graph
operations such as StableHLO control flow; ordinary Rust branching is valid
only for build-time structure.

The frontend validates rank, dtype, layout, broadcasting, contraction,
sharding, alias, and result-shape contracts before producing malformed MLIR.
Its serialized module is deterministic for the same program and configuration.

## 5. Compiler and runtime

XLA is NML's compiler. The compiler graph is pinned through OpenXLA/XLA commit
`41370d1124c74d7b93a207136a636d8c631cbed9`; PJRT headers, XLA schemas,
LLVM/MLIR, StableHLO, and Shardy come from that coherent graph rather than
independently drifting pins.

NML emits owned, verified MLIR and negotiates the newest StableHLO portable
artifact version accepted by both NML and the loaded plugin. XLA compile
options carry replicas, partitions, device assignment, Shardy enablement, and
backend options. Invalid assignments and backend combinations are rejected
before the PJRT compile call.

The common PJRT layer owns C API discovery, clients, devices, memories,
buffers, events, executables, loaded executables, transfers, execution, and
status destruction. CPU and CUDA are separate loaders because their plugin
artifacts, initialization, runtime closures, and capability discovery differ.
They reuse the common safe object model; they are not one interchangeable
plugin implementation.

A loaded PJRT library remains resident for the process lifetime because the
API provides plugin initialization but no symmetric plugin shutdown. Every
dependent safe object retains sufficient API/library ownership to remain valid
independently of Rust lexical borrows.

PJRT GPU custom-call registration is foundational. NML retains typed and
untyped calls and the instantiate, prepare, initialize, and execute lifecycle.
Extension-chain traversal and status handling are safe Rust; converting PJRT's
untyped handler addresses remains an explicitly audited unsafe boundary.

## 6. Platforms and capability dispatch

### 6.1 Host and backend matrix

| Host | CPU | CUDA | Runtime evidence |
| --- | ---: | ---: | --- |
| Linux x86-64 | supported | supported | Four-device CPU and local SM75 CUDA executed. |
| Linux AArch64 | supported | supported | Selection and packaging contracts; native execution pending. |
| macOS AArch64 | supported | no | Host contract; native CPU execution pending. |
| Windows | no | no | Outside product scope. |
| Intel macOS | no | no | Outside product scope. |

CPU is both the correctness oracle and a performance backend. A CPU path may
not remain intentionally slow merely because CUDA has an optimized path.
macOS is CPU-only; Metal is not an NML backend. AMD ROCm, TPU, Neuron, oneAPI,
and Metal accelerator implementations are outside the product.

CPU and CUDA are independent Bazel build settings. CPU defaults on, CUDA
defaults off, and `--config=cuda` enables CUDA without disabling CPU. Platform
configuration selects product code and dependency closures; it does not claim
that the machine executing a Bazel action owns a GPU.

### 6.2 CUDA device policy

Portable XLA CUDA supports the retained runtime range beginning at SM75. A GPU
below the supported floor fails during platform creation with a diagnostic;
NML does not silently fall back to CPU. All devices in one CUDA PJRT client
must report the same compute capability because one compiled graph contains
one backend choice.

Optimized attention dispatch follows the kernel actually built:

| Device | Ordinary attention | Paged attention | Grouped MoE |
| --- | --- | --- | --- |
| SM75 | Portable XLA CUDA | Portable blockwise XLA CUDA | Portable XLA CUDA |
| SM80-SM89 | FA2 where geometry permits, otherwise portable | FA2 for compatible pages/positions, otherwise Triton or portable | Triton where supported, otherwise portable |
| SM90 | FA3 where geometry permits, otherwise portable | FA3 where supported, otherwise Triton or portable | Triton where supported, otherwise portable |

Explicitly requesting an incompatible optimized kernel is an error. Automatic
dispatch may select a semantically complete portable fallback. A statically
large index geometry that cannot enter an I32 kernel ABI stays on the portable
I64 graph instead of truncating values.

## 7. Sharding and distributed execution

NML uses Shardy as its only XLA SPMD partitioner. It does not expose a
GSPMD/Shardy selector, emit legacy `mhlo.sharding`, or maintain a second manual
sharding custom-call model.

One `Sharding` contract connects:

```text
logical mesh axes
  -> Shape partition metadata
  -> sdy.mesh and tensor shardings
  -> XLA replicas/partitions and device assignment
  -> checkpoint span dispatch
  -> PJRT argument placement and result assembly
```

Logical mesh axes use `AxisTag`, and tensor dimensions declare which mesh axes
they consume. Single-device, replicated, and partitioned execution use the
same model. Four-device CPU placement, compiler-inserted communication,
explicit collectives, tiled parameters, and expert sharding execute as real
numerical contracts. Multi-GPU CUDA execution is implemented at the shared
architecture boundary but remains unvalidated until suitable hardware is
scheduled.

## 8. Storage, buffers, and executable ownership

`Slice` supports owned or borrowed host storage, explicit shapes and layouts,
byte offsets and strides, subviews, negative strides, typed access, and dense
materialization. Alignment, extent, mutability, dtype, byte order, and address
arithmetic are checked.

`Buffer` owns persistent allocations in supported PJRT memory kinds. Parameters
are loaded once and reused by compiled programs. Transfers, readiness, device
placement, donation, output aliasing, deletion, and destruction remain
observable contracts. A failed execution or partial result must release every
owned PJRT object exactly once.

Mutable state is explicit. A program donates uniquely owned state buffers and
reinstalls their aliased outputs. Hidden host copies and incidental reference
counting are not mutation semantics.

Checkpoint loading uses bounded parallelism and memory-kind-aware staging. On
CUDA, mapped staging buffers support chunked DMA without forcing a persistent
full-size converted copy. Load reports distinguish unique tensors, aliases,
transferred bytes, staging, and completion.

## 9. Checkpoints, tokenization, and model products

SafeTensors is the current model container. NML validates metadata, shapes,
dtypes, aliases, tied weights, path containment, and byte extents before
transfer. Model declarations map local Hugging Face-style repositories and
`config.json` into structurally derived Rust values.

Remote acquisition is currently outside the runtime. Product inputs are local,
revision-pinned files with declared hashes. NML does not currently promise a
transparent Hugging Face, S3, or GCS virtual filesystem or an additional model
container.

Text tokenization uses IREE's tokenizer runtime at pinned commit
`4d4e97d00f099a21f38eeff26f82a6d9e3643a11`. A narrow C bridge owns IREE
status and buffer lifecycles; one safe Rust `Tokenizer` owns encoding,
decoding, incremental decoding, reset, partial consumption, and UTF-8 fragment
behavior. The tokenizer dependency is built from the original IREE repository
with audited local compatibility patches.

`products/serve` owns Qwen as NML's first model and serving product. Its current
Qwen3 engine implements BF16 checkpoint validation, tied embeddings, Q/K
normalization, RoPE, causal GQA, SwiGLU, persistent per-layer K/V state,
prompt-specific prefill, static single-token decode, greedy selection, and
incremental text decoding. The serving layer will add request scheduling,
global paged-cache ownership, streaming, tool-call protocol, and metrics
without exporting those concerns through the `nml` facade. The official
`Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca` is the permanent initial
end-to-end artifact.

The serving control plane uses Tokio for network, timer, cancellation, and
bounded-channel orchestration; Axum and Tower own the HTTP boundary. PJRT does
not execute opportunistically on Tokio workers. One dedicated engine owner
holds the platform, loaded parameters, executables, scheduler state, and cache
arena, and receives commands through bounded channels. This preserves device
ownership, prevents blocking XLA/PJRT calls from starving the async reactor,
and gives overload and shutdown deterministic boundaries.

## 10. Attention, caches, and custom kernels

### 10.1 Portable attention

Ordinary attention and blockwise paged attention have backend-independent
semantics. They support MHA, GQA, MQA, causal and noncausal masks, sliding
windows, RoPE, prefill, single-token decode, and multi-token decode.

Portable paged attention consumes physical K/V pages and a logical page table
directly. A bounded `stablehlo.while` traverses pages while carrying online
softmax state: running score maximum, rescaled exponential sum, and accumulated
value output. It does not materialize a persistent dense logical cache or the
complete attention-score matrix.

The portable graph is the CPU implementation, the independent backend oracle,
and the CUDA fallback. It is a product path, not a test-only reference.

### 10.2 Cache ownership

Dense and paged caches share one public `CacheSpec`/`Cache` contract. K/V
tensors, page tables, and sequence lengths are persistent `Buffer` values. The
host owns logical page assignments and lengths, so truncate, rollback, replay,
and speculative verification can change logical state without copying
unaffected K/V storage.

Used page IDs are validated before execution. Inactive trailing page-table
slots may be `-1`; portable lowering substitutes an in-range gather index and
masks every token from the inactive slot. Cache updates donate uniquely owned
K/V inputs and reinstall aliased outputs.

### 10.3 FlashAttention

NML builds original-upstream FlashAttention 2.8.3 rather than consuming a ZML
source fork. FA2 provides the SM80 cubin used across SM80-SM89; FA3 provides
SM90a. Adapters implement the PJRT custom-call ABI, lifecycle, validation,
output aliases, and failure propagation.

The PJRT CUDA product runtime remains CUDA 13.1. FlashAttention source is built
with its supported CUDA 12.8 compiler because source-compiler compatibility
and packaged runtime compatibility are independent contracts. This is not a
second runtime shipped to the product.

### 10.4 Triton

Triton is a private kernel-definition mechanism, not a public attention
backend API. NML owns a safe Rust builder over narrow TTIR bindings, typed named
arguments, explicit output shapes and aliases, verified TTIR, launch grids,
warps/stages, structured control flow, and deterministic errors.

The retained CUDA paths use Triton for unified paged attention and grouped
expert projections on Ampere and newer GPUs. The pinned XLA Triton compiler
rejects pre-Ampere devices, so SM75 uses portable XLA CUDA. Kernel source and
typed launch records still compile into every CUDA product graph for their
supported devices.

SM80/SM90 compilation, linking, TTIR, registration, and dispatch are verified.
Their numerical and performance execution remains explicitly deferred until
matching rented hardware is available. Compilation is required evidence, but
is never reported as device execution.

## 11. MoE and operation substrate

Portable MoE performs top-k routing, stable assignment construction, grouped
expert execution, weighting, and combination in StableHLO. Shardy owns expert
partitioning. Private Triton kernels specialize grouped expert projections on
SM80 and newer; CPU and SM75 use the portable graph.

The operation substrate presently includes:

- constants, scalar broadcasting, arithmetic, comparisons, selection, casts,
  bit operations, classification, rounding, and selected transcendental math;
- reshape, transpose, concatenation, static/dynamic slice, dynamic update,
  gather, scatter, and embedding lookup;
- general reductions, arg reductions, normalization, softmax, and log-sum-exp;
- matrix contractions and rank-polymorphic linear layers;
- convolution, grouped/depthwise convolution, pooling, and spatial resize;
- explicit-state uniform, normal, and Gumbel random generation;
- stable/unstable sort, argsort, top-k, greedy, and stochastic sampling;
- FFT/IFFT and complex construction/decomposition;
- StableHLO control flow and explicit collectives;
- portable attention, paged attention, KV state, and portable MoE;
- recurrent/state-space primitives including compiled Gated DeltaNet graphs.

[`TASKS.md`](./TASKS.md) remains authoritative for the exact acceptance state
of each family. New operations should be added for concrete workload needs and
composed from existing primitives where that preserves semantics and compiler
quality. Coverage does not justify exporting opcode hierarchies or backend
types.

## 12. Build and dependency architecture

Bazel/Bzlmod is the only top-level build graph. `rules_rust` supplies an exactly
pinned latest stable Rust compiler; nightly is prohibited. Cargo metadata may
describe Rust ecosystem dependencies for Bazel import, but Cargo does not own
PJRT, XLA, CUDA, cross-language linking, or product construction.

The repository builds native dependencies hermetically. Important invariants
include:

- original OpenXLA sources and their upstream-pinned compiler graph;
- separate CPU and CUDA PJRT plugin packages with complete runtime closures;
- hermetic Clang and GCC 13.4 libstdc++ headers/static runtime for CUDA source;
- original-upstream FlashAttention and IREE sources plus narrow audited local
  patches;
- exact source/runtime toolchain pins and SHA-256-pinned binary archives;
- no dependency on `references/zml` or any ZML Bazel target.

Two ZML-derived inputs remain deliberate and traceable: the local
`cuda-root-path-local-defines.patch` adapted against original OpenXLA, and
SHA-256-pinned CPU/CUDA plugin archives hosted at `zml/pjrt-artifacts`. The
plugin archives are a packaging dependency, not a ZML source-fork dependency.
Any new ZML-hosted source dependency requires an explicit architecture review.

Local Bazel state uses one sibling cache. The relative path is normative and
is already encoded in `.bazelrc`:

```text
../nml-bazel-cache
```

The equivalent explicit spelling places the startup option before the command:

```sh
bazel --output_user_root=../nml-bazel-cache test --config=cpu //:cpu_contracts
```

BuildBuddy is opt-in and stores no credential in the repository. The `bb` CLI
supplies authentication. Remote workers use the repository-pinned Ubuntu 20.04
execution image and a bounded 80-action concurrency. Multi-gigabyte NVIDIA ELF
rewrite actions run locally with `no-remote` while their expensive producers
remain remotely built and cached; this avoids transferring huge files for a
small deterministic `patchelf` action.

Build and target platforms are distinct. A CUDA target platform describes the
binary and its ABI. An execution platform describes the machine running an
action. NML does not label a GPU-less remote worker as CUDA hardware merely to
make a test analyzable.

## 13. Verification topology

Verification is divided by the machine that can truthfully own the resource:

```text
//:cpu_contracts
    CPU numerical, ownership, sharding, collective, and performance contracts

//:cuda_remote_contracts
    CUDA-configured compiler/runtime-structure/failure contracts needing no GPU

//:cuda_contract_binaries
    exact GPU executable build outputs populated into the authenticated cache

//:cuda_package_contracts
    hermetic distribution and system-driver runtime closure integrity

//:cuda_device_contracts
    real NVIDIA execution, local or rented, unsandboxed and never result-cached
```

The standard commands are:

```sh
bb test --config=buildbuddy --config=cpu //:cpu_contracts
bb test --config=buildbuddy --config=cuda //:cuda_remote_contracts
bb build --config=buildbuddy --config=cuda //:cuda_contract_binaries
bb test --config=buildbuddy --config=cuda //:cuda_package_contracts
bb test --config=buildbuddy --config=cuda --cache_test_results=no \
  //:cuda_device_contracts
```

Remote CUDA compilation is not CUDA execution evidence. Device tests are
`exclusive`, unsandboxed, and non-cacheable because the installed GPU, driver,
and `/dev/nvidia*` state are external singleton resources. Hosted workflows do
not schedule them. Package tests, by contrast, are hermetic file-closure tests
and belong on BuildBuddy.

Performance contracts report compilation, parameter upload, first execution,
steady execution, and download separately. CPU and CUDA performance statements
must identify build mode, workload, device, and phase rather than folding
compiler or transfer time into a misleading single number.

## 14. Capability boundary and forward work

The source-guided substrate phase is complete: NML has its own coherent
CPU/CUDA graph, compiler, runtime, checkpoint, sharding, attention, MoE,
tokenizer, and real-model architecture. ZML remains valuable reference
material, but NML's requirements and this document now govern new work.

The remaining validation debt is explicit:

- execute FA2, FA3, Triton paged attention, and grouped Triton MoE on rented
  SM80/SM90 hardware;
- execute and measure multi-GPU CUDA Shardy placement and collectives;
- run native Linux AArch64 CPU/CUDA contracts, including DGX Spark;
- run native Apple Silicon CPU contracts.

The main new product territory is:

- W4A16, W8A8, and NVFP4 execution verticals;
- explicitly authored analytic backward graphs and LoRA optimizer/state flows;
- additional model families and modalities selected by concrete products;
- orchestration such as speculative decoding built over existing cache and
  multi-executable ownership.

### 14.1 Quantization support definition

A quantization recipe is supported only when all of the following hold:

1. A declared checkpoint encoding and metadata contract are parsed and
   incompatible inputs are rejected.
2. Packed weights remain packed in host storage with a documented byte layout.
3. Persistent device storage does not retain a hidden full-size FP16/BF16 copy.
4. Scale, zero-point, grouping, transpose, and accumulation semantics are
   explicit.
5. A correct CPU implementation executes and meets its declared performance
   requirement, unless a feature has a recorded CUDA-only exception.
6. The real CUDA kernel executes only on supported hardware.
7. Layer output is compared with an independent high-precision oracle using a
   declared metric and tolerance.
8. At least one real model executes end to end.
9. Device memory, load time, compilation, prefill, decode, and steady execution
   are measured separately.
10. Unsupported model, layout, dtype, recipe, and GPU combinations produce
    diagnostic errors.

W4A16 still requires a concrete choice of signedness, calibration/checkpoint
convention, group axis and size, scale/zero-point dtype, nibble order,
prepacked transpose layout, activation dtype, and accumulation behavior. W8A8
must separately choose integer or FP8 values, static or dynamic activation
quantization, scale granularity, accumulation, and requantization. NVFP4 must
declare E2M1 packing, local E4M3 scales over 16 elements, the global FP32 scale,
1D/2D scaling, checkpoint encoding, Blackwell capability, and whether KV-cache
quantization is a distinct feature.

### 14.2 Workload-preserving requirements

The substrate must remain usable beyond one autoregressive model:

- multimodal products need convolution, resizing, variable semantic axes,
  cross/bidirectional attention, FFT/signal operations, and explicit
  preprocessing boundaries;
- speculative decoding needs multiple executables, deterministic sampling,
  block verification, and KV checkpoint/truncate/rollback/replay semantics;
- analytic LoRA needs mutable adapter/optimizer buffers, explicit forward and
  backward graphs, mixed-precision accumulation, saved-activation policy,
  outer products/reductions, and adapter-only checkpointing.

These are design constraints, not standing permission to add speculative APIs.
Each new surface is justified by a concrete product and permanent acceptance
workload.

## 15. Stable decision index

The identifiers below are retained because source comments and the historical
ledger cite them. The present-tense sections above are authoritative; this
table is a compact compatibility index, not a migration checklist.

| ID | Stable decision |
| --- | --- |
| D-001 | NML is independently implemented product code informed by readable references; dependency/build reuse is separately audited. |
| D-002 | CPU and NVIDIA CUDA are the only accelerator backends. |
| D-003 | The ordinary dtype set is bool, 8/16/32/64-bit signed and unsigned integers, F16/BF16/F32/F64, and C64/C128. |
| D-004 | W4A16, W8A8, and NVFP4 are first-class future product goals, not enum-only claims. |
| D-005 | Upstream editor, CI, and repository-personalization files are not inherited. |
| D-006 | NML is an acceleration substrate, not LLMD or a hosted serving clone. |
| D-007 | General autograd is outside scope; analytic backward programs may be authored explicitly. |
| D-008 | Rust owns the safe host, graph, model, and orchestration layers. |
| D-009 | Bazel/Bzlmod with `rules_rust` is the single top-level build. |
| D-010 | Proven PJRT/native Bazel integration may be deliberately adapted with provenance. |
| D-011 | Product architecture and permanent verification begin immediately; throwaway phases are prohibited. |
| D-012 | Supported hosts are Linux x86-64, Linux AArch64, and macOS AArch64; Windows and Intel macOS are out. |
| D-013 | CPU is both correctness/reference and performance backend. |
| D-014 | CPU/CUDA PJRT loading and complete plugin packaging are retained and Rust-owned. |
| D-015 | StableHLO/XLA/PJRT is the compiler/runtime path. |
| D-016 | The retained CUDA range is supported with hard errors outside it; feature-specific floors remain explicit. |
| D-017 | `rules_rust` uses an exactly pinned latest stable Rust compiler, never nightly. |
| D-018 | Historical default was to follow ZML absent a reason to depart; NML's established architecture and this document now govern. |
| D-019 | Unrequested release, deployment, editor, and repository-support surface remains outside scope; BuildBuddy is the explicit exception. |
| D-020 | Bazel owns independent additive CPU and CUDA settings. |
| D-021 | Local Bazel state reuses the sibling `../nml-bazel-cache`. |
| D-022 | PJRT GPU custom-call registration and its complete lifecycle are foundational. |
| D-023 | MLIR `index` is compiler-only and never a runtime dtype. |
| D-024 | The coherent compiler graph is pinned at OpenXLA commit `41370d1124c74d7b93a207136a636d8c631cbed9`. |
| D-025 | Original upstream sources are preferred; ZML-hosted source forks require explicit review. |
| D-026 | Attention and complete selected CPU/CUDA substrate coverage precede novel quantization. |
| D-027 | BuildBuddy is opt-in, credential-free in-repo, hermetic, image-pinned remote execution/cache. |
| D-028 | Quantization work remains deferred until explicitly scheduled. |
| D-029 | Shardy is the only SPMD partitioner; legacy GSPMD is out. |
| D-030 | Placement metadata is part of the graph/checkpoint/runtime contract, not a later retrofit. |
| D-031 | Compile, package, CPU, and real-GPU contracts run where their resources truthfully exist. |
| D-032 | Portable blockwise paged attention is a CPU product path and CUDA fallback. |
| D-033 | Dense and paged KV state share one explicit persistent `CacheSpec`/`Cache` ownership model. |
| D-034 | Dispatch follows actual kernel capability: portable SM75, Triton SM8x/SM90, FA2 SM80-SM89, and FA3 SM90. |
| D-035 | CUDA source compilation uses hermetic Clang and a coherent hermetic GCC libstdc++ static runtime. |
| D-036 | FlashAttention's CUDA 12.8 source compiler and the CUDA 13.1 PJRT runtime are separate compatible contracts. |
| D-037 | Bazel target platforms describe products; execution platforms describe action machines. |
| D-038 | SM80/SM90 optimized kernels remain mandatory build inputs; execution evidence is deferred, never fabricated. |
| D-039 | IREE tokenization and dense Qwen3 BF16 generation are product capabilities over local pinned model artifacts. |
| D-040 | `products/serve` owns the Qwen serving control plane; Tokio tasks communicate through bounded channels with one dedicated PJRT engine owner. |

## 16. Provenance and relationship to ZML

NML is an opinionated, source-informed fork of
[ZML](https://github.com/zml/zml). The read-only snapshot at
`references/zml`, commit `ed2bd190e8dd1e47bda2819015f443874b68243a`, was
instrumental in identifying useful architecture, behavior, and build patterns.
ZML is Apache-2.0 licensed; reused or adapted infrastructure must remain
traceable and retain applicable notices.

`references/zml` is never:

- an NML Bazel input;
- an implicit compatibility specification;
- a source of mechanically renamed runtime/model implementation;
- a reason to carry excluded accelerators, dtypes, or product surface.

NML now owns its architecture. ZML remains a mature open-source project that we
continue to study and learn from, but new NML work is selected by NML product
requirements, the invariants in this document, and evidence in `TASKS.md`.

## 17. North star

NML makes unusual compiled model systems practical without turning the
substrate into a serving platform or a general training framework. Rust owns a
small, safe product API; Bazel owns one hermetic cross-language graph;
StableHLO, Shardy, XLA, and PJRT own portable compilation and execution; CPU is
a first-class correctness and performance backend; CUDA adds capability-checked
specialized kernels without changing model meaning.

Every claimed feature extends from checkpoint representation through persistent
storage, compiled semantics, runtime ownership, real execution, diagnostics,
and measured performance. That complete vertical—not API breadth or a compiled
artifact—is the unit of progress.
