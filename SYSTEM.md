# NML system architecture

Status: authoritative system document

Last architectural review: 2026-07-18

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

The system separates six concepts that must not collapse into one another:

- `Tensor` is a symbolic value used while constructing a compiled program.
- `Parameter` is an immutable logical model value with one closed physical
  representation and one or more named components.
- `Shape` describes dtype, dimensions, semantic axes, layout, and partitions.
- `Slice` is shaped host storage or a shaped view over host storage.
- `Buffer` owns one or more device allocations and their placement.
- `Exe` owns a compiled executable and its argument/result contract.

One parameter-tree derive connects Rust model structures to loaded parameter
trees. It maps each logical `Parameter` leaf to one `LoadedParameter`; dense
storage is the one-component case and structured representations retain all
components behind the same leaf. Checkpoint records, transfer guards,
individual PJRT shards, launch records, and MLIR objects remain private.

### 2.1 Package responsibilities

| Package | Responsibility |
| --- | --- |
| `crates/nml` | Compact public facade and product-facing composition. |
| `crates/nml-types` | Dtypes, bounded shapes, semantic axes, layouts, and partition metadata. |
| `crates/nml-parameter` | Logical parameter identity, closed representation specifications, and physical-component contracts. |
| `crates/nml-tensor` | Typed/aligned host tensor storage and views. |
| `crates/nml-ir` | Symbolic tensor programs, validation, StableHLO/Shardy lowering, attention, and portable MoE. |
| `crates/nml-derive` | Auditable Rust structural traversal generated for model values. |
| `crates/nml-checkpoint` | Physical artifact indexing plus parameter declaration, binding, aliases, tied weights, and component loading. It never owns graph operations. |
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
| `products/serve` | GPT-OSS artifact, protocol, model execution, and the serving control plane built above the substrate. |

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

NML is pre-alpha and owes no backward compatibility. An API or internal model
that obstructs a simpler, safer, or more scalable design is replaced directly;
we do not accumulate deprecated aliases, migration adapters, or parallel
legacy/new subsystems. Compatibility begins only when a later release contract
explicitly says so. This freedom does not weaken verification: every redesign
must preserve or replace the applicable permanent product evidence.

The public root surface stays comparable in magnitude to ZML's useful core.
Its principal concepts are `DataType`, `Shape`, `Tensor`, `Parameter`, `Graph`,
`Slice`, `Buffer`, `Exe`, `Memory`, `Platform`, and `Sharding`.
`Parameter` is the immutable logical model-weight concept justified by dense
and structured quantized storage; backend representations remain private.
Backend launch records, custom-call ABIs, MLIR owners, PJRT handles, checkpoint
plans, physical encoding records, and kernel selectors are not root-level
product types.

Rust traits and procedural derives replace the structural role of Zig
reflection. One parameter-tree derive flattens `Parameter`-bearing fields,
constructs loaded counterparts, and rebuilds nested results; generated behavior
remains explicit and inspectable. It never maps arbitrary graph `Tensor` values
to `Buffer`s. Ordinary graph operations live on `Graph`, independently of
artifact indexing, parameter declaration, loading, and preparation.

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

NVFP4 is the current first-priority execution vertical, defined in
[`NVFP4.md`](./NVFP4.md). It is represented as one logical `Parameter` backed
by packed payload, scale, and global-factor components, not as an ordinary
dtype or an arbitrary tuple of tensors. The representation must remain packed
through loading and device residency and lower through capability-selected CPU,
pre-Blackwell emulation, or native Blackwell kernels.

Its canonical recipe-v3 storage is operation-shaped. Logical contraction
weights remain `[N, K]`, `[E, 2I, K]`, and `[E, H, I]`, while payload and
block-scale components store encoded K before contiguous N:
`[packed K, N]`, `[E, packed K, 2I]`, and `[E, packed I, H]`. Indexed
embeddings alone remain rowwise because lookup selects complete vocabulary
rows. Source expert tensors are transposed into logical `[E, N, K]` before
quantization, then encoded contraction components swap their final two axes.
CPU, SM75, and Triton consume these operation-shaped components directly;
runtime repacking, a second prepared copy, and earlier-recipe compatibility
are not part of the architecture.

W4A16 and W8A8 remain later independent product goals. Implementing NVFP4 does
not select their signedness, calibration, grouping, scale, or checkpoint
contracts. Section 14 defines the common acceptance boundary.

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
| Linux x86-64 | supported | supported | Four-device CPU, local SM75, rented SM86, and rented SM90 executed. |
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
| SM80-SM89 | FA2 where geometry permits, otherwise portable | FA2 for its exact ABI, otherwise Triton | Triton where supported, otherwise portable |
| SM90 | FA3 where geometry permits, otherwise portable | FA3 for its exact ABI, otherwise Triton | Triton where supported, otherwise portable |
| SM91+ | Portable until an ordinary-attention adapter is retained | Triton | Triton where supported, otherwise portable |

Explicitly requesting an incompatible optimized kernel is an error. Automatic
dispatch chooses exactly one implementation from semantic features, physical
representation, geometry, and device capability. An optimized-lowering error
is never caught and retried as portable execution. Portable lowering is chosen
only when the graph lies outside a retained accelerator contract; for example,
a statically large index geometry that cannot enter an I32 kernel ABI stays on
the portable I64 graph instead of truncating values.

GPT-OSS authors paged attention even while the single-request engine owns one
contiguous donated cache buffer. The graph views that storage as 16-token
physical pages behind an identity page table and supplies the logical sequence
length, so its supported CUDA geometry always selects FA2, FA3, or Triton
rather than an ordinary-attention runtime branch over unused cache capacity.
Physical page width and kernel tile width are separate contracts: CUDA decode
tiles are capped at 64 even when another product chooses coarser pages, so
allocation policy cannot silently create a register-spilling attention kernel.
Continuous batching can replace the page table and cache owner without
changing model semantics.

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
behavior. Ordinary-text encoding explicitly disables special-token matching so
protocol delimiters can only be introduced from validated structural IDs. The
tokenizer dependency is built from the original IREE repository with audited
local compatibility patches.

`products/serve` owns GPT-OSS model execution and serving policy. Its
package-private `gpt_oss` owner contains the exact artifact contract,
configuration, checkpoint schema, Harmony protocol, bounded component graphs,
and request execution. The public product surface is one persistent
`Generator`: loading validates the artifact, constructs the tokenizer, compiles
every configured execution profile while the checkpoint is metadata-only, and
only then uploads parameters once. The resulting immutable plan and resident
parameters persist across requests; generation owns only one request's tokens,
parser, positions, page metadata, and K/V storage.

The GPT-OSS lifecycle is a strict product-owned type-state transition:
`ModelDefinition -> CompiledDefinition -> ResidentModel -> RequestState`.
Compilation profiles declare maximum prompt and total-sequence capacities and
normalize to the finite prefill/cache families described below. Equivalent
families compile once. A request selects the smallest resident profile that
covers it and hard-fails when no profile fits; request execution never invokes
XLA. This follows ZML's compile-before-buffer-load ordering and leaves device
memory available to compiler autotuning without weakening NML's bounded
multi-profile serving contract.

Framework crates do not contain a model adapter, model registry, protocol, or
checkpoint-family taxonomy. They expose semantic tensor operations, closed
parameter representations, compilation, reusable executable slots, buffers,
and asynchronous PJRT dependency chaining. A model product composes those
mechanisms directly. This direction of dependency is strict: framework crates
must never import `products`, artifact identities, Harmony, or GPT-OSS layer
types. Product-specific math becomes a framework operation only when it has a
model-independent semantic name and complete portable/specialized contracts.

The versioned GPT-OSS protocol identity is
`openai-harmony-gpt-oss-v1` over `o200k_harmony`. Its package-private owner
validates the exact required token spellings and IDs when it opens the local
artifact's `tokenizer.json`; renders system, developer, user, assistant,
function-call, and function-result messages; and parses assistant output one
token at a time into UTF-8-safe channel, text, tool-call, and terminal events.
Structural IDs are appended directly and all caller-controlled content uses
ordinary-text encoding. Invalid roles, channels, content types, JSON calls,
stop transitions, trailing tokens, or incomplete output fail deterministically.
The protocol returns a parsed tool request but never executes it.

NML deliberately reimplements this narrow product boundary instead of
depending on the `openai-harmony` crate. The official crate at reference commit
`abd677f7ac962629c808197caa1feb9e3e95d2b0` owns a tiktoken `CoreBPE`, supports
runtime tokenizer acquisition/caching, unconditionally brings broader CLI,
HTTP, and image dependencies, and also carries optional Python/Wasm bindings.
Those concerns conflict with NML's one IREE tokenizer, local artifact, hermetic
OCI, and Bazel-owned runtime. NML retains byte-compatible decoded fixtures and
exact streaming token fixtures from that Apache-2.0 reference; the adapted
JSON-Schema-to-TypeScript rendering remains traceable in source.

GPT-OSS 20B is the selected serving model. NML does not infer support
for its architecture or checkpoint representation from a model-family name.
The first GPT-OSS product vertical uses one exact NVFP4 artifact selected and
pinned by distributor, revision, file hashes, configuration, tensor records,
packing, scales, layout, and conversion provenance. The official MXFP4 release
does not create an NML requirement to implement MXFP4. Unknown or ambiguous
GPT-OSS checkpoint variants are rejected rather than interpreted
heuristically. [`NVFP4.md`](./NVFP4.md) owns the complete representation,
kernel, hardware, and acceptance architecture.

The GPT-OSS model vertical reuses the existing RMSNorm, GQA, RoPE/YaRN,
dense and sliding-window attention, paged cache, top-k MoE, grouped expert,
Shardy, sampling, and runtime substrate. It implements the selected artifact's
exact configuration and tensor mapping,
attention-sink denominator bias, clamped/residual SwiGLU semantics, alternating
dense/window attention schedule, `o200k_harmony` tokenization behavior, Harmony
roles/channels, and end-to-end output contract. The private adapter
authenticates the small, byte-exact artifact manifest and validates its
ingestion-issued materialization receipt before opening the manifest-selected
SafeTensors index. The expensive content hash of every payload is an ingestion
operation, never a model-start operation. Successful materialization makes the
manifest and payloads read-only and atomically records each file's device,
inode, size, mode, modification time, and change time. Startup compares that
bounded receipt with the authenticated manifest and current filesystem
identities; a missing or stale receipt is a hard error that requires
rematerialization, not a fallback full scan. This protects the deployment
lifecycle from incomplete or accidentally mutated artifacts. It assumes the
artifact host administrator is trusted: an actor who can replace both payload
and receipt is outside the local receipt's threat model and requires an
externally signed or filesystem-verified deployment boundary.

The adapter declares 411 logical parameters over 703 compact physical
components. Each finite prefill shape family contains four bounded executables:
embedding, one reusable sliding-attention layer, one reusable full-attention
layer, and the final normalization/output head. Each decode family instead
contains three: embedding, one reusable alternating sliding/full layer pair,
and the head. Structural parameter-slot binding applies the appropriate loaded
layer or adjacent layer pair only after verifying logical shape,
representation, component roles, physical storage, platform, and placement.

Execution enqueues embedding, the 24 prefill layer invocations or 12 decode
pair invocations, and the head through PJRT readiness dependencies,
synchronizing the host only when the selected token is needed. Hidden state and
every per-layer K/V pair use explicit output aliasing; one request-owned I32
page table describes all layer caches. Prefill and cache capacities use finite
power-of-two/page buckets so compiled families are reusable across requests
without making XLA compile a monolithic 24-layer module. This lifecycle does
not claim continuous batching or cross-request cache sharing; those require the
separate server-owned arena and scheduler.

Loaded executables resolve immutable result arity once at compilation. A hot
enqueue must not reacquire an executable metadata handle merely to rediscover
its output count. Product timing separates end-to-end device execution from
host submission at the embedding, sliding-layer, full-layer, and head
boundaries; submission counters never add synchronization and are not labeled
as GPU kernel timings.

Readable full-checkpoint generation and independent-oracle acceptance are two
explicit executions of the same product contract. Both require the immutable
model and enforce structural, memory, cache, event, and timing invariants. The
acceptance target additionally requires an independently produced token/event
fixture; absence of that fixture can never silently downgrade acceptance into
generation. A BF16 artifact adds no new
quantization format but carries its full memory cost; it may be used as a
bounded independent oracle or conversion source when explicitly selected, but
it is no longer a prerequisite product milestone. The NVFP4 artifact is
admitted only through the complete quantization contract in section 14.1.

The serving layer adds request scheduling, global paged-cache ownership,
streaming, tool-call protocol, and metrics without exporting those concerns
through the `nml` facade. Its engine boundary is model-neutral: a model owns
checkpoint validation, prompt protocol, graph construction, token semantics,
and model-specific sharding, while the server owns admission, batching, page
leases, transport, cancellation, and observability.

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

The builder's immutable result carries the canonical TTIR, public function
name, and exact ordered argument kinds as one artifact. `KernelSpec` binds
tensor shapes to that authored ABI and rejects differences in argument count,
order, global-pointer address space, or element type before emitting
StableHLO. Raw TTIR strings and separately reconstructed signatures never
cross the custom-call boundary. Split-K paged attention keeps its producer
sink-free and supplies learned sinks only to the global segment reduction,
where their softmax correction is applied exactly once.

Paged-attention page addressing is lane-wise and therefore permits a compute
tile to cross physical page boundaries. Launch selection uses that property to
bound decode tile width independently of cache allocation granularity. Product
cache pages remain small for memory utilization; the framework cap prevents a
different product's large pages from becoming unbounded per-program register
state.

The retained CUDA paths use Triton for unified paged attention and grouped
expert projections on Ampere and newer GPUs. The pinned XLA Triton compiler
rejects pre-Ampere devices, so SM75 uses portable XLA CUDA. Kernel source and
typed launch records still compile into every CUDA product graph for their
supported devices.

SM80/SM90 compilation, linking, TTIR, registration, and dispatch are verified.
Real SM86 and SM90 acceptance runs additionally execute FA2/FA3 ordinary
attention, unified paged attention, and grouped expert projections through the
retained FlashAttention and Triton paths. The phase-separated performance
contract executes GPT-OSS-sized compact embedding, decode, prefill, and grouped
Triton MoE on SM86 and SM90, with a corresponding SM75 custom-call baseline.
Dedicated optimized-attention performance/tuning and homogeneous multi-GPU
execution remain explicit hardware debt. Compilation is required evidence, but
is never reported as device execution.

## 11. MoE and operation substrate

Portable MoE performs top-k routing, stable assignment construction, grouped
expert execution, weighting, and combination in StableHLO. Shardy owns expert
partitioning. Private Triton kernels specialize grouped expert projections on
SM80 and newer; CPU and SM75 use the portable graph.

Static schedule capacity is not executable work. Sparse decode uses one expert
block per selected route when `assignments * 4 <= experts`, matching ZML's
direct assignment crossover. General routing retains an aligned expert
schedule plus an explicit active-block scalar. Every grouped CUDA kernel tests
that scalar and expert locality before forming a weight address, decoding a
scale, or entering a contraction; an inactive/non-local block contributes zero
for the later expert-parallel reduction. Clamping an invalid expert to expert
zero is prohibited because a masked activation does not prevent weight
traffic. Decode and larger-batch tiles use ZML's finite, source-owned threshold
family, including eight-warps above 128 tokens on all retained Triton-capable
NVIDIA generations, rather than request-specific heuristics.

Every expert backend shares one semantic projection boundary. Gate/up performs
both paired contractions, owns its bias and activation exactly once, and emits
`[assignments, intermediate]`; down consumes that activated tensor and owns
only its bias and route weighting. This prevents an output-tiled down kernel
from recomputing the same nonlinearity and keeps dense, compact CPU, SM75, and
SM80+ lowering interchangeable at the graph boundary. Private compact CUDA
lowering selects a dedicated F32-accumulating GEMV for `M = 1`; prefill retains
the finite tensor-core matrix family. Weight/scale decode is register-local and
exact, and one scale load is reused across its complete representation block.

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
- no dependency on `references/zml`, `references/harmony`, or their Bazel/Cargo
  targets.

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

### 12.1 OCI product artifacts

OCI images are NML's common Linux CUDA execution envelope for local NVIDIA
hosts and rented RunPod hosts. They standardize the NML executable, runfiles,
CUDA/PJRT user-space closure, entrypoint, and selected contract set. They do
not standardize the kernel driver, physical GPU, compute capability, or device
state, so an image build or pull is never GPU execution evidence.

NML produces three distinct CUDA artifacts over the same versioned CUDA/PJRT
runtime contract: the production-serving image, a substrate-contract image,
and a GPT-OSS product-contract image. Production images contain no test
binaries. The substrate image contains no product code or product input names.
The product-contract image owns only GPT-OSS acceptance executables and their
closed mounted-input contract. Each image
groups its slowly changing runfiles closure separately from its executable and
configuration. That grouping makes an application-only edit a small layer
change without claiming that the serving and contract closures are identical
registry blobs. The image graph uses digest-pinned bases and stable layer order
so unrelated multi-gigabyte content is reusable across revisions. Linux x86-64
and Linux AArch64 are distinct manifests and may be combined under one
multi-platform index. macOS remains a native CPU target, not an OCI CUDA
target.

Model weights are never ordinary OCI image layers. Local execution mounts an
exact local artifact; RunPod either downloads the same revision into ephemeral
storage or mounts an explicitly selected persistent model-cache volume. The
download/materialization workflow fully hashes the pinned manifest and every
declared payload once, removes write permissions, and atomically publishes the
local materialization receipt. The container receives a fixed model path,
authenticates the small pinned manifest, and verifies that bounded receipt
before loading it. It never rehashes checkpoint payloads during ordinary
startup. Registry references used for execution are exact digests. Mutable
tags may be human-facing aliases but never acceptance inputs.

Checkpoint conversion is a CPU-only artifact-production job. It runs on a
BuildBuddy remote runner with explicit CPU, disk, timeout, and redacted secret
properties, then publishes directly to the selected artifact repository. It
never rents a GPU, executes on the operator host, or sends model payloads back
through the operator connection. CUDA venues begin only at immutable runtime
evidence. [`tools/publish-nvfp4-artifact.sh`](./tools/publish-nvfp4-artifact.sh)
is the canonical invocation; a disconnected terminal reconnects to its printed
invocation with `bb view` and never starts a duplicate publisher speculatively.

`rules_img` is NML's only OCI construction rule set. Its provider-oriented
image graph, shallow base pulls, compact layer representation, Rust runfiles
support, multi-platform transitions, and incremental load/push behavior fit
BuildBuddy and NML's multi-gigabyte CUDA closure. `rules_oci` is not introduced
as an evaluation or compatibility graph. The permanent targets cover Bazel 9,
BuildBuddy construction, deterministic layout, digest output, Linux x86-64 and
AArch64 images, local loading, registry pushing, and incremental transfer.

BuildBuddy builds and caches the expensive image closure. Local Docker loading
is eager: `bb` downloads only the completed compressed OCI layers through its
authenticated BuildBuddy connection, then hands them to the daemon. This keeps
credentials out of post-build loaders and does not materialize the LLVM/XLA/CUDA
action cache. A Docker daemon using its legacy image store rewrites the local
manifest during `docker load`; that local image ID is execution convenience,
not the canonical OCI digest used for publication or remote acceptance.

Normal publication is one-hop: a BuildBuddy remote runner materializes the
completed compressed layers beside the remote cache and executes the
`rules_img` publisher there. Only that runner receives the named
least-scope `GHCR_TOKEN` organization secret, restricted with `env-secrets`.
The public owner name `NarendraPatwardhan` is configuration, not secret
material. A runner-local, owner-only registry-auth file or credential
helper bridges those values into `rules_img` and is removed when publication
ends; it is never a Bazel input or remotely executed action environment. The
publish operation disables automatic remote-run retries.
Local publication is unsupported because it transfers the completed layers
through the laptop twice and makes operator connectivity part of release
correctness. Registry inspection by exact digest is the authority after the
remote push, not a missing terminal event or mutable tag.

CUDA product images expose `/usr/local/lib/nml` as the one dynamic-loader
boundary. A platform-selected symlink there resolves rules_cuda's hermetic
CUDA runtime from the binary runfiles before Rust initialization; the complete
PJRT CUDA 13 runtime remains selected separately by
`NML_CUDA_RUNTIME_RLOCATION`. Local Docker execution requests devices only
through the NVIDIA Container Toolkit's `--gpus all` contract. Missing toolkit
or CDI configuration is a hard environment failure; manually enumerating
`/dev/nvidia*` is not a supported product execution path.

Public GHCR at `ghcr.io/narendrapatwardhan/nml` is the image distribution
boundary. Public Container Registry storage and transfer are currently free,
and RunPod consumers pull anonymously by canonical digest. `rules_img`
publishes through the OCI Registry API; GitHub package inspection and other
supported administration use the versioned GitHub REST API. The GitHub CLI is
not part of this workflow. GitHub does not currently expose package-visibility
mutation through its Packages REST surface, so changing the first published
package from its private default to public is a one-time web control-plane
action. A classic PAT with only the required package scopes is stored as the
encrypted BuildBuddy organization secret `GHCR_TOKEN`; its matching public
username is fixed by the repository owner. An explicitly selected token file
may instead be injected into one remote run through BuildBuddy's redacted
short-lived secret channel. Neither form enters Bazel inputs, source files,
ordinary environment properties, logs, OCI layers, or RunPod.

### 12.2 RunPod control-plane boundary

RunPod orchestration is repository tooling, not part of `products/serve` or the
`nml` facade. It is independently rewritten from the readable
`/mnt/workspace/llmd-remote` reference and built as a hermetic Python 3.12
`py_binary` with `rules_python`. The controller is standard-library-only while
that remains sufficient. `rules_uv` is not introduced merely to wrap a source
file with no third-party Python dependencies; if real dependencies are later
accepted, one reviewed locking mechanism is selected rather than layering two
Python dependency graphs.
All Python executables use rules_python's script bootstrap. They enter the
registered hermetic interpreter directly and never require `/usr/bin/python3`
on a BuildBuddy coordinator, remote worker, or minimal operator host.

The RunPod protocol split is deliberate. REST owns template discovery,
creation, and other operations for which it is reliable. GraphQL owns Pod
placement, creation, live runtime state, public SSH-port discovery, and
termination because those responses provide the lifecycle and mapping data NML
requires. Application readiness is a third, independent signal obtained from
the actual NML health or contract-result endpoint. A control-plane status is
never substituted for application readiness, and the Pod lifecycle is not
migrated to REST merely because a newer REST endpoint exists.

Direct ephemeral Pod creation is the default; reusable private templates are
an optional, explicitly requested policy. A template is reused only when its
private/public mode, exact image digest, entrypoint, command, disk, ports,
environment, and volume fields still match; drift is a hard error rather than
an implicit mutation. Every execution uses an immutable image digest, bounded
GPU fallback list, deadline, and cleanup policy. RunPod
credentials and model-access tokens remain outside Bazel action inputs and
repository state. Public model artifacts require no runtime access token.
Lease records live under the user's state directory, support concurrent Pods,
and retain enough information to identify and terminate a possibly billable
orphan after partial failure. Normal completion, test failure, timeout, signal,
and readiness failure all attempt termination. The current controller does not
offer retention; any future retention policy must be explicit and bounded by a
maximum deadline.

Persistent full-model runs attach the explicitly selected network volume in
its owning data center and mount it at `/workspace`. RunPod tooling transports
only a sorted map of canonical, absolute `NML_*` contract-input paths below
that mount; it does not know model names, artifact schemas, or which inputs are
required together. The product-owned runner definition supplies those names
and requirements. The controller records the opaque input map with volume,
data-center, and mount identity in the durable lease and terminal result.
Before every child launch the reusable runner removes the complete product
input set, then restores only the inputs declared by that exact contract.

Each device-contract image starts the same authenticated, model-agnostic Rust
runner and never contains Bazel, a shell, or SSH. A tiny image-owned binary
provides the closed contract allowlist, runfile identities, arguments, and
isolated inputs. The runner discovers hardware through `nvidia-smi`, reports
exact artifact and GPU identity at readiness, accepts one serial contract set,
enforces contract and total deadlines, kills children on interruption, and
retains one immutable bounded terminal result. The same digest and protocol
are used through local Docker `--gpus all` and RunPod; only hardware and
control-plane evidence differ.

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
    real NVIDIA execution, including the complete compact NVFP4 contract from
    SM75 onward, local or rented, unsandboxed and never result-cached

OCI CUDA device-contract image
    separate substrate and GPT-OSS acceptance envelopes, sharing only the
    reusable runner and CUDA userspace closure
```

The standard commands are:

```sh
bb test --config=buildbuddy --config=cpu //:cpu_contracts
bb test --config=buildbuddy --config=cuda //:cuda_remote_contracts
bb build --config=buildbuddy --config=cuda //:cuda_contract_binaries
bb test --config=buildbuddy --config=cuda //:cuda_package_contracts
```

The final device gate is not a BuildBuddy test command: BuildBuddy workers do
not own an NVIDIA device. An operator publishes the exact contract-image
digest and asks the repository RunPod controller—or an explicitly approved
local OCI runtime with `--gpus all`—to run the image's allowlisted contract.

Remote CUDA compilation is not CUDA execution evidence. Device tests are
`exclusive`, unsandboxed, and non-cacheable because the installed GPU, driver,
and `/dev/nvidia*` state are external singleton resources. Hosted workflows do
not schedule them. Package tests, by contrast, are hermetic file-closure tests
and belong on BuildBuddy.

BuildBuddy remains the compilation, CPU-execution, CUDA-remote, package, OCI
image construction, and registry-publication venue. Publication runs on a
BuildBuddy remote runner so the image layers stay colocated with the remote
cache instead of being downloaded to an operator machine and uploaded again.
The runner receives only the named `GHCR_TOKEN` organization secret through
its `env-secrets` execution property. Registry
credentials must never be Bazel inputs, command-line values, ordinary action
environment variables, cached outputs, or log content. The publication action
is deliberately non-retrying because pushing a tag is an external mutation;
completion is established by independently resolving the public tag to its
immutable registry digest.

Local publication through `bb run` or a host Docker credential store is not a
supported path. Local and RunPod GPU executors pull the same digest and run the
same in-image contract selection; neither recompiles NML, runs Bazel, nor
carries a source checkout. Provisioning a paid external Pod is an explicit
remote-runner action, never a hermetic or cacheable `bb test`.

The one-hop procedure is repository-owned rather than reconstructed by an
operator: `bash tools/publish-serve-image.sh`. It requires a clean, pushed
commit; reads the package token from `../github.packages.key` by default (or
`GHCR_TOKEN_FILE` when deliberately overridden); injects it through
BuildBuddy's redacted short-lived secret header; and creates and removes the
owner-only Docker credential store on the remote runner. The runner invokes
the canonical CUDA publication target beside BuildBuddy's cache, so no OCI
layer traverses the operator machine. This is the only supported token-file
publication flow; ad hoc `docker login`, local `bb run` publication,
token-bearing command lines, and locally constructed images are not.

Every OCI GPU result records the NML commit, image digest, contract selection,
GPU model and UUID when available, compute capability, driver, execution
venue, start/end time, and structured result. RunPod evidence additionally
records Pod identity and requested/allocated GPU profile. The permanent
contract runner serializes GPU use and publishes a bounded machine-readable
result; a successful container start, SSH connection, or readiness response is
not a numerical contract result.

Performance contracts report compilation, parameter upload, first execution,
steady execution, and download separately. CPU and CUDA performance statements
must identify build mode, workload, device, and phase rather than folding
compiler or transfer time into a misleading single number.
Product reports additionally separate asynchronous host submission by
component class. GPU kernel attribution comes from a device profiler or
compiler diagnostics; host timers must not be relabeled as device time.

Every paid RunPod product execution uses one inseparable diagnostic harness:
Nsight Systems launches GDB, and GDB launches the exact OCI product entrypoint
as its inferior. There is no debugger-only or profiler-only acceptance path.
Success requires a normal inferior exit, the product's numerical completion
contract, a complete `.nsys-rep`, and the exported CUDA kernel, API, launch,
and memory summaries in one attempt directory. The Pod is not terminated until
that entire directory has been copied to the operator machine and validated.
This cost is intentional during pre-alpha development: a successful result
without crash evidence or attribution data is incomplete evidence and must not
consume a paid execution slot.

## 14. Capability boundary and forward work

The source-guided substrate phase is complete: NML has its own coherent
CPU/CUDA graph, compiler, runtime, checkpoint, sharding, attention, MoE,
tokenizer, and real-model architecture. ZML remains valuable reference
material, but NML's requirements and this document now govern new work.

The remaining validation debt is explicit:

- extend the established SM86/SM90 FA2, FA3, and Triton paged-attention
  numerical evidence with representative optimized-attention performance
  coverage; grouped compact-MoE phase coverage is established separately;
- execute and measure multi-GPU CUDA Shardy placement and collectives;
- run native Linux AArch64 CPU/CUDA contracts, including DGX Spark;
- run native Apple Silicon CPU contracts.

The current first-priority product territory is the exact GPT-OSS 20B NVFP4
vertical in [`NVFP4.md`](./NVFP4.md): artifact identity, parameter/storage
redesign, CPU execution, fused pre-Blackwell emulation, native Blackwell
execution, grouped MoE, Shardy placement, and complete model evidence.

Later product territory includes:

- W4A16 and W8A8 execution verticals when selected by a concrete product
  artifact or workload;
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

Quantized formats are introduced only for an exact selected product artifact
or workload. Distributor reputation alone is insufficient: the pinned files
and their metadata must define one auditable representation. Similar names do
not make two FP4 recipes interchangeable. NVFP4 is now explicitly selected;
MXFP4 and generic four-bit cases are not. A trusted BF16 artifact remains a
valid oracle or later product choice when its declared provenance, memory, and
performance costs are accepted.

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
| D-004 | NVFP4 is a selected first-priority product vertical; W4A16 and W8A8 remain independent future goals, and none is an enum-only claim. |
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
| D-019 | Unrequested release, deployment, editor, and repository-support surface remains outside scope; BuildBuddy and the explicitly approved OCI/RunPod execution surface are the exceptions. |
| D-020 | Bazel owns independent additive CPU and CUDA settings. |
| D-021 | Local Bazel state reuses the sibling `../nml-bazel-cache`. |
| D-022 | PJRT GPU custom-call registration and its complete lifecycle are foundational. |
| D-023 | MLIR `index` is compiler-only and never a runtime dtype. |
| D-024 | The coherent compiler graph is pinned at OpenXLA commit `41370d1124c74d7b93a207136a636d8c631cbed9`. |
| D-025 | Original upstream sources are preferred; ZML-hosted source forks require explicit review. |
| D-026 | Attention and complete selected CPU/CUDA substrate coverage precede novel quantization. |
| D-027 | BuildBuddy is opt-in, credential-free in-repo, hermetic, image-pinned remote execution/cache. |
| D-028 | GPT-OSS 20B NVFP4 is the first-priority product vertical; BF16 may support its oracle/conversion evidence but is not a prerequisite serving milestone. |
| D-029 | Shardy is the only SPMD partitioner; legacy GSPMD is out. |
| D-030 | Placement metadata is part of the graph/checkpoint/runtime contract, not a later retrofit. |
| D-031 | Compile, package, CPU, and real-GPU contracts run where their resources truthfully exist. |
| D-032 | Portable blockwise paged attention is a CPU product path and CUDA fallback. |
| D-033 | Dense and paged KV state share one explicit persistent `CacheSpec`/`Cache` ownership model. |
| D-034 | Dispatch follows actual kernel capability: portable SM75, Triton SM8x/SM90, FA2 SM80-SM89, and FA3 SM90. |
| D-035 | CUDA source compilation uses hermetic Clang and a coherent hermetic GCC libstdc++ static runtime. |
| D-036 | FlashAttention's CUDA 12.8 source compiler and the CUDA 13.1 PJRT runtime are separate compatible contracts. |
| D-037 | Bazel target platforms describe products; execution platforms describe action machines. |
| D-038 | SM80/SM90 optimized kernels remain mandatory build inputs; suitable-device execution accepts an implementation path rather than every GPU SKU. SM86 proves FA2 plus Triton, SM90 proves FA3 plus Triton, and both prove grouped-MoE numerics. Dedicated optimized-attention performance/tuning and multi-GPU execution remain explicit debt and are never fabricated. |
| D-039 | IREE tokenization is a framework service; model protocols and product token semantics remain product-owned. |
| D-040 | `products/serve` owns a model-neutral serving control plane; Tokio tasks communicate through bounded channels with one dedicated PJRT engine owner. |
| D-041 | OCI images are the shared Linux CUDA userspace envelope for local and RunPod execution; hardware and driver evidence remains venue-specific. |
| D-042 | BuildBuddy owns compilation, CPU execution, GPU-independent/package contracts, and OCI construction; GPU execution consumes the immutable image elsewhere. |
| D-043 | RunPod templates use REST where reliable, while Pod placement, lifecycle, runtime/SSH mapping, and termination remain GraphQL-owned; application readiness is independent. |
| D-044 | RunPod orchestration is a rewritten standard-library Python 3.12 `rules_python` tool; `rules_uv` is absent until accepted third-party dependencies require locking. |
| D-045 | GPT-OSS 20B with one exact trustworthy NVFP4 artifact is the sole selected serving model. |
| D-046 | The selected GPT-OSS checkpoint vertical is NVFP4; BF16 is optional oracle/source evidence, MXFP4 is not implied, and unknown recipes hard-fail. |
| D-047 | Model weights remain outside OCI layers and are mounted or revision-pinned into an optional model cache. |
| D-048 | `rules_img` is NML's sole OCI rule set; `rules_oci` and parallel image graphs are prohibited. |
| D-049 | Product work proceeds with the complete GPT-OSS NVFP4 vertical first; OCI closure, general serving improvements, and unrelated formats resume afterward unless directly required for NVFP4 evidence. |
| D-050 | Public GHCR is the OCI registry; rules_img uses its registry API, supported GitHub administration uses REST, the GitHub CLI is prohibited, and package visibility uses a one-time web action only because REST lacks that mutation. |
| D-051 | NML is pre-alpha and has no backward-compatibility obligation; obstructive APIs and internals are replaced without legacy adapters when a better verified design is available. |
| D-052 | Ordinary `Tensor`/`Buffer` values remain single-shape dense values; one logical `Parameter`/loaded-parameter boundary owns dense or structured physical components. `TensorStore`, `NmlStruct`, `Bufferized<T>`, dense-only parameter input markers, and logical-shape checkpoint upload are deleted rather than adapted. |
| D-053 | NVFP4 dispatch is truthful by capability: CPU and pre-Blackwell devices consume compact weights through exact/fused emulation, while only proven SM100+ block-scaled execution is called native NVFP4. |
| D-054 | GPT-OSS uses one package-private `openai-harmony-gpt-oss-v1` protocol owner over IREE `o200k_harmony`; NML reimplements the narrow wire contract and does not consume the broader `openai-harmony` crate or execute user tools. |
| D-055 | Framework crates expose model-independent mechanisms only; GPT-OSS artifact, architecture, protocol, scheduling, and lifecycle policy remain under `products/serve`. |
| D-056 | GPT-OSS shape families compose bounded embedding, reusable layer-kind, and head executables through asynchronous PJRT dependencies; a full transformer is not one compiler module. |
| D-057 | GPT-OSS compilation profiles are complete before parameter residency; request execution selects an existing bounded profile and never invokes XLA. |
| D-058 | Every Triton custom call binds StableHLO tensors against the immutable builder-authored TTIR ABI; mismatched count, order, address space, or element type fails before XLA. |
| D-059 | Static MoE capacity never implies expert work: sparse decode launches only selected route blocks, general schedules carry an active-block scalar, and inactive/non-local kernels touch no weights. |
| D-060 | GPT-OSS cache pages are 16 tokens; CUDA attention tile width is independently bounded so cache allocation policy cannot create spill-heavy kernels. |
| D-061 | Loaded PJRT executable arity is resolved once, and product timings distinguish asynchronous component submission from synchronized device execution. |
| D-062 | GPT-OSS generation uses explicit-state runtime top-k/temperature/top-p/min-p sampling by default; greedy decoding is an explicit `top_k = 1` request, not product policy. |
| D-063 | Every grouped expert backend exposes gate/up plus one activation as `[assignments, intermediate]`; down never receives interleaved gate/up channels or recomputes their activation. |
| D-064 | Compact CUDA decode uses dedicated `M = 1` GEMV with exact register-local E2M1/E4M3FN decoding and block-scale reuse; tensor-core matrix tiles remain the distinct prefill family. |
| D-065 | Every paid RunPod product run is one combined Nsight-Systems-over-GDB execution; acceptance requires debugger, product, and profiler artifacts before Pod termination. |

## 16. Provenance and reference relationships

### ZML

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

### OpenAI Harmony

The official [OpenAI Harmony](https://github.com/openai/harmony) repository is
cloned read-only at `references/harmony`, commit
`abd677f7ac962629c808197caa1feb9e3e95d2b0`. It is an Apache-2.0 protocol and
fixture reference, never a build or runtime dependency. NML stays compatible
with the GPT-OSS subset while owning tokenizer lifetime, artifact validation,
failure behavior, and incremental serving events in Rust. Any future Harmony
feature must be justified by an actual serving consumer and added to this one
versioned owner rather than introducing a parallel renderer, parser, tokenizer,
or dependency graph.

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
