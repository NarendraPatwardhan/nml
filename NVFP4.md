# NVFP4 execution architecture

Status: current first-priority design and implementation contract

This document defines NML's NVFP4 vertical. It is intentionally narrower and
more demanding than "add an FP4 enum." It covers the selected checkpoint,
logical parameter representation, physical storage, loading, sharding,
lowering, CPU and CUDA execution, numerical acceptance, and end-to-end
GPT-OSS 20B evidence.

NML is pre-alpha. There is no compatibility obligation to preserve an API or
internal representation that prevents a coherent design. Existing code should
be reused when its ownership boundary remains sound, but checkpoint, graph,
buffer, or model APIs may be replaced rather than wrapped in compatibility
layers. No migration adapters, deprecated aliases, or duplicate old/new paths
are required.

[`SYSTEM.md`](./SYSTEM.md) remains the system contract and
[`TASKS.md`](./TASKS.md) remains the implementation ledger. Where the previous
milestone order put BF16 serving before quantization, this document and the
updated system decision index supersede it: the GPT-OSS 20B NVFP4 vertical is
now the first priority. OCI closure, general serving work, metrics, speculative
decoding, W4A16, W8A8, and unrelated model work wait unless they are a direct
prerequisite for proving this vertical.

## 1. Product objective

NML will load one exact, trustworthy GPT-OSS 20B NVFP4 representation, retain
its weights in compact form, and execute it correctly and efficiently on:

- CPU, as both an independent correctness target and a performance target;
- SM75, including the local GTX 1660 Ti;
- SM8x, with real suitable-device evidence;
- SM90, with real suitable-device evidence; and
- SM100 or newer, using native NVFP4 instructions when the complete pinned
  compiler/runtime path supports them.

"NVFP4 support" is accepted only when the representation crosses the complete
vertical. A parser without execution, a dtype without scales, a kernel without
a model, a compiled kernel without a real launch, or a model that silently
retains BF16 weights does not satisfy the objective.

The target is not identical speed on every architecture. The target is one
logical model representation with truthful capability dispatch:

| Device class | Required execution identity |
| --- | --- |
| CPU | NVFP4 storage with an exact CPU implementation; no persistent BF16 expansion. |
| SM75 | NVFP4 checkpoint executed by a fused W4A16 emulation kernel. |
| SM8x | NVFP4 checkpoint executed by a fused W4A16 emulation kernel. |
| SM90 | NVFP4 checkpoint executed by a fused W4A16 path first; FP8-assisted alternatives are admitted only after measurement and numerical acceptance. |
| SM100+ | Native NVFP4 block-scaled tensor-core execution when the selected shape/layout is supported; otherwise a named emulation path or a hard diagnostic. |

Only the SM100+ route may be described as *native NVFP4*. Earlier devices can
still receive the memory-bandwidth benefit of compact weights and a useful
speedup, but their arithmetic is emulated by unpacking and scaling tiles inside
the contraction kernel.

## 2. Terms and boundaries

### 2.1 Dtype, encoding, recipe, and execution path

These concepts must remain distinct:

- A **dtype** is a complete scalar contract. One element's bits independently
  define its value and fixed width. BF16 is a dtype.
- A **storage encoding** defines how physical bits represent one component.
  Packed E2M1 pairs and raw E4M3 scale bytes are storage encodings.
- A **quantization recipe** combines payload encoding, block geometry, scales,
  global factors, layout, padding, rounding, and reconstruction algebra into a
  tensor representation. NVFP4 is a recipe.
- An **execution path** consumes a recipe on a particular device. Native
  Blackwell block-scaled MMA and SM90 BF16 tile upcasting are different paths
  for the same logical representation.

Consequences:

- `DType::NvFp4` must not be added.
- E2M1 must not become a generally usable public tensor dtype merely to make a
  kernel ABI convenient.
- E4M3 scale bytes do not justify exposing general FP8 graph arithmetic. They
  are representation components loaded as physical bytes and interpreted only
  by NVFP4-aware code.
- W4A16 and W8A8 are not aliases for NVFP4. They require their own exact
  recipes if they are scheduled later.
- MXFP4 is not admitted. It uses a different scale encoding and block size;
  the fact that both recipes use E2M1 payload values does not make their
  checkpoints interchangeable.

### 2.2 Scope

The first vertical includes:

- weights for embeddings, ordinary projections, and GPT-OSS experts;
- BF16 or F16 activations with F32 accumulation on emulated paths;
- transient, explicitly specified activation quantization for a Blackwell
  native block-scaled path when the hardware instruction requires both
  operands in NVFP4;
- native block-scaled execution on Blackwell where supported;
- output in the model's declared ordinary activation dtype;
- dense and grouped/MoE contractions;
- exact checkpoint and transform identity;
- logical Shardy placement and representation-aware physical sharding;
- end-to-end GPT-OSS generation and memory/performance evidence.

The first vertical does not automatically include:

- NVFP4 KV-cache storage;
- NVFP4 logits or model outputs;
- activation quantization on pre-Blackwell devices;
- training, stochastic rounding, gradient recipes, or random Hadamard
  transforms;
- a generic quantization plugin system;
- GGUF, Q4_K_M, MXFP4, arbitrary distributor formats, or heuristic format
  detection; or
- serving features unrelated to proving the model representation.

Weight-only inference is deliberately narrower than Transformer Engine's full
training recipe. Training-specific 2D scaling, stochastic rounding, gradient
state, and transforms are not inherited unless the selected inference artifact
actually requires them.

## 3. Normative NVFP4 representation

NVIDIA describes an NVFP4 value as:

```text
x = x_e2m1 * s_block * s_global
```

where:

- `x_e2m1` is a four-bit E2M1 value;
- `s_block` is an FP8 E4M3 scale shared by 16 consecutive values; and
- `s_global` is an FP32 factor for the tensor.

The E2M1 magnitudes are `0, 0.5, 1, 1.5, 2, 3, 4, 6`, with a sign bit. Two
payload values are packed into each byte. The compact lower bound is therefore
0.5 payload bytes plus 1/16 scale byte per logical weight, or 0.5625 bytes per
weight before global factors, padding, metadata, and alignment. This explains
the approximately 3.5x storage reduction relative to a 16-bit tensor; it does
not by itself prove a runtime speedup.

NVIDIA's native matmul contract additionally arranges block scales in hardware
layouts, requires packed contraction dimensions, and constrains native problem
geometry. Current nvmath documentation describes 128x64 scale tiles, M/N
multiples of 128, K multiples of 64, and two E2M1 values per byte. Transformer
Engine documents rowwise/columnwise scale layouts and scale padding to 128 rows
and four scale columns. Those are backend/layout facts, not permission to
assume that every downloadable checkpoint already uses that exact swizzle.

The selected artifact manifest must settle all of the following before code is
allowed to interpret its payload:

1. Logical tensor shape and axis order.
2. Whether the payload is packed along K, N, or another named logical axis.
3. Which nibble stores the earlier logical element.
4. Exact E2M1 special-value and signed-zero behavior.
5. Block size, block axis, and edge padding.
6. E4M3 variant and raw byte interpretation.
7. Whether stored scales are direct scales, inverse scales, or preconditioned
   values.
8. Global scale direction and whether it is per tensor, row, fiber, or expert.
9. One-dimensional versus two-dimensional weight scaling.
10. Source scale layout, padding, and swizzle.
11. Logical transpose and any prepacked contraction layout.
12. Rounding/conversion provenance and independent oracle.

The formula above is the semantic reference. An artifact may store an inverse
or folded factor for efficient kernels, but its manifest must name that
algebra. Loader or kernel code must never infer the direction from a familiar
tensor suffix.

Authoritative format references:

- [NVIDIA Transformer Engine NVFP4 format and layouts](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- [NVIDIA nvmath NVFP4 matmul layout and hardware requirements](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/linalg/generated/nvmath.linalg.advanced.Matmul-class.html)
- [NVIDIA NVFP4 and MXFP4 comparison](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [CUDA 12.8 FP4 intrinsics and emulation note](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP4.html)
- [cuDNN block-scale quantization semantics](https://docs.nvidia.com/deeplearning/cudnn/v1.15.0/operations/BlockScaling.html)

## 4. Current NML assessment

NML now has the representation-neutral parameter boundary needed by compact
weights. Dense parameters use that boundary as the one-component case; NVFP4 still
requires its exact artifact schema, packed encodings, sharding rules, semantic
operations, and capability-selected kernels. The implementation must extend
this substrate rather than reintroducing a parallel quantized model stack.

| Area | Current state | Remaining NVFP4 work |
| --- | --- | --- |
| Ordinary types | `DType` and `Shape` describe one fixed-width scalar tensor. | Keep this invariant. Do not add NVFP4 to `DType`. |
| Host tensor storage | `Slice` remains one ordinary dense value; encoded components use checked packed-E2M1x2 and E4M3FN-bit storage specs. | Preserve this separation when adding prepared native layouts. |
| SafeTensors registry | `TensorRegistry` owns bounded artifact records; `ParameterSet` binds the selected artifact's exact physical components and streams them directly. | Add independent layer/generation fixtures, not another loader path. |
| Parameter model | Immutable `Parameter` and `LoadedParameter` own logical identity and physical component buffers; dense is the one-component case and `NvFp4` is the closed three-component case. | Extend only when a genuinely different recipe is admitted. |
| Structural loading | `ParameterTree` maps parameter leaves to loaded-parameter leaves across nested product structure, transactionally accounting compact components. The former dense-only traversal is gone. | Preserve one traversal; no quantized side channel is permitted. |
| Runtime buffer | `Buffer` remains one ordinary physical tensor with one `Shape` and PJRT shards. | Keep this boundary; NVFP4 is a bundle of component buffers, never a `Buffer` variant. |
| Sharded loading | Component upload derives co-sharded payload/scale spans from logical ranges and rejects non-block-aligned geometry before allocation. | Add a versioned prepared-layout transform only when a backend requires one. |
| Graph IR | Semantic linear, embedding, and grouped-MoE operations consume `Parameter`; executable manifests flatten and validate physical component bindings. CPU, SM75, and SM80+ lowering preserve one semantic graph. | Add native Blackwell lowering and complete product execution evidence. |
| Compiler target | CUDA lowering uses one private named `CudaCapabilities` value for attention and compact-weight dispatch. | Add native-layout predicates when the native path lands. |
| Triton builder | Typed verified TTIR owns packed NVFP4 emulation kernels and the pinned XLA custom-call boundary. | Add the exact `tt.dot_scaled` surface needed by native Blackwell execution. |
| Triton dependency | XLA's pinned Triton commit is `c05aa65087a9a1a6b8a08fdbb474aba834d5cddf`. It contains E2M1, `tt.dot_scaled`, SM90 FP4 upcast/decomposition, NVFP4 helpers, and Blackwell lowering. | Lift only the missing typed bindings/builder surface. Do not introduce another Triton pin or Python dependency. |
| CUDA custom calls | The process-lifetime typed XLA/PJRT lifecycle registers product SM75 linear, embedding, and grouped-expert adapters on the PJRT stream. | Reuse it only if native Blackwell needs a vendor adapter. |
| MoE | CPU, schedule-driven SM75 WMMA, and grouped SM80+ Triton paths execute compact gate/up/down weights without a host expert loop or dense conversion. | Measure and tune real model shapes; prove multi-GPU expert sharding. |
| Model identity | The selected artifact manifest and parameter representation identity are immutable, but future prepared-layout/cache keys are not complete. | Carry recipe and transform identity into prepared-weight and cache keys. |

The foundational redesign is complete. The next change is not another API
redesign or a `DType` variant; it is the exact NVFP4 representation added to
this logical-parameter/physical-component boundary.

## 5. Parameter architecture

### 5.1 Core concepts

NML has three distinct leaves:

```text
Tensor
  an executable ordinary graph value with one Shape

Parameter
  an immutable logical model value with one logical Shape and one declared
  representation; it is not general graph data

LoadedParameter
  the runtime owner of the physical Buffer components required by a Parameter
```

`Parameter` is the justified addition to the compact public model-building
surface. It prevents `Tensor` from serving as both an ordinary graph value and
an encoded checkpoint parameter. Representation implementation types
remain private or under a deliberately narrow checkpoint/parameter module;
there must not be public classes for every backend packing.

A representation uses one closed, exhaustively matched specification plus a
common physical-component envelope:

```text
ParameterSpec
  logical_shape: Shape
  representation: RepresentationSpec

RepresentationSpec
  Dense(DenseSpec)
  NvFp4(NvFp4Spec)

ComponentSpec
  role
  artifact record
  physical storage encoding and dimensions
  logical-axis mapping
  padding and alignment

NvFp4Spec
  payload component
  block-scale component
  global-scale component
  block axis and size
  payload packing
  scale algebra and E4M3 variant
  source layout and padding
  logical transpose
  transform identity
```

`RepresentationSpec` is deliberately a closed enum, not a dynamically
registered plugin trait. Adding a later block-INT4 or sparse representation
adds one validated variant and forces exhaustive review of loading, sharding,
dispatch, accounting, and diagnostics. Generic infrastructure operates on the
derived `ComponentSpec` list; only representation-aware operations interpret
component meaning. A target-specific prepared layout, Shardy placement, tied
identity, LoRA composition, and mutable optimizer state remain orthogonal and
must not become representation variants.

This is a design schema, not a demand for those exact Rust field names. Private
validated constructors prevent invalid combinations from being assembled.

### 5.2 Artifact records are not logical tensors

SafeTensors parsing should first produce bounded physical records:

```text
ArtifactRecord
  file span
  declared storage spelling
  physical dimensions
  byte length
```

Model-specific binding then turns one or more records into a `ParameterSpec`.
For an NVFP4 weight, those records normally include packed values, local
scales, and a global factor. The registry validates spans and storage sizes;
the model manifest validates their joint logical meaning.

Internal storage encodings may include:

- ordinary fixed-width `DType`;
- `PackedE2M1x2` stored in one byte; and
- `E4M3FnBits` stored in one byte.

These encodings must not be re-exported as general graph dtypes. PJRT sees
their physical buffers as `u8`/`i8` tensors; only an NVFP4-aware lowering may
interpret the bits.

### 5.3 Symbolic graph inputs

A symbolic `Parameter` owns stable component input IDs. Operations such as
linear, embedding, and grouped expert projection consume the logical
parameter. Lowering expands it to the exact component operands required by the
selected backend.

The ordinary user-level operation remains semantic:

```text
linear(activation, weight_parameter, optional_bias_parameter) -> Tensor
```

It must not accept a backend selector, a tuple of raw payload/scales, or a
caller-authored dequant graph. Dense and NVFP4 parameters share the operation;
capability and representation select a private lowering.

Operations that require ordinary small parameters, such as RMSNorm weights,
may lower a dense `Parameter` to an ordinary graph input. Operations that do
not define quantized semantics must reject an NVFP4 parameter before MLIR
construction.

### 5.4 Structural model traversal

The former `NmlStruct` derive, trait, and `Bufferized<T>` alias are deleted,
not adapted or retained. Their `Tensor -> Buffer` leaf transformation encodes
the dense-only model that this redesign replaces. The resulting single
parameter-tree system must:

- traverse nested structs, enums, arrays, and options;
- declare logical `Parameter` leaves;
- resolve each leaf to one `LoadedParameter`;
- flatten component buffers deterministically for executable binding;
- preserve parameter name and component role in diagnostics/accounting; and
- reject missing, extra, duplicate, or representation-mismatched records.

There is no legacy dense-only traversal beside the new system. Dense parameters
are the one-component case of the same system.

### 5.5 Declaration, graph, and loading separation

The former `TensorStore` is deleted. It improperly combined SafeTensors
lookup, checkpoint aliases, logical declaration, graph operations, program
finalization, and device loading. Its responsibilities become four boundaries:

```text
ArtifactIndex
  bounded raw physical records and file spans

ParameterSet
  prefixed logical declarations, artifact binding, aliases, and tied identity

Graph
  activations, semantic operations, outputs, and program finalization

ParameterLoader
  component planning, transactional upload, preparation, and accounting
```

The concrete dense starting point names the first boundary `TensorRegistry`
because it indexes validated SafeTensors records. `ParameterLoader` is a
private responsibility behind `ParameterSet::load`, not another public product
type. The boundary is enforced by dependencies and ownership: the loader may
read the registry and create `LoadedParameter`s, but neither it nor
`ParameterSet` owns a graph or exposes graph-operation proxies.

`ParameterSet` never proxies graph operations and `Graph` never reads a
checkpoint. Model declaration produces a parameter tree once; any number of
graphs reuse it. Loading the same tree produces its loaded counterpart once.

The former dense `InputKind::Parameter` marker is replaced by a private
component binding manifest carrying logical parameter identity,
representation identity, component role, and physical specification. XLA still
receives a flat argument list, but that list is an output of the manifest, not
the source of truth.

The former `upload_checkpoint_from(Shape, ...)` contract is also replaced by
physical-component upload. Ordinary dense upload is its one-component case;
logical `Shape::byte_count` is never used to size an encoded component.

### 5.6 Runtime ownership

`Buffer` remains one ordinary physical tensor. A `LoadedParameter` contains:

- the validated logical `ParameterSpec`;
- immutable payload, scale, and global-factor `Buffer`s;
- source artifact identity;
- optional prepared-layout identity; and
- exact resident-byte accounting.

Backend preparation may create a target-specific packed/swizzled
`LoadedParameter`, but it must be explicit and immutable. The source buffers
may be released after a verified one-time transform when they are not the
prepared representation. The runtime must never keep both source and prepared
copies accidentally, and must never retain a full BF16 expansion.

Executable arguments bind a logical parameter, not an arbitrary list of
buffers. The executable's internal manifest expands that binding and validates
component count, shapes, representation identity, transform version, sharding,
platform, and memory placement.

## 6. Artifact selection and identity

One exact GPT-OSS 20B artifact must be selected before implementation assumes
tensor names or packing. A repository title containing "NVFP4" is not enough.

The audit must record:

- distributor and immutable repository revision;
- license and relation to the upstream GPT-OSS release;
- every file path, size, cryptographic hash, and role;
- model configuration, tokenizer, and Harmony assets;
- every tensor/record name, storage spelling, physical shape, and byte extent;
- logical parameter mapping and transpose;
- payload nibble order and scale/global-factor algebra;
- quantizer/conversion provenance and software version;
- excluded or retained higher-precision layers;
- independent reference loader and fixed oracle values; and
- claimed hardware/runtime requirements.

The checked manifest is part of the source tree. Downloading remains outside
the Bazel graph, but no downloaded directory becomes a model merely because
its filenames match. Validation completes before graph construction, device
allocation, or layout transformation.

If no trustworthy existing NVFP4 artifact survives the audit, NML may define
one product conversion from an explicitly selected dense source. That is not a
generic converter. The conversion algorithm, scale computation, rounding,
layout, seed if any, source hashes, produced hashes, and quality evaluation
become part of the single representation identity. This decision must be
recorded before implementation; an MXFP4 payload must never be relabeled.
Artifact conversion is CPU-only and executes on a resource-bounded BuildBuddy
remote runner. GPU rental is reserved for runtime evidence; conversion does
not gain correctness or identity from CUDA.

A representation identity must include at least:

```text
model family and configuration identity
artifact revision and manifest digest
NVFP4 recipe version
logical-to-physical mapping version
prepared-layout version
```

It is used by executable caches, prepared-weight caches, prefix-cache keys,
distributed compatibility checks, results, and diagnostics. A free-form
`"nvfp4"` string is insufficient.

## 7. Logical shape, physical layout, and Shardy

The logical parameter shape remains model semantics. Physical packing must not
overwrite its axis tags, partitions, or layout.

For a logical matrix `[N, K]`, a common rowwise source representation has:

```text
logical values       [N, K]
packed payload       [N, ceil(K / 2)] bytes
block scales         [N, ceil(K / 16)] E4M3 bytes before padding/swizzle
global factor        scalar or artifact-declared fiber shape
```

NML recipe v3 derives physical storage from the operation. Indexed embedding
keeps the rowwise form above because lookup selects a complete vocabulary row.
Contractions retain logical `[N, K]`, `[E, 2I, K]`, and `[E, H, I]` shapes but
store their encoded components as `[packed K, N]`, `[E, packed K, 2I]`, and
`[E, packed I, H]`; block scales use the same order with `K/16`. This makes
adjacent lanes consume adjacent outputs for one reduction slice. The converter
first transposes GPT-OSS source experts into logical `[E, N, K]`, quantizes K
blocks, and only then swaps the encoded component axes. CPU, SM75, Triton
matrix, and Triton decode kernels consume these components directly. There is
no runtime transpose, second prepared copy, or earlier-recipe compatibility
path.

The selected artifact may differ. Every physical extent uses checked
arithmetic. Odd K, incomplete blocks, padding bytes, and scale padding have a
declared value and are masked or verified; they must not become observable
weights.

Shardy partitions logical axes. Representation code derives physical component
shards from those logical ranges:

- K-axis shards must align to the 16-value scale block and two-value payload
  packing, or use a declared padded/repacked boundary.
- Native Blackwell layouts may impose stronger K=64 and tile constraints.
- N/expert-axis shards slice payload, scale, and global factors together.
- Expert sharding must not create a hidden all-gather of all expert weights.
- A future prepared layout is local to its logical shard; it is never produced
  by preparing the whole model and slicing opaque bytes afterward. Recipe v3
  currently requires no prepared layout because its operation-shaped forms
  are directly consumable by every retained backend.

The physical representation carries a mapping from each logical axis to
payload and scale axes. Generic `Shape::byte_count` and ordinary `Slice`
regions are not used to infer quantized shard spans.

Distributed inference initially keeps stored weight scales immutable. Dynamic
activation NVFP4 would require synchronized global amax for some collectives;
that is outside the first weight-only pre-Blackwell path. If native activation
quantization is added, its collective semantics must become explicit rather
than inheriting training-library behavior accidentally.

## 8. Compiler and dispatch architecture

### 8.1 Named device capabilities

The compiler should receive one private `DeviceCapabilities` value instead of
passing major/minor integers into individual operations. It derives named
facts such as:

```text
is_cuda
supports_bf16_tensor_core
supports_fp8_tensor_core
supports_native_nvfp4
supports_xla_triton
supports_nvfp4_scale_layout
```

The exact facts remain compiler-private. Model code never branches on GPU
marketing names. Unsupported devices or shapes fail at the highest boundary
that has enough information, and error messages name the representation,
operation, shape, device capability, and missing requirement.

### 8.2 One semantic operation, private lowerings

NVFP4 linear dispatch follows the established attention pattern:

```text
semantic parameter-aware linear/grouped projection
  -> CPU exact/optimized path
  -> SM75 CUDA emulation
  -> SM8x Triton emulation
  -> SM90 Triton emulation
  -> SM100+ native Triton or native CUDA/library path
```

Dispatch is deterministic from parameter representation, logical shape,
activation dtype, device capabilities, and compiled kernel availability. There
is no environment-variable or public backend override used to manufacture
tests.

A fallback is allowed only if it preserves the product contract. In
particular, an unsupported native shape may select a fused emulation kernel,
but it must not silently allocate a persistent BF16 copy. The selected path is
available in structured diagnostics and benchmark output.

### 8.3 XLA ownership

XLA continues to own:

- the StableHLO graph;
- ordinary operations, scheduling, and memory planning;
- Shardy partitioning and collectives;
- PJRT compilation and execution;
- activation and state buffers; and
- custom-call ordering and dependencies.

Private custom calls own only the contraction region for which packed storage
and device-specific instructions matter. There is no second eager runtime and
no framework-level kernel scheduler.

The custom call receives physical payload/scales/global factor plus ordinary
activation tensors and returns an ordinary output tensor. Full dequantization
must not appear as a persistent graph value on optimized paths.

## 9. Kernel plan by backend

### 9.1 Independent scalar semantics

Before an optimized kernel exists, implement one small, exhaustive codec that:

- decodes all 16 E2M1 bit patterns;
- decodes every relevant E4M3 bit class, including zero, subnormal, normal,
  saturation, NaN, and unsupported special values according to the artifact;
- applies block and global scale algebra in F32;
- validates nibble order and edge padding; and
- produces deterministic high-precision reference values.

This codec is product code used by CPU execution, checkpoint inspection, and
oracles. It is not a disposable probe.

### 9.2 CPU

CPU has two stages that both remain supported:

1. An exact blockwise implementation that decodes tiles and contracts without
   retaining the complete dequantized matrix. This is the independent oracle.
2. An optimized implementation with architecture-appropriate vectorized
   nibble decode, scale expansion, cache tiling, and BF16/F32 contraction.

The optimized path may use Rust intrinsics or a narrowly scoped native kernel
when necessary, built through Bazel. It must retain a scalar tail, checked
alignment, and identical representation semantics on x86-64 and AArch64.
Materializing one temporary tile is allowed; materializing the full parameter
is not.

CPU acceptance includes both correctness and declared performance. A very slow
reference cannot be the final CPU product path merely because CUDA is faster.

### 9.3 SM75

The established XLA Triton integration does not currently serve SM75 product
kernels. SM75 therefore uses a dedicated CUDA custom-call kernel unless the
pinned compiler path is first proven to support the required TTIR end to end.

The kernel performs weight-only W4A16 execution:

1. Load a packed E2M1 weight tile and its E4M3 scales.
2. Extract low/high nibbles in registers.
3. Map E2M1 codes to F16/BF16 values.
4. Decode and apply local and global scales in registers.
5. Feed the tile to the best available Turing half-precision contraction,
   accumulating in F32.
6. Apply only explicitly supported epilogues and write the ordinary output.

Turing has no native BF16 tensor-core path. A BF16 model activation therefore
uses a declared tile-local BF16-to-F16 conversion before contraction and a
declared ordinary output conversion afterward. Its tolerance is measured
separately from native-BF16 SM8x/SM90 execution; it must never be presented as
bitwise BF16 arithmetic.

The local GTX 1660 Ti is the permanent real-device acceptance venue. A kernel
that compiles for `sm_75` but is not selected and launched there is incomplete.

### 9.4 SM8x

SM8x uses a typed Triton kernel with packed loads, local upcast, and ordinary
BF16/F16 `tt.dot` or the pinned compiler's verified E2M1 decomposition. The
generated TTIR is reparsed and verified through the existing `KernelSpec`
boundary.

The first accepted route is W4A16 with F32 accumulation. It must be benchmarked
for decode and prefill independently; decode is expected to benefit most from
reduced weight traffic, while large-M prefill may expose unpack overhead.

### 9.5 SM90

SM90 first uses the same semantically simple W4A16 route, tuned for Hopper
layouts and warp-group MMA. The pinned Triton source already contains an
E2M1-plus-scale to BF16 decomposition feeding MMAv3, as well as NVFP4 tile
upcast helpers. NML should expose that capability through typed builder APIs
rather than reproduce string IR.

An FP8-assisted W4A8 route may be evaluated later because Hopper has native FP8
tensor cores. It is accepted only if:

- activation quantization semantics are explicit;
- output error remains within the declared model tolerance;
- its workspace and amax/scales are accounted;
- it beats W4A16 for a declared workload; and
- it does not change the checkpoint identity.

### 9.6 SM100 and newer

Native execution uses Blackwell block-scaled instructions via the pinned
Triton `tt.dot_scaled` lowering or a product-owned CUDA/cuBLASLt/CUTLASS adapter.
Triton is preferred when the pinned XLA integration accepts the complete TTIR
and generated code because NML already owns its typed custom-call boundary.

If the native instruction requires two block-scaled FP4 operands, the operation
quantizes the ordinary activation transiently using an explicit 1D NVFP4
recipe, consumes its payload/scales immediately, and returns an ordinary
output. Activation scale computation, global-factor algebra, rounding, and
workspace are part of the operation contract and independent oracle. A mixed
NVFP4-weight/BF16-activation path that merely upcasts the weight and executes a
BF16 MMA remains an emulation path even on Blackwell; device generation alone
does not make it native.

Native acceptance requires:

- CUDA/PJRT and compiler versions that support the actual device;
- source-to-native scale repacking/swizzling when needed;
- checked M/N/K and alignment constraints;
- a real SM100+ launch through normal product dispatch;
- numerical comparison with the same CPU representation oracle; and
- generated-code or profiler evidence that native block-scaled instructions
  executed.

The current pinned Triton source is promising, not proof: its tests contain
Blackwell NVFP4 lowering, while its high-level NVFP4 matmul suite deliberately
skips devices below compute capability 10. NML must add the missing typed
`tt.dot_scaled` surface and prove that XLA's embedded Triton custom call accepts
it at the pinned revisions.

If the native vendor interface requires scale layouts distinct from the source
artifact, preparation performs a versioned one-time transform. Runtime kernels
must not reswizzle the complete parameter on every invocation.

## 10. Kernel algorithms and fusion

### 10.1 Ordinary projection

For `Y = X W^T`, the emulation kernel is tiled so each packed weight byte and
scale is loaded once per useful activation tile. Dequantization occurs inside
the K loop and feeds tensor-core fragments directly. The full dense `W` never
exists.

Kernel configuration is selected from a finite, source-owned family keyed by:

- device capability class;
- activation dtype;
- M regime (decode, small prefill, larger prefill);
- N/K alignment and tail behavior; and
- epilogue.

Do not compile a unique kernel for every request sequence length. Runtime M
tails use masks within bounded families.

For the retained pre-Blackwell Triton path, `M = 1` is a separate compact GEMV
family: one useful output tile per program, packed-pair loads, exact bitwise
E2M1/E4M3FN decoding, one scale load broadcast across its representation
block, and F32 reduction without a dead-row `tt.dot`. Decode ordinary,
gate/up, and down projections all follow this rule. Multi-row prefill retains
the finite tensor-core family established by ZML's grouped-MoE policy: small M
uses 64 output columns, 128 reduction lanes, four warps, and four stages;
larger regimes widen output tiles only when activation reuse justifies them.
These are named compiled families, not an autotuner hidden inside request
execution.

### 10.2 GPT-OSS grouped experts

GPT-OSS is MoE, so quantized grouped projection is part of the first vertical,
not a later optimization. The kernel consumes routed-token offsets plus packed
expert weights/scales and performs expert-local projection without launching a
host loop or constructing one dense expert matrix at a time.

Required cases include:

- gate/up projection;
- the exact GPT-OSS clamped/residual SwiGLU semantics;
- down projection;
- empty experts;
- uneven routed counts;
- capacity/tail masking; and
- expert-axis sharding.

Routing has two distinct quantities. `capacity` is the maximum statically
allocated schedule extent required by XLA shapes; `active_blocks` is the
runtime prefix that may execute. When the route set is sparse relative to the
expert set, each selected assignment owns one padded block directly and launch
capacity is exactly the number of selected routes. Otherwise the aligned
schedule carries its active prefix explicitly. An inactive or non-local Triton
program returns zero before it forms a compact-weight address. Masking input
rows while clamping an invalid expert to expert zero is forbidden: it still
loads and decodes an entire unselected matrix and destroys MoE sparsity.

Fusion should begin with transformations that remove unavoidable memory
traffic and have an exact model semantic boundary: bias, gated activation, and
the paired expert projections where numerically appropriate. Broad arbitrary
fusion is not a goal. Every fused path retains an unfused CPU oracle.

The expert interface itself is fixed: gate/up performs the paired contractions,
adds paired biases, applies the selected activation exactly once, and writes
`[assignments, intermediate]`. Down accepts only that activated tensor, then
adds down bias and applies routing weight. Down must not receive an interleaved
gate/up tensor or recompute activation per output tile. This boundary is shared
by dense and NVFP4 lowering; representation-specific code may choose matrix or
GEMV geometry but may not change model semantics.

### 10.3 Embedding and output projection

Embedding lookup should gather and decode only selected rows. Dequantizing the
whole embedding matrix is prohibited. Tied input/output weights share one
loaded representation and accounting owner.

Output projection is an NVFP4 linear operation; logits remain BF16/F32 as
declared. Sampling stays ordinary graph work.

### 10.4 Higher-precision exceptions

The selected artifact may retain norms, routers, embeddings, or other sensitive
parameters at BF16/F16. These are explicit dense `Parameter` leaves in the same
manifest. NML does not quantize them at load time or call a mixed checkpoint
"all NVFP4." Memory reports separate dense and NVFP4 resident bytes.

## 11. Triton and MLIR work

The private Rust TTIR builder needs only the surface required by accepted
kernels:

- packed integer loads and bitwise extraction;
- any missing shifts, masks, broadcasts, reshapes, transposes, and layout
  conversions;
- an internal E2M1 operand-format attribute;
- raw E4M3 scale representation or the exact typed FP8 value required by the
  pinned dialect;
- typed `tt.dot_scaled` construction, including optional scales, K packing,
  fast-math policy, and F32 accumulator;
- constraints for operand shapes and scale layouts; and
- canonical TTIR reparse/verification before StableHLO embedding.

Do not expose Triton types from `nml`. Do not accept arbitrary TTIR strings from
model code. Do not import Python Triton into the build. The relevant compiler
is already part of XLA's coherent dependency graph at commit
`c05aa65087a9a1a6b8a08fdbb474aba834d5cddf`.

Compiler-source tests are design evidence only. NML adds its own contracts at
three boundaries:

1. Builder tests verify exact typed TTIR and reject invalid formats/layouts.
2. XLA compile tests prove the pinned custom-call pipeline lowers the kernel for
   every supported capability class.
3. Real-device tests prove dispatch, launch, and numerics.

## 12. Loading and preparation pipeline

The load pipeline is transactional:

```text
bounded SafeTensors/index parse
  -> authenticated artifact-manifest and materialization-receipt validation
  -> logical ParameterSpec construction
  -> sharding plan validation
  -> physical component allocation/upload
  -> optional target-layout preparation
  -> prepared representation verification
  -> publish LoadedParameter to the model owner
```

Failure before the final step releases all partial allocations. A prepared
parameter is immutable and may be shared by all compiled executables for the
same platform/topology.

Full payload hashing belongs to the artifact materializer that precedes this
pipeline. It authenticates the pinned manifest, hashes every declared payload,
makes the verified materialization read-only, and atomically issues a bounded
filesystem-identity receipt. The load pipeline hashes only the small manifest
and rejects a missing, writable, or stale receipt. It never turns an invalid
receipt into an implicit multi-gigabyte scan, so launch latency is independent
of checkpoint size while ingestion retains byte-exact verification.

Preparation rules:

- CPU may retain source rowwise packed storage when its kernels consume it.
- Pre-Blackwell CUDA should prefer a source layout consumable by coalesced
  fused-dequant kernels; a one-time transpose/repack is allowed if measured.
- Blackwell may require hardware scale swizzling and contraction-aligned
  packing.
- Transform output has a versioned identity and is validated by decoding
  sampled/boundary blocks against source semantics.
- Persistent memory accounting names source, prepared, and discarded bytes.
- Compilation must not capture model payload in executables; weights remain
  reusable runtime arguments.

Direct checkpoint-to-PJRT streaming remains desirable, but its span calculation
must use component physical extents rather than logical `Shape::byte_count`.
Each component retains bounded chunking, staging ownership, exact reads, and
failure cleanup.

### 12.1 Product composition and executable reuse

NVFP4 representation code does not own a transformer. The selected GPT-OSS
product declares its exact checkpoint and builds four bounded executables for
each finite request shape family: embedding, a reusable sliding-attention
layer, a reusable full-attention layer, and the final head. The product binds
each loaded decoder layer to the matching representative executable slots only
after the runtime verifies logical shapes, representation identity, component
roles and physical storage, platform, placement, and compiled bindings.

The product enqueues these components through PJRT readiness dependencies.
Hidden state and K/V storage are donated between calls, while the host waits
only at the token observation boundary. Uploaded parameters and compiled shape
families are process-persistent; tokens, positions, the shared I32 page table,
and K/V buffers are request-local. GPT-OSS names, Harmony, attention schedules,
and artifact identities never enter representation, kernel, IR, checkpoint, or
runtime crates. The framework operation used by the model is the semantic
`routed_clamped_swiglu`, not a model-named escape hatch.

## 13. Correctness contract

### 13.1 Representation tests

Permanent CPU contracts cover:

- exhaustive E2M1 decode;
- E4M3 decode and rejected special values;
- low/high nibble order;
- direct/inverse scale algebra;
- zero blocks, maximum values, signed zero, subnormals, NaN policy, and
  saturation;
- odd logical dimensions and padded blocks;
- rowwise/columnwise transpose equivalence where declared;
- source-to-prepared layout equivalence; and
- checked overflow for all physical extent calculations.

Fixtures are deterministic serialized component tensors, not hand-waved shape
probes.

### 13.2 Operation tests

For each supported dtype and backend class:

- embedding row selection;
- decode-shaped GEMV/small-M linear;
- prefill-shaped GEMM;
- gate/up and down expert projections;
- grouped uneven/empty expert routing;
- bias and fused activation epilogues; and
- sharded logical slices

are compared with an independently decoded F32/F64 oracle. Tolerances are
declared per operation and accumulation mode. The oracle begins from the exact
stored NVFP4 values, not from an unavailable pre-quantization model, so kernel
error and quantization error remain distinguishable.

### 13.3 Model tests

The complete selected GPT-OSS model must compare:

- fixed decoded parameter samples;
- embedding and representative layer outputs;
- attention sinks and alternating attention windows;
- router logits, selected experts, expert outputs, and combined MoE output;
- final logits plus explicit-greedy and fixed-seed stochastic tokens;
- Harmony channel structure and incremental decoding; and
- a fixed set of end-to-end prompts

with a trustworthy implementation of the same artifact.

Where a dense source/baseline exists, a separate quality report measures the
artifact's quantization delta. It is not used to excuse a mismatch between NML
and the artifact oracle.

## 14. Performance and memory contract

Every result separates:

```text
artifact download
manifest validation
host read
device upload
layout preparation
XLA compilation
first execution
steady prefill
steady decode
```

Memory reports separately:

- checkpoint bytes;
- host staging and source storage;
- persistent source component buffers;
- persistent prepared component buffers;
- executable bytes;
- temporary workspace;
- activations;
- KV cache; and
- peak device memory.

The non-negotiable memory invariant is that no accepted path retains a complete
FP16/BF16 copy of an NVFP4 parameter. Temporary tile-local expansion is
expected and accounted as workspace.

Performance workloads include:

- decode M=1 and representative continuous-batch M values;
- short, medium, and long prefill buckets;
- ordinary attention plus projection layer composition;
- grouped MoE with realistic and adversarial expert distributions;
- the complete GPT-OSS model; and
- CPU x86-64, CPU AArch64 when available, local SM75, rented SM8x, rented SM90,
  and rented SM100+.

Baselines are the best truthful dense or dequantized path available on the same
device, with identical output semantics. A pre-Blackwell optimized route is
retained only if it improves at least one declared product workload without an
unacceptable regression in the others. If an emulation route is slower than a
dense resident model but the dense model does not fit, report that tradeoff
plainly; do not call it an acceleration.

Native Blackwell acceptance additionally proves the expected compact resident
memory and verifies native instruction use. Peak marketing throughput is never
used as evidence.

## 15. Failure policy

NML hard-fails with a diagnostic when it encounters:

- an unknown artifact revision or manifest digest;
- missing, extra, duplicate, or mismatched component records;
- an unsupported E2M1/E4M3 variant or scale convention;
- inconsistent payload/scale/global-factor extents;
- nonzero or invalid padding where zero padding is required;
- logical shards that cannot be represented safely;
- an unprepared layout bound to an executable requiring another layout;
- a heterogeneous CUDA capability set for one executable;
- a device/shape combination with neither an accepted native nor emulation
  kernel;
- a kernel compile/launch error; or
- numerical non-finiteness outside the representation's declared policy.

There is no heuristic retry that converts the whole parameter to BF16. A
fallback must be an explicit accepted path over the same compact representation
and must be visible in structured execution identity.

## 16. Build and execution topology

Bazel owns all representation libraries, compiler bindings, CPU kernels, CUDA
kernels, product binaries, and contracts. The existing rules remain:

- BuildBuddy compiles CPU/CUDA targets and runs CPU or GPU-independent tests.
- Local SM75 executes the permanent SM75 device contracts directly or through
  the immutable OCI runner.
- RunPod executes unchanged immutable images on SM8x, SM90, and SM100+.
- CUDA execution is never inferred from a successful remote compile.
- The relative Bazel cache remains `../nml-bazel-cache`.

Any native CUDA library added for NVFP4 must be built and packaged through the
same hermetic CUDA runtime contract. Python reference tools may be used outside
the product graph to generate oracle artifacts, but product loading and
execution remain Rust/XLA/PJRT owned.

## 17. Implementation sequence

The work is ordered to settle representation correctness before optimizing
kernels, while producing permanent product components at every step.

### Phase A: select and freeze the artifact

- Audit candidate GPT-OSS 20B NVFP4 artifacts and select exactly one.
- Commit its complete immutable manifest and representation specification.
- Pin an independent loader/oracle and fixed decoded samples.
- Decide whether the artifact is accepted as distributed or NML owns one
  deterministic conversion from a named dense source.

Exit: every bit required to reconstruct every logical weight is unambiguous.

### Phase B: replace the parameter/storage model

- Introduce validated artifact records and internal storage encodings.
- Introduce `Parameter`/`LoadedParameter` and one dense-or-quantized structural
  traversal.
- Replace dense-only `TensorStore` and executable binding assumptions.
- Add representation-aware component loading, accounting, and sharding.
- Delete superseded one-record/one-buffer compatibility paths instead of
  preserving aliases.

Exit: dense parameter execution remains supported through the new one-component parameter
case, and an NVFP4 parameter loads without expansion.

### Phase C: establish exact CPU execution

- Implement exhaustive codecs and blockwise CPU reference contraction.
- Add embedding, linear, and grouped expert semantics over `Parameter`.
- Implement optimized x86-64 and portable/AArch64-capable CPU tiling.
- Compare every operation with independent artifact decoding.

Exit: an NVFP4 GPT-OSS block executes correctly on CPU with bounded temporary
memory, and CPU performance is measured.

### Phase D: implement pre-Blackwell CUDA

- Add typed Triton E2M1/scale/dot support required by the pinned compiler.
- Implement and tune SM8x and SM90 fused W4A16 linear and grouped MoE paths.
- Implement the SM75 CUDA custom-call path and run it locally.
- Add ordinary and grouped projection numerical/performance contracts on real
  SM75, SM8x, and SM90 devices.

Exit: compact weights execute through normal dispatch on every retained
pre-Blackwell CUDA class without persistent dense expansion.

### Phase E: implement native Blackwell

- Prove the pinned XLA/Triton pipeline can compile `tt.dot_scaled` for SM100.
- Implement versioned native payload/scale preparation and native kernels.
- Run linear and grouped MoE contracts on a real SM100+ device.
- Verify native instructions and compare against the CPU oracle.

Exit: the native path is real, numerically accepted, and separately identified
from emulation.

### Phase F: complete GPT-OSS 20B

- Implement the exact GPT-OSS configuration and tensor mapping from the
  selected manifest.
- Add attention sinks, clamped/residual SwiGLU, alternating dense/window
  attention, YaRN, tokenizer, and Harmony semantics.
- Execute the complete model on CPU where feasible and on retained CUDA
  capability classes, using rented memory capacity where required.
- Compare end-to-end outputs and record phase-separated memory/performance.

Exit: the selected GPT-OSS 20B NVFP4 artifact generates accepted output without
hidden dense weights.

### Phase G: sharding and product closure

- Define logical tensor/expert-parallel placement over representation-aware
  shards.
- Run homogeneous multi-GPU CUDA Shardy and collectives.
- Make representation identity part of executable/prepared-weight/cache keys.
- Publish the immutable product artifact and retain structured device evidence.

Exit: NVFP4 is a supported NML product vertical rather than a single-device
demonstration.

## 18. Acceptance checklist

NVFP4 may be marked complete only when every applicable item is true:

- [x] One GPT-OSS 20B artifact and immutable manifest are selected.
- [x] Payload, scale, global factor, layout, padding, and transform algebra are
  completely specified.
- [x] The parameter/storage redesign replaces dense-only assumptions without a
  compatibility fork.
- [x] Packed host and device storage never retains a full dense copy.
- [x] CPU codec, embedding, linear, and grouped MoE execute correctly.
- [ ] CPU has a measured optimized path on x86-64 and a retained AArch64 design.
- [ ] The current SM75 exact-decode/GEMV adapter launches and passes on a real
  device. Its established activation-boundary predecessor passed locally; the
  refactored source is compile-gated but is not reused as evidence for SM8x.
- [x] SM8x fused emulation launches and passes on a real suitable GPU. The
  immutable full contract image at digest
  `sha256:17040fd252bac543bb3b02e9abc253d309d05a7b64cf6ee7b8c6cc8b64c426b4`
  passed on an RTX A6000 (SM86) in RunPod lease
  `1c30d217-cd03-4a61-bd8c-2d5ce10d1eb3`.
- [x] SM90 fused emulation launches and passes on a real suitable GPU. The same
  digest and contract selection passed on an H100 80GB HBM3 (SM90a) in RunPod
  lease `09de1c2c-6304-44f5-bb03-81d9d64cf4c1`.
- [ ] SM100+ native execution launches, passes, and proves native instructions.
- [x] Ordinary and grouped projection performance is measured by phase and
  shape family on local SM75, rented SM86, and rented SM90. The immutable
  performance image digest is
  `sha256:4f4619f040bd8c59549b90e5b3606c930bc063389aee4e280a25853c15fdf0ff`;
  detailed GPT-OSS-sized phase baselines and lease identities are retained in
  `TASKS.md`.
- [x] Logical Shardy placement maps correctly to physical payload/scale shards.
- [ ] The complete GPT-OSS 20B model matches the independent artifact oracle.
- [ ] Checkpoint, resident, workspace, compilation, prefill, and decode costs
  are reported separately.
- [ ] Every unsupported artifact/layout/device combination fails clearly.

Until then, status should name the exact completed layer—for example,
"NVFP4 checkpoint parsing," "SM90 W4A16 emulation," or "native SM100 linear"—
rather than claiming general NVFP4 support.
