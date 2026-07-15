# NML implementation milestones

This file is the live implementation and acceptance ledger for NML's ordered
product milestones. The first milestone established this complete execution
path:

```text
typed Rust graph -> verified StableHLO/MLIR -> XLA compile options
                 -> PJRT compilation -> owned buffers -> CPU/CUDA execution
```

A checkbox is marked complete only after its stated product contract exists and
the corresponding acceptance target passes. Partial implementations, generated
bindings without a safe owner, and compilation without execution remain
unchecked.

## 1. Canonical types and tensor metadata

- [x] Add `nml-types` with the canonical scalar set: bool; signed and unsigned
      8/16/32/64-bit integers; F16, BF16, F32, F64; C64 and C128.
- [x] Add stable host representations for F16, BF16, C64 and C128, including
      explicit size and alignment contracts.
- [x] Add dtype classification, byte width, alignment, StableHLO spelling, and
      rejection of unordered operations for complex values.
- [x] Add rank-8 bounded shapes with checked element and byte counts, logical
      axis tags, partition metadata, and explicit physical layouts.
- [x] Keep MLIR `index` outside `DType` and make it impossible to use as a PJRT
      tensor element type.
- [x] Pass the complete dtype, complex-layout, shape-overflow, rank, layout, and
      metadata contract suite.

## 2. Pinned XLA compiler graph

- [x] Lift the XLA module graph at commit
      `41370d1124c74d7b93a207136a636d8c631cbed9`, following ZML's dependency and
      patch structure for the compiler portions NML consumes.
- [x] Obtain PJRT, MLIR C API, StableHLO, Shardy, XLA compiler APIs, and generated
      option schemas from that single pinned source graph.
- [x] Remove the standalone PJRT-header repository after all ABI bindings use
      headers from the XLA source graph.
- [x] Preserve the current CPU/CUDA plugin packages, supported hosts, CUDA
      runtime closure, and hard-error capability policy.
- [x] Pass clean-cache default, CPU, and CUDA dependency-resolution and Bazel
      graph contracts.

## 3. Rust MLIR ownership

- [x] Add `nml-mlir-sys` bindings for the exact MLIR C API surface used by NML.
- [x] Add `nml-mlir` RAII ownership for contexts, modules, regions, blocks,
      operations, types, attributes, locations, diagnostics, and pass managers.
- [x] Register Func, Arith, StableHLO, and Shardy dialects.
- [x] Implement compiler-internal MLIR index types, index constants, and signed
      and unsigned index casts without adding an index runtime dtype.
- [x] Map every canonical `DType`, including C64/C128, to MLIR tensor element
      types.
- [x] Add permanent builders for functions, returns, constants, `dot_general`,
      complex/real/imaginary operations, and StableHLO FFT operations.
- [x] Verify modules and support deterministic textual and bytecode
      serialization with owned diagnostic errors.
- [x] Pass ownership, invalid-module, complex, index, and serialization
      contracts.

## 4. PJRT execution ownership

- [x] Refactor plugin/client ownership so every dependent PJRT object keeps the
      necessary API and library state alive independent of Rust lexical borrows.
- [x] Add owned `Event`, `Memory`, `Buffer`, `Executable`, and
      `LoadedExecutable` wrappers.
- [x] Add checked host-to-device and device-to-host transfers for scalar, empty,
      and multidimensional buffers.
- [x] Add executable metadata, addressable-device discovery, execute options,
      argument/result ownership, readiness, deletion, and completion errors.
- [x] Preserve useful plugin diagnostics while destroying every PJRT error and
      owned object exactly once.
- [x] Pass lifecycle, transfer, metadata, failure, CPU, and CUDA runtime
      contracts through the same safe Rust API.

## 5. XLA compile options

- [x] Add `nml-xla-sys` bindings to the pinned generated upb/protobuf option
      representations used by ZML.
- [x] Add a safe compile-options model for replicas, partitions, device
      assignment, mandatory Shardy partitioning, and backend-specific settings.
- [x] Negotiate the StableHLO target version with the pinned compiler and retain
      serialized buffers for the complete PJRT compile call.
- [x] Preserve ZML's CUDA latency-hiding scheduler override and PJRT device
      capability discovery, then apply NML's hard unsupported-GPU policy.
- [x] Reject invalid topology, assignment, partition, and backend combinations
      before crossing the PJRT ABI.
- [x] Pass deterministic serialization and CPU/CUDA compilation contracts.

## 6. Typed graph and permanent execution contracts

- [x] Add `nml-ir` with a scoped compilation context, deterministic symbols,
      programs, inputs, outputs, and typed symbolic tensors.
- [x] Validate dtypes, ranks, layouts, contracting axes, batch axes, and result
      shapes before emitting MLIR.
- [x] Implement `dot_general`, two-dimensional `matmul`, complex construction,
      and real/imaginary extraction.
- [x] Execute an F32 `[3, 5] x [5, 4]` matmul on CPU and CUDA and compare each
      result with the Rust reference using
      `abs(actual - expected) <= 1e-4 + 1e-4 * abs(expected)`.
- [x] Execute a C64 construction/real/imaginary round trip on CPU and CUDA.
- [x] Exercise GPU custom-call registration through the real CUDA plugin
      initialization path.
- [x] Pass deterministic-IR, pre-emission validation, CPU execution, CUDA
      execution, numerical, and unsupported-GPU contracts.

## Milestone acceptance

- [x] `bazel --output_user_root=../nml-bazel-cache test //...`
- [x] `bazel --output_user_root=../nml-bazel-cache test --config=cpu //...`
- [x] `bazel --output_user_root=../nml-bazel-cache test --config=cuda //...`
- [x] `git diff --check`
- [x] The complete typed graph -> StableHLO -> XLA -> PJRT -> CPU/CUDA numerical
      execution path passes on supported real hardware.

Numerical FFT execution, quantization, model loading, decoding, training, and
higher-level model APIs are subsequent product milestones. FFT builders and
complex compiler types do not constitute an end-to-end FFT support claim.

---

# Milestone 2: parameter, buffer, and checkpoint substrate

This milestone adds ZML's host-slice, persistent-buffer, structural-model, and
safetensors lifecycle in Rust. The implementation may use focused internal
types, but they do not automatically become public API.

The public surface remains comparable to ZML's. The root `nml` facade exposes
the existing `DataType`, `Shape`, and symbolic `Tensor` concepts together with
`Slice`, `Buffer`, `Exe`, `Bufferized<T>`, `Memory`, `Platform`, and `Sharding`.
Checkpoint details remain under `nml::safetensors` and loading remains under
`nml::io`, as in ZML. Executable argument/result helpers may live under
`nml::exe`, but are not additional root concepts.

Storage allocation implementations, mutable-view helpers, transfer guards,
events, individual device shards, buffer identities, binding tables, load
plans, load reports, and safetensors parser records remain private. A later
caller requirement and explicit API review are required before exposing one of
them.

A checkbox below is marked complete immediately after its stated product
contract and durable tests pass. Compile-only work, parser-only work, and a
partial backend remain unchecked.

## 1. Bazel graph and public facade

- [x] Import exact upstream Rust dependencies through `rules_rust`
      crate-universe direct specifications. Bazel remains the only build graph;
      no Cargo workspace or ZML-hosted source fork is introduced.
- [x] Add the repository's `rust_proc_macro` wrapper with the same stable Rust,
      edition, warning, unsafe-block, and supported-host policy as every other
      NML Rust target.
- [x] Add only the internal crate boundaries required by the implementation:
      host tensor storage, runtime buffer/executable ownership, structural
      derive generation, and checkpoint loading. Their implementation types
      stay private to the root facade.
- [x] Add one `nml` facade crate and route product callers through it. Keep raw
      PJRT, MLIR, XLA, loader-planning, and derive-support modules out of the
      ordinary public surface.
- [x] Pass default, CPU, and CUDA dependency-resolution and facade-visibility
      contracts.

## 2. `Slice`: typed and aligned host tensor storage

- [x] Implement the single public `Slice` abstraction for shaped host storage.
      It may either borrow caller storage or own a dtype-aligned allocation,
      while allocation strategy and mutability machinery remain private.
- [x] Record and validate shape, physical layout, byte offset, byte strides,
      backing extent, mutability, and byte order. Scalars, empty dimensions,
      sub-slices, transposed views, and negative strides must use checked
      address arithmetic.
- [x] Implement contiguity detection, typed reads/writes, strided copies, and
      explicit dense materialization. Dtype, alignment, bounds, mutability, and
      endian mismatches are diagnostic errors.
- [x] Add correct F16 and BF16 conversion helpers for checkpoint fixtures and
      independent host reference calculations.
- [x] Replace raw byte vectors in ordinary buffer transfer APIs with `Slice`.
      Raw bytes remain available only at explicit FFI or serialization
      boundaries.
- [x] Pass permanent alignment, offset, stride, layout, endian, scalar, empty,
      overflow, and ownership contracts.

## 3. PJRT memory and transfer completion

- [x] Extend the checked PJRT ABI for client/device memory discovery, default
      memory, memory kind and addressability, explicit layouts, uninitialized
      buffers, on-device sizes, memory/device copies, DMA mapping, and
      asynchronous host-to-device transfer managers.
- [x] Pass typed client creation options to PJRT, including ZML's CPU device
      count, without adding backend-specific public client types.
- [x] Preserve ZML's `Memory.Kind` choices: default, device, pinned host, and
      unpinned host. Unsupported requested memory is a structured error rather
      than a fallback.
- [x] Make transfer completion retain every borrowed host allocation until PJRT
      releases it. Transfer guards and individual PJRT buffers remain runtime
      implementation details behind public `Buffer` construction.
- [x] Implement `Buffer.toSlice`/`toSliceAlloc`, readiness, explicit deletion,
      device/memory copies, and deterministic destruction without returning raw
      byte vectors.
- [x] Pass the same lifecycle, memory selection, transfer, copy, error, and
      destruction contracts through the CPU and CUDA loaders.

## 4. `Sharding` and persistent `Buffer`

- [x] Port the CPU/CUDA-relevant ZML physical-mesh, logical-axis sharding,
      placement, canonical-device ordering, replicated placement, and tiled
      placement semantics.
- [x] Implement public `Buffer` as one logical tensor with private PJRT shards,
      its `Shape`, `Sharding`, `Platform`, and selected memory kind.
- [x] Upload contiguous and strided `Slice` views to the correct shard
      placement without materializing a persistent dense or converted copy.
- [x] Implement download/reassembly, shard readiness, byte accounting, explicit
      physical copy, and shared ownership for tied parameters. Sharing a
      buffer is distinguishable internally from allocating another device
      buffer.
- [x] Require unique ownership for donation and preserve one destruction point
      for shared/tied storage.
- [x] Exercise replicated and tiled placement through real multi-device CPU
      PJRT configuration and every available real CUDA device.
- [x] Pass placement arithmetic, shard reconstruction, replication,
      copy-versus-share, donation eligibility, and lifetime contracts.

## 5. `Exe`: named parameters and reusable arguments

- [x] Extend `Program` with deterministic named inputs and outputs, retaining
      whether each input is a parameter or an activation. Reject duplicate
      names before MLIR emission.
- [x] Add the StableHLO operations required by a conventional linear layer:
      general dot against `[out, in]` weights, elementwise addition, and bias
      broadcasting.
- [x] Have compilation return public `Exe`, which owns the loaded PJRT
      executable and keeps input/output shapes, shardings, names, and alias
      expectations private.
- [x] Follow ZML's `Exe.args`, `Arguments.set`, `Arguments.bake`, `Exe.results`,
      and `Exe.call` lifecycle. Argument/result helpers live in `nml::exe`, not
      as new root-level concepts.
- [x] Let baked parameter buffers be reused across calls while activations are
      replaced. Validate name, order, dtype, shape, layout, platform, sharding,
      missing arguments, and excess arguments before PJRT execution.
- [x] Implement ZML-style multi-device argument flattening and result
      reconstruction.
- [x] Preserve explicit output/input alias declarations. Parameters are
      non-donatable unless a later mutable-parameter API says otherwise;
      activation donation consumes uniquely owned storage.
- [x] Pass repeated-call, baked-parameter, result reconstruction, donation,
      output-alias, and invalid-binding contracts on CPU and CUDA.

## 6. Rust `Bufferized<T>` structural generation

- [x] Implement one public `NmlStruct` derive and the public
      `Bufferized<T> = <T as NmlStruct>::Buffers` mapping. Generated companion
      types and traversal support do not become root exports.
- [x] Support named and tuple structs, enums, nested derived values, `Option`,
      `Vec`, arrays, `Box`, tuples, and explicit skipped metadata.
- [x] Generate deterministic symbolic-tensor traversal, buffer traversal,
      argument flattening, result reconstruction, and checkpoint field paths.
- [x] Strip non-tensor metadata from bufferized structures and preserve the
      source structure wherever tensor fields exist, matching ZML's behavior.
- [x] Deduplicate repeated symbolic tensor identities during loading and bind
      every tied occurrence to the same underlying buffer.
- [x] Provide a manual trait implementation path for structures that cannot be
      derived without exposing another public traversal framework.
- [x] Pass nested-model, optional-bias, layer-vector, enum, skipped-metadata,
      deterministic-order, and tied-field contracts.

## 7. Safetensors registry and `TensorStore`

- [x] Implement `nml::safetensors::TensorRegistry` for a direct safetensors
      file, `model.safetensors`, and `model.safetensors.index.json` repositories.
      Parse only the bounded header through the upstream safetensors metadata
      representation.
- [x] Validate header size, JSON, shape products, dtype byte counts, offsets,
      file extent, duplicate names, index/shard agreement, missing shards, path
      containment, and integer overflow before device allocation.
- [x] Map safetensors encodings in NML's canonical dtype set and reject FP8,
      sub-byte, or otherwise unsupported encodings until their own product
      milestones.
- [x] Preserve safetensors' little-endian row-major contract and reject a
      non-native transfer rather than performing an implicit conversion.
- [x] Implement `nml::io::TensorStore` and its prefix/layer view behavior,
      binding checkpoint records to symbolic tensor identities as ZML does.
- [x] Resolve aliases deterministically: the primary name wins; with no primary,
      exactly one present alias is accepted; multiple present aliases are an
      ambiguity error.
- [x] Resolve tied weights through shared symbolic/storage identity so they are
      read and uploaded once.
- [x] Pass single-file, sharded-index, prefix, optional-field, alias,
      tied-weight, malformed-file, unsupported-dtype, and path-containment
      contracts.

## 8. Parallel checkpoint-to-buffer loading

- [x] Build and validate the complete unique-storage load plan before allocating
      device memory. Loader plan records and accounting stay private.
- [x] On CPU, read each unique tensor into an aligned `Slice`, upload it, wait
      for completion, and release staging storage immediately afterward.
- [x] On CUDA, port ZML's bounded double-buffered DMA path using mapped staging
      chunks, asynchronous transfer managers, reusable buffers, and completion
      events.
- [x] Dispatch file-order spans to tiled and replicated shards without rereading
      replicated bytes or retaining a full converted checkpoint.
- [x] Support bounded parallelism, staging-buffer count, chunk size, selected
      memory kind, and progress reporting through `nml::io.LoadOptions`, the
      single additional public configuration type already analogous to ZML.
- [x] Clean up every submitted transfer, event, staging allocation, and
      completed device buffer exactly once after any partial failure.
- [x] Keep allocation/read/upload counters private but inspectable by in-crate
      acceptance tests to prove deduplication and bounded staging memory.
- [x] Pass parallel loading, span dispatch, deduplication, bounded-memory,
      truncated-read, transfer-failure, and cleanup contracts.

## 9. FP16/BF16 linear-layer product contract

- [x] Define a derived linear structure with `[out, in]` weight and optional
      `[out]` bias, constructed through a `TensorStore` view and loaded as
      `Bufferized<Linear>`.
- [x] Generate real single-file and sharded safetensors fixtures for FP16 and
      BF16, both with and without bias, using the upstream serializer.
- [x] For each variant, load parameters once, compile once, bake the resulting
      buffers once, and execute at least three distinct activation inputs.
- [x] Run identical model, loading, binding, and execution code on CPU and
      CUDA.
- [x] Compute an independent F32 host reference, round to the output dtype, and
      require each result to be within four output-dtype ULPs with a `1e-5`
      absolute floor.
- [x] Prove through private runtime identity/accounting that each unique
      parameter is uploaded once, tied fields share storage, parameter storage
      remains unchanged across calls, and execution performs no checkpoint
      reads or parameter uploads.
- [x] Execute a separate real alias/donation contract that consumes a unique
      activation and returns the correctly aliased result through `Exe`.

## Milestone 2 acceptance

- [x] All added tests are permanent product contracts; no probe, smoke, spike,
      demo, compatibility-only, or temporary target remains.
- [x] `bazel --output_user_root=../nml-bazel-cache test //...`
- [x] `bazel --output_user_root=../nml-bazel-cache test --config=cpu //...`
- [x] `bazel --output_user_root=../nml-bazel-cache test --config=cuda //...`
- [x] The CUDA loading, parameterized execution, donation, and alias contracts
      pass on a real supported NVIDIA device.
- [x] The pushed revision's BuildBuddy workflow passes while reusing the remote
      cache.
- [x] `git diff --check`
- [x] Milestone 2 is complete only when FP16/BF16 safetensors parameters load
      once into persistent CPU/CUDA buffers, flow through `Bufferized<T>`, and
      execute repeatedly with correct results through the compact ZML-shaped
      public API.

---

# Milestone 3: Shardy-native execution and nonlinear graph foundations

This milestone makes partitioning part of the ordinary graph/runtime contract
before attention is introduced. It also adds the algebra, shape transforms,
and nonlinear operations required to express attention and genuine MLPs. The
public surface remains one `Shape`, `Tensor`, `Sharding`, `Buffer`, and `Exe`
model; compiler implementation records do not become product concepts.

## 1. Sharding model and compiler ownership

- [x] Remove the GSPMD selector and make Shardy the only XLA partitioner.
- [x] Replace positional tiled placement with logical mesh axes identified by
      `AxisTag`, retaining single-device and replicated placement.
- [x] Validate unique nonzero mesh axes, checked device products, referenced
      tensor axes, even partitioning, and exact platform device availability.
- [x] Pass the selected `Sharding` into compilation so one topology-independent
      `Program` can be compiled for single, replicated, or mesh execution.
- [x] Store the resolved topology in `Exe` and reject buffers whose platform,
      placement, shard count, or logical mesh differs before PJRT execution.
- [x] Resolve single execution as one replica/partition, replicated execution
      as one partition with one replica per device, and mesh execution as one
      replica with the mesh product as its partition count.
- [x] Generate deterministic device assignments and diagnostic hard errors for
      unavailable or inconsistent topologies.

## 2. Shardy MLIR integration

- [x] Bind the pinned Shardy C attribute API and add context-owned Rust wrappers
      for mesh, dimension-sharding, tensor-sharding, and per-value attributes.
- [x] Emit one deterministic `sdy.mesh` declaration for mesh programs and lower
      `Shape` partition metadata to function input/output `sdy.sharding` attrs.
- [x] Add an explicit graph operation for intermediate
      `sdy.sharding_constraint` placement.
- [x] Add a region-safe internal `sdy.manual_computation` builder for future
      custom kernels without exposing another root public abstraction.
- [x] Reject invalid and cross-context Shardy objects before module verification
      and keep textual and bytecode output deterministic.

## 3. Runtime placement and checkpoint loading

- [x] Derive tensor-local upload ranges from the logical mesh and each
      `Shape::partitions()` entry rather than positional partition factors.
- [x] Replicate tensor data across unused mesh axes and reject duplicate or
      ambiguous consumption of a logical mesh axis.
- [x] Reassemble globally ordered host tensors from partitioned PJRT buffers.
- [x] Preserve direct checkpoint-to-local-shard loading without a second full
      persistent host tensor.
- [x] Accept tiled executable arguments/results only when their resolved
      placement exactly matches the compiled topology.
- [x] Preserve replicated loading and repeated-execution behavior and pass
      permanent placement, mismatch, cleanup, and reconstruction contracts.

## 4. Primitive algebra

- [x] Add owned tensor constants and typed rank-zero scalars whose storage
      remains valid through MLIR construction.
- [x] Add subtraction, multiplication, division, minimum, maximum, and negation.
- [x] Add equality, inequality, ordered comparisons, boolean selection, and
      dtype conversion; identical conversion reuses the symbolic value.
- [x] Follow ZML's elementwise rule: exact shape/metadata match or explicit
      rank-zero scalar broadcasting, never implicit NumPy rank expansion.
- [x] Preserve logical axes and partitions and reject unsupported dtype or
      metadata combinations before MLIR emission.
- [x] Expose the operations through `ProgramBuilder` and `TensorStore` without
      public operation, comparison, or activation enums.

## 5. Compiled reshape and transpose

- [x] Add StableHLO reshape with an explicit output `Shape`, checked equal
      element counts, and mesh-valid output partition metadata.
- [x] Express an explicit reshape partition change as a Shardy constraint
      rather than silently reinterpreting local storage.
- [x] Add StableHLO transpose with a complete unique permutation, moving
      dimensions, semantic axis tags, and partitions together.
- [x] Define compiled transpose as a materialized row-major result while
      retaining separate zero-copy host-view physical-layout semantics.
- [x] Pass permanent attention-head reshape/transpose contracts for
      `[batch, sequence, heads, head_dim]` layouts.

## 6. Unary math and activations

- [x] Add `exp`, `log`, `sqrt`, `rsqrt`, `tanh`, `sin`, and `cos`, preserving
      the complete input shape metadata.
- [x] Add ReLU, StableHLO logistic sigmoid, SiLU, ZML-compatible
      tanh-approximate GELU, leaky ReLU, and quickGELU as graph compositions.
- [x] Verify FP32, FP16, and BF16 behavior against independent host references
      with dtype-appropriate tolerances.

## 7. Integrated product contracts

- [x] Load and execute a checkpoint-backed two-layer nonlinear MLP on CPU and
      CUDA rather than treating individual emitted operations as completion.
- [x] Execute the MLP on CPU in single, replicated, and logical-mesh modes.
- [x] Execute a partitioned dot whose contracting axis requires
      compiler-inserted communication and compare its reconstructed result with
      the unsharded reference.
- [x] Exercise the same Shardy-aware graph-building path on single-device CUDA.
- [x] Verify repeated execution, parameter ownership, constants, comparisons,
      selection, conversion, reshape, transpose, and activations through PJRT.

## Milestone 3 acceptance

- [x] All tests are permanent product contracts; no probe, smoke, spike, demo,
      compatibility-only, or temporary target remains.
- [x] BuildBuddy executes `bb test --config=buildbuddy --config=cpu
      //:cpu_contracts` with compile actions on RBE.
- [x] BuildBuddy executes the CUDA-configured contracts that do not own the
      packaged runtime or a physical device with `bb test
      --config=buildbuddy --config=cuda //:cuda_remote_contracts`.
- [x] BuildBuddy compiles the exact CUDA contract binaries without assembling
      their runtime data and populates the authenticated action cache with `bb
      build --config=buildbuddy --config=cuda //:cuda_contract_binaries`.
- [x] The NVIDIA host assembles the pinned system-driver runtime and executes
      only those cached contracts with `bb test --config=buildbuddy
      --config=cuda --cache_test_results=no //:cuda_device_contracts`.
- [x] The real CUDA contracts pass on a supported NVIDIA device and unsupported
      GPU capabilities remain hard diagnostic errors.
- [x] The pushed revision's BuildBuddy workflow passes while reusing the remote
      cache.
- [x] `git diff --check`
- [x] Milestone 3 is complete only when Shardy-partitioned execution and the
      checkpoint-backed nonlinear model work through the compact public API on
      the applicable CPU/CUDA targets.

---

# Milestone 4: portable attention semantics and paged KV state

This milestone ports ZML's real tensor/StableHLO attention foundation and adds
the portable paged path required by D-032. CPU and CUDA consume the same graph
semantics. The root API gains only the small attention configuration and cache
descriptions that a model author must provide; operation records, reduction
regions, loop state, page traversal, and backend dispatch remain internal.

## 1. Attention prerequisite operations

- [x] Add dimension-aware `iota`, concatenate, static slice, dynamic slice,
      dynamic update slice, and general gather operations with checked shapes,
      indices, layouts, logical axes, and partitions.
- [x] Add generic single-input sum and maximum reductions with correctly typed
      identities, region ownership, retained non-reduced metadata, and explicit
      accumulation dtype where required.
- [x] Add numerically stable softmax and RMS normalization composites, including
      FP32 accumulation for FP16/BF16 and conversion back to the input dtype.
- [x] Preserve ZML's explicit broadcasting rules and reject implicit rank
      expansion, invalid gather dimension numbers, out-of-range static slices,
      and unsupported reduction dtypes before MLIR emission.
- [x] Expose the required operations through `TensorStore` without exporting
      public operation, reduction, or gather-configuration enums at NML's root.

## 2. StableHLO control-flow ownership

- [x] Add RAII builders for `stablehlo.reduce`, `stablehlo.return`,
      `stablehlo.while`, and loop/reduction regions using the existing owned
      `Region`, `Block`, `Operation`, and `Value` model.
- [x] Represent attention loop state without allowing foreign tensors,
      cross-context values, mismatched result types, or unterminated regions.
- [x] Carry a bounded runtime loop index and tensor state through
      `stablehlo.while`; general paged attention must not scale graph size with
      maximum page count.
- [x] Pass deterministic text, bytecode, verification, invalid-region, and XLA
      compilation contracts for reductions and loop-carried tensors.

## 3. RoPE, masks, and ordinary attention

- [x] Implement interleaved and sequential rotary embeddings from position
      tensors, with configurable base/scaling and checked even rotary width.
- [x] Implement causal, non-causal, and sliding-window masks from runtime query
      and key positions, using negative infinity plus ZML-compatible zero output
      for a completely masked row.
- [x] Implement scaled dot-product attention with FP32 score/softmax
      accumulation and input-dtype output.
- [x] Support MHA, GQA, and MQA without materializing repeated persistent KV
      heads; validate query-to-KV head divisibility and semantic head axes.
- [x] Support prefill, single-token decode, and multi-token decode through the
      same ordinary attention graph.

## 4. Persistent dense and paged KV state

- [x] Define the compact public cache description needed to allocate split K/V
      storage, page tables, sequence lengths, and compile-time capacity bounds.
- [x] Allocate dense and paged caches as ordinary persistent `Buffer` values on
      the selected platform and sharding, with no backend-specific public cache
      type.
- [x] Implement append/update graphs for dense and paged K/V storage using
      dynamic updates, returning aliasable cache outputs rather than allocating
      replacement storage on every decode step.
- [x] Validate page size, physical/logical capacity, page identifiers, sequence
      lengths, batch ownership, dtype, head geometry, platform, and sharding
      before execution.
- [x] Support deterministic truncate, rollback, and replay by updating logical
      lengths/page tables without copying unaffected K/V pages.

## 5. Portable blockwise paged attention

- [x] Traverse logical pages with bounded `stablehlo.while` and gather physical
      K/V pages directly from the page table.
- [x] Carry the online-softmax running maximum, rescaled denominator, and value
      accumulator in FP32 across pages, including fully masked pages without
      producing NaNs.
- [x] Apply runtime tail-token, causal, non-causal, and sliding-window masks
      before page-local reductions.
- [x] Support MHA, GQA, and MQA head mapping, prefill, single-token decode, and
      multi-token decode without expanding persistent KV heads.
- [x] Prove through graph and allocation contracts that the product path does
      not materialize a contiguous logical KV cache or the complete attention
      score matrix.
- [x] Keep the same StableHLO implementation executable on CPU and CUDA as the
      correctness/performance path and CUDA fallback required by D-032.

## 6. Integrated product contracts

- [x] Compare ordinary and paged attention against independent dense host math
      for FP32, FP16, and BF16 with dtype-appropriate tolerances.
- [x] Cover empty context, partially occupied final pages, non-contiguous and
      shared physical pages, boundary capacities, invalid page identifiers,
      and invalid sequence lengths.
- [x] Cover causal and non-causal attention, sliding windows, MHA/GQA/MQA,
      prefill, single-token decode, and multi-token decode.
- [x] Execute successive cache updates, repeated decode calls, truncation,
      rollback, and replay while proving storage identity and unaffected-page
      contents remain stable.
- [x] Execute the same checkpoint-backed attention block and cache lifecycle on
      CPU and real CUDA through the compact public API.
- [x] Exercise Shardy-compatible head/batch placement without introducing a
      second attention or cache representation.

## Milestone 4 acceptance

- [x] All tests are permanent product contracts; no probe, smoke, spike, demo,
      compatibility-only, or temporary target remains.
- [x] BuildBuddy executes the CPU and CUDA-remote contract suites and compiles
      the exact CUDA runtime contract binaries without assembling CUDA runtime
      data remotely.
- [x] The NVIDIA host executes the cached ordinary/paged attention and cache
      contracts on the real supported GPU.
- [x] The pushed revision's BuildBuddy workflow passes while reusing the remote
      cache.
- [x] `git diff --check`
- [x] Milestone 4 is complete only when ordinary attention, portable blockwise
      paged attention, and persistent rollback-capable KV state execute through
      the compact public API on CPU and CUDA without dense-cache materialization.

---

# Pre-Milestone 5: model-enabling parity closure

This bounded slice closes the inexpensive ZML capabilities that are already
natural compositions of NML's typed StableHLO substrate. It does not introduce
a model hierarchy, sampling policy object, backend-specific dispatch, or a
reference-model package. Operations remain on the existing `ProgramBuilder`
and `TensorStore` surfaces so later CUDA kernels, richer reductions, and
sampling strategies can replace or reuse their lowering without changing model
code.

## 1. Coherent elementwise and reduction tail

- [x] Add typed absolute value, power, remainder, clamp, floor, and ceil with
      correct scalar broadcasting, complex-result dtype behavior, and
      pre-emission dtype validation.
- [x] Add reduction minimum, mean, and numerically stable log-sum-exp while
      preserving retained axis tags and partition metadata.
- [x] Extend the narrow MLIR builders only where a distinct StableHLO operation
      is required; composites remain explicit graph compositions so XLA can
      fuse them normally.
- [x] Add deterministic StableHLO and invalid-contract coverage for every new
      primitive and composite.

## 2. Embedding, normalization, and gated activations

- [x] Add token embedding as the checked rank-two vocabulary gather already
      used by ZML, accepting any supported integer index tensor without adding
      a public embedding-layer type.
- [x] Add variance normalization, LayerNorm, and L2 normalization with F16 and
      BF16 accumulation in F32, explicit epsilon validation, and optional
      LayerNorm affine parameters.
- [x] Add SwiGLU and GeGLU as shape-safe composites of the existing activation
      and multiplication operations; do not couple gating to a particular
      projection layout or checkpoint naming scheme.
- [x] Execute embedding, normalization, and gating numerical contracts for F32,
      F16, and BF16 on every applicable CPU/CUDA product backend.

## 3. Argmax and compiled greedy selection

- [x] Add an axis-reducing argmax that returns both values and indices, chooses
      I32 indices unless the reduced dimension requires I64, selects the first
      index on ties, and propagates the first encountered NaN as ZML does.
- [x] Represent argmax as a general two-result StableHLO reduction rather than
      a sampling-specific custom operation, leaving top-k, sorting, and
      stochastic sampling unconstrained.
- [x] Expose argmax through `TensorStore`; its index result is the compiled
      greedy-selection path without adding a premature sampling-policy type.
- [x] Execute numerical, tie, NaN, dtype, axis, metadata, CPU, and CUDA
      contracts.

## Acceptance

- [x] `rustfmt` passes for every changed Rust source.
- [x] BuildBuddy executes `bb test --config=buildbuddy --config=cpu
      //:cpu_contracts` with compile actions on RBE.
- [x] BuildBuddy executes `bb test --config=buildbuddy --config=cuda
      //:cuda_remote_contracts` and builds `//:cuda_contract_binaries`.
- [x] Applicable real-device CUDA contracts execute locally after their exact
      binaries have been populated through the remote cache.
- [x] `git diff --check`
- [x] The capability ledger is updated only for complete product families; a
      new operation does not overstate convolution, stochastic sampling,
      general scatter, or other unfinished ZML parity work.

---

# Milestone 5: CUDA FlashAttention and Triton paged attention

This milestone adds two independent CUDA acceleration mechanisms behind the
attention semantics completed in Milestone 4.  The portable StableHLO paths
remain the oracle and CUDA fallback. A kernel is hardware-validated only after
its numerical result and lifecycle behavior execute on compatible hardware;
parsing TTIR, compiling a target, or registering a symbol is never presented as
that validation. D-038 nevertheless lets the implementation milestone close
with those real-device runs explicitly deferred, provided the complete product
artifacts and unchanged future execution contracts continue to compile.

The pinned ZML snapshot is the architectural reference.  NML departs where
Rust needs explicit ownership and where D-025 requires original upstream
FlashAttention rather than ZML's hosted source fork.  Internal kernel types
stay crate-private; the public surface remains `nml::attention` plus the
existing tensor/cache operations.

## 1. Backend contracts and capability policy

- [x] Define one internal attention-backend decision with `Portable`,
      `CudaTriton`, `CudaFlash2`, and `CudaFlash3` implementations; do not add
      public per-kernel parameter or metadata families.
- [x] Read CUDA compute capability from the PJRT device description and make
      feature support explicit: upstream FA2 requires SM80+, FA3 requires
      SM90, and the pinned XLA Triton backend decides its own supported CUDA
      range.  An explicitly requested unsupported backend returns a hard,
      diagnostic error; automatic selection may use the portable path required
      by D-032.
- [x] Keep all shape, dtype, head-ratio, causal/window, layout, page-table, and
      alias validation above backend dispatch so every implementation receives
      the same already-validated semantic contract.
- [x] Record the important reference departure: ZML routes every non-SM90 CUDA
      device to its hosted FA2 fork, while NML's original-upstream FA2 cannot
      run on the local SM75 GTX 1660 Ti, and the pinned XLA compiler also
      rejects Triton below SM80.  SM75 therefore validates the portable CUDA
      fallback and capability diagnostics; Triton/FA2/FA3 need compatible
      remote hardware.

## 2. Pinned TTIR ownership and bindings

- [x] Add the Triton dialect from the already pinned XLA dependency graph to
      NML's MLIR C boundary.  Bind only dialect registration, pointer/tensor
      descriptor types, and enum attributes actually needed by retained CUDA
      kernels.
- [x] Extend `nml-mlir` with safe context-bounded TTIR handles and generic
      operation construction without exposing raw `Mlir*` objects or allowing
      TTIR operations in the long-lived StableHLO program context.
- [x] Create each kernel in an isolated non-threaded MLIR context, register the
      `tt`, `arith`, `math`, `scf`, and `cf` dialects it needs, verify the
      finished module, serialize deterministic textual TTIR, and destroy the
      complete context before ordinary graph compilation continues.
- [x] Add permanent contracts for pointer/type attributes, invalid ownership,
      malformed operations, deterministic serialization, verification
      failures, and repeated context creation/destruction.

## 3. Private Rust Triton kernel substrate

- [x] Implement crate-private `DType`, `Value`, named argument declarations,
      and a builder covering the arithmetic, pointer, load/store, broadcast,
      reduction, dot, range, program-id, and structured-control-flow operations
      used by unified attention.  Dtypes are limited to NML's retained set.
- [x] Implement typed kernel specifications with ordered named inputs and
      outputs, explicit result shapes, output/operand aliases, three-dimensional
      launch grids, warp/stage counts, and deterministic configuration errors.
- [x] Lower a typed kernel invocation to `stablehlo.custom_call` target
      `__gpu$xla.gpu.triton` with typed-FFI backend configuration, row-major
      operand/result layouts, embedded verified TTIR, and validated aliases.
- [x] Add permanent builder and custom-call contracts that exercise every
      operation family used by attention, reject cross-context values, malformed
      TTIR, and bad launch/alias contracts, verify and serialize the exact
      TTIR-bearing StableHLO artifact, and compile the permanent CUDA contract
      binary against the pinned PJRT plugin.
- [ ] `DEFERRED` Run XLA's device-specific compilation of those exact TTIR
      artifacts on SM80+ hardware. The test binary and its Triton source graph
      must continue to build before the RunPod execution gate is scheduled.

## 4. Unified Triton paged attention

- [x] Port ZML's CUDA-relevant 2D unified paged-attention kernel, retaining
      blockwise online softmax, causal and sliding-window masks, MHA/GQA/MQA
      head mapping, arbitrary valid page tables, padded head dimensions, and
      FP32 accumulation for FP16/BF16 inputs.
- [x] Port the split-K 3D attention kernel and segment-reduction kernel with
      explicit intermediate shapes and no logical-KV or full-score-matrix
      materialization.
- [x] Port the CUDA launch-selection policy (2D prefill and sufficiently large
      decode; 3D split-K otherwise) without the excluded oneAPI specialization.
      Configuration choices are deterministic functions of validated geometry
      and CUDA device attributes.
- [x] Integrate Triton as the preferred CUDA paged-attention implementation
      while preserving the same cache storage, page table, update, rollback,
      replay, and portable fallback contracts from Milestone 4.
- [x] Compile the permanent numerical contract covering prefill, single-token
      and multi-token decode, mixed sequence lengths, page boundaries, shared
      and non-contiguous pages, sliding windows, MHA/GQA/MQA, repeated
      execution, and both 2D and 3D launch paths. The same binary selects the
      portable fallback on SM75 and the private optimized implementation from
      the real device capability.
- [ ] `DEFERRED` Execute that unchanged contract's Triton 2D and split-K paths
      on rented SM80+ RunPod hardware.

## 5. Original-upstream FlashAttention integration

- [x] Pin an original Dao-AILab FlashAttention revision and its transitive
      CUTLASS inputs by immutable digest.  Build only forward inference kernels
      and supported FP16/BF16 head dimensions through Bazel; do not introduce
      PyTorch, Python packaging, a prebuilt wheel, or ZML's hosted fork.
- [x] Carry a small audited C ABI adapter as a local NML source file.  It owns
      no tensors, accepts explicit dimensions/strides/stream, translates to
      upstream FA2/FA3 parameter records, reports launch/configuration errors,
      and contains no model or dispatch policy.
- [x] Implement Rust-side typed XLA FFI handlers and process-lifetime
      registration through the existing PJRT GPU custom-call extension.
      Registration is idempotent per loaded plugin and handler code outlives
      every executable that may call it.
- [x] Lower ordinary dense attention to FA2 on SM80-SM89 and FA3 on
      SM90, including causal/sliding-window behavior, GQA/MQA, workspace/result
      aliases, and deterministic rejection of unsupported dtypes, head sizes,
      layouts, or compute capabilities.
- [x] Integrate upstream-supported paged prefill/decode variants only where
      their semantics cover NML's page-table contract; configurations not
      covered by upstream remain on Triton rather than acquiring a second cache
      representation or unaudited downstream patch.
- [x] Compile and link the unchanged numerical/lifecycle contract with the
      original-upstream FA2 SM80 and FA3 SM90a products, including every dense
      and supported paged shape selected by the future hardware runs.
- [ ] `DEFERRED` Execute FA2 on rented SM80+ RunPod hardware and FA3 on rented
      SM90 RunPod hardware. Remote compilation remains a required build
      condition and is not recorded as runtime validation.

## 6. Integrated product and failure contracts

- [x] Keep one device-polymorphic numerical contract that compares portable
      CPU, portable CUDA, Triton CUDA, and FlashAttention CUDA against
      independent dense host math for every mutually supported retained dtype
      and geometry with dtype-appropriate tolerances. CPU and SM75 branches
      execute now; optimized branches compile now and execute under the D-038
      deferred hardware gate.
- [x] Prove in-place cache updates and declared output aliases preserve
      unaffected pages, rollback/replay state, and repeated-execution behavior
      without cache reallocation. Keep its I32 index geometry eligible for the
      future Triton run instead of accidentally validating only the portable
      fallback.
- [x] Cover malformed TTIR/backend configuration, cross-context values,
      duplicate registration, unsupported SM/dtype/head geometry, and
      upstream adapter argument errors with stable failures. Registration
      extension absence remains a hard platform-construction error; automatic
      dispatch may use portable semantics, but no public or test-only backend
      selector is introduced.
- [ ] `DEFERRED` Exercise real FA2/FA3/Triton kernel-launch failures and verify
      their propagated diagnostics on compatible RunPod hardware; a launch
      failure cannot be manufactured honestly on the local unsupported GPU.
- [x] Keep CPU and portable CUDA product contracts independent of external
      FlashAttention packaging so unsupported hosts build and use NML without
      loading a CUDA attention library. Keep the full distribution runtime and
      the lighter system-driver runtime under separate package contracts.

## Milestone 5 acceptance

- [x] `rustfmt` passes for every changed Rust source.
- [x] The authenticated `bb` coordinator executes `bb test
      --config=buildbuddy --config=cpu //:cpu_contracts` with compile actions
      distributed through BuildBuddy RBE.
- [x] The authenticated `bb` coordinator executes `bb test
      --config=buildbuddy --config=cuda //:cuda_remote_contracts` and builds
      `//:cuda_contract_binaries` plus the exact FlashAttention/Triton product
      artifacts without running them on a GPU-less executor.
- [x] `//:cuda_package_contracts` proves the complete distributable runtime
      contains the pinned driver-compatibility overlay while the local-device
      runtime contains the same user-space closure and intentionally omits only
      that overlay.
- [x] The local SM75 device executes the portable CUDA runtime, linear,
      nonlinear, ordinary-attention, and paged-attention contracts. Permanent
      dispatch contracts prove Triton is excluded below SM80, while the local
      adapter contract proves explicit FA2/FA3 requests fail before launch
      with the expected capability diagnostics. Every device contract is an
      exclusive Bazel test so independent XLA clients never contend for the
      singleton physical GPU.
- [ ] `DEFERRED` The unchanged attention contract executes Triton and FA2 on
      rented SM80+ hardware and FA3 on rented SM90 hardware. These remain
      explicit hardware-validation gates and are not prerequisites for closing
      the implementation milestone under D-038.
- [x] `git diff --check`
- [x] Milestone 5 implementation is complete when every Triton/FlashAttention
      product artifact and unchanged future hardware contract compiles through
      the compact public API, and every hardware-independent numerical,
      ownership, packaging, lifecycle, and failure gate passes. Real SM80/SM90
      execution remains visibly deferred rather than being represented as
      completed validation.

## Deferred RunPod execution procedure

Use the same committed revision and the same `attention_cuda_contract_test`
binary on both machines; do not add a backend selector or a reduced hardware
test. On an SM80-SM89 machine it exercises FA2 dense/paged attention plus
Triton 2D and split-K through the F32/small-page cases. On an exact SM90
machine it exercises FA3 dense/paged attention plus the same Triton paths.

```text
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
bb build --config=buildbuddy --config=cuda //:cuda_contract_binaries
bb test --config=buildbuddy --config=cuda --cache_test_results=no \
  //crates/nml:attention_cuda_contract_test
```

Record the RunPod GPU model, compute capability, driver, invocation link, and
numerical result before checking the deferred SM80 or SM90 boxes. A successful
remote build, an SM75 fallback run, or an adapter capability diagnostic does
not satisfy either execution gate.

---

# Capability ledger

This high-level ledger remains at the end of this file while detailed work is
added to the milestone sections above. It tracks usable product capabilities,
not individual IR operations or implementation artifacts. Check an item only
when its applicable CPU/CUDA numerical, ownership, failure, and performance
contracts are permanent and passing. Parser support, emitted StableHLO,
successful compilation, registered symbols, or an unexecuted kernel do not by
themselves complete an item.

- [x] ReLU, GELU, SiLU, sigmoid, and other activations.
- [ ] Multiplication, subtraction, division, and the remaining elementwise
      operation families. Core arithmetic, ordering, absolute value, power,
      remainder, clamp, floor, and ceil are complete; logical, bitwise, and
      other retained ZML elementwise families remain.
- [x] Reshape and transpose in compiled graphs.
- [x] Reductions, normalization, and softmax, including sum/min/max/mean,
      log-sum-exp, argmax, RMSNorm, LayerNorm, and L2 normalization.
- [ ] Gather/scatter and embedding lookup. Gather, gather-slices, and token
      embedding are complete; general scatter remains.
- [ ] Convolution and pooling.
- [ ] Random-number generation.
- [ ] Sorting, top-k, and sampling. Argmax and compiled greedy selection are
      complete; sorting, top-k, RNG, and stochastic sampling remain.
- [x] Portable ordinary and blockwise paged attention, RoPE, and masks.
- [x] Persistent KV-cache allocation, page-table updates, paging, truncation,
      rollback, and replay without a persistent dense KV copy.
- [ ] CUDA FlashAttention and Triton kernels.
- [ ] MoE routing and grouped matrix multiplication.
- [ ] Quantization: W4A16, W8A8, and NVFP4. `DEFERRED` by D-028 until the
      CPU/CUDA ZML parity gate passes and the owner explicitly schedules it.
- [ ] Training or explicitly authored analytic backward graphs.
- [ ] Real distributed sharding and collectives.
