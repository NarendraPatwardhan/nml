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
      assignment, Shardy/GSPMD configuration, and backend-specific settings.
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
