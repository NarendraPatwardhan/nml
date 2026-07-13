# NML first execution milestone

This file is the live implementation and acceptance ledger for NML's first
complete execution path:

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
