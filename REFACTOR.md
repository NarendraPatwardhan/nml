# GPT-OSS compilation, residency, and Triton ABI refactor

Status: implementation contract

This document records the correction to GPT-OSS lifecycle orchestration. It is
deliberately product-scoped: NML's framework already exposes abstract parameter
declarations, compilation, parameter upload, reusable executable slots, and
PJRT execution as independent mechanisms.

## Problem

The original GPT-OSS owner combined three different states in one
`CompiledModel`:

1. the abstract model and checkpoint contract;
2. a lazily populated map of compiled shape families;
3. the complete resident checkpoint.

`Generator::load` uploaded all 703 physical checkpoint components. The first
request then selected its prefill and cache buckets and invoked XLA from
`generate`. This reversed the lifecycle used by ZML and made production
compilation/autotuning occur while the complete model was already resident on
the accelerator. Individual kernel contracts did not exercise that ordering.

The failure was incorrectly attributed to the number of Triton kernels and to
possible compiler parallelism. Every isolated kernel path had passed its
applicable standalone contract, and serializing XLA compilation reproduced the
same failure. The first compiler backtrace identified where XLA terminated but
was insufficient by itself. Exact-binary disassembly plus the completed-call
register state subsequently proved a defective custom-call ABI.

### Split-K learned-sink root cause

The split-K paged-attention producer intentionally excludes learned sinks: each
producer computes independent KV segments, and the segment reduction applies
the sink correction exactly once. Its TTIR function therefore had 18 input
pointers and three output pointers. The StableHLO lowering accidentally reused
a shared operand list containing the sink, describing 19 inputs and three
outputs.

XLA creates kernel argument metadata from the StableHLO operands and results.
After Triton compilation it annotates the resulting LLVM function arguments
without first checking that the function has the same visible arity. At
zero-based argument 21, the malformed 22-buffer call indexed one past NML's
21-argument TTIR function and `llvm::Value::setName` interpreted the LLVM
function object as an argument. The asynchronous completion worker was merely
where this deterministic mismatch became undefined behavior; compiler
parallelism and LLVM-context lifetime were not the cause.

ZML avoids this class of failure because its declared input/output record also
drives TTIR argument declaration. Its split-K kernel retains a fixed sink slot
and passes a dummy value while sink semantics are disabled. NML keeps its
different, cleaner semantic split—the producer is genuinely sink-free and the
reducer owns the sink—but now carries the builder-authored function ABI with
the verified TTIR into `KernelSpec`. Count, order, pointer address space, and
element type must match the StableHLO tensor ABI before a custom call exists.

## ZML reference

ZML keeps model description, compilation, buffer loading, and session state
separate. Its LLM entry point performs these operations in this order:

```text
parse and declare model
        |
        v
compile prefill/decode executables
        |
        v
load model buffers
        |
        v
allocate session caches and execute
```

The referenced implementation explicitly loads buffers after compilation so
the accelerator remains available to XLA autotuning. Model buffers are passed
to reusable layer executables at invocation time; compiling an executable does
not require those buffers to be resident.

NML retains its existing improvements over that execution model: named and
representation-aware parameter slots, ownership-checked donation, asynchronous
PJRT dependency chaining, and finite reusable profile buckets. Those mechanisms
do not justify reversing compilation and residency.

## Required ownership states

GPT-OSS uses four non-overlapping lifecycle states:

```text
ModelDefinition
    immutable config, abstract Checkpoint, ParameterSet
        |
        | compile every configured profile
        v
ExecutionPlan
    normalized profiles, bounded component executables
        |
        | upload the complete checkpoint only after compilation succeeds
        v
ResidentModel
    ExecutionPlan plus LoadedCheckpoint
        |
        | select an existing profile
        v
RequestState
    tokens, positions, page table, K/V buffers, parser state
```

The transition order is part of correctness:

- `ModelDefinition` performs artifact/schema work and allocates no device
  parameter buffers.
- `ExecutionPlan` compiles all distinct prefill/decode families before the
  first parameter upload begins.
- `ResidentModel` is constructed only after the complete plan exists.
- `generate` may select and execute a plan but may not call XLA.
- An unsupported request fails with a capacity error rather than compiling a
  new family while the model is resident.
- A failed compilation drops already-created executables and never begins
  checkpoint upload. A failed upload drops the plan and any partial parameter
  owners through their ordinary ownership paths.

## Compilation profiles

Serving capacity is configuration, not an accidental property of the first
request. One public `CompilationProfile` declares:

- maximum prompt tokens;
- maximum total sequence tokens, including generated tokens.

Profiles normalize to the established power-of-two prefill bucket and
page-aligned power-of-two cache capacity. Duplicate normalized profiles and
shared decode families compile once. Profile validation happens before the
first compiler invocation and rejects zero, inverted, overflowing, or
out-of-context capacities.

At request time the engine selects the smallest compiled profile that covers
both the encoded prompt and the requested/required total sequence capacity.
The prompt is padded to that profile's prefill family. Absence of a fitting
profile is a configuration error and never a request-time compilation trigger.

This is the bounded-profile analogue of ZML's fixed `seqlen`: it preserves
compile-before-load while avoiding one unnecessarily maximal prefill program.
Future continuous batching may add batch and chunk dimensions to the same
explicit profile contract; it must not restore implicit JIT compilation in the
request path.

## Metrics and acceptance

Compilation timings are startup metrics owned by `ExecutionPlan`, not request
metrics. Parameter upload starts after both prefill and decode compilation
timers have stopped. Generation reports may expose these retained startup
measurements, but repeated requests do not claim they recompiled anything.

Permanent contracts must establish:

- profile normalization, deduplication, ordering, and capacity selection;
- rejection of requests not covered by the resident plan;
- absence of a compilation path from `generate`;
- compilation metrics are determined before parameter upload;
- the full CUDA product follows definition -> plan -> residency -> request;
- existing parameter rebinding, donation, cache, Harmony, and numerical
  contracts remain unchanged.

## Implementation ledger

- [x] Introduce validated public compilation profiles.
- [x] Split abstract definition, compiled execution plan, resident model, and
  request state into distinct product-owned types.
- [x] Compile and deduplicate every configured family before loading parameter
  buffers.
- [x] Remove lazy compilation and the mutable family cache from `generate`.
- [x] Select the smallest fitting resident profile and hard-fail unsupported
  requests.
- [x] Move compilation accounting into startup metrics.
- [x] Update the CLI and complete-checkpoint contracts to provide explicit
  profiles.
- [x] Update `SYSTEM.md` and `TASKS.md` with the permanent lifecycle invariant.
- [x] Pass the focused product contract and the applicable repository CPU/CUDA
  BuildBuddy gates.
- [x] Remove the learned-sink operand from the split-K producer while retaining
  it in the segment reduction.
- [x] Replace raw TTIR strings at the call boundary with an immutable verified
  kernel carrying its builder-authored name and argument ABI.
- [x] Reject TTIR/custom-call count, order, pointer-address-space, and element-
  type drift before StableHLO lowering.
- [x] Cover split-K plus learned sinks in structural lowering and the unchanged
  suitable-device numerical attention contract.
