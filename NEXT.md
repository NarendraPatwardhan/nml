# GPT-OSS 20B NVFP4 A40 performance analysis

## Bottom line

150+ TPS on an A40 is plausible, but the branch evidence says it will not come from another artifact-layout rewrite or a global GEMV tile change.

The strongest path is:

1. Keep recipe v2 and its output-major, K-contiguous representation.
2. Keep the restored decode geometry: `block_n=8`, `block_k=256`, 4 warps, 1 stage; retain `block_n=32` for the vocabulary head.
3. Optimize the software E2M1/E4M3 decode instruction path without changing the public format.
4. Fuse or overlap Q/K/V, especially the severely under-occupied K/V projections.
5. Reduce graph parameter patching and inter-component gaps while retaining bounded layer-pair graphs.
6. Pipeline the synchronous host token observation, which currently costs about 1 ms/token.

The latest trace is 100.566 steady device TPS but only 91.236 end-to-end decode-loop TPS ([current performance](./references/runpod/reports/20260720T200639Z-tca1p7gmof152c-9628b6916dcc-diagnostic/performance.json)). Reaching 150 therefore requires:

- 1.49× on steady device execution.
- 1.64× on the actual decode loop.
- A reduction from roughly 10.96 ms to 6.67 ms per delivered token.

Two times performance—around 200 TPS—looks possible only with a substantially better SM8x software decoder and orchestration, probably a purpose-built CUDA/PTX path rather than Triton scheduling tweaks alone.

## What was analyzed

The analysis inspected all 25 commits from `master` at `3d3ee0d` through `987e4f7`, the recipe and execution contracts, 29 retained report directories, all 18 successful `performance.json` records, the failed attempts, and the Nsight Systems SQLite/CSV exports.

The branch was clean and matched `origin/nvfp4-gpt-oss` during analysis. The analysis phase made no changes and ran no tests or model executions.

A limitation is important: these are Nsight Systems reports, not Nsight Compute reports. They provide exact launch counts, timing, grids, registers, streams, graph activity, and memory-copy timing, but not the counters needed to prove whether the decoder is limited by DRAM, instruction issue, MIO/LG throttling, scoreboarding, or occupancy. `ncu` is not installed locally.

## Performance progression

| Architecture | Steady device TPS | Decode-loop TPS | Conclusion |
|---|---:|---:|---|
| Initial executable path | 7.700 | 7.626 | Correct vertical, catastrophically inefficient decode |
| Dirty pre-`22f3541` | 9.065 | 8.996 | Small improvement only |
| `22f3541` architecture | 59.611 unprofiled / 55.475 profiled | 58.488 / 52.191 | First major success |
| `355863c` direct/composed experiment | 5.601 | 5.583 | Severe regression |
| `6f8dd0b` restored architecture | 57.248 | 54.504 | Restoration confirmed |
| Recipe v2 | 100.083 | 92.660 | Largest later improvement |
| Recipe v3 initial | 64.654 | 62.975 | Functional, much slower |
| Recipe v3 best | 81.461 | 75.129 | Improved but still 19% below v2 |
| Recipe v2 restored | 105.246 | 100.441 | Best retained run; likely normal run variance |
| Whole-model monolith | 67.915 | 65.905 | Kernel work unchanged; orchestration collapsed |
| Layer-pair restoration | 102.930 steady | 58.248 overall | Good steady state, poisoned by a 2.19-second cold graph instantiation |
| Wider tiles/two stages | 66.963 | 63.314 | Confirmed regression |
| Current restored geometry | 100.566 | 91.236 | Correct baseline |

The durable history in [TASKS.md](./TASKS.md) correctly separates functional acceptance from throughput promotion.

## Every commit

| Commit | Analysis and verdict |
|---|---|
| `feceae4` | Established the complete CPU/CUDA NVFP4 vertical and original artifact. Necessary foundation, but the eventual exact-image A40 result was only 7.7 TPS. |
| `59277b9` | Componentized the GPT-OSS product, protocol, checkpoint, execution, and generation path. Primarily architecture and E2E ownership; no independent throughput claim. |
| `5cd5c96` | Moved compilation before parameter residency. Architecturally sound, but its three A40 attempts produced two runfiles failures and an LLVM/XLA compilation segmentation fault. No performance conclusion can be assigned to this commit. |
| `3a7ea04` | Added custom-call ABI integrity. This enabled trustworthy execution, but its exact image was still 7.7 TPS. Infrastructure success, not a speedup. |
| `22f3541` | The largest successful kernel change: dedicated M=1 GEMV, exact packed-bit decode, sparse expert execution, fused gate/up SwiGLU, reused metadata, and tighter PJRT ownership. It moved the system to roughly 55–60 TPS—about 6.6×. |
| `c8ba0d6` | License-only. No performance effect. |
| `355863c` | Tried direct expert kernels, larger composed segments, and streaming head work. Result: 5.601 TPS. Direct down consumed 73.4% of GPU kernel time and gate/up 18.7%; together they were 92.1%. This architecture is conclusively rejected. |
| `6f8dd0b` | Restored the accepted `22f` design. Fresh profiled result: 57.248 TPS, confirming the regression came from `355`, not the machine or harness. |
| `f9ab291` | Recorded the restored baseline. Evidence-only. |
| `f652a30` | Introduced recipe v2: `[N,K/2]`, output-major, contiguous K, no prepared copy, and rowwise decode GEMVs. The later 100.083 TPS run validates the tranche, although layout, kernels, baked arguments, and two-layer execution changed together, so the 1.75× improvement cannot honestly be attributed to layout alone. |
| `feaf370` | Fixed recipe-v2 artifact byte accounting and unblocked the immutable run. The associated run reached 100.083 TPS ([report](./references/runpod/reports/20260720T074402Z-ylcav28r6vy6kf-d4da39627c61-diagnostic/performance.json)). |
| `644aad9` | Recorded recipe-v2 evidence. No runtime change. |
| `623e96c` | Added operation-shaped recipe v3, split-K and finalizers, but its first A40 attempts failed in `ptxas`: incompatible cache and eviction hints. No successful performance result. |
| `85e5e89` | Correctly repaired pre-Blackwell cache-policy legality. Recipe v3 then ran at 64.654 TPS ([report](./references/runpod/reports/20260720T105545Z-1nty175dxjdf8b-881acf6a63dd-diagnostic/performance.json)). The PTX fix worked; the v3 performance design did not. |
| `b5600ae` | Recorded recipe-v3 runtime acceptance while explicitly admitting that 64.654 failed the 143.12 TPS promotion threshold. Good evidence discipline; no speed change. |
| `9a466a2` | Attempted to restore M=1 occupancy and pipeline depth, but ordinary M=1 projections accidentally took the padded matrix path. Result: 47.454 TPS. |
| `19f5eca` | Forced M=1 back through GEMV with recipe-v2-like tiles. Recovered to 68.068 TPS, but split-K finalizers and recipe-v3 costs remained. |
| `5a7fabd` | Increased grouped expert tiles to `N=32,K=256`. Improved v3 to 81.461 TPS ([report](./references/runpod/reports/20260720T142341Z-heby6o0z2ndxl2-cbebd86932bd-diagnostic/performance.json)), its best result, but still materially below recipe v2. |
| `e0fe7be` | Reverted recipe v3. This was the correct architectural decision. The first restored run was blocked by the stale artifact-byte expectation. |
| `484707d` | Corrected the byte count and produced a 105.246 TPS recipe-v2 run. Most other source changes are formatting churn, but it also accidentally tracks `tools/runpod/__pycache__/api.cpython-314.pyc`, making the commit noisier than its purpose. |
| `d0fc6e2` | Compiled all 24 layers, embedding, and head into one StableHLO program. Steady performance fell to 67.915 TPS ([report](./references/runpod/reports/20260720T160400Z-q3z96lg19ubvuf-760026cf60b9-diagnostic/performance.json)). The dominant kernel work remained essentially unchanged; the GPU developed large gaps inside the giant graph. |
| `ab6b24a` | Warning suppression only. |
| `c20e001` | Reverted the monolith and restored 102.930 steady TPS. Its overall rate was only 60.230 because one first-use `cuGraphInstantiateWithFlags` call took 2.1936 seconds ([report](./references/runpod/reports/20260720T161829Z-2h28yb5uhkn7zw-290dbca720e7-diagnostic/performance.json)). This is a cold-start measurement failure, not a steady regression. |
| `72ed331` | Bundled wider tiles, `K=128`, two warps, two stages, streaming `.cg` loads, and scale hoisting. It regressed to 66.963 TPS ([report](./references/runpod/reports/20260720T195042Z-i70qee04j2zzk0-39cb278614ea-diagnostic/performance.json)). The tile/pipeline geometry failed. |
| `987e4f7` | Reverted only the tile/warp/stage schedule and restored 100.566 TPS. The streaming loads survived, proving that `.cg` was not the cause of the 67 TPS result. Current geometry is documented in [nvfp4_backend.rs](./crates/nml-ir/src/nvfp4_backend.rs), while the retained streaming loads are visible in [nvfp4.rs](./crates/nml-kernel-triton/src/nvfp4.rs). |

## What recipe v3 got wrong

Recipe v3 borrowed superficially appropriate ideas—output-contiguous slices, split-K, finalization, larger tiles—but changed the representation and execution cost model together.

Its best run had:

- Gate/up around 125 µs plus a finalizer, using 233 registers and only `4 × 90` CTAs.
- Down around 65 µs plus a finalizer, using 96 registers and `4 × 90` CTAs.
- Extra partial-result storage, finalization launches, and reductions on every projection.
- Lower grid concurrency than recipe v2.

Recipe v2’s current gate/up uses 72 registers and `4 × 360` CTAs; down uses 48 registers. Recipe v3 traded away occupancy and added intermediate traffic before demonstrating that split-K was necessary.

This is exactly why the reference ledger warns that reference scheduling ideas must survive NML’s representation and epilogue contracts rather than being copied wholesale ([KERNEL_REFERENCES.md](./references/KERNEL_REFERENCES.md)).

## What the current Nsight trace says

The model uses four of 32 experts per token, 24 layers, hidden/intermediate width 2,880, Q width 4,096, K/V width 512, and a 201,088-token vocabulary ([config](./artifacts/gpt-oss-20b-nvfp4/config.json)). Recipe v2 quantizes attention, experts, embedding, and output projection ([recipe](./artifacts/gpt-oss-20b-nvfp4/recipe.json)).

Approximate active compact-weight traffic is 2.03 GB/token. With A40’s reported 696 GB/s:

- Ideal weight-only floor: 2.92 ms/token.
- Current 100.566 TPS corresponds to about 204 GB/s aggregate effective bandwidth, 29% of peak.
- 150 TPS requires about 305 GB/s, 44% of peak.
- 200 TPS requires about 406 GB/s, 58% of peak.

That makes 150 physically reasonable.

### Projection decomposition

| Current kernel work | Time per token | Estimated effective bandwidth |
|---|---:|---:|
| 24 gate/up projections | 2.921 ms | 307 GB/s |
| 24 down projections | 1.534 ms | 292 GB/s |
| 24 Q projections | 0.760 ms | 210 GB/s |
| 24 K and 24 V projections | 0.619 ms | only ~64 GB/s each |
| 24 O projections | 0.681 ms | 234 GB/s |
| Vocabulary head | 1.020 ms | 319 GB/s |
| **NVFP4 total** | **7.535 ms** | — |

Repeated non-NVFP4 kernels add approximately 0.82 ms/token. The steady device result is roughly 9.94 ms/token, leaving about 1.59 ms/token in recurring graph/submission gaps.

The clearest kernel-level finding is K/V under-occupancy: each launches only 64 CTAs on an 84-SM A40. Those tiny projections achieve roughly one-fifth the effective throughput of gate/up and the head.

### The graph finding

Current layer-pair execution produced:

- 4,172 `cuGraphLaunch` calls, about 13 per token.
- 122,865 `cuGraphExecKernelNodeSetParams_v2` calls, about 385 per token.
- 451 ms total host API time in graph launch.
- 448 ms total in graph-node parameter patching.

The single full-model graph reduced graph launches to 344, but steady device time grew from about 3.16 seconds to 4.68 seconds for the run. Dominant NVFP4 kernel time was virtually unchanged. The giant XLA graph therefore reduced launch count but made GPU scheduling much less continuous.

Conclusion: do not repeat the 24-layer StableHLO monolith. The useful target is persistent/bounded graph composition and fewer parameter patches, not maximum source-level fusion.

### The host boundary

The current loop waits for and downloads each selected token before proceeding ([execution.rs](./products/serve/src/gpt_oss/execution.rs)). The token itself is already fed device-to-device into the next embedding; it is not uploaded again.

Nsight records negligible actual device-copy work, while the application charges 320.286 ms—about 1 ms/token—to `decode_download`. This is readiness/host-observation latency, not PCIe bandwidth.

## What the external benchmark proves—and does not prove

The linked article is directionally useful, but its quantization labeling is imprecise. It describes the 3090 result as generic Q4/Q4_K_M, while the primary llama.cpp benchmark uses `gpt-oss-20b-mxfp4.gguf` and reports approximately 161.8 TPS on an RTX 3090. [Article](https://runaihome.com/blog/gpt-oss-20b-local-ai-hardware-guide-2026/), [primary benchmark](https://github.com/ggml-org/llama.cpp/discussions/15396).

It is also not an apples-to-apples format comparison. OpenAI’s published model retains only MoE weights in MXFP4; non-MoE tensors are BF16. NML recipe v2 quantizes attention and the very large vocabulary head as well. [OpenAI GPT-OSS repository](https://github.com/openai/gpt-oss).

That means:

- The llama.cpp result proves that software MXFP4 decode can be fast on Ampere.
- It does not directly prove 161 TPS on an A40.
- NML’s estimated active weight traffic is substantially lower, so 150 TPS does not require matching the 3090’s raw bandwidth.

More recent llama.cpp work is especially relevant: combined GEMV fusion, top-k/MoE work, fused normalization, and concurrent Q/K/V streams moved a reported 4090 GPT-OSS result from 232.05 to 271.99 TPS—about 17%. [llama.cpp optimization discussion](https://github.com/ggml-org/llama.cpp/discussions/17621).

Another llama.cpp investigation found that on non-native-FP4 GPUs the fallback can be instruction-bound, with much higher instruction counts and MIO/LG throttling; simple alignment repacking reportedly provided only 1–2%. [Software fallback/layout discussion](https://github.com/ggml-org/llama.cpp/discussions/18427). That matches NML’s evidence: another representation rewrite is unlikely to supply 1.5×.

## Recommended solution set

### 1. Freeze recipe v2 as the control

Do not introduce recipe v4 or another public quantization format.

Keep:

- `[N,K/2]` output-major, K-contiguous payload.
- One E4M3FN scale per 16 E2M1 values.
- No persistent BF16 expansion.
- Fused gate/up activation.
- Sparse top-four expert execution.
- Current M=1 tile family.
- Current `.cg` weight loads until an isolated `.ca`/`.cg` comparison says otherwise.

The former content of `NEXT.md` is contradicted by measurement: fewer CTAs and deeper pipelines did not improve occupancy. Its predicted 35% gain became a 33% regression.

### 2. Build an SM8x-specific software FP4 decoder

This is the likely centerpiece.

The current kernels perform nibble extraction, E2M1 decode, E4M3FN decode, scale expansion, global scaling, activation multiplication, and F32 reduction in generic Triton IR. NCU must determine whether instruction issue is the principal limiter, but the llama.cpp evidence strongly suggests it.

Candidate implementation:

- Preserve recipe-v2 bytes.
- Use aligned 16-byte or wider packed loads.
- Decode E2M1 with a constant LUT/gather or explicit SM80-friendly vector operations.
- Reuse one decoded E4M3 scale across all 16 values without materializing expanded scale tensors.
- Hoist address arithmetic and invariant global-scale operations.
- Use half2/BF16 vector arithmetic where exactness allows, retaining F32 accumulation.
- Keep current CTA geometry initially so decoder improvements are measured independently.
- If Triton lowering cannot express an efficient decoder, add a purpose-built SM80/SM86 CUDA/PTX backend behind the same `NvFp4` representation.

A backend-specific kernel is not a custom user format. Users still consume canonical NVFP4; only code generation is device-specific.

### 3. Fuse or overlap Q/K/V

This is the most obvious shape-specific win.

K and V each launch only 64 CTAs and achieve about 64 GB/s. Options:

- One fused NVFP4 QKV kernel with a combined output domain and three output buffers.
- Three concurrent stream launches, as llama.cpp uses.
- Selective split-K only for K/V, enough to produce at least two A40 waves.
- A narrower K/V-only `block_n`, such as four outputs, if NCU shows instruction cost is acceptable.

Do not apply split-K to gate/up, down, or the vocabulary head: those already have ample CTAs, and recipe v3 proved that partial buffers/finalizers can erase the benefit.

A reasonable QKV target is saving 0.5–0.7 ms/token.

### 4. Reduce graph patches without creating another monolith

Keep the two-layer executable as the known-good unit, then test bounded changes:

- Make KV cache addresses stable and update contents in place.
- Use fixed hidden-state ping-pong buffers.
- Move position updates fully onto the device.
- Preinstantiate all graph variants before timed decode.
- Patch only truly request-varying nodes rather than hundreds of stable parameter nodes.
- Experiment with four-layer segments—six submissions/token—but stop if recurrent kernel gaps grow.
- If the runtime permits it, capture an outer CUDA Graph that launches the already-optimized bounded components instead of recompiling the whole transformer as one StableHLO program.

The target should be reducing the approximately 1.59 ms/token recurrent gap to around 0.5 ms.

### 5. Pipeline token observation

Queue the selected-token D2H copy on a separate stream and immediately begin one speculative next decode using the device token buffer.

The host then observes the previous token while the GPU works. If the previous token is a stop token, discard the speculative result; the request-local cache is ending anyway.

This can hide most of the current ~1 ms/token host boundary without changing model semantics or uploading the token again.

### 6. Fuse only measured lightweight boundaries

After projection work:

- QKV reshape/RoPE where legal.
- Residual plus RMSNorm.
- Router logits plus top-k preparation.
- Sampling reductions/sorts.
- KV update bookkeeping.

These repeated non-NVFP4 kernels total only about 0.82 ms/token, so they cannot produce 1.5× alone. A realistic target is 0.2–0.3 ms/token.

## A credible 150 TPS budget

| Component | Current | Required target |
|---|---:|---:|
| NVFP4 projections | 7.54 ms | ~5.20 ms |
| Other repeated GPU kernels | 0.82 ms | ~0.55 ms |
| Graph/submission gaps | 1.59 ms | ~0.55 ms |
| Host token observation | ~1.00 ms | ~0.30 ms |
| **Decode loop** | **~10.95 ms / 91 TPS** | **~6.60 ms / 152 TPS** |

This asks for approximately:

- 31% faster projection kernels, including QKV overlap.
- About 1 ms less graph/runtime gap.
- Most token-download latency hidden.
- A smaller contribution from conventional fusion.

That is aggressive but consistent with the trace. In contrast, 200 TPS requires a 5 ms complete loop; that likely needs the optimized SM8x decoder to approach 400–450 GB/s on the large gate/down/head workloads.

## Required next profiling gate

Before implementation decisions, obtain Nsight Compute reports for these exact decode shapes:

- Gate/up: active top-four experts.
- Down: active top-four experts.
- Q: `2880 → 4096`.
- K/V: `2880 → 512`.
- O: `4096 → 2880`.
- Head: `2880 → 201088`.

Collect:

- DRAM bytes and percent of sustained peak.
- L1/L2 sectors, hit rates, and replay.
- Executed instruction count.
- Issue-active percentage.
- LG, MIO, long-scoreboard, not-selected, and dependency stalls.
- Achieved occupancy and active warps.
- Registers, spills, and local-memory traffic.

Every experiment should use the same A40, locked clocks, driver, artifact revision, prompt, and 320-token workload; report median and spread across at least five warm runs. Cold graph instantiation and warm steady performance must be separate. Promotion should require 150+ **decode-loop** TPS, not only steady-device TPS.

There is also a small evidence-hygiene issue: the monolith report’s metadata does not cleanly identify `d0fc6e2`, although its workload behavior is unambiguously the single-graph implementation. Future reports should embed the exact source SHA, dirty-tree state, kernel schedule identity, compiler revision, PTX hash, artifact revision, and image digest.

Finally, the checked-in artifact metadata is inconsistent: [published.json](./artifacts/gpt-oss-20b-nvfp4/published.json) records `11,805,934,204` bytes, while [README.md](./artifacts/gpt-oss-20b-nvfp4/README.md) still says `11,805,938,322`. It does not affect current throughput, but it illustrates why benchmark provenance needs to be byte-exact.
