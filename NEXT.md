# GPT-OSS 20B NVFP4 A40 performance analysis

## 2026-07-21 measured decoder result

Commit `29036b7` implemented the first recommended SM8x decoder tranche without
changing recipe v2, launch geometry, or the public NVFP4 representation. The
exact image
`sha256:f4fd5d8506070e5d582078c9ea87e0bc2aba9e01c2e9512bf284a5422222b671`
completed the same 320-token A40 workload under GDB and Nsight Systems at
121.694 steady-device TPS and 117.384 decode-loop TPS
([report](./references/runpod/reports/20260721T200016Z-33llh2wpqva86j-f4fd5d850607-diagnostic/performance.json)).

Against the fastest recent warm control on the prior image, steady-device TPS
improved from 107.624 to 121.694 (+13.1%) and decode-loop TPS improved from
103.130 to 117.384 (+13.8%). Against the retained official baseline, the gains
are +21.0% and +28.7%, respectively. The kernel evidence, rather than the more
variable token-download time, establishes the actual result:

| Decode projection | Prior warm control | `29036b7` | Change | Registers |
|---|---:|---:|---:|---:|
| Gate/up | 122.519 us | 94.316 us | -23.0% | 72 -> 56 |
| Down | 64.035 us | 53.737 us | -16.1% | 48 -> 40 |
| Q | 31.928 us | 25.737 us | -19.4% | 56 -> 40 |
| K or V | 12.970 us | 14.236 us | **+9.8%** | 56 -> 40 |
| O | 28.699 us | 22.261 us | -22.4% | 48 -> 39 |
| Vocabulary head | 1018.680 us | 809.833 us | -20.5% | 128 -> 128 |

Total decode NVFP4 kernel time fell from 7.579 to 6.203 ms/token (-18.2%).
The unchanged trace contains exactly 179,350 kernel launches; total GPU kernel
time fell from 3.080 to 2.622 seconds. This proves that applying the tensor-wide
scale after the complete F32 K reduction and simplifying exact E2M1/E4M3FN
construction produced real kernel alpha rather than a host-side measurement
artifact.

This particular algebraic tranche is substantially harvested. More decoder
headroom may exist in a purpose-built SM86 CUDA/PTX implementation, vectorized
packed conversion, or a counter-guided E4M3 subnormal path, but those are
different and higher-risk implementations. Repeating generic expression
rewrites now has lower expected value than addressing the only regressing,
severely under-occupied projection family: K/V.

## Bottom line

150+ TPS on an A40 is plausible, but the branch evidence says it will not come from another artifact-layout rewrite or a global GEMV tile change.

The strongest path, updated after the measured decoder gain, is:

1. Keep recipe v2 and its output-major, K-contiguous representation.
2. Keep the restored decode geometry: `block_n=8`, `block_k=256`, 4 warps, 1 stage; retain `block_n=32` for the vocabulary head.
3. Keep the successful decoder simplification from `29036b7` as the new control.
4. Fuse Q/K/V decode into one launch so Q's 512 CTAs absorb the two under-filled 64-CTA K/V tails.
5. Reduce graph parameter patching and inter-component gaps while retaining bounded layer-pair graphs.
6. Pipeline the synchronous host token observation.
7. Return to a purpose-built SM86 decoder only after the fused-QKV and orchestration gains are measured.

The latest trace is 121.694 steady-device TPS and 117.384 end-to-end decode-loop TPS. Reaching 150 now requires:

- 1.23x on steady-device execution.
- 1.28x on the actual decode loop.
- A reduction from roughly 8.22 ms to 6.67 ms per steady device token.

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
| Simplified exact decoder and post-reduction global scale (`29036b7`) | **121.694** | **117.384** | Successful: NVFP4 decode kernels -18.2% |

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

### Projection decomposition after `29036b7`

| Current kernel work | Time per token | Conclusion |
|---|---:|---:|
| 24 gate/up projections | 2.264 ms | Decoder simplification worked |
| 24 down projections | 1.290 ms | Decoder simplification worked |
| 24 Q projections | 0.618 ms | Decoder simplification worked |
| 24 K and 24 V projections | 0.683 ms | Only regression; two 64-CTA tails per layer |
| 24 O projections | 0.534 ms | Decoder simplification worked |
| Vocabulary head | 0.812 ms | Decoder simplification worked despite 128 registers |
| **NVFP4 total** | **6.203 ms** | **1.376 ms/token saved** |

The first plus steady decode executions now total 2.631 seconds. After removing
prefill from the Nsight kernel sum, decode kernels occupy approximately 2.288
seconds, leaving roughly 0.34 seconds, or 1.07 ms/token, in recurring
device-execution gaps. The prior warm trace exposed about 0.75 ms/token of such
gaps. The new run used a different A40 host and an older 570 driver, so the
increase cannot yet be assigned entirely to code, but faster kernels also make
fixed launch and graph costs a larger fraction of the budget.

The clearest kernel-level finding remains K/V under-occupancy: each launches
only 64 CTAs on an 84-SM A40, and K/V was the only projection family to regress
after the decoder change. Q launches 512 otherwise-identical CTAs. A single
640-CTA QKV grid can turn three separately scheduled tails into approximately
eight full-device waves while retaining recipe-v2 bytes and the proven
`N=8,K=256` CTA body.

### The graph finding

Current layer-pair execution produced:

- 4,172 `cuGraphLaunch` calls, about 13 per token.
- 122,865 `cuGraphExecKernelNodeSetParams_v2` calls, about 385 per token.
- 449 ms total host API time in graph launch.
- 237 ms total in graph-node parameter patching.

The single full-model graph reduced graph launches to 344, but steady device time grew from about 3.16 seconds to 4.68 seconds for the run. Dominant NVFP4 kernel time was virtually unchanged. The giant XLA graph therefore reduced launch count but made GPU scheduling much less continuous.

Conclusion: do not repeat the 24-layer StableHLO monolith. The useful target is persistent/bounded graph composition and fewer parameter patches, not maximum source-level fusion.

### The host boundary

The current loop waits for and downloads each selected token before proceeding ([execution.rs](./products/serve/src/gpt_oss/execution.rs)). The token itself is already fed device-to-device into the next embedding; it is not uploaded again.

Nsight records negligible actual device-copy work. The new run charges 86.853
ms, or 0.27 ms/token, to `decode_download`, versus 0.39 ms/token in the fastest
recent warm control and 1.00 ms/token in the retained official baseline. This
is readiness/host-observation latency, not PCIe bandwidth, and it is variable
enough that promotion must be supported by kernel and steady-device evidence.

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

### 2. Preserve the successful SM8x decoder and defer the lower-level rewrite

The generic-IR decoder tranche is now measured and accepted. Keep its exact
bit construction and post-reduction global scaling. Do not combine further
decoder work with QKV launch fusion: separate images and reports are required
to preserve attribution.

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

### 3. Fuse Q/K/V decode as the next isolated experiment

This is the most obvious shape-specific win.

K and V each launch only 64 CTAs and achieve about 64 GB/s. First implementation:

- Author one semantic compact QKV operation for three NVFP4 projections that
  share the same activation.
- Lower only `M=1` SM8x execution to one Triton kernel and one custom call.
- Retain three independent payload/scale/global/bias operands and three output
  buffers; do not concatenate or rewrite checkpoint tensors.
- Use one combined 640-CTA launch for GPT-OSS widths 4096/512/512 with the
  proven `block_n=8`, `block_k=256`, four-warps, one-stage body.
- Keep prefill, CPU, and Turing paths as three existing projections.

Fallback options only if the fused launch does not promote:

- One fused NVFP4 QKV kernel with a combined output domain and three output buffers.
- Three concurrent stream launches, as llama.cpp uses.
- Selective split-K only for K/V, enough to produce at least two A40 waves.
- A narrower K/V-only `block_n`, such as four outputs, if NCU shows instruction cost is acceptable.

Do not apply split-K to gate/up, down, or the vocabulary head: those already have ample CTAs, and recipe v3 proved that partial buffers/finalizers can erase the benefit.

A reasonable QKV target is saving 0.35–0.55 ms/token. Promotion requires a
fresh Nsight trace showing one 640-CTA QKV kernel per layer, no numerical or
device-contract regression, and a decode-loop improvement outside run variance.

### 4. Reduce graph patches without creating another monolith

Keep the two-layer executable as the known-good unit, then test bounded changes:

- Make KV cache addresses stable and update contents in place.
- Use fixed hidden-state ping-pong buffers.
- Move position updates fully onto the device.
- Preinstantiate all graph variants before timed decode.
- Patch only truly request-varying nodes rather than hundreds of stable parameter nodes.
- Experiment with four-layer segments—six submissions/token—but stop if recurrent kernel gaps grow.
- If the runtime permits it, capture an outer CUDA Graph that launches the already-optimized bounded components instead of recompiling the whole transformer as one StableHLO program.

The target should be reducing the approximately 1.07 ms/token recurrent gap to around 0.5 ms.

### 5. Pipeline token observation

Queue the selected-token D2H copy on a separate stream and immediately begin one speculative next decode using the device token buffer.

The host then observes the previous token while the GPU works. If the previous token is a stop token, discard the speculative result; the request-local cache is ending anyway.

This can hide most of the remaining 0.27-0.39 ms/token host boundary without
changing model semantics or uploading the token again.

### 6. Fuse only measured lightweight boundaries

After projection work:

- QKV reshape/RoPE where legal.
- Residual plus RMSNorm.
- Router logits plus top-k preparation.
- Sampling reductions/sorts.
- KV update bookkeeping.

These repeated non-NVFP4 kernels total about 0.97 ms/token in the new trace, so
they cannot produce the remaining speedup alone. A realistic saving is
0.2-0.3 ms/token.

## A credible 150 TPS budget

| Component | Current | Required target |
|---|---:|---:|
| NVFP4 projections | 6.20 ms | ~5.25 ms |
| Other repeated GPU kernels | ~0.97 ms | ~0.75 ms |
| Device graph/submission gaps | ~1.07 ms | ~0.55 ms |
| Host token observation beyond device execution | ~0.30 ms | ~0.10 ms |
| **Decode loop** | **~8.52 ms / 117 TPS** | **~6.65 ms / 150 TPS** |

This asks for approximately:

- 15% less projection time, starting with QKV tail elimination.
- About 0.5 ms less graph/runtime gap.
- Most remaining token-observation latency hidden.
- A smaller contribution from conventional fusion.

That is aggressive but consistent with the trace. In contrast, 200 TPS requires a 5 ms complete loop; that likely needs the optimized SM8x decoder to approach 400–450 GB/s on the large gate/down/head workloads.

## Required promotion profiling gate

The fused-QKV implementation is justified directly by the Nsight Systems launch
geometry and does not need NCU to begin. Before promoting it, run the exact
image on A40 and require one `nvfp4_qkv_gemv` launch with grid 640 per layer.
If it fails to improve outside run variance, obtain Nsight Compute reports for
these exact decode shapes before choosing split-K, narrower tiles, or another
decoder rewrite:

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
