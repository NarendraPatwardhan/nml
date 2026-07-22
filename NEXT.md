# GPT-OSS 20B NVFP4 A40 performance analysis

## 2026-07-22 expert-kernel result and next selected tranche

Commit `bc7b830` combined exact K=2880 tail handling, E2M1/E4M3FN decode
codebooks, eight decode gate/up warps, and four decode down warps. Image
`sha256:6b7b883efa28b2931986ee04012e5f1aab9eded60898ed3aeea3d7aa2dd03fdb`
reached 155.203 steady-device TPS, 154.713 device-decode TPS, and 138.269
decode-loop TPS on A40
([report](./references/runpod/reports/20260722T013718Z-y6n806pc8gtemu-6b7b883efa28-diagnostic/performance.json)).
The direct `d4ad426` control measured 150.113, 149.606, and 136.742 TPS,
respectively.

The improvement is real but uneven. Down fell from 55.538 to 49.304 us
(-11.2%), while gate/up fell from 96.401 to 93.043 us (-3.5%). Unchanged
repeated kernels were approximately 2.18% faster on the new pod, leaving about
9.3% attributable improvement for down but only about 1.3% for gate/up. The
exact final 64-wide K tail is the strongest explanation for the down result;
the codebook and gate/up warp effects remain bundled and are not independently
proven.

The chronological Nsight trace shows that the kernel gain exposed a host
submission boundary. Clean GPU busy time fell from 6.822 to 6.522 ms/token,
but clean non-kernel gaps rose from 0.481 to 0.643 ms/token. The recurring gap
is immediately before layer pair four: 433.8 us median and approximately
443.5 us clean average, versus roughly 4.8 us at ordinary graph boundaries.
Layer-pair starts are approximately 459 us apart. A five-pair lookahead models
to about 10 us residual at this boundary and 148.5-148.8 clean decode-loop TPS;
a small kernel or host improvement should then clear 150.

The selected next tranche is therefore:

1. Increase bounded decode lookahead from three to five layer pairs.
2. Restore decode gate/up to four warps while retaining the exact-tail and
   codebook paths; keep down at four warps unchanged.
3. Avoid the redundant token-buffer await when readiness can be established
   without blocking. The host-download event remains the correctness
   synchronization, while the readiness fast path preserves the existing
   device-execution and download timing boundary.

### Future tranche: bounded four-layer graph composition

If five-pair lookahead plus the isolated gate/wait corrections do not sustain
150+ decode-loop TPS, combine two adjacent layer-pair executables into one
bounded four-layer component. Keep embedding, head, and the remaining schedule
separate; do not repeat the failed 24-layer monolith.

The current trace records approximately 13 graph launches/token at about
1.405 ms/token of host API time and roughly 336 graph-node parameter patches at
about 0.893 ms/token. Six four-layer components would remove six graph launches
per token, with an upper-bound launch saving near 0.65 ms/token; parameter
patching will remain unless stable arguments are also retained. Prove the
schedule first with one isolated four-layer boundary, checking GPU continuity,
then expand to all six only if it promotes end-to-end decode-loop TPS.

The periodic 5-15 ms trace stalls must not drive this design: they align with
16 CUPTI buffer flushes totaling 24.616 ms and are profiler overhead. The
deterministic pair-four boundary is the real orchestration target.

## 2026-07-22 measured fused-QKV result

Commit `12ce228` fused the three M=1 Q/K/V projections into one semantic NVFP4
operation without changing recipe v2, checkpoint tensors, prefill, CPU, or the
public NVFP4 representation. The exact image
`sha256:b0d387f67cdcca02c5dcd36a5ea8c336de8793770f354eb21b8985c7142aeb65`
completed the same 320-token A40 workload under GDB and Nsight Systems at
136.250 steady-device TPS and 132.249 decode-loop TPS
([report](./references/runpod/reports/20260721T221141Z-v0pgn2uihkknxw-b0d387f67cdc-diagnostic/performance.json)).
The source commit embedded in the report is exactly `12ce228`; GDB exited
normally, and the paid pod was terminated after collection.

The direct control is `29036b7`, which already contained the accepted SM8x
decoder simplification. The fused result is therefore isolated:

| Metric | `29036b7` control | Fused QKV `12ce228` | Change |
|---|---:|---:|---:|
| Steady-device TPS | 121.694 | **136.250** | **+12.0%** |
| Device-decode TPS | 121.260 | **136.028** | **+12.2%** |
| Decode-loop TPS | 117.384 | **132.249** | **+12.7%** |
| Steady decode execution | 2613.121 ms | **2333.945 ms** | **-10.7%** |
| Complete decode loop | 2717.565 ms | **2412.119 ms** | **-11.2%** |
| GPU kernel calls | 179,350 | **164,038** | **-15,312** |
| Total GPU kernel time | 2622.156 ms | **2435.431 ms** | **-7.1%** |

Nsight confirms the intended mechanism. The old per-layer Q + K + V sequence
cost `25.737 + 14.236 + 14.236 = 54.209` us and launched grids
`512 + 64 + 64`. The new `nvfp4_qkv_gemv` costs 27.908 us, launches one
640-CTA grid, and remains at 40 registers/thread with no local memory. Across
the run, QKV GPU time fell from 415.030 to 213.667 ms (-48.5%), and the launch
count fell from 22,968 to 7,656. Non-QKV kernel time changed by only +0.7%, so
the gain cannot be explained by host or cross-machine variance.

The two successful tranches are now harvested as the control: exact software
decode from `29036b7`, then QKV tail elimination from `12ce228`. Repeating QKV
tiling, split-K, layout, or generic decoder-expression experiments has lower
expected value than attacking the newly exposed recurring execution bubbles.

## Bottom line

150+ TPS on an A40 is plausible, but the branch evidence says it will not come from another artifact-layout rewrite or a global GEMV tile change.

The strongest path, updated after the measured QKV gain, is:

1. Keep recipe v2 and its output-major, K-contiguous representation.
2. Keep the restored decode geometry: `block_n=8`, `block_k=256`, 4 warps, 1 stage; retain `block_n=32` for the vocabulary head.
3. Keep the successful decoder simplification from `29036b7` and fused QKV from `12ce228` as the new control.
4. Pipeline one bounded speculative unit: enqueue the next embedding and first layer pair before observing the previous token on the host.
5. If a boundary remains, fuse decode embedding with the first layer pair rather than enlarging all layer segments.
6. Reduce stable graph parameter patching only after the bounded pipeline is measured.
7. Return to a purpose-built SM86 decoder only after orchestration gains and NCU counters justify it.

The latest trace is 136.250 steady-device TPS and 132.249 end-to-end decode-loop
TPS. Reaching 150 now requires:

- 1.10x on steady-device execution.
- 1.13x on the actual decode loop.
- A reduction from 7.56 to 6.67 ms per decode-loop token, about 0.90 ms.

Two times performance—around 200 TPS—looks possible only with a substantially better SM8x software decoder and orchestration, probably a purpose-built CUDA/PTX path rather than Triton scheduling tweaks alone.

## What was analyzed

The original analysis inspected all 25 commits from `master` at `3d3ee0d`
through `987e4f7`, the recipe and execution contracts, the retained successful
and failed reports, and the Nsight Systems SQLite/CSV exports. This update also
analyzes the documentation commit `ee29dff`, decoder commit `29036b7`, fused-QKV
commit `12ce228`, and their exact A40 reports.

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
| Fused M=1 QKV scheduling (`12ce228`) | **136.250** | **132.249** | Successful: QKV time -48.5%, 15,312 launches removed |

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
| `ee29dff` | Replaced the speculative roadmap with the commit-by-commit and Nsight-backed analysis in this file. Evidence-only. |
| `29036b7` | Applied the tensor-wide scale after the complete F32 K reduction and simplified exact E2M1/E4M3FN construction. The exact A40 image reached 121.694 steady-device and 117.384 decode-loop TPS; decode NVFP4 kernels fell 18.2%. Accepted. |
| `12ce228` | Added one semantic M=1 QKV operation and one 640-CTA SM8x kernel while preserving three physical parameter sets and outputs. The exact A40 image reached 136.250 steady-device and 132.249 decode-loop TPS. QKV GPU time fell 48.5% and 15,312 launches disappeared. Accepted. |

## What recipe v3 got wrong

Recipe v3 borrowed superficially appropriate ideas—output-contiguous slices, split-K, finalization, larger tiles—but changed the representation and execution cost model together.

Its best run had:

- Gate/up around 125 µs plus a finalizer, using 233 registers and only `4 × 90` CTAs.
- Down around 65 µs plus a finalizer, using 96 registers and `4 × 90` CTAs.
- Extra partial-result storage, finalization launches, and reductions on every projection.
- Lower grid concurrency than recipe v2.

Recipe v2’s accepted decoder now uses 56 registers for gate/up and 40 for down,
with the same `4 × 360` CTA geometry. Recipe v3 traded away occupancy and added
intermediate traffic before demonstrating that split-K was necessary.

This is exactly why the reference ledger warns that reference scheduling ideas must survive NML’s representation and epilogue contracts rather than being copied wholesale ([KERNEL_REFERENCES.md](./references/KERNEL_REFERENCES.md)).

## What the current Nsight trace says

The model uses four of 32 experts per token, 24 layers, hidden/intermediate width 2,880, Q width 4,096, K/V width 512, and a 201,088-token vocabulary ([config](./artifacts/gpt-oss-20b-nvfp4/config.json)). Recipe v2 quantizes attention, experts, embedding, and output projection ([recipe](./artifacts/gpt-oss-20b-nvfp4/recipe.json)).

Approximate active compact-weight traffic is 2.03 GB/token. With A40’s reported 696 GB/s:

- Ideal weight-only floor: 2.92 ms/token.
- Current 136.250 steady-device TPS corresponds to about 277 GB/s aggregate effective bandwidth, 40% of peak.
- 150 TPS requires about 305 GB/s, 44% of peak.
- 200 TPS requires about 406 GB/s, 58% of peak.

That makes 150 physically reasonable.

### Projection decomposition after `12ce228`

| Current kernel work | Time per token | Conclusion |
|---|---:|---:|
| 24 gate/up projections | 2.267 ms | Largest remaining NVFP4 family |
| 24 down projections | 1.293 ms | Second-largest NVFP4 family |
| 24 fused QKV projections | 0.670 ms | Down from 1.301 ms; fusion worked |
| 24 O projections | 0.536 ms | Stable against control |
| Vocabulary head | 0.812 ms | Stable despite 128 registers |
| **NVFP4 total** | **5.579 ms** | **0.624 ms/token saved by QKV fusion** |

Across the 318 complete embedding-to-embedding intervals in the new trace, the
average token interval is 7.607 ms: 6.594 ms of GPU kernels and 1.013 ms with no
kernel executing. The direct control averaged 8.579 ms: 7.199 ms busy and
1.380 ms non-kernel. The busy-time reduction is the fused QKV kernel; the
non-kernel reduction is consistent with removing 48 kernel nodes per token.

The remaining raw-kernel target is no longer QKV. Gate/up plus down consume
3.559 ms/token, 64% of NVFP4 time and 47% of the full traced token interval.
Those expert kernels are the only family large enough to provide another major
kernel tranche, but changing them without NCU counters would be guesswork. The
trace exposes a lower-risk orchestration opportunity first.

### The graph finding

Current layer-pair execution produces:

- 4,172 `cuGraphLaunch` calls, about 13 per token.
- 107,557 `cuGraphExecKernelNodeSetParams_v2` calls, about 337 per token.
- 322 ms total host API time in graph launch.
- 297 ms total in graph-node parameter patching.

QKV fusion left executable-level graph launches unchanged but removed 15,308
kernel-node patches, essentially the expected two nodes per layer per decode
step. This proves bounded semantic fusion can remove real graph bookkeeping;
the absolute API durations are host-sensitive and should not be compared across
pods as if they were kernel timings.

The single full-model graph reduced graph launches to 344, but steady device time grew from about 3.16 seconds to 4.68 seconds for the run. Dominant NVFP4 kernel time was virtually unchanged. The giant XLA graph therefore reduced launch count but made GPU scheduling much less continuous.

Conclusion: do not repeat the 24-layer StableHLO monolith. The useful target is persistent/bounded graph composition and fewer parameter patches, not maximum source-level fusion.

### The next alpha: two deterministic boundaries

The chronological trace localizes 87% of non-kernel time to two boundaries that
occur exactly once per steady token:

| Boundary | Calls | Average gap | Total |
|---|---:|---:|---:|
| Final head position update -> next embedding | 318 | 420.732 us | 133.793 ms |
| Embedding -> first layer-pair kernel | 318 | 461.569 us | 146.779 ms |
| **Combined** | | **882.301 us/token** | **280.572 ms** |

The first boundary contains completion observation, the 67.015 ms aggregate
`decode_download`, token emission/stop handling, and next submission. The
second is the cost of preparing the first bounded layer-pair graph after a
two-microsecond embedding kernel. All other non-kernel time is only about
0.13 ms/token.

The current loop waits for and downloads each selected token before proceeding
([execution.rs](./products/serve/src/gpt_oss/execution.rs)). Yet `enqueue`
already returns dependency-carrying device buffers that may be passed directly
to another enqueue without a host synchronization
([runtime](./crates/nml-runtime/src/lib.rs)). The token is already fed
device-to-device into embedding; it is never uploaded again. This makes bounded
lookahead an implementation opportunity, not a new runtime or format project.

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

### 2. Preserve both accepted SM8x kernel tranches

Keep the exact decoder construction and post-reduction global scaling from
`29036b7`. Keep the semantic QKV operation, independent parameter buffers,
640-CTA grid, and 40-register kernel from `12ce228`. QKV beat its stated
0.35-0.55 ms/token target by saving 0.631 ms/token at the kernel level.

Do not retile QKV or revive selective split-K. The fused kernel is now only
0.670 ms/token, while gate/up and down together are 3.559 ms/token. Further QKV
work cannot close the remaining gap.

### 3. Pipeline one layer pair ahead as the next isolated experiment

This is the highest-confidence remaining alpha and requires no NCU prerequisite.
Restructure decode as a bounded one-pair lookahead state machine:

1. Enqueue the current head and retain its dependency-carrying device outputs:
   token, sampling state, and position.
2. Before `token_buffer.wait()` or `download_token`, bind that device token to
   the next embedding and enqueue embedding plus the first two-layer executable.
3. While the GPU executes that bounded unit, wait for and download the previous
   token, emit it, and evaluate the stop condition on the host.
4. If generation continues, submit the remaining 11 layer pairs and head. If
   the previous token is terminal, discard the speculative buffers; its cache
   is request-local and the request is ending.
5. Never speculate past `max_new_tokens`, and keep the visible token stream and
   sampling-state sequence identical to the current implementation.

Submitting only embedding plus one pair bounds terminal waste while providing
roughly one layer-pair execution window for host observation. It should attack
both measured bubbles: it queues the first pair before the two-microsecond
embedding can outrun host submission, and overlaps previous-token observation
with useful GPU work.

The theoretical exposed opportunity is 0.882 ms/token. Removing all of it
would move the measured 7.56 ms decode loop to about 6.68 ms, approximately
149.7 TPS. A realistic first-run target is a 0.65-0.80 ms reduction, or roughly
145-148 TPS; a small bounded fusion or conventional-kernel gain may still be
needed to clear 150 reliably.

### 4. Fuse only the remaining bounded boundary if lookahead leaves it exposed

If Nsight still shows a material embedding-to-first-pair gap, build a dedicated
decode entry component containing embedding plus the first sliding/full layer
pair. Keep the other 11 two-layer executables unchanged. This removes one
executable boundary per token without repeating the failed 24-layer StableHLO
monolith or changing the reusable layer-pair schedule globally.

Only after that experiment should broader graph work resume:

- Make KV cache addresses stable and update contents in place.
- Use fixed hidden-state ping-pong buffers.
- Patch only truly request-varying nodes rather than stable parameter nodes.
- Capture an outer CUDA Graph around accepted bounded components if PJRT/XLA
  exposes a safe ownership boundary.

Four-layer or whole-model StableHLO segments are not the next experiment. The
monolith already proved that fewer source-level submissions can increase GPU
gaps even when dominant kernel time is unchanged.

### 5. Use NCU before another decoder or expert-kernel rewrite

The current kernels still perform packed E2M1/E4M3FN software decode in generic
Triton IR. Gate/up and down are large enough that a 20% improvement would save
about 0.71 ms/token, but Nsight Systems cannot determine whether their next
limit is DRAM, instruction issue, MIO/LG throttling, scoreboarding, or occupancy.

If orchestration does not promote, profile gate/up, down, and the vocabulary
head with NCU before changing their implementation. A purpose-built SM80/SM86
CUDA/PTX backend remains valid behind the same `NvFp4` representation; it is a
device-specific decoder, not a custom user format.

### 6. Fuse only measured lightweight boundaries

After projection work:

- QKV reshape/RoPE where legal.
- Residual plus RMSNorm.
- Router logits plus top-k preparation.
- Sampling reductions/sorts.
- KV update bookkeeping.

These repeated non-NVFP4 kernels total about 1.01 ms/token in the new trace, so
they cannot produce the remaining speedup alone. They are useful for the final
0.05-0.15 ms after orchestration, not as a substitute for removing the two
measured bubbles.

## A credible 150 TPS budget

| Component | Current | Required target |
|---|---:|---:|
| NVFP4 projections | 5.58 ms | ~5.50 ms |
| Other repeated GPU kernels | ~1.01 ms | ~0.98 ms |
| Head -> embedding bubble | 0.42 ms | ~0.04 ms |
| Embedding -> first-pair bubble | 0.46 ms | ~0.04 ms |
| Other recurring gaps | ~0.13 ms | ~0.10 ms |
| **Traced token interval** | **~7.61 ms** | **~6.66 ms / 150 TPS** |

This asks for approximately:

- Most of the 0.882 ms in the two deterministic boundaries overlapped.
- Only about 0.1 ms of additional kernel or residual-gap improvement.
- No new artifact layout, quantization format, or global tile schedule.

That is aggressive but consistent with the trace. In contrast, 200 TPS requires a 5 ms complete loop; that likely needs the optimized SM8x decoder to approach 400–450 GB/s on the large gate/down/head workloads.

## Required promotion profiling gate

Fused QKV is accepted: the exact image produced one 640-CTA
`nvfp4_qkv_gemv` per layer, retained 40 registers/thread, passed the full GDB
workload, and improved both kernel and end-to-end rates. The next promotion gate
is the bounded lookahead pipeline. Require:

- Identical visible tokens for fixed-seed non-stop, early-stop, and
  max-token-bound requests.
- No speculation beyond the requested token budget.
- The accepted QKV kernel geometry and timing to remain intact.
- Head-to-embedding and embedding-to-first-pair gaps reported separately; the
  target is below 0.10 ms combined on a warm A40 trace.
- Terminal speculative work and cache mutation to remain request-local and be
  discarded before any externally visible state is reused.
- Decode-loop improvement outside run variance; steady-device TPS alone is not
  sufficient.

If bounded orchestration does not promote, obtain Nsight Compute reports for
these exact decode shapes before another kernel rewrite:

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
