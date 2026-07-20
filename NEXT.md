# GEMV 30% Speedup Plan

Current decode GEMV floor: 15.31 ms/token. Target: <10.7 ms/token.

## Historical lessons (from git)

| Commit | Change | Result |
|---|---|---|
| `19f5eca` | Set M=1 tiles: block_n=8, block_k=256, warps=4, stages=1 | "Recipe v2 proven geometry" |
| `5a7fabd` | Grouped M=1 → block_n=32, block_k=256 | Fixed "192us gate_up" — block_n=8 was underutilizing warps |
| `9a466a2` | block_n=64, block_k=128, warps=4, stages=4 | Preceded recipe v3 revert — tile change was valid |
| `85e5e89` | Made cache policy portable: `.cs` + `.evict_first` illegal pre-Blackwell | Using standalone `.cg` (cache=2) is valid |

The revert (`e0fe7be`) was about the **weight layout schema** ([packed K,N] → [N,K/2]), not tile sizes. The block_n=8 bottleneck is real.

## Changes

### 1. Tile rebalance (`nvfp4_backend.rs`)

| Parameter | Current | Proposed | Why |
|---|---|---|---|
| block_n | 8 (N<65536), 32 (N≥65536) | **32** (all) | 4× fewer grid blocks; 1 warp handles 16 cols with 32 threads (2:1) |
| block_k | 256 | **128** | Half register footprint per iteration; better occupancy |
| warps | 4 | **2** | Frees 2 warps/SM for more concurrent CTAs |
| stages | 1 | **2** | Software pipelining overlaps weight loads with compute |

Same changes for grouped GEMV (M=1).

### 2. Streaming loads for weights (`lib.rs` + `nvfp4.rs`)

Weight data is streamed once and never reused. Using `.cg` (cache=2 = streaming) avoids polluting L1, keeping activation data resident. This is **valid on pre-Blackwell** — `.cg` with normal eviction has no illegal hint combination (confirmed by `85e5e89`).

- Add `load_masked_streaming` to Builder (`cache=2, evict=1`)
- Use for payload and block_scales in all 3 GEMV kernels
- Keep `load_masked` (cache-all) for activations

### 3. Pre-decode global_scale outside K-loop (`nvfp4.rs`)

The `global_scale` is a per-tensor F32 scalar loaded once per program. Currently loaded inside the K-loop. Hoist it before the loop.

### 4. What NOT to change (from references)

| Reference technique | Reason skipped |
|---|---|
| GemLite `tl.gather` E2M1 LUT | Requires gather op not in builder |
| Split-K atomic_add | Partial-sum traffic wastes BW on bandwidth-bound GEMV |
| TMA / warp specialization | SM90+ only |
| Eviction policy hints | `.evict_first`/`.evict_last` with cache modifiers illegal pre-Blackwell |

## Expected speedup

| Change | Est. gain | Rationale |
|---|---|---|
| block_n=8→32 | ~10% | 4× fewer grid blocks, contiguous 32-col weight loads |
| warps=4→2 + stages=1→2 | ~15% | Higher occupancy + memory latency hiding |
| Streaming weight loads | ~5% | Less L1 thrashing, activations stay cached |
| block_k=256→128 | ~5% | Lower register pressure, better scheduling |
| **Total** | **~35%** | 15.31 ms → ~10 ms |

No schema changes. Recipe v2 weights untouched.
