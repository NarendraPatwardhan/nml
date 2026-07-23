# Current inference-server efficiency requirements

This document covers the efficiency of ordinary target-model execution only.
Future capabilities in `NEXT.md` are deliberately excluded: they cannot be
used to conceal overhead in batching, prefill, paging, or decode.

## 1. What C, B, and Q mean

- **C (concurrency)** is the number of client requests currently in flight.
- **B (batch family)** is the static GPU row capacity selected for one model
  invocation.
- **Q (query family)** is the static number of tokens processed per row.
- **Active rows/tokens** are the useful portion of that static rectangle.

For example, three decoding clients select `B4,Q1`; one row is padding. A
106-token prompt now selects `B1,Q128`; 22 tokens are padding.

```text
client arrivals (C)
        |
        v
continuous scheduler
        |
        +-- prefill: choose B and Q for prompt chunks
        |
        `-- decode: choose smallest B containing active rows
                         |
                         v
               one generic model pipeline
                         |
                         v
              process-wide paged KV arena
```

The product goal is not to maximize one synthetic number. It is to keep
interactive B1 fast while increasing useful aggregate throughput as
concurrency creates larger batches.

## 2. Governing design

There is one serving architecture across every retained batch family:

```text
active request membership
          |
          v
smallest compiled B family
          |
          v
resident token/RNG/position/page-table slab
          |
          v
embedding -> 24 layers -> sampling
          |
          v
one compact token/RNG download per visible step
```

Static family specialization is allowed inside graph construction and Triton
lowering. It must not create a second scheduler, cache owner, request
lifecycle, or B1-only engine.

The generic path must satisfy these invariants:

- all families share one process-wide paged K/V arena;
- inactive rows and padded prompt positions create no MoE assignments and
  perform no cache writes;
- stable membership keeps token, RNG, position, page table, graph arguments,
  and the bounded layer-prefix dependency on device;
- host work occurs at visible-token boundaries and membership changes, not
  between every pair of layer graphs;
- every family reuses its compiled executable bindings and result workspace;
- the steady transfer contract is zero H2D and one compact D2H per step; and
- membership changes return to the same generic path with a new smallest
  family.

## 3. Interactive cold request

```text
106 prompt tokens
       |
       v
B1,Q128 prefill
       |
       v
first sampled token
       |
       v
stable B1 decode -> token -> token -> token
```

End-to-end latency includes prefill, first-token work, every decode step, and
the server boundary. Decode-only compute is useful diagnostic evidence but is
not the product result.

### 3.1 Padded prompt MoE work

The recovered trace used `Q256` for a 106-token prompt:

```text
[ 106 real positions ][ 150 padded positions ]
```

Those 150 positions still entered routing and expert computation. The trace
attributed about 264 ms, or 84.5% of prefill, to combined gate/up and down
expert work.

The implemented repair has two parts:

1. `Q128` is now a retained prefill family, reducing this example from 150
   padded positions to 22.
2. The active `[B,Q]` mask is flattened into routed MoE. Inactive tokens receive
   expert ID `-1`, touch no expert weights, and produce exact zero routed
   output. Sparse fixed schedules may retain a masked route slot when that
   avoids runtime compaction.

The mask is semantic, not merely a final output select. Portable IR remains
available, while CUDA keeps the compact expert schedule. Masked B1/B2 decode
retains the sparse one-block-per-selected-route path without runtime
compaction. Inactive routes keep expert ID `-1`; the grouped Triton kernels
reject those blocks before weight loads or dot products.

### 3.2 Decomposed paged-cache updates

The recovered B1 trace showed approximately 552 kernels per token versus 468
on the accepted control. Runtime masks, index construction, and two
scatter-based K/V updates contributed 84 extra kernels and about 134
microseconds per token.

The implemented repair is one paired generic paged append:

```text
active row + position + page table + K row + V row
                         |
                         v
              paired Triton append
              - resolve page once
              - write K and V
              - skip inactive rows
                         |
                         v
               donated global K/V buffers
```

The portable meaning is still expressed by two StableHLO scatters. CUDA lowers
the pair to one typed custom call whose two results alias the two input cache
buffers. The operation is identical across batch families.

### 3.3 Graph-boundary bubbles

The accepted trace spent about 66 microseconds per token at recurring
layer-pair boundaries. The first generic serving trace spent about 153
microseconds. The largest hole occurred after the five-pair lookahead, when
the host decoded results, updated bookkeeping, re-entered the scheduler,
rebuilt a slab, and only then submitted more GPU work.

The implemented stable-batch transaction now behaves as follows:

```text
current nine-pair prefix already queued
                 |
                 v
remaining pairs -> head/sampling
                 |
                 +---- compact result D2H ----> host visible token
                 |
                 `---- next embedding + nine pairs queued immediately
```

The serving head also produces the donated next batch slab. It updates token,
RNG, position, and sequence length on device. While membership and page-table
bytes are unchanged, the next token therefore needs no H2D reconstruction.

The engine continues the same stable batch without scheduler re-entry until a
new command, cancellation, deadline, output backpressure, terminal row, or
page-table/membership change requires replanning. The rule is generic for
every retained B family.

### 3.4 Repeated allocation and binding

Compiled family components and their arguments already have process lifetime.
The implementation now also owns one reusable compact result-download slice
per family. During a stable period, the donated batch slab and cache buffers
retain identity across steps.

This removes per-token result allocation and avoids reconstructing graph
bindings during ordinary stable decode. It does not claim CUDA Graph capture;
that requires separate runtime support and evidence.

## 4. Concurrent conversations

```text
time ------------------------------------------------------------>

active rows:      1       3       8       7       4       2
selected family: B1      B4      B8      B8      B4      B2
inactive rows:    0       1       0       1       0       0
```

The same repairs apply:

- inactive rows are excluded from MoE schedules;
- inactive rows do not append cache state;
- stable B4/B8 periods retain their complete device batch state;
- a row joining or leaving causes one token-boundary replan; and
- survivors then continue in the new smallest family.

There is no B1-to-batch state export protocol because there is no separate B1
engine. The stable lane itself changes family.

The recovered C2/C4/C8 trace answered the small-B dispatch question. Stable
decode selected the intended B2/B4/B8 families, but those families fell off
both compact decode paths:

- Q, K, and V became three separate NVFP4 matrix launches per layer, averaging
  roughly 83, 85, and 189-192 microseconds respectively across B2/B4/B8,
  while accepted B1 fused all three projections in roughly 28 microseconds;
- expert gate/up used a `[B*4,45]` grid and averaged 795, 969, and 1,368
  microseconds at B2, B4, and B8, versus roughly 98 microseconds at B1; and
- expert down averaged 303, 374, and 504 microseconds, versus roughly 48
  microseconds at B1.

The cause was geometric, not a dense-weight conversion: through B8 there are
only 8/16/32 routed assignments across 32 experts, so the grouped matrix
kernel's 16-row tile was usually almost empty. The implementation now selects
the compact selected-route GEMV family through `M=8`, authors one fixed block
per route, and fuses all small-batch rows plus Q/K/V projections into one
weight-tile-major, row-minor launch. Adjacent QKV CTAs therefore read the same
payload/scales for different active rows and can reuse them from L2. B1 is the
same member of this generic dispatch, not a separate server route. B16 and
larger retain grouped matrix lowering where route collisions can supply useful
rows.

BuildBuddy proves TTIR construction, CUDA lowering, and the complete server
build. The next A40 run must measure the new B2/B4/B8 kernel times and determine
the empirical crossover; no throughput improvement is claimed from compile
evidence alone.

## 5. Long prompt during decode

```text
prefill chunk -> decode -> decode -> prefill chunk -> decode -> ...
```

Decode-first chunking prevents one long prompt from blocking all interactive
users. The current efficiency repairs improve the work inside those chunks:

- `Q128` reduces common final-chunk padding;
- mask-aware MoE eliminates expert work for remaining padding;
- the paired append eliminates decomposed K/V update overhead; and
- stable decode batches remain resident around interleaved prefill.

The A40 matrix must report both TTFT and p95 TPOT under mixed prompt/decode
load. Isolated prefill throughput cannot prove decode responsiveness.

## 6. Slow or disconnected client

Bounded response channels prevent a slow client from blocking unrelated
device work. Backpressure, cancellation, a deadline, or disconnect ends the
stable period at a visible-token boundary and lets the scheduler remove that
row. This is intentional product work, not avoidable per-token overhead.

## 7. Implemented solution set and remaining proof

Implemented in the current tree:

1. mask-aware routed MoE for prompt padding and inactive batch rows;
2. a paired Triton K/V paged append for every retained batch family;
3. a donated device-resident batch slab advanced by the serving head;
4. generic stable execution with nine-pair lookahead;
5. direct stable-batch continuation without ordinary scheduler re-entry;
6. reusable per-family executable bindings and result workspaces;
7. the `Q128` prefill family; and
8. weight-tile-major fused QKV and selected-route expert GEMV for every
   retained `B<=8` decode family.

BuildBuddy CPU/IR/serve contracts and the full CUDA server build validate
construction and lowering. They do not establish runtime performance.

The parity profile deliberately compiles only `B={1,2,4,8}` and
`Q={16,128,256}`, cutting the serving family count from 30 to 16. It retains
the exact C1/Q128 control, exercises real B2-B8 batching, and keeps Q256
chunked prefill. After B1 recovers at least 150 end-to-end tokens/s, expand the
same generic machinery to the production B/Q envelope.

The first corrected-ABI image compiled and became ready, then failed on the
first decode before its head launch. This was not a kernel-performance result.
The lane and the already-submitted embedding/layer argument sets retained
aliases of the slab that the head must donate, so runtime ownership validation
rejected the head deterministically. The implementation now transfers the sole
lane owner into the body and releases every read-only component binding after
submission. This exact ownership transition is covered on CPU before another
paid A40 run.

The remaining current-phase proof is one published immutable image on A40:

- the exact 106+320 C1 end-to-end control must recover at least 150 tokens/s;
- B1-B8 must be exercised through real concurrent arrivals;
- Nsight must verify the paired append, masked expert schedule, transfer
  contract, and removal of the recurring orchestration hole;
- aggregate output throughput, TTFT, TPOT, queue time, batch histogram, page
  use, and GPU busy time must be reported together; and
- any small-B kernel tuning must be selected from that trace rather than from
  speculation.

No future feature may be credited toward these gates.
