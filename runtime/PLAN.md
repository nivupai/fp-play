# FP Play runtime — plan

Goal: a small, verifiable inference systems project in the same spirit as
FP Play. Every claim is backed by an independent check or recorded data, and
a written boundary says what is not claimed. Milestones are ordered so each
one ships something usable on its own.

## R1 — Collectives numerics ✅

Exact superaccumulator numerics; five AllReduce schedules as message-level
simulations; α–β cost model; row-parallel linear experiment; recorded
artifact with stale detection. See [README.md](README.md).

## R1.1 — Policies people actually choose ✅

Done in two parts. PR A added wire policies and reduce-scatter / all-gather,
recorded in `results/r1.1-policies.*`. PR B added the shard tree and the
sweep, recorded in `results/r1.1-shard-tree.*` and `results/r1.1-sweep.*`. The
shard-tree comparison has its own artifact instead of a table inside
r1.1-policies, so that file stays unchanged. Deferred: a chunked, pipelined
shard tree, and non-contiguous shard ownership.

- **Wire ≠ accumulator**: BF16 on the wire with FP32 accumulation inside
  reduce-scatter (2× bytes in that phase). Measure how much of exact-gather's
  accuracy it recovers, and at what cost.
- **Reduce-scatter only** (sequence-parallel output) and all-gather as
  separate collectives.
- **A cheaper deterministic AllReduce**: a fixed logical reduction tree over
  shards, independent of the physical topology, compared with exact-gather.
- **Sensitivity**: sweep seeds, hidden size and P up to 64; report spread,
  not one run.

## R2 — Serving runtime (idea #3) ✅

A deterministic discrete-event simulator in Rust:

- Request lifecycle: queued → prefill → decode → finished / preempted / failed.
- **Paged KV cache**: fixed-size blocks, a free-list allocator, per-sequence
  block tables, reference-counted prefix sharing, copy-on-write.
- **Continuous batching scheduler**: a per-step token budget mixing prefill
  chunks and decode tokens; FCFS with a configurable priority policy.
- **Preemption**: recompute versus swap, with block accounting checked after
  every step.
- **Failure handling**: inject a rank failure mid-collective; abort, retry or
  drain, with invariants (no leaked blocks, no duplicated tokens).
- Metrics: time to first token, time per output token, throughput,
  p50/p99 under Poisson arrivals with configurable prompt/output lengths.
- Step time comes from the R1 cost model plus a simple compute model, so
  tensor-parallel communication shows up in latency.

Verification: property tests on allocator invariants, a reference scheduler
written as a slow obvious loop, and recorded traces.

Progress, the approved design answers and every design call made while
building are tracked in [`R2_PROGRESS.md`](R2_PROGRESS.md); the decisions
are also summarized below.

### R2 decisions

Approved before building (answers 1–9 and additions A–C):

1. Model: 8B-class public dimensions (hidden 4096, 32 layers, 32 query and
   8 KV heads, head dim 128, FFN 14336, BF16 weights and KV). Device
   constants are illustrative, not a named device.
2. Freed blocks holding a cached prefix stay reusable until evicted, LRU,
   ties broken by block ID.
3. Parallel sampling with n ∈ {1, 2, 4}, so copy-on-write does real work.
4. Drain: finish admitted work and reject new arrivals after recovery.
5. Reference scheduler in Rust only.
6. No compute/communication overlap.
7. The r2-serving stale test may take about 10 s.
8. Preemption victim: lowest priority, then latest arrival.
9. Vendor names in disclaimers replaced with generic wording
   (`tp-allreduce.json` byte-identical).

- A. Device memory is a per-scenario setting; the preemption scenario's
  memory forces regular preemption and a test asserts it does.
- B. P must divide the KV head count (8); other values are rejected.
- C. The dry-run equality test covers every algorithm, the shard tree and
  both wire policies.

Made while building (details and rationale in
[`R2_PROGRESS.md`](R2_PROGRESS.md)):

- R2a: the dry run reuses each schedule's plan code and skips only data
  movement; vocabulary 128,000 (illustrative) with embedding and head
  counted (8,028,164,096 parameters); P must also divide the query heads,
  hidden and FFN widths; `prompt_len` includes the shared prefix and
  tokens are 32-bit hashes; planted bugs use a hidden mutation switch so
  their tests run in CI; KV invariants live in `KvCache::check`; prefix
  adoption leaves at least one prompt token and the free list is used
  before evicting; identical full blocks are not deduplicated and
  swapped-in blocks are not re-registered; prefix hit rate = adopted ÷
  eligible full prompt blocks.
- R2b: no admission in a step that preempted; decodes may preempt their
  own request when no victim remains, a prefill chunk without blocks is
  skipped for the step, and an empty step preempts the last running
  request (a counted stall); the watermark gates admission and swap-in
  only; requests that could never fit are rejected on arrival; KV appends
  happen while the step is built; forked siblings take their first token
  from the carrier's prompt logits and re-prefill their own outputs after a
  recompute; priority sets order and victim only; step time skips the
  embedding read, runs the output head only for sampled tokens and adds a
  20 µs overhead; TPOT is per sample, rounded down; the KV check runs every
  k steps (k = 1 in tests and the pressure and prefix scenarios, 256 for
  the nearly empty 80 GB pools) while scheduler and token-stream checks
  run every step; the reference shares only the step-time model and token
  functions; golden traces have their own `traces` command; scenario
  settings are listed in `results/r2-serving.md`; the chain-key hash is a
  multiply–rotate loop with hits verified against stored tokens.
- R2c: a failed step is charged its share of compute and communication up
  to the failing collective plus detection and recovery, and commits
  nothing; faults are given by step index; retry limit per request; abort
  and retry leave swapped and waiting requests alone; drain retries running
  requests, then rejects never-admitted requests and later arrivals; G1 is
  checked right after a failure, G2 every step, G3 and G4 at the end, G5 by
  rerunning; interrupted steps count as `fault_ns`, not compute or
  communication; scenario 5 runs at 20 req/s with the failure in step 1500
  (10 req/s left only 2 requests in flight).

### R2 future work

- After a rank failure, continue serving at P − 1 instead of draining. The
  shard tree makes the P − 1 numerics bit-identical to P; it needs KV
  re-sharding.

## R3 — End to end: does sharding change the tokens? ✅

A tiny Llama-style decoder (RMSNorm, RoPE, SwiGLU, grouped-query attention)
run tensor-parallel across virtual devices inside the R2 runtime, with the
exact numerics from R1.

Headline experiments:

- Does greedy decoding diverge across P, AllReduce schedule, or **batch
  composition** (batch invariance)? After how many tokens?
- What does making it deterministic cost in latency?

Uses random weights by default; a small real checkpoint is optional (it needs
network access to download).

Progress, the approved answers and the design calls made while building are
tracked in [`R3_PROGRESS.md`](R3_PROGRESS.md). Recorded in
`results/r3-decode.*`.

### R3 decisions (details in R3_PROGRESS.md)

- Approved: hidden 256, 2 layers, 16/8 heads, FFN 704, vocabulary 2048;
  native binary32 local kernels; BF16 weights and KV, binary32 residual,
  attention and logits; no latency recomputation (R2's pairing is cited);
  greedy, n = 1, fixed length; baseline P = 1 with the same policy; seven
  batch variants plus a P = 1 control; one AllReduce buffer per step; a
  logit scale at two values; divergence against the top-2 margin.
- Made while building: a native binary32/BF16 fast path in `ops` (checked
  against the exact path); the model runs at step completion and K/V live
  in a paged tensor store replaying the cache's block copies; column-parallel
  work computed once; the prediction corrected before measuring (only the
  ring, not halving-doubling, can depend on batch composition); a positive
  logit scale preserves the argmax, so it cannot affect greedy tokens;
  measured as neutral.

### R3 future work

- Realistic, confident margins: train a tiny model deterministically in
  the repo on a synthetic task (copy or count), with fixed-order binary32
  training steps so the weights are bit-identical on every platform, then
  rerun r3-decode on it. A logit scale cannot do this: a positive scale
  preserves the argmax.
- Batch-shape-dependent local kernels (tiling, split-K) to model GPU-style
  local batch variance.

## R4 — Per-layer precision policies (idea #5) ✅

Choose a storage format and accumulation width per layer (FP32 / BF16 /
FP16, later FP8 / INT8 with scales). Measure logit drift and perplexity
against FP32, reusing FP Play's mixed-precision semantics as the reference.
Report which layers are sensitive.

Done: `results/r4-precision.*`; plan and decisions in
[`ROADMAP_PROGRESS.md`](ROADMAP_PROGRESS.md). FP8/INT8 with scales remain
future work.

## R5 — Dataflow mapping simulator (idea #2) ✅

Describe a transformer block as a static graph and map it onto a toy
fixed-function pipeline (compute units, SRAM, HBM bandwidth). Report
utilization, stall reasons and a roofline for prefill versus decode. FP Play's
Nangate45 delays can feed one unit model, clearly labeled as an assumption.

Done: `results/r5-dataflow.*`; decisions in
[`ROADMAP_PROGRESS.md`](ROADMAP_PROGRESS.md). Future work: a mapping built
for matrix–vector products (decode), operator fusion and overlap between
ops.

## R6 — FP Play UI integration

- A `#collectives` workspace that reads the recorded JSON: an
  animated-on-request schedule view (rank × round grid, text alternative),
  per-element association trees, and the error-versus-P chart.
- Landing and overall UI refresh (deferred by request).
- Optional: compile the crate to WebAssembly so the browser runs the real
  simulator rather than a JavaScript copy (once a wasm toolchain is
  available).

## Housekeeping

- ✅ CI job running `cargo fmt --check`, `cargo clippy -D warnings`,
  `cargo test --release` and both Python vector comparisons, with actions
  pinned to commit IDs and the toolchain installed from
  `rust-toolchain.toml`.
- Python bindings (PyO3) once crates can be fetched; the crate has no
  dependencies until then.
- Optional: an independent stdlib-only Python reference for the R2 serving
  scheduler, with a CI trace comparison like the vector scripts.
