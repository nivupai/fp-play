# R2 progress (serving runtime)

Working file for milestone R2, updated at every commit so the work can resume
after an interruption. The approved design is summarized in
[PLAN.md](PLAN.md#r2-decisions); this file adds the build state and the
design calls made while building.

## Pieces

| Piece                                                                                  | Branch               | State                      |
| -------------------------------------------------------------------------------------- | -------------------- | -------------------------- |
| R2a: dry-run collectives, workload, KV cache, invariants, property tests, planted bugs | `runtime-r2-serving` | merged: PR #19 (`2c5acb9`) |
| R2b: step time, scheduler, engine, metrics, reference scheduler, traces, scenarios 1–4 | —                    | not started                |
| R2c: failure injection, G1–G5, scenario 5, docs                                        | —                    | not started                |

## Done

- Vendor names in the R1 disclaimers replaced with generic wording;
  `results/` regenerated with the documented command, all four recorded JSON
  files byte-identical (only `tp-allreduce.md` changed).
- PLAN.md: approved R2 decisions, P − 1 continuation as future work, optional
  Python reference under housekeeping.
- Dry run: `collectives::schedule_rounds` and `shard_tree::schedule_rounds`
  record the same rounds without data. `tests/dry_run.rs` compares them with
  executed rounds for every algorithm × 5 policies (3 uniform, both mixed) ×
  P = 1…16 × 7 lengths, and the shard tree for S ∈ {64, 16, 5, 1}, every
  P ≤ S, including identical rejection of `bf16-wire/fp32-add`.

- `serving::model` (8B-class dimensions, illustrative device, KV capacity,
  P must divide 8 KV heads), `serving::detmath::ln`, `serving::workload`
  (Poisson arrivals, length tables, prefix groups, n ∈ {1, 2, 4}, priority
  classes, synthetic tokens) and `tests/serving_workload.rs` with a pinned
  fingerprint (`0x6dc5be0175451b62`, generated on macOS arm64).

- `serving::kv` with `KvCache::check` (invariants: conservation, refcount,
  shape, content, prefix cache, swap) and `tests/serving_kv.rs`:
  hand-derived cases, 400 seeds × 400 random operations against a plain
  model, five planted bugs.

- R2b so far: `serving::step_time` (roofline + dry-run AllReduce costs),
  `serving::scheduler` (the written policy in its module docs),
  `serving::engine` (event loop, per-step invariants 7–9 plus the KV
  check), `serving::metrics`, `serving::reference` (slow independent
  scheduler + cache). Tests: `tests/serving_engine.rs` (hand-derived
  three-request trace; 300 random workloads, every invariant every step,
  trace equal to the reference; planted bugs inside the engine) and
  `tests/serving_step_time.rs` (hand-computed step times).

## Next

R2b: golden JSON-lines traces under `tests/traces/` (+ a `traces` CLI
command to regenerate them), `serving_experiment` + report for scenarios
1–4 in `results/r2-serving.{json,md}` via the documented `experiment`
command, the stale test (target ≤ 10 s), scenario 2 must assert that
preemption occurs, README/PLAN updates. Then PR, CI, review, merge.

## Open questions

None.

## Planted bugs and the check that catches each

| #   | Bug                                                     | Caught by                                                                      | How recorded              |
| --- | ------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------- |
| D1  | Dry run records accumulator bytes instead of wire bytes | `tests/dry_run.rs` (first mixed-policy case, `sequential fp32-rs/bf16-ag P=2`) | manual mutation, reverted |

## Decisions made while building

(Recorded here and in the PR description of the piece that introduces them.)

- R2a-2. The model's vocabulary is an illustrative 128,000 tokens; the
  embedding and output head are counted in the weights (8,028,164,096
  parameters, 16.06 GB in BF16). With 80 GB, a 4 GB reserve and B = 16, the
  default capacity is 28,583 blocks (457,328 tokens) at P = 1.
- R2a-3. P must also divide the query heads, hidden width and FFN width; all
  hold whenever P divides the 8 KV heads at the default dimensions.
- R2a-4. A request's prompt is its shared prefix (if any) followed by its own
  tokens, so `prompt_len` includes the prefix. Tokens are 32-bit SplitMix64
  hashes, not a vocabulary index.
- R2a-5. Planted bugs are a hidden `kv::Mutation` switch on the cache
  (`#[doc(hidden)]`, set only by tests), so the catching tests run on every
  CI build instead of once by hand.
- R2a-6. KV invariants live in `KvCache::check` rather than a separate
  `invariants` module; scheduler, token-stream and event invariants (plan
  items 7–9) join in R2b with the engine.
- R2a-7. Prefix adoption leaves at least one prompt token to compute (the
  last token's logits are needed), so at most floor((len − 1) ÷ B) blocks
  are adopted. The free list is used before evicting a retained block.
- R2a-8. A newly full block whose key is already registered is not
  registered again (no deduplication of identical blocks); swapped-in
  blocks are private and not re-registered.
- R2a-9. The prefix hit rate is adopted blocks ÷ eligible full prompt
  blocks, counted at admission.
- R2b-1. Admission (phase 4) is skipped in any step that preempted, so a
  victim is not readmitted at once and memory pressure pauses new work.
- R2b-2. Decodes may preempt the requesting request itself when no victim
  remains; a prefill chunk that cannot get blocks is skipped for that step
  (not shrunk). A step with no work while requests run preempts the last
  running request (a counted _stall_; none occur in the tests so far).
- R2b-3. The watermark applies to admission and swap-in, not to the growth
  of running sequences.
- R2b-4. A request is rejected on arrival if its unshared worst case,
  samples × ceil((prompt + output − 1) ÷ B) blocks, plus the watermark
  exceeds the device pool, or its samples exceed the sequence limit.
- R2b-5. KV appends for scheduled work happen while the step is built
  (their tokens are known), so availability is exact at every decision.
- R2b-6. Forked siblings commit their first token from the carrier's
  prompt logits. After a recompute preemption the carrier rebuilds the
  prompt, forks, and each sibling re-prefills its own committed outputs.
- R2b-7. Priority sets the order and the victim; a waiting high-priority
  request does not preempt running work.
- R2b-8. Step time: the embedding table is not read (only indexed), the
  output head runs only for sampled tokens, attention reads each sequence's
  context and reads and writes its new tokens, and a 20 µs per-step
  overhead (illustrative) is added.
- R2b-9. TPOT is per sample (sequence), rounded down to whole ns;
  percentiles are nearest rank on integer ns.
- R2b-10. With checks on, the scheduler and token-stream invariants run
  after every step; the KV-cache check (O(pool size)) runs after every k-th
  step and after the last one. Tests use k = 1. A recorded scenario uses
  k = 1 unless its pool is large and nearly empty, and the results state k.
- R2b-11. The reference shares only the step-time model and the workload's
  token functions with the engine.
- R2a-1. The dry run shares each schedule's plan code and skips only the data
  movement, so the equality test checks the skipping, not a second schedule
  derivation. The executed schedules remain pinned by the frozen R1
  regression and the recorded JSON.
