# R2 progress (serving runtime)

Working file for milestone R2, updated at every commit so the work can resume
after an interruption. The approved design is summarized in
[PLAN.md](PLAN.md#r2-decisions); this file adds the build state and the
design calls made while building.

## Pieces

| Piece                                                                                  | Branch               | State       |
| -------------------------------------------------------------------------------------- | -------------------- | ----------- |
| R2a: dry-run collectives, workload, KV cache, invariants, property tests, planted bugs | `runtime-r2-serving` | in progress |
| R2b: step time, scheduler, engine, metrics, reference scheduler, traces, scenarios 1–4 | —                    | not started |
| R2c: failure injection, G1–G5, scenario 5, docs                                        | —                    | not started |

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

## Next

3. `serving::kv`: block pool, LIFO free list, refcounts, block tables, prefix
   cache with LRU retention, fork + copy-on-write, swap pool.
4. `serving::kv` invariants, property tests, planted bugs.

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
- R2a-1. The dry run shares each schedule's plan code and skips only the data
  movement, so the equality test checks the skipping, not a second schedule
  derivation. The executed schedules remain pinned by the frozen R1
  regression and the recorded JSON.
