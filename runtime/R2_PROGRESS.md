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

## Next

2. `serving::detmath` (deterministic `ln`) and `serving::workload`.
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

- R2a-1. The dry run shares each schedule's plan code and skips only the data
  movement, so the equality test checks the skipping, not a second schedule
  derivation. The executed schedules remain pinned by the frozen R1
  regression and the recorded JSON.
