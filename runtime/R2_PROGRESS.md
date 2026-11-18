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

## Next

1. Dry-run mode for `collectives` and `shard_tree`, with the equality test
   (every algorithm, the shard tree, uniform and both mixed wire policies).
2. `serving::detmath` (deterministic `ln`) and `serving::workload`.
3. `serving::kv`: block pool, LIFO free list, refcounts, block tables, prefix
   cache with LRU retention, fork + copy-on-write, swap pool.
4. `serving::kv` invariants, property tests, planted bugs.

## Open questions

None.

## Decisions made while building

(Recorded here and in the PR description of the piece that introduces them.)
