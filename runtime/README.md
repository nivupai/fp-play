# FP Play runtime

**Part 2 of FP Play: from one correctly rounded operation to many devices.**
FP Play's browser lab shows how a single FP16 add, FMA or dot product rounds.
This Rust crate asks what happens when an inference layer is split across
devices and the partial sums are combined by a collective: which bits change,
why, and what it costs to make them stop changing.

Milestone **R1** is an exact, message-level AllReduce simulator plus a
tensor-parallel experiment. **R1.1** adds mixed wire policies, reduce-scatter /
all-gather as separate collectives, a deterministic AllReduce over a fixed tree
of shards, and a sweep across seeds. **R2**, a deterministic serving
simulator (paged KV cache, continuous batching), is in progress — see
[PLAN.md](PLAN.md) and [R2_PROGRESS.md](R2_PROGRESS.md).
For a two-page summary of the results, read the
[runtime case study](../docs/RUNTIME_CASE_STUDY.md).

## Run it

Rust 1.97.0, pinned in [`rust-toolchain.toml`](rust-toolchain.toml) (rustup installs it on first use; CI installs from the same file). No dependencies; works offline.

```sh
cd runtime
cargo test --release                              # 83 tests, including the recorded-artifact checks
cargo run --release -- experiment --out results   # regenerates results/ byte for byte (~18 s on 10 cores)
cargo run --release -- schedule --algorithm ring --world 4 --len 8
cargo run --release -- schedule --algorithm ring --world 4 --len 8 --policy fp32-rs/bf16-ag
```

`schedule` prints a schedule round by round:

```
round  1 [reduce-scatter] 0→1 (2)  1→2 (2)  2→3 (2)  3→0 (2)
...
round  6 [all-gather] 0→1 (2)  1→2 (2)  2→3 (2)  3→0 (2)
```

## What it models

| Module             | Role                                                                                                                                                                                                |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`           | FP16 / BF16 / FP32 encodings, exact decoding, ulp distance                                                                                                                                          |
| `exact`            | 768-bit fixed-point superaccumulator; **one** round-to-nearest-even per operation, with gradual underflow and overflow                                                                              |
| `ops`              | add, convert, exact products, sequential fused-chain dot, exact dot                                                                                                                                 |
| `collectives`      | sequential chain, ring, binomial tree, halving-doubling (Rabenseifner), exact-gather — executed as synchronous rounds over virtual ranks; wire policies; reduce-scatter and all-gather on their own |
| `cost`             | α–β estimate: per round, latency + largest message / bandwidth                                                                                                                                      |
| `tensor_parallel`  | row-parallel `Y = X·W` (BF16 inputs, FP32 local accumulation) AllReduced in a chosen wire format                                                                                                    |
| `shard_tree`       | R1.1: a fixed binary tree over S logical shards, any P ≤ S ranks, single-port rounds; output bits independent of P                                                                                  |
| `shard_experiment` | R1.1: the shard tree on the R1 problem for every P from 1 to S, against ring and exact-gather                                                                                                       |
| `sweep`            | R1.1: min / median / max across 8 seeds × 3 hidden sizes × P up to 64 × 4 policies × every schedule                                                                                                 |
| `policies`         | R1.1: BF16 output under three wire policies, recovery of exact-gather's accuracy, reduce-scatter-only cost                                                                                          |

Every addition a simulated rank performs is a real correctly rounded addition
in the policy's accumulator format, and every message is converted (one
correct rounding) to its wire format. The association order comes from the schedule being
executed, not from a separate formula, so the output bits are an observation
of the schedule.

## Recorded result

[`results/tp-allreduce.md`](results/tp-allreduce.md) (and the JSON beside
it): 8 tokens × 4096 hidden × 512 outputs, P ∈ {2, 3, 4, 8}.

- **With two devices every schedule agrees**: there is only one way to add
  two partials. From P = 3 on they diverge; at P = 8, any two of the four
  reduction schedules (sequential, ring, binomial-tree, halving-doubling)
  disagree on **51–58% of output elements**, in FP32 and in BF16 alike, and
  exact-gather disagrees with each of them on 46–52%.
- **BF16 on the wire**: error against the exact result grows with P for
  every reduction schedule (sequential: 37% of elements wrong at P = 2, 61% at
  P = 8). Exact-gather stays nearly flat (37% at P = 2, 39–40% from P = 3 to 8) because the collective itself adds only one rounding, but at P = 8 each
  rank sends 4× the bytes that ring does. Its floor is not the collective:
  each FP32 partial is rounded to BF16 before it goes on the wire, which is why
  exact-gather sits near 39% while the unsharded kernel differs from exact on
  only 0.05% of elements.
- **FP32 on the wire**: every schedule is within 4×10⁻⁷ relative RMS error
  of exact, yet the bits still depend on the schedule.
- **Top-1 changes: 0** of 8 tokens in every configuration. Bit differences at
  this magnitude did not flip an argmax over 512 outputs here; whether they
  flip greedy decoding over a real vocabulary is an R3 experiment.

## R1.1 result: FP32 where it can help

[`results/r1.1-policies.md`](results/r1.1-policies.md) (and the JSON beside
it): the same problem, BF16 output, three modeled wire policies (not
descriptions of any library): `bf16` (R1), `fp32-rs/bf16-ag` (FP32 partials
and reduce phases, one rounding to BF16, BF16 gather) and
`bf16-wire/fp32-add` (FP32 additions, every message BF16).

Each policy applies to all five schedules: every round is a reduce phase
(payload combined or summed) or a gather phase (payload copied); the
results page has the per-schedule table. Only ring and halving-doubling can
stop after the reduce-scatter.

- **`fp32-rs/bf16-ag` is as good as the ideal**: at P = 8, 55–61% of
  elements wrong under `bf16` become 0.02% for every reduction schedule,
  for 1.5× the bytes per rank in a ring AllReduce (2× in the reduce-scatter,
  equal in the all-gather). Across P = 2…8 it is 0.00–0.10% wrong and closes
  100% of the gap from `bf16` to the ideal for these partials (their exact sum
  rounded once to BF16), on mismatch count and on RMS error, except 99.9% for
  binomial-tree at P = 4.
- **`bf16-wire/fp32-add`** sends the same bytes as `bf16` and helps much less:
  26–54% of elements wrong, closing 12–30% of the gap to the ideal (11–29%
  on RMS error), less as P grows. Under exact-gather it equals `bf16`
  exact-gather bit for bit.
- **BF16 exact-gather is not the target**: it stays near 39% wrong because it
  rounds each FP32 partial to BF16 before its one rounding. Ratios against it
  exceed 100%, so they appear only in the JSON.
- **Reduce-scatter only**: the owned slices equal the AllReduce output bit for
  bit in every case, at half the rounds and half the estimated time (e.g.
  ring P = 8: 7 rounds, ~14 µs in the illustrative model, versus 14 and ~28).

## R1.1 result: a deterministic AllReduce over a fixed tree of shards

[`results/r1.1-shard-tree.md`](results/r1.1-shard-tree.md): the hidden
dimension is split into **S = 64 logical shards** (`--shards` changes S), each
an FP32 partial, and one fixed binary tree combines them. At level k, node i
absorbs node i + 2^k. P devices own contiguous blocks of shards. The
invariant: every tree node is one accumulator-format addition, whether its
children are on one device or two. So the reduce wire must carry accumulator
values exactly, and `bf16-wire/fp32-add` is rejected.

- **It buys determinism, not accuracy.** Under `bf16` it is the least
  accurate option up to P = 8: 66.8% of elements wrong, because each of the
  64 shard partials is rounded to BF16 before the tree. In the sweep, from
  P = 16 the deeper `bf16` sequential and ring reductions become worse still.
  The shard tree is meant to be paired with `fp32-rs/bf16-ag`, which gets
  0.02% wrong (one element of 4,096).
- **The same bits for every P from 1 to 64**, under both policies, on every
  rank. This holds by construction: the tree and every node's formats depend
  on S alone. The checks verify that the distributed schedule implements that
  tree faithfully:
  - the recorded problem for every P from 1 to 64;
  - raw patterns (NaN, infinities, signed zeros) for S ∈ {64, 48, 16, 5, 1};
  - agreement with the binomial-tree reduce at P = S;
  - the sweep (432 of 432 comparisons).
- **It reproduces a single-device shard-tree kernel** (S partials, then the
  same tree on one device), not the plain single-device kernel that sums the
  whole hidden dimension in one chain. Under `fp32-rs/bf16-ag` the two differ
  on 0.07% of elements.
- **Cost depends on P's shape.** With P a power of two, log2 P tree levels
  cross devices. At P = 8 under `fp32-rs/bf16-ag` that gives 6 rounds and 32 KB
  per rank (~12.7 µs, illustrative model), against 7 rounds and 115 KB for
  exact-gather and 14 rounds for ring. Any other P touches all six levels: at
  P = 3 it takes 8 rounds against exact-gather's 2.
- **Serialization never added a round.** With contiguous ownership each device
  sends and receives at most one message per tree level. The general
  serializer is still exercised on conflicting message sets in the tests.

## R1.1 result: the spread across seeds

[`results/r1.1-sweep.md`](results/r1.1-sweep.md): 8 seeds × hidden
{1024, 4096, 16384} × P {2, 3, 4, 8, 16, 32, 64} × 4 policies × every
schedule, at 1,024 elements per run. Each cell is min / median / max over the
seeds.

- **`fp32-rs/bf16-ag`**: the median is 0.0–0.15% of elements wrong for
  every schedule, P and hidden size; the worst single seed is 0.29% (3 of
  1,024 elements).
- **`bf16`**: the reduction schedules' medians are 56–60% wrong at P = 8 and
  66–81% at P = 64. Exact-gather stays at 37–44%, and the shard tree at
  66–67% for every P.
- **`bf16-wire/fp32-add`**: 44–54% at P = 8 and 54–80% at P = 64.
- **`fp32`** (FP32 output): 37–97% of elements are not the correctly rounded
  FP32 result, yet relative RMS error stays below 1×10⁻⁶. Max ulp reaches
  about 1.6 million on elements whose exact value is near zero, where an ulp
  is tiny; R1 shows the same effect at smaller scale.
- **Top-1**: 35 of 3,768 runs changed the top-1 output of one token (never
  more). All of them are under `bf16` (25) or `bf16-wire/fp32-add` (10).
  None are under `fp32` or `fp32-rs/bf16-ag`, and none involve the shard
  tree. 22 come from one seed (`0xf991ae`). The sweep page lists every one,
  with the unsharded kernel's top-1 lead over its runner-up: the 6 tokens
  that changed led by only 1–4 BF16 ulps, against a median of 16 ulps over
  all 96 tokens, so these are near-ties.

## R2 (in progress): the serving simulator

The first piece, R2a, is a tested paged-KV-cache library with no scheduler
yet:

| Module                                                        | Role                                                                                                                                              |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `serving::model`                                              | 8B-class dimensions (not a specific released model), illustrative device constants, KV capacity per device; P must divide the 8 KV heads          |
| `serving::detmath`                                            | `ln` from IEEE basic operations only, so Poisson arrivals match on every platform                                                                 |
| `serving::workload`                                           | Poisson arrivals, prompt and output length tables, shared system prompts, n ∈ {1, 2, 4} samples, priority classes, deterministic synthetic tokens |
| `serving::kv`                                                 | Block pool (LIFO free list, refcounts), block tables, prefix cache with LRU retention, fork and copy-on-write, host swap pool, invariant checks   |
| `collectives::schedule_rounds`, `shard_tree::schedule_rounds` | Dry-run schedules: the rounds of a collective without moving data, for step-time costing                                                          |

With the default device (80 GB, 4 GB reserve) and 16-token blocks, the KV
capacity is 28,583 blocks (457,328 tokens) at P = 1. Evidence:
`tests/dry_run.rs` (dry-run rounds equal executed rounds for every algorithm,
policy and P ≤ 16, and the shard tree for every P ≤ S), `tests/serving_kv.rs`
(hand-derived cases, 400 random operation sequences checked against a plain
model after every operation, and five planted bugs each caught by a named
invariant) and `tests/serving_workload.rs` (a pinned workload fingerprint that
CI checks on Linux, and `ln` against the host).

## Evidence

- **Independent oracles** — Python `Fraction` reference vectors
  (`scripts/gen_sum_vectors.py`, 18,000 sums of 1–12 terms across all three
  formats, shares no code with the crate); 400,000 FP32 additions and
  ~400,000 binary64→FP32 conversions against the host's IEEE hardware; BF16
  addition and FP32→BF16 conversion against the standard integer bit-rounding
  identity; the fused chain against a host FP32 chain where products are
  exact.
- **Exhaustive** — all 65,536 FP16 and all 65,536 BF16 patterns round-trip
  through the exact path and through binary64.
- **Schedules** — exact results on integer data for P = 1…9 and lengths
  0, 1, 2, 5, 9, 16 and 37 in every format; the single-port model (each rank sends and receives
  at most one message per round) and the round counts for P = 2…16;
  bandwidth-optimal bytes for ring and halving-doubling; a hand-derived case
  where ring folds each chunk from a different rank.
- **Wire policies** — the uniform policy reproduces R1 exactly against a
  frozen copy of the R1 simulator (2,898 cases: bits, rounds, transfers,
  additions); a hand-derived three-rank case for each policy; measured
  rounding counts per element against hand-derived formulas; all-gather after
  reduce-scatter equals the AllReduce in bits, rounds and roundings.
- **Shard tree** — for S ∈ {64, 48, 16, 5, 1}, every P from 1 to S gives the
  bits, rounding counts and additions of P = 1, with one message per node
  whose children cross devices. Rounds are single-port, and the scheduled
  count is reported. At P = S the tree agrees bit for bit with the
  independently implemented binomial-tree reduce. A separate script,
  `scripts/gen_tree_vectors.py`, evaluates the same tree with `Fraction`
  arithmetic: 6,000 cases over FP16, BF16, FP32 and FP32 into BF16, with S up
  to 100, including infinities, NaN and signed zeros. CI regenerates and
  compares them. Planting a bug that rounds remote children through BF16 fails
  both checks.
- **Stale-artifact detection** — tests re-run the documented command and fail
  if `results/` differs; `tp-allreduce.json` is also pinned by size and hash.

## Limits

Round-to-nearest-even only. NaNs become the format's canonical quiet NaN (no
payload propagation, unlike FP Play's FP16 engine). An exactly zero sum is
`-0` only when every addend is `-0`. The local kernel is one sequential
order, not a blocked GEMM. The shard tree moves whole buffers; a chunked,
pipelined tree is not modeled. The sweep covers one synthetic input
distribution. Inputs are synthetic (Irwin–Hall approximate
normals, chosen because they are bit-identical on every platform). The cost
model ignores link contention, compute, overlap and topology; its constants
are illustrative. Nothing here is a measurement of real hardware, an
interconnect, or any particular vendor's system.
