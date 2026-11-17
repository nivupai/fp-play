# FP Play runtime

**Part 2 of FP Play: from one correctly rounded operation to many devices.**
FP Play's browser lab shows how a single FP16 add, FMA or dot product rounds.
This Rust crate asks what happens when an inference layer is split across
devices and the partial sums are combined by a collective: which bits change,
why, and what it costs to make them stop changing.

Milestone **R1** is an exact, message-level AllReduce simulator plus a
tensor-parallel experiment. **R1.1** (in progress) adds mixed wire policies
and reduce-scatter / all-gather as separate collectives. The serving runtime (paged KV
cache, continuous batching) is the next milestone — see [PLAN.md](PLAN.md).

## Run it

Rust 1.80+; no dependencies, works offline.

```sh
cd runtime
cargo test --release                              # 43 tests, including the recorded-artifact checks
cargo run --release -- experiment --out results   # regenerates results/ byte for byte (~1.5 s on 10 cores)
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

| Module            | Role                                                                                                                                                                                                |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`          | FP16 / BF16 / FP32 encodings, exact decoding, ulp distance                                                                                                                                          |
| `exact`           | 768-bit fixed-point superaccumulator; **one** round-to-nearest-even per operation, with gradual underflow and overflow                                                                              |
| `ops`             | add, convert, exact products, sequential fused-chain dot, exact dot                                                                                                                                 |
| `collectives`     | sequential chain, ring, binomial tree, halving-doubling (Rabenseifner), exact-gather — executed as synchronous rounds over virtual ranks; wire policies; reduce-scatter and all-gather on their own |
| `cost`            | α–β estimate: per round, latency + largest message / bandwidth                                                                                                                                      |
| `tensor_parallel` | row-parallel `Y = X·W` (BF16 inputs, FP32 local accumulation) AllReduced in a chosen wire format                                                                                                    |
| `policies`        | R1.1: BF16 output under three wire policies, recovery of exact-gather's accuracy, reduce-scatter-only cost                                                                                          |

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

- **`fp32-rs/bf16-ag` is essentially exact**: 0.00–0.10% of elements differ
  from exact for every schedule and P, against 37–61% for `bf16`. It differs
  from the ideal for these partials (their exact sum rounded once) on at most
  0.02% (binomial tree, P = 4). It sends 1.5× the bytes of `bf16` for ring and
  halving-doubling (2× in the reduce-scatter, equal in the all-gather).
- **It beats BF16 exact-gather by far**, so the recovery ratio is 283–766%
  for P = 3…8: BF16 exact-gather still rounds each FP32 partial to BF16, and
  that input rounding, not the collective, sets its ~39% floor.
- **`bf16-wire/fp32-add`** (same bytes as `bf16`) helps much less: 26–54% of
  elements differ from exact. It closes more than the whole gap to BF16
  exact-gather at P = 3 (158–200%) and for three of four schedules at P = 4
  (ring: 90%), but only 34–75% at P = 8. Measured on relative RMS error
  instead of mismatch counts the ratios are lower (63–103% at P = 4). Under exact-gather it equals `bf16`
  exact-gather bit for bit.
- **Reduce-scatter only**: the owned slices equal the AllReduce output bit for
  bit in every case, at half the rounds and half the estimated time (e.g.
  ring P = 8: 7 rounds, ~14 µs in the illustrative model, versus 14 and ~28).

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
- **Stale-artifact detection** — tests re-run the documented command and fail
  if `results/` differs; `tp-allreduce.json` is also pinned by size and hash.

## Limits

Round-to-nearest-even only. NaNs become the format's canonical quiet NaN (no
payload propagation, unlike FP Play's FP16 engine). An exactly zero sum is
`-0` only when every addend is `-0`. The local kernel is one sequential
order, not a blocked GEMM. Inputs are synthetic (Irwin–Hall approximate
normals, chosen because they are bit-identical on every platform). The cost
model ignores link contention, compute, overlap and topology; its constants
are illustrative. Nothing here is a measurement of real hardware, an
interconnect, or an Etched system.
