# FP Play runtime

**Part 2 of FP Play: from one correctly rounded operation to many devices.**
FP Play's browser lab shows how a single FP16 add, FMA or dot product rounds.
This Rust crate asks what happens when an inference layer is split across
devices and the partial sums are combined by a collective: which bits change,
why, and what it costs to make them stop changing.

Milestone **R1** (this crate today) is an exact, message-level AllReduce
simulator plus a tensor-parallel experiment. The serving runtime (paged KV
cache, continuous batching) is the next milestone — see [PLAN.md](PLAN.md).

## Run it

Rust 1.80+; no dependencies, works offline.

```sh
cd runtime
cargo test --release                              # 23 tests, including the recorded-artifact check
cargo run --release -- experiment --out results   # regenerates results/ byte for byte (~4 s)
cargo run --release -- schedule --algorithm ring --world 4 --len 8
```

`schedule` prints a schedule round by round:

```
round  1 [reduce-scatter] 0→1 (2)  1→2 (2)  2→3 (2)  3→0 (2)
...
round  6 [all-gather] 0→1 (2)  1→2 (2)  2→3 (2)  3→0 (2)
```

## What it models

| Module | Role |
| --- | --- |
| `format` | FP16 / BF16 / FP32 encodings, exact decoding, ulp distance |
| `exact` | 768-bit fixed-point superaccumulator; **one** round-to-nearest-even per operation, with gradual underflow and overflow |
| `ops` | add, convert, exact products, sequential fused-chain dot, exact dot |
| `collectives` | sequential chain, ring, binomial tree, halving-doubling (Rabenseifner), exact-gather — executed as synchronous rounds over virtual ranks |
| `cost` | α–β estimate: per round, latency + largest message / bandwidth |
| `tensor_parallel` | row-parallel `Y = X·W` (BF16 inputs, FP32 local accumulation) AllReduced in a chosen wire format |

Every addition a simulated rank performs is a real correctly rounded addition
in the wire format. The association order comes from the schedule being
executed, not from a separate formula, so the output bits are an observation
of the schedule.

## Recorded result

[`results/tp-allreduce.md`](results/tp-allreduce.md) (and the JSON beside
it): 8 tokens × 4096 hidden × 512 outputs, P ∈ {2, 3, 4, 8}.

- **With two devices every schedule agrees**: there is only one way to add
  two partials. From P = 3 on they diverge; at P = 8, any two schedules
  disagree on **51–58% of output elements**, in FP32 and in BF16 alike.
- **BF16 on the wire**: error against the exact result grows with P for
  every reduction schedule (sequential: 37% of elements wrong at P = 2, 61% at
  P = 8). Exact-gather stays nearly flat (37% at P = 2, 39–40% from P = 3 to 8) because it rounds once, but at
  P = 8 each rank sends 4× the bytes that ring does.
- **FP32 on the wire**: every schedule is within 4×10⁻⁷ relative RMS error
  of exact, yet the bits still depend on the schedule.
- **Top-1 changes: 0** of 8 tokens in every configuration. Bit differences at
  this magnitude did not flip an argmax over 512 outputs here; whether they
  flip greedy decoding over a real vocabulary is an R3 experiment.

## Evidence

- **Independent oracles** — Python `Fraction` reference vectors
  (`scripts/gen_sum_vectors.py`, 18,000 multi-term sums across all three
  formats, shares no code with the crate); 400,000 FP32 additions and
  ~400,000 binary64→FP32 conversions against the host's IEEE hardware; BF16
  addition and FP32→BF16 conversion against the standard integer bit-rounding
  identity; the fused chain against a host FP32 chain where products are
  exact.
- **Exhaustive** — all 65,536 FP16 and all 65,536 BF16 patterns round-trip
  through the exact path and through binary64.
- **Schedules** — exact results on integer data for P = 1…9 and lengths
  0…37 in every format; the single-port model (each rank sends and receives
  at most one message per round) and the round counts for P = 2…16;
  bandwidth-optimal bytes for ring and halving-doubling; a hand-derived case
  where ring folds each chunk from a different rank.
- **Stale-artifact detection** — a test re-runs the documented command and
  fails if `results/` differs.

## Limits

Round-to-nearest-even only. NaNs become the format's canonical quiet NaN (no
payload propagation, unlike FP Play's FP16 engine). An exactly zero sum is
`-0` only when every addend is `-0`. The local kernel is one sequential
order, not a blocked GEMM. Inputs are synthetic (Irwin–Hall approximate
normals, chosen because they are bit-identical on every platform). The cost
model ignores link contention, compute, overlap and topology; its constants
are illustrative. Nothing here is a measurement of real hardware, an
interconnect, or an Etched system.
