# Runtime case study: does combining partial sums change the bits?

FP Play's runtime is a dependency-free Rust simulator of how tensor-parallel
inference combines partial sums across devices. This page summarizes what it
found; the [runtime README](../runtime/README.md) has the details and the
commands to reproduce every number.

## The question

When a layer is split across P accelerators, each device computes a partial
sum and a collective (an AllReduce) combines them. Floating-point addition is
not associative, so every schedule — ring, tree, halving-doubling — fixes a
different order of roundings. Does that choice, and the number format on the
wire, change the output bits? And what does it cost, in bytes and
communication rounds, to stop that from happening? The simulator runs every
schedule message by message, and every addition a simulated device performs
is a real correctly rounded BF16 or FP32 addition, so the output bits are
observed, not estimated. The main problem is a row-parallel `Y = X·W` (8
tokens × 4096 hidden × 512 outputs, BF16 inputs, FP32 accumulation on each
device, P ∈ {2, 3, 4, 8}). A sweep over 8 seeds, hidden sizes 1024–16384 and P
up to 64 checks that the conclusions hold beyond one run.

## Three results

**1. BF16 on the wire: 53–58% of output elements change with the schedule
alone.** At P = 8, with the same inputs and the same arithmetic, any two of the
four reduction schedules disagree on 52.6–58.3% of the 4,096 outputs, so the
choice of collective is part of the numerical result, and the error against
the exact answer grows from 37.2% of elements at P = 2 to 55.0–61.1% at P = 8.
([results](../runtime/results/tp-allreduce.md))

**2. FP32 reduce-scatter with a BF16 all-gather: 0.02% of elements wrong, for
1.5× the bytes.** Keeping partials and additions in FP32 until a single final
rounding to BF16 takes ring at P = 8 from 59.8% of elements wrong to one
element in 4,096, closing the whole gap to the best result these partials
allow; it doubles the bytes of the reduce phase only, 1.5× per rank overall.
([results](../runtime/results/r1.1-policies.md))

**3. A fixed shard tree: identical bits for every P from 1 to 64.** Reducing
64 logical shards with one tree that does not depend on how many devices hold
them makes the output bit-identical for every device count; this buys
determinism, not accuracy, so it is paired with the FP32 policy above (0.02%
wrong), and at P = 8 it needs 6 rounds and 32 KiB per device against
exact-gather's 7 rounds and 112 KiB.
([results](../runtime/results/r1.1-shard-tree.md),
[sweep](../runtime/results/r1.1-sweep.md))

## How it is verified

- **Independent oracles.** Python `Fraction` arithmetic, sharing no code with
  the crate, produces 18,000 exact multi-term sums and 6,000 shard-tree sums.
  The tree sums are rich in ties, cancellation, subnormals and special values.
  The host's IEEE FP32 hardware checks 400,000 additions and conversions, and
  all 65,536 FP16 and BF16 bit patterns round-trip exhaustively.
- **Planted bugs.** Rounding cross-device tree children through BF16 fails
  three separate checks. Changing one schedule's association order fails a
  2,898-case regression against a frozen copy of the original simulator.
- **Cross-checks.** With one shard per device, the shard tree matches a
  separately written binomial-tree reduce bit for bit.
- **Stale-artifact tests.** Normal tests regenerate every recorded result and
  fail unless it matches the committed file byte for byte. CI runs them on a
  pinned toolchain.

## What is not claimed

- **Inputs are synthetic**: one approximately normal distribution, no real
  activations or outliers.
- **The cost model is illustrative**: an α–β model (2 µs per message, 100 GB/s
  per link) with no contention, compute or overlap. Rounds and bytes are
  counted; microseconds are estimates.
- **No hardware or interconnect is measured**, and none of the modeled
  policies describes a particular library.
- **Top-1 changes are rare and concentrated.** None of the 8 tokens changed in
  any configuration of the main problem. In the sweep, 35 of 3,768 runs changed
  one token's top-1 output, all under BF16-wire policies, and they fall in 6 of
  the 24 seed-and-size problems (22 from one seed). The sweep records each
  token's top-1 lead over its runner-up: the 6 distinct tokens that changed led
  by only 1–4 BF16 ulps, against a median lead of 16 ulps over all 96 tokens.
  These are near-ties. The "vocabulary" is 256 outputs, not a real one.
  ([sweep](../runtime/results/r1.1-sweep.md))

## Addendum: R2, a serving simulator

R2 puts these collectives inside a deterministic serving loop: a paged KV
cache with prefix sharing and copy-on-write, continuous batching, recompute
or swap preemption, and fail-stop rank failures. Step time is an
illustrative roofline plus the AllReduce schedules simulated above. All
numbers are from [r2-serving](../runtime/results/r2-serving.md) (200
requests per run, an 8B-class model on an illustrative device).

- **The collective shows up in latency.** At 10 req/s, TPOT p50 is 2.33 ms
  at P = 4 but 2.73 ms at P = 8, where ring AllReduces take 73.7% of step
  time. At P = 8, halving-doubling cuts TPOT p50 by 41.2% against ring; the
  shard tree cuts it by 31.4% but raises TTFT p50 from 12.69 ms to 24.32 ms.
  In this model the tree sends the whole buffer at every level, so large
  prefill steps cost it the most.
- **Memory pressure and failures.** With 449 KV
  blocks, recompute preempted 131 times and swap 84 times. After one rank
  failure at P = 4, retry finished all 200 requests and recomputed 3,832
  tokens; abort failed the 6 in flight; drain rejected 112.
- **How it is verified.** A slow reference scheduler must produce the same
  step trace as the engine on 600 random workloads, 300 of them with
  failures. In those tests every step is checked for leaked blocks,
  refcounts, KV contents and duplicated tokens. Eight planted bugs are each
  caught by a named check, and one of them (a wrong preemption victim) only
  by the reference.

## What is next

- **R3, does sharding change the tokens?** Greedy decoding over a real
  vocabulary, comparing schedules and policies token by token.
