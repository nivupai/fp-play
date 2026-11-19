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
  all 65,536 FP16 and BF16 bit patterns round-trip exhaustively. FP32 and
  BF16 additions in the collectives use the host's correctly rounded binary32
  arithmetic (since R3, for speed), tested equal to the exact
  superaccumulator on subnormals, ties, cancellation, overflow, NaNs and
  random inputs; every recorded result is unchanged.
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

## R2: what determinism costs when serving

R2 puts these collectives inside a deterministic serving loop: a paged KV
cache with prefix sharing and copy-on-write, continuous batching, recompute
or swap preemption, and fail-stop rank failures. Step time is an
illustrative roofline plus the AllReduce schedules simulated above, with no
compute/communication overlap. Every number is from
[r2-serving](../runtime/results/r2-serving.md) (200 requests per run, an
8B-class model on an illustrative device).

- **The deterministic pairing has a prefill price.** The shard tree with
  `fp32-rs/bf16-ag` (output bits independent of P) against ring with `bf16`:
  at P = 4, TPOT p50 is 2.23 against 2.21 ms (+1.2%) but TTFT p50 is 26.86
  against 12.61 ms (+112.9%); at P = 8, TPOT p50 is 1.98 against 2.60 ms
  (−23.7%) and TTFT p50 is 34.39 against 12.69 ms (+171.0%). In this model
  the tree sends the whole buffer at every level, so its cost grows with the
  tokens in a step: decode steps are small, prefill steps are not.
- **The collective shows up in latency.** At 10 req/s, TPOT p50 is 2.33 ms
  at P = 4 but 2.73 ms at P = 8, where ring AllReduces take 73.7% of step
  time. At P = 8 with `bf16`, halving-doubling cuts TPOT p50 by 41.2%
  against ring.
- **Swap versus recompute depends on more than the link.** With 449 KV
  blocks, recompute preempted 131 times (TTFT p50 593.52 ms) and swap 84
  times over a 25 GB/s host link (TTFT p50 1280.26 ms). Raising the link to
  1600 GB/s brings swap's TTFT p50 to 996.49 ms, but its E2E p50 stays above
  recompute's at every tested bandwidth. With one sample per request instead
  of the 1/2/4 mix, swap is ahead at 25 GB/s (E2E p50 1008.37 against
  1028.74 ms). With four, 48.5% of the blocks a swap copies are still
  shared with a sibling and come back as private copies, so a swapped-in
  request needs 1.94× the blocks it released; swap's E2E p50 is 8999.62 ms
  against recompute's 5186.36 ms, and 8961.78 ms even at 1600 GB/s. That
  is consistent with the reversal; a swap that keeps sharing was not
  simulated.
- **Failures.** After one rank failure at P = 4, retry finished all 200
  requests and recomputed 3,832 tokens; abort failed the 6 in flight; drain
  rejected 112. Every failure run reran byte-identically.
- **How it is verified.** A slow reference scheduler must produce the same
  step trace as the engine on 600 random workloads, 300 of them with
  failures. In those tests every step is checked for leaked blocks,
  refcounts, KV contents and duplicated tokens. Eight planted bugs, compiled
  in only by a test-only Cargo feature, are each caught by a named check;
  one of them (a wrong preemption victim) only by the reference.

## R3: does sharding change the tokens?

R3 decodes greedily with a tiny Llama-style decoder (2 layers, hidden 256,
random BF16 weights) inside the R2 engine, with every AllReduce executed.
Every number is from [r3-decode](../runtime/results/r3-decode.md) (32
prompts, 48 tokens each).

- **Yes, with P and the schedule.** Against P = 1 with the same policy,
  6–18 of 32 prompts diverge under a BF16 wire, some at the first generated
  token, and 1–2 under `fp32-rs/bf16-ag`, first at token 18 or 21. No run at
  P > 1 keeps any prompt's logits bit-identical. The shard tree's tokens are
  the same at P = 1, 2, 4 and 8.
- **How close the calls were.** Pooling every run, 14.4% of at-risk tokens
  under BF16 had a baseline top-2 margin within reach of the run's logit
  perturbation (at most twice its largest logit change), and 257 flipped;
  under `fp32-rs/bf16-ag`, 1.0% were within reach and 24 flipped. The median
  perturbation was 97,340 ulps of the top logit under BF16 and 9,228 under
  `fp32-rs/bf16-ag`, 10.5× smaller. That every flip was within reach is a
  consistency check that holds by construction, not a finding.
- **Logit scale.** A positive scale preserves the argmax, so it cannot affect
  greedy tokens; measured as neutral (38 of 38 runs chose the same tokens at
  scale 8 as at scale 1).
- **Batch composition matters only for the ring.** At P = 4, sequential,
  binomial tree, halving-doubling, exact-gather and the shard tree are
  bit-identical across seven batch compositions (one batch, reversed
  arrival, 8- and 64-token budgets, prefix caching, recompute preemption, a
  failure with retry), as predicted from the schedules. The ring changes
  every prompt's logits under BF16 (10–13 prompts' tokens) and 18–24
  prompts' logits but no token under `fp32-rs/bf16-ag`. At P = 1 every
  composition is bit-identical.
- **The price** of the deterministic pairing is R2's: TTFT p50 +112.9% at
  P = 4 and +171.0% at P = 8 against ring with `bf16`
  ([r2-serving](../runtime/results/r2-serving.md)).
- **Verified** by a stdlib-only Python reimplementation that matches every
  logit bit for bit (P = 1, ring at P = 2 and 4, shard tree at P = 2), the
  P = 1 control, a tensor invariant on the paged K/V store, and three
  planted bugs, each caught. The weights are random, so the divergence rates
  do not transfer to trained models, whose margins differ.

## R4: which layers need precision?

R4 gives each site of the same decoder (FP32 master weights) a storage and
an accumulator format and compares it with the all-FP32 policy,
teacher-forced on 8 prompts × 32 tokens. Every number is from
[r4-precision](../runtime/results/r4-precision.md).

- **Accumulation width matters more than storage format.** BF16 storage
  everywhere with FP32 accumulation gives a mean KL of 5.05e-5 and 253 of
  256 top-1 agreements; adding BF16 accumulation gives 2.78e-3 and 227,
  and 7 of 8 greedy continuations change. FP16 storage gives 7.98e-7, FP16
  accumulation 4.45e-5.
- **The first layer's attention is the most sensitive site** at every
  level when lowered alone (mean KL 1.32e-3 at BF16 accumulation); the
  second layer's sites and the KV cache are among the least.
- **Verified** by a fast fused multiply–accumulate tested bit-identical to
  the exact superaccumulator and to 3,600 independent Python `Fraction`
  chains, and two planted bugs, each caught. Random weights and two layers
  limit how far the ranking transfers.

## R5: where a layer's cycles go

R5 maps one 8B-class layer onto an illustrative accelerator (128 × 128 MAC
array, 16 MiB SRAM, 1.5 TB/s HBM) and times it in cycles, checked against
an independent cycle-by-cycle simulation. Every number is from
[r5-dataflow](../runtime/results/r5-dataflow.md).

- **Prefill** uses 89.2–95.7% of the array (128–2048 tokens), within
  1.05–1.12× of the roofline bound.
- **Decode** uses 0.8% at batch 1 and 43.3% at batch 128 (context 1024).
  Batch 1 and 8 sit below the ridge point (21.8 FLOPs per HBM byte), but in
  this mapping the array is the limit: tokens occupy array rows, so
  padding takes 98.9% of batch-1 cycles, 5.88× the roofline bound. A
  mapping built for matrix–vector products was not modeled.
- **Assumption, not a clock:** taking FP Play's recorded FP16 FMA path
  delay (4.179512 ns) as the cycle time moves the ridge to 5.2 FLOPs per
  byte.

## What is next

- **R6, the `#collectives` workspace** in the FP Play app.
