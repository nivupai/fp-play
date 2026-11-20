# FP Play runtime

**Part 2 of FP Play: from one correctly rounded operation to many devices.**
FP Play's browser lab shows how a single FP16 add, FMA or dot product rounds.
This Rust crate asks what happens when an inference layer is split across
devices and the partial sums are combined by a collective: which bits change,
why, and what it costs to make them stop changing.

Milestone **R1** is an exact, message-level AllReduce simulator plus a
tensor-parallel experiment. **R1.1** adds mixed wire policies, reduce-scatter /
all-gather as separate collectives, a deterministic AllReduce over a fixed tree
of shards, and a sweep across seeds. **R2** is a deterministic serving
simulator (paged KV cache, continuous batching, preemption, rank failures);
see [below](#r2-the-serving-simulator) and [PLAN.md](PLAN.md).
For a two-page summary of the results, read the
[runtime case study](../docs/RUNTIME_CASE_STUDY.md).

## Run it

Rust 1.97.0, pinned in [`rust-toolchain.toml`](rust-toolchain.toml) (rustup installs it on first use; CI installs from the same file). No dependencies; works offline.

```sh
cd runtime
cargo test --release                              # 172 tests, including the recorded-artifact checks;
                                                  # 6 slow trained-model tests are ignored; add
                                                  # `--test train_experiment --test trained_experiment -- --include-ignored`
cargo run --release -- experiment --out results   # regenerates results/ byte for byte (about 4 min on 10 cores, mostly training and the reruns)
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
| `ops`              | correctly rounded add, convert, exact products, sequential fused-chain dot, exact dot; binary32/BF16 add and convert use a native fast path tested equal to the exact path                          |
| `collectives`      | sequential chain, ring, binomial tree, halving-doubling (Rabenseifner), exact-gather — executed as synchronous rounds over virtual ranks; wire policies; reduce-scatter and all-gather on their own |
| `cost`             | α–β estimate: per round, latency + largest message / bandwidth                                                                                                                                      |
| `tensor_parallel`  | row-parallel `Y = X·W` (BF16 inputs, FP32 local accumulation) AllReduced in a chosen wire format                                                                                                    |
| `shard_tree`       | R1.1: a fixed binary tree over S logical shards, any P ≤ S ranks, single-port rounds; output bits independent of P                                                                                  |
| `shard_experiment` | R1.1: the shard tree on the R1 problem for every P from 1 to S, against ring and exact-gather                                                                                                       |
| `sweep`            | R1.1: min / median / max across 8 seeds × 3 hidden sizes × P up to 64 × 4 policies × every schedule                                                                                                 |
| `policies`         | R1.1: BF16 output under three wire policies, recovery of exact-gather's accuracy, reduce-scatter-only cost                                                                                          |

Every addition a simulated rank performs is a real correctly rounded addition
in the policy's accumulator format, and every message is converted (one
correct rounding) to its wire format. Since R3, binary32 and BF16 additions
and conversions use the host's IEEE binary32 arithmetic (correctly rounded
by IEEE 754) instead of the superaccumulator; the two paths are tested equal
(see Evidence), and every recorded result is unchanged. The association order comes from the schedule being
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

## R2: the serving simulator

A deterministic discrete-event simulator of one tensor-parallel serving
group: paged KV cache, continuous batching with chunked prefill, forked
samples, recompute or swap preemption, fail-stop rank failures with abort,
retry or drain, and step times that include the real AllReduce schedules.

| Module                | Role                                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `serving::model`      | 8B-class dimensions (not a specific released model), illustrative device constants, KV capacity per device; P must divide the 8 KV heads          |
| `serving::detmath`    | `ln` from IEEE basic operations only, so Poisson arrivals match on every platform                                                                 |
| `serving::workload`   | Poisson arrivals, prompt and output length tables, shared system prompts, n ∈ {1, 2, 4} samples, priority classes, deterministic synthetic tokens |
| `serving::kv`         | Block pool (LIFO free list, refcounts), block tables, prefix cache with LRU retention, fork and copy-on-write, host swap pool, invariant checks   |
| `serving::step_time`  | Roofline compute plus two AllReduces per layer, costed from the schedule each collective records in a dry run                                     |
| `serving::scheduler`  | The written policy: token budget, decodes first, chunked prefill, FCFS admission within priority, victim = lowest priority then latest arrival    |
| `serving::engine`     | Event loop with per-step scheduler, token-stream and KV-cache invariant checks; fault plans and guarantees G1–G4                                  |
| `serving::reference`  | A slow independent scheduler and cache (test oracle)                                                                                              |
| `serving::experiment` | The recorded scenarios in [`results/r2-serving.md`](results/r2-serving.md)                                                                        |

Selected results from `results/r2-serving.md` (illustrative model, 200
requests per run):

- **Load:** at 10 req/s, TPOT p50 is 6.45 ms at P = 1, 3.29 ms at P = 2,
  2.33 ms at P = 4 and 2.73 ms at P = 8, where ring AllReduces take 73.7% of
  step time.
- **Memory pressure** (449 KV blocks): recompute preempted 131 times and
  recomputed 61,427 tokens (TTFT p50 593.52 ms); swap preempted 84 times
  and moved 23.04 GB over the host link (TTFT p50 1280.26 ms). Swap's cost
  depends on the host-link bandwidth, 25 GB/s here: at 1600 GB/s its TTFT
  p50 is 996.49 ms, and its E2E p50 stays above recompute's at every tested
  bandwidth (25–1600 GB/s), so there is no crossover in that range. With
  one sample per request instead of the 1/2/4 mix, swap is ahead at 25 GB/s
  (E2E p50 1008.37 ms against recompute's 1028.74 ms). With four samples,
  48.5% of the blocks a swap copies are still shared with a sibling (0.0%
  with one); each returns as a private block, so a swapped-in request needs
  1.94× the blocks it released, and swap's E2E p50 (8999.62 ms; 8961.78 ms
  at 1600 GB/s) is far above recompute's (5186.36 ms). This is consistent
  with the reversal, not isolated by a sharing-preserving swap.
- **Prefix sharing:** with 4 shared 512-token system prompts, the cache hit
  42.9% of eligible blocks; TTFT p50 fell from 53.47 ms to 28.46 ms and peak
  KV blocks from 1809 to 1027.
- **Collectives** at P = 8 with a BF16 wire: halving-doubling lowers TPOT p50
  by 41.2% against ring and the shard tree by 31.4%, but the shard tree's
  TTFT p50 is 24.32 ms against ring's 12.69 ms. In this model the tree sends
  the whole buffer at every level, so its message size grows with the
  step's token count.
- **The deterministic pairing** (shard tree + `fp32-rs/bf16-ag`, whose output
  bits do not depend on P) against ring + `bf16`, p50: at P = 4, TPOT 2.23
  against 2.21 ms (+1.2%) and TTFT 26.86 against 12.61 ms (+112.9%); at
  P = 8, TPOT 1.98 against 2.60 ms (−23.7%) and TTFT 34.39 against 12.69 ms
  (+171.0%). Scenario 4 reports ring, halving-doubling and the shard tree
  under both `bf16` and `fp32-rs/bf16-ag`.
- **Failures** (P = 4, 20 req/s, one rank failure in step 1500, 0.5 s
  detection and 5 s recovery): abort failed the 6 requests in flight; retry
  finished all 200 and recomputed 3,832 tokens; drain finished 88 and
  rejected 112. Every failure run reran byte-identically, and G1–G4 held in
  every run.

Invariant checking in the recorded runs: the scheduler and token-stream
invariants run after every step in every scenario. The KV-cache invariants
(which walk the whole block pool) run after every step in the
memory-pressure, swap-sensitivity and prefix-sharing scenarios, and every
256 steps (plus after every failure and after the last step) in the 80 GB
load, collective and failure scenarios, whose pools hold up to 282,260
blocks. Every test runs every check after every step.

Evidence: `tests/dry_run.rs` (dry-run rounds equal executed rounds for every
algorithm, policy and P ≤ 16, and the shard tree for every P ≤ S);
`tests/serving_kv.rs` (hand-derived cases, 400 random operation sequences
checked against a plain model after every operation, five planted bugs);
`tests/serving_engine.rs` (a hand-derived three-request trace, 300 random
workloads with every invariant after every step and a trace equal to the
reference scheduler's, seven planted bugs caught inside the engine);
`tests/serving_traces.rs` (golden JSON-lines traces, two with failures);
`tests/serving_faults.rs` (hand-derived retry, abort and drain cases, 300
random fault plans whose traces equal the reference's and rerun
identically, a planted double commit on retry);
`tests/planted_build.rs` (planted bugs exist only with the `planted-bugs`
Cargo feature, and such a build refuses to write results or traces). The
planted-bug tests run with
`cargo test --release --features planted-bugs --test planted_build --test serving_kv --test serving_engine --test serving_faults`;
CI runs them as a separate step.
`tests/serving_step_time.rs` (hand-computed step times);
`tests/serving_workload.rs` (a pinned workload fingerprint that CI checks on
Linux); `tests/serving_experiment.rs` (the recorded results are current, the
memory-pressure scenario preempts in both modes, and each failure policy
behaves as specified). Every planted bug and the check that caught it, and
every design decision, are listed in [R2_PROGRESS.md](R2_PROGRESS.md).
`cargo run --release -- traces` regenerates `tests/traces/`.

## R3: does sharding change the tokens?

A tiny Llama-style decoder (RMSNorm, RoPE, grouped-query attention, SwiGLU,
tied embeddings; hidden 256, 2 layers, 16/8 heads, FFN 704, vocabulary
2048, deterministic random BF16 weights) decodes greedily inside the R2
engine. Its row-parallel AllReduces run through the executed R1
collectives, so every rounding of the schedule and wire policy happens.
Local kernels are native binary32 in one written order.

| Module                | Role                                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| `decoder::forward`    | One step's forward pass; K/V through a `KvStore`, row-parallel projections through a `RowReduce`     |
| `decoder::parallel`   | P = 1, and tensor parallelism: per-rank or per-shard partials through a collective or the shard tree |
| `decoder::mathf`      | `exp`, `sin`, `cos` from binary64 basic operations                                                   |
| `decoder::serve`      | The R2 engine drives the model; a paged K/V tensor store replays copy-on-write and swap block copies |
| `decoder::variants`   | The workload (32 prompts in 4 shared-prefix groups, 48 tokens each) and seven batch compositions     |
| `decoder::experiment` | The recorded experiment in [`results/r3-decode.md`](results/r3-decode.md)                            |

Selected results from `results/r3-decode.md`:

- **P and the schedule change tokens.** Against P = 1 with the same policy,
  6–18 of 32 prompts diverge under a BF16 wire (some at the first generated
  token) and 1–2 under `fp32-rs/bf16-ag` (first at token 18 or 21). No run
  at P > 1 keeps any prompt's logits bit-identical to P = 1.
- **The shard tree does not depend on P.** Its tokens are identical at
  P = 1, 2, 4 and 8. It differs from the unsharded baseline in 15 prompts
  (BF16) or 1 (`fp32-rs/bf16-ag`): a fixed change of arithmetic.
- **How close the calls were.** Under BF16, 14.4% of at-risk tokens had a
  baseline top-2 margin within reach of the run's logit perturbation (at
  most twice its largest logit change) and 257 flipped; under
  `fp32-rs/bf16-ag`, whose median perturbation is 10.5× smaller (9,228
  against 97,340 ulps of the top logit), 1.0% were within reach and 24
  flipped. Every flip being within reach is a consistency check that holds
  by construction.
- **Logit scale.** A positive scale preserves the argmax, so it cannot
  affect greedy tokens; measured as neutral: at scale 8, 38 of 38 runs
  chose the same tokens as at scale 1.
- **Only the ring depends on batch composition.** At P = 4, sequential,
  binomial tree, halving-doubling, exact-gather and the shard tree give
  bit-identical logits in all seven batch variants under both policies, as
  predicted. The ring changes every prompt's logits in every variant under
  BF16 (10–13 prompts' tokens) and 18–24 prompts' logits but no token under
  `fp32-rs/bf16-ag`. The P = 1 control is bit-identical in all seven.
- **What determinism costs** is R2's result: the shard tree with
  `fp32-rs/bf16-ag` against ring with `bf16` is +1.2% TPOT and +112.9% TTFT
  (p50) at P = 4, −23.7% and +171.0% at P = 8
  ([r2-serving](results/r2-serving.md), illustrative model).

Evidence: `tests/decoder.rs` (a stdlib-only Python reimplementation,
`scripts/gen_decoder_vectors.py`, matches every logit bit for bit at P = 1,
ring P = 2 and 4, and shard tree P = 2 under both policies; CI regenerates
and compares it; pinned cross-platform hashes); `tests/ops_fast_path.rs`
(the native binary32/BF16 fast path of `ops::add` and `ops::convert` equals
the exact superaccumulator); `tests/decoder_parallel.rs` (shard-tree
P-invariance, P = 2 agreement, the batch-invariance prediction, the P = 1
control, scheduling independent of numerics, the tensor store under
copy-on-write and swap); `tests/decoder_experiment.rs` (the recorded result
is current and its stated properties hold). Three planted bugs (skipped
tensor copy-on-write, RoPE off by one on recompute, a batch-dependent shard
order) are each caught by a named check; see [R3_PROGRESS.md](R3_PROGRESS.md).

## R4: per-layer precision policies

The R3 decoder with FP32 master weights, where each site (each layer's
attention projections and MLP, the KV cache, the tied embedding and head)
gets a storage format and an accumulator format. Every matrix product is a
fused chain (exact product, one rounding per term) computed by a fast
multiply–accumulate that is tested bit-identical to `ops::fused_chain_dot`
and to independent Python `Fraction` vectors
(`scripts/gen_chain_vectors.py`). Policies are compared with the all-FP32
policy teacher-forced on its own greedy continuation (8 prompts × 32
tokens); see [`results/r4-precision.md`](results/r4-precision.md).

- **Accumulation width matters more than storage.** All sites at BF16
  storage with FP32 accumulation: mean KL 5.05e-5, top-1 agreement 253 of 256. With BF16 accumulation as well: mean KL 2.78e-3, 227 of 256, and 7
  of 8 prompts' greedy continuations change. FP16: 7.98e-7 (255 of 256)
  with FP32 accumulation, 4.45e-5 (252 of 256) with FP16 accumulation.
- **The first layer's attention is the most sensitive site** at every
  level when lowered alone (mean KL 1.32e-3 at `bf16-acc`); the second
  layer's sites and the KV cache are among the least.
- Every run's logits stayed finite.

### FP8 and INT8 levels with explicit scales

`precision/narrow.rs` adds three levels: FP8 E4M3 (OCP: no infinities,
NaN `S.1111.111`, largest 448), FP8 E5M2 (IEEE-style, largest 57344) and
symmetric INT8 (−127…127). Each weight row, each activation vector entering
a product, and each KV head's K or V of one token gets one binary32 scale
`fl32(absmax ÷ Qmax)`. An element is quantized from the **exact** quotient
(integer division of the significands) with one round-to-nearest-even
onto the format's grid, saturating at ±Qmax. FP8 products accumulate in a
fused FP32 chain (FP8 products are exact), INT8 products in exact integers
converted once to binary32, and the scales are applied afterwards
(`fl32(fl32(acc · s_w) · s_x)`); the KV cache dequantizes to binary32 on
read.

- Exhaustive: all 256 patterns of each FP8 format decode as an independent
  table, encode ∘ decode is the identity, ordering and the NaN/∞ counts are
  checked; every midpoint between adjacent values (and INT8 integers) and
  its binary32 neighbours rounds correctly in both signs and two scales.
- `scripts/gen_narrow_vectors.py` (stdlib `Fraction`, brute-force nearest
  value) gives the 512 decodings, 15,300 quantizations and 480 scaled
  products; CI regenerates and compares them.
- Two planted bugs (ties away from zero; an INT8 clamp to −128) are caught.

### A model trained deterministically in the repo

Random weights give near-uniform logits. To rerun R3 and R4 on a model
with confident predictions, `train/` trains the R3 decoder's exact
architecture (2,000,128 parameters) on a synthetic copy task
(`BOS s1 … s24 SEP s1 … s24`, scored on the 24 tokens after `SEP`), with a
hand-written backward pass and Adam, all in binary32 basic operations and
the crate's deterministic `exp`, in a fixed order. Each sequence's gradient
is computed on its own and the batch gradient is their sum in sequence
order, so the weights do not depend on the thread count. See
[`results/r4-training.md`](results/r4-training.md).

- 600 steps of 16 sequences (about a minute). Batch loss falls from
  8.12 to 0.00143 nats per token; held-out teacher-forced accuracy is
  100.00% on 256 sequences, and greedy decoding copies 64 of 64 held-out
  prompts exactly through the R3 decoder (BF16 weights and KV) and the R4
  all-FP32 model.
- The weights are committed as `results/r4-trained-weights.bin` (binary32
  little-endian) with their SHA-256 pinned in `train::TRAINED_WEIGHTS_SHA256`;
  the stale test retrains and compares every byte, and CI also checks the
  file's hash with Python's `hashlib`.
- Checks (`tests/train.rs`): the backward pass equals central finite
  differences in binary64 for every parameter of a small model; the
  binary32 training forward pass matches the R4 all-FP32 inference model on
  the same weights to rounding (so the trained model is the model the
  experiments run); the multi-threaded batch gradient equals the in-order
  single-thread sum bit for bit.

### R3 and R4 rerun on the trained model

`results/r3-trained.*` and `results/r4-trained.*` rerun R3's divergence and
batch-invariance experiment and R4's precision study (every level,
including FP8 and INT8) on the trained model beside the original
random-weight models, on the same copy prompts (`train/rerun.rs`).

- Median top-2 margin at P = 1: 9.3781 logits trained, 0.2447 random.
- Flips over the divergence grid: BF16 wire 160 of 12,527 at-risk tokens
  (random) and 0 of 14,592 (trained); FP32 reduce-scatter 6 of 14,484 and 0
  of 14,592. No prompt's logits are bit-identical to the baseline in any
  run for either model; 0 of 29,184 trained at-risk tokens are within reach
  of a flip (margin ≤ 2 × perturbation). The batch-invariance prediction
  held for both models.
- Every site at one level, mean KL against FP32 (random / trained):
  `fp8-e4m3` 1.13e-2 / 2.04e-5, `fp8-e5m2` 4.51e-2 / 9.24e-5, `int8`
  9.98e-4 / 1.34e-6, `bf16-acc` 2.74e-3 / 5.70e-5. The trained model keeps
  192 of 192 top-1 tokens and every greedy continuation at every level.

## R5: a transformer layer on a toy accelerator

One 8B-class layer as a static graph (prefill and decode), mapped onto an
illustrative accelerator (128 × 128 MAC array, 256 vector lanes, 16 MiB
SRAM, 1.5 TB/s HBM, one DMA engine, double buffering) and timed in cycles.
The timing recurrence is checked against an independent cycle-by-cycle
simulation; see [`results/r5-dataflow.md`](results/r5-dataflow.md).

- **Prefill fills the array:** 95.7%, 94.3% and 89.2% utilization at 128,
  512 and 2048 tokens (1 ns cycle), within 1.05–1.12× of the roofline
  bound. The gate/up projection alone takes 50.3% of prefill-512 cycles.
- **Decode does not, in this mapping:** 0.8%, 6.0%, 19.2% and 43.3%
  utilization at batch 1, 8, 32 and 128 (context 1024), 5.88× to 2.31×
  its roofline bound. Decode at batch 1 and 8 is below the ridge point
  (21.8 FLOPs per HBM byte), yet the array, not HBM, is the limit: tokens
  map to array rows, so padding takes 98.9% of batch-1 cycles.
- **The assumed slower cycle** (FP Play's recorded FP16 FMA path delay,
  4.179512 ns, an assumption, not a clock) moves the ridge to 5.2 FLOPs per
  byte and puts batch-1 decode at 24.53× its bound.

## R6: the app workspaces

The FP Play app's lazily loaded `#collectives` workspace replays recorded
schedules: a rank × round table stepped (or played on request) round by
round, each element's association as text and a tree, the fixed shard tree,
and error against the exact sum versus P with an exact table. Its data,
[`results/r6-collectives.json`](results/r6-collectives.json), comes from
`collectives::trace_association`, which traces each transfer's element
range and each element's association beside the executed R1 simulator. The
trace confirms R3's corrected prediction directly: up to commutativity,
only the ring's association depends on the element's index (P > 2).

`#serving`, `#decode`, `#precision` and `#dataflow` present R2–R5 (and the
trained-model reruns) the same way: each reads its committed JSON directly,
validates it, and pairs every chart with an exact table; the landing page
links to all five workspaces from its first screen.

## Evidence

- **Independent oracles** — Python `Fraction` reference vectors
  (`scripts/gen_sum_vectors.py`, 18,000 sums of 1–12 terms across all three
  formats, shares no code with the crate); 400,000 FP32 additions and
  ~400,000 binary64→FP32 conversions against the host's IEEE hardware; BF16
  addition and FP32→BF16 conversion against the standard integer bit-rounding
  identity; the fused chain against a host FP32 chain where products are
  exact.
- **Fast path** — `tests/ops_fast_path.rs`: binary32 and BF16 `add`, and
  binary32 ↔ BF16 `convert`, equal the exact superaccumulator path on
  hand-derived subnormal, tie, cancellation, signed-zero, overflow and NaN
  cases, every special-value pair, every BF16 rounding boundary, and 4M
  random pairs. Because `add` itself now uses the host's binary32 addition,
  the host-hardware check of FP32 addition above applies to the fast path;
  the exact path keeps its own 400,000-case host check there.
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
interconnect, or any particular vendor's system. The R2 serving simulator
uses synthetic tokens (no model numerics), a roofline with illustrative
constants, no compute/communication overlap, one tensor-parallel group,
uncontended links and fail-stop failures with constant detection and
recovery times; P must divide the 8 KV heads (no KV-head replication), and
it has no speculative decoding, LoRA or quantized KV. The R3 decoder has
random weights (its margins do not transfer to trained models), 2 layers,
synthetic prompts, greedy decoding with fixed lengths, sequential local
kernels (real GPU kernels can also break batch invariance locally), and
deterministic but not correctly rounded `exp`, `sin` and `cos`. R4 changes
only matrix products, their inputs and the KV cache (element-wise work,
the residual stream and the attention core stay binary32), uses one
sequential summation order and has no FP8 or INT8. R5's machine is
illustrative: one simple tiling with tokens on the array rows, no operator
fusion, no overlap between ops, no SRAM-bandwidth, NoC, power or area
model; the recorded FMA delay is used only as an assumed cycle time.
