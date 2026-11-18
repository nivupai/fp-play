# R3 progress

R3 runs a tiny Llama-style decoder tensor-parallel through the R2 engine
with R1's collectives, and asks whether greedy tokens change with P, the
AllReduce schedule, the wire policy and batch composition. Plan:
[`PLAN.md`](PLAN.md#r3--end-to-end-does-sharding-change-the-tokens).

## Status

| Part | Scope                                                                                                                                                  | State                          |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------ |
| R3a  | Model config, deterministic weights, `exp`/`sin`/`cos`, single-device forward and greedy decoding, Python reference vectors with a CI comparison       | merged (#23)                   |
| R3b  | Tensor-parallel forward (executed collectives, shard tree), engine integration with a K/V tensor store, batch variants, invariance tests, planted bugs | PR open (branch `runtime-r3b`) |
| R3c  | Experiment, `results/r3-decode.*`, stale test, docs and case study                                                                                     | not started                    |

Also in R3a: the R2 swap-mechanism check requested with the R3 approval
(swap bytes per preemption at 1 vs 4 samples; see `results/r2-serving.md`).

## Approved answers (before building)

1. Model: hidden 256, 2 layers, 16 query / 8 KV heads of dimension 16,
   SwiGLU FFN 704, vocabulary 2048, tied embeddings, BF16 weights. The
   r3-decode stale test is a normal test (option a).
2. Local kernels use native binary32 in a fixed order.
3. Formats: BF16 weights and KV cache; binary32 residual stream,
   attention and logits; binary32 partials into the collective, the wire
   policy applied there. Shard-tree per-shard partials use a fixed order
   within each shard.
4. No latency recomputation in R3: cite R2's deterministic-pairing results;
   R3 is about tokens.
5. Greedy, n = 1, fixed output length.
6. Baseline: P = 1 with the same policy (P = 1 runs no collective).
7. All seven batch variants, plus a P = 1 control (every variant
   bit-identical); the Python reference also covers ring and shard tree at
   P = 2.
8. One AllReduce buffer per step.

- Addition: a logit-scale parameter, the experiment at two values, the
  margin distribution for both, divergence presented mainly as a function
  of the top-2 margin.

## Decisions made while building

- **D-1 Logit scale is applied after the output projection, in binary32.**
  Any positive scale preserves the order of the logits, so greedy tokens
  cannot change; with a power of two the product is exact, so margins in
  ulps are unchanged too (`tests/decoder.rs`). The approved addition's
  premise, that a wider spread lowers divergence, does not hold for a pure
  scale: the numerical perturbation of the logits scales with them. It is
  implemented as asked and its neutrality is reported as a finding; see
  the R3c notes.
- **D-2 Weight scales:** projections `1/√fan_in`, tied embedding
  `1/√hidden` (logit standard deviation about 1 for a normalized hidden
  state), RMSNorm gains `1 + 0.1·normal`. Each tensor has its own SplitMix
  stream (seed XOR index × golden ratio).
- **D-3 Transcendentals:** `exp` (Cody–Waite + 14-term Taylor) and
  `sin`/`cos` (three-part π/2 reduction + Taylor) in binary64 from basic
  operations, rounded once to binary32. Tested against the host to ≤ 4
  ulps (`exp`) and ≤ 8 × 2⁻⁵² absolute (`sin`/`cos`). The Python
  reference mirrors them; everything else in it is independent.
- **D-4 Python reference size:** a reduced config (hidden 32, 4/2 heads of
  dimension 8, FFN 48, vocabulary 64), 4 prompts × 5 tokens, 6 greedy
  steps; full logits compared bit for bit. It runs in about 0.3 s.
- **D-5 RoPE** uses the rotate-half pairing (i with i + d/2).
- **D-6 Embedding lookup** is replicated (every rank reads the row it
  needs); the vocabulary-parallel head computes whole logits per rank and
  gathers them without arithmetic, so neither depends on P.

- **D-7 Native fast path for binary32/BF16 `ops::add` and `ops::convert`**
  (R3b). Host IEEE binary32 addition is correctly rounded; a BF16 sum of
  two BF16 values added in binary32 and rounded again equals one rounding
  (24 ≥ 2·8 + 2). Checked against the exact superaccumulator on specials,
  ties, overflow and 4M random pairs (`tests/ops_fast_path.rs`); every
  recorded R1/R1.1/R2 artifact regenerates byte-identically. Collectives
  run about 15× faster; without it the shard tree (63 additions per
  element) would not fit the stale-test budget.
- **D-8 Engine integration.** `engine::run_model` takes prompts and a
  `StepExecutor`. The scheduler is unchanged; requests carry model tokens
  (prompt plus committed outputs) instead of synthetic hashes. The model
  runs when a step completes (a failed step commits nothing, so it is not
  computed); the KV cache records block copies (copy-on-write, swap-out,
  swap-in) and the executor replays them into the tensor store before
  computing. Logits are computed only for tokens whose step commits.
- **D-9 Tensor invariant:** every covered slot carries a hash of the token
  prefix it was computed for; checked after every step against every
  device and swapped sequence.
- **D-10 Column-parallel work is computed once.** Each Q/K/V, attention,
  gate/up and head value is a whole dot product on whichever rank owns it,
  so it does not depend on P; only the two row-parallel projections per
  layer are split into per-rank (or per-shard) partials.
- **D-11 Variants** (`decoder/variants.rs`): 32 prompts of 16 tokens in 4
  groups sharing an 8-token prefix (block size 4, so 2 shareable blocks),
  48 greedy tokens. Reference: each prompt alone. Together: budget 1024
  (every prefill in one step). Reversed arrival. Chunk 8 and chunk 64
  budgets. Prefix cache: chunk 64 with caching on, so reused K/V was
  computed in earlier steps. Pressure: 5/8 of the needed blocks (320),
  recompute. Failure: a rank fails in step 12 (a quarter of decoding) at
  the last layer's MLP AllReduce; retry, limit 1.
- **D-12 Corrected prediction.** The plan said ring and halving-doubling
  can vary with batch composition. Halving-doubling cannot: at a
  power-of-two P every element is reduced as (x0 + x2) + (x1 + x3) up to
  commutativity, whatever its index. Only the ring folds each chunk from a
  different rank. The prediction tested is therefore: only the ring varies.
  Also, at P = 2 sequential, ring, binomial tree and halving-doubling agree
  (one commutative addition); exact-gather agrees with them under BF16 but
  not under fp32-rs/bf16-ag, where it rounds the exact sum once instead of
  rounding to binary32 first (my first derivation wrongly applied the
  double-rounding argument to binary32 inputs; the test caught it).
- **D-13 Forked samples** are supported by the executor (a sibling's first
  token comes from the carrier's prompt logits), used only in the tensor
  store tests; R3 workloads use n = 1.
- **D-14 The Python reference** now uses 8 query / 4 KV heads of dimension
  4 so it can run P = 4; it covers P = 1, ring at P = 2 and 4, and the
  shard tree (S = 8) at P = 2, for both policies.

## Planted bugs (R3)

| Bug                                                        | Caught by                                                          |
| ---------------------------------------------------------- | ------------------------------------------------------------------ |
| The tensor store skips copy-on-write copies                | tensor invariant, step 1 (`model: tensor: …`)                      |
| RoPE position off by one on recomputed positions           | the P = 1 control: pressure and failure variants differ from alone |
| Shard-tree partials of odd buffer tokens summed in reverse | the batch-invariance check for the shard tree                      |

## Observations so far

- Greedy decoding with these random tied weights often settles into
  repeating one token for many steps (visible in the vector file). It is
  a property of the random model, reported, not tuned away.
- The TINY forward pass costs about 0.28 ms per token on the development
  machine (single thread, P = 1); a 32-prompt run through the engine takes
  about 0.4–0.5 s at P = 4.
- First look (R3b probe, not a recorded result): at P = 4 the ring changes
  every prompt's logits in every batch variant; halving-doubling,
  sequential and the shard tree are bit-identical across all seven; the
  P = 1 control is bit-identical in all seven.
