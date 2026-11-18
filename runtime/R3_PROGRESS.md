# R3 progress

R3 runs a tiny Llama-style decoder tensor-parallel through the R2 engine
with R1's collectives, and asks whether greedy tokens change with P, the
AllReduce schedule, the wire policy and batch composition. Plan:
[`PLAN.md`](PLAN.md#r3--end-to-end-does-sharding-change-the-tokens).

## Status

| Part | Scope                                                                                                                                                  | State                              |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------- |
| R3a  | Model config, deterministic weights, `exp`/`sin`/`cos`, single-device forward and greedy decoding, Python reference vectors with a CI comparison       | in progress (branch `runtime-r3a`) |
| R3b  | Tensor-parallel forward (executed collectives, shard tree), engine integration with a K/V tensor store, batch variants, invariance tests, planted bugs | not started                        |
| R3c  | Experiment, `results/r3-decode.*`, stale test, docs and case study                                                                                     | not started                        |

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

## Observations so far

- Greedy decoding with these random tied weights often settles into
  repeating one token for many steps (visible in the vector file). It is
  a property of the random model, reported, not tuned away.
- The TINY forward pass costs about 0.28 ms per token on the development
  machine (single thread, P = 1).
