# Roadmap progress: R4, R5, R6

The long run after R3. The file `runtime/ROADMAP_PROGRESS.md` that the run
was asked to resume from did not exist anywhere (main, every branch, every
stash). Asked how to proceed, the user chose **"proceed from PLAN.md"**: this
file expands `PLAN.md`'s R4–R6 bullets into plans, and every design call
made without approval is recorded below as a decision.

Rules (unchanged from R2/R3): small commits; before every commit
`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` (default and
`planted-bugs`), `cargo test --release --locked`, the planted-bug tests, every
Python vector comparison and `npm run check`; push, PR, green CI, self-review,
squash-merge. Recorded results change only through
`cargo run --release -- experiment --out results`; earlier recorded JSON
stays byte-identical. Every number in docs comes from committed results.
Every planted bug is caught by a named check. No tags, releases or deploys.
Before each milestone free disk space is checked with `df`; below 5 GB only
build caches created by this run (`cargo clean`, test scratch files) are
removed, and that is noted here.

## Status

| Step | Scope                                                                                                                                | State                          |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------ |
| 0    | R3 follow-up                                                                                                                         | merged (#26)                   |
| R4a  | Fused multiply–accumulate in any format, FP32 master weights, per-site precision policies, policy forward pass, Python chain vectors | merged (#27)                   |
| R4b  | Experiment `results/r4-precision.*`, stale test, docs                                                                                | PR open (branch `runtime-r4b`) |
| R5a  | Dataflow graph of a transformer block, toy pipeline model, simulator, reference model, tests                                         | not started                    |
| R5b  | Experiment `results/r5-dataflow.*`, docs                                                                                             | not started                    |
| R6a  | `#collectives` data export and presentation library                                                                                  | not started                    |
| R6b  | `#collectives` workspace UI, tests, docs                                                                                             | not started                    |

## Disk checks

| Before | Free  | Action |
| ------ | ----- | ------ |
| R4     | 34 GB | none   |

## R4 plan: per-layer precision policies

Question: which layers of the R3 decoder are sensitive to storage format and
accumulation width, measured as logit drift and perplexity against FP32?

- **Model.** The R3 TINY dimensions with **FP32 master weights** (same seed
  and Irwin–Hall draws, rounded once to binary32 instead of BF16), so that
  storing a site's weights in BF16 or FP16 actually rounds them.
- **Arithmetic: every dot product is a fused chain** in the site's
  accumulator format: `acc = round_acc(acc + a·b)`, the product exact, one
  rounding per term, starting from +0. This is `ops::fused_chain_dot`'s
  semantics and FP Play's FMA semantics (exact product, one destination
  rounding). A fast implementation (binary64 product, TwoSum, and a
  rounding that breaks ties with the TwoSum error) is tested bit-identical
  to `ops::fused_chain_dot` and to independent Python `Fraction` vectors.
- **Levels** (per site): `fp32` (FP32 storage and accumulation), `bf16`
  (BF16 weights and activations, FP32 accumulation), `bf16-acc` (BF16
  accumulation too), `fp16`, `fp16-acc`. FP8/INT8 with scales are out of
  scope (PLAN.md "later").
- **Sites:** each layer's attention projections (Q, K, V, O), each layer's
  MLP (gate, up, down), the KV cache (storage only), and the tied
  embedding/head. With 2 layers: 6 sites. Element-wise work (RMSNorm,
  RoPE, softmax, SiLU, residual adds) and the attention core (QKᵀ, PV,
  FP32 fused chains) stay in FP32 for every policy.
- **Reference:** the all-`fp32` policy. **Metrics,** teacher-forced on the
  reference's own greedy continuation so every policy sees the same
  contexts: max and RMS logit difference, KL(p_ref ‖ p_policy), top-1
  agreement, and the perplexity of that continuation (a self-consistency
  measure with random weights, not a language-modeling result). Metrics
  are computed in binary64 from the binary32 logits.
- **Policies:** the 5 uniform levels; one site at a time lowered to each of
  the 4 non-FP32 levels (24); two mixed policies.
- **Checks:** fast fused chain = exact chain (random, subnormals, ties,
  cancellation, overflow, NaN); Python `Fraction` chain vectors in CI; the
  all-`fp32` policy forward pinned; a policy with every site at `fp32`
  equals the reference exactly; metric functions on hand-computed cases;
  stale test. Planted bugs: accumulating a BF16 site in FP32 (caught by the
  chain vectors / level test), skipping the KV-cache rounding (caught by
  the per-site difference test).
- **PRs:** R4a (arithmetic, weights, policy forward, tests), R4b
  (experiment, report, docs).

## R5 plan (refined when R5 starts)

A static graph of one transformer block (prefill and decode shapes, the
R2 8B-class dimensions) mapped onto a toy fixed-function pipeline: a MAC
array, a vector unit, an SRAM of fixed capacity and bandwidth, and HBM
bandwidth. Per-op tiling, double buffering, a deterministic cycle
simulation with stall reasons (memory, capacity, dependency), utilization
and a roofline for prefill versus decode. One MAC-unit model may take FP
Play's recorded Nangate45 FMA delay (ideal-interconnect OpenSTA, from the
M12 artifact) as its clock: labeled as an assumption, not a timing claim.

## R6 plan (refined when R6 starts)

A lazily loaded `#collectives` workspace in the FP Play app that reads
recorded runtime JSON: a rank × round schedule view animated only on
request with a text table alternative, per-element association trees, and
an error-versus-P chart with an exact data table. The four landing entries
stay as they are; the workspace is linked from the research showcase. The
landing refresh stays deferred.

## Decisions (made without approval)

- **RD-1** The missing progress file: proceed from PLAN.md (the user's
  answer); this file is the plan of record.
- **RD-2 Fused MAC implementation.** Binary64 product (exact for ≤ 24-bit
  inputs), TwoSum, and a rounding that breaks exact-midpoint ties with the
  TwoSum error. The rounding works on the binary64 bit pattern when the
  result is normal in the target format and falls back to a scale-and-floor
  reference otherwise; the two are tested equal (600,000 cases with
  constructed midpoints and tails). About 7× faster than the reference
  rounding alone (3.8 ms per TINY token).
- **RD-3 FP32 master weights** use `as f32` (IEEE round to nearest even)
  on the same binary64 draws as R3; they therefore differ from R3's BF16
  weights (which round the draw once to BF16) and R4's numbers are not
  comparable bit for bit with R3's.
- **RD-4 The attention core** (scores and Σ p·v) is a fused FP32 chain over
  the KV cache's stored values for every policy; only its storage format
  is a policy choice (the KV-cache site).
- **RD-5 R4a checks:** fused chains equal `ops::fused_chain_dot` for every
  accumulator × input format (27,000 random chains with subnormals,
  overflow and NaN/∞ inputs), mixed-format products equal the exact path,
  3,600 Python `Fraction` chain vectors (CI regenerates them), every level's
  site product equals the exact chain on its stored values, the all-FP32
  policy is pinned, and lowering any single site changes the logits.
- **RD-6 R4 planted bugs:**

  | Bug                                            | Caught by                                                 |
  | ---------------------------------------------- | --------------------------------------------------------- |
  | BF16-accumulating sites accumulate in FP32     | the per-level exact-chain check (33 of 48 outputs differ) |
  | The KV cache keeps binary32 whatever its level | the every-site-affects-the-logits check                   |

- **RD-7 R4 experiment:** 8 prompts × 16 tokens, 32-token reference
  continuation; 5 uniform policies, 22 one-site policies (the KV cache's
  `-acc` levels are dropped: it only stores values), 2 mixed policies.
  Teacher-forced comparison at every continuation token plus free-running
  greedy divergence. The stale test takes about 18 s locally.

## R4 results (results/r4-precision.md)

- BF16 storage with FP32 accumulation everywhere: mean KL 5.05e-5, 253 of
  256 top-1; with BF16 accumulation: 2.78e-3, 227 of 256, 7 of 8 greedy
  continuations change. FP16: 7.98e-7 / 4.45e-5.
- The first layer's attention is the most sensitive site at every level.
