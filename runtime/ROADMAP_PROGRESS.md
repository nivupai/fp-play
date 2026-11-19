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

| Step | Scope                                                                                                                                | State        |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------ |
| 0    | R3 follow-up                                                                                                                         | merged (#26) |
| R4a  | Fused multiply–accumulate in any format, FP32 master weights, per-site precision policies, policy forward pass, Python chain vectors | merged (#27) |
| R4b  | Experiment `results/r4-precision.*`, stale test, docs                                                                                | merged (#28) |
| R5a  | Dataflow graph of a transformer block, toy pipeline model, simulator, reference model, tests                                         | merged (#29) |
| R5b  | Experiment `results/r5-dataflow.*`, docs                                                                                             | merged (#30) |
| R6a  | `#collectives` data export and presentation library                                                                                  | merged (#31) |
| R6b  | `#collectives` workspace UI, tests, docs                                                                                             | merged (#32) |
| S0   | Scope corrections: the landing refresh was not deferred; SG-1…SG-5 recorded                                                          | merged (#34) |
| R4c  | Deterministic training on the copy task, committed weights, `results/r4-training.*`                                                  | merged (#35) |
| R4d  | FP8 E4M3 / E5M2 and INT8 levels with scales, Python vectors, exhaustive checks                                                       | in progress  |

## Disk checks

| Before | Free  | Action |
| ------ | ----- | ------ |
| R4     | 34 GB | none   |
| R5     | 31 GB | none   |
| R6     | 29 GB | none   |
| R4c    | 15 GB | none   |

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

## Scope-guard deferrals (recorded after the fact, 2026-11-19)

The scope-guard rule says anything the run was asked for and did not do is
recorded here as a deferral. The R4–R6 run merged (#27–#33) without recording
these; they are recorded now, as the user pointed out. Each was asked for
and not delivered; none was deferred by the user.

| #    | Milestone | Asked for                                                                                                 | State after #33                                                                                                           | Why it was not done                                         |
| ---- | --------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| SG-1 | R4        | FP8 and INT8 levels with explicit scales                                                                  | missing; `PLAN.md` called them "later" and the R4 plan put them out of scope                                              | scoped out by me, not by the user                           |
| SG-2 | R4        | A tiny model trained deterministically in the repo (the R3 follow-up's PLAN item), with R3/R4 rerun on it | missing; only listed as future work                                                                                       | scoped out by me                                            |
| SG-3 | R6        | App workspaces for the other runtime results (serving R2, decode R3, precision R4, dataflow R5)           | only `#collectives` was built                                                                                             | I scoped R6 to the one workspace PLAN.md named              |
| SG-4 | R6        | The landing-page refresh                                                                                  | not done; recorded wrongly as "deferred by request" here, in `PLAN.md` and in `docs/RUNTIME_CASE_STUDY.md`                | misread as deferred; the user's R6 instructions included it |
| SG-5 | R6        | Accessibility verification beyond static and DOM checks                                                   | one headless Chromium check of `#collectives`; no automated audit, no scripted keyboard walkthrough, no saved screenshots | not attempted                                               |

All five are the scope of the next run (R4 completion and R6 completion
below).

## Completion run plan (R4 completion, R6 completion)

Started 2026-11-19 under the same rules and authority. Order and PRs:

| Step | Scope                                                                                                                                                                                                                |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0   | This correction                                                                                                                                                                                                      |
| R4c  | `train/`: a tiny Llama-style model trained deterministically in binary32 on a copy task, hand-written backprop, finite-difference gradient check, committed weights with a pinned SHA-256, CI retrains and compares  |
| R4d  | FP8 E4M3 / E5M2 and INT8 levels with explicit scales; Python `Fraction` vectors; exhaustive 256-pattern checks                                                                                                       |
| R4e  | Recorded reruns on the trained model: `results/r4-training.*` (loss curve, accuracy), `results/r4-trained.*` (precision study with every level), `results/r3-trained.*` (divergence, margins, batch invariance)      |
| R6c  | `#serving` (R2) and `#dataflow` (R5) workspaces                                                                                                                                                                      |
| R6d  | `#decode` (R3) and `#precision` (R4, trained model included) workspaces                                                                                                                                              |
| R6e  | Landing refresh: Part 2 (the runtime) visible from the first screen, FP Play's visual identity kept                                                                                                                  |
| R6f  | Automated accessibility audit, scripted keyboard walkthrough of every new workspace, headless screenshots (desktop and phone, light and dark) in the verification folder, recorded in `docs/RELEASE_VERIFICATION.md` |

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
- **RD-8 R5 machine and mapping** (`dataflow/`; illustrative, not a named
  device): a 128 × 128 BF16 MAC array, 256 vector lanes, 16 MiB SRAM, 1.5
  TB/s HBM, one DMA engine, two chunk buffers. An activation stays in SRAM
  if it fits in half of it. Matrix products run in output passes sized to a
  quarter of SRAM, column blocks of the array width and reduction chunks
  sized so two fit a quarter of SRAM; causal tiles above the diagonal are
  skipped, partly masked tiles are computed in full. Vector ops cost
  illustrative per-element operation counts (RMSNorm 4, RoPE 6, softmax 5,
  SiLU × up 4, add 1). Ops run strictly in sequence (no inter-op overlap).
- **RD-9 Results in cycles.** A second machine takes the recorded Nangate45
  FP16 FMA path delay (4.179512 ns, M12 artifact, ideal interconnect) as an
  _assumed_ cycle time, which only changes HBM bytes per cycle; it is
  labeled an assumption and implies no frequency or timing result.
- **RD-10 R5 checks:** the timing recurrence equals an independent
  cycle-by-cycle simulation (3,000 random chunk lists and every op of five
  small layers); mapped useful MACs equal the analytic count for every op;
  cycles lie between max(compute, DMA) and their sum; weights and the KV
  cache are read at least once; a hand-derived product (80 cycles); the
  delay constant matches the hardware artifact.
- **RD-11 R5 planted bugs:**

  | Bug                            | Caught by                                                         |
  | ------------------------------ | ----------------------------------------------------------------- |
  | A buffer freed one chunk early | the cycle-by-cycle reference (2,587 of 3,000 random lists differ) |
  | Decode attention reads no K/V  | the KV-traffic check                                              |

- **RD-12 R5 experiment:** prefill 128, 512, 2048 tokens and decode batch
  1, 8, 32, 128 at context 1024, on the 1 ns machine and the assumed
  4.179512 ns machine. A roofline-bound column (max of useful MACs ÷ peak
  and this mapping's HBM bytes ÷ bytes per cycle) shows each run's distance
  from what the machine allows. Finding, stated as a property of this
  mapping: decode is limited by padding (tokens on array rows), not by HBM,
  even where its intensity is below the ridge point; a GEMV-oriented
  mapping is future work.

## R5 results (results/r5-dataflow.md)

- Prefill: 95.7% / 94.3% / 89.2% array utilization (128 / 512 / 2048
  tokens), 1.05–1.12× the roofline bound.
- Decode: 0.8% / 6.0% / 19.2% / 43.3% (batch 1 / 8 / 32 / 128), 5.88× to
  2.31× the bound; padding takes 98.9% of batch-1 cycles.
- **RD-13 R6 data path.** The schedule view needs each transfer's element
  range and each element's association, which the simulator did not
  record. `collectives::trace_association` adds an optional tracker to the
  R1 simulator that follows the same messages as its rounding-count
  tracker (never touching values); its rounds equal the dry run's, and the
  R1 regression against the frozen simulator still passes. The shard
  tree's association comes from `shard_tree::tree`, the list its executor
  iterates. `results/r6-collectives.json` (schedules for P = 2, 3, 4, 8 on 8
  elements, the S = 8 shard tree at P = 1, 2, 4, 8, and each schedule's
  error from the tp-allreduce run of the same command) is stale-checked.
  The app imports it directly from `runtime/results/`, so there is no copy
  to drift.
- **RD-14 R6 planted bug:** the trace records an addition's operands
  swapped; caught by the hand-derived association check (ring, P = 3).
- **RD-15** The executed-simulator trace confirms R3's corrected
  prediction directly: up to commutativity, only the ring's association
  depends on the element index (P > 2); `tests/association.rs`.
- **RD-16 R6 UI.** A lazily loaded `#collectives` `<details>` entry like
  `#research` and `#datapath`, a footer link, and no change to the four
  landing entries or other hashes. Native selects and buttons; a rank ×
  round table that is itself the text alternative of the schedule view;
  Left/Right/Home/End on the round controls; **Play rounds** only on request
  (slower with reduced motion), stopping at the last round; the current
  round marked by text and `aria-current`, not only by outline; trees as
  SVG with the expression as text and as the accessible name; the error
  chart distinguishes schedules by marker shape and dash, with an exact
  table. Malformed data throws rather than rendering a guess.
- **RD-17 Browser check (actual):** headless Chromium via the
  browser-automation skill on the production build (`vite preview`):
  stepping and selection worked, no console errors or failed requests, no
  horizontal overflow at 390 px; screenshots in light and dark themes were
  inspected. The round buttons first rendered without button styling; they
  now use the datapath explorer's button style. No screen reader or other
  assistive technology was used.

## Completion run decisions

- **RD-18 The trained model has R3's exact dimensions** (`DecoderConfig::TRAINED`
  = `TINY` with its own seed, 2,000,128 parameters), not a smaller model:
  training takes about a minute, and it makes every random-versus-trained
  comparison use one architecture, with every P of R3 (1, 2, 4, 8). Cost: an
  8,000,512-byte weights file in git (`results/r4-trained-weights.bin`,
  marked binary in `.gitattributes`).
- **RD-19 Task:** copy with a fixed length, `BOS s1 … s24 SEP s1 … s24`,
  symbols uniform over IDs 2–2047 (0 is BOS, 1 is SEP), scored on the 24
  tokens after `SEP`. A fixed length makes the source a fixed relative
  offset, which RoPE attention can represent; prompts are `BOS s1 … s24 SEP`
  and generate 24 tokens.
- **RD-20 Training:** initialization from the R4 master-weight generator
  (seed `0xdec0de0074`); Adam (β1 0.9, β2 0.99, ε 1e-8), 600 steps of 16
  fresh sequences, linear warm-up to 3e-3 over 50 steps then linear decay
  to 10%, global-norm clip 1; bias corrections by repeated multiplication;
  all binary32 basic operations plus `mathf::exp_f32`. Each sequence's
  gradient is computed on its own (on any thread) and summed in sequence
  order; matrix products keep each output's sum sequential in the input
  index (vectorized only across outputs). The loss is reported in binary64
  with the crate's `exp` and `ln`; it does not feed the gradient.
- **RD-21 Recording:** training runs inside the experiment command, which
  writes `results/r4-training.*` and `results/r4-trained-weights.bin`; the
  stale test retrains (about 110 s locally) and compares every byte and the
  pinned SHA-256; CI also checks the file's hash with `hashlib`. Later
  experiments load the committed file through `train::load_trained`, which
  refuses a file whose hash is not the pinned one.
- **RD-22 R4c checks and planted bugs:** backward = central finite
  differences (binary64, every parameter of a 1,064-parameter model, two
  examples); training forward = the R4 all-FP32 model to 1e-4 of the largest
  logit (measured worst ≈ 1e-6); threaded batch gradient = in-order sum, bit
  for bit; SHA-256 against the FIPS 180-2 examples.

  | Bug                                                        | Caught by                                                                         |
  | ---------------------------------------------------------- | --------------------------------------------------------------------------------- |
  | RoPE gradients rotated forward instead of by the transpose | the finite-difference check (the inference-forward check passes)                  |
  | RoPE pairs adjacent dimensions, consistently in backward   | the inference-forward check (finite differences pass: the gradient is consistent) |

- **RD-23 Greedy accuracy** is measured through the inference code, not the
  trainer: the R3 decoder (BF16 weights and KV, P = 1) under both R3 wire
  policies, and the R4 all-FP32 model, on 64 held-out prompts.

- **RD-24 FP8/INT8 semantics** (`precision/narrow.rs`): E4M3 and E5M2 as
  in the OCP FP8 specification; conversion saturates to ±max (E4M3 has no
  ∞; for E5M2 saturation rather than ∞ is chosen so a rounding overflow
  cannot create ∞); INT8 symmetric −127…127. One binary32 scale per
  weight row, per activation vector and per (token, KV head) slice of K and
  V: `fl32(absmax ÷ Qmax)`, 1 for an all-zero slice, and a non-finite
  element makes the product NaN. The quotient is exact (integer division),
  so there is one rounding. FP8 accumulates in a fused FP32 chain, INT8 in
  exact integers converted once to binary32; output
  `fl32(fl32(acc · s_w) · s_x)`. The KV cache dequantizes to binary32
  (`fl32(q · s)`) on write, which the FP32 attention core reads. Existing
  levels and `results/r4-precision.*` are unchanged (`Level::ALL` keeps the
  five float levels; `Level::SCALED` and `Level::EVERY` are new).
- **RD-25 R4d checks and planted bugs:** the 512 FP8 decodings against the
  Python table; round trips, ordering, NaN/∞ counts for all 256 patterns;
  every midpoint and neighbour; 15,300 quantizations and 480 scaled
  products against `scripts/gen_narrow_vectors.py`; every site at every
  scaled level changes the logits; the site product equals the scaled
  product.

  | Bug                                 | Caught by                                                      |
  | ----------------------------------- | -------------------------------------------------------------- |
  | Ties round away from zero           | the exhaustive midpoint check                                  |
  | INT8 clamps to −128 instead of −127 | the Python vectors (and the midpoint check's saturation cases) |

## R4c results (results/r4-training.md)

- CI on Linux x86-64 retrained the model and reproduced the SHA-256 first
  recorded on macOS arm64 (PR #35).
- Batch loss 8.12 → 0.00143 nats per token over 600 steps; first perfect
  batch at step 144. Held-out: 100.00% teacher-forced (256 sequences); 64 of
  64 prompts copied exactly by all three inference models.
