# Changelog

## Unreleased — runtime R4 completion: a trained model

- New `runtime/src/train/`: the R3 decoder's architecture trained deterministically in binary32 on a synthetic copy task, with a hand-written backward pass (checked against central finite differences in binary64 for every parameter of a small model) and Adam. The training forward pass is checked against the R4 all-FP32 inference model.
- New recorded artifact `runtime/results/r4-training.{json,md}` (loss curve, held-out accuracy) and the weights `runtime/results/r4-trained-weights.bin`, SHA-256 pinned in the source; the stale test retrains and compares every byte, and CI checks the hash with Python's `hashlib`.
- Two planted bugs, each caught by a different check. `runtime/src/sha256.rs` (dependency-free SHA-256, FIPS 180-2 examples).

## Unreleased — runtime milestone R6

- New `#collectives` workspace in the FP Play app (lazy, native hash link, footer entry): a rank × round schedule table stepped or played on request, element associations as text and trees, the fixed shard tree, and error against the exact sum versus P with an exact data table. Presentation in `src/lib/collectives/`, components in `src/components/collectives/`.
- `collectives::trace_association` traces transfer ranges and element associations beside the executed R1 simulator; `runtime/results/r6-collectives.json` is written by the documented experiment command, stale-checked, and imported by the app directly.
- One planted bug (swapped operands in the trace), caught. Checked with jsdom tests and in headless Chromium; no assistive-technology check.

## Unreleased — runtime milestone R5

- New `runtime/src/dataflow/`: a static graph of one 8B-class transformer layer (prefill and decode) mapped onto an illustrative accelerator (MAC array, vector unit, SRAM, HBM, one DMA engine, double buffering), timed in cycles with stall reasons and a roofline.
- The timing recurrence is checked against an independent cycle-by-cycle simulation; mapped work and HBM traffic are checked against analytic counts; two planted bugs, each caught.
- New recorded artifact `runtime/results/r5-dataflow.{json,md}`. One machine variant takes FP Play's recorded FP16 FMA path delay as an assumed cycle time, labeled as an assumption.

## Unreleased — runtime milestone R4

- New `runtime/src/precision/`: per-site precision policies (storage and accumulator format for each layer's attention and MLP, the KV cache and the embedding/head) for the R3 decoder with FP32 master weights.
- A fast fused multiply–accumulate in FP32, BF16 or FP16, tested bit-identical to `ops::fused_chain_dot` and to 3,600 independent Python `Fraction` chain vectors (`runtime/scripts/gen_chain_vectors.py`, compared in CI).
- New recorded artifact `runtime/results/r4-precision.{json,md}`: logit drift, KL, top-1 agreement and perplexity against FP32 for uniform, one-site-at-a-time and mixed policies.
- Two planted bugs, each caught (listed in `runtime/ROADMAP_PROGRESS.md`).

## Unreleased — runtime milestone R3

- New `runtime/src/decoder/`: a tiny Llama-style decoder (RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings; deterministic random BF16 weights) with native binary32 local kernels in a written order and `exp`/`sin`/`cos` from binary64 basic operations.
- Tensor parallelism: the row-parallel projections' per-rank or per-shard partials run through the executed R1 collectives or the shard tree under each wire policy.
- The R2 engine drives the model (`engine::run_model`, a `StepExecutor`); a paged K/V tensor store replays copy-on-write and swap block copies and is checked after every step.
- `ops::add` and `ops::convert` use a native binary32/BF16 fast path, checked against the exact superaccumulator; every earlier recorded artifact is byte-identical.
- New recorded artifact `runtime/results/r3-decode.{json,md}`: divergence from P = 1 across P, schedule and policy at two logit scales; divergence against the top-2 margin; a batch-invariance matrix over seven batch compositions. The prediction that only the ring depends on batch composition held exactly.
- Verification: a stdlib-only Python reference (`runtime/scripts/gen_decoder_vectors.py`) matched bit for bit and compared in CI; three planted bugs, each caught (listed in `runtime/R3_PROGRESS.md`).
- `r2-serving` adds the swap-mechanism check: shared blocks copied by swap at 1 and 4 samples.

## Unreleased — runtime milestone R2

- New `runtime/src/serving/`: a deterministic discrete-event serving simulator for one tensor-parallel group. It models an 8B-class model (not a specific released model) on an illustrative device.
  - A paged KV cache: LIFO free list, refcounts, block tables, a prefix cache with LRU retention, fork with copy-on-write, and a host swap pool.
  - Continuous batching with chunked prefill and n ∈ {1, 2, 4} samples, plus recompute or swap preemption.
  - Fail-stop rank failures handled by abort, retry or drain.
  - Step times combine a roofline compute model with the AllReduce schedules, recorded by a new dry-run mode for the collectives and the shard tree.
- Verification:
  - A slow reference scheduler whose trace must equal the engine's on 600 random workloads (300 with failures).
  - Hand-derived traces and golden JSON-lines traces (`runtime/tests/traces/`, regenerated by `cargo run --release -- traces`).
  - Per-step invariant checks and model-based KV-cache property tests.
  - Eight planted bugs (plus one manual mutation), each caught by a named check. They are listed in `runtime/R2_PROGRESS.md`.
- New recorded artifact `runtime/results/r2-serving.{json,md}`, written by the documented `experiment` command and stale-checked. It covers a load sweep, preemption under memory pressure, prefix sharing, collective choice and failures. Every earlier recorded JSON file is byte-identical.
- A named company in the R1 disclaimers is replaced with generic wording. The only text change is in `tp-allreduce.md`.
- Planted bugs are compile-time only: they exist only with the non-default `planted-bugs` Cargo feature, and such a build refuses to write results or traces. CI runs the planted-bug tests in a separate `--features planted-bugs` step.
- `r2-serving` adds swap sensitivity rows (host link 25–1600 GB/s, and one sample per request) and a table of the deterministic pairing (shard tree + `fp32-rs/bf16-ag`) against ring under both wire policies. The runtime README states the KV-check interval of each scenario, and the runtime case study gains an R2 section.

## Unreleased — runtime milestone R1.1

- The Rust toolchain is pinned in `runtime/rust-toolchain.toml` (1.97.0, clippy, rustfmt). CI installs from that file and checks the active version, so local and CI runs cannot drift apart.
- Shard tree: a deterministic AllReduce over a fixed binary tree of S logical shards (default 64, `--shards`), for any P ≤ S with contiguous ownership. Every node uses the same formats whether its children are on one rank or two, and policies with a lossy reduce wire are rejected. Messages are packed into single-port rounds, and the scheduled count is reported.
- Independent shard-tree vectors: `runtime/scripts/gen_tree_vectors.py` (6,000 `Fraction` cases) and `runtime/tests/vectors/tree-sums.txt`, with their own CI `cmp` step.
- New recorded artifacts, each stale-checked by a normal test:
  - `runtime/results/r1.1-shard-tree.{json,md}`: identical bits for every P from 1 to 64, compared with ring and exact-gather;
  - `runtime/results/r1.1-sweep.{json,md}`: min / median / max over 8 seeds × 3 hidden sizes × 7 world sizes × 4 policies × every schedule, at 1,024 elements per run, with every run that changed a top-1 output listed by seed, hidden size, P, policy and schedule. It also records each token's top-1 lead over its runner-up, in ulps of the output format, for every token and for each changed token: the 6 tokens that changed led by 1–4 BF16 ulps, against a median of 16 over all 96 tokens.
- The shard-tree page and README say plainly that the tree buys determinism, not accuracy. Under `bf16` it is the least accurate option up to P = 8, so it is meant to be paired with `fp32-rs/bf16-ag`. P-invariance holds by construction, and the tests check the distributed implementation. The output equals a single-device shard-tree kernel, not the plain single-device kernel.
- Wire policies: a policy names the accumulator, reduce-wire, gather-wire and output formats. Two mixed BF16-output policies are modeled, `fp32-rs/bf16-ag` and `bf16-wire/fp32-add`, and neither is attributed to any library. The uniform policy reproduces R1 bit for bit against a frozen copy of the R1 simulator.
- Reduce-scatter and all-gather are available as separate collectives (ring and halving-doubling). Per-element rounding counts are now measured from the schedule.
- New recorded artifact `runtime/results/r1.1-policies.{json,md}`, which is stale-checked like R1's. It leads with absolute error and bytes, and gives the share of the gap from `bf16` to the ideal for these partials (their exact sum rounded once) that each mixed policy closes. The ratio against BF16 exact-gather is in the JSON only. Every policy applies to all five schedules through their reduce and gather phases; the reduce-scatter-only comparison covers ring and halving-doubling.
- `runtime/results/tp-allreduce.md` regenerated with corrected **text only**: it now defines the roundings/elem column, says the pairwise list covers the four reduction schedules (pairs with exact-gather are in the JSON), and states that the BF16 wire rounds each FP32 partial before the collective. No number changed; `tp-allreduce.json` is byte-identical to R1 and a test pins it.

## Unreleased — runtime milestone R1

- New `runtime/` Rust crate (no dependencies): exact FP16/BF16/FP32 superaccumulator numerics, five message-level AllReduce schedules with a single-port round model and α–β cost estimate, and a row-parallel linear-layer experiment with a recorded, stale-checked artifact.
- Independent checks: Python `Fraction` sum vectors, host IEEE FP32 add/convert differentials, BF16 bit-rounding oracle, exhaustive FP16/BF16 round trips. New `runtime.yml` CI workflow.
- Frontend, numerical APIs, RTL and historical records are unchanged.

## 0.2.0-rc.1 — research showcase candidate (local preparation)

- Compact landing evidence, exact FP16/FMA examples and architecture overview, with entry points to all existing workspaces.
- Dedicated interactive hardware showcase: seven source-hashed measurements, baseline/tree area-delay comparison, selected critical endpoints, conditions and observation-versus-hypothesis explanations.
- Technical case study, fast/advanced reproduction guide, documentation index, contributor workflows, issue/PR templates and explicit third-party attribution.
- Candidate version synchronized across package/lockfile/UI; manual-only Pages behavior preserved. No new tag, GitHub release or deployment is created by this preparation.
- Preserves the 502 Milestone 13 tests, numerical APIs, RTL and historical records. New presentation checks and actual browser evidence are recorded in `docs/RELEASE_VERIFICATION.md`.

## Milestone 13 — Hardware Datapath Explorer

- Added logical FP16 add/sub, multiply, true FMA and two-round nonfused diagrams with accessible stage navigation.
- Connected 97 recorded operation/examples to exact numerical traces, result/flag/internal-signal agreement and simulator/source provenance; unavailable wires and special bypasses stay explicit.
- Added stepwise GRS/parity/increment/carry explanations and source-linked baseline/tree whole-unit synthesis metrics with exact library/tool/constraint metadata.
- Added a bounded offline VCD exporter, 51 correlated directed waveform examples, native signal/zoom/pan controls and genuine raw downloads.
- Preserved historical tests, numerical engines, RTL and synthesis records; documented evidence layers, regeneration and outstanding manual browser checks. Actual headless Chrome desktop/mobile/theme/keyboard/emulated-touch checks and screenshots caught and resolved mobile grid overflow.

## Milestone 12 — RTL synthesis and design-space analysis

- Reproducible functional-only Yosys/ABC mapping with pinned public Nangate45, constrained OpenSTA timing, graph depth, raw netlists/reports, per-run manifests, and a read-only Hardware Analysis view.
- Seven measured baseline/normalization-variant configurations; all result/flag/signal comparisons pass. The changed rounder additionally passes two-state combinational formal equivalence.
- Preserves 457 historical tests, numerical APIs, and original RTL. Cell area/timing are synthesis estimates, not physical implementation, power, or post-layout claims. See [the methodology and actual results](docs/HARDWARE_SYNTHESIS.md).

## Milestone 11 — FP16 multiplier and true FMA RTL (local preparation)

- Exact 11×11 multiplication, 81-bit exact fused accumulation, shared nearest-even rounder, and nonfused comparison through the unchanged adder.
- 301,544 triples per path, independent lattice checks, 904,632 direct SoftFloat outputs, and all supported flags validated.
- Genuine source-hashed hardware inspection, bounded VCD replay, read-only RTL CI, and generic Yosys synthesis sanity checks.
- Preserves all 430 previous tests and numerical APIs. No physical implementation, new formats, HLS, publication, or deployment. See [the contract](rtl/MULTIPLY_FMA.md).

## Milestone 10 — FP16 RTL arithmetic (local preparation)

- Original combinational SystemVerilog FP16 add/subtract, lossless fixed-width alignment, explicit GRS/nearest-even rounding, NaN/signed-zero policies, and all four indicators.
- Separate Verilator lint/build/differential workflow, VCD output, strict flags/payloads, reproducible failure records, and read-only RTL CI.
- 462,172 RTL transactions pass against TypeScript and directly against SoftFloat under the stated metadata policy; source-hashed recorded signals power optional browser inspection.
- All 424 earlier tests remain unchanged; 430 tests now pass. No multiplication/FMA RTL, HLS, transformer hardware, area/power optimization, actual synthesis/device/timing claims, publication, or deployment. See [the RTL contract](rtl/README.md).

## Milestone 9 — attention precision (local preparation)

- Bounded Q/K/V attention with five named precision policies, exact modeled arithmetic/normalization, stable and deliberately unstable naive softmax, and explicitly labeled binary64 exp/scaling approximations.
- Accessible matrix editing, dual-configuration/shared-scale heatmaps, raw encodings, probability rows, reference/error views, complete boundary inspection, and twelve verified experiments.
- Preserve all 390 earlier tests; 424 now pass. Independent verification covers 240 pipelines, 12,916 stage results/flags, aligned NumPy FP32, and Decimal(100) references. Full deterministic JSON export is development-only.
- No training, backpropagation, GPU kernels, benchmarking, RTL, publication, or deployment. Real-browser review remains unperformed. See [the attention contract](docs/ATTENTION_PRECISION.md).

## Milestone 8 — multi-precision lab (local preparation)

- Typed FP16/BF16/FP32 conversion, raw decoding, exact dyadics, adjacency, and ULP; BF16 rounds directly from binary64.
- Exact BigInt multi-format add/multiply/FMA, destination rounding, and documented NaN/tininess/exception semantics.
- Format Comparison plus a bounded five-mode Dot Product Precision Lab with sequential/pairwise graphs, exact original/quantized references, computation steps, error chart, and twelve verified presets.
- Preserve all 344 historical tests; add exhaustive BF16, native FP32, independent arithmetic/reduction, and actual-engine UI checks. Optional SoftFloat FP32 verification compares 169,011 operations with zero result/flag mismatches under the stated NaN policy.
- No browser automation, publication, push, merge, deployment, attention, arbitrary formats, or RTL in this milestone. See [the model contract](docs/MIXED_PRECISION.md).

## 0.1.0 — existing foundation tag

The repository contains `v0.1.0` at `977ccc8` (Milestone 7). The following notes describe that foundation release preparation; they do not assert GitHub release or deployment status. Milestone 14 prepares a separate subsequent candidate.

- Interactive IEEE 754 binary16 Bit Explorer, decimal quantization, exact decoding, signed zeros, subnormals, infinities, NaNs, and neighbors.
- Precision Microscope with viewport-generated FP16 points, zoom/pan, rounding cells, quantization error, and ULP/binade exploration.
- Exact BigInt addition, subtraction, multiplication, single-rounding FMA, nonfused comparison, typed computation traces, and per-operation indicators.
- Ten multiplication/FMA presets, deterministic counterexamples, exhaustive numerical checks, and independent lattice/SoftFloat validation.
- Four-experience landing page, direct FMA hash entry, responsive typography, larger controls, stronger borders, visible stages, keyboard-scrollable data, metadata/favicon, and a no-JavaScript explanation.
- Manual-only, main-only GitHub Pages workflow, `/fp-play/` assets, immutable action pins, complete validation before deployment, and narrowly scoped permissions.
- Contributor, usage, deployment, and verification guides. Numerical coverage is preserved; the suite contains 344 tests.

Historical Milestone 7 limits: binary16 nearest-even only; no full IEEE conformance claim, other formats, division, FP32 accumulation simulator, transformer attention, or RTL. Real-browser, mobile/touch, and screen-reader review remain unperformed because no browser was connected. See [the verification record](docs/RELEASE_VERIFICATION.md) and [model contracts](src/lib/arithmetic/MULTIPLY_FMA.md).
