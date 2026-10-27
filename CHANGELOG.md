# Changelog

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

## 0.1.0 — prepared, not yet published

First public-release preparation for FP Play. No Git tag, GitHub release, or deployment has been published by this work.

- Interactive IEEE 754 binary16 Bit Explorer, decimal quantization, exact decoding, signed zeros, subnormals, infinities, NaNs, and neighbors.
- Precision Microscope with viewport-generated FP16 points, zoom/pan, rounding cells, quantization error, and ULP/binade exploration.
- Exact BigInt addition, subtraction, multiplication, single-rounding FMA, nonfused comparison, typed computation traces, and per-operation indicators.
- Ten multiplication/FMA presets, deterministic counterexamples, exhaustive numerical checks, and independent lattice/SoftFloat validation.
- Four-experience landing page, direct FMA hash entry, responsive typography, larger controls, stronger borders, visible stages, keyboard-scrollable data, metadata/favicon, and a no-JavaScript explanation.
- Manual-only, main-only GitHub Pages workflow, `/fp-play/` assets, immutable action pins, complete validation before deployment, and narrowly scoped permissions.
- Contributor, usage, deployment, and verification guides. Numerical coverage is preserved; the suite contains 344 tests.

Historical Milestone 7 limits: binary16 nearest-even only; no full IEEE conformance claim, other formats, division, FP32 accumulation simulator, transformer attention, or RTL. Real-browser, mobile/touch, and screen-reader review remain unperformed because no browser was connected. See [the verification record](docs/RELEASE_VERIFICATION.md) and [model contracts](src/lib/arithmetic/MULTIPLY_FMA.md).
