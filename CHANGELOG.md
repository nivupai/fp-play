# Changelog

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
