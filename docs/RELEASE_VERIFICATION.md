# v0.1.0 verification record

This record distinguishes source/DOM/static-build evidence from real-browser verification. It is not a full IEEE 754 or WCAG conformance certification.

## Automated evidence

- Latest `main` included Milestone 6 before `feature/public-release` was created. All 338 earlier tests and all numerical modules are unchanged.
- 344 Vitest tests pass on Node 22.9.0 and Node 24: the earlier 338 plus six release entry-point, FMA hash-navigation, and keyboard-access cases. The FMA link loads an existing independently verified example rather than a new arithmetic path.
- Lint, Prettier, TypeScript, and production build pass. The generated index references JavaScript, CSS, and favicon beneath `/fp-play/`; `npm run check:release` verifies those files exist and package/lockfile versions agree.
- `npm run check:contrast` checks 74 actual light/dark token pairs, including pressed bits, colored panels, labels, control borders, and focus colors. Minimum tested text contrast: 5.46:1; minimum tested control/focus contrast: 3.39:1. System-dark tokens match the manual dark palette. Opacity, compositing, rendered geometry, and every possible state require browser review.
- A loopback-only production preview served the index and all four hash URLs as HTML; JavaScript, CSS, and favicon returned HTTP 200 with expected MIME types. This checks static serving, not client execution or visual behavior.
- CI and manual Pages workflows pass actionlint 1.7.12 validation. The manual workflow is main-only, has no automatic triggers, and scopes write permissions to the deployment job. Workflow execution on GitHub and actual deployment remain unperformed.
- Milestone 6 previously checked 536,888 operations against pinned Berkeley SoftFloat with zero mismatches under the documented NaN comparison policy, plus 21 checks for the seven counterexamples. Numerical code and reference corpora are untouched in this milestone; those historical counts are not new release-only checks.

## Browser availability and screenshots

Browser inventory was empty during Milestone 7. No real browser was connected. No screenshots were captured, and no successful rendered-browser or screen-reader review is claimed. Native focus rendering, touch capture/scrolling, responsive geometry, OS font rendering, contrast compositing, and assistive-technology announcements remain unverified.

## Manual review before publication

Use a real desktop browser and at least one mobile browser, recording browser/OS versions, viewport, input method, and results. Include widths around 360, 390, 768, and 1440 pixels, landscape orientation, and 200% zoom.

1. **Landing/navigation:** four experience cards reach named sections; source and technical documentation links work. Reload `/fp-play/#fma` and confirm three operands, fused `0x0010`, nonfused `0x0000`. Reopen the FMA link after editing another operation. Browser back/forward must retain ordinary static hash navigation.
2. **Keyboard/focus:** use the skip link, native controls, Tab/Shift+Tab, Space/Enter, and theme selector. Confirm focus remains visible and is not clipped. Verify all bit switches' positions and pressed states are announced. There must be no keyboard trap.
3. **Bit Explorer:** toggle sign/fraction/exponent, enter `0.1`, `-0`, `Infinity`, and `NaN`, and submit invalid text. Ensure original/quantized values stay distinct and invalid input preserves the applied pattern. Check subnormal/normal and signed-zero neighbors.
4. **Microscope:** exercise all zoom/pan/fit/overview buttons and chart arrow/+/- shortcuts. Inspect exact point selection, target slider, sampling notices, rounding cells, tooltips, and the expanded exact table. Verify native plot/table scrolling, touch target movement and pan mode, pointer capture, and vertical page scrolling. Text controls must remain usable without dragging.
5. **Arithmetic:** inspect finite and special traces, previous/next bounds, Home/End/Left/Right, and long binary rows. Focus a long row and scroll it; its keys must not change the stage. Check all operations and invalid A/B/C fields. Read complete values without page-wide overflow.
6. **Fusion:** compare lost-residual, cancellation, and intermediate-overflow presets. Verify exact sum, rounded product, final encodings, flags, and numerical difference/undefined explanations. Switch trace paths and verify their actual stage counts. Keep fused and nonfused labels distinct on mobile.
7. **Themes/motion:** test manual light/dark and OS-driven system themes, high zoom, reduced motion, and visible focus on colored panels. Check contrast in disabled, selected, error, and pressed states in the rendered page.
8. **Screen reader:** review landmarks/headings, form labels/descriptions, errors, exact numerical strings, selection announcements, and plot alternatives with VoiceOver or NVDA. Static token checks and jsdom cannot establish this behavior.
9. **Production:** verify asset HTTP responses and MIME types at the deployed subpath, reload all hash entry points, and check browser console/network failures. An unrelated path such as `/fp-play/arithmetic` is not supported navigation.

Record any failures and limitations in the release notes; do not mark unperformed items as passed. Screenshots should be added only after they are captured from and reviewed against a real browser.

## Milestone 8 verification update

`feature/mixed-precision` starts at main `977ccc8`, which includes Milestones 6/7. All 344 historical tests and the five historical FP16 numerical modules remain unchanged. The new typed format/arithmetic/dot modules run beside those APIs. The full suite now has **390 passing tests in 14 files**, including 46 new numerical and actual-engine UI tests. Node 22.9.0 and Node 24.22.0 execute the complete suite.

New independent evidence: exhaustive 65,536 BF16 raw classifications/values/metadata; 195,840 BF16 midpoint-neighbor conversions; 20,000 binary64→FP32 native conversions; 20,000 native raw FP32 decodes/round trips; 2,540 signed FP32 exponent-boundary cases; 24,000 finite arithmetic comparisons; 2,000 reductions with intermediate bits/flags; all twelve experiment outputs; all ten mode/order combinations at 64 terms. The integer-distance reference imports no production code. See [the exact model and corpus definitions](MIXED_PRECISION.md).

The extended public SoftFloat driver compares **169,011 FP32 operations, zero result/flag mismatches** under the documented NaN classification/flag comparison. Of 5,275 quiet NaN outputs, 369 differ in implementation-specific sign/payload selection; directed tests cover FP Play's policy. The historical FP16 corpus is also rerun unchanged using the extended driver: **536,888 operations, zero mismatches**, with 28,618 NaN-classification/flag comparisons. No reference library is bundled into the app.

Lint, formatting, strict TypeScript, all tests, static contrast checks, production build, built asset checks, and workflow actionlint pass. Workflow triggers/permissions and dependencies remain unchanged. No push, merge, publication, or deployment is performed.

There is no browser automation tool in the Milestone 8 environment; no new screenshot, real rendering, mobile/touch, native range-key behavior, theme appearance, or assistive-technology verification is claimed. Add `#formats` and `#dot-product` to the manual production review: submit valid/invalid/special values, compare formats, apply equal/mismatched/max-size vectors, select modes/orders/presets, navigate steps, read exact references and long values, and inspect error/table alternatives at narrow widths in all themes.

## Milestone 9 verification update

`feature/attention-precision` starts from main `958c81b`, which includes Milestone 8. All **390 earlier tests and numerical implementations remain unchanged**; additions are separate attention/normalization modules and barrel exports. The complete suite now passes **424 tests in 16 files** on Node 22.9.0 and Node 24.22.0. No assertions or exhaustive corpora were removed or weakened.

New verification includes 9,000 independently rounded rational quotients; normalization midpoint and precision-only tininess boundaries; 250 finite softmax-row invariant checks; maximum matrix shapes; all conceptual stage boundaries; explicit overflow/underflow/zero-denominator/nonfinite propagation; signed-zero inputs; 28 preset outcomes; and actual-engine DOM/keyboard/heatmap interactions.

The fixed-seed exporter produces 24 inputs (12 presets and 12 random matrices), five precision policies, and stable/naive methods: **240 full calculations**. The separate Python Fraction model matches **3,010 matrix values and 12,916 stage result/flag sets**. Explicitly aligned NumPy FP32 matches **48 calculations / 602 values**. Decimal(100) comparisons cover **301 baseline values** with tolerance `5e-12 * max(1, abs(reference))`; observed maximum absolute error is `3.789746294556856e-14`. **Zero mismatches** under the documented NaN-classification policy. A compact independent full-pipeline fixture is replayed in CI; [the verification artifact](experiments/attention-verification.json) records counts, versions, policies, and high-precision outputs. Two full-trace exports on the same Node runtime matched byte for byte. Exp is independently checked after host approximation/quantization, not certified as correctly rounded.

The existing SoftFloat scripts also pass unchanged corpora again: **536,888 FP16** and **169,011 FP32** operations, zero mismatches under their established NaN comparison policy. FP32 includes 5,275 quiet NaN outputs and 369 implementation-specific payload/sign differences. Those checks validate the reused arithmetic, not transcendental attention conformance.

Lint, formatting, strict TypeScript, complete tests, static contrast tokens, production build, built release assets, workflow actionlint, and optional Python Ruff formatting/lint checks pass. Dependencies, CI triggers/permissions, and manual-only Pages deployment remain unchanged. Nothing is pushed, merged, deployed, or published.

There is no browser automation tool available for Milestone 9. No screenshots or successful rendered-mobile/theme/touch/native-focus/screen-reader review are claimed. Manual review should include `#attention`: matrix editing/invalid drafts, all 12 presets, stable/naive switching, both precision selectors, all four heatmap stages with common scales, text/nonfinite/error labels, keyboard cell/step/row selection, and long-value/table scrolling at narrow widths. See [the attention contract](ATTENTION_PRECISION.md) for mathematical limits and commands.

## Milestone 10 verification update

`feature/fp16-rtl` starts from main `cb22437`, which includes Milestone 9. All **424 historical tests and numerical implementations are unchanged**. Six artifact-provenance/numerical/UI tests bring the suite to **430 tests in 18 files**, passing on Node 22.9.0 and Node 24.22.0. The new panel reports both bit and flag-only disagreements, preserves full long signals, and labels browser records as precomputed.

The original combinational SystemVerilog unit passes **Verilator 5.052 `--Wall` lint without warnings**, with no latch diagnostics or RTL warning waivers. The owned C++17 testbench passes strict compiler syntax/warning checks with third-party headers treated as system headers and clang-format 12.0.1 validation. No real-number or simulator floating-point shortcut calculates hardware results.

Full differential simulation passes **462,172 cases, zero mismatches**: 28 explicit directed cases, 262,144 exhaustive-family transactions covering every raw encoding, and 200,000 fixed-seed randomized add/subtract cases. All four flags and all result bits, including **20,268 raw NaN signs/payloads**, match the unchanged TypeScript engine strictly. **441,888 finite paths** also match retained significand, GRS, increment, and exact arithmetic magnitude. The wide 2^-24 lattice retains cancellation tails; underflow is correctly always clear because tiny FP16 add/sub results are exact.

Direct comparison of those same RTL observations against the existing external SoftFloat driver also passes **462,172 cases, zero mismatches** under its established NaN-classification policy. All flags agree. **65** NaN payload/sign selections differ from SoftFloat; RTL matches FP Play's policy strictly. Seed, source hashes, actual RTL outputs, and signal observations are committed in the small [simulation artifact](../rtl/vectors/simulation.json); full generated corpora remain ignored build outputs. Repeated exports from the same sources/tools matched byte for byte.

Directed VCD capture passes for all 28 examples, with inputs, result/flags, effective signs, aligned magnitudes, normalization, and GRS/rounding signals. File contents were inspected; no waveform GUI session or physical timing claim is made. Directed files use separate names and retain the full-corpus verification outputs. Mismatches have reproducible vectors/JSON and bounded waveform replay paths.

Frontend lint, Prettier, TypeScript, complete tests, theme-token contrast, production build, and release asset checks pass. Actionlint validates the new separate read-only RTL workflow and unchanged existing workflows. The RTL job installs/logs the Ubuntu 24.04 distribution's Verilator and runs the complete corpus plus directed VCD capture; it has not executed on GitHub because no branch was pushed. Manual-only Pages behavior and permissions remain unchanged.

The unit is designed with synthesizable fixed-width constructs; actual synthesis/netlist checks, formal equivalence, four-state X/Z behavior, timing closure, physical implementation, area/power measurements, and exhaustive operand-pair validation remain unperformed. No multiplication/FMA RTL, HLS, transformer hardware, publication, push, merge, or deployment is included. Browser tools are unavailable; the panel has DOM/keyboard evidence but no rendered-layout/theme/touch/screen-reader claim. See [the complete RTL contract and reproduction instructions](../rtl/README.md).

## Milestone 11 extension

The original 430 tests and all numerical APIs are preserved; 27 new artifact/integration tests bring the complete suite to **457 tests in 20 files**, passing on Node 22.9.0 and Node 24. The new multiplier/FMA RTL validates 301,544 triples per path against exact software and an independent lattice oracle, plus 904,632 direct SoftFloat outputs with all flags. Generic Yosys 0.70 synthesis passes latch/loop checks for multiplier, FMA, and comparison. See [the detailed evidence and limitations](../rtl/MULTIPLY_FMA.md). No connected browser or GUI waveform viewer was available; DOM/keyboard tests and generated VCDs are not visual verification. Deployment remains manual and nothing has been published.

Milestone 11 quality checks: lint, Prettier, TypeScript, 74 theme-token contrast checks, full tests, production build, release asset verification, actionlint, C++ syntax/warnings/formatting, strict Verilator lint/simulation, original adder regression, and generic synthesis all pass locally. Read-only RTL CI is extended; no GitHub run is claimed because the branch is not pushed.
