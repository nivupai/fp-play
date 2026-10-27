# FP Play release verification record

This record distinguishes source/DOM/static-build evidence from real-browser verification. It is not a full IEEE 754 or WCAG conformance certification.

**Current preparation: v0.2.0-rc.1, Milestone 14.** See the [latest verification update](#milestone-14-research-showcase-candidate). Earlier sections describe the checks and publication status at those milestones. The existing v0.1.0 tag is retained; this candidate creates no new tag or release.

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

## Milestone 12 verification update

`feature/rtl-synthesis-analysis` starts from main `ef49251`, which includes Milestone 11 and its pinned synthesis CI fix. All 457 historical tests and six original RTL sources are unchanged. The complete suite now passes **473 tests in 23 files** on Node 22.9.0 and Node 24. All lint, formatting, TypeScript, contrast, production-build, and release asset checks pass. New raw-data/graph tests reject stale source hashes, invalid netlists, and fabricated unavailable values. Hardware datasets/path lists are loaded only on request; historical test assertions and timeout budgets remain unchanged.

Strict Verilator 5.052 lint, the original 462,172 add/sub corpus, all 301,544 multiplication/FMA/nonfused triples, both directed VCD flows, and the alternative rounder's entire corpus pass without result/flag/signal mismatches. All seven actual functional measurement wrappers additionally pass strict lint and full raw result/flag corpus replay. A pre-optimization structural check rejects undriven boundary signals. Yosys/ABC map seven controlled Nangate45 configurations; OpenSTA 3.1.0 completes constrained timing for all seven. The checksum-pinned OpenSTA/CUDD builder was tested end to end on macOS ARM64. The changed rounder separately passes combinational formal equivalence (98 proven, zero unproven equivalence cells), without claiming whole-unit IEEE proof. Repeated direct SoftFloat baseline checks pass under the existing documented NaN metadata policy.

[The full methodology](HARDWARE_SYNTHESIS.md) records actual area/delay/depth values, timing endpoints, library/tool source identifiers, raw-report locations, and limits. Per-run manifests capture commands, constraints, source/report hashes, source commit, dirty state, timestamps, seed, and verification evidence. No physical area, power, post-layout timing, silicon, GUI waveform, or rendered-browser verification is claimed. The separate manual-only Linux analysis workflow passes actionlint; no GitHub execution, push, merge, publication, or deployment was performed.

## Milestone 13 verification update

`feature/hardware-datapath-explorer` starts from latest main `135670a`, which
contains merged Milestone 12. All **473 historical tests**, numerical APIs,
original RTL, historical simulation/synthesis records, workflow permissions and
deployment triggers are unchanged. Six bounded pure record/provenance/reference/
VCD modules, a lazy dedicated workspace, and an offline waveform exporter add no
runtime or npm dependency. The suite now has **502 tests in 25 files** (29 new),
passing on **Node 22.9.0 and Node 24.22.0**. Lint, Prettier, TypeScript, 74 theme
contrast checks, production build, release asset checks and actionlint pass.

All 97 selectable operation/examples (28 add/sub plus 23 shared triples) agree
with the actual exact engine in raw result bits and every exception flag. They
also pass **768 available finite internal-signal checks**, using exact grid
scales and BigInt for wide values. Tests detect bit/flag/GRS corruption, stale
source hashes, unsupported signal widths, malformed artifacts, missing optional
data, mismatched synthesis sources, unsafe download paths and VCD inconsistencies.
Missing nonfused-adder wires remain explicitly unavailable. Two historical
product midpoint titles are corrected only in presentation using actual retained
parity/increment; original titles remain inspectable and old artifacts unchanged.

**Verilator 5.052** strict lint, full **462,172-case add/sub** regression and
**301,544-triple multiply/FMA/nonfused** regression pass with zero mismatches.
Both directed waveform flows pass (28 and 23). The new exporter checks full and
directed certificates/source hashes and correlates genuine VCD operands,
results/flags, GRS/increment, and available magnitude/product observations with
all 51 recorded sets before writing the curated artifact and raw downloads.
Normal tests independently reparse both raw VCDs and compare the exported subset.
Generic multiplier/FMA/comparison synthesis sanity checks also pass with the
CI-pinned **Yosys 0.70+219 (5053d0e26)**. No RTL arithmetic or synthesis
architecture changes, new physical measurements, new SoftFloat run, whole-unit
formal proof or new timing analysis are claimed. Milestone 12 measurements and
exact library/tool/constraint metadata are reused with core-source compatibility
checks.

An isolated **Chrome 156.0.8078.48** headless instance was driven using its local
DevTools protocol and Node's native WebSocket API. Actual rendered desktop
1440×1200 and mobile-emulated 390×844 / 320×800 checks covered direct `#datapath`
opening, SVGs, light/dark themes, visible focus outlines, bounded keyboard stage
controls, native waveform range-arrow behavior, long-code focus isolation,
emulated touch stage selection, signed zero/NaN/infinity, exact midpoint parity,
subnormal transition, fused/nonfused residuals, waveform controls and both raw
VCD URLs. Accessibility-tree checks found labeled combobox/navigation controls;
no runtime exceptions occurred. Every operation/stage with expanded software,
provenance/synthesis and waveform details was checked at 320 px. This caught and
fixed implicit grid minimum-width overflow and long provenance-label wrapping.
The diagram/long codes scroll internally while the page stays within its client
width. Actual light desktop and dark mobile screenshots are committed under
`docs/images/`.

This is not screen-reader certification, physical-device testing, cross-browser
verification, full contrast sampling or a complete accessibility audit. System
appearance and 200% zoom also remain manual. Follow the remaining
[reproducible manual checklist](HARDWARE_DATAPATH.md#browser-verification-and-remaining-manual-checklist).
Synthesis values are whole-unit mapping/timing estimates; logical blocks and VCD
observation indices are not physical placement, measured stage delay, hardware
clocks or latency. No push, merge, deployment, publication or repository-settings
change was performed.

## Milestone 14 research showcase candidate

`feature/research-showcase` starts from latest main **600aca6**, which includes
merged Milestone 13. The existing **v0.1.0** tag points to **977ccc8** (Milestone 7).
The subsequent work is prepared as **0.2.0-rc.1** in package/lockfile and the UI,
with private npm status retained. Old global documentation implying v0.1.0 was
still untagged has been corrected; no GitHub release/deployment status is inferred
from the tag. No new tag is created.

All **502 historical tests** are unchanged, as are numerical engines, original
RTL, historical simulation/waveform/synthesis artifacts and workflow behavior.
Twelve new evidence/UI/release/documentation tests bring the suite to **514 tests
in 27 files**, passing on **Node 22.9.0 and Node 24.22.0**. Checks bind all seven
compact showcase records and their metrics/endpoints/verification/conditions to
the original M12 JSON SHA-256, preserve measurement dirty-state provenance,
verify independent example encodings and percentage denominators, permit missing
timing, and exercise lazy navigation, keyboard controls, filtering and source
links. Local documentation target tests catch broken file links.

Lint, Prettier, TypeScript, all 74 theme-token contrast checks, production build,
release-path/version/private-package checks and actionlint pass. Static text
contrast remains at least 5.46:1 and tested control/focus pairs at least 3.39:1.
The release checker now also verifies current candidate text in built JavaScript,
third-party notices copied into the build, and both raw VCD files byte-for-byte
and against their recorded hashes. The lockfile changes only candidate versions;
no dependency is added/upgraded. The Pages workflow remains manual/main-only
with deployment writes confined to its deploy job; no settings were changed.

Applicable RTL checks were rerun with **Verilator 5.052**: strict lint, all
**462,172 add/sub transactions**, all **301,544 multiplication/FMA/nonfused triples**
and the **28/23 directed VCD flows**, zero mismatches. Generic synthesis/latch
sanity checks pass with CI-pinned **Yosys 0.70+219 / 5053d0e26**. No new mapped
area/timing measurement, SoftFloat run, formal proof or silicon validation is
claimed. The case study/showcase reuses the source-hashed M12 measurements and
previously recorded independent references, identifying overlapping corpora and
NaN-policy differences rather than summing them into an inflated count.

### Actual production-browser verification

The committed optional `scripts/check-showcase-browser.mjs` uses Node's native
WebSocket API and an isolated local **Chrome 156.0.8078.48** DevTools target, with
no new npm dependency. It passed against both the development server and the
**built production preview under /fp-play/**. Coverage includes:

- Desktop **1440×1100**, mobile-emulated **390×844** and **320×844**, with page
  scroll width bounded by client width across every hardware-unit filter and
  expanded provenance. Plot/table/code scrolling remains local.
- Native keyboard entry into `#research`, experiment-row Enter activation,
  visible focus outline, exact selection/endpoints, bounded filter results,
  and readable labels with native ArrowRight plot scrolling.
- Explicit light/dark switching and system-theme emulated dark/light preference,
  plus rendered SVG circle/diamond alternatives and exact table controls.
- Verified FP16 FMA/nonfused residual results, microscope zoom/pan, logical
  datapath stage keys, waveform signal/zoom/pan/range controls and native range
  ArrowRight behavior; emulated touch activates a datapath stage.
- Accessibility-tree audit: **207 nonignored interactive controls**
  (buttons, links, comboboxes, sliders and textboxes) all have nonempty names;
  research and stage-navigation labels are present. No runtime exceptions.
- HTTP 200 and exact source-byte equality for both genuine raw VCD downloads and the
  third-party notice asset at the production base.

Actual screenshots were captured and visually inspected: [desktop landing](images/research-landing-desktop-light.png),
[mobile dark landing](images/research-landing-mobile-dark.png),
[desktop hardware](images/research-hardware-desktop.png), and
[mobile dark hardware](images/research-hardware-mobile-dark.png).
Visual review caught tiny chart labels on narrow screens; the final plot uses a
focusable horizontal scroll region instead of shrinking labels. These are real
interface captures, not new scientific measurements.

The [reproduction guide](REPRODUCIBILITY.md#optional-real-browser-check-no-npm-dependency)
provides exact preview/browser/checker commands. Browser checks are local-only;
the checker closes only its own tab. It does not deploy or contact GitHub APIs.

### Remaining acceptance checks

Automated control names and contrast tokens do not establish full accessibility
or WCAG conformance. Physical touch devices, actual screen-reader announcements,
other browser engines, complete 200% zoom/device coverage and every composited
contrast state remain unverified. Review them with the existing
[manual checklist](HARDWARE_DATAPATH.md#browser-verification-and-remaining-manual-checklist),
adding the landing proof strip, all `#research` unit filters/table rows,
horizontal plot scrolling, source provenance, and documentation/download links.

Candidate source/build readiness is established locally; CI must still run on
GitHub after an authorized push. New documentation links to main become available
after the approved merge. Maintainer review, merge, Pages enablement/workflow
dispatch, candidate tag and GitHub prerelease all remain separate explicitly
authorized steps. Nothing was pushed, merged, deployed, published or tagged by
this milestone. See [the release procedure](RELEASING.md).
