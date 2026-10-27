# Contributing to FP Play

FP Play is an educational scientific computing tool. Correct results, understandable explanations, accessible controls, and reproducible validation take priority over adding features quickly. Read [AGENTS.md](AGENTS.md) and the relevant [numerical](src/lib/numerics/README.md), [microscope](src/lib/microscope/README.md), or [arithmetic](src/lib/arithmetic/MULTIPLY_FMA.md) contract before changing behavior.

## Development

Use `.nvmrc` (Node 22.9.0) or Node 24, install with `npm ci`, and run `npm run dev`. Open the printed `/fp-play/` URL. Keep dependencies minimal and retain the lockfile. Package majors are deliberately selected for the supported runtimes; inspect engines, peer requirements, and transitive dependencies before upgrading. The private npm package flag prevents accidental registry publication; the public release is a static web app and source repository.

Keep pure numerical code in `src/lib/numerics/`, presentation/examples under the corresponding `src/lib/` directory, React UI in `src/components/`, and tests in `src/tests/`. Do not duplicate decoding, rounding, or arithmetic in visualization components. Hash navigation serves this static app; no client router or server fallback is needed.

Before submitting:

```sh
npm run check
npm run check:release
npm run preview
```

`check` runs lint, formatting, contrast-token checks, the complete test suite, TypeScript, and production build. `check:release` validates built `/fp-play/` asset paths without deploying. Preview at `http://localhost:4173/fp-play/`; follow the [manual review cases](docs/RELEASE_VERIFICATION.md) and report browser/device details and any checks you could not perform.

## Numerical changes

Preserve all existing assertions and exhaustive loops. New operations need independent expected values and boundary cases, rather than testing an encoder against its paired decoder or a trace against an approximate animation. Cover signed zeros, subnormals, exponent boundaries, NaN metadata, infinities, midpoint parity, carry, cancellation, overflow, underflow, and inexactness as applicable. Keep randomness deterministic. Use targeted per-test timeouts for expensive exhaustive checks; never reduce coverage or broadly increase timeouts to hide failures.

Use exact BigInt dyadics for arithmetic and one final nearest-even rounding where specified. FMA must not round or range-limit its product before addition. Tininess after rounding is a precision-only, unbounded-exponent concept, not simply the final encoded exponent. FMA's Number projection can round; exact coefficient/exponent fields drive exact displays.

For numerical changes, run the optional [SoftFloat corpus and counterexample experiment](src/lib/arithmetic/MULTIPLY_FMA.md#verification-and-reproducible-counterexamples) when available. Report upstream revision, cases, flags, NaN policy differences, and every unresolved mismatch. Do not silently exempt flags, approximate ties with epsilons, or claim full IEEE conformance from a finite corpus.

## UI and documentation changes

Use semantic landmarks/headings, labeled fields and errors, native buttons, visible focus, keyboard/text alternatives to pointer interaction, and reduced-motion support. Preserve exact numerical strings even when they need their own scroll area. Test at narrow widths and in light, dark, and system themes. Static contrast checks and jsdom do not replace rendered-browser and assistive-technology review.

Document original text, parsed binary64 inputs, and quantized operands separately. Invalid submissions retain the previous applied result. Explain special-value short circuits and the distinction between conceptual stages and physical hardware cycles. Keep README, user guide, and model contracts synchronized.

## Reports and pull requests

Bug reports should include steps, expected/actual behavior, raw patterns, decimal text, operation/rounding mode, and a minimal reproducible case. Include browser/OS details for UI issues and a seed/reference version for numerical differences. Avoid secrets, private code, and proprietary hardware information.

A focused pull request should explain the concrete problem, resulting behavior, validation performed, and limitations. Supply screenshots only if actually captured and reviewed. Preserve MIT attribution and include only original or appropriately licensed public material.

Pushing, merging, enabling Pages, deploying, tagging, and publishing a release are separate maintainer actions. The [manual release guide](docs/RELEASING.md) describes them; local checks never perform them automatically.

## Authoritative and generated files

| Files                                                                                   | Role and edit policy                                                                                                                                                   |
| --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/lib/numerics/`, typed model contracts, `rtl/src/`, original verification harnesses | Authoritative behavior. Changes need independent numerical/flag tests and relevant full RTL verification.                                                              |
| `src/lib/*/` presentation helpers, React components, CSS and Markdown                   | Authored presentation. Consume the real engine/records; never invent stage data or performance claims.                                                                 |
| `src/tests/reference/`, attention reference vectors and directed vectors                | Independent test evidence. Preserve historical cases/assertions and deterministic seeds; do not regenerate merely to accept implementation output.                     |
| `rtl/vectors/simulation.json`, `mul-fma-simulation.json`                                | Historical actual simulation records, source-hashed. Intentional replacement requires complete real verification and review; see the RTL contracts.                    |
| `rtl/vectors/hardware-analysis.json`                                                    | Authoritative portable **recorded measurement**, generated only by the complete verified hardware flow. Never hand-edit values.                                        |
| `src/lib/showcase/hardware-summary.json`                                                | Compact presentation derivative. Run `node scripts/export-showcase.mjs`, then Prettier on that file. Tests bind every value to the authoritative measurement artifact. |
| `rtl/vectors/datapath-waveforms.json`, `public/artifacts/*.vcd`                         | Derived from actual full/directed certificates and VCDs using the checked exporter; see the reproduction guide.                                                        |
| `rtl/build/`, `dist/`, downloaded tools/libraries                                       | Ignored generated output. Never commit binaries, downloaded cell libraries, private traces or fabricated reports.                                                      |
| `docs/images/`                                                                          | Actual reviewed screenshots; record browser/viewport/theme and limitations. They are not numerical or hardware measurement evidence.                                   |

## Three example contribution workflows

1. **Clarify an explanation:** choose a small issue, update its component/contract,
   verify links and actual values, then run `npm run check` and `check:release`.
   For visual changes, capture and inspect the real browser; state untested areas.
2. **Investigate a numerical discrepancy:** retain a minimal raw-input/flag case,
   establish an independent expected result, add boundary coverage, fix the pure
   engine, and run all historical tests plus relevant SoftFloat/RTL checks.
   Do not tune the reference or exempt flags to match an implementation.
3. **Change a hardware experiment:** preserve the baseline; document a controlled
   variant, run complete strict differential verification and the pinned analysis
   flow, and review new hashes/conditions/reports. Refresh showcase derivatives
   only after the authoritative measurement is valid. Label hypotheses separately.

[The reproduction guide](docs/REPRODUCIBILITY.md) gives fast and advanced commands;
[the documentation index](docs/README.md) maps each contract. The issue/PR templates
ask for minimal inputs, actual checks and licensing provenance. FP Play's MIT
license does not replace [external licenses](docs/THIRD_PARTY.md).
