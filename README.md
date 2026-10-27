# FP Play

**Make precision visible—from bits to verified RTL.** FP Play is an open-source
interactive laboratory for floating-point arithmetic, numerical precision in AI
workloads, and reproducible hardware experiments. Explore a number, follow an
exact calculation, and inspect the implementation and evidence behind it.

**Release candidate: v0.2.0-rc.1.** The repository already has a v0.1.0 tag;
this candidate prepares the subsequent numerical/hardware work for review.
No new tag, GitHub release or deployment is created by local preparation.

[Technical case study](docs/CASE_STUDY.md) · [Documentation](docs/README.md) ·
[Reproduce the results](docs/REPRODUCIBILITY.md) · [Contributing](CONTRIBUTING.md)

[Desktop landing screenshot](docs/images/research-landing-desktop-light.png) · [Hardware showcase screenshot](docs/images/research-hardware-desktop.png)

## Explore

| Experience                                              | What you can learn                                                                                                     |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Bit Explorer** (`#explorer`)                          | Flip FP16 bits; inspect exact values, signed zeros, subnormals, NaN metadata, decimal quantization and neighbors.      |
| **Precision Microscope** (`#microscope`)                | Zoom/pan over actual representable values, halfway boundaries, rounding cells and ULP spacing.                         |
| **Arithmetic / FMA** (`#arithmetic`, `#fma`)            | Follow exact add/subtract/multiply/FMA traces, GRS decisions, flags and one-round versus two-round results.            |
| **Formats / dot products** (`#formats`, `#dot-product`) | Compare FP16/BF16/FP32 and five explicit mixed-precision accumulation policies, sequential or pairwise.                |
| **Attention Precision Lab** (`#attention`)              | Inspect bounded QK/softmax/PV pipelines, stable/naive behavior, raw encodings and stage errors.                        |
| **Research showcase** (`#research`)                     | Compare seven recorded Nangate45 mappings, area/delay tradeoffs, critical endpoints and exact experimental conditions. |
| **Hardware / datapaths** (`#hardware`, `#datapath`)     | Inspect actual RTL observations, synthesis paths, software agreement, GRS and genuine recorded VCD waveforms.          |

The compact landing page links to every experience. Hash navigation works on
static hosting without server rewrites. Native controls, exact tables and full
scrollable binary values complement SVGs; light/dark/system themes share labeled
fields and visible focus states.

## Run in minutes

Use Node **22.9.0** / npm **10.8.3**, or Node **24**, and retain the lockfile.
No HDL tools, Python or SoftFloat installation is required for frontend use.

```sh
git clone https://github.com/nivupai/fp-play.git
cd fp-play
npm ci
npm run dev
# Open http://localhost:5173/fp-play/ (or the port Vite prints).
```

```sh
npm run check          # lint, formatting, contrast, full tests, types and build
npm run check:release  # built /fp-play/ paths, version, notices and VCD downloads
npm run preview       # http://localhost:4173/fp-play/
```

[Reproducibility](docs/REPRODUCIBILITY.md) separates these fast checks from
optional SoftFloat, Verilator, Yosys/ABC and OpenSTA setup. Full controls and
input conventions are in the [user guide](docs/USER_GUIDE.md). Decimal text parses
as binary64 first; special inputs include `-0`, `Infinity`, `-Infinity`, `NaN`
and arithmetic raw hex such as `0xfc11`.

## Two experiments to try

- Enter **1.0001** in Bit Explorer or the microscope. Its parsed binary64 value
  rounds to FP16 **1** (`0x3c00`); spacing near 1 is `2^-10`.
- Open **FMA Comparison**: `(1 + 2^-10)^2 − (1 + 2^-9)` is exactly `2^-20`.
  Fused returns `0x0010`; separately rounding the product first loses the
  residual and returns `0x0000`. Inspect the same triple in `#datapath`.

For attention, try **QK accumulation width**, then compare the scores,
probabilities and output. For hardware, inspect the baseline FMA and nonfused
critical paths; the showcase separates measured observations from hypotheses.

## Evidence, with boundaries

The existing **502 Milestone 13 tests are preserved unchanged**. Tests include
all **65,536 binary16 encodings**, independent lattice/midpoint checks, exact
trace reconstruction, BF16/FP32 and attention references, source-hash checks,
and actual-engine UI integration. CI runs the complete suite on Node 22.9.0 and 24. [Release verification](docs/RELEASE_VERIFICATION.md) records the new showcase
test count and commands.

Recorded independent checks include **536,888 FP16 SoftFloat operations**,
**169,011 FP32 operations**, and **240 attention pipelines / 12,916 stage
comparisons**. RTL passes **462,172 add/sub transactions** and **301,544 triples
per multiply/FMA/nonfused path**. Raw bits and all modeled flags are checked;
SoftFloat NaN metadata differences follow its documented specialization policy.
These overlapping corpora are not summed into a single verification claim.

The source-hashed [Nangate45 artifact](rtl/vectors/hardware-analysis.json) records:

| Baseline unit         | Mapped cell area (µm²) | Constrained estimated delay (ns) |
| --------------------- | ---------------------: | -------------------------------: |
| Add/subtract          |               1685.908 |                         2.917984 |
| Multiply              |               2048.466 |                         3.318500 |
| True FMA              |               3588.074 |                         4.179512 |
| Nonfused multiply-add |               3981.754 |                         5.652442 |

Under this one library/corner/configuration, baseline FMA is smaller and faster
than nonfused; no tree variant improves both area and delay. This is an
experimental observation, not a universal FMA cost law. Cell area is not chip
area; ideal-interconnect OpenSTA delay is not post-layout timing. No power,
achieved clock frequency, throughput or silicon measurement is claimed.
See the [case study](docs/CASE_STUDY.md) and [full synthesis conditions](docs/HARDWARE_SYNTHESIS.md).

## Architecture and limits

React/strict TypeScript render the interface; pure BigInt dyadic models perform
exact arithmetic before destination rounding. Presentation helpers stay outside
numerical modules. Original combinational SystemVerilog implements FP16 units;
separate verification tools produce small immutable browser records. Vite,
Tailwind, Vitest, ESLint and Prettier provide tooling. There is no plotting,
routing, numerical runtime or browser HDL dependency.

Only nearest-even is supported. This is **not full IEEE 754 conformance**.
NaN propagation has an explicit FP Play policy. FMA's Number projection can round;
exact coefficient/exponent fields are authoritative. Attention uses host
binary64 exp/scaling approximations, not correctly rounded format-specific
transcendentals. Workloads are bounded and do not model a particular accelerator.
RTL simulation and finite corpora are not whole-unit formal/silicon validation.

Actual headless Chrome checks and screenshots cover the landing, research showcase, charts and datapath/waveform workspaces;
[the verification record](docs/RELEASE_VERIFICATION.md) distinguishes current
showcase browser evidence from pending physical-device, screen-reader and
cross-browser review. Automated accessibility checks are not WCAG certification.

## Contribute, reproduce, release

Start with [CONTRIBUTING.md](CONTRIBUTING.md), the [documentation index](docs/README.md),
and [AGENTS.md](AGENTS.md). Documentation, accessibility and independent verification
are useful first contributions. Numerical APIs, historical assertions and
source-hashed artifacts must remain trustworthy. See the authoritative/generated
file map before refreshing records.

Future research may isolate alignment/rounding costs, examine fanout-aware
normalization, add controlled pipeline studies, or expand independent verification.
These are research directions, not measured improvements or release promises.

GitHub Pages uses `/fp-play/` and a **manual, main-only** workflow. Enabling Pages,
merging, deploying, tagging and publishing remain explicit maintainer actions;
see [the release procedure](docs/RELEASING.md) and [changelog](CHANGELOG.md).
No local check performs publication.

## License

Original FP Play code is [MIT-licensed](LICENSE), with the original attribution
preserved. Dependencies and external hardware tools/libraries retain their own
terms; see [third-party attribution](docs/THIRD_PARTY.md). No proprietary hardware
designs or internal Broadcom information are included.
