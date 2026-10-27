# Reproduce FP Play

[Documentation index](README.md) · [Case study](CASE_STUDY.md) ·
[Source and generated files](../CONTRIBUTING.md#authoritative-and-generated-files)

## Fast start: frontend and numerical checks

Use Node **22.9.0** with npm **10.8.3**, or Node **24**, and the committed lockfile.
No HDL tools, Python, external libraries or reference binaries are needed here.

```sh
git clone https://github.com/nivupai/fp-play.git
cd fp-play
npm ci
npm run dev
# http://localhost:5173/fp-play/
```

```sh
npm run check          # lint, formatting, 74 contrast pairs, all tests, types, build
npm run check:release  # built base paths, version, notices and VCD downloads
npm run preview       # http://localhost:4173/fp-play/
```

Try `#microscope`, `#fma`, `#research`, and `#datapath`. The last two load detailed
records only on request. Normal tests retain exhaustive raw encodings, independent
rounding/arithmetic oracles, deterministic attention vectors, source hashes, and
actual-engine UI integration. CPU-heavy tests keep individual timeout budgets;
never shorten corpora to accommodate CI. Browser screenshots and remaining manual
checks are described in [release verification](RELEASE_VERIFICATION.md).

## Independent numerical references (optional)

SoftFloat is external development tooling. It is not an npm/browser dependency.
Build the pinned public upstream revision with C/make; the following tested
upstream build directory is for Linux x86-64 and may need adjustment elsewhere:

```sh
git clone https://github.com/ucb-bar/berkeley-softfloat-3.git /tmp/fp-play-softfloat
cd /tmp/fp-play-softfloat
git checkout a0c6494cdc11865811dec815d5c0049fba9d82a8
make -C build/Linux-x86_64-GCC -j2
# Back in the FP Play repository:
cc -O2 -I/tmp/fp-play-softfloat/source/include \
  scripts/softfloat-driver.c \
  /tmp/fp-play-softfloat/build/Linux-x86_64-GCC/softfloat.a \
  -o /tmp/fp-play-softfloat-driver
node scripts/validate-softfloat.mjs /tmp/fp-play-softfloat-driver
node scripts/validate-fp32-softfloat.mjs /tmp/fp-play-softfloat-driver
```

The driver sets nearest-even and after-rounding tininess, clearing status per
operation. The established corpora contain 536,888 FP16 and 169,011 FP32 operations.
Compare non-NaN bits and every flag strictly; quiet NaN classification/flags are
strict while specialization-specific metadata differences are counted separately.
See [the driver protocol](../src/lib/arithmetic/MULTIPLY_FMA.md#verification-and-reproducible-counterexamples)
and [FP32 extension](MIXED_PRECISION.md). Do not combine overlapping corpus counts.
Optional Python/NumPy/Decimal attention verification is documented in
[the attention contract](ATTENTION_PRECISION.md); normal CI replays its committed
240 independently generated vectors without Python.

## Advanced: RTL simulation and genuine waveforms

Install Verilator and a C++17 toolchain separately. Local recorded verification
uses Verilator 5.052; RTL CI logs its Ubuntu distribution's version. These
commands run strict lint and the full deterministic result/flag/signal corpora:

```sh
make -C rtl test waveform
make -C rtl -f mul-fma.mk test waveform
node rtl/scripts/softfloat.mjs /tmp/fp-play-softfloat-driver
node rtl/scripts/softfloat-mul-fma.mjs /tmp/fp-play-softfloat-driver
```

Full counts are 462,172 add/sub transactions and 301,544 triples per new path.
Directed VCDs contain 28 and 23 recorded sets. Values hold across omitted unchanged
timestamps. VCD observation indices are not hardware clocks or propagation delay.
Open `rtl/build/directed.vcd` and `rtl/build/mul-fma/directed.vcd` in an external
viewer, or use the browser's curated waveform subset.

Only after successful full and directed verification, refresh the **new waveform
snapshot and raw downloads** with:

```sh
node rtl/scripts/export-datapath-waveforms.mjs
npx prettier --write rtl/vectors/datapath-waveforms.json
```

This validates certificates, source hashes and all directed correlations before
export. Historical `simulation.json` and `mul-fma-simulation.json` are not replaced.
Their intentional regeneration requires the commands in the
[add/sub guide](../rtl/README.md) and [multiplier/FMA guide](../rtl/MULTIPLY_FMA.md).

## Advanced: synthesis and constrained timing

Use checksum-pinned OSS CAD Suite **2026-10-24**, Yosys **0.70+219 / 5053d0e26**,
and ABC **1.01**. The read-only RTL workflow and
[toolchain contract](../rtl/MULTIPLY_FMA.md#reproducible-ci-synthesis-toolchain)
provide Linux/macOS archive URLs, digests and version checks. Do not substitute a
parser/tool revision silently.

```sh
make -C rtl -f mul-fma.mk synth YOSYS=/path/to/oss-cad-suite/bin/yosys
```

For the full seven-unit Nangate45 analysis, follow the OS-specific build
prerequisites and pinned downloads in [HARDWARE_SYNTHESIS.md](HARDWARE_SYNTHESIS.md):

```sh
bash rtl/analysis/build-opensta.sh  # verifies public OpenSTA 3.1.0 and CUDD sources
make -C rtl -f analysis.mk analysis \
  YOSYS=/path/to/oss-cad-suite/bin/yosys \
  STA="$PWD/rtl/build/analysis/tools/opensta-build/sta"
node rtl/analysis/summarize.mjs
```

Analysis includes full baseline/variant/wrapper verification, rounder equivalence,
Yosys/ABC mapping and constrained OpenSTA timing. Without OpenSTA, timing is
explicitly unavailable; a tool failure is not replaced with cached measurements.
The public Nangate45 library has its own terms/notices, independent of FP Play's
MIT license. Tools/libraries stay in ignored build directories. Never vendor them
or claim mapped cell area/ideal-interconnect timing is physical chip validation.

## Artifacts, source hashes and intentional refresh

Raw scripts/netlists/reports/manifests stay under `rtl/build/analysis/`, with
source/report hashes, source commit and dirty state, UTC timestamp, library/tool
versions and constraints. The portable committed measurement snapshot is
`rtl/vectors/hardware-analysis.json`. To replace it, perform a complete verified
analysis and use its documented `--export` command; review all changed provenance.
Normal tests reject stale original source hashes.

The compact **showcase derivative** may be regenerated from that unchanged
snapshot without running synthesis:

```sh
node scripts/export-showcase.mjs
npx prettier --write src/lib/showcase/hardware-summary.json
npm run check
npm run check:release
```

Its source path, SHA-256, original measurement commit/dirty state, constraints and
all seven values are checked against the authoritative artifact. This command
copies measurements; it never creates new ones. Screenshots are actual browser
captures, not generated scientific evidence. See the
[contributor file map](../CONTRIBUTING.md#authoritative-and-generated-files) for
which files to edit and which certificates require real tool runs.

## Optional real-browser check (no npm dependency)

An installed Chrome/Chromium with DevTools and Node 22/24 can exercise the real
production preview using the committed local-only checker. Use three terminals:

```sh
# Terminal 1, after npm run check / check:release:
npm run preview -- --host 127.0.0.1

# Terminal 2: isolated temporary profile; adjust the executable on Linux.
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --no-first-run --no-default-browser-check \
  --remote-debugging-port=9230 \
  --user-data-dir="$(mktemp -d /tmp/fp-play-browser.XXXXXX)" about:blank

# Terminal 3, repository root:
node scripts/check-showcase-browser.mjs
```

The checker creates/closes only its own tab, uses localhost, and prints temporary
screenshot paths for visual inspection. `FP_PLAY_CDP_PORT` changes the local
DevTools port; `FP_PLAY_PREVIEW_URL=http://127.0.0.1:5173/fp-play/` can target the
dev server. Stop the isolated browser/preview when finished. Never expose a
DevTools port publicly or run this against a personal browser profile.

It checks native keyboard/focus and range behavior, all themes including emulated
system preference, desktop/mobile width bounds, exact examples, research
filter/selection and critical endpoints, existing microscope/datapath/wave controls,
static downloads, runtime exceptions and accessible control names. Review the
actual screenshots; a passing script is not a screen-reader, physical-device,
cross-browser or WCAG certification. The remaining checklist is in
[release verification](RELEASE_VERIFICATION.md).
