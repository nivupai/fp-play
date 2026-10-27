# RTL synthesis and hardware design-space exploration

Milestone 12 adds a read-only **Hardware Analysis** workspace (`#hardware`) backed by real Yosys/ABC/OpenSTA measurements. Milestone 11 is merged into main as `ef49251`. Its six original RTL source files, 457 tests, numerical APIs, and existing simulation artifacts remain unchanged. The framework measures educational hardware complexity, not physical silicon performance.

## Controlled experiment

`rtl/analysis/units/` defines functional-only boundaries for add/subtract, multiplication, true FMA, and nonfused multiply-then-add. Outputs are the 16-bit result and four exception flags. Debug ports terminate in explicitly unused local wires and are pruned in every run; measuring them as outputs would retain hardware that is unnecessary for arithmetic. Add/subtract has 33 input bits (two operands plus opcode), multiply 32, and FMA/nonfused 48. All have 20 output bits. All synthesized cells are combinational; sequential count is zero.

Nonfused uses the original separately rounded FP16 multiplier feeding the original adder, with the actual intermediate multiplication flags unioned with addition flags. It is not approximated by FMA, and the multiplier's exceptions remain observable through that union. FMA retains the exact product and rounds once. The synthesis wrappers do not change these semantics.

Every run uses the same `synth -flatten -noabc` optimizations, checks, functional interfaces, library, ABC mapping settings, and timing assumptions. The generic snapshot is taken before technology mapping. ABC then maps using the Liberty library, `INV_X1` driving-cell model, 10 fF output load, and a 1,000 ps delay target. This is an optimization request, not a guaranteed achieved delay or supported clock frequency. Per-run scripts preserve the exact commands and paths. `check -assert`, sequential/latch rejection, and graph-cycle/undriven-net checks remain active.

## Library and tools

- Yosys **0.70+219**, git **5053d0e26**, and ABC **1.01** (compiled 2026-10-24), from checksum-pinned OSS CAD Suite **2026-10-24**, as in the [RTL toolchain contract](../rtl/MULTIPLY_FMA.md#reproducible-ci-synthesis-toolchain). Analysis rejects a different Yosys revision and records the complete ABC version.
- OpenSTA **3.1.0**, source **150bc0dbd103cbc97c0f85f63c4c8338826121fd**. `build-opensta.sh` verifies the source archive SHA-256, builds public CUDD **3.0.0** (`f54f533303640afd5dbe47a05ebeabb3066f2a25`) and verifies its archive too. Tcl **8.6**, Eigen, SWIG, Flex/Bison, CMake, and a C++ compiler are development-only tools. The builder disables OpenSTA's optional unit-test build, not any FP Play checks. Local simulation uses Verilator **5.052**.
- **Nangate45 typical**, 1.10 V, 25 °C, from OpenROAD-flow-scripts revision **64939f3717ad7c694199ba04cb98924b98022868**. Liberty SHA-256: **8d540a4d4cf6d09d27c87ad067857a9c0c2eeb023ab7a56e058cd3113db4e9b1**. The script downloads and verifies it under ignored build output, without vendoring it. The public [platform README](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/64939f3717ad7c694199ba04cb98924b98022868/flow/platforms/nangate45/README.md) identifies `PDKv1.3_v2010_12.Apache.CCL`, its research purpose, and non-manufacturability; see its [license](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/64939f3717ad7c694199ba04cb98924b98022868/flow/platforms/nangate45/LICENSE). Original library notices remain in downloaded files. No library cells, models, proprietary accelerator designs, or tool binaries are included in this repository.

Liberty time is ns and capacitance is fF. Mapped area is the sum of library cell-area values, conventionally square micrometres for Nangate45. It excludes placement utilization, routing, buffers added during physical design, IO pads, power distribution, and physical-design overhead. It is not die area or a manufacturable chip estimate.

## Timing assumptions and path interpretation

The checked-in `rtl/analysis/timing.sdc` is retained alongside generated reports and is source-hashed in each manifest. Configuration/SDC disagreement fails the run. The constraints are:

- Virtual IO clock period: **100 ns**, with no hardware clock port or sequential logic.
- Input arrivals and output delays: **0 ns**, both min and max, associated with that virtual clock.
- Input transition: **0.05 ns**; every output load: **10 fF**.
- **Ideal interconnect**: no SPEF, placement, routing, extracted parasitics, or clock tree.

OpenSTA's reported max input-to-output path is constrained by those IO assumptions. It is not an unconstrained graph path. Data arrival equals path delay because input delay is zero. The arbitrary virtual clock does not establish an achievable clock frequency for a pipelined implementation. No clock uncertainty, PVT sweep, physical design-rule closure, or post-layout timing is claimed.

Each timing record retains the reported startpoint, endpoint, actual cell/pin sequence, incremental cell delay, cumulative arrival, and raw report hash. Mapped **logic depth** is separately computed by topologically traversing the synthesized netlist: one level per combinational cell, zero per wire/constant. Its maximum-depth path can differ from the timing-critical path because cell arcs, slew, and fanout affect delay. Unknown port directions, multiple drivers, cycles, undriven nets, or sequential cells fail rather than producing plausible depth numbers.

ABC optimizes across normalization, alignment, rounding, and packing boundaries. A reliable per-RTL-stage critical-path breakdown is therefore **unavailable**, explicitly shown in the interface; no invented stage percentages are assigned. A concrete rounding-related observation is the tree-FMA worst path terminating at `flags[1]`, the underflow indicator. This identifies the endpoint's function, not a measured delay attributed to an entire RTL stage.

## Variant and verification

The immutable baseline uses a linear leading-index priority scan. `rtl/variants/tree-leading-bit/fp16_round_grid48.sv` replaces only that search with bounded binary partitions (64, 32, 16, 8, 4, 2, 1) on a zero-extended 128-bit vector. The exact 81-bit magnitude, retained precision, GRS, increment, packing, tininess, exception behavior, and module interface remain the same. The variant is evaluated for multiplier, FMA, and nonfused units; the unchanged adder is not duplicated as a fake alternative.

`make -C rtl -f analysis.mk verify` runs:

1. The original **462,172** add/sub transactions and directed VCD flow.
2. The original **301,544** triples per multiply/FMA/nonfused path, including 100,000 seeded random triples, exhaustive raw-pattern families, 4,913 special cross-product triples, 23 directed cases, all four flags, and strict FP Play NaN payload/sign policy.
3. Variant Verilator `--Wall` lint/build with no warning waivers, then the same complete and directed corpora. Every raw output, intermediate flag, and recorded internal signal must equal the independently verified baseline. No NaN exemptions are used. A mismatch fails the command; the directed VCD, input vectors, and observed rows remain available for replay. The reference numerical engine and independent lattice oracle are unchanged.

4. Each of the **seven actual functional measurement wrappers** additionally passes strict Verilator lint and replays its complete and directed numerical corpus, comparing the raw result and all four flags. This independently checks the measurement-boundary wiring; a wrapper must have a current source/vector-hashed certificate before analysis. A structural `check -assert` also runs **before optimization**, so undriven outputs cannot be silently converted into constants.

The analysis flow also executes `equiv-rounder.ys`: **98 equivalence cells proven, zero unproven**, establishing two-state combinational equality of all rounder outputs for every 81-bit magnitude and sign. This is a proof of the changed rounder, not an IEEE specification proof of the whole arithmetic unit. Whole-unit coverage is still reported as corpus-based. Verification certificates, source hashes, and vector/observation hashes must match the actual current inputs before measurement.

## Actual results

All seven runs completed mapping and constrained timing. No result/flag/signal mismatches, latch inference, sequential cells, or combinational cycles were observed.

| Unit / variant    | Generic cells | Mapped cells | Cell area (µm²) | Estimated delay (ns) | Mapped depth | I/O bits |
| ----------------- | ------------: | -----------: | --------------: | -------------------: | -----------: | -------- |
| Add/sub baseline  |          2563 |         1539 |        1685.908 |             2.917984 |           99 | 33/20    |
| Multiply baseline |          3308 |         1851 |        2048.466 |             3.318500 |          109 | 32/20    |
| True FMA baseline |          5540 |         3221 |        3588.074 |             4.179512 |          140 | 48/20    |
| Nonfused baseline |          5877 |         3374 |        3981.754 |             5.652442 |          189 | 48/20    |
| Multiply tree     |          3767 |         1801 |        2085.440 |             3.422190 |           99 | 32/20    |
| True FMA tree     |          6004 |         3205 |        3801.406 |             5.031622 |          149 | 48/20    |
| Nonfused tree     |          6336 |         3333 |        4069.268 |             6.009750 |          196 | 48/20    |

Under this configuration, baseline FMA maps to less area and shorter delay than separately rounded multiply-then-add. That is an observation about these exact implementations and assumptions, not a universal FMA cost claim. The tree reduces multiplier depth and cell count but increases area and delay; FMA/nonfused are also slower and larger despite slightly fewer cells. Different drive strengths and cell types explain why fewer cells need not mean less area or delay. It is **not promoted as an optimization win**.

Baseline critical paths are `a[13] → result[5]` (add/sub), `b[0] → result[1]` (multiply), `b[0] → result[6]` (FMA), and `b[1] → result[2]` (nonfused). Corresponding cell sequences and all alternative paths are inspectable in the recorded interface and raw reports. Future experiments could investigate the unusually wide exact alignment/rounding cones, sticky reductions, or fanout-aware normalization, but any numerical change requires full result/flag validation and new controlled measurements. These are opportunities, not implemented variants or promised improvements.

## Reproduce

Frontend-only setup still needs only npm. Install HDL tools separately. Use the pinned suite's absolute Yosys path. For optional timing builds on Ubuntu, install `build-essential cmake bison flex libfl-dev swig libeigen3-dev tcl8.6-dev zlib1g-dev libreadline-dev libspdlog-dev`. On macOS, the tested dependencies are Homebrew `cmake bison flex eigen swig tcl-tk@8`; the builder supplies the required keg paths.

```sh
# Optional: builds the exact public OpenSTA/CUDD sources, checking both digests.
bash rtl/analysis/build-opensta.sh

# Full baseline/variant verification, formal rounder check, mapping, timing, reports:
make -C rtl -f analysis.mk analysis \
  YOSYS=/path/to/oss-cad-suite/bin/yosys \
  STA="$PWD/rtl/build/analysis/tools/opensta-build/sta"
node rtl/analysis/summarize.mjs

# Explicitly refresh the portable browser snapshot after a complete verified run:
YOSYS=/path/to/oss-cad-suite/bin/yosys \
STA="$PWD/rtl/build/analysis/tools/opensta-build/sta" \
  node rtl/analysis/run.mjs --export rtl/vectors/hardware-analysis.json
npx prettier --write rtl/vectors/hardware-analysis.json
npm run check
npm run check:release
```

Without OpenSTA, mapping/depth can still complete, with `timing: null` and a precise unavailable reason. The UI omits fabricated scatter points and retains the real area/depth table. Installed timing-tool errors or missing constrained reports fail the run. Missing Yosys, failed ABC mapping, wrong library checksum, mismatches, or stale verification inputs also fail. Generated output never replaces a failed analysis with cached metrics.

Each `rtl/build/analysis/<variant>-<unit>/` contains the actual synthesis script, ABC constraints, generic/mapped JSON statistics and netlists, mapped Verilog, timing script/report, and machine-readable manifest. Manifests record source commit, dirty-worktree flag, UTC timestamp, source hashes, tool versions, library revision/checksum, settings, commands, metrics, report hashes, verification method/count/seed, and timing limitations. Source hashes describe the exact measured working tree, even before a commit. `summary.json` and `summarize.mjs` collect runs without mixing configurations. The small `rtl/vectors/hardware-analysis.json` is the browser's recorded data; tests reject stale source hashes. Raw reports and downloaded tools/libraries stay ignored under `rtl/build/`.

`.github/workflows/hardware-analysis.yml` is an expensive **manual-only, read-only** workflow. It builds pinned public timing tools and uses the same pinned Yosys/ABC suite; it does not publish artifacts, push commits, deploy, or change repository settings. Ordinary frontend CI does not install or invoke synthesis tools. The existing RTL CI remains intact. Local measurements and the tool builder were tested on macOS ARM64; the new Linux workflow is syntax-checked but has not been run on GitHub.

## Limits

One library/corner, one ABC configuration, and one normalization variant are measured. There are no alternative alignment, GRS, shared-rounder scheduling, pipelining, placement, power, or multi-corner experiments. Gate depth is not delay; mapped cell area is not chip area; estimated timing is not post-layout timing. No silicon measurements or proprietary hardware models are used. The proof is two-state rounder equivalence, not four-state semantics or full IEEE conformance. Browser tools are unavailable, so DOM/keyboard tests do not establish visual layout, touch, theme, or screen-reader correctness.
