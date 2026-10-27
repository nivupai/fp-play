# FP Play: from a rounding decision to a hardware experiment

FP Play connects interactive numerical experiments to inspectable software,
original SystemVerilog, and recorded synthesis evidence. It is an educational
laboratory and an engineering case study, not an accelerator benchmark or an
IEEE 754 conformance certification. This study describes implemented work through
Milestone 13; Milestone 14 improves its presentation and release preparation.

[Documentation index](README.md) · [Reproduce](REPRODUCIBILITY.md) ·
[Source](https://github.com/nivupai/fp-play)

## 1. Motivation and goals

A numerical expression is incomplete without its representation, rounding
boundaries, accumulation format, and operation order. Those choices matter in
scientific computing and AI attention, while their hardware implementations have
costs that cannot be inferred from algebra alone. FP Play makes both kinds of
choices visible. A learner can flip an exponent bit, inspect an exact discarded
tail, compare fused and separately rounded results, then inspect a real RTL wire
and the measured mapping of the corresponding unit.

The engineering priorities are exact arithmetic, independently checkable results,
small pure models, accessible text alternatives, and source-hashed provenance.
The project uses public standards and original or appropriately licensed code;
it does not model a proprietary accelerator or use employer hardware designs.

## 2. IEEE 754 representation

Binary16 has one sign bit, five exponent bits, ten stored fraction bits, and bias 15. For a normal, its value is `(-1)^s × (1 + f/1024) × 2^(e-15)`.
For a subnormal it is `(-1)^s × f × 2^-24`. Raw exponent 31 encodes infinities
or NaNs; exponent/fraction zero encodes signed zero. Raw encodings retain NaN
metadata that a JavaScript Number cannot reliably preserve.

Bit Explorer and Precision Microscope show this lattice rather than interpolating
an imaginary continuum of FP16 values. Decimal entry first parses binary64:
`1.0001` rounds to FP16 `1` (`0x3c00`). Spacing at 1 is `2^-10`; smaller
magnitudes have finer absolute spacing until the fixed subnormal quantum
`2^-24`. BF16 and FP32 share the format lab, with explicit conversion and
accumulation policies. See [representation/API semantics](../src/lib/numerics/README.md).

## 3. Exact FP16 arithmetic and rounding

The TypeScript engine represents finite intermediates as a signed BigInt
coefficient and a power-of-two exponent. Alignment, multiplication and addition
are exact before a destination rounding. Normalization and discarded bits feed
round-to-nearest, ties-to-even: increment when `G AND (R OR S OR retained LSB)`.
Traces are returned by the actual calculation, not an independent animation.

Signed zero, gradual underflow, infinity, signaling/quiet NaNs and all four modeled
indicators are explicit. Tininess is detected after precision-only rounding with
an unbounded exponent; a packed normal `0x0400` may still raise underflow when
inexact. Tiny binary16 additions are exact on the subnormal grid and cannot raise
arithmetic underflow. Overflow raises inexact. Only nearest-even is supported;
conversion flags, traps, alternate modes and global sticky state are not modeled.
Exact coefficient/exponent is authoritative; the compatibility Number projection
can round wide FMA sums. See [the arithmetic contract](../src/lib/arithmetic/MULTIPLY_FMA.md).

## 4. Fused versus nonfused multiply-add

Use A=B=`0x3c01` (`1 + 2^-10`) and C=`0xbc02` (`-(1 + 2^-9)`).
The exact expression is `2^-20`. True FMA rounds once and returns `0x0010`.
Separately rounding the product gives `0x3c02`; adding C then returns `0x0000`.
The second operation cannot restore discarded product bits.

A second verified case is A=`0x7bff`, B=`0x4000`, C=`0xfbff`.
The exact fused result is 65504 (`0x7bff`), but the nonfused product overflows,
and the nonfused output remains infinity (`0x7c00`). FMA does not inherit
intermediate-product flags; nonfused unions the two real operations' flags.
These are counterexamples, not a guarantee that FMA improves every application.
Seven discovered examples and 21 additional SoftFloat checks are recorded in
[the experiment](experiments/fma-counterexamples.json).

## 5. Mixed precision and transformer attention

The dot-product lab distinguishes five explicit input/product/accumulator policies,
including FP16 products rounded before FP32 accumulation versus exact products
inside FP32 FMA. Sequential and pairwise evaluation are different computation
graphs. Exact original-input and quantized-input references remain separate.

The bounded attention lab follows QK, scaling, stable/naive softmax, stored
probabilities, and PV. Stable max subtraction can avoid exponential overflow,
but wider accumulation cannot recover input quantization. Twelve presets expose
these effects with raw encodings, stage events and shared-scale heatmaps.
The five fixed configurations model numerical policies, not a named chip.
`Math.exp` and reciprocal-square-root scaling are host binary64 approximations,
followed by format quantization; format-specific correctly rounded transcendental
operations are not claimed. Quotient normalization uses exact rational rounding.
Independent verification records 240 full pipelines, 12,916 stage result/flag
comparisons, 48 aligned NumPy FP32 calculations, and a Decimal(100) reference.
See [attention scope and evidence](ATTENTION_PRECISION.md).

## 6. Original SystemVerilog architecture

The combinational add/sub unit aligns exact magnitudes on a `2^-24` lattice with
40-bit operands and a 41-bit result. The multiplier retains its exact 22-bit
11×11 product. FMA places that product and C on a `2^-48` grid with 80-bit aligned
product, 64-bit addend and 81-bit accumulator, then rounds once. Nonfused connects
the real rounded multiplier to the original adder and unions their flags.

These intentionally wide educational datapaths retain cancellation tails.
Observation ports expose actual internal wires. Logical blocks in the viewer
are not pipeline cycles, physical placement or an extracted gate topology.
The alternative rounder changes the leading-bit search to bounded binary
partitions; it leaves the arithmetic grid and rounding/exception semantics intact.
See [add/sub](../rtl/README.md) and [multiply/FMA widths](../rtl/MULTIPLY_FMA.md).

## 7. Verification from software to RTL

Evidence is layered rather than summarized as a universal proof:

| Evidence                   | Recorded extent and interpretation                                                                                                                                                  |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Representation             | Every one of 65,536 binary16 encodings, including NaN canonicalization, plus independently derived midpoint boundaries.                                                             |
| Independent FP16 reference | 536,888 SoftFloat operations; non-NaN bits, quiet NaN classification and all four flags match. Implementation-specific NaN metadata is checked separately against FP Play's policy. |
| Mixed arithmetic           | 169,011 FP32 SoftFloat operations, zero mismatches under the documented NaN policy; exhaustive BF16 classifications/midpoints and independent arithmetic/reduction references.      |
| RTL add/sub                | 462,172 transactions; all raw results, flags and applicable internal observations match. Direct SoftFloat uses its documented metadata policy.                                      |
| RTL multiply/FMA/nonfused  | 301,544 triples per path; strict FP Play raw metadata and every flag, with independent lattice checks. Direct SoftFloat compares 904,632 outputs.                                   |
| Alternative rounder        | Same deterministic differential corpus plus 98 proven two-state equivalence cells, zero unproven. This proves the changed rounder, not whole-unit IEEE conformance.                 |
| Interactive datapath       | 97 selectable operation/examples from 51 recorded sets, with 768 available finite internal-signal checks; absent wires remain unavailable.                                          |

These recorded corpora overlap and are not added into a single headline count.
Full operand-pair/triple enumeration, TestFloat, silicon and whole-unit formal
conformance are not claimed. NaN payload selection is explicitly an FP Play
policy, not a universal IEEE requirement. The 502-test Milestone 13 suite is
preserved as Milestone 14 adds presentation/release checks.

## 8. Controlled synthesis methodology

Functional-only wrappers retain the result and four indicators; unused debug
outputs are consistently pruned. Seven configurations use Yosys 0.70+219
(5053d0e26), ABC 1.01, OpenSTA 3.1.0 and the checksum-pinned Nangate45 typical
library at 1.10 V / 25 °C. Mapping uses INV_X1 drive, 10 fF load and a 1000 ps ABC
optimization target. IO timing uses 0.05 ns input transition, 10 fF output load,
zero input/output delays and a 100 ns virtual clock, with ideal interconnect.
The optimization target is not an achieved path delay or a hardware clock.

The measurement snapshot records source commit
`f5d11a0b86548072402c8c5a314f585d4279037a`, dirty working tree, and timestamp
`2026-10-26T17:56:12.182Z`. Source hashes identify the actual measured inputs;
the commit alone does not. Raw per-run manifests preserve scripts, constraints,
report hashes and tools. The authoritative portable evidence is
[`rtl/vectors/hardware-analysis.json`](../rtl/vectors/hardware-analysis.json).
Milestone 14 reuses these measurements; it does not create new benchmark results.

## 9. Hardware design-space results

| Unit / variant    | Mapped cells | Cell area (µm²) | Estimated delay (ns) | Logic depth |
| ----------------- | -----------: | --------------: | -------------------: | ----------: |
| Add/sub baseline  |         1539 |        1685.908 |             2.917984 |          99 |
| Multiply baseline |         1851 |        2048.466 |             3.318500 |         109 |
| True FMA baseline |         3221 |        3588.074 |             4.179512 |         140 |
| Nonfused baseline |         3374 |        3981.754 |             5.652442 |         189 |
| Multiply tree     |         1801 |        2085.440 |             3.422190 |          99 |
| True FMA tree     |         3205 |        3801.406 |             5.031622 |         149 |
| Nonfused tree     |         3333 |        4069.268 |             6.009750 |         196 |

**Observation:** baseline FMA uses 9.89% less mapped area and 26.06% lower
estimated delay than nonfused, with nonfused as the denominator. Its selected
critical path is `b[0] → result[6]`; nonfused's is `b[1] → result[2]`.
**Hypothesis:** one exact accumulation and one final rounding may avoid serial
intermediate-rounding and second-unit costs. No isolated ablation measures those
individual contributions, so this is not a causal proof or a general FMA cost law.

The tree multiplier reduces maximum graph depth from 109 to 99 while increasing
area and delay. Fewer cells/levels need not mean smaller/faster mapping: selected
cell types, drive strengths, fanout and slew matter. Tree FMA's timing endpoint
is `flags[1]`, the underflow flag; it is not valid to attribute its entire delay
to a named RTL rounding stage. ABC optimizes across those logical boundaries.

Functional simulation demonstrates sampled result/flag agreement. Mapped cell
area sums library areas; it is not physical die area. Constrained estimated
timing uses cell arcs and ideal interconnect; it is not post-layout timing.
Graph depth is a separate structural metric. No power, manufacturable layout,
throughput, achieved frequency or physical chip measurements are reported.
See [the full synthesis methodology](HARDWARE_SYNTHESIS.md).

## 10. Limitations and future research

The experiments cover one library/corner and configuration, original educational
combinational units, and bounded numerical workloads. Four-state behavior,
physical design, clocked pipelines, PVT sweeps, routing/parasitics and silicon
validation remain outside the evidence. The lightweight waveform viewer displays
real directed VCD observations; indices are not propagation delays or cycles.
Nonfused adder internals absent from the recorded artifact are never fabricated.

Future studies could isolate rounding/alignment costs, measure fanout-aware
normalization, explore constrained pipelines, and extend independent verification.
Each would require new numerical/flag checks and controlled measurements.
No new format, arithmetic operation, optimization win or physical implementation
is introduced in this showcase. The [reproduction guide](REPRODUCIBILITY.md) and
[release evidence](RELEASE_VERIFICATION.md) separate available automation from
remaining browser, device and assistive-technology review.
