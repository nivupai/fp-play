# FP16 multiplication and fused multiply-add RTL

Milestone 11 extends the original adder without changing its implementation, APIs, 462,172-case corpus, or recorded artifact. These independently developed educational combinational modules use only public numerical specifications. They are intentionally wide and unoptimized, with no vendor or employer architecture.

## Modules and interfaces

- `src/fp16_mul.sv`: binary16 A × B, one nearest-even rounding.
- `src/fp16_fma.sv`: true A × B + C, exact intermediate product and sum, one final rounding.
- `src/fp16_mul_fma_core.sv`: common unpack/classify, exact product and signed-magnitude alignment/accumulation. `include_c` selects multiplication or true FMA.
- `src/fp16_round_grid48.sv`: finite normalization, GRS, parity-controlled increment, carry, packing, and after-rounding tininess.
- `src/fp16_mul_fma_compare.sv`: independent multiplier and true FMA instances plus the unchanged `fp16_addsub` receiving the **rounded** product and C.

Operands and results are raw 16-bit patterns. The multiplier/FMA expose separate `invalid`, `overflow`, `underflow`, and `inexact` outputs. The comparison top exposes masks: invalid=8, overflow=4, underflow=2, inexact=1. It retains `multiply_flags`, `fused_flags`, and `addition_flags`; `nonfused_flags` unions multiplication and subsequent addition flags. FMA never imports intermediate multiplier flags or range limits. There is no clock/reset/valid protocol, pipeline latency, sticky state, trapping, alternate rounding, or synthesis-specific primitive.

## Fixed widths and exactness

A finite input has integer significand M in 0…2047 and effective biased exponent E in 1…30 (raw exponent zero uses E=1). Its exact value is (−1)^sign × M × 2^(E−25). Subnormal significands have no implicit bit.

| Quantity                              | Width and bound                                                                |
| ------------------------------------- | ------------------------------------------------------------------------------ |
| Unpacked significands                 | 11 bits each                                                                   |
| Raw product                           | 22 bits: 2047² < 2²²                                                           |
| Product exponent                      | Signed 8 bits: Ea+Eb−50 ∈ [−48,10]                                             |
| Product on global 2⁻⁴⁸ grid           | 80 bits: raw product shifted by Ea+Eb−2 ∈ [0,58]; strictly < 2⁸⁰               |
| C on same grid                        | 64 bits: 11-bit Mc shifted by Ec+23 ∈ [24,53]; strictly < 2⁶⁴                  |
| Signed-magnitude accumulation         | 81 bits: sum < 2⁸⁰ + 2⁶⁴ < 2⁸¹; subtraction cannot exceed the larger magnitude |
| Leading-bit index                     | Bounded priority encoder over 81 bits                                          |
| Discard count and normalization shift | 7 bits; no negative or unbounded shifts                                        |
| Retained significand                  | 12 bits including rounding carry                                               |
| Rounded biased exponent               | 6 bits, retaining out-of-range product exponents before overflow packing       |

Every finite binary16 product is an integer multiple of 2⁻⁴⁸. Thus even minimum-subnormal × minimum-subnormal is retained, and maximum-finite products need no truncation before cancellation. FMA can cancel an out-of-FP16-range product back to a finite answer. Sign XOR applies to the product; accumulation compares full magnitudes and resolves exact opposite-sign cancellation to +0 under nearest-even. Same-sign negative zero terms retain −0.

For leading index L ≥34, the initial biased exponent is L−33 and discard count L−10. Smaller magnitudes use a discard count of 24, rounding to the binary16 subnormal quantum 2⁻²⁴. Guard, round, and sticky come from the exact magnitude. Increment iff guard AND (round OR sticky OR retained-odd). Carry can cross binade or subnormal/normal boundaries. Debug normalization shift is 80−L (zero uses 0); it reports position, not lost bits. Raw `productExponent` scales the 22-bit product, not an already normalized significand.

After-rounding tininess means precision-only eleven-bit rounding with **unbounded exponent**, not merely testing the packed exponent. Its positive threshold on this grid is `4095 << 22` = (2⁻¹⁴−2⁻²⁶) / 2⁻⁴⁸. Magnitudes strictly below that threshold are tiny; equality rounds into the normal range. Underflow requires tininess **and** inexactness. A packed `0x0400` can therefore raise underflow. Overflow sets inexact and returns signed infinity. Exact subnormals do not raise underflow.

## Special values

The RTL matches FP Play's existing policy: first signaling NaN, otherwise first quiet NaN in A/B/C order (A/B for multiply), quiet while preserving sign and low-nine-bit payload. Zero × infinity raises invalid **even with a quiet NaN C**. A selected NaN retains its payload; otherwise invalid operations use `0x7e00`. An actual infinite product plus opposing infinite C is invalid. A large finite exact product is not treated as infinity before FMA addition. Special finite-stage signals are bypass defaults, never evidence of a computed finite product/rounding.

Berkeley SoftFloat can choose different NaN signs/payloads. Direct comparisons require quiet NaN classification and all four flags strictly, recording metadata differences separately. The RTL matches FP Play metadata strictly. This does not claim exhaustive IEEE 754 conformance.

## Reproduce verification and synthesis

Install Verilator and a Yosys release supporting the used SystemVerilog subset separately. No HDL tools or SoftFloat binaries enter npm/browser dependencies.

```sh
make -C rtl test                         # Preserve original add/sub corpus
make -C rtl -f mul-fma.mk test            # Strict lint, build, complete new corpus
make -C rtl -f mul-fma.mk waveform        # 23 bounded directed examples
make -C rtl -f mul-fma.mk synth           # Generic synthesis of three tops
node rtl/scripts/softfloat-mul-fma.mjs /path/to/softfloat-driver
# Refresh the small source-hashed browser artifact only after a full RTL run:
node rtl/scripts/verify-mul-fma.mjs --export rtl/vectors/mul-fma-simulation.json
```

Build the external SoftFloat driver using [the existing instructions](../src/lib/arithmetic/MULTIPLY_FMA.md#verification-and-reproducible-counterexamples). Local reference: upstream `a0c6494cdc11865811dec815d5c0049fba9d82a8`, 8086-SSE specialization, nearest-even, after-rounding tininess. The driver protocol is the existing four-field hexadecimal A/B/operation/C input; returned flags use invalid=16. No source/binary is vendored.

The C++ harness takes `vectors observations [wave.vcd]`. Each input row has ten hexadecimal fields: A B C multiplyResult multiplyFlags fusedResult fusedFlags nonfusedResult nonfusedFlags additionFlags. Observations include these actual outputs followed by nineteen multiplier and nineteen FMA signals, in `keys` order in `verify-mul-fma.mjs`. Wide integers are hex strings; JavaScript comparisons use BigInt throughout. Software alone supplies expected outputs. The independent lattice oracle additionally validates every multiplication and fused result/flag without sharing the production rounding algorithm.

Generated files stay under ignored `rtl/build/mul-fma/`. Full and directed runs use distinct prefixes; failures save up to eight reproducible triples and a bounded failing VCD. Simulator errors and malformed/empty/count/protocol mismatches fail. Directed constants include independently derived ties, tininess thresholds, cancellation, and exceptional cases. Complete outputs are not bundled. Artifact provenance hashes bind numerical references, RTL, harness, testbench, and vectors to actual observations; ordinary tests reject stale hashes.

## Actual validation record

Local tools: **Verilator 5.052**, **Yosys 0.70**, Node **22.9.0**. Corpus seed `0x4d554c46`:

| Family                                                 |     Triples |
| ------------------------------------------------------ | ----------: |
| Directed boundary/special cases                        |          23 |
| All 65,536 encodings: A×1+0, A×1+sign-flipped-A, A×0+A |     196,608 |
| Seventeen-value special/boundary cross product         |       4,913 |
| Deterministic full-width random triples                |     100,000 |
| **Total per operation**                                | **301,544** |

Every triple checks multiplication, true FMA, and nonfused results and all flags, including separate addition flags. **568,836 finite multiplication/FMA paths** additionally pass exact-product, alignment, result sign, normalization shift, discard count, GRS, parity increment, rounded exponent, and tininess comparisons. **Zero mismatches** against TypeScript and the independent multiplication/FMA lattice oracle. This is a corpus of triples, not a claim of unique multiplication pairs or exhaustive pair/triple coverage.

Direct SoftFloat: **301,544 comparisons per operation**, **904,632 result comparisons total**, zero finite-result, quiet-classification, or flag mismatches. NaN results: multiplication 14,532; FMA 18,296; nonfused 18,300. Implementation-specific NaN sign/payload differences: **231 / 429 / 1,550**, respectively. For nonfused, SoftFloat's own rounded product feeds its adder and flags are unioned; FP Play's payload can differ at either stage.

Yosys successfully reads the sources, synthesizes multiplier, FMA, and comparison tops with `synth -noabc`, then runs `check -assert` and asserts no latch/register cells. No unsupported constructs, inferred latches, or combinational-loop warnings were observed. This is generic logic synthesis, **not physical area, timing, power, placement, or device validation**. Distribution versions used in CI may differ; local 0.70 is the version actually tested.

## Recorded inspection and limitations

Arithmetic Explorer → **Inspect multiplier and true FMA RTL** displays the 23 actual recorded examples, live software agreement, raw/decoded fused and nonfused outputs, flags, significands, product exponent/product, fixed-grid alignment/accumulation, normalization, GRS, increment, and tininess. Selection is independent of the custom operand form. A browser never runs arbitrary HDL. The residual example yields fused `0x0010` versus nonfused `0x0000`; the intermediate-overflow example yields fused `0x7bff` versus nonfused infinity. This illustrates how a fused accumulator can retain small contributions in neural-network workloads; it does not model any particular accelerator.

Directed VCD generation was verified, without claiming GUI waveform viewing. Timestamps are settled observation indices, not hardware delay. No browser automation is available; DOM/keyboard tests do not establish rendered layout/theme/touch/screen-reader correctness. Formal equivalence, exhaustive operand pairs/triples, X/Z four-state behavior, silicon, and physical implementation remain unverified. HLS and transformer RTL are outside this milestone.
