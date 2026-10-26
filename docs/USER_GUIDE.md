# FP Play usage guide

[Back to the project overview](../README.md). Start the app with `npm run dev` and open `http://localhost:5173/fp-play/`.

The landing page links to Bit Explorer, Precision Microscope, Arithmetic Pipeline, and FMA Comparison. The FMA entry opens a verified residual-preservation example. These four original experiences share the validated binary16 numerical library. The additional multi-precision entry links open Format Comparison and Dot Product Precision Lab; see their controls below.

## Multiplication and fused multiply-add

Open **Arithmetic**, select **A × B** or **FMA(A, B, C) / nonfused comparison**, then enter decimal/scientific or raw hex operands and press **Trace operation**. C appears for FMA. Decimal text is parsed into binary64 and independently quantized to FP16 before arithmetic; raw hex preserves signaling NaN metadata. Invalid input leaves the applied calculation intact.

The FMA view shows all three applied operands, the exact product and mathematical result, **fused output**, **rounded intermediate product**, **nonfused output**, binary/hex encodings, flags, and the numerical difference when both outputs are finite. Use **Inspect computation path** to follow the fused operation, nonfused multiplication, or nonfused addition with the same keyboard-accessible stage viewer. The paths show their actual computation stages, not chip cycles.

```ts
import {
  multiplyFP16,
  fmaFP16,
  nonfusedMultiplyAddFP16,
} from './src/lib/numerics'

multiplyFP16(0x3e00, 0x4000).bits // 1.5 × 2 = 3 / 0x4200
fmaFP16(0x3c01, 0x3c01, 0xbc02).bits // exact residual 2^-20 / 0x0010
nonfusedMultiplyAddFP16(0x3c01, 0x3c01, 0xbc02).bits // rounded product loses residual: +0
fmaFP16(0x7bff, 0x4000, 0xfbff).bits // 65504 × 2 − 65504 = 65504 / 0x7bff
nonfusedMultiplyAddFP16(0x7bff, 0x4000, 0xfbff).bits // intermediate overflow: Infinity
```

Multiplication forms the exact up-to-22-bit significand product and adds operand scale exponents, then rounds once. **True FMA** aligns that unrounded product with C, sums exactly in BigInts, and rounds only the final result. **Nonfused** calculation rounds the FP16 product first and then performs a separately rounded FP16 addition; its flags are the union of both operations. Binary16 FMA is different from FP32 accumulation. Ten new explained presets cover sign/exponent arithmetic, exact and subnormal products, underflow, overflow, invalid zero × infinity, midpoint rounding, cancellation, lost residuals, and avoided intermediate overflow.

NaN selection extends the existing first-signaling/otherwise-first-quiet policy to A/B/C order. Zero × infinity raises invalid even with a quiet NaN addend; the selected NaN keeps its original sign and nine payload bits. An infinite product plus the opposite infinity is invalid; a finite FMA product has no intermediate FP16 range restriction. All returned indicators are per operation; input conversion flags and global sticky state are outside this model.

**Tininess after rounding** uses eleven-bit precision with unbounded exponent range, not just the final stored exponent. Underflow additionally requires inexactness. Near the normal boundary a stored `0x0400` can still raise underflow; this is explicitly tested against SoftFloat. Exact subnormals never raise underflow. The exact dyadic coefficient/exponent is authoritative: FMA's compatibility `exact.value` Number projection can round, while every displayed exact decimal is formatted directly from BigInts.

MAC operations are central to neural-network dot products. Retaining product bits through an addition can preserve small cancellation residuals and avoid transient overflow. This view teaches those single-operation effects without modeling an accelerator's throughput, accumulation format, reduction ordering, or proprietary hardware.

## Using the Arithmetic Explorer

Open **Arithmetic** in the header, or scroll below the microscope.

1. Choose **Decimal / special value** or **Raw hexadecimal** for each operand. Enter decimal/scientific text, `-0`, `Infinity`, `-Infinity`, or `NaN`; raw mode accepts exactly `0x` and four hex digits, such as `0xfc11`. Choose **A + B** or **A − B**, then **Trace operation** (or Enter in an input). Draft edits do not change applied results; invalid input retains the previous operation.
2. Inspect the two applied **quantized FP16 operands**, with original input text and parsed binary64 value separately identified. Decimal parsing and operand quantization happen **before** arithmetic. Raw hex retains NaN sign/payload/signaling metadata; decimal `NaN` is canonical quiet `0x7e00`.
3. Compare the **exact mathematical result of the FP16 operands** with the final FP16 decimal, binary, and hex result. All finite decimal readouts are exact terminating expansions, including an overflow's finite pre-rounding sum; signed zero remains visible.
4. Use **Previous stage / Next stage**, or select any stage directly. Focus the detail panel for **Left/Right** and **Home/End** shortcuts. Native stage buttons support Tab, Space, and Enter. Inspect the lossless alignment tail, signed integer sum, normalization shifts, guard/round/sticky bits, tie parity, carry, result encoding, and per-operation flags. Long binary rows scroll within their panels on small screens.
5. Try all ten explained presets: exact addition, large exponent difference, nearest-even tie, cancellation, subnormal arithmetic, transition to normal, overflow, infinity arithmetic, invalid infinities, and signaling NaN. Each preset displays its expected raw result.

### Addition/subtraction semantics

`addFP16(aBits, bBits)`, `subtractFP16(aBits, bBits)`, and `calculateFP16(aBits, bBits, 'add' | 'subtract')` accept validated raw uint16 patterns. They return result `bits`, `decoded`, an exact dyadic `{ coefficient: bigint, exponent, value }` for finite operands, a discriminated stage `trace`, and `flags`. `value` is an exact binary64 display projection; the core uses BigInts, not Number addition. Finite dyadics are aligned without discarding any tail, summed with signs, normalized, and rounded **once, nearest, ties to even**. The GRS increment rule is `guard && (round || sticky || retainedOdd)`.

```ts
import { addFP16, subtractFP16 } from './src/lib/numerics'

addFP16(0x3c00, 0x1000).bits // 0x3c00: 1 + 2^-11, tie to even
addFP16(0x3c01, 0x1000).bits // 0x3c02: odd tie increments
subtractFP16(0x3c01, 0x3c00).bits // 0x1400: cancellation gives 2^-10
addFP16(0x7bff, 0x4c00).flags // overflow + inexact: 65520 rounds to Infinity
addFP16(0xfc11, 0x3c00).bits // 0xfe11: quiet signaling NaN, preserve sign/payload
```

- **NaNs:** select the first signaling NaN in A/B order, otherwise the first quiet NaN. Set the quiet bit and preserve the selected operand's original sign and low nine payload bits, including for subtraction. Any signaling NaN raises invalid; quiet NaNs alone do not. Opposite effective infinities (including Infinity − Infinity) return canonical quiet NaN `0x7e00` with invalid. NaN classification/invalid behavior is distinct from this **implementation-specific payload policy**.
- **Zeros:** exact cancellation is +0 under nearest-even; two effective negative zero operands sum to −0. Subtraction first reverses B's arithmetic sign. Subnormal results are retained rather than flushed to zero.
- **Flags:** `invalid`, `overflow`, `underflow`, and `inexact` describe this operation only; there is no sticky global status. Overflow returns signed infinity and sets inexact, even when no significand bits were discarded. Underflow means **tiny after rounding AND inexact**. All finite FP16 operands are multiples of `2**-24`; any tiny sum/difference is consequently exactly representable, so add/sub cannot raise underflow. Input conversion flags are outside this arithmetic result. Division-by-zero is irrelevant to add/sub and is not exposed.
- **Trace:** finite operations have eight conceptual stages; specials have decode → special-value short circuit → encode. These are explanatory algorithm stages, **not clock cycles, latency, RTL, or a proprietary hardware pipeline**. Returned BigInt fields require explicit conversion before JSON serialization.

FP16's eleven-bit normal significand makes accelerator storage and arithmetic economical, but alignment can move a small contribution entirely below retained precision. Repeated rounded additions can lose contributions or amplify cancellation effects; accumulation format and operation ordering therefore matter. These add/sub traces demonstrate single operations; they do not model accelerator performance, reductions, or a specific device's handling of denormals.

See [the arithmetic model, trace contract, and reproducible validation](../src/lib/arithmetic/README.md). This limited educational model does **not claim full IEEE 754 conformance**: binary16 add/sub, multiplication, and FMA use nearest-even. Division, alternate rounding modes, hardware timing, and global exception state remain unsupported.

## Using the Precision Microscope

Open **Microscope** in the header, or scroll below the existing Explorer.

1. Enter a decimal/scientific binary64 target and press **Analyze target** or Enter. Input conventions match the Explorer, including `-0`, infinities, and NaN. The last applied text, parsed binary64 Number, quantized FP16 value, neighbors, rounding direction, absolute error, relative error, and selected binary/hex encoding are reported separately.
2. Use **Zoom in/out**, **Pan left/right**, **Fit target**, and **Overview**. Round markers are actual FP16 values; the cross marks the binary64 target and the diamond its quantized value. Shading shows the selected rounding cell; square endpoints and dashed boundaries distinguish it from discrete points. Tooltips explain exact point values and their ULP spacing.
3. Choose **Move target** or **Pan viewport** drag mode. The target slider moves between representable values; clicking a point or using **Inspect a visible FP16 point** selects an exact FP16 value. The expandable point table gives exact values, binary, hex, and ULP.
4. Keyboard users can focus the number line: **Left/Right** pan, **+/−** zoom, and **Up/Down** move the binary64 target by 1% of the current view. Native buttons, the slider, and point/region dropdowns provide text controls for the same exploration.
5. Try all seven explained experiments: why **1.0001 becomes 1**, spacing near **1** and **1000**, subnormal precision, gradual underflow, exponent boundaries, and maximum finite/overflow behavior.
6. Inspect the positive **ULP-versus-magnitude** chart. Both axes are logarithmic (base 2). Click a region or use **Inspect spacing region** to focus the local number line on that binade. Negative magnitudes mirror positive spacing.

### Mathematical definitions and examples

- A normal binade `[2**e, 2**(e+1))` has constant absolute spacing `2**(e-10)`. Normal exponent transitions double spacing; subnormals and the first normal binade both have spacing `2**-24`. There is **no spacing jump at the smallest normal**, `2**-14`.
- A nearest-even rounding cell is bounded by exact midpoints to adjacent FP16 values. Even retained significands own their halfway endpoints; odd ones do not. The cell of 1 is **[0.999755859375, 1.00048828125]**: 1.0001 rounds to 1, and the upper endpoint also rounds to 1 by ties-to-even. Cells at powers of two can be asymmetric.
- Signed zeros share a geometric point but have sign-dependent half-cells ending at ±`2**-25`. Zero itself is a sign seam, not a halfway tie. Maximum finite FP16 is 65504; the exact overflow tie is 65520, halfway to virtual 65536, which overflows after significand carry. Negative overflow is symmetric.
- Absolute error is `abs(FP16 - binary64 input)`. Relative error divides that by `abs(input)` and is undefined at zero. Errors for nonfinite original inputs are undefined; finite inputs that overflow FP16 have infinite error. Relative division uses binary64 arithmetic and may round.
- Constant absolute spacing is different from relative precision: spacing near 1000 is 0.5, versus `2**-10` immediately above 1. Subnormal spacing stays fixed as relative precision deteriorates toward zero. For input magnitudes at least `2**-14` with finite quantization, nearest-rounding relative error is bounded by `2**-11`.

### Microscope limits and validation

The plot domain is ±131072, with viewport widths from `2**-30` to 262144. Analysis still accepts arbitrary binary64 values; extreme or nonfinite targets are explicitly marked off-axis. The finite numerical lattice has **63,487 positions**, combining both zero encodings geometrically. Visible values are located by binary search, and the renderer caps points at **121**. Wide views explicitly disclose sampling: omitted points are still representable. Zoom in to resolve individual gaps. Tooltips report engine ULP, never a misleading gap between sampled markers.

Decimal parsing is binary64-first, not exact decimal-rational conversion. Binary64/error readouts use shortest round-trip strings; FP16 readouts and finite rounding boundaries use exact decimal expansions. Axis labels are rounded to six significant digits, and SVG coordinates are approximate screen projections. The target slider has 1,001 normalized positions; text input remains independent of slider resolution. Charts and tables scroll horizontally on narrow screens while controls/readouts reflow.

See the [microscope model, API, mathematics, and validation notes](../src/lib/microscope/README.md) for complete definitions and limitations.

## Using the FP16 Explorer

1. Start the app with the setup commands below. The initial pattern is **1.5 / 0x3e00**.
2. Toggle any of the sixteen bits by clicking its button, or use **Tab** to focus it and **Space/Enter** to toggle. Sign, exponent, and fraction have distinct colors, visible labels, bit positions, and accessible pressed states.
3. Enter a decimal or scientific-notation value and select **Convert to FP16** (or press Enter). The comparison shows the entered text, parsed binary64 Number, and the rounded FP16 value separately. Conversion uses nearest, ties-to-even. Incomplete or invalid submissions show an error and preserve the selected pattern.
4. Read the exact decoded decimal value, classification, raw fields, exponent bias, unbiased exponent, implicit leading bit, hexadecimal encoding, and class-specific mathematical explanation.
5. Select a reference preset, or step to a previous/next numerical neighbor. The precision panel explains binade ULP, signed zeros, infinities, and unordered NaNs. Neighbor buttons follow numerical order, not unsigned encoding order; both zeros skip the other zero encoding when stepping.
6. Choose **System**, **Light**, or **Dark** theme in the header. System follows the OS color preference; manual selections last for the current page session. Layouts adapt to mobile widths, and reduced-motion preferences disable transitions.

Accepted input examples: `0.1`, `-0.125`, `.5`, `1.5e0`, `-0`, `Infinity`, `+Infinity`, `-Infinity`, and `NaN`. Special tokens are case-insensitive and surrounding whitespace is ignored. Use ASCII `-` for negative input. Hexadecimal strings, digit separators, malformed numbers, empty input, and decimals outside the finite binary64 range are rejected. Explicit infinity tokens are accepted.

Presets use explicit validated bit patterns for +0, -0, 1.0, 1.5, the smallest/largest positive subnormals, the smallest positive normal, maximum finite positive, positive infinity, and quiet NaN with payload 1 (`0x7e01`). Editing bits, selecting a preset, or stepping to a neighbor clears the last conversion comparison and synchronizes the decimal field with the decoded value. Editing the decimal text alone does not change the selected bits; comparison results are labeled **Last converted input** until another conversion.

### Display and input limitations

- Finite FP16 output uses its exact terminating decimal expansion (up to 24 fractional places), with no display rounding; negative zero is shown as `-0`.
- Decimal text is first parsed into a JavaScript binary64 Number. Parsing may round, or underflow to signed zero, before FP16 conversion. FP Play does not parse decimal strings as exact rationals. The original text and parsed Number are shown to make this boundary visible.
- Raw NaN edits preserve sign, quiet/signaling bit, and payload. Submitting `NaN` canonicalizes to positive quiet NaN `0x7e00`; JavaScript Number cannot retain the original FP16 NaN metadata. The representation Explorer does not execute signaling NaNs or report conversion exception flags; the Arithmetic Explorer separately models arithmetic indicators.
- ULP uses the engine's sign-independent binade-spacing definition, not necessarily the gap to the next greater value. Infinity returns Infinity and NaN returns NaN; neither has a finite ULP. Outward stepping at infinity saturates. NaN neighbor controls are disabled because NaNs are unordered.
- Real-browser visual verification is unavailable in the current environment. Automated DOM tests cover interactions and semantics; manual review of layout, themes, focus rendering, and screen-reader announcements is still needed.

## IEEE 754 binary16 (FP16)

Binary16 uses **1 sign bit**, **5 exponent bits**, and **10 stored fraction bits**, with an **exponent bias of 15**. Normal values have an implicit leading 1, giving eleven bits of significand precision. Exponent zero encodes signed zero or subnormals; exponent 31 encodes infinities or NaNs.

For a normal value, `value = (-1)**sign * (1 + fraction/1024) * 2**(exponent-15)`. Subnormals use `value = (-1)**sign * fraction * 2**-24`. The smallest positive subnormal is `2**-24`, the smallest positive normal is `2**-14`, and the maximum finite value is **65504**.

```ts
import {
  decodeFP16,
  floatToFP16Bits,
  formatFP16Bits,
  fp16BitsToFloat,
  fp16BitsToHex,
  isExactlyRepresentableFP16,
  nextUpFP16,
  ulpFP16,
} from './src/lib/numerics'

const bits = floatToFP16Bits(1.5) // 0x3e00
fp16BitsToHex(bits) // '0x3e00'
formatFP16Bits(bits) // '0011111000000000'
decodeFP16(bits) // normal; sign 0, exponent 15, fraction 512, value 1.5
fp16BitsToFloat(nextUpFP16(bits)) // 1.5009765625
ulpFP16(bits) // 0.0009765625 (2**-10)
isExactlyRepresentableFP16(0.1) // false
Object.is(fp16BitsToFloat(0x8000), -0) // true
floatToFP16Bits(1 + 2 ** -11) // 0x3c00: exact halfway tie rounds to even
```

Encoding rounds directly from the input binary64 Number to binary16 using round-to-nearest, ties-to-even, including gradual underflow and overflow. Raw-bit APIs accept integers from 0 to 65535 and throw `RangeError` for invalid patterns. Adjacent-value APIs accept and return **raw bit patterns**. ULP means sign-independent binade spacing: `2**-24` at zero/subnormals, 32 at maximum finite magnitude, Infinity at infinities, and NaN at NaNs.

Raw NaN inspection preserves the sign, fraction, quiet/signaling bit, and payload. JavaScript Numbers cannot preserve the original FP16 NaN payload or sign through conversion; encoding any Number NaN produces positive quiet NaN `0x7e00`. Exact representability includes signed zeros and infinities but excludes NaN by policy. The conversion/introspection APIs do not expose exception flags or alternative rounding modes. Arithmetic is provided by the separate add/sub/multiply/FMA models described above. See the [full API contract, algorithms, validation, and limitations](../src/lib/numerics/README.md).

## Format Comparison and Dot Product Precision Lab

Follow the landing page's **Compare FP16, BF16, and FP32** and **Dot Product Precision Lab** links. In Format Comparison, enter a decimal/special value and press Enter or **Compare formats**. Three cards compare quantized values, complete S/E/F bits, exact decimals, ULP, range, and errors against the applied binary64 input.

The dot lab accepts 1–64 comma/whitespace values in each equal-length vector. Apply both vectors together; invalid input leaves the prior calculation intact. Select any of twelve explained experiments, a precision configuration A–E, and sequential or balanced pairwise order. The comparison table always shows all five modes for the applied vectors/order. Inspect exact original and quantized references separately.

Use **Previous step**, **Next step**, the native step range, or **Inspect error point** to read operands, exact intermediate, output encoding, tie/increment decisions, error, and flags. The log-error chart has a separate zero row; pairwise steps compare their subtree's terms. Expand **Quantized input encodings** for every converted operand. Tables and long numbers have their own keyboard-focusable scroll areas.

C/D sequential modes use FP32 FMA per term; pairwise modes instead use FMA-with-zero leaves and rounded tree additions. B first rounds each product to FP16, even though the accumulator is FP32. Conversion flags are not part of the arithmetic flag union. Signed-zero seeding and nonfinite-reference limitations are explained beside the results. See [the mathematical contract](MIXED_PRECISION.md) for exact definitions, examples, and verified results.

## Attention Precision Lab

Open **Attention Precision Lab** from the landing links (`#attention`). Enter Q/K/V with newline/semicolon rows and comma/space columns. Q supports up to 4×8, K up to 8×8, V up to 8×4; matching dimensions are required. **Apply attention matrices** updates all three together; errors preserve the last calculation. Decimal text is binary64-first, including explicit -0/Infinity/NaN tokens.

Select left/right precision policies and stable or deliberately unstable naive softmax. Choose **Attention stage view** to compare QK scores, scaled logits, probabilities, and output. Corresponding heatmaps share a scale and display signed values/hex rather than relying on tint. Left cells are native buttons: Tab then Space/Enter opens exact decimal, full bits, and classification. Any policy can be selected on the left to inspect all of its boundaries.

Read stable FP32 and separate binary64 baselines, absolute/relative error maps, and **Final output difference · left minus right**. Select a probability row for its stored denominator, weights, and displayed sum. Quantized probabilities are not renormalized. Use previous/next/select stage controls to inspect actual operands, rounding ties/increments, encodings, and flags. The twelve experiments show expected outcomes and verification methods.

Math.exp and reciprocal Math.sqrt are host binary64 computations followed by explicit quantization, not correctly rounded format-specific transcendental functions. A nonfinite result is retained with diagnostics; stable softmax cannot undo an overflowing input/score. Baselines in the UI are numerical estimates; the higher-precision Decimal reference is offline. See [the complete attention policy and reproduction guide](ATTENTION_PRECISION.md).

## Optional RTL inspection

In Arithmetic Explorer, expand **Inspect precomputed FP16 RTL simulations**. Select one of 28 recorded add/subtract examples; this selection is separate from the custom operands above. The panel recomputes the current software result and compares it against a real precomputed Verilator record, including all flags and NaN payload/sign bits. A disagreement is reported explicitly, including flag-only differences.

Try **Tie even stays** and **Tie odd increments**: both show G/R/S = 1/0/0, but only the odd retained significand increments. **Sticky tiny addend** shows 0/0/1 and inexact. Read the aligned 40-bit magnitudes, 41-bit sum/difference, normalization/discard coordinates, and final rounding decision. Native selection and keyboard-scrollable values provide text alternatives. Special paths explain that finite-stage zeros are bypass defaults.

The browser does not run Verilator, synthesize hardware, or simulate arbitrary operands. These are combinational observation stages; neither stage order nor VCD timestamps imply clock cycles or measured delay. See [the RTL guide](../rtl/README.md) for local tool installation, the full differential corpus, VCD generation, failure replay, and honest synthesis/browser limitations.
