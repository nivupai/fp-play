# Multi-precision arithmetic and accumulation contract

Milestone 8 adds fixed FP16, BF16, and FP32 models, a format comparison, and a bounded dot-product laboratory. Existing FP16 APIs and their computation-stage traces remain unchanged. These are public-format educational models, not a model of any particular accelerator or a full IEEE 754 conformance certification.

Milestone 9 now builds a separate [Attention Precision Lab](ATTENTION_PRECISION.md) on these APIs. The scope limits below describe the Milestone 8 dot/format workspaces; the attention contract documents the new supported graph.

## Formats and representation

| Format | Sign / exponent / stored fraction | Bias | Normal precision | Normal exponent range | Smallest positive subnormal | Maximum finite      |
| ------ | --------------------------------- | ---- | ---------------- | --------------------- | --------------------------- | ------------------- |
| FP16   | 1 / 5 / 10                        | 15   | 11 bits          | −14…15                | 2^-24                       | 65504               |
| BF16   | 1 / 8 / 7                         | 127  | 8 bits           | −126…127              | 2^-133                      | (2 − 2^-7) × 2^127  |
| FP32   | 1 / 8 / 23                        | 127  | 24 bits          | −126…127              | 2^-149                      | (2 − 2^-23) × 2^127 |

Wider exponents extend range; more significand bits reduce spacing at a fixed normal magnitude. BF16 has FP32's exponent range but fewer significant bits than FP16. Within a normal binade [2^E, 2^(E+1)), ULP is 2^(E−fractionBits). Subnormal/zero spacing is the fixed minimum quantum. Relative precision deteriorates toward zero in the subnormal region.

`src/lib/numerics/formats.ts` exports frozen `FORMATS` descriptors and pure functions taking a `FormatName` (`'FP16' | 'BF16' | 'FP32'`). User-defined formats are not accepted. Raw patterns are **unsigned Numbers**, including uint32 FP32 patterns; invalid integers are rejected without masking. FP32 is never exposed as a signed JavaScript bitwise integer.

- `encodeFloat(format, number)` extracts the exact binary64 dyadic with DataView and rounds directly once using BigInt. In particular, BF16 does **not** pass through FP32. For example, `1 + 2^-8 + 2^-30` rounds to BF16 `0x3f81`; rounding through FP32 incorrectly gives `0x3f80`.
- `decodeFloat(format, bits)` returns raw fields, sign, classification, unbiased exponent (null for zero/nonfinite; effective minimum exponent for subnormals), implicit-bit presence, exact finite Number, and authoritative finite `{ coefficient: bigint, exponent: number }` such that value = coefficient × 2^exponent. Zero's sign is in `sign`/`value`, since BigInt cannot represent negative zero.
- `floatBinary`/`floatHex` retain every bit, including raw NaN metadata. Decoding exposes `isQuietNaN` and `nanPayload` (fraction excluding its quiet bit). Number NaNs cannot preserve original raw metadata: encoding canonicalizes to positive quiet `0x7e00`, `0x7fc0`, or `0x7fc00000`.
- `nextUp`/`nextDown` accept and return raw encodings ordered **numerically**, skip the other zero, saturate outward at infinity, and preserve NaN patterns. Both zeros step up to the minimum positive subnormal and down to its negative. `floatULP` is sign-independent binade spacing, not necessarily the distance to `nextUp` at negative exponent boundaries. It returns minimum quantum at zero, Infinity at infinity, and NaN at NaN.
- `numberToDyadic`, `addDyadic`, `multiplyDyadic`, `subtractDyadic`, and `roundDyadic` support exact intermediate computations. Dyadic exponents are integer powers of two. The public barrel exports conversion/rounding and main model types; internal arithmetic helpers can be imported from their module.

```ts
import {
  encodeFloat,
  decodeFloat,
  addFloat,
  dotProduct,
} from './src/lib/numerics'

encodeFloat('BF16', 1 + 2 ** -8 + 2 ** -30) // 0x3f81
const x = decodeFloat('FP32', 0x80000000) // value is -0
Object.is(x.value, -0) // true
addFloat('BF16', 0x3f80, 0x3b80) // 1 + 2^-8 ties to 1; inexact
const dot = dotProduct([1.0009765625, -1.001953125], [1.0009765625, 1], 'C')
// FP32 2^-20; mode B instead yields zero.
```

## Exact arithmetic, rounding, and flags

`formatArithmetic.ts` provides `addFloat`, `multiplyFloat` and `calculateFloat(output, operation, a, b, c?)`. Each operand carries `{ format, bits }`. Mixed-format FMA supports input products feeding FP32 accumulation without an intermediate product rounding. Addition aligns full signed BigInts; multiplication multiplies integer significands; FMA aligns the **unrounded, unbounded-range** product with C. The result rounds once to output precision with round-to-nearest, ties-to-even. The original FP16 rounder remains intact for backward compatibility; the new rounder generalizes the same mathematical policy without changing the old trace contract.

A rounded result includes `exact` (null for special short circuits), `halfway`, `increment`, output bits/format, and per-operation `invalid`, `overflow`, `underflow`, `inexact`. Exact intermediates carry no Number projection. Stored fraction/subnormal rounding uses a single exact remainder comparison; exponent carries and overflow are applied afterward. Overflow implies inexact. Exact tiny results do not raise underflow.

Tininess is detected **after precision rounding with unbounded exponent range**, rather than merely checking a stored subnormal exponent. With normal precision P = fractionBits + 1 and minimum normal exponent Emin, a nonzero magnitude is tiny iff it is strictly below `2^Emin − 2^(Emin−P−1)` under nearest-even. Underflow requires this condition and final inexactness. A rounded normal can consequently raise underflow. This convention matches the [Berkeley SoftFloat explanation](https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat-FAQ.html).

NaN selection: first signaling, otherwise first quiet, in A/B/C order. Quiet the chosen NaN, retain its sign, and align the high payload bits when changing format (discard unavailable low bits on narrowing). Any signaling input raises invalid. Zero × infinity raises invalid even with quiet NaN C; input NaNs still propagate. Opposing genuine infinities in a sum give canonical quiet NaN and invalid. Finite FMA products do not independently overflow/underflow or raise flags. Multiplication zero uses XOR sign; exact cancellation gives +0, except summing two negative zero terms gives −0. Negative inexact underflow retains the negative sign.

## Dot-product configurations

`dotProduct(a, b, mode, order)` accepts equal-length binary64 Number arrays of **1–64 terms**. Quantize each operand once to the named input format, preserving signed zero and canonicalizing Number NaNs. Conversion flags are not accumulated; applied arithmetic flags are unioned, including intermediate exceptions. There is no implicit flush-to-zero or saturation.

| Mode | Inputs | Product boundary | Accumulator | Sequential step                               |
| ---- | ------ | ---------------- | ----------- | --------------------------------------------- |
| A    | FP16   | Round FP16       | FP16        | rounded multiply, then rounded add            |
| B    | FP16   | Round FP16       | FP32        | rounded multiply, exact widening, rounded add |
| C    | FP16   | Exact inside FMA | FP32        | one FP32 FMA per term                         |
| D    | BF16   | Exact inside FMA | FP32        | one FP32 FMA per term                         |
| E    | FP32   | Round FP32       | FP32        | rounded multiply, then rounded add            |

Sequential accumulation starts at **+0** and processes terms in vector order. B and C are deliberately different: widening an already rounded FP16 product cannot recover its lost bits or avoid its overflow.

**Pairwise reduction:** recursively split contiguous terms at `floor(length/2)`. Evaluate left subtree before right subtree. A/B/E leaves execute their configured rounded multiplication; C/D leaves execute `FMA(a,b,+0)` rounded to FP32. Every internal node executes an accumulator-format rounded addition. A one-term B tree widens its FP16 leaf exactly to FP32, preserving its sign and adding no rounding/flags. There is no extra +0 root addition. These are explicit leaf rounding boundaries, **not** a sequential FMA chain or one globally fused dot product. BF16 product range can exceed FP32 even though its significand is short. Reordering can change results, NaN selection, flags, and signed zeros.

For sequential arithmetic, +0 seeding can turn a lone negative-zero product into +0. Pairwise A/B/E leaves retain multiplication's zero sign; C/D use the FMA zero-sum rules. Infinity and NaN follow each executed operation; an overflowing intermediate remains infinity for later steps, potentially producing invalid on cancellation. No exact finite reference is assigned to a dot containing nonfinite inputs, even when a partial expression appears to cancel.

Each `DotStep` contains actual typed operands, operation, represented term indices, result, exact intermediate, rounding decision, per-operation flags, exact quantized-input subsum, and exact signed error. Sequential traces interleave multiplication and accumulation; pairwise traces are postorder tree evaluations. This is a deterministic computation log, not hardware timing.

## References, errors, and interface

The exact **original binary64 dot** sums the exact products of the parsed Numbers using BigInts. It is not Number multiplication/summation, nor exact decimal-text semantics. It can retain residuals beyond binary64 precision and products beyond binary64 range. The exact **quantized-input dot** multiplies/sums the quantized operands without product or sum rounding. Their difference measures input quantization; output minus each reference distinguishes total error from arithmetic-only error. `originalError`/`quantizedError` are exact signed dyadics, null when either reference or output is nonfinite. Exact zero references have no negative-zero BigInt sign; arithmetic outputs retain theirs.

Format Comparison (`#formats`) displays original and quantized values, labeled S/E/F bit groups, raw fields, exact decimal, hex, class, NaN metadata, range, precision, ULP, numerical neighbors, and errors. Absolute finite conversion error is exact dyadic subtraction; relative error is explicitly a binary64 estimate and undefined at zero/nonfinite values. Short headline values are round-trip binary64 decimal displays; the exact terminating decimals appear separately.

Dot Product Precision Lab (`#dot-product`) accepts comma/whitespace vectors, compares all five modes, selects sequential/pairwise order, and steps through the actual computation with native buttons, a range, and a select alternative. Invalid vector input retains the applied calculation. Twelve independently verified presets cover representation/range, ties, gradual precision, cancellation, small increments, reduction order, product rounding versus fusion, widening, and irreversible input loss.

The error SVG plots log₂ absolute error **against the exact quantized-input subsum for that step's terms**. Product steps represent one term; accumulation steps represent a prefix/subtree. The zero row is separate, and undefined errors have no point. A pairwise evaluation is not a single growing prefix; read the term indices. Coordinates/logarithms and tooltip Number summaries are approximate presentation only; full exact signed values remain available in step detail. Keyboard/text step controls provide the chart's selection alternative. Tables and long exact values scroll independently on narrow screens; themes reuse the validated application tokens.

## Verification and reproduction

All **344 earlier tests remain unchanged**. Milestone 8 adds independent numerical and actual-engine DOM integration tests. The mathematical oracle in `src/tests/reference/formatReference.ts` uses a fixed integer lattice of **2^-2148**, independent raw-field decoding, and binary search of positive encoding distances. It imports no production functions and does not reuse the production normalization/remainder rounder. This scale covers every product of finite binary64 inputs, including minimum subnormal products.

The suite checks:

- All **65,536 BF16 patterns**: fields, class, quiet/payload metadata, independently decoded finite values, signed zeros, and re-encoding/canonicalization.
- Every **32,640 positive finite BF16 rounding gap**, including the virtual overflow neighbor: exact midpoint and immediately neighboring binary64 values, both signs (**195,840 conversions**). Directed BF16 double-rounding counterexamples are separate.
- **20,000 binary64→FP32 native DataView comparisons**, another **20,000 random raw FP32 decodes/round trips**, and **2,540 signed exponent-boundary inputs** checked against both native conversion and the integer oracle.
- **24,000** deterministic finite BF16/FP32 add/multiply results and all flags against the independent distance oracle. Directed specials, signed zeros, precision-only tininess, exponent carry, and overflow supplement the corpus.
- **2,000** independent dot reductions across all five modes and both orders, including every executed intermediate's bits and flags. Random dot inputs are bounded to finite output ranges; directed tests separately exercise arithmetic/input overflow, underflow, nonfinites, signed-zero boundaries, and cancellation. The 64-term maximum is exercised in every mode/order. All twelve preset outputs also have independent references.
- Exhaustive compatibility of generic FP16 conversion/decoding/adjacency/ULP with the historical API, in addition to all unchanged historical numerical tests.
- DOM tests of original/quantized displays, validation, presets, signed zero, specials, subnormal transitions, reduction/precision selection, exact references, and keyboard-native entry/step alternatives using the real numerical engine. jsdom cannot execute native range-arrow behavior; its change handler and the native select alternative are tested explicitly.

Optional SoftFloat verification uses the previously documented Berkeley upstream revision **a0c6494cdc11865811dec815d5c0049fba9d82a8**, 8086-SSE specialization, nearest-even, after-rounding tininess. No upstream source or binaries are bundled. Extend the existing [build instructions](../src/lib/arithmetic/MULTIPLY_FMA.md#verification-and-reproducible-counterexamples) with:

```sh
node scripts/validate-fp32-softfloat.mjs /path/to/softfloat-driver
node scripts/validate-softfloat.mjs /path/to/softfloat-driver
```

The updated C driver preserves operations 0–3 for FP16 and adds 4/5/6 for FP32 add/multiply/FMA. It must be rebuilt before using the new script. The FP32 corpus includes directed triples, every normal exponent boundary, and 50,000 random triples (seed `0x8f32bf16`). This milestone's run compared **169,011 FP32 operations**: 54,171 add, 54,171 multiply, 60,669 FMA, **zero mismatches**. All four flags and non-NaN bits agree. All **5,275 NaN outputs** agree in quiet classification/flags; **369** have differing implementation-specific payload/sign selections, deliberately excluded from strict bit equality and covered by directed FP Play policy tests. BF16 has no native SoftFloat oracle here; its exhaustive/differential checks use the independent mathematical reference.

These finite corpora do not exhaust all FP32 operands, arithmetic pairs/triples, or dot products. There is no conformance certification. Node 22.9.0 and 24 run the full tests; see the current verification record for totals and runtime results.

## Scope and limitations

Only the three fixed formats and nearest-even are supported; there are no alternate rounding modes, division, global sticky processor state, traps, conversion flag reporting in the UI, arbitrary format definitions, transformer attention, or RTL. Dot vectors are deliberately bounded. Original decimal text may already round or underflow during binary64 parsing. Exact full decimal expansions can be long; they remain scrollable and are never silently truncated numerically. Pairwise fused-mode leaf boundaries differ from sequential FMA. The existing FP16 FMA workspace still means FP16 output, distinct from this lab's FP32 accumulation.

No browser automation is available in the Milestone 8 environment. There are no new screenshots or rendered-layout, touch, focus-appearance, screen-reader, or full accessibility-conformance claims. The responsive CSS, inherited theme contrast tokens, semantic controls, and DOM checks need real-browser review before publication. Nothing in this milestone pushes, merges, publishes, or deploys.
