# Binary16 numerical library

Import the public API and types from `src/lib/numerics/index.ts`. This pure TypeScript module has no React, DOM, external library, or native FP16 dependency. It requires BigInt and DataView binary64 support (ES2022 target).

## Representation

IEEE 754 binary16 has one sign bit `s`, five exponent bits `E`, ten fraction bits `F`, and exponent bias 15. All finite binary16 values are exactly representable as JavaScript binary64 Numbers.

| Raw exponent | Fraction | Classification | Decoded value                              |
| ------------ | -------- | -------------- | ------------------------------------------ |
| 0            | 0        | zero           | Signed zero                                |
| 0            | nonzero  | subnormal      | `(-1)**s * F * 2**-24`                     |
| 1–30         | any      | normal         | `(-1)**s * (1 + F/1024) * 2**(E-15)`       |
| 31           | 0        | infinity       | Signed infinity                            |
| 31           | nonzero  | NaN            | Number NaN; metadata remains in raw fields |

The minimum positive subnormal is `2**-24`, minimum positive normal is `2**-14`, and maximum finite value is 65504.

## API contract

All `bits` arguments must be integer Numbers from 0 through 65535. Invalid patterns throw `RangeError`; they are never coerced or masked to sixteen bits. Raw `-0` is accepted as pattern zero. Numeric-value APIs are typed to accept JavaScript Numbers.

| Function                            | Input → output           | Semantics                                                                                               |
| ----------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------- |
| `floatToFP16Bits(value)`            | Number → raw bits        | Direct binary64-to-binary16 round-to-nearest, ties-to-even; preserves signed zero; NaN becomes `0x7e00` |
| `fp16BitsToFloat(bits)`             | Raw bits → Number        | Exact finite value and signed zero, signed infinity, or Number NaN                                      |
| `decodeFP16(bits)`                  | Raw bits → `DecodedFP16` | Structured value, fields, classification, and raw NaN metadata                                          |
| `formatFP16Bits(bits)`              | Raw bits → string        | Sixteen binary digits, sign first, no spaces                                                            |
| `fp16BitsToHex(bits)`               | Raw bits → string        | `0x` plus four lowercase hexadecimal digits                                                             |
| `nextUpFP16(bits)`                  | Raw bits → raw bits      | Next strictly greater value; see special cases below                                                    |
| `nextDownFP16(bits)`                | Raw bits → raw bits      | Next strictly smaller value; see special cases below                                                    |
| `ulpFP16(bits)`                     | Raw bits → Number        | Sign-independent binade spacing in numeric units                                                        |
| `isExactlyRepresentableFP16(value)` | Number → boolean         | True iff conversion preserves the input under `Object.is`; NaN is explicitly false                      |

`DecodedFP16` includes `bits`, `sign` (0 or 1), `exponent` (raw field), `fraction` (all ten stored bits), `unbiasedExponent`, `classification`, `value`, and `hasImplicitLeadingBit`. The unbiased exponent is `E - 15` for normals, -14 for subnormals, and null for zeros, infinities, and NaNs. The implicit leading bit is present only for normals. Positive and negative zero share classification `zero`; inspect `sign` or use `Object.is(value, -0)` to distinguish them.

For NaNs, `isQuietNaN` reads fraction bit 9, and `nanPayload` contains the low nine fraction bits, excluding the quiet bit. Both fields are null for non-NaNs. `fraction` preserves the entire raw NaN fraction, and `bits` and `sign` preserve the original pattern and sign. JavaScript's Number abstraction cannot preserve the original FP16 NaN payload or sign through conversion; do not use a Number NaN as raw-pattern storage. Every Number NaN re-encodes as the positive quiet NaN `0x7e00`.

### Adjacent values

- Both signed zeros step upward to `0x0001` and downward to `0x8001` (minimum subnormals).
- `nextUpFP16(0x8001)` returns negative zero `0x8000`; `nextDownFP16(0x0001)` returns positive zero `0x0000`.
- Upward from maximum positive finite reaches positive infinity; downward from minimum negative finite reaches negative infinity.
- Moving inward from either infinity returns its nearest finite neighbor. Moving outward saturates at the same infinity.
- NaNs return the input raw pattern unchanged, preserving sign, quiet/signaling bit, and payload. These inspection helpers do not emulate IEEE exception flags or signaling-NaN exceptions.

### ULP definition

For finite nonzero values with magnitude in binade `[2**e, 2**(e+1))`, ULP is `2**max(e - 10, -24)`. Zero and subnormals have ULP `2**-24`. Sign is ignored. Infinities return positive Infinity, and NaNs return NaN.

At a power of two, this means spacing on the larger-magnitude side: `ulpFP16(0x3c00)` and `ulpFP16(0xbc00)` are both `2**-10`. The gap immediately below +1, or immediately above -1, is `2**-11`. At maximum finite magnitude, ULP remains 32 even though the next outward representable value is infinity. ULP is therefore not defined as `nextUp(x) - x`.

Signed zeros and infinities count as exactly representable. NaN returns false by policy because a Number cannot specify the original binary16 NaN pattern.

## Encoding algorithm and scope

DataView extracts the input's binary64 sign, exponent, and fraction. A BigInt significand and a power-of-two scale describe the Number exactly. For normals, retain eleven significant bits and compare all discarded bits with the exact halfway remainder. Increment above halfway, or at halfway if the retained integer is odd. A significand carry advances the exponent; exceeding exponent 15 yields infinity. Subnormals round directly to integer multiples of `2**-24`, including signed-zero underflow and carry into the smallest normal.

There is exactly one rounding to binary16, with no binary32 conversion, logarithmic exponent estimation, or approximate epsilon comparison. Magnitudes at or above 65520 round to infinity. A magnitude of exactly `2**-25` rounds to signed zero, since the zero significand is even.

The input is the Number already supplied by JavaScript. Parsing decimal strings to exact rationals is outside this milestone; a decimal literal may already have been rounded to binary64 before calling the API. These conversion/introspection APIs do not execute arithmetic or expose exception flags. The separate [binary16 add/sub model](../arithmetic/README.md) and [multiplication/FMA extension](../arithmetic/MULTIPLY_FMA.md) execute arithmetic with explicit NaN policies, exact BigInt intermediates, and per-operation indicators; alternative rounding modes remain unsupported. Raw-bit NaN inspection is lossless; Number conversion canonicalizes NaNs as described above.

## Validation

`src/tests/reference/fp16Reference.ts` independently enumerates all positive finite values in exact integer units of `2**-24` and encodes by binary-searching their exact midpoint thresholds. It imports no production functions and does not extract binary64 fields. This provides an independent oracle rather than relying only on encoder/decoder round trips.

Tests cover every one of the 65,536 raw patterns, including independent decoded-value checks, exact finite round trips, and NaN canonicalization. Every gap between consecutive positive finite values is probed at its midpoint and the immediately adjacent binary64 values, on both signs. This covers ties, carries at every exponent boundary, gradual underflow, and the subnormal-to-normal transition. Additional tests cover overflow, all binary64 exponent fields, fixed-seed full-width binary64 samples, exhaustive adjacency and ULP checks, signed zeros, and raw NaN metadata.

The same oracle test also compares against `Math.f16round` and DataView `setFloat16` when available. Node 22.9 has neither, so the independent oracle always runs and no numerical cases are skipped. On newer CI runtimes with these APIs, native comparisons run automatically; native NaN encodings are checked by classification because their payloads are implementation-defined.

Public references:

- [ECMAScript Math.f16round](https://tc39.es/ecma262/multipage/numbers-and-dates.html#sec-math.f16round), which specifies direct binary16 roundTiesToEven conversion and discusses double rounding.
- [NVIDIA CUDA Math API: half representation](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html) and [public format ranges](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/mathematical-functions.html).

The implementation and test oracle are original code based on public format semantics; they contain no proprietary hardware designs or internal Broadcom information.

## Multi-precision extension

Milestone 8 retains every FP16 API above and adds fixed format descriptors, direct BF16/FP32 conversion, exact arithmetic, and a typed mixed-precision dot-product engine. See [the complete multi-precision contract](../../../docs/MIXED_PRECISION.md) for API semantics, rounding boundaries, NaN/tininess policies, independent references, and reproduction instructions.

## Attention extension

Milestone 9 composes existing dot products into a bounded attention graph with explicit scaling/softmax/probability/value/output boundaries and an attention-specific exact-rational quotient helper. Existing APIs remain unchanged. See [the attention contract](../../../docs/ATTENTION_PRECISION.md) for policies, host transcendental limitations, stage flags, independent references, and reproduction.
