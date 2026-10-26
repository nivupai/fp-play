# Binary16 arithmetic model

This page documents the original **raw binary16 addition and subtraction** APIs. See [multiplication, FMA, and nonfused comparison](MULTIPLY_FMA.md) for the Milestone 6 extension and current verification. The add/sub implementation rounds once with **round-to-nearest, ties-to-even**. It is independent of React and of proprietary hardware. It does not claim full IEEE 754 conformance.

## API

```ts
import { addFP16, subtractFP16, calculateFP16 } from '../numerics'

const tie = addFP16(0x3c00, 0x1000) // exact 1.00048828125 → 1 / 0x3c00
const cancellation = subtractFP16(0x3c01, 0x3c00) // 2^-10 / 0x1400
const overflow = calculateFP16(0x7bff, 0x4c00, 'add') // 65520 → Infinity
```

Both operands are integer uint16 patterns (0…65535); invalid patterns and unsupported operation names throw `RangeError`. `FP16ArithmeticResult` returns:

- `operation`, `aBits`, `bBits`: applied operation and original patterns.
- `bits`, `decoded`: final packed result and existing `DecodedFP16` fields.
- `roundingMode: 'nearest-even'`, `tininessDetection: 'after-rounding'`.
- `flags: { invalid, overflow, underflow, inexact }`: booleans for this operation, not sticky global state.
- `exact: { coefficient: bigint, exponent: number, value: number } | null`: mathematical signed integer times a power of two, before rounding. `null` for special operands; a finite overflow still has an exact finite sum. `value` preserves signed zero and is an exact Number projection for these add/sub APIs, not the computation method. FMA can require more than 53 bits; its coefficient/exponent, rather than Number projection, is authoritative.
- `trace: ArithmeticStage[]`: a discriminated union of actual computation records. BigInts need explicit serialization; plain `JSON.stringify(result)` is unsupported.

Result objects are fresh per call. Consumers should treat their fields as immutable; mutating a returned object does not change subsequent calculations.

## Exact model and rounding

The existing decoder supplies sign, exponent, and fraction. A finite operand is `(-1)^sign * M * 2^(E-10)`, with integer `M = 1024 + fraction` and `E = rawExponent - 15` for normals, or `M = fraction` and `E = -14` for subnormals/zeros.

Let `Emax` and `Emin` be the effective operand exponents, `f = Emax - Emin`, and `q = Emin - 10`. Store each aligned magnitude as `M << (E - Emin)`. Integers multiply `2^q` to give the original magnitudes exactly. Viewed at the common exponent, they are fixed-point integers with **f fractional significand bits**, making the conceptual right shifts visible without throwing away a tail. The alignment stage's binary point belongs to the integer significand; it is not a binary point in ordinary value units.

Apply effective signs (reverse B's arithmetic sign for subtraction) and sum the BigInts. Locate the magnitude's leading bit. Choose eleven significant bits for normals, clamping the rounding quantum to `2^-24` for subnormals. Normalize by shifting right to identify discarded bits or left after cancellation. Retain the complete exact remainder.

The first discarded bit is **guard**, the second **round**, and the OR of every remaining bit **sticky**. Missing bits are zero. Increment the retained integer exactly when `G && (R || S || retainedOdd)`. `G=1,R=0,S=0` means an exact halfway tie. An increment carrying to 2048 renormalizes to 1024 and advances the exponent. Apply the finite exponent range, then pack sign/exponent/fraction directly. The arithmetic engine never calls the Number encoder or performs Number operand addition.

All finite operands are integer multiples of `2^-24`. Even the largest sum has a global coordinate below `2^41`, so its exact mathematical result fits binary64's 53-bit precision. This proves the returned Number projection is exact and enables an **independent** test oracle using native Number addition/subtraction plus a separately enumerated lattice search. That oracle does not use the production arithmetic, decoder, or encoder.

### Special values and indicators

| Condition                          | Result / indicators                                                                                                   |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Signaling NaN in either operand    | Select first signaling NaN in A/B order; quiet it, retain its original sign and low nine payload bits; invalid raised |
| Only quiet NaNs                    | Select first in A/B order; retain raw sign/payload; flags clear                                                       |
| Opposite effective infinities      | Canonical quiet NaN `0x7e00`; invalid raised                                                                          |
| Other infinity operations          | Signed infinity; flags clear                                                                                          |
| Exact finite cancellation          | +0 under nearest-even                                                                                                 |
| Both effective operands −0         | −0                                                                                                                    |
| Finite overflow after rounding     | Signed infinity; overflow and inexact raised                                                                          |
| Finite nonzero discarded remainder | Inexact raised                                                                                                        |
| Tiny after rounding AND inexact    | Underflow raised by definition; unreachable for valid FP16 add/sub operands                                           |

NaN payload selection is a deterministic **FP Play policy**, distinct from required NaN/invalid behavior. A selected NaN keeps its original sign even when it is operand B of subtraction. The quiet bit is fraction bit 9 and the payload is the low nine bits. Undefined infinity combinations use a canonical result because there is no input NaN to propagate.

Subnormals are never flushed to zero. Tiny FP16 add/sub results are exact multiples of the subnormal quantum and need no rounding; therefore no valid input pair can raise underflow. Input conversion can underflow but is a separate operation whose flags are not exposed here. Overflow sets inexact even if the unbounded significand happened to need no discarded bits (for example, 65504 + 65504).

Multiplication and FMA are now provided by the [Milestone 6 extension](MULTIPLY_FMA.md). There are no division-by-zero indicators, global sticky state, traps, directed rounding modes, division or reductions in this original API. Milestone 10 adds a separate [FP16 RTL add/subtract unit and recorded inspection](../../../rtl/README.md), without changing this numerical model.

## Trace contract

Finite operations always return these eight records:

| `kind`       | Computation fields / purpose                                                                                      |
| ------------ | ----------------------------------------------------------------------------------------------------------------- |
| `decode`     | Original decoded patterns and B's effective sign                                                                  |
| `compare`    | Integer significands, effective signs/exponents, signed exponent difference                                       |
| `align`      | Common exponent, conceptual right shifts, all retained fractional bits, aligned BigInts, exact integer scale      |
| `arithmetic` | Signed aligned integers, exact signed sum, scale exponent                                                         |
| `normalize`  | Magnitude, leading bit index, exact leading exponent, working exponent, rounding quantum, shift, retained integer |
| `grs`        | Guard/round/sticky, count and string of all discarded bits, exact remainder                                       |
| `round`      | Tie/parity/increment decision, rounded integer, carry, final significand/exponent                                 |
| `encode`     | Packed pattern, decoded result, final flags                                                                       |

For zero, the leading-bit index and exact leading exponent are `null`; no leading 1 is invented. For exponent overflow, the working final exponent can be 16 before encoding infinity. Specials return only **decode → special → encode**, with the reason, selected NaN operand if any, and signaling presence. They do not fabricate ordinary alignment or rounding records.

`describeStage` formats these fields for the UI and adds explanatory text. It does not simulate a second arithmetic path. Stages are conceptual algorithm boundaries, not clock cycles or claimed accelerator latency. Real hardware may compress alignment tails with jamming, share or fuse stages, pipeline differently, or support different modes; no such physical design is implied here.

## Examples and UI

The ten `ARITHMETIC_EXAMPLES` entries contain explicit raw operand patterns, operation, expected raw result, and explanation. Tests independently check them. The UI distinguishes draft input from applied operands, shows the exact sum separately from its rounded result, and provides native buttons plus Left/Right/Home/End shortcuts to inspect stages. Long binary strings remain intact in horizontally scrollable rows; field names and GRS labels supplement colors.

Decimal/scientific input is parsed to binary64 before **operand** quantization. It does not promise exact decimal semantics. Special tokens are case-insensitive; raw hex requires exactly four digits and preserves metadata unavailable through Number NaNs. Arithmetic flags cover only the applied raw FP16 operation, not decimal parsing or conversion.

## Verification

The prior **191 tests are unchanged**. New numerical tests cover directed signed/normal/subnormal/overflow boundaries, all signed-zero combinations, NaN priority and payloads, every raw pattern with zero, exhaustive finite self-cancellation, every attainable outward rounding midpoint on both signs (including overflow), 25,000 deterministic random pairs in both operations, and 5,000 trace reconstructions. Midpoints below normal exponent −13 cannot be sums of two binary16 operands because half-spacing is below `2^-24`; existing conversion tests still cover those midpoints independently. Heavy loops use native strict assertions and per-test 30-second budgets without weakening numerical checks or raising the global timeout.

The independent oracle uses an enumerated positive finite lattice and binary-search rounding thresholds. It shares no production decoding, encoding, or arithmetic logic. Directed NaN policy tests supply explicit expected raw patterns rather than relying only on that oracle. UI tests exercise the actual numerical engine, all ten presets, decimal quantization, raw signaling metadata, ties, normalization, overflow, invalid input, draft/applied separation, and keyboard controls.

### Optional Berkeley SoftFloat differential validation

An external run used [Berkeley SoftFloat 3](https://github.com/ucb-bar/berkeley-softfloat-3), revision **a0c6494cdc11865811dec815d5c0049fba9d82a8**, with the supplied `Linux-x86_64-GCC` build and `8086-SSE` specialization, compiled with Apple clang on arm64. The [SoftFloat documentation](https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat.html) describes its rounding modes, tininess selection, exception flags, and specialization behavior. Source and binaries remain outside this repository; the upstream BSD 3-Clause license was inspected. Only our original driver/script are included.

Reproduce with a C compiler and make (build directory may need adjustment on other platforms):

```sh
# Choose a temporary directory outside this repository.
git clone https://github.com/ucb-bar/berkeley-softfloat-3.git /tmp/fp-play-softfloat
cd /tmp/fp-play-softfloat
git checkout a0c6494cdc11865811dec815d5c0049fba9d82a8
make -C build/Linux-x86_64-GCC -j2

# Back in the FP Play repository, after npm ci:
cc -O2 -I/tmp/fp-play-softfloat/source/include \
  scripts/softfloat-driver.c \
  /tmp/fp-play-softfloat/build/Linux-x86_64-GCC/softfloat.a \
  -o /tmp/fp-play-softfloat-driver
node scripts/validate-softfloat.mjs /tmp/fp-play-softfloat-driver
```

This section records Milestone 5 validation. The current driver accepts four fields (`hexA hexB operation hexC`) and adds multiplication/FMA; recompile it before running the [expanded Milestone 6 corpus](MULTIPLY_FMA.md#verification-and-reproducible-counterexamples).

The driver sets `softfloat_round_near_even` and `softfloat_tininess_afterRounding`, clears status before each operation, and reports raw output and flags. The JavaScript runner uses the existing TypeScript dependency to transpile the production engine into a cleaned temporary directory; no new dependency is required.

**Observed: 231,872 operations, 10,572 NaN results, zero mismatches.** The corpus contains all 65,536 patterns against zero in add/sub, all pairs from a 20-pattern directed boundary set in add/sub, and 50,000 deterministic random pairs in add/sub. Non-NaN result bits and all four flags must match exactly. NaN outputs must be quiet and flags must match; payload/sign equality is deliberately not required because SoftFloat's selected specialization has a different propagation policy. Our explicit payload/sign policy has separate directed tests.

This comparison is optional manual validation, not a normal-test dependency or an exhaustive test of all `2^32` operand pairs. TestFloat was not run, and neither this corpus nor the internal tests establish full IEEE 754 conformance.

## Accelerator context and limits

Eleven normal significand bits can push small addends below retained precision. Cancellation can expose low-order bits; repeated rounding and accumulator choice can affect numerical accuracy in accelerator workloads. These exact single-operation traces help explain those effects without claiming to model a particular accelerator, reduction order, throughput, or proprietary datapath.

No connected browser was available for visual review. DOM tests verify semantic controls and keyboard behavior, but browser layout, native focus appearance, mobile scrolling, themes, and actual assistive-technology announcements remain unverified.
