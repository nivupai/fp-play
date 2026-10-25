import type { ArithmeticFlags } from './fp16Arithmetic'
import type { FP16Classification } from './fp16'

export type FormatName = 'FP16' | 'BF16' | 'FP32'
export interface FloatFormat {
  readonly name: FormatName
  readonly exponentBits: number
  readonly fractionBits: number
  readonly width: number
  readonly bias: number
}
export const FORMATS = Object.freeze({
  FP16: Object.freeze({
    name: 'FP16',
    exponentBits: 5,
    fractionBits: 10,
    width: 16,
    bias: 15,
  }),
  BF16: Object.freeze({
    name: 'BF16',
    exponentBits: 8,
    fractionBits: 7,
    width: 16,
    bias: 127,
  }),
  FP32: Object.freeze({
    name: 'FP32',
    exponentBits: 8,
    fractionBits: 23,
    width: 32,
    bias: 127,
  }),
} satisfies Record<FormatName, FloatFormat>)
export interface Dyadic {
  readonly coefficient: bigint
  readonly exponent: number
}
export interface FloatValue {
  readonly format: FormatName
  readonly bits: number
}
export interface DecodedFloat extends FloatValue {
  readonly sign: 0 | 1
  readonly exponent: number
  readonly fraction: number
  readonly unbiasedExponent: number | null
  readonly classification: FP16Classification
  readonly value: number
  readonly exact: Dyadic | null
  readonly hasImplicitLeadingBit: boolean
  readonly isQuietNaN: boolean | null
  readonly nanPayload: number | null
}
export function emptyFlags(): ArithmeticFlags {
  return { invalid: false, overflow: false, underflow: false, inexact: false }
}
export function unionFlags(...flags: ArithmeticFlags[]): ArithmeticFlags {
  return {
    invalid: flags.some((f) => f.invalid),
    overflow: flags.some((f) => f.overflow),
    underflow: flags.some((f) => f.underflow),
    inexact: flags.some((f) => f.inexact),
  }
}
export function assertBits(format: FormatName, bits: number): void {
  if (!Number.isInteger(bits) || bits < 0 || bits >= 2 ** FORMATS[format].width)
    throw new RangeError(
      `${format} requires an unsigned ${FORMATS[format].width}-bit integer`,
    )
}
export function infinityBits(format: FormatName): number {
  const f = FORMATS[format]
  return (2 ** f.exponentBits - 1) * 2 ** f.fractionBits
}
export function canonicalNaN(format: FormatName): number {
  return infinityBits(format) + 2 ** (FORMATS[format].fractionBits - 1)
}
export function decodeFloat(format: FormatName, bits: number): DecodedFloat {
  assertBits(format, bits)
  const f = FORMATS[format],
    unit = 2 ** f.fractionBits
  const sign = Math.floor(bits / 2 ** (f.width - 1)) as 0 | 1
  const exponent = Math.floor(bits / unit) % 2 ** f.exponentBits
  const fraction = bits % unit
  const classification =
    exponent === 2 ** f.exponentBits - 1
      ? fraction
        ? 'nan'
        : 'infinity'
      : exponent === 0
        ? fraction
          ? 'subnormal'
          : 'zero'
        : 'normal'
  const unbiasedExponent =
    classification === 'normal'
      ? exponent - f.bias
      : classification === 'subnormal'
        ? 1 - f.bias
        : null
  const exact =
    classification === 'nan' || classification === 'infinity'
      ? null
      : {
          coefficient: BigInt(
            (sign ? -1 : 1) * (exponent ? unit + fraction : fraction),
          ),
          exponent:
            (exponent ? exponent - f.bias : 1 - f.bias) - f.fractionBits,
        }
  const value = exact
    ? exact.coefficient === 0n
      ? sign
        ? -0
        : 0
      : Number(exact.coefficient) * 2 ** exact.exponent
    : classification === 'nan'
      ? NaN
      : sign
        ? -Infinity
        : Infinity
  return {
    format,
    bits,
    sign,
    exponent,
    fraction,
    unbiasedExponent,
    classification,
    value,
    exact,
    hasImplicitLeadingBit: classification === 'normal',
    isQuietNaN: classification === 'nan' ? fraction >= unit / 2 : null,
    nanPayload: classification === 'nan' ? fraction % (unit / 2) : null,
  }
}
/** Exact finite binary64 decomposition; NaN/infinity have no dyadic value. */
export function numberToDyadic(value: number): Dyadic | null {
  if (!Number.isFinite(value)) return null
  const view = new DataView(new ArrayBuffer(8))
  view.setFloat64(0, value)
  const raw = view.getBigUint64(0),
    exp = Number((raw >> 52n) & 2047n)
  const coefficient = (raw & ((1n << 52n) - 1n)) + (exp ? 1n << 52n : 0n)
  return {
    coefficient: raw >> 63n ? -coefficient : coefficient,
    exponent: exp ? exp - 1075 : -1074,
  }
}
export function addDyadic(a: Dyadic, b: Dyadic): Dyadic {
  const exponent = Math.min(a.exponent, b.exponent)
  return {
    coefficient:
      (a.coefficient << BigInt(a.exponent - exponent)) +
      (b.coefficient << BigInt(b.exponent - exponent)),
    exponent,
  }
}
export function multiplyDyadic(a: Dyadic, b: Dyadic): Dyadic {
  return {
    coefficient: a.coefficient * b.coefficient,
    exponent: a.exponent + b.exponent,
  }
}
export function subtractDyadic(a: Dyadic, b: Dyadic): Dyadic {
  return addDyadic(a, { coefficient: -b.coefficient, exponent: b.exponent })
}
function lessDyadic(a: bigint, ae: number, b: bigint, be: number): boolean {
  return ae >= be ? a << BigInt(ae - be) < b : a < b << BigInt(be - ae)
}
export interface RoundedFloat extends FloatValue {
  readonly flags: ArithmeticFlags
  readonly halfway: boolean
  readonly increment: boolean
  readonly exact: Dyadic | null
}
/** One RNE rounding of an exact dyadic. Tininess uses unbounded-exponent precision rounding. */
export function roundDyadic(
  format: FormatName,
  exact: Dyadic,
  zeroSign: 0 | 1 = 0,
): RoundedFloat {
  const f = FORMATS[format],
    p = f.fractionBits,
    emin = 1 - f.bias
  const sign =
    exact.coefficient < 0n ? 1 : exact.coefficient > 0n ? 0 : zeroSign
  const magnitude =
    exact.coefficient < 0n ? -exact.coefficient : exact.coefficient
  let exponent = Math.max(
    emin,
    magnitude ? magnitude.toString(2).length - 1 + exact.exponent : emin,
  )
  const shift = exponent - p - exact.exponent
  let retained =
    shift > 0 ? magnitude >> BigInt(shift) : magnitude << BigInt(-shift)
  const remainder = shift > 0 ? magnitude - (retained << BigInt(shift)) : 0n
  const half = shift > 0 ? 1n << BigInt(shift - 1) : 0n
  const halfway = shift > 0 && remainder === half
  const increment =
    shift > 0 && (remainder > half || (halfway && (retained & 1n) !== 0n))
  retained += increment ? 1n : 0n
  if (retained === 1n << BigInt(p + 1)) {
    retained >>= 1n
    exponent++
  }
  const flags = emptyFlags()
  flags.overflow = exponent > f.bias
  flags.inexact = remainder !== 0n || flags.overflow
  // p stored fraction bits => p+1 significant bits; threshold (2^(p+2)-1)*2^(emin-p-2).
  flags.underflow =
    magnitude !== 0n &&
    flags.inexact &&
    lessDyadic(
      magnitude,
      exact.exponent,
      (1n << BigInt(p + 2)) - 1n,
      emin - p - 2,
    )
  const magnitudeBits = flags.overflow
    ? infinityBits(format)
    : retained < 1n << BigInt(p)
      ? Number(retained)
      : (exponent + f.bias) * 2 ** p + Number(retained - (1n << BigInt(p)))
  return {
    format,
    bits: sign * 2 ** (f.width - 1) + magnitudeBits,
    flags,
    exact,
    halfway,
    increment,
  }
}
/** Direct binary64 conversion, including BF16: no binary32 intermediate. */
export function encodeFloat(format: FormatName, value: number): number {
  const exact = numberToDyadic(value)
  if (exact)
    return roundDyadic(format, exact, Object.is(value, -0) ? 1 : 0).bits
  return Number.isNaN(value)
    ? canonicalNaN(format)
    : infinityBits(format) + (value < 0 ? 2 ** (FORMATS[format].width - 1) : 0)
}
export function floatHex(format: FormatName, bits: number): string {
  assertBits(format, bits)
  return `0x${bits.toString(16).padStart(FORMATS[format].width / 4, '0')}`
}
export function floatBinary(format: FormatName, bits: number): string {
  assertBits(format, bits)
  return bits.toString(2).padStart(FORMATS[format].width, '0')
}
export function nextUp(format: FormatName, bits: number): number {
  const d = decodeFloat(format, bits)
  if (d.classification === 'nan' || d.value === Infinity) return bits
  if (d.value === 0) return 1
  return bits + (d.sign ? -1 : 1)
}
export function nextDown(format: FormatName, bits: number): number {
  const d = decodeFloat(format, bits)
  if (d.classification === 'nan' || d.value === -Infinity) return bits
  if (d.value === 0) return 2 ** (FORMATS[format].width - 1) + 1
  return bits + (d.sign ? 1 : -1)
}
/** Sign-independent binade spacing; zero/subnormal share minimum quantum. */
export function floatULP(format: FormatName, bits: number): number {
  const d = decodeFloat(format, bits),
    f = FORMATS[format]
  return d.classification === 'nan'
    ? NaN
    : d.classification === 'infinity'
      ? Infinity
      : 2 ** ((d.unbiasedExponent ?? 1 - f.bias) - f.fractionBits)
}
