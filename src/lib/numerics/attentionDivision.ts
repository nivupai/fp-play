import {
  canonicalNaN,
  decodeFloat,
  emptyFlags,
  FORMATS,
  infinityBits,
} from './formats'
import type { FloatValue, FormatName } from './formats'
import type { ArithmeticFlags } from './fp16Arithmetic'
export interface DivisionResult extends FloatValue {
  flags: ArithmeticFlags
  halfway: boolean
  increment: boolean
  divideByZero: boolean
}
/** Attention normalization helper: exact rational quotient, one destination RNE.
 * No binary64 division is used for finite operands. Not a new arithmetic workspace.
 */
export function divideForAttention(
  format: FormatName,
  a: FloatValue,
  b: FloatValue,
): DivisionResult {
  const x = decodeFloat(a.format, a.bits),
    y = decodeFloat(b.format, b.bits),
    f = FORMATS[format],
    flags = emptyFlags()
  const sign = (x.sign ^ y.sign) as 0 | 1,
    signBits = sign * 2 ** (f.width - 1)
  const result = (
    bits: number,
    halfway = false,
    increment = false,
    divideByZero = false,
  ): DivisionResult => ({
    format,
    bits,
    flags,
    halfway,
    increment,
    divideByZero,
  })
  if (x.classification === 'nan' || y.classification === 'nan') {
    flags.invalid =
      (x.classification === 'nan' && !x.isQuietNaN) ||
      (y.classification === 'nan' && !y.isQuietNaN)
    // Attention's quotient NaNs canonicalize; no raw-NaN operand entry is offered.
    return result(canonicalNaN(format))
  }
  if (
    (x.value === 0 && y.value === 0) ||
    (x.classification === 'infinity' && y.classification === 'infinity')
  ) {
    flags.invalid = true
    return result(canonicalNaN(format))
  }
  if (y.value === 0)
    return result(
      signBits + infinityBits(format),
      false,
      false,
      x.classification !== 'infinity',
    )
  if (x.classification === 'infinity')
    return result(signBits + infinityBits(format))
  if (x.value === 0 || y.classification === 'infinity') return result(signBits)
  let numerator = x.exact!.coefficient,
    denominator = y.exact!.coefficient
  if (numerator < 0n) numerator = -numerator
  if (denominator < 0n) denominator = -denominator
  const scale = x.exact!.exponent - y.exact!.exponent
  if (scale >= 0) numerator <<= BigInt(scale)
  else denominator <<= BigInt(-scale)
  let exponent = numerator.toString(2).length - denominator.toString(2).length
  const below =
    exponent >= 0
      ? numerator < denominator << BigInt(exponent)
      : numerator << BigInt(-exponent) < denominator
  if (below) exponent--
  const emin = 1 - f.bias,
    quantum = Math.max(emin, exponent) - f.fractionBits
  let n = numerator,
    d = denominator
  if (quantum >= 0) d <<= BigInt(quantum)
  else n <<= BigInt(-quantum)
  let retained = n / d
  const remainder = n % d,
    halfway = remainder * 2n === d
  const increment = remainder * 2n > d || (halfway && (retained & 1n) === 1n)
  retained += increment ? 1n : 0n
  exponent = Math.max(emin, exponent)
  if (retained === 1n << BigInt(f.fractionBits + 1)) {
    retained >>= 1n
    exponent++
  }
  flags.overflow = exponent > f.bias
  flags.inexact = remainder !== 0n || flags.overflow
  const tinyExponent = emin - f.fractionBits - 2,
    tinyCoefficient = (1n << BigInt(f.fractionBits + 2)) - 1n
  const tiny =
    tinyExponent >= 0
      ? numerator < (denominator * tinyCoefficient) << BigInt(tinyExponent)
      : numerator << BigInt(-tinyExponent) < denominator * tinyCoefficient
  flags.underflow = tiny && flags.inexact
  const magnitude = flags.overflow
    ? infinityBits(format)
    : retained < 1n << BigInt(f.fractionBits)
      ? Number(retained)
      : (exponent + f.bias) * 2 ** f.fractionBits +
        Number(retained - (1n << BigInt(f.fractionBits)))
  return result(signBits + magnitude, halfway, increment)
}
