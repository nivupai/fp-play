import { parseDecimalInput, formatNumber } from '../explorer/presentation'
import { numberToDyadic, subtractDyadic } from '../numerics/formats'
import type { Dyadic } from '../numerics/formats'
export function formatDyadic(value: Dyadic | null): string {
  if (!value) return 'Undefined (nonfinite input or result)'
  const negative = value.coefficient < 0n,
    n = negative ? -value.coefficient : value.coefficient
  if (n === 0n) return '0'
  if (value.exponent >= 0)
    return `${negative ? '-' : ''}${n << BigInt(value.exponent)}`
  const places = -value.exponent,
    digits = (n * 5n ** BigInt(places)).toString().padStart(places + 1, '0')
  return `${negative ? '-' : ''}${digits.slice(0, -places)}.${digits.slice(-places)}`.replace(
    /\.?0+$/,
    '',
  )
}
export function exactValue(value: number): string {
  return value === 0 || !Number.isFinite(value)
    ? formatNumber(value)
    : formatDyadic(numberToDyadic(value))
}
export function quantizationError(
  original: number,
  quantized: number,
): { absolute: string; relative: string } {
  const a = numberToDyadic(original),
    b = numberToDyadic(quantized)
  if (!a || !b)
    return {
      absolute: 'Undefined for nonfinite values',
      relative: 'Undefined for nonfinite values',
    }
  const error = subtractDyadic(b, a)
  return {
    absolute: formatDyadic({
      ...error,
      coefficient:
        error.coefficient < 0n ? -error.coefficient : error.coefficient,
    }),
    relative:
      original === 0
        ? 'Undefined at zero'
        : `${Math.abs((quantized - original) / original)} (binary64 estimate)`,
  }
}
export function parseVector(text: string): number[] {
  const tokens = text.trim().split(/[\s,]+/)
  if (tokens.length > 64)
    throw new RangeError('Use at most 64 values per vector.')
  return tokens.map((token) => {
    const parsed = parseDecimalInput(token)
    if (!parsed.ok) throw new TypeError(parsed.error)
    return parsed.value
  })
}
