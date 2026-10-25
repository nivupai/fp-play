/** Independent fixed integer lattice, unit 2^-2148. No production imports.
 * Decode by raw field masks; quantize by binary search of ordered encodings,
 * comparing exact distances (no normalization, GRS, or logarithmic exponent selection).
 */
export type RefFormat = 'FP16' | 'BF16' | 'FP32'
const spec = {
  FP16: [10, 5, 15, 16],
  BF16: [7, 8, 127, 16],
  FP32: [23, 8, 127, 32],
} as const
export function refDecode(format: RefFormat, bits: number) {
  const [p, e, bias, width] = spec[format],
    raw = BigInt(bits)
  const sign = Number(raw >> BigInt(width - 1)),
    fraction = Number(raw & ((1n << BigInt(p)) - 1n)),
    exponent = Number((raw >> BigInt(p)) & ((1n << BigInt(e)) - 1n))
  const kind =
    exponent === 2 ** e - 1
      ? fraction
        ? 'nan'
        : 'infinity'
      : exponent
        ? 'normal'
        : fraction
          ? 'subnormal'
          : 'zero'
  const units =
    kind === 'infinity' || kind === 'nan'
      ? null
      : BigInt((sign ? -1 : 1) * (exponent ? 2 ** p + fraction : fraction)) <<
        BigInt((exponent ? exponent - bias : 1 - bias) - p + 2148)
  return {
    sign,
    exponent,
    fraction,
    kind,
    units,
    quiet: kind === 'nan' && fraction >= 2 ** (p - 1),
  }
}
export function refNumber(value: number): bigint | null {
  if (!Number.isFinite(value)) return null
  const view = new DataView(new ArrayBuffer(8))
  view.setFloat64(0, value)
  const raw = view.getBigUint64(0),
    exp = Number((raw >> 52n) & 2047n),
    sig = (raw & 0xfffffffffffffn) + (exp ? 0x10000000000000n : 0n)
  return (raw >> 63n ? -sig : sig) << BigInt(2148 + (exp ? exp - 1075 : -1074))
}
export function refRound(
  format: RefFormat,
  units: bigint,
  negativeZero = false,
) {
  const [p, e, bias, width] = spec[format],
    inf = (2 ** e - 1) * 2 ** p
  const negative = units < 0n || (units === 0n && negativeZero),
    n = units < 0n ? -units : units
  let lo = 0,
    hi = inf - 1
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2)
    if (refDecode(format, mid).units! <= n) lo = mid
    else hi = mid - 1
  }
  const lower = refDecode(format, lo).units!,
    upper =
      lo === inf - 1
        ? 1n << BigInt(bias + 1 + 2148)
        : refDecode(format, lo + 1).units!
  const up = n - lower > upper - n || (n - lower === upper - n && lo % 2 === 1)
  const magnitude = lo + (up ? 1 : 0),
    overflow = magnitude === inf
  const inexact = overflow || refDecode(format, magnitude).units! !== n
  // Separately derived precision-only minimum-normal midpoint on the global lattice.
  const tinyThreshold =
    (1n << BigInt(2148 + 1 - bias)) -
    (1n << BigInt(2148 + 1 - bias - (p + 1) - 1))
  return {
    bits: magnitude + (negative ? 2 ** (width - 1) : 0),
    flags: {
      invalid: false,
      overflow,
      underflow: n !== 0n && n < tinyThreshold && inexact,
      inexact,
    },
  }
}
export function refArithmetic(
  format: RefFormat,
  operation: 'add' | 'multiply',
  a: number,
  b: number,
) {
  const x = refDecode(format, a),
    y = refDecode(format, b)
  if (x.units === null || y.units === null)
    throw new Error(
      'Finite oracle only; special-value expectations are directed separately',
    )
  const units =
    operation === 'add' ? x.units + y.units : (x.units * y.units) >> 2148n
  return refRound(
    format,
    units,
    operation === 'multiply'
      ? x.sign !== y.sign
      : x.units === 0n && y.units === 0n && x.sign === 1 && y.sign === 1,
  )
}
export function seededRandom(seed = 0x8f32bf16) {
  return () => {
    seed ^= seed << 13
    seed ^= seed >>> 17
    seed ^= seed << 5
    return seed >>> 0
  }
}
