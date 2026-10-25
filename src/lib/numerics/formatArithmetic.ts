import {
  addDyadic,
  canonicalNaN,
  decodeFloat,
  emptyFlags,
  FORMATS,
  infinityBits,
  multiplyDyadic,
  roundDyadic,
} from './formats'
import type { FloatValue, FormatName, RoundedFloat } from './formats'

/** Mixed-format operands are decoded exactly; only the selected output is rounded. */
export function calculateFloat(
  output: FormatName,
  operation: 'add' | 'multiply' | 'fma',
  a: FloatValue,
  b: FloatValue,
  c?: FloatValue,
): RoundedFloat {
  if (operation === 'fma' && !c) throw new TypeError('FMA requires operand C')
  const operands = [a, b, ...(operation === 'fma' && c ? [c] : [])].map((x) =>
    decodeFloat(x.format, x.bits),
  )
  const [x, y, z] = operands
  const flags = emptyFlags()
  const productInvalid =
    operation !== 'add' &&
    ((x.classification === 'zero' && y.classification === 'infinity') ||
      (y.classification === 'zero' && x.classification === 'infinity'))
  flags.invalid =
    productInvalid ||
    operands.some((d) => d.classification === 'nan' && !d.isQuietNaN)
  const nan =
    operands.find((d) => d.classification === 'nan' && !d.isQuietNaN) ??
    operands.find((d) => d.classification === 'nan')
  const special = (bits: number): RoundedFloat => ({
    format: output,
    bits,
    flags,
    exact: null,
    halfway: false,
    increment: false,
  })
  if (nan) {
    // Align the payload's high bits on conversion; always quiet, preserving sign.
    const from = FORMATS[nan.format].fractionBits,
      to = FORMATS[output].fractionBits
    const payload =
      from > to
        ? Math.floor(nan.fraction / 2 ** (from - to))
        : nan.fraction * 2 ** (to - from)
    return special(
      nan.sign * 2 ** (FORMATS[output].width - 1) +
        infinityBits(output) +
        (payload | (2 ** (to - 1))),
    )
  }
  if (productInvalid) return special(canonicalNaN(output))
  const productSign = (x.sign ^ y.sign) as 0 | 1
  const firstInfinity =
    operation === 'add'
      ? x.classification === 'infinity'
      : x.classification === 'infinity' || y.classification === 'infinity'
  const second = operation === 'add' ? y : z
  const firstSign = operation === 'add' ? x.sign : productSign
  if (firstInfinity || second?.classification === 'infinity') {
    if (
      firstInfinity &&
      second?.classification === 'infinity' &&
      firstSign !== second.sign
    ) {
      flags.invalid = true
      return special(canonicalNaN(output))
    }
    const sign = firstInfinity ? firstSign : second!.sign
    return special(
      sign * 2 ** (FORMATS[output].width - 1) + infinityBits(output),
    )
  }
  const first =
    operation === 'add' ? x.exact! : multiplyDyadic(x.exact!, y.exact!)
  const exact =
    operation === 'multiply' ? first : addDyadic(first, second!.exact!)
  const zeroSign =
    operation === 'multiply'
      ? productSign
      : first.coefficient === 0n &&
          second!.exact!.coefficient === 0n &&
          firstSign === 1 &&
          second!.sign === 1
        ? 1
        : 0
  return roundDyadic(output, exact, zeroSign)
}
export function addFloat(
  format: FormatName,
  a: number,
  b: number,
): RoundedFloat {
  return calculateFloat(format, 'add', { format, bits: a }, { format, bits: b })
}
export function multiplyFloat(
  format: FormatName,
  a: number,
  b: number,
): RoundedFloat {
  return calculateFloat(
    format,
    'multiply',
    { format, bits: a },
    { format, bits: b },
  )
}
