import {
  addDyadic,
  decodeFloat,
  encodeFloat,
  emptyFlags,
  multiplyDyadic,
  numberToDyadic,
  subtractDyadic,
  unionFlags,
} from './formats'
import type { Dyadic, FloatValue, FormatName, RoundedFloat } from './formats'
import { calculateFloat } from './formatArithmetic'

export type DotMode = 'A' | 'B' | 'C' | 'D' | 'E'
export type ReductionOrder = 'sequential' | 'pairwise'
export const DOT_MODES = {
  A: {
    input: 'FP16',
    product: 'FP16',
    accumulator: 'FP16',
    fused: false,
    label: 'A · FP16 products → FP16 sum',
  },
  B: {
    input: 'FP16',
    product: 'FP16',
    accumulator: 'FP32',
    fused: false,
    label: 'B · FP16 products → FP32 sum',
  },
  C: {
    input: 'FP16',
    product: 'FP32',
    accumulator: 'FP32',
    fused: true,
    label: 'C · FP16 inputs → FP32 FMA',
  },
  D: {
    input: 'BF16',
    product: 'FP32',
    accumulator: 'FP32',
    fused: true,
    label: 'D · BF16 inputs → FP32 FMA',
  },
  E: {
    input: 'FP32',
    product: 'FP32',
    accumulator: 'FP32',
    fused: false,
    label: 'E · FP32 products → FP32 sum',
  },
} as const satisfies Record<
  DotMode,
  {
    input: FormatName
    product: FormatName
    accumulator: FormatName
    fused: boolean
    label: string
  }
>
export interface DotStep {
  readonly operation: 'multiply' | 'add' | 'fma'
  readonly terms: readonly number[]
  readonly operands: readonly FloatValue[]
  readonly result: RoundedFloat
  /** Exact dot/subsum of quantized inputs for the represented terms. */
  readonly reference: Dyadic | null
  readonly error: Dyadic | null
}
export interface DotResult {
  readonly mode: DotMode
  readonly order: ReductionOrder
  readonly a: readonly FloatValue[]
  readonly b: readonly FloatValue[]
  readonly result: FloatValue
  readonly flags: ReturnType<typeof emptyFlags>
  readonly steps: readonly DotStep[]
  readonly exactOriginal: Dyadic | null
  readonly exactQuantized: Dyadic | null
  readonly quantizedError: Dyadic | null
  readonly originalError: Dyadic | null
}
const zero: Dyadic = { coefficient: 0n, exponent: 0 }
function sumExact(values: (Dyadic | null)[]): Dyadic | null {
  return values.some((x) => x === null)
    ? null
    : (values as Dyadic[]).reduce(addDyadic, zero)
}
export function exactDot(
  a: readonly number[],
  b: readonly number[],
): Dyadic | null {
  if (a.length !== b.length)
    throw new RangeError('Vectors must have equal lengths')
  return sumExact(
    a.map((x, i) => {
      const ax = numberToDyadic(x),
        bx = numberToDyadic(b[i])
      return ax && bx ? multiplyDyadic(ax, bx) : null
    }),
  )
}
/** Deterministic simulation, 1–64 terms. Conversion flags are not arithmetic flags.
 * Pairwise: rounded product (or FMA(a,b,+0)) leaves, then balanced rounded additions.
 * Sequential: +0 seed, multiply/add or one FMA per term. No reassociation inside a step.
 */
export function dotProduct(
  a: readonly number[],
  b: readonly number[],
  mode: DotMode,
  order: ReductionOrder = 'sequential',
): DotResult {
  if (
    !Object.hasOwn(DOT_MODES, mode) ||
    !['sequential', 'pairwise'].includes(order)
  )
    throw new RangeError('Unknown dot-product configuration')
  if (a.length !== b.length || a.length < 1 || a.length > 64)
    throw new RangeError('Enter equal-length vectors of 1–64 values')
  const config = DOT_MODES[mode]
  const qa = a.map((value) => ({
      format: config.input,
      bits: encodeFloat(config.input, value),
    })),
    qb = b.map((value) => ({
      format: config.input,
      bits: encodeFloat(config.input, value),
    }))
  const products = qa.map((value, i) => {
    const x = decodeFloat(value.format, value.bits).exact,
      y = decodeFloat(qb[i].format, qb[i].bits).exact
    return x && y ? multiplyDyadic(x, y) : null
  })
  const steps: DotStep[] = []
  const seed: FloatValue = { format: config.accumulator, bits: 0 }
  const perform = (
    operation: DotStep['operation'],
    operands: FloatValue[],
    terms: number[],
    output: FormatName,
  ): FloatValue => {
    const result = calculateFloat(
      output,
      operation,
      operands[0],
      operands[1],
      operands[2],
    )
    const reference = sumExact(terms.map((i) => products[i]))
    const exactResult = decodeFloat(result.format, result.bits).exact
    steps.push({
      operation,
      operands,
      terms,
      result,
      reference,
      error:
        reference && exactResult
          ? subtractDyadic(exactResult, reference)
          : null,
    })
    return { format: result.format, bits: result.bits }
  }
  const leaf = (i: number) =>
    config.fused
      ? perform('fma', [qa[i], qb[i], seed], [i], config.accumulator)
      : perform('multiply', [qa[i], qb[i]], [i], config.product)
  let result: FloatValue
  if (order === 'sequential') {
    result = seed
    for (let i = 0; i < a.length; i++) {
      const terms = Array.from({ length: i + 1 }, (_, j) => j)
      result = config.fused
        ? perform('fma', [qa[i], qb[i], result], terms, config.accumulator)
        : perform('add', [result, leaf(i)], terms, config.accumulator)
    }
  } else {
    const reduce = (start: number, end: number): FloatValue => {
      if (end - start === 1) return leaf(start)
      const middle = start + Math.floor((end - start) / 2)
      return perform(
        'add',
        [reduce(start, middle), reduce(middle, end)],
        Array.from({ length: end - start }, (_, i) => start + i),
        config.accumulator,
      )
    }
    result = reduce(0, a.length)
    // A one-leaf B tree still produces an FP32 accumulator encoding.
    if (result.format !== config.accumulator)
      result = {
        format: config.accumulator,
        bits: encodeFloat(
          config.accumulator,
          decodeFloat(result.format, result.bits).value,
        ),
      }
  }
  const exactOriginal = exactDot(a, b),
    exactQuantized = sumExact(products),
    decoded = decodeFloat(result.format, result.bits)
  return {
    mode,
    order,
    a: qa,
    b: qb,
    result,
    flags: unionFlags(...steps.map((s) => s.result.flags)),
    steps,
    exactOriginal,
    exactQuantized,
    quantizedError:
      decoded.exact && exactQuantized
        ? subtractDyadic(decoded.exact, exactQuantized)
        : null,
    originalError:
      decoded.exact && exactOriginal
        ? subtractDyadic(decoded.exact, exactOriginal)
        : null,
  }
}
