// @vitest-environment node
/// <reference types="node" />
import { strictEqual, deepStrictEqual } from 'node:assert'
import { describe, expect, it } from 'vitest'
import { DOT_MODES, dotProduct, exactDot } from '../lib/numerics/dotProduct'
import type { DotMode } from '../lib/numerics/dotProduct'
import { decodeFloat } from '../lib/numerics/formats'
import { MIXED_EXAMPLES } from '../lib/mixed/examples'
import { formatDyadic, parseVector } from '../lib/mixed/presentation'
import {
  refDecode,
  refNumber,
  refRound,
  seededRandom,
} from './reference/formatReference'

const toUnits = (d: { coefficient: bigint; exponent: number }) =>
  d.coefficient << BigInt(d.exponent + 2148)
/** Independent lattice execution. Each rounding boundary uses binary search, not production rounding. */
function referenceDot(
  a: number[],
  b: number[],
  mode: DotMode,
  pairwise: boolean,
) {
  const c = DOT_MODES[mode]
  const qa = a.map((v) => refRound(c.input, refNumber(v)!).bits),
    qb = b.map((v) => refRound(c.input, refNumber(v)!).bits)
  const products = qa.map(
    (bits, i) =>
      (refDecode(c.input, bits).units! * refDecode(c.input, qb[i]).units!) >>
      2148n,
  )
  type Value = { format: 'FP16' | 'BF16' | 'FP32'; bits: number }
  const leaf = (i: number): Value => ({
    format: c.product,
    bits: refRound(
      c.product,
      products[i],
      !c.fused &&
        refDecode(c.input, qa[i]).sign !== refDecode(c.input, qb[i]).sign,
    ).bits,
  })
  const add = (a: Value, b: Value): Value => {
    const x = refDecode(a.format, a.bits),
      y = refDecode(b.format, b.bits)
    return {
      format: c.accumulator,
      bits: refRound(
        c.accumulator,
        x.units! + y.units!,
        x.units === 0n && y.units === 0n && x.sign === 1 && y.sign === 1,
      ).bits,
    }
  }
  const tree = (start: number, end: number): Value => {
    if (end - start === 1) return leaf(start)
    const mid = start + Math.floor((end - start) / 2)
    return add(tree(start, mid), tree(mid, end))
  }
  let result: Value = { format: c.accumulator, bits: 0 }
  if (pairwise) result = tree(0, a.length)
  else
    for (let i = 0; i < a.length; i++) {
      if (c.fused) {
        const acc = refDecode(result.format, result.bits)
        const negativeZero =
          products[i] === 0n &&
          acc.units === 0n &&
          acc.sign === 1 &&
          refDecode(c.input, qa[i]).sign !== refDecode(c.input, qb[i]).sign
        result = {
          format: c.accumulator,
          bits: refRound(c.accumulator, products[i] + acc.units!, negativeZero)
            .bits,
        }
      } else result = add(result, leaf(i))
    }
  const raw = refDecode(result.format, result.bits)
  return {
    bits: refRound(c.accumulator, raw.units!, raw.sign === 1).bits,
    exact: products.reduce((x, y) => x + y, 0n),
  }
}
describe('mixed-precision dot products', () => {
  for (const example of MIXED_EXAMPLES)
    it(`independently verifies experiment: ${example.title}`, () => {
      const result = dotProduct(
        example.a,
        example.b,
        example.mode,
        example.order,
      )
      expect(decodeFloat(result.result.format, result.result.bits).value).toBe(
        example.expected,
      )
      expect(result.result.bits).toBe(
        referenceDot(
          example.a,
          example.b,
          example.mode,
          example.order === 'pairwise',
        ).bits,
      )
      expect(toUnits(result.exactOriginal!)).toBe(
        example.a.reduce(
          (sum, x, i) =>
            sum + ((refNumber(x)! * refNumber(example.b[i])!) >> 2148n),
          0n,
        ),
      )
    })
  it('distinguishes rounded FP16 products from FP32 FMA for identical inputs', () => {
    const a = [1 + 2 ** -10, -(1 + 2 ** -9)],
      b = [1 + 2 ** -10, 1]
    const rounded = dotProduct(a, b, 'B'),
      fused = dotProduct(a, b, 'C')
    expect(decodeFloat('FP32', rounded.result.bits).value).toBe(0)
    expect(decodeFloat('FP32', fused.result.bits).value).toBe(2 ** -20)
    expect(rounded.flags.inexact).toBe(true)
    expect(fused.flags.inexact).toBe(false)
    expect(rounded.steps.map((s) => s.operation)).toEqual([
      'multiply',
      'add',
      'multiply',
      'add',
    ])
    expect(fused.steps.map((s) => s.operation)).toEqual(['fma', 'fma'])
  })
  it('validates 2000 reductions and each intermediate against the separate lattice oracle', () => {
    const random = seededRandom(0x12345678)
    for (let sample = 0; sample < 200; sample++) {
      const count = 1 + (random() % 16)
      const vector = () =>
        Array.from(
          { length: count },
          () => ((random() % 4097) - 2048) * 2 ** -(7 + (random() % 18)),
        )
      const a = vector(),
        b = vector()
      for (const mode of Object.keys(DOT_MODES) as DotMode[])
        for (const order of ['sequential', 'pairwise'] as const) {
          const result = dotProduct(a, b, mode, order),
            reference = referenceDot(a, b, mode, order === 'pairwise')
          strictEqual(
            result.result.bits,
            reference.bits,
            `${sample} ${mode} ${order}`,
          )
          strictEqual(toUnits(result.exactQuantized!), reference.exact)
          for (const step of result.steps) {
            const operands = step.operands.map(
              (o) => refDecode(o.format, o.bits).units!,
            )
            const exact =
              step.operation === 'add'
                ? operands[0] + operands[1]
                : ((operands[0] * operands[1]) >> 2148n) +
                  (step.operation === 'fma' ? operands[2] : 0n)
            const signs = step.operands.map(
              (o) => refDecode(o.format, o.bits).sign,
            )
            const negativeZero =
              step.operation === 'multiply'
                ? signs[0] !== signs[1]
                : step.operation === 'add'
                  ? operands[0] === 0n &&
                    operands[1] === 0n &&
                    signs[0] === 1 &&
                    signs[1] === 1
                  : operands[0] * operands[1] === 0n &&
                    operands[2] === 0n &&
                    signs[0] !== signs[1] &&
                    signs[2] === 1
            const expected = refRound(step.result.format, exact, negativeZero)
            strictEqual(step.result.bits, expected.bits)
            deepStrictEqual(step.result.flags, expected.flags)
          }
        }
    }
  }, 30000)
  it('retains exact original binary64 products beyond binary64 range and precision', () => {
    expect(toUnits(exactDot([1, 2 ** -54, -1], [1, 1, 1])!)).toBe(1n << 2094n)
    expect(
      formatDyadic(exactDot([2 ** 1023, -(2 ** 1023)], [2 ** 1023, 2 ** 1023])),
    ).toBe('0')
    expect(exactDot([Infinity], [0])).toBeNull()
    expect(() => exactDot([1], [])).toThrow()
  })
  it('separates irreversible input quantization error from arithmetic error', () => {
    const result = dotProduct([1 + 2 ** -12, -1], [1, 1], 'C')
    expect(result.quantizedError?.coefficient).toBe(0n)
    expect(toUnits(result.originalError!)).toBe(-(1n << 2136n))
  })
  it('documents pairwise FMA leaves and rounding boundaries', () => {
    const result = dotProduct([1, 2, 3, 4], [1, 1, 1, 1], 'C', 'pairwise')
    expect(result.steps.map((s) => s.operation)).toEqual([
      'fma',
      'fma',
      'add',
      'fma',
      'fma',
      'add',
      'add',
    ])
    for (const step of result.steps.filter((s) => s.operation === 'fma'))
      expect(step.operands[2].bits).toBe(0)
    expect(decodeFloat('FP32', result.result.bits).value).toBe(10)
  })
  it('handles signed zeros and exact widening of one-leaf trees', () => {
    for (const mode of Object.keys(DOT_MODES) as DotMode[]) {
      const seq = dotProduct([-0], [1], mode),
        tree = dotProduct([-0], [1], mode, 'pairwise')
      expect(
        Object.is(decodeFloat(seq.result.format, seq.result.bits).value, 0),
      ).toBe(true)
      expect(
        Object.is(
          decodeFloat(tree.result.format, tree.result.bits).value,
          DOT_MODES[mode].fused ? 0 : -0,
        ),
      ).toBe(true)
    }
  })
  it('propagates exceptional arithmetic and reports undefined exact references', () => {
    for (const order of ['sequential', 'pairwise'] as const) {
      const invalid = dotProduct([0], [Infinity], 'D', order)
      expect(decodeFloat('FP32', invalid.result.bits).classification).toBe(
        'nan',
      )
      expect(invalid.flags.invalid).toBe(true)
      expect(invalid.exactOriginal).toBeNull()
      const opposite = dotProduct([Infinity, -Infinity], [1, 1], 'E', order)
      expect(opposite.flags.invalid).toBe(true)
      const quiet = dotProduct([NaN], [1], 'C', order)
      expect(quiet.flags.invalid).toBe(false)
      expect(quiet.quantizedError).toBeNull()
    }
    const overflow = dotProduct([65504], [2], 'B')
    expect(decodeFloat('FP32', overflow.result.bits).value).toBe(Infinity)
    expect(overflow.flags.overflow).toBe(true)
    expect(
      decodeFloat('FP32', dotProduct([65504], [2], 'C').result.bits).value,
    ).toBe(131008)
  })
  it('handles BF16 underflow and fused cancellation without intermediate FP32 overflow', () => {
    const tiny = dotProduct([2 ** -133], [2 ** -133], 'D')
    expect(tiny.flags.underflow).toBe(true)
    expect(tiny.flags.inexact).toBe(true)
    const result = dotProduct([-(2 ** 127), 2 ** 127], [1, 2], 'D')
    expect(result.flags.overflow).toBe(false)
    expect(decodeFloat('FP32', result.result.bits).value).toBe(2 ** 127)
    const tree = dotProduct([-(2 ** 127), 2 ** 127], [1, 2], 'D', 'pairwise')
    expect(tree.flags.overflow).toBe(true)
    expect(decodeFloat('FP32', tree.result.bits).value).toBe(Infinity)
  })
  it('supports the full 64-term bound with explicit rounding at each step', () => {
    for (const mode of Object.keys(DOT_MODES) as DotMode[])
      for (const order of ['sequential', 'pairwise'] as const) {
        const result = dotProduct(
          Array(64).fill(1),
          Array(64).fill(1),
          mode,
          order,
        )
        expect(
          decodeFloat(result.result.format, result.result.bits).value,
        ).toBe(64)
        expect(result.steps.length).toBe(
          order === 'pairwise' ? 127 : DOT_MODES[mode].fused ? 64 : 128,
        )
        expect(result.flags).toEqual({
          invalid: false,
          overflow: false,
          underflow: false,
          inexact: false,
        })
      }
  })
  it('rejects invalid vector sizes, syntax, and configurations', () => {
    expect(() => dotProduct([], [], 'A')).toThrow()
    expect(() => dotProduct([1], [1, 2], 'A')).toThrow()
    expect(() =>
      dotProduct(Array(65).fill(1), Array(65).fill(1), 'A'),
    ).toThrow()
    expect(() => dotProduct([1], [1], 'bad' as DotMode)).toThrow()
    expect(() => parseVector('')).toThrow()
    expect(() => parseVector('1, hello')).toThrow()
    expect(() => parseVector(Array(65).fill('1').join(','))).toThrow()
    expect(
      parseVector('-0, Infinity NaN').map((x) =>
        Object.is(x, -0) ? '-0' : String(x),
      ),
    ).toEqual(['-0', 'Infinity', 'NaN'])
  })
})
