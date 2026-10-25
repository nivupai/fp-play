// @vitest-environment node
/// <reference types="node" />
import { strictEqual, deepStrictEqual } from 'node:assert'
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  ATTENTION_POLICIES,
  attentionSoftmax,
  attentionValues,
  binary64Attention,
  calculateAttention,
  validateAttentionInput,
} from '../lib/numerics/attention'
import type {
  AttentionInput,
  AttentionMode,
  SoftmaxMethod,
} from '../lib/numerics/attention'
import { divideForAttention } from '../lib/numerics/attentionDivision'
import { decodeFloat, encodeFloat, FORMATS } from '../lib/numerics/formats'
import type { FormatName } from '../lib/numerics/formats'
import { ATTENTION_EXAMPLES } from '../lib/attention/examples'
import {
  parseAttentionMatrices,
  parseAttentionMatrix,
} from '../lib/attention/presentation'
import { refDecode, seededRandom } from './reference/formatReference'

const simple: AttentionInput = { q: [[1]], k: [[1], [0]], v: [[2], [0]] }
function soft(
  values: number[],
  work: FormatName = 'FP32',
  method: SoftmaxMethod = 'stable',
  probability: FormatName = work,
) {
  return attentionSoftmax(
    values.map((v) => ({ format: work, bits: encodeFloat(work, v) })),
    work,
    probability,
    method,
  )
}
function quotientReference(format: FormatName, a: number, b: number) {
  const x = refDecode(format, a),
    y = refDecode(format, b),
    f = FORMATS[format],
    sign = x.sign !== y.sign,
    inf = (2 ** f.exponentBits - 1) * 2 ** f.fractionBits
  const n = (x.units! < 0n ? -x.units! : x.units!) << 2148n,
    d = y.units! < 0n ? -y.units! : y.units!
  let lo = 0,
    hi = inf - 1
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2)
    if (refDecode(format, mid).units! * d <= n) lo = mid
    else hi = mid - 1
  }
  const lower = refDecode(format, lo).units!,
    upper =
      lo === inf - 1
        ? 1n << BigInt(2148 + f.bias + 1)
        : refDecode(format, lo + 1).units!
  const delta = 2n * n - (lower + upper) * d,
    bits = lo + (delta > 0n || (delta === 0n && lo % 2 === 1) ? 1 : 0)
  const overflow = bits === inf,
    inexact = overflow || refDecode(format, bits).units! * d !== n
  const tiny =
    (1n << BigInt(2148 + 1 - f.bias)) -
    (1n << BigInt(2148 + 1 - f.bias - f.fractionBits - 2))
  return {
    bits: bits + (sign ? 2 ** (f.width - 1) : 0),
    flags: {
      invalid: false,
      overflow,
      inexact,
      underflow: n !== 0n && n < tiny * d && inexact,
    },
  }
}
describe('attention exact normalization', () => {
  for (const format of Object.keys(FORMATS) as FormatName[])
    it(`${format}: 3000 independent rational quotient comparisons`, () => {
      const random = seededRandom(),
        modulus = 2 ** FORMATS[format].width
      for (let i = 0; i < 3000; i++) {
        let a = random() % modulus,
          b = random() % modulus
        while (refDecode(format, a).units === null) a = random() % modulus
        while (
          refDecode(format, b).units === null ||
          refDecode(format, b).units === 0n
        )
          b = random() % modulus
        const actual = divideForAttention(
            format,
            { format, bits: a },
            { format, bits: b },
          ),
          expected = quotientReference(format, a, b)
        strictEqual(actual.bits, expected.bits)
        deepStrictEqual(actual.flags, expected.flags)
      }
    }, 30000)
  it('reports zeros, infinities, signed zero, NaNs, and divide-by-zero without fallback', () => {
    const divide = (a: number, b: number) =>
      divideForAttention(
        'FP32',
        { format: 'FP32', bits: encodeFloat('FP32', a) },
        { format: 'FP32', bits: encodeFloat('FP32', b) },
      )
    expect(divide(0, 0).flags.invalid).toBe(true)
    expect(divide(Infinity, Infinity).flags.invalid).toBe(true)
    expect(divide(1, 0).divideByZero).toBe(true)
    expect(decodeFloat('FP32', divide(1, -0).bits).value).toBe(-Infinity)
    expect(
      Object.is(decodeFloat('FP32', divide(-1, Infinity).bits).value, -0),
    ).toBe(true)
    expect(divide(NaN, 1).flags.invalid).toBe(false)
    expect(
      divideForAttention(
        'FP32',
        { format: 'FP32', bits: 0x7f800001 },
        { format: 'FP32', bits: 0x3f800000 },
      ).flags.invalid,
    ).toBe(true)
  })
  it('retains exact ties and rounds third probabilities explicitly', () => {
    const result = divideForAttention(
      'FP16',
      { format: 'FP32', bits: encodeFloat('FP32', 1 + 2 ** -11) },
      { format: 'FP16', bits: 0x3c00 },
    )
    expect(result.halfway).toBe(true)
    expect(result.increment).toBe(false)
    expect(result.bits).toBe(0x3c00)
    const nearNormal = (value: number) =>
      divideForAttention(
        'FP16',
        { format: 'FP32', bits: encodeFloat('FP32', value) },
        { format: 'FP16', bits: 0x3c00 },
      )
    const tinyNormal = nearNormal(2 ** -14 - 2 ** -26 - 2 ** -38)
    expect(tinyNormal.bits).toBe(0x0400)
    expect(tinyNormal.flags.underflow).toBe(true)
    expect(nearNormal(2 ** -14 - 2 ** -26).flags.underflow).toBe(false)

    expect(soft([0, 0, 0], 'FP16').probabilities.map((v) => v.bits)).toEqual([
      0x3555, 0x3555, 0x3555,
    ])
  })
})
describe('independent full-pipeline regression vectors', () => {
  it('replays all 240 independently generated attention calculations in CI', () => {
    const fixture = JSON.parse(
      readFileSync(
        new URL('./reference/attention-vectors.json', import.meta.url),
        'utf8',
      ),
    ) as {
      cases: {
        id: string
        input: AttentionInput
        results: {
          mode: AttentionMode
          method: SoftmaxMethod
          stages: Record<
            'scores' | 'logits' | 'probabilities' | 'output',
            number[][]
          >
          flags: ReturnType<typeof divideForAttention>['flags']
        }[]
      }[]
    }
    let calculations = 0
    for (const sample of fixture.cases)
      for (const expected of sample.results) {
        const actual = calculateAttention(
          sample.input,
          expected.mode,
          expected.method,
        )
        for (const name of [
          'scores',
          'logits',
          'probabilities',
          'output',
        ] as const) {
          const expectedFormat =
            name === 'probabilities' || name === 'output'
              ? ATTENTION_POLICIES[expected.mode].input
              : ATTENTION_POLICIES[expected.mode].scale
          for (let i = 0; i < expected.stages[name].length; i++)
            for (let j = 0; j < expected.stages[name][i].length; j++) {
              const bits = expected.stages[name][i][j],
                value = actual[name][i][j]
              strictEqual(value.format, expectedFormat)
              if (refDecode(expectedFormat, bits).kind === 'nan')
                strictEqual(
                  decodeFloat(value.format, value.bits).isQuietNaN,
                  true,
                )
              else
                strictEqual(
                  value.bits,
                  bits,
                  `${sample.id} ${expected.mode} ${expected.method} ${name} (${i},${j})`,
                )
            }
        }
        deepStrictEqual(actual.flags, expected.flags)
        calculations++
      }
    strictEqual(calculations, 240)
  }, 30000)
})
describe('bounded attention and softmax policies', () => {
  for (const example of ATTENTION_EXAMPLES)
    it(`experiment: ${example.title}`, () => {
      for (const check of example.checks) {
        const result = calculateAttention(
            example.input,
            check.mode,
            check.method,
          ),
          value = attentionValues(result[check.stage])[check.row][check.column]
        if (check.relation === 'equal') expect(value).toBe(check.value)
        else if (check.relation === 'nan') expect(value).toBeNaN()
        else {
          expect(value).toBeGreaterThan(0)
          expect(Number.isFinite(value)).toBe(true)
        }
      }
    })
  it('enforces dimensions, rectangular shapes, number inputs, and supported configurations', () => {
    for (const input of [
      { ...simple, q: [] },
      { ...simple, q: [[1, 2]] },
      { ...simple, v: [[1]] },
      { ...simple, q: [[1], [1, 2]] },
      { ...simple, k: Array(9).fill([1]) },
      {
        ...simple,
        v: [
          [1, 2, 3, 4, 5],
          [0, 0, 0, 0, 0],
        ],
      },
      { ...simple, q: [Array(9).fill(1)] },
      { ...simple, q: Array(5).fill([1]) },
    ])
      expect(() => validateAttentionInput(input)).toThrow()
    expect(() =>
      validateAttentionInput({
        ...simple,
        q: [['1']],
      } as unknown as AttentionInput),
    ).toThrow()
    expect(() => validateAttentionInput({ ...simple, q: Array(1) })).toThrow()
    expect(() => validateAttentionInput({ ...simple, q: [Array(1)] })).toThrow()
    expect(() =>
      calculateAttention(simple, 'unknown' as AttentionMode),
    ).toThrow('Unsupported')
    expect(() =>
      calculateAttention(simple, 'fp32', 'unknown' as SoftmaxMethod),
    ).toThrow('Unsupported')
    expect(() => attentionSoftmax([], 'FP32', 'FP32', 'stable')).toThrow()
    expect(() =>
      attentionSoftmax(
        Array(9).fill({ format: 'FP32', bits: 0 }),
        'FP32',
        'FP32',
        'stable',
      ),
    ).toThrow()
  })
  it('handles binary64 and format exp overflow, exp underflow, and zero denominators', () => {
    expect(soft([1000, 1001], 'FP32', 'naive').issues.join(' ')).toContain(
      'binary64 Math.exp overflow',
    )
    const fp16 = soft([12, 11], 'FP16', 'naive')
    expect(fp16.events.find((e) => e.stage === 'exp')!.flags.overflow).toBe(
      true,
    )
    const overflowedSum = soft([11, 11], 'FP16', 'naive')
    expect(
      overflowedSum.exponentials.every((v) =>
        Number.isFinite(decodeFloat(v.format, v.bits).value),
      ),
    ).toBe(true)
    expect(overflowedSum.issues.join(' ')).toContain(
      'nonfinite normalization denominator',
    )
    expect(
      overflowedSum.probabilities.map(
        (v) => decodeFloat(v.format, v.bits).value,
      ),
    ).toEqual([0, 0])
    const zero = soft([-1000, -1001], 'FP32', 'naive')
    expect(zero.issues.join(' ')).toContain('denominator is zero')
    expect(
      zero.probabilities.every((v) =>
        Number.isNaN(decodeFloat(v.format, v.bits).value),
      ),
    ).toBe(true)
    const tiny = soft([0, -100], 'FP16')
    expect(
      tiny.events.some((e) => e.stage === 'exp' && e.flags.underflow),
    ).toBe(true)
    expect(
      soft([-10000, -10001]).probabilities.every((v) =>
        Number.isFinite(decodeFloat(v.format, v.bits).value),
      ),
    ).toBe(true)
  })
  it('propagates nonfinite stable inputs instead of inventing a probability distribution', () => {
    for (const values of [
      [Infinity, 1],
      [-Infinity, -Infinity],
      [NaN, 0],
    ]) {
      const result = soft(values)
      expect(
        result.probabilities.every((v) =>
          Number.isNaN(decodeFloat(v.format, v.bits).value),
        ),
      ).toBe(true)
      expect(result.issues.join(' ')).toContain('no replacement distribution')
    }
    const valid = soft([0, -Infinity])
    expect(
      valid.probabilities.map((v) => decodeFloat(v.format, v.bits).value),
    ).toEqual([1, 0])
  })
  it('uses max subtraction and separate probability quantization without renormalizing', () => {
    const stable = soft([1000, 999]),
      shifted = soft([0, -1])
    expect(stable.probabilities.map((v) => v.bits)).toEqual(
      shifted.probabilities.map((v) => v.bits),
    )
    const reduced = soft([0, -20], 'FP32', 'stable', 'FP16')
    expect(
      decodeFloat('FP32', reduced.exponentials[1].bits).value,
    ).toBeGreaterThan(0)
    expect(reduced.probabilities[1].bits).toBe(0)
    const thirds = soft([0, 0, 0], 'FP16')
    expect(
      thirds.probabilities.reduce(
        (sum, v) => sum + decodeFloat(v.format, v.bits).value,
        0,
      ),
    ).toBe(0.999755859375)
  })
  it('records actual operands and modeled arithmetic at all computation stages', () => {
    const result = calculateAttention(simple, 'fp16-wide')
    expect(new Set(result.events.map((e) => e.stage))).toEqual(
      new Set([
        'input',
        'score',
        'scale',
        'softmax-input',
        'shift',
        'exp',
        'denominator',
        'normalize',
        'probability',
        'aggregate',
        'output',
      ]),
    )
    expect(
      result.events
        .filter((e) => e.stage === 'score')
        .every((e) => e.operation === 'fma'),
    ).toBe(true)
    expect(
      result.events
        .filter((e) => e.stage === 'aggregate')
        .every((e) => e.operation === 'fma'),
    ).toBe(true)
    expect(result.events.find((e) => e.stage === 'exp')!.before).toBe(1)
    expect(result.policy.probability).toBe('FP16')
    expect(result.policy.softmax).toBe('FP32')
    const reduced = calculateAttention(simple, 'fp16-softmax')
    expect(
      reduced.events
        .filter((e) => e.stage === 'softmax-input')
        .every((e) => e.result.format === 'FP16'),
    ).toBe(true)
    expect(reduced.policy.scale).toBe('FP32')
  })
  it('preserves signed-zero input and final output encodings', () => {
    const result = calculateAttention(
      { q: [[-0]], k: [[1]], v: [[-0]] },
      'fp16',
    )
    expect(result.quantized.q[0][0].bits).toBe(0x8000)
    expect(result.quantized.v[0][0].bits).toBe(0x8000)
    // Sequential +0 seeding in the inherited dot model makes the aggregate +0.
    expect(result.output[0][0].bits).toBe(0)
  })
  it('exercises the maximum supported shapes', () => {
    const input = {
      q: Array.from({ length: 4 }, () => Array(8).fill(1)),
      k: Array.from({ length: 8 }, () => Array(8).fill(1)),
      v: Array.from({ length: 8 }, () => Array(4).fill(1)),
    }
    for (const mode of Object.keys(ATTENTION_POLICIES) as AttentionMode[]) {
      const result = calculateAttention(input, mode)
      expect(result.output).toHaveLength(4)
      expect(result.output[0]).toHaveLength(4)
      expect(attentionValues(result.output).flat()).toEqual(Array(16).fill(1))
    }
  })
  it('checks 250 randomized finite softmax rows and binary64 baseline consistency', () => {
    const random = seededRandom(0x6174746e)
    for (let i = 0; i < 50; i++) {
      const values = Array.from(
        { length: 1 + (random() % 8) },
        () => ((random() % 1025) - 512) / 64,
      )
      for (const mode of Object.values(ATTENTION_POLICIES)) {
        const result = soft(values, mode.softmax, 'stable', mode.probability)
        const sum = result.probabilities.reduce(
          (s, v) => s + decodeFloat(v.format, v.bits).value,
          0,
        )
        const tolerance =
          mode.probability === 'BF16'
            ? 0.012
            : mode.probability === 'FP16'
              ? 0.004
              : 1e-6
        expect(Math.abs(sum - 1)).toBeLessThan(tolerance)
        for (const v of result.probabilities) {
          const value = decodeFloat(v.format, v.bits).value
          expect(value).toBeGreaterThanOrEqual(0)
          expect(value).toBeLessThanOrEqual(1)
        }
      }
    }
    const reference = binary64Attention(simple)
    expect(reference.output[0][0]).toBeCloseTo(2 / (1 + Math.exp(-1)), 14)
    expect(reference.probabilities[0].reduce((a, b) => a + b, 0)).toBeCloseTo(
      1,
      15,
    )
  })
  it('parses matrices atomically with explicit special-value conventions', () => {
    expect(parseAttentionMatrices('1, 2; 3 4', '1 0\n0 1', '1;2').q).toEqual([
      [1, 2],
      [3, 4],
    ])
    expect(Object.is(parseAttentionMatrix('-0 Infinity NaN')[0][0], -0)).toBe(
      true,
    )
    expect(() => parseAttentionMatrices('bad', '1', '1')).toThrow()
    expect(() => parseAttentionMatrices('1', '1;2', '1')).toThrow()
  })
})
