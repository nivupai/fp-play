// @vitest-environment node
/// <reference types="node" />
import { strictEqual, deepStrictEqual } from 'node:assert'
import { describe, expect, it } from 'vitest'
import {
  decodeFP16,
  floatToFP16Bits,
  nextDownFP16,
  nextUpFP16,
  ulpFP16,
} from '../lib/numerics/fp16'
import {
  decodeFloat,
  encodeFloat,
  floatBinary,
  floatHex,
  floatULP,
  FORMATS,
  nextDown,
  nextUp,
  roundDyadic,
} from '../lib/numerics/formats'
import type { FormatName } from '../lib/numerics/formats'
import {
  addFloat,
  multiplyFloat,
  calculateFloat,
} from '../lib/numerics/formatArithmetic'
import {
  refArithmetic,
  refDecode,
  refNumber,
  refRound,
  seededRandom,
} from './reference/formatReference'

describe('typed formats and direct conversion', () => {
  it('exhausts all BF16 raw patterns against the independent lattice', () => {
    for (let bits = 0; bits < 65536; bits++) {
      const d = decodeFloat('BF16', bits),
        r = refDecode('BF16', bits)
      strictEqual(d.classification, r.kind)
      strictEqual(d.sign, r.sign)
      strictEqual(d.exponent, r.exponent)
      strictEqual(d.fraction, r.fraction)
      strictEqual(d.isQuietNaN, r.kind === 'nan' ? r.quiet : null)
      strictEqual(d.nanPayload, r.kind === 'nan' ? r.fraction % 64 : null)
      if (r.units !== null) {
        strictEqual(refNumber(d.value), r.units)
        strictEqual(encodeFloat('BF16', d.value), bits)
      } else
        strictEqual(
          encodeFloat('BF16', d.value),
          r.kind === 'nan' ? 0x7fc0 : bits,
        )
    }
  }, 30000)
  it('preserves the complete historical FP16 API semantics', () => {
    for (let bits = 0; bits < 65536; bits++) {
      const d = decodeFloat('FP16', bits),
        old = decodeFP16(bits)
      for (const key of [
        'sign',
        'exponent',
        'fraction',
        'unbiasedExponent',
        'classification',
        'value',
        'hasImplicitLeadingBit',
        'isQuietNaN',
        'nanPayload',
      ] as const)
        strictEqual(d[key], old[key])
      strictEqual(encodeFloat('FP16', d.value), floatToFP16Bits(old.value))
      strictEqual(nextUp('FP16', bits), nextUpFP16(bits))
      strictEqual(nextDown('FP16', bits), nextDownFP16(bits))
      strictEqual(floatULP('FP16', bits), ulpFP16(bits))
    }
  }, 30000)
  it('checks every positive finite BF16 midpoint and its binary64 neighbors', () => {
    const view = new DataView(new ArrayBuffer(8))
    for (let bits = 0; bits < 0x7f80; bits++) {
      const lo = refDecode('BF16', bits).units!,
        hi = bits === 0x7f7f ? 1n << 2276n : refDecode('BF16', bits + 1).units!
      // Midpoints fit binary64 exactly, including virtual overflow neighbor.
      const midpoint =
        bits === 0x7f7f
          ? (2 - 2 ** -8) * 2 ** 127
          : (decodeFloat('BF16', bits).value +
              decodeFloat('BF16', bits + 1).value) /
            2
      view.setFloat64(0, midpoint)
      const raw = view.getBigUint64(0)
      for (const delta of [-1n, 0n, 1n]) {
        view.setBigUint64(0, raw + delta)
        const value = view.getFloat64(0)
        strictEqual(
          encodeFloat('BF16', value),
          refRound('BF16', refNumber(value)!).bits,
        )
        strictEqual(
          encodeFloat('BF16', -value),
          refRound('BF16', -refNumber(value)!).bits,
        )
      }
      strictEqual(refNumber(midpoint), (lo + hi) / 2n)
    }
  }, 30000)
  it('avoids BF16 double rounding on either side of a midpoint', () => {
    const above = 1 + 2 ** -8 + 2 ** -30,
      belowOdd = 1 + 3 * 2 ** -8 - 2 ** -30
    expect(encodeFloat('BF16', above)).toBe(0x3f81)
    expect(encodeFloat('BF16', Math.fround(above))).toBe(0x3f80)
    expect(encodeFloat('BF16', belowOdd)).toBe(0x3f81)
    expect(encodeFloat('BF16', Math.fround(belowOdd))).toBe(0x3f82)
  })
  it('matches native FP32 conversion on 20000 deterministic binary64 patterns', () => {
    const rand = seededRandom(),
      source = new DataView(new ArrayBuffer(8)),
      target = new DataView(new ArrayBuffer(4))
    for (let i = 0; i < 20000; i++) {
      source.setUint32(0, rand())
      source.setUint32(4, rand())
      const value = source.getFloat64(0)
      target.setFloat32(0, value)
      strictEqual(
        encodeFloat('FP32', value),
        Number.isNaN(value) ? 0x7fc00000 : target.getUint32(0),
      )
    }
  }, 30000)
  it('checks FP32 exponent boundaries and randomized raw decoding against native DataView', () => {
    const view = new DataView(new ArrayBuffer(4)),
      random = seededRandom()
    for (let i = 0; i < 20000; i++) {
      const bits = random()
      view.setUint32(0, bits)
      strictEqual(decodeFloat('FP32', bits).value, view.getFloat32(0))
      strictEqual(
        encodeFloat('FP32', view.getFloat32(0)),
        refDecode('FP32', bits).kind === 'nan' ? 0x7fc00000 : bits,
      )
    }
    for (let exponent = -126; exponent <= 127; exponent++) {
      const boundary = 2 ** exponent
      for (const value of [
        boundary,
        boundary - 2 ** (exponent - 25),
        boundary - 2 ** (exponent - 24),
        boundary + 2 ** (exponent - 24),
        boundary + 3 * 2 ** (exponent - 24),
      ]) {
        for (const sign of [-1, 1]) {
          view.setFloat32(0, sign * value)
          strictEqual(encodeFloat('FP32', sign * value), view.getUint32(0))
          strictEqual(
            encodeFloat('FP32', sign * value),
            refRound('FP32', refNumber(sign * value)!).bits,
          )
        }
      }
    }
  }, 30000)
  for (const format of Object.keys(FORMATS) as FormatName[]) {
    const f = FORMATS[format],
      min = 2 ** (1 - f.bias - f.fractionBits),
      inf = (2 ** f.exponentBits - 1) * 2 ** f.fractionBits,
      sign = 2 ** (f.width - 1)
    it(`${format}: zeros, subnormals, ties, exponent boundaries and overflow`, () => {
      for (const value of [
        0,
        -0,
        min / 2,
        -min / 2,
        min,
        min * 1.5,
        2 ** (1 - f.bias),
        1,
        1 + 2 ** (-f.fractionBits - 1),
        2 - 2 ** (-f.fractionBits - 1),
        (2 - 2 ** (-f.fractionBits - 1)) * 2 ** f.bias,
      ])
        strictEqual(
          encodeFloat(format, value),
          refRound(format, refNumber(value)!, Object.is(value, -0)).bits,
        )
      expect(Object.is(decodeFloat(format, sign).value, -0)).toBe(true)
      expect(nextUp(format, sign)).toBe(1)
      expect(nextDown(format, 0)).toBe(sign + 1)
      expect(nextUp(format, inf - 1)).toBe(inf)
      expect(nextDown(format, inf)).toBe(inf - 1)
      expect(nextDown(format, sign + inf)).toBe(sign + inf)
      expect(nextUp(format, sign + inf)).toBe(sign + inf - 1)
      expect(nextUp(format, inf + 1)).toBe(inf + 1)
      expect(nextDown(format, inf + 1)).toBe(inf + 1)
      expect(floatULP(format, 0)).toBe(min)
      expect(floatULP(format, encodeFloat(format, 1))).toBe(
        2 ** -f.fractionBits,
      )
      expect(floatULP(format, inf)).toBe(Infinity)
      expect(floatULP(format, inf + 1)).toBeNaN()
      expect(floatBinary(format, 0)).toHaveLength(f.width)
      expect(floatHex(format, 0)).toHaveLength(f.width / 4 + 2)
      for (const bad of [-1, 0.5, 2 ** f.width, NaN])
        expect(() => decodeFloat(format, bad)).toThrow(RangeError)
    })
    it(`${format}: underflow at a rounded normal follows precision-only tininess`, () => {
      const boundaryExponent = 1 - f.bias - f.fractionBits - 2
      const threshold = (1n << BigInt(f.fractionBits + 2)) - 1n
      expect(
        roundDyadic(format, {
          coefficient: threshold * 2n - 1n,
          exponent: boundaryExponent - 1,
        }).flags.underflow,
      ).toBe(true)
      expect(
        roundDyadic(format, {
          coefficient: threshold,
          exponent: boundaryExponent,
        }).flags.underflow,
      ).toBe(false)
      expect(
        roundDyadic(format, {
          coefficient: 1n,
          exponent: 1 - f.bias - f.fractionBits,
        }).flags.underflow,
      ).toBe(false)
    })
  }
})
describe('exact multi-format arithmetic', () => {
  for (const format of ['BF16', 'FP32'] as const) {
    it(`${format}: 12000 finite arithmetic results and flags against integer-distance oracle`, () => {
      const rand = seededRandom(),
        mask = format === 'BF16' ? 65536 : 2 ** 32
      for (let i = 0; i < 6000; i++) {
        let a = rand() % mask,
          b = rand() % mask
        while (refDecode(format, a).units === null) a = rand() % mask
        while (refDecode(format, b).units === null) b = rand() % mask
        for (const op of ['add', 'multiply'] as const) {
          const actual =
            op === 'add' ? addFloat(format, a, b) : multiplyFloat(format, a, b)
          const expected = refArithmetic(format, op, a, b)
          strictEqual(
            actual.bits,
            expected.bits,
            `${format} ${op} ${a.toString(16)} ${b.toString(16)}`,
          )
          deepStrictEqual(actual.flags, expected.flags)
        }
      }
    }, 30000)
    it(`${format}: signed zeros, infinities, and signaling NaN policy`, () => {
      const f = FORMATS[format],
        sign = 2 ** (f.width - 1),
        inf = (2 ** f.exponentBits - 1) * 2 ** f.fractionBits,
        quiet = 2 ** (f.fractionBits - 1)
      expect(addFloat(format, sign, sign).bits).toBe(sign)
      expect(addFloat(format, sign, 0).bits).toBe(0)
      expect(multiplyFloat(format, sign, encodeFloat(format, 1)).bits).toBe(
        sign,
      )
      expect(addFloat(format, inf, sign + inf).flags.invalid).toBe(true)
      expect(multiplyFloat(format, 0, inf).flags.invalid).toBe(true)
      const result = addFloat(format, inf + quiet + 3, sign + inf + 5)
      expect(result.bits).toBe(sign + inf + quiet + 5)
      expect(result.flags.invalid).toBe(true)
      expect(addFloat(format, inf + quiet + 3, 0).flags.invalid).toBe(false)
      expect(
        calculateFloat(
          format,
          'fma',
          { format, bits: 0 },
          { format, bits: inf },
          { format, bits: inf + quiet },
        ).flags.invalid,
      ).toBe(true)
      expect(() =>
        calculateFloat(format, 'fma', { format, bits: 0 }, { format, bits: 0 }),
      ).toThrow()
    })
  }
})
