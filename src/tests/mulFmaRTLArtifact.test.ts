// @vitest-environment node
/// <reference types="node" />
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import artifact from '../../rtl/vectors/mul-fma-simulation.json'
import {
  multiplyFP16,
  fmaFP16,
  nonfusedMultiplyAddFP16,
} from '../lib/numerics/fp16Multiply'
import { referenceMultiply } from './reference/fp16MultiplyReference'
const mask = (f: ReturnType<typeof multiplyFP16>['flags']) =>
  (f.invalid ? 8 : 0) |
  (f.overflow ? 4 : 0) |
  (f.underflow ? 2 : 0) |
  (f.inexact ? 1 : 0)
describe('genuine multiplier/FMA RTL records', () => {
  it('pins sources and the complete deterministic corpus', () => {
    for (const [path, hash] of Object.entries(artifact.metadata.sourceHashes))
      expect(
        createHash('sha256')
          .update(readFileSync(new URL(`../../${path}`, import.meta.url)))
          .digest('hex'),
        path,
      ).toBe(hash)
    expect(artifact.metadata.cases).toBe(301544)
    expect(artifact.metadata.counts).toEqual({
      directed: 23,
      exhaustive: 196608,
      special: 4913,
      randomized: 100000,
    })
    expect(artifact.metadata.mismatches).toBe(0)
  })
  for (const sample of artifact.samples)
    it(sample.name, () => {
      const m = multiplyFP16(sample.a, sample.b),
        f = fmaFP16(sample.a, sample.b, sample.c),
        n = nonfusedMultiplyAddFP16(sample.a, sample.b, sample.c)
      expect([sample.multiply, sample.fused, sample.nonfused]).toEqual([
        m.bits,
        f.bits,
        n.bits,
      ])
      expect([
        sample.multiplyFlags,
        sample.fusedFlags,
        sample.nonfusedFlags,
        sample.additionFlags,
      ]).toEqual([
        mask(m.flags),
        mask(f.flags),
        mask(n.flags),
        mask(n.addition.flags),
      ])
      for (const [result, independent, signals] of [
        [m, referenceMultiply(sample.a, sample.b), sample.multiplySignals],
        [f, referenceMultiply(sample.a, sample.b, sample.c), sample.fmaSignals],
      ] as const) {
        expect(result.bits).toBe(independent.bits)
        expect(result.flags).toEqual(independent.flags)
        if (independent.units !== null) {
          expect(BigInt(signals.arithmeticMagnitude)).toBe(
            independent.units < 0n ? -independent.units : independent.units,
          )
          const grs = result.trace.find((s) => s.kind === 'grs'),
            round = result.trace.find((s) => s.kind === 'round')
          expect([signals.guard, signals.round, signals.sticky]).toEqual([
            grs?.guard,
            grs?.round,
            grs?.sticky,
          ])
          expect(!!signals.increment).toBe(round?.increment)
          expect(!!signals.tiny).toBe(round?.tinyAfterPrecisionRounding)
        } else expect(signals.special).toBe(1)
      }
    })
  it('retains product residuals and avoids intermediate overflow', () => {
    expect(
      artifact.samples.find((s) => s.name === 'Lost product residual'),
    ).toMatchObject({ fused: 16, nonfused: 0, fusedFlags: 0, nonfusedFlags: 1 })
    expect(
      artifact.samples.find((s) => s.name === 'Avoid intermediate overflow'),
    ).toMatchObject({
      fused: 0x7bff,
      nonfused: 0x7c00,
      fusedFlags: 0,
      nonfusedFlags: 5,
    })
  })
})
