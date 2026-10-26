// @vitest-environment node
/// <reference types="node" />
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  compareRecordedRTL,
  RTL_METADATA,
  RTL_SIMULATIONS,
} from '../lib/arithmetic/rtl'
import { calculateFP16 } from '../lib/numerics/fp16Arithmetic'

describe('real RTL artifact provenance and numerical consistency', () => {
  it('pins every recorded source hash to the current files', () => {
    for (const [path, hash] of Object.entries(RTL_METADATA.sourceHashes)) {
      const source = readFileSync(new URL(`../../rtl/${path}`, import.meta.url))
      expect(createHash('sha256').update(source).digest('hex'), path).toBe(hash)
    }
    expect(RTL_METADATA.cases).toBe(462172)
    expect(RTL_METADATA.mismatches).toBe(0)
    expect(RTL_METADATA.signalComparisons).toBe(441888)
    expect(RTL_METADATA.counts).toEqual({
      directed: 28,
      exhaustive: 262144,
      randomized: 200000,
    })
  })
  it('checks every recorded result, flag, NaN payload/sign, and finite GRS decision against the actual engine', () => {
    expect(RTL_SIMULATIONS).toHaveLength(28)
    for (const sample of RTL_SIMULATIONS) {
      const comparison = compareRecordedRTL(sample)
      expect(comparison.agrees, sample.name).toBe(true)
      expect(sample.softwareBits).toBe(comparison.software.bits)
      expect(sample.softwareFlags).toEqual(comparison.software.flags)
      const ref = calculateFP16(sample.a, sample.b, sample.operation)
      const grs = ref.trace.find((s) => s.kind === 'grs'),
        round = ref.trace.find((s) => s.kind === 'round')
      if (grs && round) {
        expect([
          sample.signals.guard,
          sample.signals.round,
          sample.signals.sticky,
        ]).toEqual([grs.guard, grs.round, grs.sticky])
        expect(Boolean(sample.signals.increment)).toBe(round.increment)
        const magnitude = BigInt(`0x${sample.signals.magnitude}`)
        expect(magnitude).toBe(
          (ref.exact!.coefficient < 0n
            ? -ref.exact!.coefficient
            : ref.exact!.coefficient) << BigInt(ref.exact!.exponent + 24),
        )
      } else expect(sample.signals.special).toBe(true)
    }
  })
})
