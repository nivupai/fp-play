// @vitest-environment node
/// <reference types="node" />
import { createHash } from 'node:crypto'
import { readFileSync, existsSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import artifact from '../../rtl/vectors/hardware-analysis.json'
import type { HardwareArtifact } from '../lib/hardware/types'
const data: HardwareArtifact = artifact
describe('real synthesis artifact provenance', () => {
  it('binds every analysis input to current source and preserves baseline HDL', () => {
    for (const [path, hash] of Object.entries(data.sourceHashes))
      expect(
        createHash('sha256')
          .update(readFileSync(new URL(`../../${path}`, import.meta.url)))
          .digest('hex'),
        path,
      ).toBe(hash)
    expect(data.sourceHashes['rtl/src/fp16_addsub.sv']).toBe(
      '29e2de0c304d637dacb171b8e0d41b89df66a13b3524777c6fcd5577fdc66e12',
    )
    expect(data.sourceHashes['rtl/src/fp16_round_grid48.sv']).toBe(
      '65b67987562ec5cac6a4a6577726504574b81fcb9b15c75495f4ba7cb2aaed9b',
    )
  })
  it('uses one controlled configuration and successful verification for seven real runs', () => {
    expect(data.records.map((r) => r.id)).toEqual([
      'baseline-add',
      'baseline-mul',
      'baseline-fma',
      'baseline-nonfused',
      'tree-leading-bit-mul',
      'tree-leading-bit-fma',
      'tree-leading-bit-nonfused',
    ])
    expect(data.tools.yosys).toContain('5053d0e26')
    expect(data.config.library.sha256).toBe(
      '8d540a4d4cf6d09d27c87ad067857a9c0c2eeb023ab7a56e058cd3113db4e9b1',
    )
    for (const r of data.records) {
      expect(r.mapped.area).toBeGreaterThan(0)
      expect(r.mapped.cells).toBeGreaterThan(0)
      expect(r.mapped.sequentialCells).toBe(0)
      expect(r.io).toEqual({
        inputs: r.unit === 'add' ? 33 : r.unit === 'mul' ? 32 : 48,
        outputs: 20,
      })
      expect(r.verification).toMatchObject({
        cases: r.unit === 'add' ? 462172 : 301544,
        mismatches: 0,
        proof: false,
      })
      expect(r.verification.boundaryVerification).toMatchObject({
        cases: r.unit === 'add' ? 462172 : 301544,
        mismatches: 0,
        proof: false,
      })
      expect(r.depthPath).toHaveLength(r.mapped.logicDepth)
      if (r.timing) {
        expect(r.timing.delayNs).toBeGreaterThan(0)
        expect(r.timing.startpoint).toBeTruthy()
        expect(r.timing.endpoint).toBeTruthy()
        expect(r.timing.cells.length).toBeGreaterThan(0)
      } else expect(r.timingUnavailable).toBeTruthy()
    }
  })
  it('checks raw report integrity when local reproduction outputs exist', () => {
    // Raw reports/netlists stay out of the browser/repository; CI can validate the portable snapshot alone.
    for (const r of artifact.records)
      for (const [key, path] of Object.entries(r.reports)) {
        const file = new URL(`../../${path}`, import.meta.url)
        if (existsSync(file))
          expect(
            createHash('sha256').update(readFileSync(file)).digest('hex'),
          ).toBe(r.reportHashes[key as keyof typeof r.reportHashes])
      }
  })
})
