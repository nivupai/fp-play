// @vitest-environment node
/// <reference types="node" />
import { createHash } from 'node:crypto'
import { readFileSync, existsSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import authoritative from '../../rtl/vectors/hardware-analysis.json'
import {
  hardwareEvidence,
  fusedComparison,
  numericalExamples,
} from '../lib/showcase/evidence'
const read = (path: string) =>
  readFileSync(new URL('../../' + path, import.meta.url))
describe('showcase evidence and release preparation', () => {
  it('binds the compact summary to the actual authoritative snapshot and all measurement conditions', () => {
    const source = hardwareEvidence.source
    expect(source.sha256).toBe(
      createHash('sha256').update(read(source.path)).digest('hex'),
    )
    expect(source.config).toEqual(authoritative.config)
    expect(source.tools).toEqual(authoritative.tools)
    expect(source.sourceHashes).toEqual(authoritative.sourceHashes)
    expect(source.gitCommit).toBe(authoritative.gitCommit)
    expect(source.gitDirty).toBe(true)
    expect(source.timestamp).toBe(authoritative.timestamp)
  })
  it('preserves every measured number, path endpoint and verification record without interpolating results', () => {
    expect(hardwareEvidence.records).toHaveLength(7)
    for (const actual of authoritative.records) {
      const r = hardwareEvidence.records.find((r) => r.id === actual.id)!
      expect(r, actual.id).toMatchObject({
        unit: actual.unit,
        variant: actual.variant,
        area: actual.mapped.area,
        cells: actual.mapped.cells,
        depth: actual.mapped.logicDepth,
        delay: actual.timing.delayNs,
        startpoint: actual.timing.startpoint,
        endpoint: actual.timing.endpoint,
        pathCells: actual.timing.cells.length,
        verification: actual.verification,
      })
      expect(r.stageBreakdownUnavailable).toBe(
        actual.timing.stageBreakdownUnavailable,
      )
    }
  })
  it('computes contextual FMA reductions relative to nonfused, never latency or frequency', () => {
    const comparison = fusedComparison()!
    expect(comparison.areaReduction).toBeCloseTo(
      (100 * (3981.754 - 3588.074)) / 3981.754,
      10,
    )
    expect(comparison.delayReduction).toBeCloseTo(
      (100 * (5.652442 - 4.179512)) / 5.652442,
      10,
    )
    expect(comparison.fused.id).toBe('baseline-fma')
    expect(comparison.nonfused.id).toBe('baseline-nonfused')
    expect(fusedComparison([])).toBeNull()
    const missing = structuredClone(hardwareEvidence.records)
    missing.find((r) => r.id === 'baseline-fma')!.delay = null
    expect(fusedComparison(missing)?.delayReduction).toBeNull()
  })
  it('checks displayed FP16 and FMA examples using independently specified encodings', () => {
    expect(numericalExamples.conversion).toEqual({
      original: '1.0001',
      bits: 0x3c00,
      value: 1,
    })
    expect(numericalExamples.fma.fused.bits).toBe(0x0010)
    expect(numericalExamples.fma.fused.exact).toMatchObject({
      coefficient: 1n,
      exponent: -20,
    })
    expect(numericalExamples.fma.nonfused.bits).toBe(0x0000)
    expect(numericalExamples.fma.nonfused.product.bits).toBe(0x3c02)
  })
  it('keeps the current release candidate private and synchronized with the lockfile', () => {
    const p = JSON.parse(read('package.json').toString()),
      lock = JSON.parse(read('package-lock.json').toString())
    expect(p.version).toBe('0.2.0-rc.1')
    expect(p.private).toBe(true)
    expect(lock.version).toBe(p.version)
    expect(lock.packages[''].version).toBe(p.version)
    const notices = read('public/THIRD_PARTY_NOTICES.txt').toString()
    expect(notices).toContain('Meta Platforms')
    expect(notices).toContain('Tailwind Labs')
    expect(notices).toContain('MIT License')
  })
  it('keeps case-study and README measurements tied to source values', () => {
    for (const path of ['README.md', 'docs/CASE_STUDY.md']) {
      const doc = read(path).toString()
      for (const r of authoritative.records.filter(
        (r) => r.variant === 'baseline',
      )) {
        expect(doc, `${path}: ${r.id}`).toContain(String(r.mapped.area))
        expect(doc, `${path}: ${r.id}`).toContain(r.timing.delayNs.toFixed(6))
      }
    }
    const study = read('docs/CASE_STUDY.md').toString()
    for (const r of authoritative.records)
      expect(study).toContain(r.timing.delayNs.toFixed(6))
    expect(study.match(/^## \d+\./gm)).toHaveLength(10)
  })
  it('checks local documentation navigation targets including generated notices', () => {
    for (const path of [
      'README.md',
      'CONTRIBUTING.md',
      'docs/README.md',
      'docs/CASE_STUDY.md',
      'docs/REPRODUCIBILITY.md',
      'docs/RELEASING.md',
      'docs/THIRD_PARTY.md',
    ]) {
      const url = new URL('../../' + path, import.meta.url)
      for (const match of readFileSync(url, 'utf8').matchAll(
        /\]\(([^)]+)\)/g,
      )) {
        const target = match[1].split('#')[0]
        if (target && !/^(?:https?:|#)/.test(target))
          expect(existsSync(new URL(target, url)), `${path}: ${target}`).toBe(
            true,
          )
      }
    }
  })
})
