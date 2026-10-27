// @vitest-environment node
/// <reference types="node" />
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import add from '../../rtl/vectors/simulation.json'
import product from '../../rtl/vectors/mul-fma-simulation.json'
import synthesis from '../../rtl/vectors/hardware-analysis.json'
import waves from '../../rtl/vectors/datapath-waveforms.json'
import { loadCatalog } from '../lib/datapath/catalog'
import {
  parseSimulationArtifact,
  signalBinary,
} from '../lib/datapath/artifacts'
import {
  compareExample,
  exampleLabel,
  exactReference,
  explainGRS,
} from '../lib/datapath/reference'
import { linkSynthesis, parseSynthesis } from '../lib/datapath/synthesis'
import { curateVCD, validateWaveData } from '../lib/datapath/vcd'
import { parseWaves, validateCorrelation, valueAt } from '../lib/datapath/waves'
const examples = [
  ...parseSimulationArtifact(add, 'add'),
  ...parseSimulationArtifact(product, 'product'),
]
describe('recorded datapath artifacts and numerical consistency', () => {
  it('parses every actual example without inventing nonfused adder observations', () => {
    expect(examples).toHaveLength(97)
    expect(examples.filter((r) => r.operation === 'nonfused')).toHaveLength(23)
    for (const r of examples.filter((r) => r.operation === 'nonfused'))
      expect(r.phases[1]).toMatchObject({ signals: {}, bypassed: null })
  })
  it('compares all raw bits, flags and available finite signals against the actual exact engine', () => {
    let checks = 0
    for (const record of examples) {
      const c = compareExample(record)
      expect(c.agrees, record.id).toBe(true)
      for (const p of c.phases) {
        expect(p.signalMismatches, record.id).toEqual([])
        checks += p.signalChecks
      }
    }
    expect(checks).toBe(768)
  })
  it('preserves signed zero and quiet/signaling NaN payload policy through raw patterns', () => {
    const zero = examples.find(
        (r) => r.operation === 'add' && r.name === 'Two negative zeros',
      )!,
      nan = examples.find(
        (r) =>
          r.operation === 'fma' && r.name === 'Signaling C outranks quiet A',
      )!
    expect(compareExample(zero).result.bits).toBe(0x8000)
    expect(exactReference(compareExample(zero).phases[0].software)).toContain(
      '-0',
    )
    expect(compareExample(nan).result.bits).toBe(0xfe11)
    expect(nan.flags.invalid).toBe(true)
    expect(explainGRS(nan.phases[0]).status).toBe('bypassed')
  })
  it('explains independently specified even/odd ties and retains historical title provenance', () => {
    const odd = examples.find((r) => r.id === 'product:20:multiply')!,
      even = examples.find((r) => r.id === 'product:21:multiply')!
    expect(odd.bits).toBe(0x3e02)
    expect(explainGRS(odd.phases[0])).toMatchObject({
      status: 'available',
      halfway: true,
      lsb: 1,
      increment: 1,
      agrees: true,
    })
    expect(even.bits).toBe(0x3e04)
    expect(explainGRS(even.phases[0])).toMatchObject({
      halfway: true,
      lsb: 0,
      increment: 0,
      agrees: true,
    })
    expect(exampleLabel(odd)).toBe(
      'Product midpoint · retained odd · increment',
    )
    expect(odd.name).toBe('Even midpoint rounds down')
  })
  it('explains carry, cancellation, gradual underflow and subnormal transitions from real records', () => {
    const carry = examples.find((r) => r.name === 'Significand carry')!,
      cancel = examples.find((r) => r.name === 'Cancellation')!,
      tiny = examples.find((r) => r.id === 'product:4:multiply')!,
      transition = examples.find(
        (r) => r.name === 'Smallest normal transition',
      )!
    expect(explainGRS(carry.phases[0])).toMatchObject({ inferredCarry: true })
    expect(cancel.bits).toBe(0x1400)
    expect(
      examples.find((r) => r.name === 'Signed zero cancellation')!.bits,
    ).toBe(0)
    expect(tiny.bits).toBe(0)
    expect(tiny.flags).toMatchObject({ underflow: true, inexact: true })
    expect(transition.bits).toBe(0x400)
  })
  it('preserves true FMA versus separately rounded results and exception unions', () => {
    const fma = examples.find((r) => r.id === 'product:1:fma')!,
      nf = examples.find((r) => r.id === 'product:1:nonfused')!,
      overflow = examples.find((r) => r.id === 'product:2:nonfused')!
    expect(fma.bits).toBe(0x10)
    expect(nf.bits).toBe(0)
    expect(nf.phases[0].bits).toBe(0x3c02)
    expect(exactReference(compareExample(fma).phases[0].software)).toContain(
      '0.00000095367431640625',
    )
    expect(overflow.flags).toMatchObject({ overflow: true, inexact: true })
  })
  it('detects corrupted result, flags and available GRS observations without quiet NaN exemptions', () => {
    const r = structuredClone(examples[0])
    r.bits ^= 1
    expect(compareExample(r).agrees).toBe(false)
    const f = structuredClone(examples[0])
    f.flags.invalid = true
    expect(compareExample(f).flagsAgree).toBe(false)
    const g = structuredClone(examples[1])
    g.phases[0].signals.guard.value = 0n
    expect(compareExample(g).phases[0].signalMismatches).toContain(
      'FP16 addition: guard',
    )
    expect(compareExample(g).agrees).toBe(false)
  })
  it('permits missing observations but rejects invalid schemas, widths, masks and hashes', () => {
    const missing = structuredClone(product) as unknown as {
      samples: Record<string, unknown>[]
    }
    delete missing.samples[0].fmaSignals
    const parsed = parseSimulationArtifact(missing, 'product').find(
      (r) => r.operation === 'fma',
    )!
    expect(explainGRS(parsed.phases[0]).status).toBe('missing')
    expect(() =>
      parseSimulationArtifact(
        { ...add, metadata: { ...add.metadata, mismatches: 1 } },
        'add',
      ),
    ).toThrow(/verified/)
    expect(() =>
      parseSimulationArtifact(
        { ...add, samples: [{ ...add.samples[0], a: 65536 }] },
        'add',
      ),
    ).toThrow(/raw FP16/)
    expect(() =>
      parseSimulationArtifact(
        { ...product, samples: [{ ...product.samples[0], fusedFlags: 16 }] },
        'product',
      ),
    ).toThrow(/mask/)
    expect(() =>
      parseSimulationArtifact(
        {
          ...product,
          samples: [
            {
              ...product.samples[0],
              fmaSignals: {
                ...product.samples[0].fmaSignals,
                arithmeticMagnitude: '0x' + 'f'.repeat(22),
              },
            },
          ],
        },
        'product',
      ),
    ).toThrow(/range/)
    expect(() =>
      parseSimulationArtifact(
        {
          ...add,
          metadata: { ...add.metadata, sourceHashes: { bad: 'nope' } },
        },
        'add',
      ),
    ).toThrow(/hash/)
  })
  it('keeps wide signals exact with BigInt and rejects unsupported operation kinds', () => {
    const record = examples.find((r) => r.id === 'product:14:fma')!,
      s = record.phases[0].signals.arithmeticMagnitude
    expect(s.value).toBeGreaterThan(BigInt(Number.MAX_SAFE_INTEGER))
    expect(signalBinary(s)).toHaveLength(81)
    expect(() =>
      parseSimulationArtifact(
        { ...add, samples: [{ ...add.samples[0], operation: 'divide' }] },
        'add',
      ),
    ).toThrow(/operation/)
  })
  it('links metrics only to matching RTL source revisions and retains complete constraints', () => {
    const context = parseSynthesis(synthesis),
      fma = examples.find((r) => r.operation === 'fma')!
    expect(linkSynthesis(fma, context)).toMatchObject({
      compatible: true,
      unit: 'fma',
    })
    expect(linkSynthesis(fma, context).records).toHaveLength(2)
    expect(context.config.timing).toMatchObject({
      inputTransitionNs: 0.05,
      outputLoadFf: 10,
      virtualClockPeriodNs: 100,
    })
    expect(
      linkSynthesis(fma, { ...context, sourceHashes: {} }).compatible,
    ).toBe(false)
    expect(() =>
      parseSynthesis({
        ...synthesis,
        records: [
          {
            ...synthesis.records[0],
            mapped: { ...synthesis.records[0].mapped, area: NaN },
          },
        ],
      }),
    ).toThrow(/area/)
  })
  it('isolates invalid optional data while refusing corrupt mandatory hardware artifacts', () => {
    expect(loadCatalog({ add: null }).ok).toBe(false)
    const result = loadCatalog({ synthesis: null, waves: null })
    expect(result.ok).toBe(true)
    if (result.ok) {
      expect(result.catalog.examples).toHaveLength(97)
      expect(result.catalog.synthesis).toBeNull()
      expect(result.catalog.waveError).toBeTruthy()
    }
  })
  it('checks all source hashes for simulation and exported VCD provenance', () => {
    for (const example of examples)
      for (const [path, hash] of Object.entries(
        example.provenance.sourceHashes,
      ))
        expect(
          createHash('sha256')
            .update(readFileSync(new URL(`../../${path}`, import.meta.url)))
            .digest('hex'),
        ).toBe(hash)
    for (const group of waves.groups)
      for (const [path, hash] of Object.entries(group.sourceHashes))
        expect(
          createHash('sha256')
            .update(readFileSync(new URL(`../../${path}`, import.meta.url)))
            .digest('hex'),
        ).toBe(hash)
  })
})
describe('curated actual VCD records', () => {
  it('correlates every example with real raw VCD values including held-state timestamps', () => {
    const groups = parseWaves(waves)
    validateCorrelation(groups, examples)
    for (const group of groups) {
      const actual = curateVCD(
        readFileSync(
          new URL(`../../public/${group.download}`, import.meta.url),
          'utf8',
        ),
        group.data.signals.map((s) => s.name),
      )
      expect(actual).toEqual(group.data)
    }
  })
  it('retains unknown logic as unknown rather than inventing numerical values', () => {
    const v =
      '$timescale 1ps $end\n$scope module TOP $end\n$var wire 4 ! a $end\n$upscope $end\n$enddefinitions $end\n#0\nbx !\n#2\nb10 !\n'
    const data = curateVCD(v, ['a'])
    expect(data.frames.map((f) => f.values.a)).toEqual(['xxxx', '0010'])
    expect(validateWaveData(data)).toEqual(data)
    const g = parseWaves(waves)[0]
    expect(valueAt(g, 0, 'a')).toBeDefined()
  })
  it('rejects malformed transitions, missing declarations and unsafe download paths', () => {
    expect(() => curateVCD('bad VCD', ['a'])).toThrow(/Incomplete/)
    const v =
      '$timescale 1ps $end\n$scope module TOP $end\n$var wire 1 ! a $end\n$upscope $end\n$enddefinitions $end\n#0\n0!\n#0\n1!'
    expect(() => curateVCD(v, ['a'])).toThrow(/monotone/)
    const bad = structuredClone(waves)
    bad.groups[0].download = 'javascript:alert(1)'
    expect(() => parseWaves(bad)).toThrow(/download/)
    expect(() =>
      validateWaveData({
        timescale: '1ps',
        signals: [{ name: 'a', width: 1 }],
        frames: [{ time: 0, values: { a: '7' } }],
      }),
    ).toThrow(/bits/)
  })
  it('rejects waveform/RTL output and flag mismatches', () => {
    const groups = parseWaves(structuredClone(waves))
    groups[0].data.frames[1].values.result = '0000000000000000'
    expect(() => validateCorrelation(groups, examples)).toThrow(/differs/)
    const flags = parseWaves(structuredClone(waves))
    flags[0].data.frames[1].values.invalid = '1'
    expect(() => validateCorrelation(flags, examples)).toThrow(/flags differ/)
  })
})
