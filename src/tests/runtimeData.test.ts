import { describe, expect, it } from 'vitest'
import r2 from '../../runtime/results/r2-serving.json'
import r3 from '../../runtime/results/r3-decode.json'
import r3t from '../../runtime/results/r3-trained.json'
import r4 from '../../runtime/results/r4-precision.json'
import r5 from '../../runtime/results/r5-dataflow.json'
import training from '../../runtime/results/r4-training.json'
import { parseServing, servingGroups } from '../lib/runtime/serving'
import {
  dataflow,
  parseDataflow,
  REASONS,
  shares,
} from '../lib/runtime/dataflow'
import {
  decodeSources,
  parseR3,
  parseTrained,
  pooledFlips,
} from '../lib/runtime/decode'
import {
  oneSite,
  parseR4,
  parseTraining,
  precisionSources,
  training as trainingData,
  uniformRuns,
} from '../lib/runtime/precision'
import { RecordedDataError, sci } from '../lib/runtime/validate'

const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T

// Expected values below are copied by hand from the committed Markdown
// reports (runtime/results/*.md), not computed from the JSON.

describe('recorded serving data (r2-serving)', () => {
  it('copies the scenarios the report lists', () => {
    const load = servingGroups.find((g) => g.key === 'load')!
    expect(load.runs).toHaveLength(12)
    const p1 = load.runs[0]
    expect(p1.label).toBe('load P=1 rate=5')
    expect((p1.ttft.p50 / 1e6).toFixed(2)).toBe('26.43')
    expect((p1.tpot.p99 / 1e6).toFixed(2)).toBe('7.00')
    expect(Math.round(p1.outputTokensPerS)).toBe(1503)
    const prefix = servingGroups.find((g) => g.key === 'prefix_sharing')!
    expect(prefix.runs[1].prefixHitRate!.toFixed(3)).toBe('0.429')
    expect(prefix.runs[0].prefixHitRate).toBeNull()
    const pressure = servingGroups.find(
      (g) => g.key === 'preemption_under_memory_pressure',
    )!
    expect(pressure.runs.map((r) => r.preemptions)).toEqual([131, 84])
    expect(pressure.runs[1].swapGB.toFixed(2)).toBe('23.04')
    expect(servingGroups.map((g) => g.runs.length)).toEqual([
      12, 2, 8, 2, 12, 4,
    ])
  })

  it('rejects malformed data', () => {
    const bad = clone(r2) as { schema: string }
    bad.schema = 'other'
    expect(() => parseServing(bad)).toThrow(RecordedDataError)
    const missing = clone(r2) as { scenarios: Record<string, unknown> }
    delete missing.scenarios.failures
    expect(() => parseServing(missing)).toThrow(/failures must be a list/)
  })
})

describe('recorded dataflow data (r5-dataflow)', () => {
  it('copies utilization and the cycle breakdown', () => {
    const one = dataflow.runs.filter(
      (r) => r.machine === dataflow.machines[0].name,
    )
    expect(one.map((r) => (100 * r.utilization).toFixed(1))).toEqual([
      '95.7',
      '94.3',
      '89.2',
      '0.8',
      '6.0',
      '19.2',
      '43.3',
    ])
    expect((100 * shares(one[3]).padding_or_masked_array).toFixed(1)).toBe(
      '98.9',
    )
    for (const r of dataflow.runs) {
      const sum = REASONS.reduce((a, [k]) => a + r.reasons[k], 0)
      expect(sum).toBe(r.cycles)
      expect(r.ops.reduce((a, o) => a + o.cycles, 0)).toBe(r.cycles)
    }
  })

  it('rejects malformed data', () => {
    const bad = clone(r5) as { runs: { machine: string }[] }
    bad.runs[0].machine = 'unknown'
    expect(() => parseDataflow(bad)).toThrow(/unknown machine/)
  })
})

describe('recorded decode data (r3-decode, r3-trained)', () => {
  it('copies divergence, margins and the batch-invariance verdict', () => {
    const [orig, random, trained] = decodeSources
    const seq2 = orig.grid.find(
      (g) =>
        g.policy === 'bf16' && g.schedule === 'sequential' && g.world === 2,
    )!
    expect(seq2.diverged).toBe(18)
    expect(orig.generated).toBe(48)
    expect(trained.margins[0].p50.toFixed(4)).toBe('9.3781')
    expect(random.margins[0].p50.toFixed(4)).toBe('0.2447')
    expect(trained.margins[0].min!.toFixed(4)).toBe('5.5726')
    expect(pooledFlips(random, 'bf16')).toMatchObject({
      flips: 160,
      atRisk: 12527,
    })
    expect(pooledFlips(trained, 'bf16')).toMatchObject({
      flips: 0,
      atRisk: 14592,
    })
    expect(pooledFlips(random, 'fp32-rs/bf16-ag').flips).toBe(6)
    expect(decodeSources.every((s) => s.predictionHeld)).toBe(true)
    for (const s of decodeSources) {
      for (const row of s.matrix.filter((r) => r.predicted))
        expect(row.variants.every((v) => v.logits === 0)).toBe(true)
      expect(
        s.matrix.some(
          (r) => !r.predicted && r.variants.some((v) => v.logits > 0),
        ),
      ).toBe(true)
    }
  })

  it('rejects malformed data', () => {
    const bad = clone(r3) as {
      batch_invariance: { variants: { variant: string }[] }[]
    }
    bad.batch_invariance[0].variants[0].variant = 'together'
    expect(() => parseR3(bad)).toThrow(/first variant must be alone/)
    const t = clone(r3t) as {
      models: { batch_invariance: { prediction_held: unknown } }[]
    }
    t.models[1].batch_invariance.prediction_held = 'yes'
    expect(() => parseTrained(t)).toThrow(RecordedDataError)
  })
})

describe('recorded precision data (r4-precision, r4-trained, r4-training)', () => {
  it('copies the uniform and one-site results', () => {
    const [orig, random, trained] = precisionSources
    const bf16 = uniformRuns(orig).find((r) => r.level === 'bf16')!
    expect(sci(bf16.meanKl)).toBe('5.05e-5')
    expect(bf16.top1).toBe(253)
    expect(orig.levels).toEqual([
      'fp32',
      'bf16',
      'bf16-acc',
      'fp16',
      'fp16-acc',
    ])
    expect(trained.levels).toHaveLength(8)
    const e5m2 = (s: typeof random) =>
      uniformRuns(s).find((r) => r.level === 'fp8-e5m2')!
    expect(sci(e5m2(random).meanKl)).toBe('4.51e-2')
    expect(sci(e5m2(trained).meanKl)).toBe('9.24e-5')
    expect(e5m2(trained).top1).toBe(192)
    expect(sci(oneSite(random, 'layer 0 attention', 'fp8-e5m2')!.meanKl)).toBe(
      '1.99e-2',
    )
    expect(oneSite(trained, 'KV cache', 'bf16-acc')).toBeUndefined()
  })

  it('copies the training run', () => {
    expect(trainingData.steps).toHaveLength(600)
    expect(trainingData.steps[0].loss.toFixed(5)).toBe('8.12375')
    expect(trainingData.steps[599].loss.toFixed(5)).toBe('0.00143')
    expect(trainingData.heldOut.accuracy).toBe(1)
    expect(trainingData.sha256).toBe(
      '9de08b23ec3ca33cf6650f921de2735188f223cd9582ad5fd8089aa704b15b65',
    )
    expect(
      trainingData.greedy.every((g) => g.correctSequences === g.sequences),
    ).toBe(true)
  })

  it('rejects malformed data', () => {
    const bad = clone(r4) as { runs: { kind: string }[] }
    bad.runs[0].kind = 'other'
    expect(() => parseR4(bad)).toThrow(/unknown kind/)
    const t = clone(training) as { weights_sha256: string }
    t.weights_sha256 = 'abc'
    expect(() => parseTraining(t)).toThrow(/64 hex digits/)
  })
})
