/**
 * Presentation data for the `#precision` workspace: R4's per-site precision
 * study (`runtime/results/r4-precision.json`, random weights), its rerun on
 * the trained model with FP8 and INT8 levels (`runtime/results/r4-trained.json`),
 * and the training run (`runtime/results/r4-training.json`). Only
 * validation, selection and formatting.
 */
import r4Raw from '../../../runtime/results/r4-precision.json'
import rerunRaw from '../../../runtime/results/r4-trained.json'
import trainingRaw from '../../../runtime/results/r4-training.json'
import { checker } from './validate'

export interface PolicyRow {
  policy: string
  kind: 'uniform' | 'one site' | 'mixed'
  /** For uniform runs, the level; for one-site runs, the lowered site and its level. */
  level: string | null
  site: string | null
  meanKl: number
  maxAbs: number
  top1: number
  scored: number
  diverged: number
  nonfinite: number
}

export interface PrecisionSource {
  key: string
  title: string
  description: string
  prompts: number
  sites: string[]
  levels: string[]
  runs: PolicyRow[]
}

function parseRuns(
  source: string,
  list: unknown,
): { sites: string[]; runs: PolicyRow[] } {
  const c = checker(source)
  let sites: string[] = []
  const runs = c.arr(list, 'runs').map((x, i) => {
    const r = c.obj(x, `runs[${i}]`)
    const kind = c.str(r.kind, 'kind')
    if (kind !== 'uniform' && kind !== 'one site' && kind !== 'mixed')
      c.fail(`unknown kind ${kind}`)
    const levels = c.arr(r.levels, 'levels').map((y, j) => {
      const o = c.obj(y, `levels[${j}]`)
      return [c.str(o.site, 'site'), c.str(o.level, 'level')] as const
    })
    if (i === 0) sites = levels.map(([s]) => s)
    else if (levels.map(([s]) => s).join('|') !== sites.join('|'))
      c.fail('every run must list the same sites')
    let level: string | null = null
    let site: string | null = null
    if (kind === 'uniform') {
      level = levels[0][1]
      if (levels.some(([, l]) => l !== level))
        c.fail('uniform run with mixed levels')
    } else if (kind === 'one site') {
      const lowered = levels.filter(([, l]) => l !== 'fp32')
      if (lowered.length !== 1)
        c.fail('one-site run must lower exactly one site')
      ;[site, level] = lowered[0]
    }
    return {
      policy: c.str(r.policy, 'policy'),
      kind: kind as PolicyRow['kind'],
      level,
      site,
      meanKl: c.num(r.mean_kl, 'mean_kl'),
      maxAbs: c.num(r.max_abs_logit_difference, 'max_abs'),
      top1: c.count(r.top1_agreement, 'top1'),
      scored: c.count(r.scored_tokens, 'scored'),
      diverged: c.count(r.greedy_diverged, 'diverged'),
      nonfinite: c.count(r.nonfinite_logits, 'nonfinite'),
    }
  })
  return { sites, runs }
}

function levelsOf(runs: PolicyRow[]): string[] {
  return runs.filter((r) => r.kind === 'uniform').map((r) => r.level!)
}

export function parseR4(value: unknown): PrecisionSource {
  const c = checker('r4-precision')
  const d = c.obj(value, 'file')
  const w = c.obj(d.workload, 'workload')
  const { sites, runs } = parseRuns('r4-precision', d.runs)
  return {
    key: 'r4',
    title: 'R4: random weights, random prompts',
    description:
      'The original r4-precision run: R4’s random FP32 master weights, 8 random-token prompts, the five float levels.',
    prompts: c.count(w.prompts, 'prompts'),
    sites,
    levels: levelsOf(runs),
    runs,
  }
}

export function parseRerun(value: unknown): PrecisionSource[] {
  const c = checker('r4-trained')
  const d = c.obj(value, 'file')
  const p = c.obj(d.prompts, 'prompts')
  const prompts = c.count(p.count, 'count')
  return c.arr(d.models, 'models').map((x, i) => {
    const m = c.obj(x, `models[${i}]`)
    const name = c.str(m.model, 'model')
    const { sites, runs } = parseRuns('r4-trained', m.runs)
    return {
      key: `rerun-${name}`,
      title:
        name === 'trained'
          ? 'Rerun: trained weights, copy prompts'
          : 'Rerun: random weights, copy prompts',
      description:
        name === 'trained'
          ? 'The deterministically trained copy-task model on 8 copy prompts, every level including FP8 and INT8 with explicit scales.'
          : 'R4’s random FP32 master weights on the same 8 copy prompts, every level including FP8 and INT8.',
      prompts,
      sites,
      levels: levelsOf(runs),
      runs,
    }
  })
}

export interface TrainingData {
  parameters: number
  steps: { step: number; lr: number; loss: number; accuracy: number }[]
  sha256: string
  heldOut: { sequences: number; loss: number; accuracy: number }
  greedy: {
    model: string
    sequences: number
    correctSequences: number
    tokens: number
    correctTokens: number
  }[]
}

export function parseTraining(value: unknown): TrainingData {
  const c = checker('r4-training')
  const d = c.obj(value, 'file')
  const h = c.obj(d.held_out, 'held_out')
  const sha = c.str(d.weights_sha256, 'weights_sha256')
  if (!/^[0-9a-f]{64}$/.test(sha))
    c.fail('weights_sha256 must be 64 hex digits')
  return {
    parameters: c.count(c.obj(d.model, 'model').parameters, 'parameters'),
    sha256: sha,
    steps: c.arr(d.steps, 'steps').map((x, i) => {
      const s = c.obj(x, `steps[${i}]`)
      return {
        step: c.count(s.step, 'step'),
        lr: c.num(s.lr, 'lr'),
        loss: c.num(s.loss, 'loss'),
        accuracy: c.num(s.accuracy, 'accuracy'),
      }
    }),
    heldOut: {
      sequences: c.count(h.sequences, 'sequences'),
      loss: c.num(h.teacher_forced_loss, 'loss'),
      accuracy: c.num(h.teacher_forced_accuracy, 'accuracy'),
    },
    greedy: c.arr(h.greedy, 'greedy').map((x, i) => {
      const g = c.obj(x, `greedy[${i}]`)
      return {
        model: c.str(g.model, 'model'),
        sequences: c.count(g.sequences, 'sequences'),
        correctSequences: c.count(g.correct_sequences, 'correct'),
        tokens: c.count(g.tokens, 'tokens'),
        correctTokens: c.count(g.correct_tokens, 'correct tokens'),
      }
    }),
  }
}

export const precisionSources: PrecisionSource[] = [
  parseR4(r4Raw),
  ...parseRerun(rerunRaw),
]
export const training: TrainingData = parseTraining(trainingRaw)

export function uniformRuns(s: PrecisionSource): PolicyRow[] {
  return s.runs.filter((r) => r.kind === 'uniform')
}

/** The one-site run lowering `site` to `level`, if the study ran it. */
export function oneSite(
  s: PrecisionSource,
  site: string,
  level: string,
): PolicyRow | undefined {
  return s.runs.find(
    (r) => r.kind === 'one site' && r.site === site && r.level === level,
  )
}
