/**
 * Presentation data for the `#decode` workspace: R3's divergence grid,
 * top-2 margins and batch-invariance matrix, from
 * `runtime/results/r3-decode.json` (random weights, random prompts, logit
 * scale 1 rows) and `runtime/results/r3-trained.json` (random and trained
 * weights on the same copy prompts). Only validation, selection and formatting.
 */
import r3Raw from '../../../runtime/results/r3-decode.json'
import trainedRaw from '../../../runtime/results/r3-trained.json'
import { checker } from './validate'

export interface GridRow {
  policy: string
  schedule: string
  world: number
  diverged: number
  identical: number
  atRisk: number
  flips: number
}

export interface MatrixRow {
  world: number
  schedule: string
  policy: string
  predicted: boolean
  /** Every variant except `alone`, compared with `alone`. */
  variants: { variant: string; logits: number; tokens: number }[]
}

export interface Margins {
  policy: string
  p10: number
  p50: number
  p90: number
  min: number | null
  copyCorrect: number | null
  tokens: number | null
}

export interface DecodeSource {
  key: string
  title: string
  description: string
  prompts: number
  generated: number
  grid: GridRow[]
  matrix: MatrixRow[]
  matrixWorld: number
  margins: Margins[]
  predictionHeld: boolean
}

export function parseR3(value: unknown): DecodeSource {
  const c = checker('r3-decode')
  const d = c.obj(value, 'file')
  const w = c.obj(d.workload, 'workload')
  const prompts = c.count(w.prompts, 'prompts')
  const grid = c
    .arr(d.grid, 'grid')
    .map((x, i) => c.obj(x, `grid[${i}]`))
    .filter((g) => c.num(g.logit_scale, 'scale') === 1)
    .map((g) => ({
      policy: c.str(g.policy, 'policy'),
      schedule: c.str(g.schedule, 'schedule'),
      world: c.count(g.world, 'world'),
      diverged: c.count(g.diverged, 'diverged'),
      identical: c.count(g.logits_identical, 'identical'),
      atRisk: c.count(g.at_risk_tokens, 'at risk'),
      flips: c.arr(g.flips, 'flips').length,
    }))
  const matrix = c.arr(d.batch_invariance, 'batch_invariance').map((x, i) => {
    const r = c.obj(x, `batch_invariance[${i}]`)
    const variants = c.arr(r.variants, 'variants').map((v, j) => {
      const o = c.obj(v, `variants[${j}]`)
      return {
        variant: c.str(o.variant, 'variant'),
        logits: c.count(o.prompts_with_different_logits, 'logits'),
        tokens: c.count(o.prompts_with_different_tokens, 'tokens'),
      }
    })
    if (variants[0]?.variant !== 'alone')
      c.fail('the first variant must be alone')
    return {
      world: c.count(r.world, 'world'),
      schedule: c.str(r.schedule, 'schedule'),
      policy: c.str(r.policy, 'policy'),
      predicted: c.bool(r.predicted_invariant, 'predicted_invariant'),
      variants: variants.slice(1),
    }
  })
  const margins = c
    .arr(d.margins, 'margins')
    .map((x, i) => c.obj(x, `margins[${i}]`))
    .filter((m) => c.num(m.logit_scale, 'scale') === 1)
    .map((m) => {
      const b = c.obj(m.baseline_margin, 'baseline_margin')
      return {
        policy: c.str(m.policy, 'policy'),
        p10: c.num(b.p10, 'p10'),
        p50: c.num(b.p50, 'p50'),
        p90: c.num(b.p90, 'p90'),
        min: null,
        copyCorrect: null,
        tokens: null,
      }
    })
  return {
    key: 'r3',
    title: 'R3: random weights, random prompts',
    description:
      'The original r3-decode run (logit scale 1): random BF16 weights, 32 random-token prompts of 16 tokens, 48 greedy tokens.',
    prompts,
    generated: c.count(w.generated_tokens, 'generated'),
    grid,
    matrix,
    matrixWorld: Math.max(...matrix.map((r) => r.world)),
    margins,
    predictionHeld: c.bool(d.prediction_held, 'prediction_held'),
  }
}

export function parseTrained(value: unknown): DecodeSource[] {
  const c = checker('r3-trained')
  const d = c.obj(value, 'file')
  const w = c.obj(d.workload, 'workload')
  const prompts = c.count(w.prompts, 'prompts')
  const generated = c.count(w.generated_tokens, 'generated')
  return c.arr(d.models, 'models').map((x, i) => {
    const m = c.obj(x, `models[${i}]`)
    const name = c.str(m.model, 'model')
    const bi = c.obj(m.batch_invariance, 'batch_invariance')
    return {
      key: `trained-${name}`,
      title:
        name === 'trained'
          ? 'Rerun: trained weights, copy prompts'
          : 'Rerun: random weights, copy prompts',
      description:
        name === 'trained'
          ? 'The deterministically trained copy-task model (r4-training), its weights rounded to BF16, on 32 copy prompts of 26 tokens, 24 greedy tokens (the copy).'
          : 'R3’s random BF16 weights on the same 32 copy prompts as the trained model.',
      prompts,
      generated,
      grid: c.arr(m.grid, 'grid').map((y, j) => {
        const g = c.obj(y, `grid[${j}]`)
        return {
          policy: c.str(g.policy, 'policy'),
          schedule: c.str(g.schedule, 'schedule'),
          world: c.count(g.world, 'world'),
          diverged: c.count(g.prompts_diverged, 'diverged'),
          identical: c.count(g.prompts_logits_identical, 'identical'),
          atRisk: c.count(g.at_risk_tokens, 'at risk'),
          flips: c.count(g.flips, 'flips'),
        }
      }),
      matrix: c.arr(bi.rows_detail, 'rows_detail').map((y, j) => {
        const r = c.obj(y, `rows_detail[${j}]`)
        return {
          world: c.count(r.world, 'world'),
          schedule: c.str(r.schedule, 'schedule'),
          policy: c.str(r.policy, 'policy'),
          predicted: c.bool(r.predicted_invariant, 'predicted_invariant'),
          variants: c.arr(r.variants, 'variants').map((v, k) => {
            const o = c.obj(v, `variants[${k}]`)
            return {
              variant: c.str(o.variant, 'variant'),
              logits: c.count(o.prompts_logits_differ, 'logits'),
              tokens: c.count(o.prompts_tokens_differ, 'tokens'),
            }
          }),
        }
      }),
      matrixWorld: c.count(d.matrix_world, 'matrix_world'),
      margins: c.arr(m.policies, 'policies').map((y, j) => {
        const p = c.obj(y, `policies[${j}]`)
        const b = c.obj(p.baseline, 'baseline')
        return {
          policy: c.str(p.policy, 'policy'),
          p10: c.num(b.margin_p10, 'p10'),
          p50: c.num(b.margin_p50, 'p50'),
          p90: c.num(b.margin_p90, 'p90'),
          min: c.num(b.margin_min, 'min'),
          copyCorrect: c.count(b.copy_tokens_correct, 'copy'),
          tokens: c.count(b.tokens, 'tokens'),
        }
      }),
      predictionHeld: c.bool(bi.prediction_held, 'prediction_held'),
    }
  })
}

export const decodeSources: DecodeSource[] = [
  parseR3(r3Raw),
  ...parseTrained(trainedRaw),
]

export function policiesOf(s: DecodeSource): string[] {
  return [...new Set(s.grid.map((g) => g.policy))]
}

/** Flips and at-risk tokens pooled over every grid run of one policy. */
export function pooledFlips(s: DecodeSource, policy: string) {
  const runs = s.grid.filter((g) => g.policy === policy)
  return {
    runs: runs.length,
    diverged: runs.reduce((a, g) => a + g.diverged, 0),
    prompts: runs.length * s.prompts,
    atRisk: runs.reduce((a, g) => a + g.atRisk, 0),
    flips: runs.reduce((a, g) => a + g.flips, 0),
  }
}
