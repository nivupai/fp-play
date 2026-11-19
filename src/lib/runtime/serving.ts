/**
 * Presentation data for the `#serving` workspace: the R2 serving scenarios
 * from `runtime/results/r2-serving.json`. Only validation, selection and
 * formatting; every number is copied from the file.
 */
import raw from '../../../runtime/results/r2-serving.json'
import { checker } from './validate'

export interface Percentiles {
  n: number
  p50: number
  p99: number
}

export interface ServingRun {
  label: string
  world: number
  rate: number
  preemption: string
  prefixCaching: boolean
  collective: string
  wire: string
  hostLinkGBps: number
  kvBlocks: number
  ttft: Percentiles
  tpot: Percentiles
  e2e: Percentiles
  outputTokensPerS: number
  commShare: number
  preemptions: number
  recomputedTokens: number
  swapGB: number
  prefixHitRate: number | null
  finished: number
  requests: number
  failed: number
  rejected: number
}

export interface ScenarioGroup {
  key: string
  title: string
  varies: string
  runs: ServingRun[]
}

const GROUPS: [string, string, string][] = [
  [
    'load',
    'Load sweep',
    'Arrival rate (5, 10 and 20 requests per second) and tensor-parallel width P (1, 2, 4 and 8).',
  ],
  [
    'preemption_under_memory_pressure',
    'Preemption under memory pressure',
    'Recompute versus swap when the KV cache is too small for the load.',
  ],
  [
    'swap_sensitivity',
    'Swap sensitivity',
    'Host-link bandwidth and samples per request under swap preemption.',
  ],
  [
    'prefix_sharing',
    'Prefix sharing',
    'The prefix cache off and on for requests that share prompt prefixes.',
  ],
  [
    'collective_choice',
    'Collective choice',
    'AllReduce schedule and wire policy at P = 4 and P = 8.',
  ],
  [
    'failures',
    'Failures',
    'No failure, and a rank failure mid-collective handled by abort, retry or drain.',
  ],
]

function parse(value: unknown): ScenarioGroup[] {
  const c = checker('r2-serving')
  const d = c.obj(value, 'file')
  if (c.str(d.schema, 'schema') !== 'fp-play-runtime/r2-serving/v1')
    c.fail('unknown schema')
  const scenarios = c.obj(d.scenarios, 'scenarios')
  const pct = (v: unknown, what: string): Percentiles => {
    const o = c.obj(v, what)
    return {
      n: c.count(o.n, `${what}.n`),
      p50: c.num(o.p50, `${what}.p50`),
      p99: c.num(o.p99, `${what}.p99`),
    }
  }
  return GROUPS.map(([key, title, varies]) => ({
    key,
    title,
    varies,
    runs: c.arr(scenarios[key], key).map((item, i) => {
      const s = c.obj(item, `${key}[${i}]`)
      const m = c.obj(s.metrics, `${key}[${i}].metrics`)
      const hit = m.prefix_hit_rate
      return {
        label: c.str(s.label, 'label'),
        world: c.count(s.world, 'world'),
        rate: c.num(s.rate_per_s, 'rate'),
        preemption: c.str(s.preemption, 'preemption'),
        prefixCaching: c.bool(s.prefix_caching, 'prefix_caching'),
        collective: c.str(s.collective, 'collective'),
        wire: c.str(s.wire_policy, 'wire_policy'),
        hostLinkGBps: c.num(s.host_link_bytes_per_s, 'host link') / 1e9,
        kvBlocks: c.count(s.kv_capacity_blocks, 'kv blocks'),
        ttft: pct(m.ttft_ns, 'ttft'),
        tpot: pct(m.tpot_ns, 'tpot'),
        e2e: pct(m.e2e_ns, 'e2e'),
        outputTokensPerS: c.num(m.output_tokens_per_s, 'throughput'),
        commShare: c.num(m.comm_share, 'comm share'),
        preemptions:
          c.count(m.preemptions_recompute, 'preemptions') +
          c.count(m.preemptions_swap, 'preemptions'),
        recomputedTokens: c.count(m.recomputed_tokens, 'recomputed'),
        swapGB: c.count(m.swap_bytes_all_devices, 'swap bytes') / 1e9,
        prefixHitRate: hit === null ? null : c.num(hit, 'hit rate'),
        finished: c.count(m.finished, 'finished'),
        requests: c.count(m.requests, 'requests'),
        failed: c.count(m.failed, 'failed'),
        rejected: c.count(m.rejected, 'rejected'),
      }
    }),
  }))
}

export const servingGroups: ScenarioGroup[] = parse(raw)
export { parse as parseServing }

export interface Metric {
  key: string
  label: string
  unit: string
  value: (r: ServingRun) => number
  text: (r: ServingRun) => string
}

const ms = (ns: number) => ns / 1e6

export const servingMetrics: Metric[] = [
  {
    key: 'ttft-p50',
    label: 'Time to first token, p50',
    unit: 'ms',
    value: (r) => ms(r.ttft.p50),
    text: (r) => `${ms(r.ttft.p50).toFixed(2)} ms`,
  },
  {
    key: 'ttft-p99',
    label: 'Time to first token, p99',
    unit: 'ms',
    value: (r) => ms(r.ttft.p99),
    text: (r) => `${ms(r.ttft.p99).toFixed(2)} ms`,
  },
  {
    key: 'tpot-p50',
    label: 'Time per output token, p50',
    unit: 'ms',
    value: (r) => ms(r.tpot.p50),
    text: (r) => `${ms(r.tpot.p50).toFixed(2)} ms`,
  },
  {
    key: 'tpot-p99',
    label: 'Time per output token, p99',
    unit: 'ms',
    value: (r) => ms(r.tpot.p99),
    text: (r) => `${ms(r.tpot.p99).toFixed(2)} ms`,
  },
  {
    key: 'throughput',
    label: 'Output tokens per second',
    unit: 'tokens/s',
    value: (r) => r.outputTokensPerS,
    text: (r) => `${Math.round(r.outputTokensPerS)} tokens/s`,
  },
  {
    key: 'comm',
    label: 'Communication share of step time',
    unit: '%',
    value: (r) => 100 * r.commShare,
    text: (r) => `${(100 * r.commShare).toFixed(1)}%`,
  },
  {
    key: 'preemptions',
    label: 'Preemptions',
    unit: '',
    value: (r) => r.preemptions,
    text: (r) => `${r.preemptions}`,
  },
]
