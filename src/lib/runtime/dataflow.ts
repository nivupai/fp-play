/**
 * Presentation data for the `#dataflow` workspace: one 8B-class layer on
 * the illustrative toy accelerator, from `runtime/results/r5-dataflow.json`.
 * Only validation, selection and formatting.
 */
import raw from '../../../runtime/results/r5-dataflow.json'
import { checker } from './validate'

export interface Machine {
  name: string
  rows: number
  cols: number
  lanes: number
  sramBytes: number
  hbmBytesPerCycle: number
  cycleNs: number
}

export interface OpRow {
  op: string
  cycles: number
  compute: number
  dma: number
  stall: number
  fill: number
  hbmBytes: number
  usefulMacs: number
}

/** Cycle shares, in the order the report lists them. */
export const REASONS = [
  ['useful_array', 'Useful array work'],
  ['padding_or_masked_array', 'Padding or masked array work'],
  ['vector', 'Vector unit'],
  ['memory_stall', 'Waiting for the DMA'],
  ['first_load_fill', 'Exposed first load'],
] as const

export type Reason = (typeof REASONS)[number][0]

export interface DataflowRun {
  machine: string
  shape: string
  phase: 'prefill' | 'decode'
  cycles: number
  bound: number
  utilization: number
  intensity: number
  hbmBytes: number
  reasons: Record<Reason, number>
  ops: OpRow[]
}

export interface DataflowData {
  machines: Machine[]
  runs: DataflowRun[]
}

export function parseDataflow(value: unknown): DataflowData {
  const c = checker('r5-dataflow')
  const d = c.obj(value, 'file')
  const machines = c.arr(d.machines, 'machines').map((m, i) => {
    const o = c.obj(m, `machines[${i}]`)
    return {
      name: c.str(o.name, 'name'),
      rows: c.count(o.array_rows, 'rows'),
      cols: c.count(o.array_cols, 'cols'),
      lanes: c.count(o.vector_lanes, 'lanes'),
      sramBytes: c.count(o.sram_bytes, 'sram'),
      hbmBytesPerCycle: c.count(o.hbm_bytes_per_cycle, 'hbm'),
      cycleNs: c.num(o.assumed_cycle_ns, 'cycle'),
    }
  })
  const runs = c.arr(d.runs, 'runs').map((r, i) => {
    const o = c.obj(r, `runs[${i}]`)
    const shape = c.obj(o.shape, 'shape')
    const phase = c.str(shape.phase, 'phase')
    if (phase !== 'prefill' && phase !== 'decode') c.fail('unknown phase')
    const label =
      phase === 'prefill'
        ? `prefill ${c.count(shape.tokens, 'tokens')} tokens`
        : `decode batch ${c.count(shape.batch, 'batch')}, context ${c.count(shape.context, 'context')}`
    const by = c.obj(o.cycles_by_reason, 'cycles_by_reason')
    const reasons = Object.fromEntries(
      REASONS.map(([k]) => [k, c.num(by[k], k)]),
    ) as Record<Reason, number>
    const machine = c.str(o.machine, 'machine')
    if (!machines.some((m) => m.name === machine)) c.fail('unknown machine')
    return {
      machine,
      shape: label,
      phase: phase as 'prefill' | 'decode',
      cycles: c.count(o.cycles, 'cycles'),
      bound: c.count(o.roofline_bound_cycles, 'bound'),
      utilization: c.num(o.array_utilization, 'utilization'),
      intensity: c.num(o.flops_per_hbm_byte, 'intensity'),
      hbmBytes: c.count(o.hbm_bytes, 'hbm bytes'),
      reasons,
      ops: c.arr(o.ops, 'ops').map((x, j) => {
        const p = c.obj(x, `ops[${j}]`)
        return {
          op: c.str(p.op, 'op'),
          cycles: c.count(p.cycles, 'cycles'),
          compute: c.count(p.compute_cycles, 'compute'),
          dma: c.count(p.dma_cycles, 'dma'),
          stall: c.count(p.memory_stall, 'stall'),
          fill: c.count(p.fill, 'fill'),
          hbmBytes: c.count(p.hbm_bytes, 'hbm'),
          usefulMacs: c.count(p.useful_macs, 'macs'),
        }
      }),
    }
  })
  return { machines, runs }
}

export const dataflow: DataflowData = parseDataflow(raw)

export function runsFor(machine: string): DataflowRun[] {
  return dataflow.runs.filter((r) => r.machine === machine)
}

/** Fraction of a run's cycles spent on each reason. */
export function shares(r: DataflowRun): Record<Reason, number> {
  return Object.fromEntries(
    REASONS.map(([k]) => [k, r.reasons[k] / r.cycles]),
  ) as Record<Reason, number>
}
