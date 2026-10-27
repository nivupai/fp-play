import { hashes, integer, object, text } from './artifacts'
import type { RecordedExample } from './artifacts'
export interface SynthesisRecord {
  id: string
  unit: 'add' | 'mul' | 'fma' | 'nonfused'
  variant: string
  cells: number
  area: number
  depth: number
  delay: number | null
  timingUnavailable: string | null
  cases: number
  mismatches: number
}
export interface SynthesisContext {
  records: SynthesisRecord[]
  tools: Record<string, unknown>
  config: Record<string, unknown>
  sourceHashes: Record<string, string>
  commit: string
  timestamp: string
}
const finite = (v: unknown, label: string) => {
  if (typeof v !== 'number' || !Number.isFinite(v) || v < 0)
    throw new Error(`Invalid ${label}`)
  return v
}
export function parseSynthesis(value: unknown): SynthesisContext {
  const a = object(value, 'synthesis artifact')
  if (
    a.schemaVersion !== 1 ||
    !Array.isArray(a.records) ||
    a.records.length > 32
  )
    throw new Error('Unsupported synthesis artifact')
  const config = object(a.config, 'synthesis configuration'),
    library = object(config.library, 'library')
  text(library.name, 'library name')
  text(library.corner, 'corner')
  hashes({ library: library.sha256 })
  object(config.timing, 'timing assumptions')
  object(config.mapping, 'mapping assumptions')
  const tools = object(a.tools, 'tools')
  text(tools.yosys, 'Yosys version')
  text(tools.abc, 'ABC version')
  if (tools.opensta !== null) text(tools.opensta, 'OpenSTA version')
  const records = a.records.map((value) => {
    const r = object(value, 'synthesis record'),
      mapped = object(r.mapped, 'mapped metrics'),
      verification = object(r.verification, 'verification')
    if (!['add', 'mul', 'fma', 'nonfused'].includes(String(r.unit)))
      throw new Error('Unsupported synthesis unit')
    const timing = r.timing === null ? null : object(r.timing, 'timing report')
    const missing =
      timing === null
        ? text(r.timingUnavailable, 'timing unavailable reason')
        : null
    if (timing && !tools.opensta)
      throw new Error('Timing requires an OpenSTA version')
    return {
      id: text(r.id, 'experiment id'),
      unit: r.unit as SynthesisRecord['unit'],
      variant: text(r.variant, 'variant'),
      cells: integer(mapped.cells, 1, 1000000, 'cell count'),
      area: finite(mapped.area, 'cell area'),
      depth: integer(mapped.logicDepth, 0, 100000, 'logic depth'),
      delay: timing ? finite(timing.delayNs, 'delay') : null,
      timingUnavailable: missing,
      cases: integer(verification.cases, 1, 10000000, 'verified cases'),
      mismatches: integer(verification.mismatches, 0, 10000000, 'mismatches'),
    }
  })
  if (new Set(records.map((r) => r.id)).size !== records.length)
    throw new Error('Duplicate synthesis experiments')
  return {
    records,
    tools,
    config,
    sourceHashes: hashes(a.sourceHashes),
    commit: text(a.gitCommit, 'source commit'),
    timestamp: text(a.timestamp, 'measurement timestamp'),
  }
}
export function linkSynthesis(
  example: RecordedExample,
  context: SynthesisContext,
) {
  const unit =
    example.operation === 'add' || example.operation === 'subtract'
      ? 'add'
      : example.operation === 'multiply'
        ? 'mul'
        : example.operation
  const required =
    unit === 'add'
      ? ['fp16_addsub']
      : unit === 'mul'
        ? ['fp16_mul', 'fp16_mul_fma_core', 'fp16_round_grid48']
        : unit === 'fma'
          ? ['fp16_fma', 'fp16_mul_fma_core', 'fp16_round_grid48']
          : [
              'fp16_mul_fma_compare',
              'fp16_mul',
              'fp16_addsub',
              'fp16_mul_fma_core',
              'fp16_round_grid48',
            ]
  const compatible = required.every((name) => {
    const p = `rtl/src/${name}.sv`
    return (
      !!example.provenance.sourceHashes[p] &&
      example.provenance.sourceHashes[p] === context.sourceHashes[p]
    )
  })
  return {
    unit,
    compatible,
    records: context.records.filter((r) => r.unit === unit),
    reason: compatible
      ? 'Core RTL source hashes match the measured baseline.'
      : 'Missing or different RTL source hashes: these metrics cannot be attributed to this simulation revision.',
  }
}
