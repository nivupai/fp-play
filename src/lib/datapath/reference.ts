import { calculateFP16 } from '../numerics/fp16Arithmetic'
import {
  fmaFP16,
  multiplyFP16,
  nonfusedMultiplyAddFP16,
} from '../numerics/fp16Multiply'
import {
  describeMultiplyStage,
  MULTIPLY_STAGE_TITLES,
} from '../arithmetic/multiplyPresentation'
import { STAGE_TITLES, formatExactDyadic } from '../arithmetic/presentation'
import { formatExactFP16 } from '../explorer/presentation'
import { flagMask } from './artifacts'
import type { Phase, RecordedExample } from './artifacts'
export function compareExample(record: RecordedExample) {
  const [a, b, c] = record.operands
  const result =
    record.operation === 'add' || record.operation === 'subtract'
      ? calculateFP16(a, b, record.operation)
      : record.operation === 'multiply'
        ? multiplyFP16(a, b)
        : record.operation === 'fma'
          ? fmaFP16(a, b, c)
          : nonfusedMultiplyAddFP16(a, b, c)
  const paths =
    'product' in result ? [result.product, result.addition] : [result]
  const phases = paths.map((software, i) => ({
    software,
    recorded: record.phases[i],
    bitsAgree: software.bits === record.phases[i].bits,
    flagsAgree: flagMask(software.flags) === flagMask(record.phases[i].flags),
    ...compareSignals(record.phases[i], software),
  }))
  return {
    result,
    phases,
    bitsAgree: result.bits === record.bits,
    flagsAgree: flagMask(result.flags) === flagMask(record.flags),
    agrees:
      result.bits === record.bits &&
      flagMask(result.flags) === flagMask(record.flags) &&
      phases.every(
        (p) => p.bitsAgree && p.flagsAgree && p.signalMismatches.length === 0,
      ),
  }
}
export type ReferenceResult =
  ReturnType<typeof calculateFP16> | ReturnType<typeof multiplyFP16>
export function softwareTrace(result: ReferenceResult) {
  return result.trace.map((stage) => ({
    kind: stage.kind,
    title: { ...STAGE_TITLES, ...MULTIPLY_STAGE_TITLES }[stage.kind],
    ...describeMultiplyStage(stage),
  }))
}
export function exactReference(result: ReferenceResult): string {
  return result.exact
    ? `${formatExactDyadic(result.exact)} = ${result.exact.coefficient} × 2^${result.exact.exponent}`
    : `No finite exact dyadic result; software result ${formatExactFP16(result.decoded.value)}`
}
/** Educational Boolean explanation, checked against observations; not an FP16 arithmetic engine. */
export function explainGRS(phase: Phase) {
  if (phase.bypassed === true)
    return {
      status: 'bypassed' as const,
      message:
        'Special-value bypass: finite GRS defaults do not describe executed rounding.',
    }
  const fields = ['retained', 'guard', 'round', 'sticky', 'increment'] as const
  if (fields.some((k) => !phase.signals[k]) || phase.bypassed === null)
    return {
      status: 'missing' as const,
      message:
        'GRS observations or finite-path status were not recorded for this phase.',
    }
  const values = Object.fromEntries(
    fields.map((k) => [k, Number(phase.signals[k].value)]),
  ) as Record<(typeof fields)[number], number>
  const lsb = values.retained % 2,
    expectedIncrement =
      values.guard === 1 &&
      (values.round === 1 || values.sticky === 1 || lsb === 1),
    halfway = values.guard === 1 && values.round === 0 && values.sticky === 0
  const position = halfway
    ? 'Exact midpoint tie'
    : values.guard === 0 && values.round === 0 && values.sticky === 0
      ? 'Exact retained value'
      : values.guard === 0
        ? 'Below midpoint'
        : 'Above midpoint'
  return {
    status: 'available' as const,
    ...values,
    lsb,
    halfway,
    expectedIncrement,
    agrees: expectedIncrement === !!values.increment,
    inferredCarry: values.retained === 2047 && values.increment === 1,
    message: `${position}; ${expectedIncrement ? 'increment' : 'retain'} the significand${halfway ? (lsb ? ' to reach even parity' : ' because its LSB is even') : ''}.`,
  }
}

/** Preserve historical artifact titles in provenance while correcting two mislabeled product ties. */
export function exampleLabel(record: RecordedExample): string {
  if (
    record.family !== 'product' ||
    !['Even midpoint rounds down', 'Odd midpoint rounds up'].includes(
      record.name,
    )
  )
    return record.name
  const grs = explainGRS(record.phases[0])
  if (grs.status !== 'available') return record.name
  return `Product midpoint · retained ${grs.lsb ? 'odd' : 'even'} · ${grs.increment ? 'increment' : 'stay'}`
}

/** Compare only observations that exist, using the reference's authoritative exact trace. */
function compareSignals(phase: Phase, software: ReferenceResult) {
  const signalMismatches: string[] = []
  let signalChecks = 0
  const check = (name: string, expected: bigint | number | boolean) => {
    const s = phase.signals[name]
    if (s) {
      signalChecks++
      if (s.value !== BigInt(expected))
        signalMismatches.push(`${phase.label}: ${name}`)
    }
  }
  if (phase.bypassed === true) return { signalChecks, signalMismatches }
  const grs = software.trace.find((s) => s.kind === 'grs'),
    round = software.trace.find((s) => s.kind === 'round'),
    norm = software.trace.find((s) => s.kind === 'normalize')
  if (grs)
    for (const name of ['guard', 'round', 'sticky'] as const)
      check(name, grs[name])
  if (round) {
    check('increment', round.increment)
    check('tiny', round.tinyAfterPrecisionRounding)
  }
  if (norm) check('retained', norm.retained)
  if (software.exact) {
    const magnitude =
      software.exact.coefficient < 0n
        ? -software.exact.coefficient
        : software.exact.coefficient
    check(
      phase.kind === 'add' ? 'magnitude' : 'arithmeticMagnitude',
      magnitude <<
        BigInt(software.exact.exponent + (phase.kind === 'add' ? 24 : 48)),
    )
  }
  const multiply = software.trace.find((s) => s.kind === 'multiply')
  if (multiply) {
    check('significandA', multiply.significandA)
    check('significandB', multiply.significandB)
    check('exactProduct', multiply.product)
    check('productExponent', multiply.exponent)
    check('productSign', multiply.sign)
  }
  return { signalChecks, signalMismatches }
}
