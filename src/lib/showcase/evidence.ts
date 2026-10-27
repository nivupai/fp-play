import recorded from './hardware-summary.json'
import { floatToFP16Bits, fp16BitsToFloat } from '../numerics/fp16'
import { fmaFP16, nonfusedMultiplyAddFP16 } from '../numerics/fp16Multiply'

/** Presentation only: every measured number is copied from the source-hashed M12 artifact. */
export type ShowcaseRecord = Omit<
  (typeof recorded.records)[number],
  'delay' | 'startpoint' | 'endpoint' | 'pathCells'
> & {
  delay: number | null
  startpoint: string | null
  endpoint: string | null
  pathCells: number | null
}
export const hardwareEvidence: Omit<typeof recorded, 'records'> & {
  records: ShowcaseRecord[]
} = recorded
export const unitNames: Record<string, string> = {
  add: 'Add/subtract',
  mul: 'Multiply',
  fma: 'True FMA',
  nonfused: 'Nonfused multiply-add',
}
const a = 0x3c01,
  b = 0x3c01,
  c = 0xbc02
const rounded = floatToFP16Bits(1.0001)
export const numericalExamples = {
  conversion: {
    original: '1.0001',
    bits: rounded,
    value: fp16BitsToFloat(rounded),
  },
  fma: {
    a,
    b,
    c,
    fused: fmaFP16(a, b, c),
    nonfused: nonfusedMultiplyAddFP16(a, b, c),
  },
}
/** Relative reductions use nonfused as the denominator, under the same recorded configuration. */
export function fusedComparison(
  records: readonly ShowcaseRecord[] = recorded.records,
) {
  const fused = records.find(
    (r) => r.unit === 'fma' && r.variant === 'baseline',
  )
  const nonfused = records.find(
    (r) => r.unit === 'nonfused' && r.variant === 'baseline',
  )
  if (!fused || !nonfused || nonfused.area <= 0) return null
  return {
    fused,
    nonfused,
    areaReduction: (100 * (nonfused.area - fused.area)) / nonfused.area,
    delayReduction:
      fused.delay !== null && nonfused.delay !== null && nonfused.delay > 0
        ? (100 * (nonfused.delay - fused.delay)) / nonfused.delay
        : null,
  }
}
export const formatMetric = (n: number) =>
  new Intl.NumberFormat('en-US', { maximumFractionDigits: 6 }).format(n)
