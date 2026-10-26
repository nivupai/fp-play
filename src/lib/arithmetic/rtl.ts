import artifact from '../../../rtl/vectors/simulation.json'
import { calculateFP16 } from '../numerics/fp16Arithmetic'
import type { FP16Operation } from '../numerics/fp16Arithmetic'

export const RTL_METADATA = artifact.metadata
export const RTL_SIMULATIONS = artifact.samples.map((sample) => {
  if (sample.operation !== 'add' && sample.operation !== 'subtract')
    throw new Error('Invalid operation in recorded RTL artifact')
  return { ...sample, operation: sample.operation as FP16Operation }
})
export type RTLSample = (typeof RTL_SIMULATIONS)[number]
/** Browser comparison is software versus a recorded RTL result, never live HDL execution. */
export function compareRecordedRTL(sample: RTLSample) {
  const software = calculateFP16(sample.a, sample.b, sample.operation)
  const flagsAgree = (
    ['invalid', 'overflow', 'underflow', 'inexact'] as const
  ).every((key) => software.flags[key] === sample.rtlFlags[key])
  return {
    software,
    bitsAgree: software.bits === sample.rtlBits,
    flagsAgree,
    agrees: software.bits === sample.rtlBits && flagsAgree,
  }
}
