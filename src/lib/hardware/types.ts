import type artifact from '../../../rtl/vectors/hardware-analysis.json'
export interface HardwareTiming {
  delayNs: number
  startpoint: string
  endpoint: string
  cells: { pin: string; type: string; arcDelayNs: number; arrivalNs: number }[]
  report: string
  stageBreakdown: null
  stageBreakdownUnavailable: string
}
type RecordedRecord = (typeof artifact.records)[number]
export type HardwareRecord = Omit<
  RecordedRecord,
  'timing' | 'timingUnavailable'
> & { timing: HardwareTiming | null; timingUnavailable: string | null }
export type HardwareArtifact = Omit<typeof artifact, 'records' | 'gitDirty'> & {
  gitDirty: boolean
  records: HardwareRecord[]
}
