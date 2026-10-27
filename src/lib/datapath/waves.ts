import { flagMask, hashes, integer, object, text } from './artifacts'
import { validateWaveData } from './vcd'
import type { WaveData } from './vcd'
import type { RecordedExample } from './artifacts'
export interface RecordedWave {
  family: 'add' | 'product'
  tool: string
  cases: number
  samples: number
  sampleTimes: number[]
  download: string
  sourceHashes: Record<string, string>
  data: WaveData
}
export function parseWaves(input: unknown): RecordedWave[] {
  const a = object(input, 'waveform artifact')
  if (
    a.schemaVersion !== 1 ||
    !Array.isArray(a.groups) ||
    a.groups.length !== 2
  )
    throw new Error('Invalid waveform schema')
  const groups = a.groups.map((value) => {
    const g = object(value, 'waveform group')
    if (g.family !== 'add' && g.family !== 'product')
      throw new Error('Invalid waveform family')
    if (g.download !== `artifacts/datapath-${g.family}.vcd`)
      throw new Error('Unexpected waveform download path')
    const samples = integer(g.samples, 1, 256, 'waveform samples')
    if (!Array.isArray(g.sampleTimes) || g.sampleTimes.length !== samples)
      throw new Error('Missing waveform sample indices')
    const sampleTimes = g.sampleTimes.map((v, i) => {
      const time = integer(v, 0, 10000, 'sample index')
      if (time !== 2 * i + 1)
        throw new Error('Unexpected settled observation index')
      return time
    })
    return {
      family: g.family,
      tool: text(g.tool, 'waveform simulator'),
      cases: integer(g.cases, 1, 10000000, 'waveform corpus'),
      samples,
      sampleTimes,
      download: g.download,
      sourceHashes: hashes(g.sourceHashes),
      data: validateWaveData(g.data),
    } as RecordedWave
  })
  if (new Set(groups.map((g) => g.family)).size !== 2)
    throw new Error('Duplicate waveform family')
  return groups
}
export function valueAt(
  wave: RecordedWave,
  time: number,
  name: string,
): string | undefined {
  return wave.data.frames.filter((f) => f.time <= time).at(-1)?.values[name]
}
export function validateCorrelation(
  waves: RecordedWave[],
  examples: RecordedExample[],
): void {
  for (const wave of waves) {
    const family = examples.filter((r) => r.family === wave.family)
    if (new Set(family.map((r) => r.index)).size !== wave.samples)
      throw new Error('Waveform sample count differs from RTL artifact')
    for (const example of family) {
      const time = wave.sampleTimes[example.index],
        name =
          example.operation === 'multiply'
            ? 'multiply_result'
            : example.operation === 'fma'
              ? 'fused_result'
              : example.operation === 'nonfused'
                ? 'nonfused_result'
                : 'result'
      const maskName =
        example.operation === 'multiply'
          ? 'multiply_flags'
          : example.operation === 'fma'
            ? 'fused_flags'
            : 'nonfused_flags'
      if (wave.family === 'product') {
        const bits = valueAt(wave, time, maskName)
        if (
          !bits ||
          !/^[01]+$/.test(bits) ||
          BigInt('0b' + bits) !== BigInt(flagMask(example.flags))
        )
          throw new Error('Waveform flags differ from recorded RTL')
      } else {
        for (const key of [
          'invalid',
          'overflow',
          'underflow',
          'inexact',
        ] as const) {
          const bits = valueAt(wave, time, key)
          if (bits !== (example.flags[key] ? '1' : '0'))
            throw new Error('Waveform flags differ from recorded RTL')
        }
      }
      const value = valueAt(wave, time, name)
      if (
        !value ||
        !/^[01]+$/.test(value) ||
        BigInt('0b' + value) !== BigInt(example.bits)
      )
        throw new Error('Waveform result differs from recorded RTL')
      for (const [i, operand] of example.operands.entries()) {
        const bits = valueAt(wave, time, ['a', 'b', 'c'][i])
        if (
          !bits ||
          !/^[01]+$/.test(bits) ||
          BigInt('0b' + bits) !== BigInt(operand)
        )
          throw new Error('Waveform operand differs from recorded RTL')
      }
    }
  }
}
