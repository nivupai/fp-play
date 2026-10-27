/** Offline VCD subset reader. It never runs or interpolates an RTL simulation. */
export interface WaveSignal {
  name: string
  width: number
}
export interface WaveFrame {
  time: number
  values: Record<string, string>
}
export interface WaveData {
  timescale: string
  signals: WaveSignal[]
  frames: WaveFrame[]
}
export function curateVCD(
  source: string,
  selected: readonly string[],
): WaveData {
  if (
    source.length > 2000000 ||
    selected.length > 32 ||
    new Set(selected).size !== selected.length
  )
    throw new Error('VCD subset exceeds bounds or repeats names')
  const scopes: string[] = [],
    ids = new Map<string, WaveSignal>(),
    signals: WaveSignal[] = [],
    frames: WaveFrame[] = [],
    values: Record<string, string> = {}
  let header = true,
    time: number | null = null,
    timescale = ''
  const flush = () => {
    if (time !== null) {
      for (const s of signals)
        if (values[s.name] === undefined)
          throw new Error(`Missing initial VCD value ${s.name}`)
      frames.push({ time, values: { ...values } })
    }
  }
  for (const raw of source.split(/\r?\n/)) {
    const line = raw.trim()
    if (header) {
      const scale = line.match(/^\$timescale\s+(.+?)\s+\$end$/)
      if (scale) timescale = scale[1]
      const scope = line.match(/^\$scope\s+\w+\s+(\S+)\s+\$end/)
      if (scope) scopes.push(scope[1])
      if (line.startsWith('$upscope')) scopes.pop()
      const declaration = line.match(/^\$var\s+\w+\s+(\d+)\s+(\S+)\s+(\S+)/)
      if (
        declaration &&
        scopes.length === 1 &&
        scopes[0] === 'TOP' &&
        selected.includes(declaration[3])
      ) {
        const width = Number(declaration[1])
        if (!Number.isInteger(width) || width < 1 || width > 128)
          throw new Error('Unsupported VCD width')
        const signal = { name: declaration[3], width }
        if (signals.some((s) => s.name === signal.name))
          throw new Error('Duplicate VCD signal')
        ids.set(declaration[2], signal)
        signals.push(signal)
      }
      if (line.startsWith('$enddefinitions')) header = false
      continue
    }
    if (/^#\d+$/.test(line)) {
      const next = Number(line.slice(1))
      if (!Number.isSafeInteger(next) || (time !== null && next <= time))
        throw new Error('Non-monotone VCD observation index')
      flush()
      time = next
      if (frames.length > 1024) throw new Error('Too many curated VCD frames')
      continue
    }
    const match =
      line.match(/^b([01xz]+)\s+(\S+)$/i) ?? line.match(/^([01xz])(\S+)$/i)
    if (match && ids.has(match[2])) {
      const signal = ids.get(match[2])!,
        bits = match[1].toLowerCase()
      if (time === null || bits.length > signal.width)
        throw new Error('Invalid VCD signal change')
      values[signal.name] = bits.padStart(
        signal.width,
        /^[xz]/.test(bits) ? bits[0] : '0',
      )
    }
  }
  flush()
  if (
    header ||
    !timescale ||
    !frames.length ||
    selected.some((name) => !signals.some((s) => s.name === name))
  )
    throw new Error('Incomplete VCD or missing selected signals')
  return { timescale, signals, frames }
}
/** Validate the exported subset independently of the raw VCD parser. */
export function validateWaveData(input: unknown): WaveData {
  if (!input || typeof input !== 'object')
    throw new Error('Invalid waveform data')
  const a = input as Partial<WaveData>
  if (
    typeof a.timescale !== 'string' ||
    !a.timescale ||
    !Array.isArray(a.signals) ||
    !a.signals.length ||
    a.signals.length > 32 ||
    !Array.isArray(a.frames) ||
    !a.frames.length ||
    a.frames.length > 1024
  )
    throw new Error('Invalid waveform dimensions')
  const names = new Set<string>()
  for (const s of a.signals) {
    if (
      !s ||
      typeof s.name !== 'string' ||
      !s.name ||
      names.has(s.name) ||
      !Number.isInteger(s.width) ||
      s.width < 1 ||
      s.width > 128
    )
      throw new Error('Invalid waveform signal')
    names.add(s.name)
  }
  let last = -1
  for (const f of a.frames) {
    if (
      !f ||
      !Number.isSafeInteger(f.time) ||
      f.time <= last ||
      !f.values ||
      typeof f.values !== 'object'
    )
      throw new Error('Invalid waveform frame')
    last = f.time
    for (const s of a.signals)
      if (
        typeof f.values[s.name] !== 'string' ||
        f.values[s.name].length !== s.width ||
        !/^[01xz]+$/.test(f.values[s.name])
      )
        throw new Error('Invalid waveform bits')
  }
  return { timescale: a.timescale, signals: a.signals, frames: a.frames }
}
