/** Portable, bounded validation. No filesystem, HDL tools, or approximate numerical model. */
import type { ArithmeticFlags } from '../numerics/fp16Arithmetic'
export type Operation = 'add' | 'subtract' | 'multiply' | 'fma' | 'nonfused'
export type Stage =
  | 'unpack'
  | 'exponent'
  | 'align'
  | 'arithmetic'
  | 'normalize'
  | 'grs'
  | 'round'
  | 'pack'
export interface Provenance {
  artifact: string
  tool: string
  cases: number
  seed: number
  sourceHashes: Record<string, string>
}
export interface Signal {
  name: string
  width: number
  value: bigint
  signed: boolean
}
export interface Phase {
  kind: 'add' | 'multiply' | 'fma'
  label: string
  operands: number[]
  bits: number
  flags: ArithmeticFlags
  signals: Record<string, Signal>
  bypassed: boolean | null
}
export interface RecordedExample {
  id: string
  name: string
  operation: Operation
  index: number
  family: 'add' | 'product'
  operands: number[]
  bits: number
  flags: ArithmeticFlags
  phases: Phase[]
  provenance: Provenance
}
export const STAGES: { id: Stage; title: string; description: string }[] = [
  {
    id: 'unpack',
    title: 'Unpack',
    description:
      'Classify raw operands and extract integer significands. Software operand decoding is separately labeled.',
  },
  {
    id: 'exponent',
    title: 'Exponent processing',
    description:
      'Process exponent differences or the exact product scale; coordinates differ between the software trace and fixed-grid RTL.',
  },
  {
    id: 'align',
    title: 'Exact alignment',
    description:
      'Place magnitudes on the RTL fixed grid without rounding. Multiplication has no C addend.',
  },
  {
    id: 'arithmetic',
    title: 'Significand arithmetic',
    description:
      'Perform exact integer addition/subtraction or 11 × 11 multiplication. True FMA accumulates the exact product before rounding.',
  },
  {
    id: 'normalize',
    title: 'Normalize',
    description:
      'Find the leading bit after arithmetic or cancellation. This is a conceptual function, not a hardware clock stage.',
  },
  {
    id: 'grs',
    title: 'Guard / round / sticky',
    description:
      'Inspect retained bits and discarded-tail information from recorded signals.',
  },
  {
    id: 'round',
    title: 'Nearest-even rounding',
    description:
      'Inspect the recorded increment and exponent. Special values bypass ordinary finite rounding.',
  },
  {
    id: 'pack',
    title: 'Pack result / flags',
    description:
      'Read the recorded output encoding and per-operation exception indicators.',
  },
]
const addFields: Record<string, [number, Stage]> = {
  effectiveBSign: [1, 'exponent'],
  exponentDifference: [6, 'exponent'],
  alignedA: [40, 'align'],
  alignedB: [40, 'align'],
  magnitude: [41, 'arithmetic'],
  normalizationShift: [6, 'normalize'],
  normalizedMagnitude: [41, 'normalize'],
  discardCount: [6, 'grs'],
  retained: [12, 'grs'],
  guard: [1, 'grs'],
  round: [1, 'grs'],
  sticky: [1, 'grs'],
  increment: [1, 'round'],
  exponentRounded: [5, 'round'],
}
const productFields: Record<string, [number, Stage]> = {
  significandA: [11, 'unpack'],
  significandB: [11, 'unpack'],
  productSign: [1, 'exponent'],
  productExponent: [8, 'exponent'],
  exactProduct: [22, 'arithmetic'],
  alignedProduct: [80, 'align'],
  alignedAddend: [64, 'align'],
  arithmeticMagnitude: [81, 'arithmetic'],
  resultSign: [1, 'arithmetic'],
  normalizationShift: [7, 'normalize'],
  discardCount: [7, 'grs'],
  retained: [12, 'grs'],
  guard: [1, 'grs'],
  round: [1, 'grs'],
  sticky: [1, 'grs'],
  increment: [1, 'round'],
  exponentRounded: [6, 'round'],
  tiny: [1, 'round'],
}
export function expectedFields(phase: Phase, stage: Stage): string[] {
  const fields = phase.kind === 'add' ? addFields : productFields
  const names = Object.keys(fields).filter((k) => fields[k][1] === stage)
  if (stage === 'unpack' && phase.kind === 'add')
    return ['significandA', 'significandB']
  return names
}
export function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error(`${label}: expected object`)
  return value as Record<string, unknown>
}
export function text(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value.trim() || value.length > 1000)
    throw new Error(`${label}: expected bounded nonempty text`)
  return value
}
export function integer(
  value: unknown,
  min: number,
  max: number,
  label: string,
): number {
  if (
    typeof value !== 'number' ||
    !Number.isInteger(value) ||
    value < min ||
    value > max
  )
    throw new Error(`${label}: integer outside ${min}…${max}`)
  return value
}
export function hashes(value: unknown): Record<string, string> {
  const o = object(value, 'source hashes')
  if (!Object.keys(o).length || Object.keys(o).length > 128)
    throw new Error('Missing or excessive source hashes')
  return Object.fromEntries(
    Object.entries(o).map(([p, v]) => {
      if (
        !/^[\w./-]+$/.test(p) ||
        p.startsWith('/') ||
        typeof v !== 'string' ||
        !/^[a-f0-9]{64}$/i.test(v)
      )
        throw new Error('Invalid source hash')
      return [p, v]
    }),
  )
}
export function flagMask(flags: ArithmeticFlags): number {
  return (
    (flags.invalid ? 8 : 0) |
    (flags.overflow ? 4 : 0) |
    (flags.underflow ? 2 : 0) |
    (flags.inexact ? 1 : 0)
  )
}
export function maskFlags(value: unknown): ArithmeticFlags {
  const m = integer(value, 0, 15, 'exception mask')
  return {
    invalid: !!(m & 8),
    overflow: !!(m & 4),
    underflow: !!(m & 2),
    inexact: !!(m & 1),
  }
}
function flagObject(value: unknown): ArithmeticFlags {
  const o = object(value, 'flags')
  for (const k of ['invalid', 'overflow', 'underflow', 'inexact'])
    if (typeof o[k] !== 'boolean') throw new Error(`Invalid flag ${k}`)
  return {
    invalid: o.invalid as boolean,
    overflow: o.overflow as boolean,
    underflow: o.underflow as boolean,
    inexact: o.inexact as boolean,
  }
}
function observed(
  value: unknown,
  kind: Phase['kind'],
): { signals: Phase['signals']; bypassed: boolean | null } {
  if (value === undefined || value === null)
    return { signals: {}, bypassed: null }
  const o = object(value, 'recorded signals'),
    fields = kind === 'add' ? addFields : productFields,
    signals: Phase['signals'] = {}
  for (const [name, [width]] of Object.entries(fields))
    if (o[name] !== undefined) {
      const raw = o[name],
        signed = name === 'productExponent'
      let bits: bigint
      if (typeof raw === 'string') {
        if (raw.length > 32 || !/^(?:0x)?[0-9a-f]+$/i.test(raw))
          throw new Error(`Invalid hex signal ${name}`)
        bits = BigInt(raw.toLowerCase().startsWith('0x') ? raw : '0x' + raw)
      } else
        bits = BigInt(
          integer(raw, signed ? -128 : 0, signed ? 127 : 2 ** width - 1, name),
        )
      if (
        bits < (signed ? -(1n << BigInt(width - 1)) : 0n) ||
        bits >= 1n << BigInt(signed ? width - 1 : width)
      )
        throw new Error(`Out-of-range signal ${name}`)
      signals[name] = { name, width, value: bits, signed }
    }
  let bypassed: boolean | null = null
  if (o.special !== undefined) {
    if (typeof o.special === 'boolean') bypassed = o.special
    else bypassed = !!integer(o.special, 0, 1, 'special bypass')
  }
  return { signals, bypassed }
}
const rawBits = (v: unknown) => integer(v, 0, 65535, 'raw FP16 pattern')
export function parseSimulationArtifact(
  value: unknown,
  family: 'add' | 'product',
): RecordedExample[] {
  const artifact = object(value, 'simulation artifact'),
    meta = object(artifact.metadata, 'metadata')
  if (meta.schema !== undefined && meta.schema !== 1)
    throw new Error('Unsupported simulation schema')
  if (meta.mismatches !== 0)
    throw new Error('Artifact is not a zero-mismatch verified run')
  const sourceHashes = hashes(meta.sourceHashes),
    normalized: Record<string, string> = {}
  for (const [p, h] of Object.entries(sourceHashes)) {
    if (family === 'add') {
      if (p.startsWith('../src/') || p.startsWith('../scripts/'))
        normalized[p.slice(3)] = h
      else {
        if (p.includes('..')) throw new Error('Unexpected provenance path')
        normalized['rtl/' + p] = h
      }
    } else {
      if (p.includes('..')) throw new Error('Unexpected provenance path')
      normalized[p] = h
    }
  }
  if (
    !normalized[
      family === 'add'
        ? 'rtl/src/fp16_addsub.sv'
        : 'rtl/src/fp16_mul_fma_core.sv'
    ]
  )
    throw new Error('Missing core RTL provenance')
  const provenance: Provenance = {
    artifact: `rtl/vectors/${family === 'add' ? 'simulation' : 'mul-fma-simulation'}.json`,
    tool: text(meta.tool, 'simulator'),
    cases: integer(meta.cases, 1, 10000000, 'corpus count'),
    seed: integer(meta.seed, 0, 0xffffffff, 'seed'),
    sourceHashes: normalized,
  }
  if (!Array.isArray(artifact.samples) || artifact.samples.length > 256)
    throw new Error('Invalid sample list')
  return artifact.samples.flatMap<RecordedExample>((value, index) => {
    const s = object(value, 'sample'),
      name = text(s.name, 'example name'),
      a = rawBits(s.a),
      b = rawBits(s.b)
    if (family === 'add') {
      if (s.operation !== 'add' && s.operation !== 'subtract')
        throw new Error('Unsupported add/sub operation')
      const bits = rawBits(s.rtlBits),
        flags = flagObject(s.rtlFlags),
        phase: Phase = {
          kind: 'add',
          label: s.operation === 'add' ? 'FP16 addition' : 'FP16 subtraction',
          operands: [a, b],
          bits,
          flags,
          ...observed(s.signals, 'add'),
        }
      return [
        {
          id: `add:${index}:${s.operation}`,
          name,
          operation: s.operation,
          index,
          family,
          operands: [a, b],
          bits,
          flags,
          phases: [phase],
          provenance,
        },
      ]
    }
    const c = rawBits(s.c),
      multiplyBits = rawBits(s.multiply),
      multiplyFlags = maskFlags(s.multiplyFlags),
      multiply: Phase = {
        kind: 'multiply',
        label: 'Rounded FP16 multiplier',
        operands: [a, b],
        bits: multiplyBits,
        flags: multiplyFlags,
        ...observed(s.multiplySignals, 'multiply'),
      }
    return (['multiply', 'fma', 'nonfused'] as const).map((operation) => {
      const bits =
          operation === 'multiply'
            ? multiplyBits
            : rawBits(operation === 'fma' ? s.fused : s.nonfused),
        flags =
          operation === 'multiply'
            ? multiplyFlags
            : maskFlags(operation === 'fma' ? s.fusedFlags : s.nonfusedFlags)
      const phases: Phase[] =
        operation === 'multiply'
          ? [multiply]
          : operation === 'fma'
            ? [
                {
                  kind: 'fma',
                  label: 'True fused datapath · one final rounding',
                  operands: [a, b, c],
                  bits,
                  flags,
                  ...observed(s.fmaSignals, 'fma'),
                },
              ]
            : [
                multiply,
                {
                  kind: 'add',
                  label: 'Adder after rounded product · second rounding',
                  operands: [multiplyBits, c],
                  bits,
                  flags: maskFlags(s.additionFlags),
                  signals: {},
                  bypassed: null,
                },
              ]
      return {
        id: `product:${index}:${operation}`,
        name,
        operation,
        index,
        family,
        operands: operation === 'multiply' ? [a, b] : [a, b, c],
        bits,
        flags,
        phases,
        provenance,
      }
    })
  })
}
export function signalBinary(signal: Signal): string {
  return BigInt.asUintN(signal.width, signal.value)
    .toString(2)
    .padStart(signal.width, '0')
}
