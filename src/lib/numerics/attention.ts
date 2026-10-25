import { calculateFloat } from './formatArithmetic'
import {
  decodeFloat,
  encodeFloat,
  emptyFlags,
  numberToDyadic,
  roundDyadic,
  unionFlags,
} from './formats'
import type { FloatValue, FormatName } from './formats'
import type { ArithmeticFlags } from './fp16Arithmetic'
import { dotProduct } from './dotProduct'
import type { DotMode } from './dotProduct'
import { divideForAttention } from './attentionDivision'

export type Matrix = readonly (readonly number[])[]
export interface AttentionInput {
  q: Matrix
  k: Matrix
  v: Matrix
}
export type AttentionMode =
  'fp16' | 'fp16-wide' | 'bf16-wide' | 'fp32' | 'fp16-softmax'
export type SoftmaxMethod = 'stable' | 'naive'
export interface AttentionPolicy {
  readonly label: string
  readonly input: FormatName
  readonly scoreDot: DotMode
  readonly reduction: 'sequential'
  readonly scale: FormatName
  readonly scaleSource: 'binary64-reciprocal-sqrt'
  readonly softmax: FormatName
  readonly exp: 'Math.exp-binary64-then-quantize'
  readonly probability: FormatName
  readonly valueDot: DotMode
  readonly output: FormatName
}
const policy = (
  label: string,
  input: FormatName,
  dot: DotMode,
  work: FormatName,
  softmax: FormatName,
): AttentionPolicy =>
  Object.freeze({
    label,
    input,
    scoreDot: dot,
    reduction: 'sequential',
    scale: work,
    scaleSource: 'binary64-reciprocal-sqrt',
    softmax,
    exp: 'Math.exp-binary64-then-quantize',
    probability: input,
    valueDot: dot,
    output: input,
  })
export const ATTENTION_POLICIES = Object.freeze({
  fp16: policy('FP16 inputs / FP16 accumulation', 'FP16', 'A', 'FP16', 'FP16'),
  'fp16-wide': policy(
    'FP16 inputs / FP32 FMA accumulation',
    'FP16',
    'C',
    'FP32',
    'FP32',
  ),
  'bf16-wide': policy(
    'BF16 inputs / FP32 FMA accumulation',
    'BF16',
    'D',
    'FP32',
    'FP32',
  ),
  fp32: policy(
    'FP32 inputs / FP32 multiply + add (reference)',
    'FP32',
    'E',
    'FP32',
    'FP32',
  ),
  'fp16-softmax': policy(
    'FP16 + FP32 accumulation / reduced FP16 softmax',
    'FP16',
    'C',
    'FP32',
    'FP16',
  ),
} satisfies Record<AttentionMode, AttentionPolicy>)
export type AttentionStage =
  | 'input'
  | 'score'
  | 'scale'
  | 'softmax-input'
  | 'shift'
  | 'exp'
  | 'denominator'
  | 'normalize'
  | 'probability'
  | 'aggregate'
  | 'output'
export type AttentionOperation =
  | 'quantize'
  | 'add'
  | 'multiply'
  | 'fma'
  | 'divide'
  | 'subtract maximum'
  | 'Math.exp → quantize'
export interface AttentionEvent {
  readonly stage: AttentionStage
  readonly row: number
  readonly column: number
  readonly operation: AttentionOperation
  readonly operands: readonly FloatValue[]
  readonly result: FloatValue
  readonly before: number | null
  readonly flags: ArithmeticFlags
  readonly halfway: boolean
  readonly increment: boolean
  readonly note: string
}
export interface SoftmaxResult {
  readonly probabilities: readonly FloatValue[]
  readonly shifted: readonly FloatValue[]
  readonly exponentials: readonly FloatValue[]
  readonly denominator: FloatValue
  readonly maximum: number | null
  readonly events: readonly AttentionEvent[]
  readonly issues: readonly string[]
}
const numeric = (v: FloatValue) => decodeFloat(v.format, v.bits).value
export function quantizeAttention(
  stage: AttentionStage,
  row: number,
  column: number,
  format: FormatName,
  value: number,
  note: string,
): AttentionEvent {
  const exact = numberToDyadic(value)
  const rounded = exact
    ? roundDyadic(format, exact, Object.is(value, -0) ? 1 : 0)
    : null
  return {
    stage,
    row,
    column,
    operation: 'quantize',
    operands: [],
    result: { format, bits: rounded?.bits ?? encodeFloat(format, value) },
    before: value,
    flags: rounded?.flags ?? emptyFlags(),
    halfway: rounded?.halfway ?? false,
    increment: rounded?.increment ?? false,
    note,
  }
}
/** 1–8 logits; all arithmetic boundaries are explicit, including exact-rational normalization.
 * Math.exp is a binary64 host approximation, NOT correctly rounded format exponentiation.
 */
export function attentionSoftmax(
  logits: readonly FloatValue[],
  work: FormatName,
  probability: FormatName,
  method: SoftmaxMethod,
  row = 0,
): SoftmaxResult {
  if (
    !['stable', 'naive'].includes(method) ||
    logits.length < 1 ||
    logits.length > 8 ||
    !['FP16', 'BF16', 'FP32'].includes(work) ||
    !['FP16', 'BF16', 'FP32'].includes(probability)
  )
    throw new RangeError(
      'Unsupported softmax configuration or row length (1–8)',
    )
  const events: AttentionEvent[] = [],
    issues: string[] = []
  const cast = logits.map((v, column) => {
    const event = quantizeAttention(
      'softmax-input',
      row,
      column,
      work,
      numeric(v),
      'Round scaled logit to softmax working precision.',
    )
    events.push(event)
    return event.result
  })
  const maximum = method === 'stable' ? Math.max(...cast.map(numeric)) : null
  const max: FloatValue = {
    format: work,
    bits: encodeFloat(work, maximum ?? 0),
  }
  const shifted = cast.map((value, column) => {
    if (method === 'naive') return value
    const negMax = { format: work, bits: encodeFloat(work, -numeric(max)) }
    const result = calculateFloat(work, 'add', value, negMax)
    events.push({
      stage: 'shift',
      row,
      column,
      operation: 'subtract maximum',
      operands: [value, max],
      result,
      before: null,
      flags: result.flags,
      halfway: result.halfway,
      increment: result.increment,
      note: 'Exact modeled subtraction, then one nearest-even rounding. Nonfinite max subtraction propagates NaN; no fallback.',
    })
    return result
  })
  const exponentials = shifted.map((value, column) => {
    const argument = numeric(value),
      native = Math.exp(argument)
    if (Number.isFinite(argument) && native === Infinity)
      issues.push(
        `Row ${row + 1}: binary64 Math.exp overflow at column ${column + 1}.`,
      )
    if (Number.isFinite(argument) && native === 0)
      issues.push(
        `Row ${row + 1}: binary64 Math.exp underflow to zero at column ${column + 1}.`,
      )
    const event = quantizeAttention(
      'exp',
      row,
      column,
      work,
      native,
      'Math.exp(binary64 argument), then quantize to working format. Flags describe conversion only, not transcendental conformance.',
    )
    events.push({
      ...event,
      operation: 'Math.exp → quantize',
      operands: [value],
    })
    return event.result
  })
  let denominator: FloatValue = { format: work, bits: 0 }
  for (let column = 0; column < exponentials.length; column++) {
    const result = calculateFloat(
      work,
      'add',
      denominator,
      exponentials[column],
    )
    events.push({
      stage: 'denominator',
      row,
      column,
      operation: 'add',
      operands: [denominator, exponentials[column]],
      result,
      before: null,
      flags: result.flags,
      halfway: result.halfway,
      increment: result.increment,
      note: 'Left-to-right rounded denominator sum, starting +0.',
    })
    denominator = result
  }
  if (numeric(denominator) === 0)
    issues.push(
      `Row ${row + 1}: normalization denominator is zero; division results are retained.`,
    )
  if (!Number.isFinite(numeric(denominator)))
    issues.push(
      `Row ${row + 1}: nonfinite normalization denominator; no replacement distribution.`,
    )
  const probabilities = exponentials.map((value, column) => {
    const result = divideForAttention(work, value, denominator)
    events.push({
      stage: 'normalize',
      row,
      column,
      operation: 'divide',
      operands: [value, denominator],
      result,
      before: null,
      flags: result.flags,
      halfway: result.halfway,
      increment: result.increment,
      note: `Exact rational quotient rounded once to ${work}.${result.divideByZero ? ' Divide by zero.' : ''}`,
    })
    const event = quantizeAttention(
      'probability',
      row,
      column,
      probability,
      numeric(result),
      'Quantize softmax output once before weighted-value multiplication. No renormalization after quantization.',
    )
    events.push(event)
    return event.result
  })
  if (probabilities.some((v) => !Number.isFinite(numeric(v))))
    issues.push(
      `Row ${row + 1}: nonfinite probabilities propagate to value aggregation.`,
    )
  return {
    probabilities,
    shifted,
    exponentials,
    denominator,
    maximum,
    events,
    issues,
  }
}
export function validateAttentionInput(input: AttentionInput): {
  queries: number
  keys: number
  depth: number
  values: number
} {
  const dimensions = (
    matrix: Matrix,
    name: string,
    maxRows: number,
    maxCols: number,
  ) => {
    if (
      !Array.isArray(matrix) ||
      !matrix.length ||
      matrix.length > maxRows ||
      !Array.isArray(matrix[0]) ||
      !matrix[0].length ||
      matrix[0].length > maxCols
    )
      throw new RangeError(
        `${name}: require 1–${maxRows} rows and 1–${maxCols} columns`,
      )
    const cols = matrix[0].length
    if (
      Array.from(matrix).some(
        (row) =>
          !Array.isArray(row) ||
          row.length !== cols ||
          Array.from(row).some((v) => typeof v !== 'number'),
      )
    )
      throw new TypeError(
        `${name}: use a rectangular matrix of Numbers (explicit NaN/Infinity allowed)`,
      )
    return [matrix.length, cols]
  }
  const [queries, depth] = dimensions(input.q, 'Q', 4, 8),
    [keys, kd] = dimensions(input.k, 'K', 8, 8),
    [vr, values] = dimensions(input.v, 'V', 8, 4)
  if (depth !== kd || keys !== vr)
    throw new RangeError(
      'Q/K feature dimensions must match; V rows must equal K rows',
    )
  return { queries, keys, depth, values }
}
export interface AttentionResult {
  readonly mode: AttentionMode
  readonly method: SoftmaxMethod
  readonly policy: AttentionPolicy
  readonly quantized: {
    q: readonly (readonly FloatValue[])[]
    k: readonly (readonly FloatValue[])[]
    v: readonly (readonly FloatValue[])[]
  }
  readonly scores: readonly (readonly FloatValue[])[]
  readonly logits: readonly (readonly FloatValue[])[]
  readonly probabilities: readonly (readonly FloatValue[])[]
  readonly output: readonly (readonly FloatValue[])[]
  readonly rows: readonly SoftmaxResult[]
  readonly scale: FloatValue
  readonly events: readonly AttentionEvent[]
  readonly flags: ArithmeticFlags
  readonly issues: readonly string[]
}
export function calculateAttention(
  input: AttentionInput,
  mode: AttentionMode = 'fp32',
  method: SoftmaxMethod = 'stable',
): AttentionResult {
  const { depth, values } = validateAttentionInput(input)
  if (
    !Object.hasOwn(ATTENTION_POLICIES, mode) ||
    !['stable', 'naive'].includes(method)
  )
    throw new RangeError(
      'Unsupported attention policy; use a named mode and stable/naive softmax',
    )
  const policy = ATTENTION_POLICIES[mode],
    events: AttentionEvent[] = []
  const quantize = (matrix: Matrix, name: string) =>
    matrix.map((row, i) =>
      row.map((value, j) => {
        const event = quantizeAttention(
          'input',
          i,
          j,
          policy.input,
          value,
          `${name} input: direct binary64 conversion.`,
        )
        events.push(event)
        return event.result
      }),
    )
  const quantized = {
    q: quantize(input.q, 'Q'),
    k: quantize(input.k, 'K'),
    v: quantize(input.v, 'V'),
  }
  const dot = (
    a: readonly FloatValue[],
    b: readonly FloatValue[],
    dotMode: DotMode,
    stage: 'score' | 'aggregate',
    row: number,
    column: number,
  ): FloatValue => {
    const result = dotProduct(a.map(numeric), b.map(numeric), dotMode)
    for (const step of result.steps)
      events.push({
        stage,
        row,
        column,
        operation: step.operation,
        operands: step.operands,
        result: step.result,
        before: null,
        flags: step.result.flags,
        halfway: step.result.halfway,
        increment: step.result.increment,
        note: `Terms ${step.terms.map((i) => i + 1).join(', ')}; ${dotMode} sequential reduction. ${step.operation === 'fma' ? 'Exact product retained inside the FMA.' : 'Product and addition boundaries are separately rounded.'}`,
      })
    return result.result
  }
  const scores = quantized.q.map((q, i) =>
    quantized.k.map((k, j) => dot(q, k, policy.scoreDot, 'score', i, j)),
  )
  const scaleEvent = quantizeAttention(
    'scale',
    0,
    0,
    policy.scale,
    1 / Math.sqrt(depth),
    'Compute 1 / Math.sqrt(d_k) in binary64, then quantize this constant; not correctly rounded format sqrt/reciprocal.',
  )
  events.push(scaleEvent)
  const scale = scaleEvent.result
  const logits = scores.map((row, i) =>
    row.map((score, j) => {
      const result = calculateFloat(policy.scale, 'multiply', score, scale)
      events.push({
        stage: 'scale',
        row: i,
        column: j,
        operation: 'multiply',
        operands: [score, scale],
        result,
        before: null,
        flags: result.flags,
        halfway: result.halfway,
        increment: result.increment,
        note: 'Multiply by the quantized reciprocal-square-root constant; one destination rounding.',
      })
      return result
    }),
  )
  const rows = logits.map((row, i) =>
    attentionSoftmax(row, policy.softmax, policy.probability, method, i),
  )
  rows.forEach((row) => events.push(...row.events))
  const probabilities = rows.map((row) => row.probabilities)
  const output = probabilities.map((row, i) =>
    Array.from({ length: values }, (_, j) => {
      const accumulated = dot(
        row,
        quantized.v.map((r) => r[j]),
        policy.valueDot,
        'aggregate',
        i,
        j,
      )
      const event = quantizeAttention(
        'output',
        i,
        j,
        policy.output,
        numeric(accumulated),
        'Final output quantization after the modeled weighted-value dot product.',
      )
      events.push(event)
      return event.result
    }),
  )
  const issues = rows.flatMap((row) => row.issues)
  if (events.some((event) => event.flags.overflow))
    issues.push(
      'At least one modeled arithmetic/conversion stage overflowed; inspect stage flags.',
    )
  if (events.some((event) => event.flags.underflow))
    issues.push(
      'At least one modeled arithmetic/conversion stage raised underflow; gradual underflow is enabled.',
    )
  if (
    output.some((row) =>
      Array.from(row).some((v) => !Number.isFinite(numeric(v))),
    )
  )
    issues.push(
      'Nonfinite final output. No clamping, sanitizing, or probability fallback was applied.',
    )
  return {
    mode,
    method,
    policy,
    quantized,
    scores,
    logits,
    probabilities,
    output,
    rows,
    scale,
    events,
    flags: unionFlags(...events.map((event) => event.flags)),
    issues,
  }
}
export function attentionValues(
  matrix: readonly (readonly FloatValue[])[],
): number[][] {
  return matrix.map((row) => row.map(numeric))
}
/** Separate binary64 numerical baseline. Not exact attention or a high-precision oracle. */
export function binary64Attention(input: AttentionInput): {
  scores: number[][]
  logits: number[][]
  probabilities: number[][]
  output: number[][]
} {
  const { depth, values } = validateAttentionInput(input)
  const dot = (a: readonly number[], b: readonly number[]) =>
    a.reduce((sum, value, i) => sum + value * b[i], 0)
  const scores = input.q.map((q) => input.k.map((k) => dot(q, k))),
    logits = scores.map((row) => row.map((x) => x * (1 / Math.sqrt(depth))))
  const probabilities = logits.map((row) => {
    const max = Math.max(...row),
      exps = row.map((x) => Math.exp(x - max)),
      denom = exps.reduce((a, b) => a + b, 0)
    return exps.map((x) => x / denom)
  })
  return {
    scores,
    logits,
    probabilities,
    output: probabilities.map((row) =>
      Array.from({ length: values }, (_, j) =>
        dot(
          row,
          input.v.map((v) => v[j]),
        ),
      ),
    ),
  }
}
