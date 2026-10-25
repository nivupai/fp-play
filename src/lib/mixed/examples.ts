import type { DotMode, ReductionOrder } from '../numerics/dotProduct'
export interface MixedExample {
  title: string
  explanation: string
  a: number[]
  b: number[]
  mode: DotMode
  order: ReductionOrder
  expected: number
}
export const MIXED_EXAMPLES: readonly MixedExample[] = [
  {
    title: 'FP16 versus BF16 near 1',
    explanation:
      '1 + 2^-10 survives FP16 input conversion but BF16 rounds it to 1. Compare C and D.',
    a: [1 + 2 ** -10],
    b: [1],
    mode: 'D',
    order: 'sequential',
    expected: 1,
  },
  {
    title: 'FP16 overflow, BF16 finite',
    explanation:
      '65536 overflows FP16 inputs; BF16 represents it exactly. Compare C and D.',
    a: [65536],
    b: [1],
    mode: 'D',
    order: 'sequential',
    expected: 65536,
  },
  {
    title: 'BF16 rounding loss',
    explanation: '1 + 2^-8 is an exact BF16 midpoint; the even result is 1.',
    a: [1 + 2 ** -8],
    b: [1],
    mode: 'D',
    order: 'sequential',
    expected: 1,
  },
  {
    title: 'FP32 precision near 1',
    explanation:
      '1 + 2^-24 is an FP32 midpoint and becomes 1 during input quantization.',
    a: [1 + 2 ** -24],
    b: [1],
    mode: 'E',
    order: 'sequential',
    expected: 1,
  },
  {
    title: 'Subnormal / normal boundary',
    explanation:
      'Largest FP16 subnormal plus its minimum quantum reaches the smallest normal exactly.',
    a: [1023 * 2 ** -24, 2 ** -24],
    b: [1, 1],
    mode: 'A',
    order: 'sequential',
    expected: 2 ** -14,
  },
  {
    title: 'Catastrophic cancellation',
    explanation:
      'The addition to 2048 discards 1 before subtraction cancels the large terms. Compare A and C.',
    a: [2048, 1, -2048],
    b: [1, 1, 1],
    mode: 'A',
    order: 'sequential',
    expected: 0,
  },
  {
    title: 'Sequential loses small increments',
    explanation: 'Two additions of 1 at 2048 each tie down to the even value.',
    a: [2048, 1, 1],
    b: [1, 1, 1],
    mode: 'A',
    order: 'sequential',
    expected: 2048,
  },
  {
    title: 'Pairwise saves increments',
    explanation: 'The tree adds 1 + 1 first, then 2048 + 2 = 2050.',
    a: [2048, 1, 1],
    b: [1, 1, 1],
    mode: 'A',
    order: 'pairwise',
    expected: 2050,
  },
  {
    title: 'Rounded FP16 products',
    explanation:
      '(1 + 2^-10)^2 loses 2^-20 before the FP32 sum. Widening after multiplication cannot restore it.',
    a: [1 + 2 ** -10, -(1 + 2 ** -9)],
    b: [1 + 2 ** -10, 1],
    mode: 'B',
    order: 'sequential',
    expected: 0,
  },
  {
    title: 'FP32 FMA accumulation',
    explanation:
      'An exact FP16 product enters the FP32 accumulator. The residual 2^-20 survives. Compare B and C.',
    a: [1 + 2 ** -10, -(1 + 2 ** -9)],
    b: [1 + 2 ** -10, 1],
    mode: 'C',
    order: 'sequential',
    expected: 2 ** -20,
  },
  {
    title: 'Wider accumulator helps',
    explanation:
      'FP32 accumulation retains the unit lost by FP16 accumulation. Compare A and B.',
    a: [2048, 1, -2048],
    b: [1, 1, 1],
    mode: 'B',
    order: 'sequential',
    expected: 1,
  },
  {
    title: 'Input loss is irreversible',
    explanation:
      '1 + 2^-12 becomes 1 in FP16 before either accumulator sees it; A, B, and C all lose the residual.',
    a: [1 + 2 ** -12, -1],
    b: [1, 1],
    mode: 'C',
    order: 'sequential',
    expected: 0,
  },
]
