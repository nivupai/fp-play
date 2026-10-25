import type {
  AttentionInput,
  AttentionMode,
  SoftmaxMethod,
} from '../numerics/attention'
export interface AttentionExpectation {
  mode: AttentionMode
  method: SoftmaxMethod
  stage: 'scores' | 'logits' | 'probabilities' | 'output'
  row: number
  column: number
  relation: 'equal' | 'positive' | 'nan'
  value?: number
}
export interface AttentionExample {
  id: string
  title: string
  phenomenon: string
  expected: string
  verification: string
  input: AttentionInput
  left: AttentionMode
  right: AttentionMode
  method: SoftmaxMethod
  checks: readonly AttentionExpectation[]
}
const check = (
  mode: AttentionMode,
  stage: AttentionExpectation['stage'],
  column: number,
  value: number,
  method: SoftmaxMethod = 'stable',
): AttentionExpectation => ({
  mode,
  method,
  stage,
  row: 0,
  column,
  relation: 'equal',
  value,
})
const invalid = (
  mode: AttentionMode,
  method: SoftmaxMethod,
): AttentionExpectation => ({
  mode,
  method,
  stage: 'output',
  row: 0,
  column: 0,
  relation: 'nan',
})
const positive = (
  mode: AttentionMode,
  stage: AttentionExpectation['stage'],
  column = 0,
): AttentionExpectation => ({
  mode,
  method: 'stable',
  stage,
  row: 0,
  column,
  relation: 'positive',
})
const cancellation: AttentionInput = {
  q: [[2048, 1, -2048, 0]],
  k: [
    [1, 1, 1, 0],
    [0, 0, 0, 0],
  ],
  v: [[1], [0]],
}
const verification =
  'Independent Fraction rounding at every modeled boundary; Python math.exp host check; 100-digit Decimal stable-attention reference for the mathematical phenomenon (not exact transcendental equality).'
export const ATTENTION_EXAMPLES: readonly AttentionExample[] = [
  {
    id: 'stable-naive',
    title: 'Stable versus naive',
    phenomenon:
      'Binary64 exp(1000) overflows before format conversion in the naive path.',
    expected: 'Naive FP32 output is NaN; stable output is finite and positive.',
    verification,
    input: { q: [[1]], k: [[1000], [1001]], v: [[0], [1]] },
    left: 'fp32',
    right: 'fp16-wide',
    method: 'naive',
    checks: [invalid('fp32', 'naive'), positive('fp32', 'output')],
  },
  {
    id: 'exp-overflow',
    title: 'FP16 exponential overflow',
    phenomenon:
      'Math.exp(12) is finite in binary64, but exceeds FP16 range on conversion.',
    expected:
      'Naive FP16 output is NaN; stable maximum subtraction keeps exp arguments nonpositive.',
    verification,
    input: { q: [[1]], k: [[12], [11]], v: [[1], [0]] },
    left: 'fp16',
    right: 'fp16-wide',
    method: 'naive',
    checks: [invalid('fp16', 'naive'), positive('fp16', 'output')],
  },
  {
    id: 'dynamic-range',
    title: 'BF16 versus FP16 range',
    phenomenon: '65536 is finite in BF16 but overflows the FP16 input format.',
    expected: 'Stable FP16-wide output is NaN; BF16-wide returns 1.',
    verification,
    input: { q: [[65536]], k: [[1], [0]], v: [[1], [0]] },
    left: 'fp16-wide',
    right: 'bf16-wide',
    method: 'stable',
    checks: [
      invalid('fp16-wide', 'stable'),
      check('bf16-wide', 'output', 0, 1),
    ],
  },
  {
    id: 'qk-accumulation',
    title: 'QK accumulation width',
    phenomenon:
      'FP16 loses the unit in 2048 + 1 − 2048; FP32 accumulation retains it.',
    expected:
      'Raw score 0 in FP16 versus 1 with FP32 FMA; final probabilities differ.',
    verification,
    input: cancellation,
    left: 'fp16',
    right: 'fp16-wide',
    method: 'stable',
    checks: [
      check('fp16', 'scores', 0, 0),
      check('fp16-wide', 'scores', 0, 1),
      check('fp16', 'output', 0, 0.5),
    ],
  },
  {
    id: 'near-tie',
    title: 'Near-tied scores',
    phenomenon:
      'Two keys differ by 2^-12 near 1; FP16 input conversion erases the difference.',
    expected: 'FP16-wide scores tie at 1; FP32 scores retain 1 + 2^-12.',
    verification,
    input: { q: [[1]], k: [[1], [1 + 2 ** -12]], v: [[0], [1]] },
    left: 'fp16-wide',
    right: 'fp32',
    method: 'stable',
    checks: [
      check('fp16-wide', 'scores', 1, 1),
      check('fp32', 'scores', 1, 1 + 2 ** -12),
    ],
  },
  {
    id: 'lost-probability',
    title: 'Probability quantization loss',
    phenomenon:
      'A small nonzero softmax tail is rounded to zero in FP16 before value aggregation.',
    expected:
      'Second FP16 probability and output are zero; FP32 retains a positive tail and output.',
    verification,
    input: { q: [[1]], k: [[0], [-20]], v: [[0], [65504]] },
    left: 'fp16-wide',
    right: 'fp32',
    method: 'stable',
    checks: [
      check('fp16-wide', 'probabilities', 1, 0),
      check('fp16-wide', 'output', 0, 0),
      positive('fp32', 'probabilities', 1),
      positive('fp32', 'output'),
    ],
  },
  {
    id: 'large-logits',
    title: 'Large-magnitude logits',
    phenomenon:
      'Stable softmax depends on differences; subtracting a large common offset avoids exp overflow.',
    expected:
      'Scores 10000 and 9999 give finite stable FP32 output; naive output is NaN.',
    verification,
    input: { q: [[1]], k: [[10000], [9999]], v: [[1], [0]] },
    left: 'fp32',
    right: 'bf16-wide',
    method: 'stable',
    checks: [
      check('fp32', 'scores', 0, 10000),
      positive('fp32', 'output'),
      invalid('fp32', 'naive'),
    ],
  },
  {
    id: 'dot-cancellation',
    title: 'Product cancellation',
    phenomenon:
      'FP16 product rounding removes a residual before cancellation; FP32 FMA preserves it.',
    expected:
      'First score is 0 in FP16, 2^-20 in FP16-wide. Later probability quantization can hide this residual.',
    verification,
    input: {
      q: [[1 + 2 ** -10, -(1 + 2 ** -9), 0, 0]],
      k: [
        [1 + 2 ** -10, 1, 0, 0],
        [0, 0, 0, 0],
      ],
      v: [[1], [0]],
    },
    left: 'fp16',
    right: 'fp16-wide',
    method: 'stable',
    checks: [
      check('fp16', 'scores', 0, 0),
      check('fp16-wide', 'scores', 0, 2 ** -20),
    ],
  },
  {
    id: 'scaling',
    title: 'Scaling error',
    phenomenon:
      'For d_k = 3, 1/sqrt(3) is approximated in binary64 and then quantized to the scaling format.',
    expected:
      'The FP16 scaled score is 1.7314453125, distinct from the mathematical sqrt(3).',
    verification,
    input: {
      q: [[1, 1, 1]],
      k: [
        [1, 1, 1],
        [0, 0, 0],
      ],
      v: [[1], [0]],
    },
    left: 'fp16',
    right: 'fp32',
    method: 'stable',
    checks: [check('fp16', 'logits', 0, 1.7314453125)],
  },
  {
    id: 'value-aggregation',
    title: 'Value aggregation error',
    phenomenon:
      'Uniform FP16 weights multiply [2048, 1, −2048]. FP16 partial sums lose different bits than FP32 FMA.',
    expected:
      'FP16 output is 0.5; FP16-wide output is 0.333251953125, with identical quantized probabilities.',
    verification,
    input: { q: [[0]], k: [[1], [1], [1]], v: [[2048], [1], [-2048]] },
    left: 'fp16',
    right: 'fp16-wide',
    method: 'stable',
    checks: [
      check('fp16', 'output', 0, 0.5),
      check('fp16-wide', 'output', 0, 0.333251953125),
    ],
  },
  {
    id: 'output-impact',
    title: 'Precision changes output',
    phenomenon:
      'Cancellation-sensitive scores weight opposing large values, magnifying the score error.',
    expected: 'FP16 output cancels to zero; FP16-wide output is positive.',
    verification,
    input: { ...cancellation, v: [[1000], [-1000]] },
    left: 'fp16',
    right: 'fp16-wide',
    method: 'stable',
    checks: [check('fp16', 'output', 0, 0), positive('fp16-wide', 'output')],
  },
  {
    id: 'input-loss',
    title: 'Widening cannot restore inputs',
    phenomenon:
      'A key increment lost during FP16 conversion is unavailable to either accumulator width.',
    expected: 'Both FP16 paths output 512; FP32 retains the key increment.',
    verification,
    input: { q: [[1]], k: [[1], [1 + 2 ** -12]], v: [[0], [1024]] },
    left: 'fp16',
    right: 'fp16-wide',
    method: 'stable',
    checks: [
      check('fp16', 'output', 0, 512),
      check('fp16-wide', 'output', 0, 512),
      check('fp32', 'scores', 1, 1 + 2 ** -12),
    ],
  },
]
