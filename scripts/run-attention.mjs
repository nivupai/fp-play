// Deterministic, development-only experiment export. No timing/power/silicon claims.
import { writeFileSync } from 'node:fs'
import process from 'node:process'
import console from 'node:console'
import { withArithmetic } from './arithmetic-runtime.mjs'
const destination = process.argv[2]
if (!destination)
  throw new Error('Usage: node scripts/run-attention.mjs /path/to/results.json')
await withArithmetic(
  async ({
    ATTENTION_EXAMPLES,
    ATTENTION_POLICIES,
    calculateAttention,
    attentionValues,
    binary64Attention,
    seededRandom,
  }) => {
    const seed = 0x6174746e,
      random = seededRandom(seed)
    const cases = ATTENTION_EXAMPLES.map((e) => ({
      id: e.id,
      input: e.input,
      phenomenon: e.phenomenon,
      expected: e.expected,
      checks: e.checks,
    }))
    for (let i = 0; i < 12; i++) {
      const nq = 1 + (random() % 3),
        nk = 1 + (random() % 4),
        dk = 1 + (random() % 4),
        dv = 1 + (random() % 3)
      const matrix = (rows, cols) =>
        Array.from({ length: rows }, () =>
          Array.from({ length: cols }, () => ((random() % 1025) - 512) / 256),
        )
      cases.push({
        id: `seeded-${i}`,
        input: { q: matrix(nq, dk), k: matrix(nk, dk), v: matrix(nk, dv) },
      })
    }
    const records = cases.map((c) => {
      const fp32 = calculateAttention(c.input, 'fp32', 'stable'),
        binary64 = binary64Attention(c.input)
      const results = Object.keys(ATTENTION_POLICIES).flatMap((mode) =>
        ['stable', 'naive'].map((method) => {
          const result = calculateAttention(c.input, mode, method)
          const error = attentionValues(result.output).map((row, i) =>
            row.map((x, j) =>
              Number.isFinite(x) && Number.isFinite(binary64.output[i][j])
                ? {
                    absolute: Math.abs(x - binary64.output[i][j]),
                    relative:
                      binary64.output[i][j] === 0
                        ? null
                        : Math.abs(
                            (x - binary64.output[i][j]) / binary64.output[i][j],
                          ),
                  }
                : { absolute: null, relative: null },
            ),
          )
          return { ...result, errorAgainstBinary64: error }
        }),
      )
      return {
        ...c,
        references: {
          fp32: {
            scores: attentionValues(fp32.scores),
            logits: attentionValues(fp32.logits),
            probabilities: attentionValues(fp32.probabilities),
            output: attentionValues(fp32.output),
          },
          binary64,
        },
        results,
      }
    })
    const artifact = {
      schema: 1,
      seed,
      metadata: {
        model: 'FP Play Milestone 9',
        node: process.version,
        policies: ATTENTION_POLICIES,
        exp: 'Math.exp binary64 approximation, then explicit quantization',
        scale: 'binary64 1/Math.sqrt(d_k), then explicit quantization',
        reference:
          'stable modeled FP32 and separate binary64 numerical baseline; neither is exact mathematical attention',
        numberEncoding:
          'Nonfinite Numbers and negative zero are strings NaN, Infinity, -Infinity, -0; BigInt fields are decimal strings.',
        order: 'sequential',
        cases: records.length,
        calculations:
          records.length * Object.keys(ATTENTION_POLICIES).length * 2,
      },
      records,
    }
    writeFileSync(
      destination,
      JSON.stringify(
        artifact,
        (_, v) =>
          typeof v === 'bigint'
            ? v.toString()
            : typeof v === 'number' && (!Number.isFinite(v) || Object.is(v, -0))
              ? Object.is(v, -0)
                ? '-0'
                : String(v)
              : v,
        2,
      ) + '\n',
    )
    console.log(
      `Exported ${records.length} deterministic cases / ${artifact.metadata.calculations} calculations to ${destination}`,
    )
  },
)
