// Development-only differential verification. Reuses the optional public SoftFloat build.
import { execFileSync } from 'node:child_process'
import console from 'node:console'
import process from 'node:process'
import { resolve } from 'node:path'
import { withArithmetic } from './arithmetic-runtime.mjs'
const driver = process.argv[2]
if (!driver)
  throw new Error(
    'Usage: node scripts/validate-fp32-softfloat.mjs /path/to/softfloat-driver',
  )
await withArithmetic(async ({ calculateFloat, decodeFloat, seededRandom }) => {
  const rows = [],
    directed = [
      0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x00800001,
      0x3f7fffff, 0x3f800000, 0x3f800001, 0x40000000, 0x7f7fffff, 0xff7fffff,
      0x7f800000, 0xff800000, 0x7f800001, 0xff800001, 0x7fc00001, 0xffc00002,
    ]
  for (const a of directed)
    for (const b of directed) {
      rows.push([a, b, 4, 0], [a, b, 5, 0])
      for (const c of directed) rows.push([a, b, 6, c])
    }
  // All exponent boundaries and precision-only tininess at the minimum normal.
  for (let exponent = 1; exponent < 255; exponent++) {
    const edge = exponent * 2 ** 23
    for (const a of [edge - 1, edge, edge + 1])
      for (const b of [1, 0x3f000000, 0x3f7fffff, 0x3f800001, 0xbf800000])
        rows.push([a, b, 4, 0], [a, b, 5, 0], [a, b, 6, 0x80800000])
  }
  const random = seededRandom(0x8f32bf16)
  for (let i = 0; i < 50000; i++) {
    const a = random(),
      b = random(),
      c = random()
    rows.push([a, b, 4, 0], [a, b, 5, 0], [a, b, 6, c])
  }
  const lines = execFileSync(resolve(driver), [], {
    input:
      rows
        .map(
          ([a, b, op, c]) =>
            `${a.toString(16)} ${b.toString(16)} ${op} ${c.toString(16)}`,
        )
        .join('\n') + '\n',
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
  })
    .trim()
    .split('\n')
  if (lines.length !== rows.length) throw new Error('Output count mismatch')
  let nanResults = 0,
    nanPolicyDifferences = 0
  const counts = { add: 0, multiply: 0, fma: 0 }
  rows.forEach(([a, b, op, c], index) => {
    const operation = op === 4 ? 'add' : op === 5 ? 'multiply' : 'fma'
    counts[operation]++
    const result = calculateFloat(
      'FP32',
      operation,
      { format: 'FP32', bits: a },
      { format: 'FP32', bits: b },
      { format: 'FP32', bits: c },
    )
    const [raw, flagText] = lines[index].split(' '),
      expected = Number.parseInt(raw, 16),
      flags = Number(flagText)
    const mask =
      (result.flags.invalid ? 16 : 0) |
      (result.flags.overflow ? 4 : 0) |
      (result.flags.underflow ? 2 : 0) |
      (result.flags.inexact ? 1 : 0)
    const nan = decodeFloat('FP32', expected).classification === 'nan',
      d = decodeFloat('FP32', result.bits)
    if (nan) {
      nanResults++
      if (result.bits !== expected) nanPolicyDifferences++
    }
    if (
      (nan
        ? d.classification !== 'nan' || !d.isQuietNaN
        : result.bits !== expected) ||
      mask !== flags
    )
      throw new Error(
        `Mismatch ${index}: ${[a, b, c].map((x) => x.toString(16))} ${operation}: ${result.bits.toString(16)}/${mask} expected ${raw}/${flags}`,
      )
  })
  console.log(
    JSON.stringify(
      {
        comparisons: rows.length,
        counts,
        mismatches: 0,
        nanResults,
        nanPolicyDifferences,
        rounding: 'nearest-even',
        tininess: 'after-rounding',
        nanComparison:
          'quiet classification and all flags; sign/payload selection is implementation-specific',
      },
      null,
      2,
    ),
  )
})
