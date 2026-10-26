import { strict as assert } from 'node:assert'
import { createHash } from 'node:crypto'
import { execFileSync, spawnSync } from 'node:child_process'
import console from 'node:console'
import { mkdirSync, readFileSync, writeFileSync, readdirSync } from 'node:fs'
import { resolve } from 'node:path'
import process from 'node:process'
import { fileURLToPath, URL } from 'node:url'
import { withArithmetic } from '../../scripts/arithmetic-runtime.mjs'
const root = fileURLToPath(new URL('../../', import.meta.url))
const build = resolve(root, 'rtl/build/mul-fma')
mkdirSync(build, { recursive: true })
const directedOnly = process.argv.includes('--directed')
const waveform = process.argv.includes('--waveform')
const exportIndex = process.argv.indexOf('--export')
if (exportIndex >= 0)
  assert.ok(process.argv[exportIndex + 1], '--export requires a destination')
for (let i = 2; i < process.argv.length; i++) {
  if (i === exportIndex + 1 && exportIndex >= 0) continue
  assert.ok(
    ['--directed', '--waveform', '--export'].includes(process.argv[i]),
    'Unknown argument',
  )
}
assert.ok(!waveform || directedOnly, 'Waveforms require --directed')
const directed = JSON.parse(
  readFileSync(resolve(root, 'rtl/vectors/mul-fma-directed.json')),
)
export const mask = (f) =>
  (f.invalid ? 8 : 0) |
  (f.overflow ? 4 : 0) |
  (f.underflow ? 2 : 0) |
  (f.inexact ? 1 : 0)
const keys = [
  'significandA',
  'significandB',
  'productSign',
  'productExponent',
  'exactProduct',
  'alignedProduct',
  'alignedAddend',
  'arithmeticMagnitude',
  'resultSign',
  'special',
  'normalizationShift',
  'discardCount',
  'retained',
  'guard',
  'round',
  'sticky',
  'increment',
  'exponentRounded',
  'tiny',
]
const wide = new Set(['alignedProduct', 'alignedAddend', 'arithmeticMagnitude'])
const hex = (n) => n.toString(16)
const seed = 0x4d554c46
await withArithmetic(
  async ({
    multiplyFP16,
    fmaFP16,
    nonfusedMultiplyAddFP16,
    referenceMultiply,
    seededRandom,
  }) => {
    const cases = directed.map(({ a, b, c }) => ({ a, b, c }))
    const counts = {
      directed: cases.length,
      exhaustive: 0,
      special: 0,
      randomized: 0,
    }
    if (!directedOnly) {
      for (let a = 0; a < 65536; a++) {
        cases.push(
          { a, b: 0x3c00, c: 0 },
          { a, b: 0x3c00, c: a ^ 0x8000 },
          { a, b: 0, c: a },
        )
        counts.exhaustive += 3
      }
      const special = [
        0, 0x8000, 1, 0x8001, 0x3ff, 0x400, 0x8400, 0x3c00, 0xbc00, 0x7bff,
        0xfbff, 0x7c00, 0xfc00, 0x7e11, 0xfe22, 0x7c33, 0xfc44,
      ]
      for (const a of special)
        for (const b of special)
          for (const c of special) {
            cases.push({ a, b, c })
            counts.special++
          }
      const random = seededRandom(seed)
      for (let i = 0; i < 100000; i++) {
        cases.push({
          a: random() & 65535,
          b: random() & 65535,
          c: random() & 65535,
        })
        counts.randomized++
      }
    }
    const reference = ({ a, b, c }) => {
      const m = multiplyFP16(a, b),
        f = fmaFP16(a, b, c),
        n = nonfusedMultiplyAddFP16(a, b, c)
      for (const [result, independent] of [
        [m, referenceMultiply(a, b)],
        [f, referenceMultiply(a, b, c)],
      ]) {
        assert.equal(
          result.bits,
          independent.bits,
          'Independent integer-lattice result',
        )
        assert.deepEqual(
          result.flags,
          independent.flags,
          'Independent integer-lattice flags',
        )
      }
      return { m, f, n }
    }
    const vectors = cases.map((v, i) => {
      const { m, f, n } = reference(v)
      if (i < directed.length) {
        const d = directed[i]
        for (const [key, value] of Object.entries({
          multiply: m.bits,
          fused: f.bits,
          nonfused: n.bits,
          multiplyFlags: mask(m.flags),
          fusedFlags: mask(f.flags),
          nonfusedFlags: mask(n.flags),
        }))
          if (key in d) assert.equal(value, d[key], `${d.name}: ${key}`)
      }
      return [
        v.a,
        v.b,
        v.c,
        m.bits,
        mask(m.flags),
        f.bits,
        mask(f.flags),
        n.bits,
        mask(n.flags),
        mask(n.addition.flags),
      ]
        .map(hex)
        .join(' ')
    })
    const prefix = directedOnly ? 'directed-' : ''
    const vectorPath = resolve(build, prefix + 'vectors.txt'),
      observationPath = resolve(build, prefix + 'observations.txt')
    writeFileSync(vectorPath, vectors.join('\n') + '\n')
    const run = spawnSync(
      resolve(build, 'obj/fp16_mul_fma_sim'),
      [
        vectorPath,
        observationPath,
        ...(waveform ? [resolve(build, 'directed.vcd')] : []),
      ],
      { encoding: 'utf8' },
    )
    if (run.error) throw run.error
    console.log(run.stdout)
    const rows = readFileSync(observationPath, 'utf8').trim().split('\n')
    assert.equal(rows.length, cases.length)
    let signalComparisons = 0
    const samples = [],
      failures = []
    const units = (exact) => {
      const x = exact.coefficient << BigInt(exact.exponent + 48)
      return x < 0n ? -x : x
    }
    for (let i = 0; i < rows.length; i++) {
      const tokens = rows[i].split(' ')
      try {
        assert.equal(tokens.length, 48)
        assert.equal(
          tokens.slice(0, 10).join(' '),
          vectors[i],
          'Raw outputs and every exception flag',
        )
        const { m, f } = reference(cases[i])
        const signals = []
        for (const [j, result] of [m, f].entries()) {
          const signal = Object.fromEntries(
            keys.map((key, k) => [
              key,
              wide.has(key)
                ? '0x' + tokens[10 + j * 19 + k]
                : parseInt(tokens[10 + j * 19 + k], 16),
            ]),
          )
          if (signal.productExponent >= 128) signal.productExponent -= 256
          if (result.exact) {
            signalComparisons++
            assert.equal(signal.special, 0)
            const mul = result.trace.find((s) => s.kind === 'multiply')
            assert.equal(BigInt(signal.significandA), mul.significandA)
            assert.equal(BigInt(signal.significandB), mul.significandB)
            assert.equal(signal.productSign, mul.sign)
            assert.equal(signal.productExponent, mul.exponent)
            assert.equal(signal.resultSign, result.bits >>> 15)
            assert.equal(BigInt(signal.exactProduct), mul.product)
            assert.equal(
              BigInt(signal.alignedProduct),
              units(result.exactProduct),
            )
            assert.equal(
              BigInt(signal.arithmeticMagnitude),
              units(result.exact),
            )
            assert.equal(
              BigInt(signal.alignedAddend),
              j === 0
                ? 0n
                : units({
                    coefficient: result.trace.find(
                      (s) => s.kind === 'arithmetic',
                    ).signedB,
                    exponent: result.trace.find((s) => s.kind === 'arithmetic')
                      .exponent,
                  }),
            )
            const norm = result.trace.find((s) => s.kind === 'normalize'),
              grs = result.trace.find((s) => s.kind === 'grs'),
              round = result.trace.find((s) => s.kind === 'round')
            const magnitude = units(result.exact)
            const leading =
              magnitude === 0n ? 0 : magnitude.toString(2).length - 1
            assert.equal(
              signal.normalizationShift,
              magnitude === 0n ? 0 : 80 - leading,
            )
            assert.equal(signal.discardCount, Math.max(24, leading - 10))
            assert.equal(
              signal.exponentRounded,
              round.finalSignificand < 1024n && round.finalExponent === -14
                ? 0
                : round.finalExponent + 15,
            )
            assert.equal(BigInt(signal.retained), norm.retained)
            for (const k of ['guard', 'round', 'sticky'])
              assert.equal(signal[k], grs[k])
            assert.equal(!!signal.increment, round.increment)
            assert.equal(!!signal.tiny, round.tinyAfterPrecisionRounding)
          } else assert.equal(signal.special, 1)
          signals.push(signal)
        }
        if (i < directed.length)
          samples.push({
            name: directed[i].name,
            ...cases[i],
            multiply: parseInt(tokens[3], 16),
            multiplyFlags: parseInt(tokens[4], 16),
            fused: parseInt(tokens[5], 16),
            fusedFlags: parseInt(tokens[6], 16),
            nonfused: parseInt(tokens[7], 16),
            nonfusedFlags: parseInt(tokens[8], 16),
            additionFlags: parseInt(tokens[9], 16),
            multiplySignals: signals[0],
            fmaSignals: signals[1],
          })
      } catch (error) {
        if (failures.length < 8)
          failures.push({ index: i, ...cases[i], message: error.message })
      }
    }
    writeFileSync(
      resolve(build, prefix + 'failures.json'),
      JSON.stringify(failures, null, 2) + '\n',
    )
    if (failures.length) {
      writeFileSync(
        resolve(build, 'failing-vectors.txt'),
        failures.map((f) => vectors[f.index]).join('\n') + '\n',
      )
      spawnSync(resolve(build, 'obj/fp16_mul_fma_sim'), [
        resolve(build, 'failing-vectors.txt'),
        resolve(build, 'failing-observations.txt'),
        resolve(build, 'failing.vcd'),
      ])
      throw new Error(JSON.stringify(failures))
    }
    assert.equal(run.status, 0, run.stderr)
    const sources = [
      ...readdirSync(resolve(root, 'rtl/src')).map((n) => 'rtl/src/' + n),
      'rtl/mul-fma.mk',
      'rtl/sim/mul_fma.cpp',
      'rtl/scripts/verify-mul-fma.mjs',
      'rtl/vectors/mul-fma-directed.json',
      'src/lib/numerics/fp16Multiply.ts',
      'src/lib/numerics/roundDyadicFP16.ts',
      'src/tests/reference/fp16MultiplyReference.ts',
    ]
    const metadata = {
      tool: execFileSync(process.env.VERILATOR ?? 'verilator', ['--version'], {
        encoding: 'utf8',
      }).trim(),
      seed,
      counts,
      cases: cases.length,
      mismatches: 0,
      signalComparisons,
      comparison:
        'Strict raw results including NaN metadata; all four flags; exact product, alignment, normalization and rounding observations',
      sourceHashes: Object.fromEntries(
        sources.map((p) => [
          p,
          createHash('sha256')
            .update(readFileSync(resolve(root, p)))
            .digest('hex'),
        ]),
      ),
    }
    writeFileSync(
      resolve(build, prefix + 'verification.json'),
      JSON.stringify(metadata, null, 2) + '\n',
    )
    if (exportIndex >= 0) {
      assert.ok(!directedOnly, 'Export only after complete verification')
      writeFileSync(
        resolve(process.argv[exportIndex + 1]),
        JSON.stringify({ metadata, samples }, null, 2) + '\n',
      )
    }
    console.log(JSON.stringify(metadata, null, 2))
  },
)
