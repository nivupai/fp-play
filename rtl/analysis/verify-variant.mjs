import { strict as assert } from 'node:assert'
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { fileURLToPath, URL } from 'node:url'
import { resolve } from 'node:path'
import process from 'node:process'
import console from 'node:console'
const root = fileURLToPath(new URL('../../', import.meta.url)),
  build = resolve(root, 'rtl/build/analysis/verification')
mkdirSync(build, { recursive: true })
const hash = (p) =>
  createHash('sha256')
    .update(readFileSync(resolve(root, p)))
    .digest('hex')
const baseline = JSON.parse(
  readFileSync(resolve(root, 'rtl/build/mul-fma/verification.json')),
)
for (const [p, h] of Object.entries(baseline.sourceHashes))
  assert.equal(hash(p), h, `Stale baseline ${p}`)
assert.equal(baseline.mismatches, 0)
assert.equal(baseline.cases, 301544)
const sources = [
  'fp16_addsub.sv',
  'fp16_mul_fma_core.sv',
  'fp16_mul.sv',
  'fp16_fma.sv',
  'fp16_mul_fma_compare.sv',
].map((p) => resolve(root, 'rtl/src', p))
const variant = 'rtl/variants/tree-leading-bit/fp16_round_grid48.sv'
const verilator = process.env.VERILATOR ?? 'verilator'
const args = [
  '--Wall',
  '--top-module',
  'fp16_mul_fma_compare',
  ...sources,
  resolve(root, variant),
]
execFileSync(verilator, ['--lint-only', ...args], { stdio: 'inherit' })
execFileSync(
  verilator,
  [
    '--cc',
    '--exe',
    '--build',
    '--trace',
    ...args,
    '-Mdir',
    build,
    '-o',
    'variant_sim',
    '-CFLAGS',
    '-std=c++17',
    resolve(root, 'rtl/sim/mul_fma.cpp'),
  ],
  { stdio: 'inherit' },
)
for (const prefix of ['', 'directed-']) {
  const vectors = resolve(root, `rtl/build/mul-fma/${prefix}vectors.txt`),
    observations = resolve(build, prefix + 'observations.txt')
  execFileSync(
    resolve(build, 'variant_sim'),
    [
      vectors,
      observations,
      ...(prefix ? [resolve(build, 'directed.vcd')] : []),
    ],
    { stdio: 'inherit' },
  )
  assert.equal(
    readFileSync(observations, 'utf8'),
    readFileSync(
      resolve(root, `rtl/build/mul-fma/${prefix}observations.txt`),
      'utf8',
    ),
    'Every result, flag and recorded internal signal must match, without NaN exemptions',
  )
}
const report = {
  variant: 'tree-leading-bit',
  cases: baseline.cases,
  directed: baseline.counts.directed,
  randomized: baseline.counts.randomized,
  seed: baseline.seed,
  mismatches: 0,
  proof: false,
  method:
    'Full original triples; strict raw results, all flags and all recorded signals',
  verilator: execFileSync(verilator, ['--version'], {
    encoding: 'utf8',
  }).trim(),
  sourceHashes: {
    ...baseline.sourceHashes,
    [variant]: hash(variant),
    'rtl/analysis/verify-variant.mjs': hash('rtl/analysis/verify-variant.mjs'),
  },
  vectorsSha256: hash('rtl/build/mul-fma/vectors.txt'),
  observationsSha256: hash('rtl/build/mul-fma/observations.txt'),
}
writeFileSync(
  resolve(build, 'verification.json'),
  JSON.stringify(report, null, 2) + '\n',
)
console.log(JSON.stringify(report, null, 2))
