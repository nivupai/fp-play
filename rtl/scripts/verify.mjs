// RTL observes vectors; the unchanged TypeScript engine supplies expectations.
import { strict as assert } from 'node:assert'
import { createHash } from 'node:crypto'
import { execFileSync, spawnSync } from 'node:child_process'
import console from 'node:console'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import process from 'node:process'
import { fileURLToPath, URL } from 'node:url'
import { withArithmetic } from '../../scripts/arithmetic-runtime.mjs'
const root = fileURLToPath(new URL('../', import.meta.url)),
  build = join(root, 'build')
mkdirSync(build, { recursive: true })
const args = process.argv.slice(2),
  directedOnly = args.includes('--directed'),
  waveform = args.includes('--waveform')
if (waveform && !directedOnly)
  throw new Error(
    'Use --directed --waveform for bounded trace capture; the simulator also supports explicit vector replay.',
  )
const prefix = directedOnly ? 'directed-' : ''
const exportIndex = args.indexOf('--export'),
  exportPath = exportIndex < 0 ? null : args[exportIndex + 1]
if (exportIndex >= 0 && !exportPath)
  throw new Error('--export requires a destination JSON path')
if (
  args.some(
    (arg, i) =>
      !['--directed', '--waveform', '--export'].includes(arg) &&
      !(exportIndex >= 0 && i === exportIndex + 1),
  )
)
  throw new Error('Unknown argument')
const directed = JSON.parse(
  readFileSync(join(root, 'vectors/directed.json'), 'utf8'),
)
const hex = (n) => n.toString(16)
const flagBits = (f) =>
  (f.invalid ? 8 : 0) |
  (f.overflow ? 4 : 0) |
  (f.underflow ? 2 : 0) |
  (f.inexact ? 1 : 0)
const flags = (mask) => ({
  invalid: !!(mask & 8),
  overflow: !!(mask & 4),
  underflow: !!(mask & 2),
  inexact: !!(mask & 1),
})
const seed = 0x46503136
await withArithmetic(async ({ calculateFP16, seededRandom }) => {
  const cases = directed.map((v) => ({ a: v.a, b: v.b, subtract: v.subtract })),
    counts = { directed: directed.length, exhaustive: 0, randomized: 0 }
  if (!directedOnly) {
    for (let a = 0; a < 65536; a++) {
      cases.push(
        { a, b: 0, subtract: false },
        { a, b: 0, subtract: true },
        { a, b: a, subtract: true },
        { a, b: a ^ 0x8000, subtract: false },
      )
      counts.exhaustive += 4
    }
    const random = seededRandom(seed)
    for (let i = 0; i < 100000; i++) {
      const a = random() & 65535,
        b = random() & 65535
      cases.push({ a, b, subtract: false }, { a, b, subtract: true })
      counts.randomized += 2
    }
  }
  // Keep only compact expected encodings during generation; recompute each
  // reference trace while comparing so the exhaustive corpus has bounded heap use.
  const expected = cases.map((v) => {
    const result = calculateFP16(v.a, v.b, v.subtract ? 'subtract' : 'add')
    return { bits: result.bits, flags: result.flags }
  })
  directed.forEach((v, i) => {
    assert.equal(expected[i].bits, v.expected, `Directed constant: ${v.name}`)
    assert.equal(
      flagBits(expected[i].flags),
      v.flags,
      `Directed flags: ${v.name}`,
    )
  })
  const lines = cases.map(
    (v, i) =>
      `${hex(v.a)} ${hex(v.b)} ${v.subtract ? 1 : 0} ${hex(expected[i].bits)} ${hex(flagBits(expected[i].flags))}`,
  )
  const vectorPath = join(build, `${prefix}vectors.txt`),
    observationsPath = join(build, `${prefix}observations.txt`),
    wavePath = join(build, 'directed.vcd')
  writeFileSync(vectorPath, lines.join('\n') + '\n')
  const simulator = join(build, 'obj/fp16_sim'),
    run = spawnSync(
      simulator,
      [vectorPath, observationsPath, ...(waveform ? [wavePath] : [])],
      { encoding: 'utf8', maxBuffer: 1024 * 1024 },
    )
  if (run.error) throw run.error
  if (run.stdout) console.log(run.stdout.trim())
  if (run.stderr) console.error(run.stderr.trim())
  if (run.status !== 0 && run.status !== 1)
    throw new Error(`Simulator failed: status ${run.status}`)
  const observations = readFileSync(observationsPath, 'utf8').trim().split('\n')
  assert.equal(
    observations.length,
    cases.length,
    'Every transaction must produce one observation',
  )
  const failures = [],
    samples = []
  let signalComparisons = 0,
    nanResults = 0
  for (let i = 0; i < cases.length; i++) {
    const fields = observations[i].split(' ').map((x) => Number.parseInt(x, 16))
    assert.equal(fields.length, 20, 'Observation schema')
    const [
      a,
      b,
      subtract,
      bits,
      mask,
      effectiveBSign,
      exponentDifference,
      alignedA,
      alignedB,
      magnitude,
      normalizationShift,
      normalizedMagnitude,
      discardCount,
      retained,
      guard,
      round,
      sticky,
      increment,
      exponentRounded,
      special,
    ] = fields
    const v = cases[i],
      ref = calculateFP16(v.a, v.b, v.subtract ? 'subtract' : 'add')
    assert.equal(a, v.a)
    assert.equal(b, v.b)
    assert.equal(subtract, Number(v.subtract))
    if (ref.decoded.classification === 'nan') nanResults++
    const signals = {
      effectiveBSign,
      exponentDifference,
      alignedA: hex(alignedA),
      alignedB: hex(alignedB),
      magnitude: hex(magnitude),
      normalizationShift,
      normalizedMagnitude: hex(normalizedMagnitude),
      discardCount,
      retained,
      guard,
      round,
      sticky,
      increment,
      exponentRounded,
      special: !!special,
    }
    const reasons = []
    if (bits !== ref.bits)
      reasons.push('result bits (including NaN sign/payload)')
    if (mask !== flagBits(ref.flags)) reasons.push('exception flags')
    const grs = ref.trace.find((s) => s.kind === 'grs'),
      decision = ref.trace.find((s) => s.kind === 'round'),
      norm = ref.trace.find((s) => s.kind === 'normalize')
    if (grs && decision && norm) {
      signalComparisons++
      if (
        guard !== grs.guard ||
        round !== grs.round ||
        sticky !== grs.sticky ||
        increment !== Number(decision.increment) ||
        BigInt(retained) !== norm.retained
      )
        reasons.push('rounding signals')
      const exact =
        ref.exact.coefficient < 0n
          ? -ref.exact.coefficient
          : ref.exact.coefficient
      if (BigInt(magnitude) !== exact << BigInt(ref.exact.exponent + 24))
        reasons.push('exact arithmetic magnitude')
    }
    if (reasons.length)
      failures.push({
        index: i,
        ...v,
        expected: ref.bits,
        actual: bits,
        expectedFlags: ref.flags,
        actualFlags: flags(mask),
        signals,
        reasons,
      })
    if (i < directed.length)
      samples.push({
        name: directed[i].name,
        a,
        b,
        operation: v.subtract ? 'subtract' : 'add',
        softwareBits: ref.bits,
        rtlBits: bits,
        softwareFlags: ref.flags,
        rtlFlags: flags(mask),
        signals,
      })
  }
  const version = execFileSync(
    process.env.VERILATOR ?? 'verilator',
    ['--version'],
    { encoding: 'utf8' },
  ).trim()
  const sourceHashes = Object.fromEntries(
    [
      'src/fp16_addsub.sv',
      'Makefile',
      'sim/main.cpp',
      'scripts/verify.mjs',
      'vectors/directed.json',
      '../src/lib/numerics/fp16.ts',
      '../src/lib/numerics/fp16Arithmetic.ts',
      '../src/lib/numerics/roundDyadicFP16.ts',
      '../scripts/arithmetic-runtime.mjs',
    ].map((path) => [
      path,
      createHash('sha256')
        .update(readFileSync(join(root, path)))
        .digest('hex'),
    ]),
  )
  const summary = {
    schema: 1,
    tool: version,
    seed,
    cases: cases.length,
    counts,
    mismatches: failures.length,
    nanResults,
    signalComparisons,
    nanPolicy:
      'Strict raw equality including sign/payload; all four flags match exactly',
    sourceHashes,
  }
  writeFileSync(
    join(build, `${prefix}verification.json`),
    JSON.stringify(summary, null, 2) + '\n',
  )
  writeFileSync(
    join(build, `${prefix}failures.json`),
    JSON.stringify(failures, null, 2) + '\n',
  )
  writeFileSync(
    join(build, `${prefix}failures.txt`),
    failures.map((f) => lines[f.index]).join('\n') +
      (failures.length ? '\n' : ''),
  )
  if (failures.length) {
    const small = join(build, `${prefix}first-failures.txt`)
    writeFileSync(
      small,
      failures
        .slice(0, 8)
        .map((f) => lines[f.index])
        .join('\n') + '\n',
    )
    spawnSync(
      simulator,
      [
        small,
        join(build, `${prefix}failed-observations.txt`),
        join(build, `${prefix}failures.vcd`),
      ],
      { encoding: 'utf8' },
    )
    throw new Error(
      `${failures.length} mismatches saved under rtl/build; first: ${JSON.stringify(failures[0])}`,
    )
  }
  assert.equal(
    run.status,
    0,
    'Simulator must agree with independent harness comparison',
  )
  if (exportPath)
    writeFileSync(
      resolve(exportPath),
      JSON.stringify({ metadata: summary, samples }, null, 2) + '\n',
    )
  console.log(JSON.stringify(summary, null, 2))
})
