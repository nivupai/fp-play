// Run both complete verification and directed waveform flows before exporting.
import { strict as assert } from 'node:assert'
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync, mkdirSync, copyFileSync } from 'node:fs'
import { fileURLToPath, pathToFileURL, URL } from 'node:url'
import { resolve } from 'node:path'
import ts from 'typescript'
import console from 'node:console'
const root = fileURLToPath(new URL('../../', import.meta.url)),
  out = resolve(root, 'rtl/build/datapath')
mkdirSync(out, { recursive: true })
mkdirSync(resolve(root, 'public/artifacts'), { recursive: true })
const hash = (p) =>
  createHash('sha256')
    .update(readFileSync(resolve(root, p)))
    .digest('hex')
const parser = 'src/lib/datapath/vcd.ts'
writeFileSync(
  resolve(out, 'vcd.mjs'),
  ts.transpileModule(readFileSync(resolve(root, parser), 'utf8'), {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.ESNext,
    },
  }).outputText,
)
const { curateVCD } = await import(pathToFileURL(resolve(out, 'vcd.mjs')).href)
const groups = []
for (const family of ['add', 'product']) {
  const artifactPath = `rtl/vectors/${family === 'add' ? 'simulation' : 'mul-fma-simulation'}.json`,
    artifact = JSON.parse(readFileSync(resolve(root, artifactPath)))
  const build = family === 'add' ? 'rtl/build' : 'rtl/build/mul-fma'
  const verification = JSON.parse(
      readFileSync(resolve(root, build, 'verification.json')),
    ),
    directed = JSON.parse(
      readFileSync(resolve(root, build, 'directed-verification.json')),
    )
  for (const certificate of [verification, directed, artifact.metadata]) {
    assert.equal(certificate.mismatches, 0)
    for (const [p, h] of Object.entries(certificate.sourceHashes)) {
      assert.equal(
        hash(family === 'add' ? resolve(root, 'rtl', p) : p),
        h,
        `Stale simulation source ${p}`,
      )
    }
  }
  assert.equal(verification.cases, family === 'add' ? 462172 : 301544)
  assert.equal(directed.cases, artifact.samples.length)
  assert.equal(directed.tool, verification.tool)
  const selected =
    family === 'add'
      ? [
          'a',
          'b',
          'subtract',
          'result',
          'guard_bit',
          'round_bit',
          'sticky_bit',
          'increment',
          'retained_significand',
          'arithmetic_magnitude',
          'invalid',
          'overflow',
          'underflow',
          'inexact',
        ]
      : [
          'a',
          'b',
          'c',
          'multiply_result',
          'multiply_flags',
          'fused_result',
          'fused_flags',
          'nonfused_result',
          'nonfused_flags',
          'addition_flags',
          'mul_guard_bit',
          'mul_round_bit',
          'mul_sticky_bit',
          'mul_increment',
          'fma_guard_bit',
          'fma_round_bit',
          'fma_sticky_bit',
          'fma_increment',
          'fma_exact_product',
          'fma_arithmetic_magnitude',
        ]
  const rawPath = `${build}/directed.vcd`,
    data = curateVCD(readFileSync(resolve(root, rawPath), 'utf8'), selected)
  const sampleFrame = (index) => {
    const frame = data.frames.filter((f) => f.time <= index * 2 + 1).at(-1)
    assert.ok(frame, 'Missing VCD state at settled transaction')
    return frame.values
  }
  for (const [i, s] of artifact.samples.entries()) {
    const values = sampleFrame(i),
      number = (name) => {
        assert.match(values[name], /^[01]+$/)
        return Number(BigInt('0b' + values[name]))
      }
    assert.equal(number('a'), s.a)
    assert.equal(number('b'), s.b)
    if (family === 'add') {
      assert.equal(number('subtract'), s.operation === 'subtract' ? 1 : 0)
      assert.equal(number('result'), s.rtlBits)
      for (const key of ['invalid', 'overflow', 'underflow', 'inexact'])
        assert.equal(!!number(key), s.rtlFlags[key])
      for (const [signal, key] of [
        ['guard_bit', 'guard'],
        ['round_bit', 'round'],
        ['sticky_bit', 'sticky'],
        ['increment', 'increment'],
        ['retained_significand', 'retained'],
      ])
        assert.equal(number(signal), s.signals[key])
      assert.equal(
        BigInt('0b' + values.arithmetic_magnitude),
        BigInt('0x' + s.signals.magnitude),
      )
    } else {
      assert.equal(number('c'), s.c)
      for (const [signal, key] of [
        ['multiply_result', 'multiply'],
        ['multiply_flags', 'multiplyFlags'],
        ['fused_result', 'fused'],
        ['fused_flags', 'fusedFlags'],
        ['nonfused_result', 'nonfused'],
        ['nonfused_flags', 'nonfusedFlags'],
        ['addition_flags', 'additionFlags'],
      ])
        assert.equal(number(signal), s[key])
      for (const prefix of ['mul', 'fma'])
        for (const [signal, key] of [
          ['guard_bit', 'guard'],
          ['round_bit', 'round'],
          ['sticky_bit', 'sticky'],
          ['increment', 'increment'],
        ])
          assert.equal(
            number(`${prefix}_${signal}`),
            s[prefix === 'mul' ? 'multiplySignals' : 'fmaSignals'][key],
          )
      assert.equal(number('fma_exact_product'), s.fmaSignals.exactProduct)
      assert.equal(
        BigInt('0b' + values.fma_arithmetic_magnitude),
        BigInt(s.fmaSignals.arithmeticMagnitude),
      )
    }
  }
  const publicPath = `public/artifacts/datapath-${family}.vcd`
  copyFileSync(resolve(root, rawPath), resolve(root, publicPath))
  groups.push({
    family,
    artifact: artifactPath,
    tool: directed.tool,
    cases: verification.cases,
    samples: artifact.samples.length,
    sampleTimes: artifact.samples.map((_, i) => i * 2 + 1),
    download: `artifacts/datapath-${family}.vcd`,
    sourceHashes: {
      [publicPath]: hash(publicPath),
      [artifactPath]: hash(artifactPath),
      [parser]: hash(parser),
      'rtl/scripts/export-datapath-waveforms.mjs': hash(
        'rtl/scripts/export-datapath-waveforms.mjs',
      ),
    },
    data,
  })
}
writeFileSync(
  resolve(root, 'rtl/vectors/datapath-waveforms.json'),
  JSON.stringify(
    {
      schemaVersion: 1,
      interpretation:
        'Settled reset/transaction observations from actual Verilator VCD, not propagation delay or physical clocks',
      groups,
    },
    null,
    2,
  ) + '\n',
)
console.log(
  `Exported ${groups.length} checked VCD groups, ${groups.reduce((n, g) => n + g.samples, 0)} correlated examples`,
)
