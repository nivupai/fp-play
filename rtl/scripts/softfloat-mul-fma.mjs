// Compare genuine RTL outputs directly to the separately built public SoftFloat driver.
import { strict as assert } from 'node:assert'
import { execFileSync } from 'node:child_process'
import console from 'node:console'
import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import process from 'node:process'
import { fileURLToPath, URL } from 'node:url'
const driver = process.argv[2]
assert.ok(
  driver,
  'Usage: node rtl/scripts/softfloat-mul-fma.mjs /path/to/softfloat-driver',
)
const build = fileURLToPath(new URL('../build/mul-fma/', import.meta.url))
const observed = readFileSync(resolve(build, 'observations.txt'), 'utf8')
  .trim()
  .split('\n')
  .map((l) => l.split(' '))
const call = (input) =>
  execFileSync(resolve(driver), [], {
    input: input.join('\n') + '\n',
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  })
    .trim()
    .split('\n')
    .map((l) => {
      const [b, f] = l.split(' ')
      return { bits: parseInt(b, 16), flags: Number(f) }
    })
const product = call(observed.map(([a, b]) => `${a} ${b} 2 0`))
const fused = call(observed.map(([a, b, c]) => `${a} ${b} 3 ${c}`))
const addition = call(
  observed.map((r, i) => `${product[i].bits.toString(16)} ${r[2]} 0 0`),
)
const summaries = []
for (const [name, results, bitColumn, flagColumn] of [
  ['multiply', product, 3, 4],
  ['fma', fused, 5, 6],
  ['nonfused', addition, 7, 8],
]) {
  assert.equal(results.length, observed.length)
  let nanResults = 0,
    nanPolicyDifferences = 0
  for (let i = 0; i < results.length; i++) {
    const r = results[i],
      bits = parseInt(observed[i][bitColumn], 16),
      mask = parseInt(observed[i][flagColumn], 16),
      flags = (mask & 8 ? 16 : 0) | (mask & 7)
    const expectedFlags = r.flags | (name === 'nonfused' ? product[i].flags : 0)
    if ((r.bits & 0x7fff) > 0x7c00) {
      nanResults++
      if (bits !== r.bits) nanPolicyDifferences++
      assert.ok(
        (bits & 0x7fff) > 0x7c00 && (bits & 512) !== 0,
        `${name} quiet NaN ${i}`,
      )
    } else assert.equal(bits, r.bits, `${name} bits ${i}`)
    assert.equal(
      flags,
      expectedFlags,
      `${name} flags ${i}: ${observed[i].slice(0, 3)}`,
    )
  }
  summaries.push({
    operation: name,
    cases: results.length,
    mismatches: 0,
    nanResults,
    nanPolicyDifferences,
  })
}
const report = {
  rounding: 'nearest-even',
  tininess: 'after-rounding',
  nanPolicy:
    'Quiet NaN classification; sign/payload differences counted, no flag exemptions',
  summaries,
}
writeFileSync(
  resolve(build, 'softfloat-verification.json'),
  JSON.stringify(report, null, 2) + '\n',
)
console.log(JSON.stringify(report, null, 2))
