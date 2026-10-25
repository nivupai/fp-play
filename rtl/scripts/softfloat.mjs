// Optional direct RTL-to-SoftFloat comparison using the existing external C driver.
import { strict as assert } from 'node:assert'
import { execFileSync } from 'node:child_process'
import console from 'node:console'
import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import process from 'node:process'
import { fileURLToPath, URL } from 'node:url'
const driver = process.argv[2]
if (!driver)
  throw new Error(
    'Usage: node rtl/scripts/softfloat.mjs /path/to/softfloat-driver (run full RTL verification first)',
  )
const build = fileURLToPath(new URL('../build/', import.meta.url))
const vectors = readFileSync(resolve(build, 'vectors.txt'), 'utf8')
  .trim()
  .split('\n')
const observed = readFileSync(resolve(build, 'observations.txt'), 'utf8')
  .trim()
  .split('\n')
assert.equal(vectors.length, observed.length)
const rows = execFileSync(resolve(driver), [], {
  input:
    vectors
      .map((line) => {
        const [a, b, op] = line.split(' ')
        return `${a} ${b} ${op} 0`
      })
      .join('\n') + '\n',
  encoding: 'utf8',
  maxBuffer: 32 * 1024 * 1024,
})
  .trim()
  .split('\n')
assert.equal(rows.length, vectors.length)
let nanResults = 0,
  nanPolicyDifferences = 0
for (let i = 0; i < rows.length; i++) {
  const [, , , rtlText, maskText] = observed[i].split(' '),
    [expectedText, flagsText] = rows[i].split(' ')
  const rtl = Number.parseInt(rtlText, 16),
    expected = Number.parseInt(expectedText, 16),
    mask = Number.parseInt(maskText, 16),
    flags = Number(flagsText)
  const nan = (expected & 0x7fff) > 0x7c00
  if (nan) {
    nanResults++
    if (rtl !== expected) nanPolicyDifferences++
  }
  const bitsMatch = nan
    ? (rtl & 0x7fff) > 0x7c00 && (rtl & 0x200) !== 0
    : rtl === expected
  const mapped = (mask & 8 ? 16 : 0) | (mask & 7)
  assert.ok(
    bitsMatch && mapped === flags,
    `SoftFloat mismatch at ${i}: ${vectors[i]}, RTL ${rtlText}/${mapped}, SoftFloat ${expectedText}/${flagsText}`,
  )
}
const result = {
  cases: rows.length,
  mismatches: 0,
  nanResults,
  nanPolicyDifferences,
  nanComparison:
    'Quiet classification and all four flags; implementation-specific sign/payload selection compared separately against FP Play',
  rounding: 'nearest-even',
  tininess: 'after-rounding',
}
writeFileSync(
  resolve(build, 'softfloat-verification.json'),
  JSON.stringify(result, null, 2) + '\n',
)
console.log(JSON.stringify(result, null, 2))
