// Replay the actual measured boundaries, catching miswired results/flags before mapping.
import { strict as assert } from 'node:assert'
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs'
import { fileURLToPath, URL } from 'node:url'
import { resolve } from 'node:path'
import process from 'node:process'
const root = fileURLToPath(new URL('../../', import.meta.url)),
  build = resolve(root, 'rtl/build/analysis/unit-verification')
const hash = (p) =>
  createHash('sha256')
    .update(readFileSync(resolve(root, p)))
    .digest('hex')
const verilatorVersion = execFileSync(
  process.env.VERILATOR ?? 'verilator',
  ['--version'],
  { encoding: 'utf8' },
).trim()
const records = []
for (const variant of ['baseline', 'tree-leading-bit'])
  for (const unit of ['add', 'mul', 'fma', 'nonfused']) {
    if (variant !== 'baseline' && unit === 'add') continue
    const id = `${variant}-${unit}`,
      dir = resolve(build, id),
      top = `analysis_${unit}`,
      cpp = resolve(dir, 'probe.cpp')
    mkdirSync(dir, { recursive: true })
    const sources = readdirSync(resolve(root, 'rtl/src'))
      .filter(
        (p) =>
          p.endsWith('.sv') &&
          (variant === 'baseline' || p !== 'fp16_round_grid48.sv'),
      )
      .map((p) => 'rtl/src/' + p)
    if (variant !== 'baseline')
      sources.push('rtl/variants/tree-leading-bit/fp16_round_grid48.sv')
    sources.push(`rtl/analysis/units/${top}.sv`)
    const args = [
      '--Wall',
      '--top-module',
      top,
      ...sources.map((p) => resolve(root, p)),
    ]
    execFileSync(
      process.env.VERILATOR ?? 'verilator',
      ['--lint-only', ...args],
      { stdio: 'inherit' },
    )
    const index =
      unit === 'add' ? 3 : unit === 'mul' ? 3 : unit === 'fma' ? 5 : 7
    writeFileSync(
      cpp,
      `#include "V${top}.h"
#include "verilated.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
int main(int argc, char **argv) {
 if(argc!=2) return 2;
 std::ifstream input(argv[1]);
 if(!input) return 2;
 VerilatedContext context; V${top} dut(&context);
 std::string line; unsigned count=0, failures=0;
 while(std::getline(input,line)) {
  std::istringstream row(line); std::vector<unsigned> values; unsigned value;
  while(row >> std::hex >> value) values.push_back(value);
  if(values.size()!=${unit === 'add' ? 5 : 10}) return 2;
  dut.a=values[0]; dut.b=values[1];
  ${unit === 'add' ? 'dut.subtract=values[2];' : unit === 'mul' ? '' : 'dut.c=values[2];'}
  dut.eval();
  if(dut.result!=values[${index}] || dut.flags!=values[${index + 1}]) {
   if(failures++<8) std::cerr << "${id} mismatch " << count << ": " << line << "\\n";
  }
  ++count;
 }
 dut.final();
 std::cout << "${id}: " << count << " cases, " << failures << " mismatches\\n";
 return count && !failures ? 0 : 1;
}
`,
    )
    execFileSync(
      process.env.VERILATOR ?? 'verilator',
      [
        '--cc',
        '--exe',
        '--build',
        ...args,
        '-Mdir',
        dir,
        '-o',
        'probe',
        '-CFLAGS',
        '-std=c++17',
        cpp,
      ],
      { stdio: 'inherit' },
    )
    const family = unit === 'add' ? '' : 'mul-fma/'
    let count = 0
    for (const prefix of ['', 'directed-']) {
      const path = `rtl/build/${family}${prefix}vectors.txt`
      execFileSync(resolve(dir, 'probe'), [resolve(root, path)], {
        stdio: 'inherit',
      })
      if (!prefix)
        count = readFileSync(resolve(root, path), 'utf8')
          .trim()
          .split('\n').length
    }
    assert.equal(count, unit === 'add' ? 462172 : 301544)
    records.push({
      id,
      cases: count,
      verilator: verilatorVersion,
      vectorsSha256: hash(`rtl/build/${family}vectors.txt`),
      mismatches: 0,
      proof: false,
      method:
        'Actual functional wrapper outputs, raw result and all flags against the full original numerical corpus',
      sourceHashes: Object.fromEntries(
        [...sources, 'rtl/analysis/verify-units.mjs'].map((p) => [p, hash(p)]),
      ),
    })
  }
writeFileSync(
  resolve(build, 'verification.json'),
  JSON.stringify({ records }, null, 2) + '\n',
)
