import { strict as assert } from 'node:assert'
import { createHash } from 'node:crypto'
import { execFileSync, spawnSync } from 'node:child_process'
import {
  readFileSync,
  writeFileSync,
  mkdirSync,
  existsSync,
  readdirSync,
} from 'node:fs'
import { fileURLToPath, pathToFileURL, URL } from 'node:url'
import { resolve, dirname, relative } from 'node:path'
import process from 'node:process'
import console from 'node:console'
import ts from 'typescript'
const root = fileURLToPath(new URL('../../', import.meta.url)),
  build = resolve(root, 'rtl/build/analysis')
const config = JSON.parse(
  readFileSync(resolve(root, 'rtl/analysis/config.json')),
)
const hash = (p) => createHash('sha256').update(readFileSync(p)).digest('hex')
const json = (p) => JSON.parse(readFileSync(p, 'utf8'))
mkdirSync(build, { recursive: true })
const yosys = process.env.YOSYS ?? 'yosys',
  version = execFileSync(yosys, ['-V'], { encoding: 'utf8' }).trim()
assert.ok(
  version.startsWith(config.toolchain.yosys),
  `Use pinned ${config.toolchain.yosys}; got ${version}`,
)
const yosysPath = yosys.includes('/')
  ? yosys
  : execFileSync('which', [yosys], { encoding: 'utf8' }).trim()
const abc = process.env.ABC ?? resolve(dirname(yosysPath), 'yosys-abc')
const abcVersion = execFileSync(abc, ['-c', 'version'], {
  encoding: 'utf8',
}).trim()
const sta = process.env.STA ?? 'sta'
const staProbe = spawnSync(sta, ['-version'], { encoding: 'utf8' })
const staVersion = staProbe.status === 0 ? staProbe.stdout.trim() : null
const library = resolve(build, 'library/NangateOpenCellLibrary_typical.lib')
mkdirSync(dirname(library), { recursive: true })
if (!existsSync(library))
  execFileSync(
    'curl',
    [
      '--fail',
      '--location',
      '--retry',
      '3',
      '--output',
      library,
      config.library.url,
    ],
    { stdio: 'inherit' },
  )
assert.equal(hash(library), config.library.sha256, 'Pinned Liberty checksum')
const sourcePaths = [
  'rtl/analysis.mk',
  ...readdirSync(resolve(root, 'rtl/src')).map((n) => 'rtl/src/' + n),
  'rtl/variants/tree-leading-bit/fp16_round_grid48.sv',
  ...readdirSync(resolve(root, 'rtl/analysis/units')).map(
    (n) => 'rtl/analysis/units/' + n,
  ),
  'rtl/analysis/verify-units.mjs',
  'rtl/analysis/config.json',
  'rtl/analysis/timing.sdc',
  'rtl/analysis/equiv-rounder.ys',
  'rtl/analysis/run.mjs',
  'rtl/analysis/build-opensta.sh',
  'rtl/analysis/summarize.mjs',
  'rtl/analysis/verify-variant.mjs',
  'src/lib/hardware/netlist.ts',
]
const sourceHashes = Object.fromEntries(
  sourcePaths.map((p) => [p, hash(resolve(root, p))]),
)
const baseline = json(resolve(root, 'rtl/build/mul-fma/verification.json')),
  adder = json(resolve(root, 'rtl/build/verification.json')),
  variantVerification = json(resolve(build, 'verification/verification.json'))
for (const v of [baseline, variantVerification])
  for (const [p, h] of Object.entries(v.sourceHashes)) {
    assert.equal(hash(resolve(root, p)), h, `Stale verification ${p}`)
    sourceHashes[p] = h
  }
for (const [p, h] of Object.entries(adder.sourceHashes)) {
  assert.equal(
    hash(resolve(root, 'rtl', p)),
    h,
    `Stale adder verification ${p}`,
  )
  sourceHashes[relative(root, resolve(root, 'rtl', p))] = h
}
for (const v of [baseline, adder, variantVerification])
  assert.equal(v.mismatches, 0)
assert.equal(
  hash(resolve(root, 'rtl/build/mul-fma/vectors.txt')),
  variantVerification.vectorsSha256,
)
assert.equal(
  hash(resolve(root, 'rtl/build/mul-fma/observations.txt')),
  variantVerification.observationsSha256,
)
writeFileSync(
  resolve(build, 'netlist.mjs'),
  ts.transpileModule(
    readFileSync(resolve(root, 'src/lib/hardware/netlist.ts'), 'utf8'),
    {
      compilerOptions: {
        target: ts.ScriptTarget.ES2022,
        module: ts.ModuleKind.ESNext,
      },
    },
  ).outputText,
)
const { analyzeLogic } = await import(
  pathToFileURL(resolve(build, 'netlist.mjs')).href
)
const command = (executable, args, log) => {
  const result = spawnSync(executable, args, {
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
    timeout: 180000,
    cwd: root,
  })
  writeFileSync(log, (result.stdout ?? '') + (result.stderr ?? ''))
  if (result.error) throw result.error
  assert.equal(result.status, 0, `Failed command; see ${log}`)
}
const quote = (s) => JSON.stringify(s)
const gitCommit = execFileSync('git', ['rev-parse', 'HEAD'], {
  encoding: 'utf8',
}).trim()
const gitDirty =
  execFileSync('git', ['status', '--porcelain'], {
    encoding: 'utf8',
  }).trim() !== ''
const formalLog = resolve(build, 'rounder-equivalence.log')
command(
  yosys,
  ['-Q', '-T', '-s', resolve(root, 'rtl/analysis/equiv-rounder.ys')],
  formalLog,
)
assert.ok(
  readFileSync(formalLog, 'utf8').includes('Equivalence successfully proven!'),
  'Rounder equivalence must be proven',
)
const rounderEquivalence = {
  proven: true,
  scope:
    'All rounder outputs for every 81-bit magnitude and sign; not a whole-unit proof',
  report: relative(root, formalLog),
  sha256: hash(formalLog),
}
const sdc = readFileSync(resolve(root, 'rtl/analysis/timing.sdc'), 'utf8')
for (const line of [
  `create_clock -name virtual_io -period ${config.timing.virtualClockPeriodNs}`,
  `set_input_delay -clock virtual_io -max ${config.timing.inputDelayNs} [all_inputs]`,
  `set_output_delay -clock virtual_io -max ${config.timing.outputDelayNs} [all_outputs]`,
  `set_input_transition ${config.timing.inputTransitionNs} [all_inputs]`,
  `set_load ${config.timing.outputLoadFf} [all_outputs]`,
])
  assert.ok(
    sdc.includes(line),
    'SDC must agree with recorded timing assumptions',
  )
const wrapperEvidence = json(
  resolve(build, 'unit-verification/verification.json'),
).records
for (const r of wrapperEvidence) {
  assert.equal(r.mismatches, 0)
  assert.equal(
    hash(
      resolve(
        root,
        r.id.endsWith('-add')
          ? 'rtl/build/vectors.txt'
          : 'rtl/build/mul-fma/vectors.txt',
      ),
    ),
    r.vectorsSha256,
    'Measured wrapper corpus must match',
  )
  for (const [p, h] of Object.entries(r.sourceHashes))
    assert.equal(hash(resolve(root, p)), h, `Stale measured wrapper ${p}`)
}
assert.equal(wrapperEvidence.length, 7)
const records = []
for (const variant of config.variants)
  for (const unit of config.units) {
    if (variant !== 'baseline' && unit === 'add') continue
    const name = `${variant}-${unit}`,
      directory = resolve(build, name),
      top = `analysis_${unit}`
    mkdirSync(directory, { recursive: true })
    const sources = readdirSync(resolve(root, 'rtl/src'))
      .filter((n) => n.endsWith('.sv'))
      .map((n) => 'rtl/src/' + n)
      .filter(
        (p) => variant === 'baseline' || !p.endsWith('/fp16_round_grid48.sv'),
      )
    if (variant !== 'baseline')
      sources.push('rtl/variants/tree-leading-bit/fp16_round_grid48.sv')
    sources.push(
      ...readdirSync(resolve(root, 'rtl/analysis/units')).map(
        (n) => 'rtl/analysis/units/' + n,
      ),
    )
    const constraints = resolve(directory, 'abc.constr')
    writeFileSync(
      constraints,
      `set_driving_cell ${config.mapping.driver}\nset_load ${config.mapping.outputLoadFf}\n`,
    )
    const script =
      [
        `read_verilog -sv ${sources.join(' ')}`,
        `hierarchy -check -top ${top}`,
        'proc',
        'check -assert',
        `synth -top ${top} -flatten -noabc`,
        'check -assert',
        'select -assert-none t:$_DLATCH_* t:$_DFF_* t:$dlatch t:$dff',
        `tee -o ${quote(resolve(directory, 'generic-stat.json'))} stat -json`,
        `write_json ${quote(resolve(directory, 'generic.json'))}`,
        `read_liberty -lib ${quote(library)}`,
        `abc -exe ${quote(abc)} -liberty ${quote(library)} -constr ${quote(constraints)} -D ${config.mapping.abcDelayTargetPs}`,
        'clean',
        'check -assert',
        `tee -o ${quote(resolve(directory, 'mapped-stat.json'))} stat -json -liberty ${quote(library)}`,
        `write_json ${quote(resolve(directory, 'mapped.json'))}`,
        `write_verilog -noattr -noexpr -nodec ${quote(resolve(directory, 'mapped.v'))}`,
      ].join('\n') + '\n'
    const scriptPath = resolve(directory, 'synthesis.ys'),
      log = resolve(directory, 'synthesis.log')
    writeFileSync(scriptPath, script)
    command(yosys, ['-Q', '-T', '-s', scriptPath], log)
    const generic = json(resolve(directory, 'generic.json')).modules[top],
      mapped = json(resolve(directory, 'mapped.json')).modules[top]
    const g = analyzeLogic(generic),
      m = analyzeLogic(mapped)
    const stat = json(resolve(directory, 'mapped-stat.json')).modules[
      '\\' + top
    ]
    assert.ok(
      Number.isFinite(stat.area) && stat.area > 0,
      'Mapped library area must be measured',
    )
    let timing = null,
      timingUnavailable =
        staVersion === null ? 'OpenSTA executable unavailable' : null
    const staScript = resolve(directory, 'timing.tcl')
    writeFileSync(
      staScript,
      [
        `read_liberty ${quote(library)}`,
        `read_verilog ${quote(resolve(directory, 'mapped.v'))}`,
        `link_design ${top}`,
        `read_sdc ${quote(resolve(root, 'rtl/analysis/timing.sdc'))}`,
        'check_setup -verbose',
        'report_checks -path_delay max -format full -digits 6 -group_path_count 1',
        'exit',
      ].join('\n') + '\n',
    )
    if (staVersion) {
      const staLog = resolve(directory, 'timing.log')
      command(sta, ['-exit', staScript], staLog)
      const report = readFileSync(staLog, 'utf8')
      assert.ok(
        !/^Error(?:\s+\d+)?:/m.test(report),
        'OpenSTA errors must not be treated as timing results',
      )
      const arrival = report.match(/([\d.]+)\s+data arrival time/),
        start = report.match(/Startpoint: (.+)/),
        end = report.match(/Endpoint: (.+)/)
      assert.ok(arrival && start && end, 'Constrained timing path missing')
      timing = {
        delayNs: Number(arrival[1]),
        startpoint: start[1],
        endpoint: end[1],
        cells: [
          ...report.matchAll(
            /^\s*([\d.]+)\s+([\d.]+)\s+[\^v]\s+(\S+)\s+\(([^)]+)\)/gm,
          ),
        ]
          .filter((match) => !['in', 'out'].includes(match[4]))
          .map((match) => ({
            pin: match[3],
            type: match[4],
            arcDelayNs: Number(match[1]),
            arrivalNs: Number(match[2]),
          })),
        report: relative(root, staLog),
        stageBreakdown: null,
        stageBreakdownUnavailable:
          'ABC optimizes across RTL stage boundaries; no reliable stage attribution is retained',
      }
    }
    const wrapperVerification = wrapperEvidence.find((r) => r.id === name)
    assert.ok(
      wrapperVerification,
      'Measured wrapper must have full numerical verification',
    )
    const verification =
      variant === 'baseline'
        ? unit === 'add'
          ? adder
          : baseline
        : variantVerification
    const record = {
      id: name,
      unit,
      variant,
      top,
      generic: { cells: g.combinationalCells, logicDepth: g.logicDepth },
      mapped: {
        cells: m.combinationalCells,
        area: stat.area,
        logicDepth: m.logicDepth,
        combinationalCells: m.combinationalCells,
        sequentialCells: m.sequentialCells,
      },
      io: { inputs: m.inputBits, outputs: m.outputBits },
      timing,
      timingUnavailable,
      depthPath: m.path,
      verification: {
        cases: verification.cases,
        mismatches: verification.mismatches,
        seed: verification.seed,
        proof: false,
        rounderEquivalence: variant === 'baseline' ? null : rounderEquivalence,
        boundaryVerification: wrapperVerification,
        method:
          variant === 'baseline'
            ? 'Exact TypeScript differential corpus'
            : 'Differential results/flags/signals against verified baseline',
      },
      reports: {
        synthesis: relative(root, log),
        netlist: relative(root, resolve(directory, 'mapped.v')),
        genericStat: relative(root, resolve(directory, 'generic-stat.json')),
        mappedStat: relative(root, resolve(directory, 'mapped-stat.json')),
        script: relative(root, scriptPath),
      },
    }
    record.reportHashes = Object.fromEntries(
      Object.entries(record.reports).map(([key, p]) => [
        key,
        hash(resolve(root, p)),
      ]),
    )
    if (timing) record.reportHashes.timing = hash(resolve(root, timing.report))
    records.push(record)
    console.log(
      `${name}: generic ${g.combinationalCells}, mapped ${m.combinationalCells}, area ${stat.area}, depth ${m.logicDepth}, delay ${timing?.delayNs ?? 'unavailable'}`,
    )
    writeFileSync(
      resolve(directory, 'manifest.json'),
      JSON.stringify(
        {
          schemaVersion: 1,
          gitCommit,
          gitDirty,
          timestamp: new Date().toISOString(),
          sourceHashes,
          tools: { yosys: version, abc: abcVersion, opensta: staVersion },
          config,
          commands: {
            synthesis: [yosys, '-Q', '-T', '-s', relative(root, scriptPath)],
            timing: staVersion
              ? [sta, '-exit', relative(root, staScript)]
              : null,
          },
          record,
        },
        null,
        2,
      ) + '\n',
    )
  }
const artifact = {
  schemaVersion: 1,
  gitCommit,
  gitDirty,
  timestamp: new Date().toISOString(),
  sourceHashes,
  tools: { yosys: version, abc: abcVersion, opensta: staVersion },
  config,
  records,
  rounderEquivalence,
  limitations: [
    'Synthesis-derived cell area only; no physical layout, power or silicon measurements',
    'Ideal interconnect; no placement, routing, extracted parasitics or clock implementation',
    'Verification corpora are sampled/exhaustive families, not full operand-space proofs',
    'Timing stage attribution unavailable after cross-stage ABC optimization',
  ],
}
writeFileSync(
  resolve(build, 'summary.json'),
  JSON.stringify(artifact, null, 2) + '\n',
)
const exportIndex = process.argv.indexOf('--export')
if (exportIndex >= 0) {
  assert.ok(process.argv[exportIndex + 1], '--export requires a path')
  writeFileSync(
    resolve(root, process.argv[exportIndex + 1]),
    JSON.stringify(artifact, null, 2) + '\n',
  )
}
