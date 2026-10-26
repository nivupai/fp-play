import { readFileSync } from 'node:fs'
import { fileURLToPath, URL } from 'node:url'
import process from 'node:process'
import console from 'node:console'
const artifact = JSON.parse(
  readFileSync(
    process.argv[2] ??
      fileURLToPath(new URL('../build/analysis/summary.json', import.meta.url)),
    'utf8',
  ),
)
console.log(
  `Source ${artifact.gitCommit}; dirty=${artifact.gitDirty}\n${artifact.tools.yosys}\n`,
)
console.log(
  '| Unit / variant | Generic cells | Mapped cells | Area (µm²) | Delay (ns) | Mapped depth | I/O |',
)
console.log('| --- | ---: | ---: | ---: | ---: | ---: | --- |')
for (const r of artifact.records)
  console.log(
    `| ${r.unit} / ${r.variant} | ${r.generic.cells} | ${r.mapped.cells} | ${r.mapped.area} | ${r.timing?.delayNs ?? 'Unavailable'} | ${r.mapped.logicDepth} | ${r.io.inputs}/${r.io.outputs} |`,
  )
console.log(
  '\nCell area and ideal-interconnect timing estimates; no physical area, power, or post-layout claims.',
)
