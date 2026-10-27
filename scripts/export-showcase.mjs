// Compact presentation snapshot of existing measurements; never runs synthesis.
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'
import { strict as assert } from 'node:assert'
import console from 'node:console'
import { URL } from 'node:url'
const path = 'rtl/vectors/hardware-analysis.json'
const root = new URL('../', import.meta.url)
const raw = readFileSync(new URL(path, root))
const artifact = JSON.parse(raw)
assert.equal(artifact.schemaVersion, 1)
assert.equal(artifact.records.length, 7)
const records = artifact.records.map((r) => {
  assert.equal(r.verification.mismatches, 0)
  for (const n of [r.mapped.area, r.mapped.cells, r.mapped.logicDepth])
    assert.ok(Number.isFinite(n) && n >= 0)
  if (r.timing)
    assert.ok(Number.isFinite(r.timing.delayNs) && r.timing.delayNs > 0)
  return {
    id: r.id,
    unit: r.unit,
    variant: r.variant,
    area: r.mapped.area,
    cells: r.mapped.cells,
    depth: r.mapped.logicDepth,
    delay: r.timing?.delayNs ?? null,
    startpoint: r.timing?.startpoint ?? null,
    endpoint: r.timing?.endpoint ?? null,
    pathCells: r.timing?.cells.length ?? null,
    stageBreakdownUnavailable:
      r.timing?.stageBreakdownUnavailable ?? r.timingUnavailable,
    verification: r.verification,
  }
})
const snapshot = {
  schemaVersion: 1,
  source: {
    path,
    sha256: createHash('sha256').update(raw).digest('hex'),
    gitCommit: artifact.gitCommit,
    gitDirty: artifact.gitDirty,
    timestamp: artifact.timestamp,
    sourceHashes: artifact.sourceHashes,
    tools: artifact.tools,
    config: artifact.config,
  },
  records,
}
writeFileSync(
  new URL('src/lib/showcase/hardware-summary.json', root),
  JSON.stringify(snapshot, null, 2) + '\n',
)
console.log(
  `Exported ${records.length} measured records with original source provenance; no measurements generated.`,
)
