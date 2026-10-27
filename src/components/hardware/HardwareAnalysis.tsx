import { useState } from 'react'
import type { ReactNode } from 'react'
import recorded from '../../../rtl/vectors/hardware-analysis.json'
import type { HardwareArtifact } from '../../lib/hardware/types'
export type { HardwareArtifact } from '../../lib/hardware/types'
const formatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: 6 })
const number = (value: number) => formatter.format(value)
function PathDetails({
  label,
  children,
}: {
  label: string
  children: ReactNode
}) {
  const [open, setOpen] = useState(false)
  return (
    <details onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>{label}</summary>
      {open ? children : null}
    </details>
  )
}
export function HardwareAnalysis({
  artifact = recorded,
}: {
  artifact?: HardwareArtifact
}) {
  const [selection, setSelection] = useState(artifact.records[0]?.id ?? '')
  const selected = artifact.records.find((r) => r.id === selection)
  const timed = artifact.records.filter((r) => r.timing !== null)
  const maxArea = Math.max(1, ...artifact.records.map((r) => r.mapped.area)),
    maxDelay = Math.max(1, ...timed.map((r) => r.timing!.delayNs)),
    maxDepth = Math.max(1, ...artifact.records.map((r) => r.mapped.logicDepth))
  return (
    <section
      id="hardware-results"
      aria-labelledby="hardware-heading"
      className="panel hardware-analysis"
    >
      <div className="section-heading">
        <p className="eyebrow">Synthesis laboratory · recorded experiments</p>
        <h2 id="hardware-heading">Hardware Analysis</h2>
        <p>
          Compare FP16 arithmetic under one library, toolchain, and constraint
          set. These are actual synthesis artifacts; your browser does not run
          synthesis.
        </p>
      </div>
      <p className="muted">
        Mapped cell area is not chip area. Timing uses ideal interconnect,
        without placement or extracted parasitics. No power or post-layout
        performance is measured.
      </p>
      {artifact.records.length === 0 ? (
        <p>
          No synthesis measurements are available. Run the documented analysis
          flow to generate verified artifacts.
        </p>
      ) : (
        <>
          <div
            className="hardware-table"
            tabIndex={0}
            role="region"
            aria-label="Scrollable hardware comparison"
          >
            <table>
              <caption>
                Controlled Nangate45 comparison. Every variant uses the same
                functional-only interfaces and mapping settings.
              </caption>
              <thead>
                <tr>
                  {[
                    'Unit / variant',
                    'Generic cells',
                    'Mapped cells',
                    'Cell area (µm²)',
                    'Delay (ns)',
                    'Mapped depth',
                    'I/O bits',
                    'Sequential cells',
                    'Verification',
                  ].map((h) => (
                    <th key={h} scope="col">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {artifact.records.map((r) => (
                  <tr key={r.id}>
                    <th scope="row">
                      <button
                        type="button"
                        onClick={() => setSelection(r.id)}
                        aria-pressed={selection === r.id}
                      >
                        {r.unit} / {r.variant}
                      </button>
                    </th>
                    <td>{number(r.generic.cells)}</td>
                    <td>{number(r.mapped.cells)}</td>
                    <td>{number(r.mapped.area)}</td>
                    <td>
                      {r.timing ? number(r.timing.delayNs) : 'Unavailable'}
                    </td>
                    <td>{r.mapped.logicDepth}</td>
                    <td>
                      {r.io.inputs} / {r.io.outputs}
                    </td>
                    <td>{r.mapped.sequentialCells}</td>
                    <td>
                      {r.verification.mismatches === 0 ? 'Passed' : 'FAILED'} ·{' '}
                      {number(r.verification.cases)} cases; sampled families, no
                      proof
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <label htmlFor="hardware-experiment">
            Inspect hardware experiment
          </label>
          <select
            id="hardware-experiment"
            value={selection}
            onChange={(e) => setSelection(e.target.value)}
          >
            {artifact.records.map((r) => (
              <option key={r.id} value={r.id}>
                {r.unit} / {r.variant}
              </option>
            ))}
          </select>
          <div className="hardware-charts">
            <figure>
              <figcaption>
                Cell area versus estimated constrained path delay
              </figcaption>
              {timed.length ? (
                <svg
                  viewBox="0 0 580 300"
                  role="img"
                  aria-label="Area versus delay; exact values and experiment selection are available in the table"
                >
                  <path d="M65 20V255H560" fill="none" stroke="currentColor" />
                  <text x="230" y="290">
                    Mapped cell area (µm²)
                  </text>
                  <text x="12" y="20">
                    Delay (ns)
                  </text>
                  <text x="65" y="275">
                    0
                  </text>
                  <text x="470" y="275">
                    {number(maxArea)}
                  </text>
                  <text x="5" y="42">
                    {number(maxDelay)}
                  </text>
                  {timed.map((r, i) => {
                    const x = 65 + (r.mapped.area / maxArea) * 465,
                      y = 250 - (r.timing!.delayNs / maxDelay) * 210
                    return (
                      <g
                        key={r.id}
                        className={
                          r.variant === 'baseline'
                            ? 'hardware-baseline'
                            : 'hardware-variant'
                        }
                      >
                        <title>
                          {r.id}: {r.mapped.area} µm², {r.timing!.delayNs} ns
                        </title>
                        {r.variant === 'baseline' ? (
                          <circle
                            cx={x}
                            cy={y}
                            r={selection === r.id ? 7 : 5}
                          />
                        ) : (
                          <path d={`M${x} ${y - 7}l7 7-7 7-7-7Z`} />
                        )}
                        <text x={x + 8} y={y + (i % 2 ? 14 : -10)}>
                          {r.unit} {r.variant === 'baseline' ? 'B' : 'T'}
                        </text>
                      </g>
                    )
                  })}
                </svg>
              ) : (
                <p>
                  Delay unavailable:{' '}
                  {artifact.records[0]?.timingUnavailable ??
                    'no valid timing reports'}
                  . No area-delay points are fabricated.
                </p>
              )}
              <p className="text-small">
                B / circle: baseline. T / diamond: tree leading-bit search.
                Points are recorded mapping outcomes, not a proven Pareto
                frontier.
              </p>
            </figure>
            <figure>
              <figcaption>Mapped combinational gate depth</figcaption>
              <ul className="hardware-depth">
                {artifact.records.map((r) => (
                  <li key={r.id}>
                    <span>
                      {r.unit} / {r.variant}:{' '}
                      <strong>{r.mapped.logicDepth}</strong>
                    </span>
                    <div
                      aria-hidden="true"
                      style={{
                        width: `${(r.mapped.logicDepth / maxDepth) * 100}%`,
                      }}
                    />
                  </li>
                ))}
              </ul>
              <p className="text-small">
                Depth is the maximum number of mapped cells along an
                input-to-output path. It is not delay; different cells have
                different timing arcs.
              </p>
            </figure>
          </div>
          {selected && (
            <section aria-label="Selected hardware experiment">
              <h3>
                {selected.unit} / {selected.variant}
              </h3>
              <p aria-live="polite">
                Verification:{' '}
                {selected.verification.mismatches === 0 ? 'passed' : 'FAILED'},{' '}
                {number(selected.verification.cases)} cases.{' '}
                {selected.verification.method}; whole-unit equivalence is not
                proven. The tree rounder alone has a separate combinational
                equivalence proof.
              </p>
              {selected.timing ? (
                <>
                  <p>
                    Worst constrained path:{' '}
                    <code>{selected.timing.startpoint}</code> →{' '}
                    <code>{selected.timing.endpoint}</code>. Estimated data
                    arrival:{' '}
                    <strong>{number(selected.timing.delayNs)} ns</strong>.
                  </p>
                  <p>
                    Critical-path RTL stage breakdown unavailable:{' '}
                    {selected.timing.stageBreakdownUnavailable}.
                  </p>
                  <PathDetails label="Inspect reported critical-path cell sequence">
                    <ol className="hardware-path">
                      {selected.timing.cells.map((cell, i) => (
                        <li key={i}>
                          <code>
                            {cell.pin} ({cell.type}) · +
                            {number(cell.arcDelayNs)} ns, arrival{' '}
                            {number(cell.arrivalNs)} ns
                          </code>
                        </li>
                      ))}
                    </ol>
                  </PathDetails>
                </>
              ) : (
                <p>Timing unavailable: {selected.timingUnavailable}</p>
              )}
              <PathDetails label="Inspect maximum-depth path (distinct from timing-critical path)">
                <ol className="hardware-path">
                  {selected.depthPath.map((c) => (
                    <li key={c.name}>
                      <code>
                        {c.name} ({c.type})
                      </code>
                    </li>
                  ))}
                </ol>
              </PathDetails>
            </section>
          )}
        </>
      )}
      <details>
        <summary>Experimental assumptions and provenance</summary>
        <dl className="mixed-data">
          {[
            ['Source commit', artifact.gitCommit],
            ['Uncommitted work at measurement', String(artifact.gitDirty)],
            ['Recorded time', artifact.timestamp],
            ['Yosys', artifact.tools.yosys],
            ['ABC', artifact.tools.abc],
            ['OpenSTA', artifact.tools.opensta ?? 'Unavailable'],
            [
              'Library',
              `${artifact.config.library.name}; ${artifact.config.library.corner}`,
            ],
            ['Library SHA-256', artifact.config.library.sha256],
            ['Mapping', JSON.stringify(artifact.config.mapping)],
            ['Timing constraints', JSON.stringify(artifact.config.timing)],
          ].map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>
                <code>{value}</code>
              </dd>
            </div>
          ))}
        </dl>
        <p>
          Raw reports, generated netlists, per-run manifests, and constraints
          are written to <code>rtl/build/analysis/</code> by the reproduction
          scripts. Source hashes bind this recorded artifact to the checked-in
          experiment.
        </p>
        <a href="https://github.com/nivupai/fp-play/blob/main/docs/HARDWARE_SYNTHESIS.md">
          Synthesis methodology and reproduction commands
        </a>
      </details>
    </section>
  )
}
