import { useState } from 'react'
import {
  formatMetric as fmt,
  fusedComparison,
  hardwareEvidence,
  unitNames,
} from '../../lib/showcase/evidence'
const repo = 'https://github.com/nivupai/fp-play/blob/main/'
export function ResearchShowcase() {
  const [unit, setUnit] = useState('all'),
    [selectedId, setSelectedId] = useState('baseline-fma')
  const records = hardwareEvidence.records.filter(
    (r) => unit === 'all' || r.unit === unit,
  )
  const selected = records.find((r) => r.id === selectedId) ?? records[0]
  const measured = records.filter((r) => r.delay !== null)
  const maxArea = Math.max(1, ...records.map((r) => r.area)),
    maxDelay = Math.max(1, ...measured.map((r) => r.delay!))
  const comparison = fusedComparison()
  const { config, tools } = hardwareEvidence.source
  return (
    <section className="research-showcase" aria-labelledby="research-heading">
      <div className="section-heading">
        <p className="eyebrow">
          Research showcase / recorded Nangate45 experiment
        </p>
        <h2 id="research-heading">
          One arithmetic contract. Seven hardware experiments.
        </h2>
        <p>
          Compare four baseline units and three leading-bit tree variants. All
          share functional outputs, the same library and constraints. These are
          whole-unit mapping and timing estimates, not physical chip
          measurements.
        </p>
      </div>
      {comparison && (
        <p className="research-observation">
          <strong>Observed:</strong> baseline FMA used{' '}
          {comparison.areaReduction.toFixed(2)}% less mapped cell area and{' '}
          {comparison.delayReduction === null
            ? 'unavailable estimated timing'
            : `${comparison.delayReduction.toFixed(2)}% lower estimated path delay`}{' '}
          than nonfused multiply-add in this configuration.
        </p>
      )}
      <label htmlFor="showcase-unit">Compare hardware units</label>
      <select
        id="showcase-unit"
        value={unit}
        onChange={(e) => setUnit(e.target.value)}
      >
        <option value="all">All measured units</option>
        {Object.entries(unitNames).map(([id, name]) => (
          <option key={id} value={id}>
            {name}
          </option>
        ))}
      </select>
      <div className="research-grid">
        <figure>
          <figcaption>
            Mapped cell area versus constrained estimated delay
          </figcaption>
          {measured.length ? (
            <div
              className="research-plot"
              role="group"
              aria-label="Scrollable hardware showcase plot"
              tabIndex={0}
            >
              <svg
                viewBox="0 0 600 310"
                role="img"
                aria-label="Hardware area and delay scatter plot; select experiments using the exact table"
              >
                <path d="M65 20V255H580" fill="none" stroke="currentColor" />
                <text x="200" y="299">
                  Mapped cell area (µm²)
                </text>
                <text x="8" y="15">
                  Delay (ns)
                </text>
                <text x="50" y="275">
                  0
                </text>
                <text x="475" y="275">
                  {fmt(maxArea)}
                </text>
                <text x="8" y="40">
                  {fmt(maxDelay)}
                </text>
                {measured.map((r) => {
                  const x = 65 + (480 * r.area) / maxArea,
                    y = 250 - (210 * r.delay!) / maxDelay
                  return (
                    <g
                      key={r.id}
                      className={
                        r.variant === 'baseline'
                          ? 'research-baseline'
                          : 'research-tree'
                      }
                    >
                      <title>
                        {r.id}: {r.area} µm²; {r.delay} ns
                      </title>
                      {r.variant === 'baseline' ? (
                        <circle
                          cx={x}
                          cy={y}
                          r={selected.id === r.id ? 9 : 6}
                        />
                      ) : (
                        <path d={`M${x} ${y - 8}l8 8-8 8-8-8Z`} />
                      )}
                      {selected.id === r.id && (
                        <circle
                          cx={x}
                          cy={y}
                          r="13"
                          fill="none"
                          stroke="currentColor"
                        />
                      )}
                    </g>
                  )
                })}
              </svg>
            </div>
          ) : (
            <p>No recorded timing is available.</p>
          )}
          <p className="text-small">
            Circles: baseline. Diamonds: tree. Scroll the plot horizontally on
            narrow screens. Select a row to inspect its critical path. Exact
            values and keyboard controls are in the table.
          </p>
        </figure>
        <div
          className="research-path"
          aria-label="Showcase selected experiment"
        >
          <h3>
            {unitNames[selected.unit]} · {selected.variant}
          </h3>
          <p>
            <strong>{fmt(selected.area)} µm²</strong> cell area ·{' '}
            <strong>
              {selected.delay === null
                ? 'Unavailable'
                : fmt(selected.delay) + ' ns'}
            </strong>{' '}
            estimated delay
          </p>
          <p>
            {selected.cells} mapped cells · {selected.depth} logic levels
          </p>
          <h4>Recorded timing-critical endpoints</h4>
          <p>
            <code>{selected.startpoint ?? 'Unavailable'}</code> →{' '}
            <code>{selected.endpoint ?? 'Unavailable'}</code>
          </p>
          <p>
            {selected.pathCells ?? 'Unavailable'} cell arcs in the recorded
            timing path. This is distinct from maximum graph depth.
          </p>
          <p>
            {selected.stageBreakdownUnavailable}. The tree-FMA worst path ends
            at underflow flag <code>flags[1]</code>; that identifies an
            endpoint, not a measured delay for the rounding stage.
          </p>
          <a href="#hardware">
            Inspect full cell paths and experiment manifests →
          </a>
        </div>
      </div>
      <div
        className="hardware-table"
        role="region"
        aria-label="Showcase exact hardware measurements"
        tabIndex={0}
      >
        <table>
          <caption>
            One library/corner/configuration. Values come directly from the
            source-hashed Milestone 12 artifact.
          </caption>
          <thead>
            <tr>
              {[
                'Unit / variant',
                'Cell area (µm²)',
                'Delay (ns)',
                'Cells',
                'Logic depth',
              ].map((h) => (
                <th scope="col" key={h}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {records.map((r) => (
              <tr key={r.id}>
                <th scope="row">
                  <button
                    type="button"
                    onClick={() => setSelectedId(r.id)}
                    aria-pressed={selected.id === r.id}
                  >
                    {unitNames[r.unit]} / {r.variant}
                  </button>
                </th>
                <td>{fmt(r.area)}</td>
                <td>{r.delay === null ? 'Unavailable' : fmt(r.delay)}</td>
                <td>{r.cells}</td>
                <td>{r.depth}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="research-grid">
        <div>
          <h3>Observation and hypothesis</h3>
          <p>
            <strong>Measured:</strong> baseline FMA is smaller and faster here.
            The tree multiplier has fewer logic levels (109 → 99), yet greater
            area and delay. No tree variant improved both area and delay.
          </p>
          <p>
            <strong>Hypothesis:</strong> retaining one exact product and
            rounding once may avoid costs of serial intermediate rounding and a
            second arithmetic unit. Different mapping, fanout, slew and cell
            strengths may explain the tree results. These hypotheses were not
            isolated by ablation; they are not universal claims about FMA or
            tree circuits.
          </p>
        </div>
        <div>
          <h3>Exact experimental conditions</h3>
          <p>
            {config.library.name} · {config.library.corner}.{' '}
            {config.timing.wireModel}; no placement, routing or extracted
            parasitics. Input slew {config.timing.inputTransitionNs} ns; output
            load {config.timing.outputLoadFf} fF; IO delays{' '}
            {config.timing.inputDelayNs}/{config.timing.outputDelayNs} ns;
            virtual IO clock {config.timing.virtualClockPeriodNs} ns. No
            hardware clock or achievable frequency is implied.
          </p>
          <p>
            ABC uses {config.mapping.driver} drive,{' '}
            {config.mapping.outputLoadFf} fF load and a{' '}
            {config.mapping.abcDelayTargetPs} ps optimization target; the target
            is not an achieved delay.
          </p>
          <dl className="mixed-data">
            {Object.entries(tools).map(([name, value]) => (
              <div key={name}>
                <dt>{name}</dt>
                <dd>
                  <code className="rtl-binary" tabIndex={0}>
                    {value ?? 'Unavailable'}
                  </code>
                </dd>
              </div>
            ))}
          </dl>
          <a href={`${repo}docs/HARDWARE_SYNTHESIS.md`}>
            Full conditions, tool pins and constraints →
          </a>
        </div>
      </div>
      <details>
        <summary>Measurement provenance and snapshot source</summary>
        <p>
          Measured {hardwareEvidence.source.timestamp} at source commit{' '}
          <code>{hardwareEvidence.source.gitCommit}</code>; dirty working tree:{' '}
          {String(hardwareEvidence.source.gitDirty)}. The recorded hashes, not
          the commit alone, identify the measured sources.
        </p>
        <p>
          Source:{' '}
          <a href={`${repo}${hardwareEvidence.source.path}`}>
            {hardwareEvidence.source.path}
          </a>
          . SHA-256:{' '}
          <code className="rtl-binary" tabIndex={0}>
            {hardwareEvidence.source.sha256}
          </code>
        </p>
        <p>
          Library SHA-256:{' '}
          <code className="rtl-binary" tabIndex={0}>
            {hardwareEvidence.source.config.library.sha256}
          </code>
        </p>
      </details>
      <nav className="landing-links" aria-label="Research documentation">
        <a href={`${repo}docs/CASE_STUDY.md`}>Read the technical case study</a>
        <a href={`${repo}docs/REPRODUCIBILITY.md`}>Reproduce the experiments</a>
        <a href="#datapath">Inspect recorded datapaths and waveforms</a>
      </nav>
    </section>
  )
}
