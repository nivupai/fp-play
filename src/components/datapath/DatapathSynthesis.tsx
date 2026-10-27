import { linkSynthesis } from '../../lib/datapath/synthesis'
import type { SynthesisContext } from '../../lib/datapath/synthesis'
import type { RecordedExample } from '../../lib/datapath/artifacts'
export function DatapathSynthesis({
  record,
  context,
  error,
}: {
  record: RecordedExample
  context: SynthesisContext | null
  error: string | null
}) {
  if (!context)
    return (
      <section aria-label="Datapath synthesis comparison">
        <h3>Synthesis comparison unavailable</h3>
        <p>{error ?? 'No measured synthesis data'}</p>
      </section>
    )
  const linked = linkSynthesis(record, context)
  return (
    <section aria-label="Datapath synthesis comparison">
      <h3>Source-linked synthesis · {linked.unit}</h3>
      <p>{linked.reason}</p>
      {!linked.compatible ? (
        <p>
          No automatic area/delay comparison is shown for mismatched source
          revisions.
        </p>
      ) : linked.records.length === 0 ? (
        <p>No measured experiments are available for this unit.</p>
      ) : (
        <div
          className="hardware-table"
          role="region"
          aria-label="Synthesis comparison table"
          tabIndex={0}
        >
          <table>
            <thead>
              <tr>
                {[
                  'Variant',
                  'Mapped cells',
                  'Cell area (µm²)',
                  'Estimated delay (ns)',
                  'Logic depth',
                  'Verification',
                ].map((h) => (
                  <th scope="col" key={h}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {linked.records.map((r) => (
                <tr key={r.id}>
                  <th scope="row">{r.variant}</th>
                  <td>{r.cells}</td>
                  <td>{r.area}</td>
                  <td>{r.delay === null ? 'Unavailable' : r.delay}</td>
                  <td>{r.depth}</td>
                  <td>
                    {r.mismatches === 0 ? 'Passed' : 'FAILED'} ·{' '}
                    {r.cases.toLocaleString('en-US')} cases
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {linked.unit === 'add' && linked.compatible && (
        <p>
          The tree-rounder experiment does not modify the adder. No alternative
          adder measurement exists.
        </p>
      )}
      <p>
        Fewer levels do not guarantee lower delay: mapped cell types, drive
        strengths, slew, fanout, and loading affect timing arcs. These are
        whole-unit, functional-output measurements; no block area or stage delay
        is inferred.
      </p>
      <details>
        <summary>Exact synthesis library, constraints, and toolchain</summary>
        <dl className="mixed-data">
          {[
            ['Source commit', context.commit],
            ['Measured at', context.timestamp],
            ...Object.entries(context.tools),
            ['Library', JSON.stringify(context.config.library)],
            ['Mapping settings', JSON.stringify(context.config.mapping)],
            ['Timing assumptions', JSON.stringify(context.config.timing)],
          ].map(([name, value]) => (
            <div key={name}>
              <dt>{name}</dt>
              <dd>
                <code className="rtl-binary" tabIndex={0}>
                  {String(value ?? 'Unavailable')}
                </code>
              </dd>
            </div>
          ))}
        </dl>
      </details>
      <p className="text-small">
        Mapped cell area is not physical chip area. Delay uses Nangate45 timing
        and ideal interconnect, not post-layout extraction. No physical power or
        silicon measurements.
      </p>
    </section>
  )
}
