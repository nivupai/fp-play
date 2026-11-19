import { useState } from 'react'
import { servingGroups, servingMetrics } from '../../lib/runtime/serving'
import { BarList, ScrollTable } from './charts'

const repo = 'https://github.com/nivupai/fp-play/blob/main/'

/** R2: the deterministic serving simulator's recorded scenarios. */
export function ServingWorkspace() {
  const [groupKey, setGroupKey] = useState(servingGroups[0].key)
  const [metricKey, setMetricKey] = useState(servingMetrics[0].key)
  const group = servingGroups.find((g) => g.key === groupKey)!
  const metric = servingMetrics.find((m) => m.key === metricKey)!
  const ms = (ns: number) => (ns / 1e6).toFixed(2)
  return (
    <section className="runtime-ws" aria-labelledby="serving-heading">
      <h2 id="serving-heading">Serving simulator (runtime R2)</h2>
      <p>
        A deterministic discrete-event simulation of an 8B-class model served
        with a paged KV cache, continuous batching, preemption and
        tensor-parallel AllReduce, under Poisson arrivals. Every number is read
        from{' '}
        <a href={`${repo}runtime/results/r2-serving.md`}>
          <code>runtime/results/r2-serving.json</code>
        </a>
        , written by the runtime's documented experiment command. Device and
        link constants are illustrative, not a measured or named system.
      </p>
      <div className="control-grid">
        <label htmlFor="serving-group">Scenario group</label>
        <select
          id="serving-group"
          value={groupKey}
          onChange={(e) => setGroupKey(e.target.value)}
        >
          {servingGroups.map((g) => (
            <option key={g.key} value={g.key}>
              {g.title}
            </option>
          ))}
        </select>
        <label htmlFor="serving-metric">Metric</label>
        <select
          id="serving-metric"
          value={metricKey}
          onChange={(e) => setMetricKey(e.target.value)}
        >
          {servingMetrics.map((m) => (
            <option key={m.key} value={m.key}>
              {m.label}
            </option>
          ))}
        </select>
      </div>
      <p className="muted">
        <strong>{group.title}.</strong> Varies: {group.varies}
      </p>
      <BarList
        caption={`${metric.label} per scenario (${group.title}). Exact values for every metric are in the table below.`}
        bars={group.runs.map((r) => ({
          label: r.label,
          value: metric.value(r),
          text: metric.text(r),
        }))}
      />
      <ScrollTable label="serving table">
        <table className="runtime-table">
          <caption>
            {group.title}: every scenario's recorded metrics (times in
            milliseconds; p50 / p99 over requests, TPOT over samples).
          </caption>
          <thead>
            <tr>
              <th scope="col">Scenario</th>
              <th scope="col">P</th>
              <th scope="col">Rate (req/s)</th>
              <th scope="col">Finished</th>
              <th scope="col">TTFT p50 / p99</th>
              <th scope="col">TPOT p50 / p99</th>
              <th scope="col">E2E p50 / p99</th>
              <th scope="col">Output tok/s</th>
              <th scope="col">Comm share</th>
              <th scope="col">Preemptions</th>
              <th scope="col">Recomputed tokens</th>
              <th scope="col">Swap GB</th>
              <th scope="col">Prefix hit rate</th>
            </tr>
          </thead>
          <tbody>
            {group.runs.map((r) => (
              <tr key={r.label}>
                <th scope="row">{r.label}</th>
                <td>{r.world}</td>
                <td>{r.rate}</td>
                <td>
                  {r.finished} of {r.requests}
                  {r.failed ? `, ${r.failed} failed` : ''}
                  {r.rejected ? `, ${r.rejected} rejected` : ''}
                </td>
                <td>
                  {ms(r.ttft.p50)} / {ms(r.ttft.p99)}
                </td>
                <td>
                  {ms(r.tpot.p50)} / {ms(r.tpot.p99)}
                </td>
                <td>
                  {ms(r.e2e.p50)} / {ms(r.e2e.p99)}
                </td>
                <td>{Math.round(r.outputTokensPerS)}</td>
                <td>{(100 * r.commShare).toFixed(1)}%</td>
                <td>{r.preemptions}</td>
                <td>{r.recomputedTokens}</td>
                <td>{r.swapGB.toFixed(2)}</td>
                <td>
                  {r.prefixHitRate === null
                    ? '–'
                    : `${(100 * r.prefixHitRate).toFixed(1)}%`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollTable>
      <p className="muted text-small">
        Simulated time from a roofline step model with illustrative constants
        and the R1 α–β collective cost model; no compute/communication overlap.
        Full scenario settings and the reference-scheduler checks are in{' '}
        <a href={`${repo}runtime/README.md`}>the runtime README</a>.
      </p>
    </section>
  )
}
