import { useState } from 'react'
import { dataflow, REASONS, runsFor, shares } from '../../lib/runtime/dataflow'
import { int, pct } from '../../lib/runtime/validate'
import { BarList, ScrollTable, StackedShares } from './charts'

const repo = 'https://github.com/nivupai/fp-play/blob/main/'

/** R5: one transformer layer mapped onto an illustrative accelerator. */
export function DataflowWorkspace() {
  const [machine, setMachine] = useState(dataflow.machines[0].name)
  const runs = runsFor(machine)
  const [shape, setShape] = useState(runs[0].shape)
  const run = runs.find((r) => r.shape === shape) ?? runs[0]
  const m = dataflow.machines.find((x) => x.name === machine)!
  return (
    <section className="runtime-ws" aria-labelledby="dataflow-heading">
      <h2 id="dataflow-heading">Dataflow mapping (runtime R5)</h2>
      <p>
        One 8B-class transformer layer, prefill and decode, mapped onto a toy
        accelerator ({m.rows} × {m.cols} multiply–accumulate array, {m.lanes}{' '}
        vector lanes, {m.sramBytes / 2 ** 20} MiB SRAM, one DMA engine, double
        buffering). Results are cycles from{' '}
        <a href={`${repo}runtime/results/r5-dataflow.md`}>
          <code>runtime/results/r5-dataflow.json</code>
        </a>
        . Every machine constant is illustrative; nothing here is a timing, area
        or power result.
      </p>
      <div className="control-grid">
        <label htmlFor="dataflow-machine">Machine</label>
        <select
          id="dataflow-machine"
          value={machine}
          onChange={(e) => setMachine(e.target.value)}
        >
          {dataflow.machines.map((x) => (
            <option key={x.name} value={x.name}>
              {x.name}
            </option>
          ))}
        </select>
      </div>
      <p className="muted text-small">
        {m.hbmBytesPerCycle} HBM bytes per cycle at an assumed {m.cycleNs} ns
        cycle.
        {m.cycleNs !== 1 &&
          ' This cycle time is FP Play’s recorded FP16 FMA combinational path delay, used only as an assumption; it is not an achievable clock.'}
      </p>
      <h3>Where each shape’s cycles go</h3>
      <StackedShares
        caption="Share of each run's cycles by reason. Exact cycle counts are in the table below."
        legend={REASONS.map(([key, label]) => ({ key, label }))}
        rows={runs.map((r) => {
          const s = shares(r)
          return {
            label: r.shape,
            segments: REASONS.map(([key, label]) => ({
              key,
              label,
              share: s[key],
            })),
          }
        })}
      />
      <BarList
        caption="Cycles divided by the roofline bound (1 means the mapping reaches the bound)."
        bars={runs.map((r) => ({
          label: r.shape,
          value: r.cycles / r.bound,
          text: `${(r.cycles / r.bound).toFixed(2)} × the bound`,
        }))}
      />
      <ScrollTable label="dataflow runs table">
        <table className="runtime-table">
          <caption>
            Every shape on this machine. Roofline bound: max(useful
            multiply–accumulates ÷ array size, HBM bytes ÷ bytes per cycle).
          </caption>
          <thead>
            <tr>
              <th scope="col">Shape</th>
              <th scope="col">Cycles</th>
              <th scope="col">Roofline bound</th>
              <th scope="col">× bound</th>
              <th scope="col">Array utilization</th>
              <th scope="col">FLOPs per HBM byte</th>
              {REASONS.map(([k, label]) => (
                <th scope="col" key={k}>
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.shape}>
                <th scope="row">{r.shape}</th>
                <td>{int(r.cycles)}</td>
                <td>{int(r.bound)}</td>
                <td>{(r.cycles / r.bound).toFixed(2)}</td>
                <td>{pct(r.utilization)}</td>
                <td>{r.intensity.toFixed(1)}</td>
                {REASONS.map(([k]) => (
                  <td key={k}>{int(r.reasons[k])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollTable>
      <h3>Op by op</h3>
      <div className="control-grid">
        <label htmlFor="dataflow-shape">Shape</label>
        <select
          id="dataflow-shape"
          value={run.shape}
          onChange={(e) => setShape(e.target.value)}
        >
          {runs.map((r) => (
            <option key={r.shape} value={r.shape}>
              {r.shape}
            </option>
          ))}
        </select>
      </div>
      <BarList
        caption={`Cycles per op, ${run.shape}.`}
        bars={run.ops.map((o) => ({
          label: o.op,
          value: o.cycles,
          text: `${int(o.cycles)} cycles (${pct(o.cycles / run.cycles)})`,
        }))}
      />
      <ScrollTable label="op table">
        <table className="runtime-table">
          <caption>
            Ops of {run.shape}: cycles, compute and DMA cycles (they overlap
            within an op), DMA waits, the exposed first load, and HBM traffic.
          </caption>
          <thead>
            <tr>
              <th scope="col">Op</th>
              <th scope="col">Cycles</th>
              <th scope="col">Compute</th>
              <th scope="col">DMA</th>
              <th scope="col">DMA wait</th>
              <th scope="col">First load</th>
              <th scope="col">HBM MB</th>
              <th scope="col">Bound by</th>
            </tr>
          </thead>
          <tbody>
            {run.ops.map((o) => (
              <tr key={o.op}>
                <th scope="row">{o.op}</th>
                <td>{int(o.cycles)}</td>
                <td>{int(o.compute)}</td>
                <td>{int(o.dma)}</td>
                <td>{int(o.stall)}</td>
                <td>{int(o.fill)}</td>
                <td>{(o.hbmBytes / 1e6).toFixed(2)}</td>
                <td>{o.compute >= o.dma ? 'compute' : 'memory'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollTable>
    </section>
  )
}
