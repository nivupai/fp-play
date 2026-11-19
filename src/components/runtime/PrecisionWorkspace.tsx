import { useState } from 'react'
import {
  oneSite,
  precisionSources,
  training,
  uniformRuns,
} from '../../lib/runtime/precision'
import { sci } from '../../lib/runtime/validate'
import { BarList, LogLineChart, ScrollTable } from './charts'

const repo = 'https://github.com/nivupai/fp-play/blob/main/'

/** R4: per-site precision levels, FP8/INT8 with scales, and the trained model. */
export function PrecisionWorkspace() {
  const [key, setKey] = useState(
    precisionSources[precisionSources.length - 1].key,
  )
  const source = precisionSources.find((s) => s.key === key)!
  const uniform = uniformRuns(source)
  const lower = source.levels.filter((l) => l !== 'fp32')
  const every50 = training.steps.filter(
    (s) => s.step % 50 === 0 || s.step === training.steps.length - 1,
  )
  return (
    <section className="runtime-ws" aria-labelledby="precision-study-heading">
      <h2 id="precision-study-heading">Precision policies (runtime R4)</h2>
      <p>
        The R3 decoder with FP32 master weights, each site given a storage and
        accumulator format, compared with all-FP32 teacher-forced on its own
        greedy continuation. FP8 E4M3, FP8 E5M2 and INT8 use one binary32 scale
        per weight row, activation vector and KV slice. Data from{' '}
        <a href={`${repo}runtime/results/r4-precision.md`}>
          <code>r4-precision.json</code>
        </a>
        ,{' '}
        <a href={`${repo}runtime/results/r4-trained.md`}>
          <code>r4-trained.json</code>
        </a>{' '}
        and{' '}
        <a href={`${repo}runtime/results/r4-training.md`}>
          <code>r4-training.json</code>
        </a>
        .
      </p>
      <div className="control-grid">
        <label htmlFor="precision-source">Model and prompts</label>
        <select
          id="precision-source"
          value={key}
          onChange={(e) => setKey(e.target.value)}
        >
          {precisionSources.map((s) => (
            <option key={s.key} value={s.key}>
              {s.title}
            </option>
          ))}
        </select>
      </div>
      <p className="muted">{source.description}</p>
      <h3>Every site at one level</h3>
      <BarList
        log
        caption="Mean KL divergence from the all-FP32 reference per level (0 for fp32 itself)."
        bars={uniform.map((r) => ({
          label: r.level!,
          value: r.meanKl,
          text: sci(r.meanKl),
        }))}
      />
      <ScrollTable label="uniform-level table">
        <table className="runtime-table">
          <caption>
            Every site at one level: mean KL, largest logit difference, top-1
            agreement with the reference, and prompts (of {source.prompts})
            whose own greedy continuation changed.
          </caption>
          <thead>
            <tr>
              <th scope="col">Level</th>
              <th scope="col">Mean KL</th>
              <th scope="col">Max |Δ logit|</th>
              <th scope="col">Top-1 agreement</th>
              <th scope="col">Greedy diverged</th>
            </tr>
          </thead>
          <tbody>
            {uniform.map((r) => (
              <tr key={r.level!}>
                <th scope="row">{r.level}</th>
                <td>{sci(r.meanKl)}</td>
                <td>{sci(r.maxAbs)}</td>
                <td>
                  {r.top1} of {r.scored}
                </td>
                <td>
                  {r.diverged} of {source.prompts}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollTable>
      <h3>One site at a time</h3>
      <ScrollTable label="one-site table">
        <table className="runtime-table">
          <caption>
            Mean KL with one site lowered and every other site at fp32. The KV
            cache only stores values, so its -acc levels were not run (–).
          </caption>
          <thead>
            <tr>
              <th scope="col">Site</th>
              {lower.map((l) => (
                <th scope="col" key={l}>
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {source.sites.map((site) => (
              <tr key={site}>
                <th scope="row">{site}</th>
                {lower.map((l) => {
                  const r = oneSite(source, site, l)
                  return <td key={l}>{r ? sci(r.meanKl) : '–'}</td>
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollTable>
      <h3>The trained model</h3>
      <p>
        R3’s architecture ({training.parameters.toLocaleString('en-US')}{' '}
        parameters) trained deterministically in binary32 on a copy task (
        <code>BOS s1 … s24 SEP s1 … s24</code>). Weights SHA-256{' '}
        <code className="runtime-hash">{training.sha256}</code>, reproduced bit
        for bit by CI.
      </p>
      <LogLineChart
        caption="Batch loss (nats per scored token) at every training step. Values every 50 steps are in the table below."
        ariaLabel={`Training loss from ${training.steps[0].loss.toFixed(2)} to ${training.steps[training.steps.length - 1].loss.toFixed(5)} nats per token over ${training.steps.length} steps; values in the following table`}
        series={[
          {
            name: 'batch loss',
            points: training.steps.map((s) => ({ x: s.step, y: s.loss })),
          },
        ]}
        xLabel="training step"
        yLabel="loss"
      />
      <ScrollTable label="loss table">
        <table className="runtime-table">
          <caption>Loss curve every 50 steps, and the last step.</caption>
          <thead>
            <tr>
              <th scope="col">Step</th>
              <th scope="col">Learning rate</th>
              <th scope="col">Batch loss</th>
              <th scope="col">Batch accuracy</th>
            </tr>
          </thead>
          <tbody>
            {every50.map((s) => (
              <tr key={s.step}>
                <th scope="row">{s.step}</th>
                <td>{s.lr.toExponential(3)}</td>
                <td>{s.loss.toFixed(5)}</td>
                <td>{(100 * s.accuracy).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollTable>
      <ScrollTable label="held-out accuracy table">
        <table className="runtime-table">
          <caption>
            Held-out accuracy: teacher-forced on {training.heldOut.sequences}{' '}
            sequences (loss {training.heldOut.loss.toFixed(6)}, accuracy{' '}
            {(100 * training.heldOut.accuracy).toFixed(2)}%), and greedy copies
            through the inference models.
          </caption>
          <thead>
            <tr>
              <th scope="col">Inference model</th>
              <th scope="col">Sequences copied exactly</th>
              <th scope="col">Tokens correct</th>
            </tr>
          </thead>
          <tbody>
            {training.greedy.map((g) => (
              <tr key={g.model}>
                <th scope="row">{g.model}</th>
                <td>
                  {g.correctSequences} of {g.sequences}
                </td>
                <td>
                  {g.correctTokens} of {g.tokens}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollTable>
    </section>
  )
}
