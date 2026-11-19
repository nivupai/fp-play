import { useState } from 'react'
import {
  decodeSources,
  policiesOf,
  pooledFlips,
} from '../../lib/runtime/decode'
import { BarList, ScrollTable } from './charts'

const repo = 'https://github.com/nivupai/fp-play/blob/main/'

/** R3: does sharding change the tokens? With random and trained weights. */
export function DecodeWorkspace() {
  const [key, setKey] = useState(decodeSources[0].key)
  const source = decodeSources.find((s) => s.key === key)!
  const policies = policiesOf(source)
  const [policyChoice, setPolicy] = useState(policies[0])
  const policy = policies.includes(policyChoice) ? policyChoice : policies[0]
  const runs = source.grid.filter((g) => g.policy === policy)
  const rows = source.matrix.filter((r) => r.policy === policy)
  const variants = rows[0]?.variants.map((v) => v.variant) ?? []
  const comparison = decodeSources.filter((s) => s.key !== 'r3')
  return (
    <section className="runtime-ws" aria-labelledby="decode-heading">
      <h2 id="decode-heading">
        Decode divergence and batch invariance (runtime R3)
      </h2>
      <p>
        A tiny Llama-style decoder run tensor-parallel through the executed
        AllReduce schedules inside the serving engine. Does greedy decoding
        change with P, the schedule or the batch composition? Data from{' '}
        <a href={`${repo}runtime/results/r3-decode.md`}>
          <code>r3-decode.json</code>
        </a>{' '}
        and{' '}
        <a href={`${repo}runtime/results/r3-trained.md`}>
          <code>r3-trained.json</code>
        </a>
        .
      </p>
      <div className="control-grid">
        <label htmlFor="decode-source">Model and prompts</label>
        <select
          id="decode-source"
          value={key}
          onChange={(e) => setKey(e.target.value)}
        >
          {decodeSources.map((s) => (
            <option key={s.key} value={s.key}>
              {s.title}
            </option>
          ))}
        </select>
        <label htmlFor="decode-policy">Wire policy</label>
        <select
          id="decode-policy"
          value={policy}
          onChange={(e) => setPolicy(e.target.value)}
        >
          {policies.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </div>
      <p className="muted">{source.description}</p>
      <h3>Divergence from P = 1</h3>
      <BarList
        caption={`Prompts (of ${source.prompts}) whose greedy tokens differ from the P = 1 baseline, per schedule and P (${policy} wire).`}
        bars={runs.map((g) => ({
          label: `${g.schedule}, P = ${g.world}`,
          value: g.diverged,
          text: `${g.diverged} of ${source.prompts}`,
        }))}
      />
      <ScrollTable label="divergence table">
        <table className="runtime-table">
          <caption>
            Every grid run ({policy} wire). At-risk tokens: generated tokens up
            to and including each prompt’s first divergence; a flip is the token
            where it diverges.
          </caption>
          <thead>
            <tr>
              <th scope="col">Schedule</th>
              <th scope="col">P</th>
              <th scope="col">Prompts diverged</th>
              <th scope="col">Prompts with bit-identical logits</th>
              <th scope="col">At-risk tokens</th>
              <th scope="col">Flips</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((g) => (
              <tr key={`${g.schedule}-${g.world}`}>
                <th scope="row">{g.schedule}</th>
                <td>{g.world}</td>
                <td>
                  {g.diverged} of {source.prompts}
                </td>
                <td>{g.identical}</td>
                <td>{g.atRisk}</td>
                <td>{g.flips}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollTable>
      <h3>Margins and flips: random against trained weights</h3>
      <p>
        Both rerun models see the same copy prompts. A flip needs a top-2 margin
        no larger than twice the logit perturbation.
      </p>
      <BarList
        log
        caption={`Median top-2 margin at P = 1 (logits, ${policy} wire).`}
        bars={comparison.map((s) => {
          const m = s.margins.find((x) => x.policy === policy)!
          return { label: s.title, value: m.p50, text: m.p50.toFixed(4) }
        })}
      />
      <ScrollTable label="margin table">
        <table className="runtime-table">
          <caption>
            Top-2 margins at P = 1 and flips pooled over the divergence grid (
            {policy} wire).
          </caption>
          <thead>
            <tr>
              <th scope="col">Model and prompts</th>
              <th scope="col">Margin min / p10 / p50 / p90</th>
              <th scope="col">Copy tokens correct</th>
              <th scope="col">Flips / at-risk tokens</th>
              <th scope="col">Prompts diverged</th>
            </tr>
          </thead>
          <tbody>
            {decodeSources.map((s) => {
              const m = s.margins.find((x) => x.policy === policy)!
              const f = pooledFlips(s, policy)
              return (
                <tr key={s.key}>
                  <th scope="row">{s.title}</th>
                  <td>
                    {m.min === null ? '–' : m.min.toFixed(4)} /{' '}
                    {m.p10.toFixed(4)} / {m.p50.toFixed(4)} / {m.p90.toFixed(4)}
                  </td>
                  <td>
                    {m.copyCorrect === null
                      ? 'not a copy task'
                      : `${m.copyCorrect} of ${m.tokens}`}
                  </td>
                  <td>
                    {f.flips} / {f.atRisk}
                  </td>
                  <td>
                    {f.diverged} of {f.prompts}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </ScrollTable>
      <h3>Batch invariance (P = {source.matrixWorld} and the P = 1 control)</h3>
      <p>
        Each prompt’s logits in each batch composition, compared bit for bit
        with the prompt served alone. Prediction: only the ring’s bits may
        depend on batch composition.{' '}
        <strong>
          {source.predictionHeld
            ? 'The prediction held exactly.'
            : 'The prediction did not hold.'}
        </strong>
      </p>
      <ScrollTable label="batch-invariance table">
        <table className="runtime-table">
          <caption>
            Prompts (of {source.prompts}) whose logits differ from serving
            alone, and in parentheses whose tokens differ ({policy} wire).
          </caption>
          <thead>
            <tr>
              <th scope="col">Schedule, P</th>
              <th scope="col">Predicted invariant</th>
              {variants.map((v) => (
                <th scope="col" key={v}>
                  {v}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.schedule}-${r.world}`}>
                <th scope="row">
                  {r.schedule}, P = {r.world}
                </th>
                <td>{r.predicted ? 'yes' : 'no'}</td>
                {r.variants.map((v) => (
                  <td
                    key={v.variant}
                    className={v.logits > 0 ? 'runtime-differs' : undefined}
                  >
                    {v.logits === 0
                      ? 'identical'
                      : `${v.logits} prompts (tokens: ${v.tokens})`}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollTable>
      <p className="muted text-small">
        The trained model solves one synthetic copy task with confident
        predictions; it is not a language model.
      </p>
    </section>
  )
}
