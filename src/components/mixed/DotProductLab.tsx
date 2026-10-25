import { useMemo, useState } from 'react'
import { DOT_MODES, dotProduct } from '../../lib/numerics/dotProduct'
import type { DotMode, ReductionOrder } from '../../lib/numerics/dotProduct'
import { decodeFloat, floatBinary, floatHex } from '../../lib/numerics/formats'
import { MIXED_EXAMPLES } from '../../lib/mixed/examples'
import {
  exactValue,
  formatDyadic,
  parseVector,
} from '../../lib/mixed/presentation'

export function DotProductLab() {
  const initial = MIXED_EXAMPLES[8]
  const [aDraft, setADraft] = useState(initial.a.join(', ')),
    [bDraft, setBDraft] = useState(initial.b.join(', '))
  const [applied, setApplied] = useState({ a: initial.a, b: initial.b }),
    [mode, setMode] = useState<DotMode>(initial.mode),
    [order, setOrder] = useState<ReductionOrder>('sequential')
  const [error, setError] = useState(''),
    [explanation, setExplanation] = useState(initial.explanation),
    [step, setStep] = useState(0)
  const results = useMemo(
    () =>
      (Object.keys(DOT_MODES) as DotMode[]).map((m) =>
        dotProduct(applied.a, applied.b, m, order),
      ),
    [applied, order],
  )
  const result = results.find((r) => r.mode === mode)!,
    selected = result.steps[Math.min(step, result.steps.length - 1)]
  return (
    <section
      id="dot-product"
      className="explorer"
      aria-labelledby="dot-heading"
    >
      <header className="explorer-header mixed-header">
        <p className="eyebrow">06 / MIXED PRECISION</p>
        <h2 id="dot-heading">Dot Product Precision Lab</h2>
        <p className="muted">
          Track what input conversion, product rounding, and accumulator width
          each contribute to a sum of products.
        </p>
      </header>
      <div className="explorer-content mixed-content">
        <div className="panel">
          <h3>Verified experiments</h3>
          <div className="presets">
            {MIXED_EXAMPLES.map((example) => (
              <button
                className="preset-button"
                key={example.title}
                onClick={() => {
                  setADraft(example.a.join(', '))
                  setBDraft(example.b.join(', '))
                  setApplied({ a: example.a, b: example.b })
                  setMode(example.mode)
                  setOrder(example.order)
                  setExplanation(example.explanation)
                  setError('')
                  setStep(0)
                }}
              >
                {example.title}
              </button>
            ))}
          </div>
          <p>{explanation}</p>
        </div>
        <form
          className="panel mixed-controls"
          onSubmit={(event) => {
            event.preventDefault()
            try {
              const a = parseVector(aDraft),
                b = parseVector(bDraft)
              dotProduct(a, b, mode, order)
              setApplied({ a, b })
              setStep(0)
              setError('')
              setExplanation('Custom vectors. Compare all five modes below.')
            } catch (e) {
              setError((e as Error).message)
            }
          }}
        >
          <label htmlFor="vector-a">Vector A</label>
          <textarea
            id="vector-a"
            value={aDraft}
            onChange={(e) => setADraft(e.target.value)}
            aria-invalid={!!error}
            aria-describedby="vector-help vector-error"
          />
          <label htmlFor="vector-b">Vector B</label>
          <textarea
            id="vector-b"
            value={bDraft}
            onChange={(e) => setBDraft(e.target.value)}
            aria-invalid={!!error}
            aria-describedby="vector-help vector-error"
          />
          <p id="vector-help" className="muted text-small">
            1–64 comma- or whitespace-separated values per vector. Equal lengths
            required. Binary64-first; -0, Infinity, -Infinity and NaN supported.
          </p>
          <button className="primary-button">Apply vectors</button>
          <p id="vector-error" role={error ? 'alert' : undefined}>
            {error}
          </p>
        </form>
        <div className="panel mixed-controls">
          <label htmlFor="dot-mode">Precision configuration</label>
          <select
            id="dot-mode"
            value={mode}
            onChange={(e) => {
              setMode(e.target.value as DotMode)
              setStep(0)
            }}
          >
            {Object.entries(DOT_MODES).map(([key, config]) => (
              <option key={key} value={key}>
                {config.label}
              </option>
            ))}
          </select>
          <label htmlFor="dot-order">Reduction order</label>
          <select
            id="dot-order"
            value={order}
            onChange={(e) => {
              setOrder(e.target.value as ReductionOrder)
              setStep(0)
            }}
          >
            <option value="sequential">Sequential (+0 seed)</option>
            <option value="pairwise">Balanced pairwise tree</option>
          </select>
          <p className="muted text-small">
            {order === 'sequential'
              ? 'A/B/E round each product, then each accumulator addition. C/D retain the exact product within each FP32 FMA.'
              : 'Leaves round each product, or execute FMA(a,b,+0) in C/D. Internal nodes round accumulator-format additions; this is not a sequential FMA chain.'}{' '}
            Input conversion flags are separate and not included. Flags below
            union executed arithmetic operations.
          </p>
        </div>
        <div
          className="panel mixed-scroll"
          role="group"
          aria-label="Precision mode comparison"
          tabIndex={0}
        >
          <table className="mixed-table">
            <caption>All modes · {order} reduction</caption>
            <thead>
              <tr>
                <th scope="col">Mode</th>
                <th scope="col">Result</th>
                <th scope="col">Encoding</th>
                <th scope="col">Error versus quantized exact dot</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.mode}>
                  <th scope="row">{DOT_MODES[r.mode].label}</th>
                  <td aria-label={`Mode ${r.mode} result`}>
                    {exactValue(
                      decodeFloat(r.result.format, r.result.bits).value,
                    )}
                  </td>
                  <td>
                    {r.result.format} {floatHex(r.result.format, r.result.bits)}
                  </td>
                  <td>{formatDyadic(r.quantizedError)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h3>Selected result and exact references · mode {mode}</h3>
          <dl className="mixed-data">
            {[
              [
                'Accumulator result',
                exactValue(
                  decodeFloat(result.result.format, result.result.bits).value,
                ),
              ],
              [
                'Accumulator encoding',
                `${floatHex(result.result.format, result.result.bits)} · ${floatBinary(result.result.format, result.result.bits)}`,
              ],
              [
                'Exact original binary64 dot',
                formatDyadic(result.exactOriginal),
              ],
              [
                'Exact quantized-input dot',
                formatDyadic(result.exactQuantized),
              ],
              [
                'Signed error versus original',
                formatDyadic(result.originalError),
              ],
              [
                'Signed error versus quantized',
                formatDyadic(result.quantizedError),
              ],
              [
                'Exception indicators',
                Object.entries(result.flags)
                  .map(
                    ([key, value]) => `${key}: ${value ? 'raised' : 'clear'}`,
                  )
                  .join('; '),
              ],
            ].map(([label, text]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd aria-label={label} tabIndex={0}>
                  {text}
                </dd>
              </div>
            ))}
          </dl>
          <p className="muted text-small">
            Exact references are BigInt dyadic sums, not binary64 summation.
            Nonfinite references/errors are undefined; arithmetic still applies
            NaN/infinity rules. Sequential +0 seeding can change a lone -0 to
            +0. Pairwise leaves preserve multiplication’s zero sign in A/B/E.
          </p>
        </div>
        <div className="panel">
          <h3>Computation steps</h3>
          <div className="mixed-step-controls">
            <button
              onClick={() => setStep(Math.max(0, step - 1))}
              disabled={step === 0}
            >
              Previous step
            </button>
            <label htmlFor="dot-step">Selected computational step</label>
            <input
              id="dot-step"
              type="range"
              min="0"
              max={result.steps.length - 1}
              value={step}
              onChange={(e) => setStep(Number(e.target.value))}
            />
            <button
              onClick={() =>
                setStep(Math.min(result.steps.length - 1, step + 1))
              }
              disabled={step >= result.steps.length - 1}
            >
              Next step
            </button>
          </div>
          <p aria-live="polite">
            Step {step + 1} of {result.steps.length}: {selected.operation};
            terms {selected.terms.map((i) => i + 1).join(', ')}
          </p>
          <dl className="mixed-data">
            {[
              [
                'Step operands',
                selected.operands
                  .map(
                    (o) =>
                      `${o.format} ${floatHex(o.format, o.bits)} = ${exactValue(decodeFloat(o.format, o.bits).value)}`,
                  )
                  .join(' ; '),
              ],
              ['Step exact intermediate', formatDyadic(selected.result.exact)],
              [
                'Step output',
                `${selected.result.format} ${floatHex(selected.result.format, selected.result.bits)} = ${exactValue(decodeFloat(selected.result.format, selected.result.bits).value)}`,
              ],
              [
                'Step binary',
                floatBinary(selected.result.format, selected.result.bits),
              ],
              [
                'Step rounding event',
                `halfway: ${selected.result.halfway}; increment: ${selected.result.increment}; inexact: ${selected.result.flags.inexact}`,
              ],
              [
                'Step flags',
                Object.entries(selected.result.flags)
                  .map(([key, value]) => `${key}: ${value}`)
                  .join('; '),
              ],
              ['Step signed error', formatDyadic(selected.error)],
            ].map(([label, text]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd tabIndex={0} aria-label={label}>
                  {text}
                </dd>
              </div>
            ))}
          </dl>
        </div>
        <ErrorPlot steps={result.steps} selected={step} onSelect={setStep} />
        <details className="panel">
          <summary>Quantized input encodings</summary>
          <div
            className="mixed-scroll"
            tabIndex={0}
            role="group"
            aria-label="Quantized vector data"
          >
            <table className="mixed-table">
              <thead>
                <tr>
                  <th>Term</th>
                  <th>A bits / value</th>
                  <th>B bits / value</th>
                </tr>
              </thead>
              <tbody>
                {result.a.map((a, i) => (
                  <tr key={i}>
                    <th>{i + 1}</th>
                    {[a, result.b[i]].map((v, j) => (
                      <td key={j}>
                        {v.format} {floatHex(v.format, v.bits)} ·{' '}
                        {exactValue(decodeFloat(v.format, v.bits).value)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </div>
    </section>
  )
}
function ErrorPlot({
  steps,
  selected,
  onSelect,
}: {
  steps: ReturnType<typeof dotProduct>['steps']
  selected: number
  onSelect: (step: number) => void
}) {
  const values = steps.map((s) =>
    s.error ? Number(s.error.coefficient) * 2 ** s.error.exponent : NaN,
  )
  const logs = steps.map((s) => {
    if (!s.error || s.error.coefficient === 0n) return null
    const n =
      s.error.coefficient < 0n ? -s.error.coefficient : s.error.coefficient
    const binary = n.toString(2)
    return (
      binary.length -
      1 +
      s.error.exponent +
      Math.log2(
        Number(n >> BigInt(Math.max(0, binary.length - 53))) /
          2 ** Math.min(52, binary.length - 1),
      )
    )
  })
  const finite = logs.filter((x): x is number => x !== null),
    min = Math.min(...finite, 0),
    max = Math.max(...finite, 0)
  return (
    <div className="panel">
      <h3>Error across computational steps</h3>
      <p className="muted text-small">
        Absolute error versus the exact quantized-input dot for each step’s
        terms. Product steps compare one term; addition/FMA steps compare their
        subsum. The plot uses log₂ error with a separate zero row; it is not a
        monotone accumulation timeline for pairwise trees. Exact signed data is
        in the selectable step detail.
      </p>
      <div
        className="mixed-scroll"
        tabIndex={0}
        role="group"
        aria-label="Scrollable dot error plot"
      >
        <svg
          viewBox="0 0 720 240"
          className="mixed-plot"
          role="img"
          aria-label="Absolute error by computational step"
        >
          <text x="12" y="20" fill="currentColor">
            log₂ |error|: {min.toFixed(1)} … {max.toFixed(1)}
          </text>
          <path
            d="M40 40 V180 H690 M40 210 H690"
            fill="none"
            stroke="currentColor"
          />
          <text x="0" y="215" fill="currentColor">
            0
          </text>
          {steps.map((s, i) => {
            if (!s.error) return null
            const y =
              logs[i] === null
                ? 210
                : 175 - ((logs[i]! - min) / (max - min || 1)) * 130
            return (
              <circle
                key={i}
                cx={50 + (i / Math.max(1, steps.length - 1)) * 630}
                cy={y}
                r={i === selected ? 7 : 4}
                fill="var(--accent)"
                stroke="var(--text)"
              >
                <title>
                  Step {i + 1}:{' '}
                  {Number.isFinite(values[i])
                    ? values[i]
                    : 'see exact step detail'}
                  ; {s.operation}
                </title>
              </circle>
            )
          })}
        </svg>
      </div>
      <label htmlFor="error-step">
        Inspect error point (keyboard or text control)
      </label>
      <select
        id="error-step"
        value={selected}
        onChange={(e) => onSelect(Number(e.target.value))}
      >
        {steps.map((s, i) => (
          <option key={i} value={i}>
            Step {i + 1} · {s.operation} · error{' '}
            {s.error
              ? s.error.coefficient === 0n
                ? '0'
                : 'nonzero'
              : 'undefined'}
          </option>
        ))}
      </select>
    </div>
  )
}
