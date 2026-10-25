import { useMemo, useState } from 'react'
import {
  ATTENTION_POLICIES,
  attentionValues,
  binary64Attention,
  calculateAttention,
} from '../../lib/numerics/attention'
import type {
  AttentionMode,
  AttentionResult,
  SoftmaxMethod,
} from '../../lib/numerics/attention'
import { ATTENTION_EXAMPLES } from '../../lib/attention/examples'
import {
  attentionErrors,
  attentionDifference,
  sharedHeatScale,
  matrixText,
  parseAttentionMatrices,
} from '../../lib/attention/presentation'
import { exactValue } from '../../lib/mixed/presentation'
import { decodeFloat, floatBinary, floatHex } from '../../lib/numerics/formats'
import { Heatmap } from './Heatmap'
const stages = ['scores', 'logits', 'probabilities', 'output'] as const
const titles = {
  scores: 'QKᵀ scores',
  logits: 'Scaled logits',
  probabilities: 'Softmax probabilities',
  output: 'Attention output',
}
type View = (typeof stages)[number]
function Policy({ result }: { result: AttentionResult }) {
  const p = result.policy
  return (
    <p className="attention-policy text-small">
      Input {p.input}; QK mode {p.scoreDot}, sequential; scale {p.scale};
      softmax {p.softmax}; probabilities {p.probability}; PV mode {p.valueDot},
      sequential; final {p.output}.{' '}
      {p.scoreDot === 'C' || p.scoreDot === 'D'
        ? 'Exact product inside each FP32 FMA.'
        : 'Products and sums round separately.'}
    </p>
  )
}
export function AttentionLab() {
  const initial = ATTENTION_EXAMPLES[3]
  const [input, setInput] = useState(initial.input),
    [q, setQ] = useState(matrixText(initial.input.q)),
    [k, setK] = useState(matrixText(initial.input.k)),
    [v, setV] = useState(matrixText(initial.input.v))
  const [leftMode, setLeft] = useState<AttentionMode>(initial.left),
    [rightMode, setRight] = useState<AttentionMode>(initial.right),
    [method, setMethod] = useState<SoftmaxMethod>('stable')
  const [example, setExample] = useState<typeof initial | null>(initial),
    [error, setError] = useState(''),
    [view, setView] = useState<View>('scores'),
    [step, setStep] = useState(0),
    [selected, setSelected] = useState<[number, number]>([0, 0])
  const calculation = useMemo(
    () => ({
      left: calculateAttention(input, leftMode, method),
      right: calculateAttention(input, rightMode, method),
      reference: calculateAttention(input, 'fp32', 'stable'),
      binary64: binary64Attention(input),
    }),
    [input, leftMode, rightMode, method],
  )
  const leftValues = attentionValues(calculation.left[view]),
    rightValues = attentionValues(calculation.right[view]),
    refValues = attentionValues(calculation.reference[view])
  const leftError = attentionErrors(leftValues, refValues),
    rightError = attentionErrors(rightValues, refValues),
    scale = sharedHeatScale(
      leftValues,
      rightValues,
      refValues,
      calculation.binary64[view],
    )
  const errorScale = sharedHeatScale(leftError.absolute, rightError.absolute)
  const event =
      calculation.left.events[
        Math.min(step, calculation.left.events.length - 1)
      ],
    cell =
      calculation.left[view][
        Math.min(selected[0], calculation.left[view].length - 1)
      ][Math.min(selected[1], calculation.left[view][0].length - 1)]
  const probabilityRow = Math.min(selected[0], calculation.left.rows.length - 1)
  return (
    <section
      id="attention"
      className="explorer"
      aria-labelledby="attention-heading"
    >
      <header className="explorer-header mixed-header">
        <p className="eyebrow">07 / ATTENTION PRECISION</p>
        <h2 id="attention-heading">Attention Precision Lab</h2>
        <p className="muted">
          Follow softmax(QKᵀ / √d_k)V from input bits to output error. Small,
          deterministic numerical models—not a GPU kernel or performance
          benchmark.
        </p>
      </header>
      <div className="explorer-content mixed-content">
        <aside className="panel attention-notice">
          <h3>What this simulator models</h3>
          <p>
            Products, sums, and normalization use explicit format rounding.
            Exponentials use <code>Math.exp</code> in binary64, then quantize;
            correctly rounded FP16/BF16/FP32 exponentiation is not claimed.
            Scaling starts with binary64 <code>1 / Math.sqrt(d_k)</code>. Stable
            FP32 is a numerical baseline, not an exact mathematical answer.
          </p>
        </aside>
        <div className="panel">
          <h3>Reproducible experiments</h3>
          <div className="presets">
            {ATTENTION_EXAMPLES.map((e) => (
              <button
                key={e.id}
                className="preset-button"
                onClick={() => {
                  setInput(e.input)
                  setQ(matrixText(e.input.q))
                  setK(matrixText(e.input.k))
                  setV(matrixText(e.input.v))
                  setLeft(e.left)
                  setRight(e.right)
                  setMethod(e.method)
                  setExample(e)
                  setError('')
                  setStep(0)
                  setSelected([0, 0])
                }}
              >
                {e.title}
              </button>
            ))}
          </div>
          {example && (
            <div className="attention-example">
              <p>{example.phenomenon}</p>
              <p>
                <strong>Expected:</strong> {example.expected}
              </p>
              <details>
                <summary>Experiment verification method</summary>
                <p>{example.verification}</p>
              </details>
            </div>
          )}
        </div>
        <form
          className="panel mixed-controls"
          onSubmit={(e) => {
            e.preventDefault()
            try {
              const parsed = parseAttentionMatrices(q, k, v)
              setInput(parsed)
              setError('')
              setExample(null)
              setStep(0)
              setSelected([0, 0])
            } catch (e) {
              setError((e as Error).message)
            }
          }}
        >
          <h3>Edit matrices</h3>
          <div className="attention-editors">
            {[
              { name: 'Q', value: q, setter: setQ },
              { name: 'K', value: k, setter: setK },
              { name: 'V', value: v, setter: setV },
            ].map(({ name, value, setter }) => (
              <label key={name}>
                {name} matrix
                <textarea
                  value={value}
                  onChange={(e) => setter(e.target.value)}
                  aria-invalid={!!error}
                  aria-describedby="attention-input-help attention-input-error"
                />
              </label>
            ))}
          </div>
          <p id="attention-input-help" className="muted text-small">
            Rows: newline or semicolon. Columns: comma or space. Q: 1–4 rows ×
            1–8 features; K: 1–8 rows × same features; V: K rows × 1–4 values.
            Decimal text is binary64-first; -0, Infinity, -Infinity and NaN are
            explicit inputs. Apply Q/K/V together.
          </p>
          <button className="primary-button">Apply attention matrices</button>
          <p id="attention-input-error" role={error ? 'alert' : undefined}>
            {error}
          </p>
        </form>
        <div className="panel attention-selectors">
          {[
            { label: 'Left precision', value: leftMode, setter: setLeft },
            { label: 'Right precision', value: rightMode, setter: setRight },
          ].map(({ label, value, setter }) => (
            <label key={label}>
              {label}
              <select
                value={value}
                onChange={(e) => {
                  setter(e.target.value as AttentionMode)
                  setStep(0)
                }}
              >
                {Object.entries(ATTENTION_POLICIES).map(([key, p]) => (
                  <option value={key} key={key}>
                    {p.label}
                  </option>
                ))}
              </select>
            </label>
          ))}
          <label>
            Attention softmax method
            <select
              value={method}
              onChange={(e) => {
                setMethod(e.target.value as SoftmaxMethod)
                setStep(0)
              }}
            >
              <option value="stable">Stable · subtract row maximum</option>
              <option value="naive">
                Naive · deliberately unstable baseline
              </option>
            </select>
          </label>
          <label>
            Attention stage view
            <select
              value={view}
              onChange={(e) => {
                setView(e.target.value as View)
                setSelected([0, 0])
              }}
            >
              {stages.map((stage) => (
                <option key={stage} value={stage}>
                  {titles[stage]}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="attention-notice">
          {method === 'naive'
            ? 'Naive softmax is deliberately unstable: positive exponentials can overflow, or all negative exponentials can underflow to a zero denominator.'
            : 'Stable softmax avoids large positive exponential arguments for finite logits. It cannot repair input/score overflow or nonfinite maximum subtraction.'}{' '}
          No nonfinite result is replaced, clipped, or assigned a fallback
          distribution.
        </p>
        <div className="attention-comparisons">
          {(['left', 'right'] as const).map((side, i) => (
            <article
              className="panel"
              key={side}
              aria-label={`${side === 'left' ? 'Left' : 'Right'} attention calculation`}
            >
              <h3>{side === 'left' ? 'Left' : 'Right'} configuration</h3>
              <Policy result={calculation[side]} />
              <Heatmap
                title={`${side === 'left' ? 'Left' : 'Right'} ${titles[view]}`}
                values={i ? rightValues : leftValues}
                encodings={calculation[side][view]}
                scale={scale}
                onSelect={
                  side === 'left' ? (r, c) => setSelected([r, c]) : undefined
                }
                selected={selected}
              />
              <p className="text-small">
                Flags:{' '}
                {Object.entries(calculation[side].flags)
                  .map(
                    ([key, value]) => `${key}: ${value ? 'raised' : 'clear'}`,
                  )
                  .join('; ')}
                . Includes modeled conversions; exp flags themselves are not
                modeled.
              </p>
              {calculation[side].issues.length > 0 && (
                <ul aria-label={`${side} attention diagnostics`}>
                  {calculation[side].issues.map((issue, j) => (
                    <li key={j}>{issue}</li>
                  ))}
                </ul>
              )}
            </article>
          ))}
        </div>
        <div className="panel">
          <h3>
            Selected left cell · row {selected[0] + 1}, column {selected[1] + 1}
          </h3>
          <p>
            Use Tab and Enter/Space on any left heatmap cell. Full numerical
            text accompanies each tint.
          </p>
          <dl className="mixed-data">
            <div>
              <dt>Exact decoded decimal</dt>
              <dd tabIndex={0} aria-label="Attention selected exact value">
                {exactValue(decodeFloat(cell.format, cell.bits).value)}
              </dd>
            </div>
            <div>
              <dt>Raw bits</dt>
              <dd tabIndex={0} aria-label="Attention selected bits">
                {floatBinary(cell.format, cell.bits)} ·{' '}
                {floatHex(cell.format, cell.bits)}
              </dd>
            </div>
            <div>
              <dt>Classification</dt>
              <dd>{decodeFloat(cell.format, cell.bits).classification}</dd>
            </div>
          </dl>
        </div>
        <details className="panel" open>
          <summary>FP32 and binary64 baselines</summary>
          <p className="muted text-small">
            The FP32 baseline always uses stable softmax, separate FP32
            products/additions, and explicit FP32 stage rounding. The separate
            binary64 baseline uses Number arithmetic and Math.exp. Neither is an
            exact/high-precision mathematical reference. Offline verification
            uses 100-digit Decimal.
          </p>
          <Heatmap
            title={`Stable FP32 ${titles[view]}`}
            values={refValues}
            encodings={calculation.reference[view]}
            scale={scale}
          />
          <Heatmap
            title={`Binary64 ${titles[view]}`}
            values={calculation.binary64[view]}
            scale={sharedHeatScale(
              calculation.binary64[view],
              leftValues,
              rightValues,
              refValues,
            )}
          />
        </details>
        <div className="panel">
          <h3>Error against stable FP32 · {titles[view]}</h3>
          <p className="muted text-small">
            Absolute/relative errors below use binary64 arithmetic on decoded
            values. Relative error is undefined at a zero reference; all errors
            are undefined for nonfinite operands. Naive-versus-stable
            comparisons include the algorithm change.
          </p>
          <div className="attention-comparisons">
            <Heatmap
              title="Left absolute error"
              values={leftError.absolute}
              scale={errorScale}
            />
            <Heatmap
              title="Right absolute error"
              values={rightError.absolute}
              scale={errorScale}
            />
          </div>
          <details>
            <summary>Relative error heatmaps</summary>
            <div className="attention-comparisons">
              <Heatmap
                title="Left relative error"
                values={leftError.relative}
                scale={sharedHeatScale(leftError.relative, rightError.relative)}
              />
              <Heatmap
                title="Right relative error"
                values={rightError.relative}
                scale={sharedHeatScale(leftError.relative, rightError.relative)}
              />
            </div>
          </details>
        </div>
        <details className="panel">
          <summary>Final output difference · left minus right</summary>
          <p className="muted text-small">
            Signed decoded-value differences in binary64; undefined for
            nonfinite outputs.
          </p>
          <Heatmap
            title="Left minus right final output"
            values={attentionDifference(
              attentionValues(calculation.left.output),
              attentionValues(calculation.right.output),
            )}
            scale={sharedHeatScale(
              attentionDifference(
                attentionValues(calculation.left.output),
                attentionValues(calculation.right.output),
              ),
            )}
          />
        </details>
        <div className="panel">
          <h3>Probability row {probabilityRow + 1}</h3>
          <label>
            Inspect probability row
            <select
              value={probabilityRow}
              onChange={(e) => setSelected([Number(e.target.value), 0])}
            >
              {calculation.left.rows.map((_, i) => (
                <option value={i} key={i}>
                  Row {i + 1}
                </option>
              ))}
            </select>
          </label>
          <p aria-label="Attention probability row">
            Left:{' '}
            {attentionValues(calculation.left.probabilities)[
              probabilityRow
            ].join(', ')}{' '}
            · sum (binary64 display):{' '}
            {attentionValues(calculation.left.probabilities)[
              probabilityRow
            ].reduce((a, b) => a + b, 0)}
            . Quantized probabilities are not renormalized.
          </p>
          <p>
            Denominator:{' '}
            {exactValue(
              decodeFloat(
                calculation.left.rows[probabilityRow].denominator.format,
                calculation.left.rows[probabilityRow].denominator.bits,
              ).value,
            )}
          </p>
        </div>
        <div className="panel">
          <h3>Every rounding boundary · left configuration</h3>
          <div className="mixed-step-controls">
            <button
              onClick={() => setStep(Math.max(0, step - 1))}
              disabled={step === 0}
            >
              Previous attention step
            </button>
            <label htmlFor="attention-step">Attention computation step</label>
            <select
              id="attention-step"
              value={Math.min(step, calculation.left.events.length - 1)}
              onChange={(e) => setStep(Number(e.target.value))}
            >
              {calculation.left.events.map((e, i) => (
                <option key={i} value={i}>
                  {i + 1}. {e.stage} · {e.operation} · ({e.row + 1},
                  {e.column + 1})
                </option>
              ))}
            </select>
            <button
              onClick={() =>
                setStep(Math.min(calculation.left.events.length - 1, step + 1))
              }
              disabled={step >= calculation.left.events.length - 1}
            >
              Next attention step
            </button>
          </div>
          <p aria-live="polite">
            Attention step {step + 1} of {calculation.left.events.length}:{' '}
            {event.stage} / {event.operation}
          </p>
          <p>{event.note}</p>
          <dl className="mixed-data">
            {[
              [
                'Stage operands',
                event.operands
                  .map(
                    (v) =>
                      `${v.format} ${floatHex(v.format, v.bits)} = ${exactValue(decodeFloat(v.format, v.bits).value)}`,
                  )
                  .join('; ') || 'Binary64 source',
              ],
              [
                'Before quantization',
                event.before === null
                  ? 'Modeled arithmetic; see operands'
                  : exactValue(event.before),
              ],
              [
                'Stage result',
                `${event.result.format} ${floatHex(event.result.format, event.result.bits)} = ${exactValue(decodeFloat(event.result.format, event.result.bits).value)}`,
              ],
              [
                'Stage bits',
                floatBinary(event.result.format, event.result.bits),
              ],
              [
                'Stage rounding',
                `halfway: ${event.halfway}; increment: ${event.increment}`,
              ],
              [
                'Stage flags',
                Object.entries(event.flags)
                  .map(([key, value]) => `${key}: ${value}`)
                  .join('; '),
              ],
            ].map(([label, text]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd
                  aria-label={`Attention ${label.toLowerCase()}`}
                  tabIndex={0}
                >
                  {text}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </section>
  )
}
