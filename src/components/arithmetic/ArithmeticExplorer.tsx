import { useCallback, useEffect, useState } from 'react'
import { Encoding } from './Encoding'
import { RTLInspection } from './RTLInspection'
import { FMAComparison } from './FMAComparison'
import {
  compareMultiplyAddFP16,
  multiplyFP16,
} from '../../lib/numerics/fp16Multiply'
import { MULTIPLY_EXAMPLES } from '../../lib/arithmetic/multiplyExamples'
import { ArithmeticTrace } from './ArithmeticTrace'
import type { FormEvent } from 'react'
import { decodeFP16, fp16BitsToHex } from '../../lib/numerics'
import { calculateFP16 } from '../../lib/numerics/fp16Arithmetic'
import type { FP16Operation } from '../../lib/numerics/fp16Arithmetic'
import { formatExactFP16 } from '../../lib/explorer/presentation'
import { ARITHMETIC_EXAMPLES } from '../../lib/arithmetic/examples'
import {
  formatExactDyadic,
  parseArithmeticOperand,
} from '../../lib/arithmetic/presentation'

type ExplorerOperation = FP16Operation | 'multiply' | 'fma'
type OperandDraft = { text: string; mode: 'decimal' | 'hex' }

// A hash link opens an existing, independently verified example; no new numerical path.
const FMA_START_INDEX = 8
const fmaStart = MULTIPLY_EXAMPLES[FMA_START_INDEX]
function rawSelection(index: number) {
  const entry = MULTIPLY_EXAMPLES[index]
  return {
    a: entry.a,
    b: entry.b,
    c: entry.c ?? 0,
    operation: entry.operation as ExplorerOperation,
    sources: [entry.a, entry.b, entry.c ?? 0].map(
      (bits) => `Raw ${fp16BitsToHex(bits)}`,
    ),
  }
}
function rawDrafts(index: number): OperandDraft[] {
  const entry = MULTIPLY_EXAMPLES[index]
  return [entry.a, entry.b, entry.c ?? 0].map((bits) => ({
    text: fp16BitsToHex(bits),
    mode: 'hex',
  }))
}

export function ArithmeticExplorer() {
  const [initialFma] = useState(
    () => typeof window !== 'undefined' && window.location.hash === '#fma',
  )
  const [drafts, setDrafts] = useState<OperandDraft[]>(() =>
    initialFma
      ? rawDrafts(FMA_START_INDEX)
      : [
          { text: '1', mode: 'decimal' },
          { text: '0.5', mode: 'decimal' },
          { text: '0', mode: 'decimal' },
        ],
  )
  const [operation, setOperation] = useState<ExplorerOperation>(
    initialFma ? fmaStart.operation : 'add',
  )
  const [active, setActive] = useState(() =>
    initialFma
      ? rawSelection(FMA_START_INDEX)
      : {
          a: 0x3c00,
          b: 0x3800,
          c: 0,
          operation: 'add' as ExplorerOperation,
          sources: [
            'Entered 1; binary64 1',
            'Entered 0.5; binary64 0.5',
            'Entered 0; binary64 0',
          ],
        },
  )
  const [errors, setErrors] = useState<(string | null)[]>([null, null, null])
  const [example, setExample] = useState<number | null>(initialFma ? null : 0)
  const [step, setStep] = useState(0)
  const [multiplyExample, setMultiplyExample] = useState<number | null>(
    initialFma ? FMA_START_INDEX : null,
  )
  const [tracePath, setTracePath] = useState('fused')
  const selectMultiplyExample = useCallback((index: number) => {
    setActive(rawSelection(index))
    setDrafts(rawDrafts(index))
    setOperation(MULTIPLY_EXAMPLES[index].operation)
    setStep(0)
    setErrors([null, null, null])
    setExample(null)
    setMultiplyExample(index)
    setTracePath('fused')
  }, [])
  useEffect(() => {
    const handleHash = () => {
      if (window.location.hash === '#fma')
        selectMultiplyExample(FMA_START_INDEX)
    }
    window.addEventListener('hashchange', handleHash)
    if (initialFma)
      document.getElementById('fma')?.scrollIntoView?.({ block: 'start' })
    return () => window.removeEventListener('hashchange', handleHash)
  }, [initialFma, selectMultiplyExample])
  const comparison =
    active.operation === 'fma'
      ? compareMultiplyAddFP16(active.a, active.b, active.c)
      : null
  const result = comparison
    ? comparison.fused
    : active.operation === 'multiply'
      ? multiplyFP16(active.a, active.b)
      : calculateFP16(active.a, active.b, active.operation as FP16Operation)
  const trace =
    comparison && tracePath !== 'fused'
      ? tracePath === 'product'
        ? comparison.nonfused.product.trace
        : comparison.nonfused.addition.trace
      : result.trace

  function apply(event: FormEvent) {
    event.preventDefault()
    const a = parseArithmeticOperand(drafts[0].text, drafts[0].mode)
    const b = parseArithmeticOperand(drafts[1].text, drafts[1].mode)
    const c =
      operation === 'fma'
        ? parseArithmeticOperand(drafts[2].text, drafts[2].mode)
        : { ok: true as const, bits: 0, source: 'Unused' }
    setErrors([
      a.ok ? null : a.error,
      b.ok ? null : b.error,
      c.ok ? null : c.error,
    ])
    if (!a.ok || !b.ok || !c.ok) return
    setActive({
      a: a.bits,
      b: b.bits,
      c: c.bits,
      operation,
      sources: [a.source, b.source, c.source],
    })
    setStep(0)
    setExample(null)
    setMultiplyExample(null)
    setTracePath('fused')
  }

  return (
    <section
      id="arithmetic"
      aria-labelledby="arithmetic-heading"
      className="explorer arithmetic-explorer"
    >
      <div className="explorer-header">
        <div>
          <p className="eyebrow">06 / FOLLOW THE COMPUTATION</p>
          <h2 id="arithmetic-heading">FP16 Arithmetic Explorer</h2>
          <p className="muted">
            Follow exact arithmetic. Compare one rounding with two.
          </p>
        </div>
        <span className="arithmetic-mode">
          ADD / SUB / MUL / FMA · NEAREST EVEN
        </span>
      </div>
      <div className="explorer-content arithmetic-content">
        <p className="muted text-small">
          These are conceptual stages, not clock cycles or a specific hardware
          design. Full alignment tails are retained with exact integer
          arithmetic before rounding once to binary16.
        </p>
        <form
          onSubmit={apply}
          className="arithmetic-form"
          data-operation={operation}
          aria-label="Arithmetic operands"
        >
          {drafts.slice(0, operation === 'fma' ? 3 : 2).map((draft, index) => {
            const name = 'ABC'[index]
            return (
              <div className="arithmetic-operand-input" key={name}>
                <label htmlFor={`operand-${name}`}>Operand {name}</label>
                <select
                  aria-label={`Operand ${name} format`}
                  value={draft.mode}
                  onChange={(event) => {
                    const mode = event.target.value as OperandDraft['mode']
                    setDrafts(
                      drafts.map((entry, i) =>
                        i === index ? { ...entry, mode } : entry,
                      ),
                    )
                  }}
                >
                  <option value="decimal">Decimal / special value</option>
                  <option value="hex">Raw hexadecimal</option>
                </select>
                <input
                  id={`operand-${name}`}
                  value={draft.text}
                  spellCheck={false}
                  autoComplete="off"
                  aria-invalid={!!errors[index]}
                  aria-describedby={`arithmetic-input-help${errors[index] ? ` operand-${name}-error` : ''}`}
                  onChange={(event) => {
                    setDrafts(
                      drafts.map((entry, i) =>
                        i === index
                          ? { ...entry, text: event.target.value }
                          : entry,
                      ),
                    )
                  }}
                />
                {errors[index] && (
                  <p
                    id={`operand-${name}-error`}
                    role="alert"
                    className="input-error"
                  >
                    {name}: {errors[index]}
                  </p>
                )}
              </div>
            )
          })}
          <div className="arithmetic-operation">
            <label htmlFor="arithmetic-operation">Operation</label>
            <select
              id="arithmetic-operation"
              value={operation}
              onChange={(event) =>
                setOperation(event.target.value as ExplorerOperation)
              }
            >
              <option value="add">A + B</option>
              <option value="subtract">A − B</option>
              <option value="multiply">A × B</option>
              <option value="fma">FMA(A, B, C) / nonfused comparison</option>
            </select>
            <button className="primary-button" type="submit">
              Trace operation
            </button>
          </div>
        </form>
        <p id="arithmetic-input-help" className="muted text-small">
          Decimal/scientific text is parsed to binary64, then quantized to FP16
          before arithmetic. Use -0, Infinity, -Infinity, or NaN; raw mode
          requires four hex digits such as 0x7c01 to retain a signaling NaN
          payload. Draft edits apply only on submission.
        </p>
        <div className="arithmetic-operands" data-operation={active.operation}>
          {(active.operation === 'fma'
            ? [active.a, active.b, active.c]
            : [active.a, active.b]
          ).map((bits, index) => {
            const name = 'ABC'[index]
            const decoded = decodeFP16(bits)
            return (
              <div className="panel" key={name}>
                <p className="eyebrow">APPLIED OPERAND {name}</p>
                <p
                  className="arithmetic-value"
                  aria-label={`Operand ${name} quantized`}
                >
                  {formatExactFP16(decoded.value)}
                </p>
                <p
                  className="muted text-small"
                  aria-label={`Operand ${name} source`}
                >
                  {active.sources[index]}
                </p>
                <p
                  className="font-mono"
                  aria-label={`Operand ${name} encoding`}
                >
                  {fp16BitsToHex(bits)} · {decoded.classification}
                  {decoded.classification === 'nan'
                    ? ` · ${decoded.isQuietNaN ? 'quiet' : 'signaling'} · payload ${decoded.nanPayload}`
                    : ''}
                </p>
                <Encoding bits={bits} label={`Operand ${name} bits`} />
              </div>
            )
          })}
        </div>
        <div className="arithmetic-result panel">
          <div>
            <p className="eyebrow">
              APPLIED{' '}
              {active.operation === 'add'
                ? 'A + B'
                : active.operation === 'subtract'
                  ? 'A − B'
                  : active.operation === 'multiply'
                    ? 'A × B'
                    : 'FMA(A, B, C)'}{' '}
              / FP16 RESULT
            </p>
            <p
              className="arithmetic-value"
              aria-label="Arithmetic result decimal"
            >
              {formatExactFP16(result.decoded.value)}
            </p>
            <p className="font-mono" aria-label="Arithmetic result hex">
              {fp16BitsToHex(result.bits)} · {result.decoded.classification}
            </p>
            <Encoding bits={result.bits} label="Arithmetic result binary" />
          </div>
          <div className="arithmetic-comparison">
            <p className="eyebrow">EXACT BEFORE ROUNDING</p>
            <p className="font-mono" aria-label="Arithmetic exact result">
              {result.exact
                ? formatExactDyadic(result.exact)
                : 'Not a finite operation'}
            </p>
            <p className="muted text-small">
              {result.exact
                ? 'Exact mathematical result of the applied FP16 operands, before the final nearest-even rounding. This is distinct from arithmetic on the original decimal text.'
                : 'Special values short-circuit the finite arithmetic pipeline.'}
            </p>
          </div>
        </div>
        <ul
          className="arithmetic-flags"
          aria-label="Arithmetic exception indicators"
        >
          {Object.entries(result.flags).map(([flag, raised]) => (
            <li key={flag} data-raised={raised}>
              <span>{flag}</span>
              <strong aria-label={`Arithmetic ${flag}`}>
                {raised ? 'Raised' : 'Clear'}
              </strong>
            </li>
          ))}
        </ul>
        <p className="muted text-small">
          Flags are per operation. Tininess is detected after eleven-bit
          precision rounding with unbounded exponents; a stored normal result
          can still raise underflow. Underflow requires inexactness as well.
          Multiplication and FMA can raise underflow for inexact tiny results.
          FP16 addition/subtraction cannot produce an inexact tiny result:
          finite operands share the 2⁻²⁴ grid, so subnormal sums are exact.
          Decimal input conversion flags are not included.
        </p>
        <section
          id="fma"
          className="fma-entry"
          aria-label="FMA Comparison"
          tabIndex={-1}
        >
          {comparison ? (
            <>
              <FMAComparison comparison={comparison} />
              <label className="arithmetic-trace-picker">
                Inspect computation path
                <select
                  aria-label="Trace path"
                  value={tracePath}
                  onChange={(event) => {
                    setTracePath(event.target.value)
                    setStep(0)
                  }}
                >
                  <option value="fused">
                    Fused: exact product + C, one rounding
                  </option>
                  <option value="product">
                    Nonfused: rounded multiplication
                  </option>
                  <option value="addition">
                    Nonfused: addition after rounded product
                  </option>
                </select>
              </label>
            </>
          ) : (
            <div className="panel fma-introduction">
              <div>
                <p className="eyebrow">ONE ROUNDING / TWO ROUNDED OPERATIONS</p>
                <h3>FMA Comparison</h3>
                <p className="muted">
                  Can a rounded product hide a small result? Compare fused and
                  nonfused arithmetic with the same FP16 operands and inspect
                  both paths.
                </p>
              </div>
              <button
                type="button"
                className="primary-button"
                onClick={() => selectMultiplyExample(FMA_START_INDEX)}
              >
                Start FMA comparison
              </button>
            </div>
          )}
        </section>
        <ArithmeticTrace trace={trace} step={step} setStep={setStep} />
        <section aria-labelledby="arithmetic-examples-heading">
          <h3 id="arithmetic-examples-heading">Arithmetic experiments</h3>
          <div className="presets arithmetic-examples">
            {ARITHMETIC_EXAMPLES.map((entry, index) => (
              <button
                className="preset-button"
                key={entry.label}
                type="button"
                aria-pressed={example === index}
                onClick={() => {
                  setActive({
                    a: entry.a,
                    b: entry.b,
                    c: 0,
                    operation: entry.operation,
                    sources: [
                      `Raw ${fp16BitsToHex(entry.a)}`,
                      `Raw ${fp16BitsToHex(entry.b)}`,
                    ],
                  })
                  setDrafts([
                    { text: fp16BitsToHex(entry.a), mode: 'hex' },
                    { text: fp16BitsToHex(entry.b), mode: 'hex' },
                    { text: '0', mode: 'decimal' },
                  ])
                  setOperation(entry.operation)
                  setStep(0)
                  setErrors([null, null, null])
                  setExample(index)
                  setMultiplyExample(null)
                  setTracePath('fused')
                }}
              >
                {entry.label}
                <small className="font-mono">
                  → {fp16BitsToHex(entry.expected)}
                </small>
              </button>
            ))}
          </div>
          <p className="muted" aria-label="Arithmetic experiment explanation">
            {example === null
              ? 'Custom operands. Inspect the trace to see exactly which bits survive rounding.'
              : ARITHMETIC_EXAMPLES[example].explanation}
          </p>
        </section>
        <section aria-labelledby="multiply-examples-heading">
          <h3 id="multiply-examples-heading">
            Multiplication &amp; fusion experiments
          </h3>
          <div className="presets arithmetic-examples">
            {MULTIPLY_EXAMPLES.map((entry, index) => (
              <button
                key={entry.label}
                type="button"
                className="preset-button"
                aria-pressed={multiplyExample === index}
                onClick={() => selectMultiplyExample(index)}
              >
                {entry.label}
                <small className="font-mono">
                  → {fp16BitsToHex(entry.expected)}
                </small>
              </button>
            ))}
          </div>
          <p
            aria-label="Multiplication experiment explanation"
            className="muted"
          >
            {multiplyExample === null
              ? 'Choose a verified experiment or enter custom operands.'
              : MULTIPLY_EXAMPLES[multiplyExample].explanation}
          </p>
        </section>
        <RTLInspection />
        <p className="muted text-small">
          NaN policy: first signaling NaN, otherwise first quiet NaN in A/B
          order (A/B/C for FMA); quiet and preserve its original sign and nine
          payload bits. Zero × infinity and invalid infinity combinations return
          0x7e00. Zero × infinity raises invalid even with a quiet NaN C.
          Payload choice is implementation-specific. This educational arithmetic
          model is not a claim of full IEEE 754 conformance.
        </p>
      </div>
    </section>
  )
}
