import { useState } from 'react'
import { explainGRS } from '../../lib/datapath/reference'
import type { ReferenceResult } from '../../lib/datapath/reference'
import type { Phase } from '../../lib/datapath/artifacts'
const steps = [
  'Retained LSB',
  'Guard',
  'Round',
  'Sticky',
  'Increment',
  'Carry / transition',
]
export function GRSExplorer({
  phase,
  software,
}: {
  phase: Phase
  software: ReferenceResult
}) {
  const [step, setStep] = useState(0),
    grs = explainGRS(phase)
  const decision = software.trace.find((s) => s.kind === 'round'),
    normal = software.trace.find((s) => s.kind === 'normalize')
  if (grs.status !== 'available')
    return (
      <section aria-label="GRS explorer">
        <h3>GRS microscope</h3>
        <p>{grs.message}</p>
      </section>
    )
  const descriptions = [
    `Retained significand ${grs.retained}; its LSB is ${grs.lsb} (${grs.lsb ? 'odd' : 'even'}). Derived parity of a recorded integer.`,
    `Recorded guard bit: ${grs.guard}. This is the first discarded bit.`,
    `Recorded round bit: ${grs.round}. This is the second discarded bit.`,
    `Recorded sticky bit: ${grs.sticky}. It is the OR of all remaining discarded bits.`,
    `RNE rule: G AND (R OR S OR retained LSB). Boolean expectation ${Number(grs.expectedIncrement)}; recorded increment ${grs.increment}.`,
    `Carry inferred from recorded retained + increment: ${grs.inferredCarry}. This is an annotation, not a recorded carry wire. Software trace carry: ${decision?.carry ?? 'unavailable'}.`,
  ]
  const transition =
    normal &&
    decision &&
    normal.retained < 1024n &&
    decision.finalSignificand === 1024n &&
    decision.finalExponent === -14
  return (
    <section className="datapath-grs" aria-label="GRS explorer">
      <h3>GRS microscope</h3>
      <p aria-label="GRS conclusion">
        {grs.message}{' '}
        {grs.agrees
          ? 'Recorded increment agrees with RNE.'
          : 'DISAGREEMENT: recorded increment violates the displayed RNE rule.'}
      </p>
      <div className="datapath-grs-bits" aria-label="Recorded rounding inputs">
        {[
          ['LSB (derived)', grs.lsb],
          ['Guard', grs.guard],
          ['Round', grs.round],
          ['Sticky', grs.sticky],
          ['Increment', grs.increment],
        ].map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      <ol className="datapath-stage-buttons">
        {steps.map((name, i) => (
          <li key={name}>
            <button
              type="button"
              aria-current={step === i ? 'step' : undefined}
              onClick={() => setStep(i)}
            >
              {name}
            </button>
          </li>
        ))}
      </ol>
      <div
        tabIndex={0}
        role="group"
        aria-label="GRS step navigation"
        onKeyDown={(e) => {
          if (e.target !== e.currentTarget) return
          const next =
            e.key === 'ArrowRight'
              ? Math.min(5, step + 1)
              : e.key === 'ArrowLeft'
                ? Math.max(0, step - 1)
                : e.key === 'Home'
                  ? 0
                  : e.key === 'End'
                    ? 5
                    : null
          if (next !== null) {
            e.preventDefault()
            setStep(next)
          }
        }}
      >
        <p aria-live="polite">
          {step + 1} / 6 · {descriptions[step]}
        </p>
        <button
          type="button"
          disabled={step === 0}
          onClick={() => setStep((s) => s - 1)}
        >
          Previous GRS step
        </button>
        <button
          type="button"
          disabled={step === 5}
          onClick={() => setStep((s) => s + 1)}
        >
          Next GRS step
        </button>
      </div>
      {transition && (
        <p>
          Software trace: subnormal-to-normal transition. This is distinct from
          an eleven-bit significand carry.
        </p>
      )}
      <p className="text-small">
        Software decisions and parity/carry annotations are identified
        separately from observed RTL wires. Tininess uses precision rounding
        with unbounded exponent; the packed exponent alone does not determine
        underflow.
      </p>
    </section>
  )
}
