import { formatFP16Bits } from '../lib/numerics'

const fields = [
  { label: 'Sign', className: 'sign', positions: [15] },
  { label: 'Exponent', className: 'exponent', positions: [14, 13, 12, 11, 10] },
  {
    label: 'Fraction',
    className: 'fraction',
    positions: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
  },
]

export function BitFields({
  bits,
  onToggle,
}: {
  bits: number
  onToggle: (position: number) => void
}) {
  const binary = formatFP16Bits(bits)
  return (
    <section aria-labelledby="bits-heading" className="bit-board">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">01 / Representation</p>
          <h3 id="bits-heading">Every bit is a switch.</h3>
        </div>
        <p className="muted text-small" id="bits-help">
          Toggle with a click, Enter, or Space.
          <br />
          Bit positions run from 15 to 0.
        </p>
      </div>
      <div className="bit-fields" aria-describedby="bits-help">
        {fields.map((field) => (
          <fieldset
            key={field.label}
            className={`bit-field ${field.className}`}
          >
            <legend>
              {field.label}{' '}
              <span>
                {field.positions.length}{' '}
                {field.positions.length === 1 ? 'bit' : 'bits'}
              </span>
            </legend>
            <div className="bit-grid">
              {field.positions.map((position) => {
                const value = binary[15 - position]
                return (
                  <div className="bit-cell" key={position}>
                    <button
                      type="button"
                      className="bit-button"
                      // The name starts with the visible digit (WCAG 2.5.3, label in name).
                      aria-label={`${value}, ${field.label} bit ${position}`}
                      aria-pressed={value === '1'}
                      onClick={() => onToggle(position)}
                    >
                      {value}
                    </button>
                    <span className="bit-position" aria-hidden="true">
                      {position}
                    </span>
                  </div>
                )
              })}
            </div>
          </fieldset>
        ))}
      </div>
      <p className="bit-caption">
        <span className="font-mono">binary16</span>
        <span>1 sign + 5 exponent + 10 fraction = 16 bits</span>
      </p>
    </section>
  )
}
