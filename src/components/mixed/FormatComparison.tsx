import { useState } from 'react'
import {
  decodeFloat,
  encodeFloat,
  floatBinary,
  floatHex,
  floatULP,
  FORMATS,
  nextDown,
  nextUp,
} from '../../lib/numerics/formats'
import type { FormatName } from '../../lib/numerics/formats'
import {
  parseDecimalInput,
  formatNumber,
} from '../../lib/explorer/presentation'
import { exactValue, quantizationError } from '../../lib/mixed/presentation'

export function FormatComparison() {
  const [draft, setDraft] = useState('1.0009765625'),
    [value, setValue] = useState(1.0009765625),
    [error, setError] = useState('')
  return (
    <section
      id="formats"
      className="explorer"
      aria-labelledby="formats-heading"
    >
      <header className="explorer-header mixed-header">
        <p className="eyebrow">05 / REPRESENTATION ACROSS FORMATS</p>
        <h2 id="formats-heading">Format Comparison</h2>
        <p className="muted">
          Exponent width buys range. Significand width buys precision. All
          conversions round directly from binary64, nearest-even.
        </p>
      </header>
      <div className="explorer-content mixed-content">
        <form
          className="panel mixed-controls"
          onSubmit={(event) => {
            event.preventDefault()
            const parsed = parseDecimalInput(draft)
            if (parsed.ok) {
              setValue(parsed.value)
              setError('')
            } else setError(parsed.error)
          }}
        >
          <label htmlFor="format-input">Compare a binary64 value</label>
          <input
            id="format-input"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            aria-invalid={!!error}
            aria-describedby="format-help format-error"
          />
          <button className="primary-button" type="submit">
            Compare formats
          </button>
          <p id="format-help" className="muted text-small">
            Decimal text is parsed into binary64 first, not exact decimal
            arithmetic. Use -0, Infinity, -Infinity, or NaN.
          </p>
          <p id="format-error" role={error ? 'alert' : undefined}>
            {error}
          </p>
        </form>
        <p>
          Original binary64:{' '}
          <output
            aria-label="Format original value"
            className="mixed-exact"
            tabIndex={0}
          >
            {exactValue(value)}
          </output>
        </p>
        <div className="format-grid">
          {(Object.keys(FORMATS) as FormatName[]).map((name) => {
            const f = FORMATS[name],
              bits = encodeFloat(name, value),
              d = decodeFloat(name, bits),
              binary = floatBinary(name, bits),
              error = quantizationError(value, d.value)
            return (
              <article
                className="panel"
                key={name}
                aria-label={`${name} representation`}
              >
                <h3>{name}</h3>
                <p
                  className="mixed-result"
                  aria-label={`${name} quantized value`}
                >
                  {formatNumber(d.value)}
                </p>
                <p className="muted text-small">
                  Shortest binary64 display; exact decimal below.
                </p>
                <div
                  className="mixed-bits"
                  tabIndex={0}
                  aria-label={`${name} bit layout`}
                >
                  <span className="sign-text">S {binary[0]}</span>
                  <span className="exponent-text">
                    E {binary.slice(1, 1 + f.exponentBits)}
                  </span>
                  <span className="fraction-text">
                    F {binary.slice(1 + f.exponentBits)}
                  </span>
                </div>
                <dl className="mixed-data">
                  {[
                    ['Hex', floatHex(name, bits)],
                    ['Classification', d.classification],
                    [
                      'Sign / raw exponent / fraction integer',
                      `${d.sign} / ${d.exponent} / ${d.fraction}`,
                    ],
                    [
                      'Exponent bias / unbiased exponent',
                      `${f.bias} / ${d.unbiasedExponent ?? 'Not applicable'}`,
                    ],
                    [
                      'Implicit leading bit',
                      d.hasImplicitLeadingBit
                        ? '1'
                        : d.exact
                          ? '0'
                          : 'Not applicable',
                    ],
                    ['Exact decimal', exactValue(d.value)],
                    ['Smallest normal', exactValue(2 ** (1 - f.bias))],
                    [
                      'Largest finite magnitude',
                      exactValue((2 - 2 ** -f.fractionBits) * 2 ** f.bias),
                    ],
                    ['ULP (binade spacing)', exactValue(floatULP(name, bits))],
                    [
                      'Previous / next numerical value',
                      `${formatNumber(decodeFloat(name, nextDown(name, bits)).value)} / ${formatNumber(decodeFloat(name, nextUp(name, bits)).value)}`,
                    ],
                    ['Absolute error', error.absolute],
                    ['Relative error', error.relative],
                    [
                      'NaN metadata',
                      d.classification === 'nan'
                        ? `${d.isQuietNaN ? 'Quiet' : 'Signaling'}; payload ${d.nanPayload}`
                        : 'Not applicable',
                    ],
                  ].map(([label, text]) => (
                    <div key={label}>
                      <dt>{label}</dt>
                      <dd tabIndex={text.length > 24 ? 0 : undefined}>
                        {text}
                      </dd>
                    </div>
                  ))}
                </dl>
                <p className="muted text-small">
                  1 sign · {f.exponentBits} exponent · {f.fractionBits} stored
                  fraction bits. Normal precision: {f.fractionBits + 1} bits.
                  Normal exponents {1 - f.bias}…{f.bias}. Minimum quantum 2^
                  {1 - f.bias - f.fractionBits}.
                </p>
              </article>
            )
          })}
        </div>
        <p className="muted text-small">
          ULP is sign-independent binade spacing, including at negative powers
          of two. Both zeros step up to the minimum positive subnormal and down
          to its negative. Outward infinity saturates. NaNs preserve raw
          adjacency; Number NaNs canonicalize and cannot carry their original
          payload/sign.
        </p>
      </div>
    </section>
  )
}
