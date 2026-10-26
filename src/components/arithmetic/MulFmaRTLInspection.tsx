import { useState } from 'react'
import artifact from '../../../rtl/vectors/mul-fma-simulation.json'
import {
  multiplyFP16,
  fmaFP16,
  nonfusedMultiplyAddFP16,
} from '../../lib/numerics/fp16Multiply'
import {
  fp16BitsToHex,
  formatFP16Bits,
  fp16BitsToFloat,
} from '../../lib/numerics/fp16'
import { formatExactFP16 } from '../../lib/explorer/presentation'

/** Only simulator-observed signals are displayed; software is an explicit comparison. */
export function MulFmaRTLInspection() {
  const [index, setIndex] = useState(0)
  const sample = artifact.samples[index]
  const software = [
    multiplyFP16(sample.a, sample.b),
    fmaFP16(sample.a, sample.b, sample.c),
    nonfusedMultiplyAddFP16(sample.a, sample.b, sample.c),
  ]
  const recorded = [
    [sample.multiply, sample.multiplyFlags],
    [sample.fused, sample.fusedFlags],
    [sample.nonfused, sample.nonfusedFlags],
  ]
  const mask = (f: (typeof software)[number]['flags']) =>
    (f.invalid ? 8 : 0) |
    (f.overflow ? 4 : 0) |
    (f.underflow ? 2 : 0) |
    (f.inexact ? 1 : 0)
  const agrees = software.every(
    (result, i) =>
      result.bits === recorded[i][0] && mask(result.flags) === recorded[i][1],
  )
  return (
    <details className="panel rtl-inspection">
      <summary>Inspect multiplier and true FMA RTL</summary>
      <div className="rtl-content">
        <h3>Exact product → fused or separately rounded</h3>
        <p>
          These are genuine precomputed Verilator records, not live browser
          hardware simulation. The exact fused product is never rounded before C
          is added. Neural-network accumulations can therefore retain residuals
          lost by separately rounded operations.
        </p>
        <p className="muted text-small">
          {artifact.metadata.tool}.{' '}
          {artifact.metadata.cases.toLocaleString('en-US')} verified triples;{' '}
          {artifact.metadata.mismatches} mismatches against software and the
          independent lattice oracle.
        </p>
        <label htmlFor="mul-fma-rtl-example">
          Recorded multiply/FMA example
        </label>
        <select
          id="mul-fma-rtl-example"
          value={index}
          onChange={(e) => setIndex(Number(e.target.value))}
        >
          {artifact.samples.map((s, i) => (
            <option key={s.name} value={i}>
              {s.name}
            </option>
          ))}
        </select>
        <p className="font-mono">
          A {fp16BitsToHex(sample.a)} × B {fp16BitsToHex(sample.b)} + C{' '}
          {fp16BitsToHex(sample.c)}
        </p>
        <p aria-live="polite">
          {agrees
            ? 'Agreement: all three recorded results and exception flags match the current software.'
            : 'Disagreement: recorded outputs differ from current software.'}
        </p>
        <div className="rtl-results">
          {['Multiply', 'True FMA', 'Nonfused'].map((name, i) => (
            <div key={name}>
              <h4>{name}</h4>
              <p aria-label={`${name} hardware result`}>
                {fp16BitsToHex(recorded[i][0])} ·{' '}
                {formatExactFP16(fp16BitsToFloat(recorded[i][0]))}
              </p>
              <code className="rtl-binary" tabIndex={0}>
                {formatFP16Bits(recorded[i][0])}
              </code>
              <p>
                Flags: invalid {Boolean(recorded[i][1] & 8).toString()},
                overflow {Boolean(recorded[i][1] & 4).toString()}, underflow{' '}
                {Boolean(recorded[i][1] & 2).toString()}, inexact{' '}
                {Boolean(recorded[i][1] & 1).toString()}
              </p>
            </div>
          ))}
        </div>
        <p>
          Nonfused flags union the rounded multiplier flags (
          {sample.multiplyFlags}) and subsequent adder flags (
          {sample.additionFlags}). Mask: invalid 8, overflow 4, underflow 2,
          inexact 1.
        </p>
        <p>
          Exact grid: 2⁻⁴⁸. Product alignment is 80 bits, addend alignment 64
          bits, and accumulation 81 bits. Wide recorded integers below are
          hexadecimal strings, never rounded JavaScript Numbers. Product
          exponent multiplies the raw 22-bit significand product.
        </p>
        {(
          [
            ['Multiplier', sample.multiplySignals],
            ['FMA', sample.fmaSignals],
          ] as const
        ).map(([name, signals]) => (
          <section key={name} aria-label={`${name} recorded signals`}>
            <h4>{name} signals</h4>
            {signals.special !== 0 ? (
              <p>
                Special-value bypass: finite product, alignment, normalization,
                and rounding debug defaults do not describe executed arithmetic.
              </p>
            ) : (
              <p>
                Normalize the exact magnitude, retain eleven significant bits
                (or the subnormal quantum), then use guard/round/sticky and
                retained parity for one nearest-even rounding. Tininess uses
                unbounded-exponent precision rounding.
              </p>
            )}
            <dl className="mixed-data">
              {Object.entries(signals).map(([key, value]) => (
                <div key={key}>
                  <dt>{key}</dt>
                  <dd>
                    <code
                      aria-label={`${name} RTL ${key}`}
                      tabIndex={0}
                      className="rtl-binary"
                    >
                      {String(value)}
                    </code>
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>
    </details>
  )
}
