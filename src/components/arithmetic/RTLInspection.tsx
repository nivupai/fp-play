import { useState } from 'react'
import {
  compareRecordedRTL,
  RTL_METADATA,
  RTL_SIMULATIONS,
} from '../../lib/arithmetic/rtl'
import type { RTLSample } from '../../lib/arithmetic/rtl'
import {
  decodeFP16,
  formatFP16Bits,
  fp16BitsToHex,
} from '../../lib/numerics/fp16'
import { formatExactFP16 } from '../../lib/explorer/presentation'

export function RTLInspection({
  samples = RTL_SIMULATIONS,
}: {
  samples?: readonly RTLSample[]
}) {
  const [index, setIndex] = useState(0)
  const sample = samples[index],
    comparison = compareRecordedRTL(sample),
    signals = sample.signals
  const rtl = decodeFP16(sample.rtlBits)
  return (
    <details className="panel rtl-inspection">
      <summary>Inspect precomputed FP16 RTL simulations</summary>
      <div className="rtl-content">
        <h3>Software ↔ RTL inspection</h3>
        <p>
          This panel reads small, precomputed Verilator simulation records. The
          browser does not run Verilator, synthesize hardware, or simulate
          custom operands. Choose a recorded example independently of the
          arithmetic form above.
        </p>
        <p className="muted text-small">
          Recorded with {RTL_METADATA.tool}. Corpus:{' '}
          {RTL_METADATA.cases.toLocaleString('en-US')} cases;{' '}
          {RTL_METADATA.mismatches} mismatches. Raw NaN signs/payloads and all
          four flags were compared strictly. Source hashes are included in the
          artifact.
        </p>
        <label htmlFor="rtl-example">Recorded RTL example</label>
        <select
          id="rtl-example"
          value={index}
          onChange={(e) => setIndex(Number(e.target.value))}
        >
          {samples.map((s, i) => (
            <option value={i} key={s.name}>
              {s.name}
            </option>
          ))}
        </select>
        <p className="font-mono">
          A {fp16BitsToHex(sample.a)}{' '}
          {sample.operation === 'subtract' ? '−' : '+'} B{' '}
          {fp16BitsToHex(sample.b)}
        </p>
        <p aria-live="polite" aria-label="RTL agreement">
          {comparison.agrees
            ? 'Agreement: result bits and all exception flags match.'
            : 'Disagreement: recorded RTL differs from the current software reference.'}
        </p>
        <div className="rtl-results">
          <div>
            <h4>Current software reference</h4>
            <p className="rtl-value" aria-label="RTL software result">
              {fp16BitsToHex(comparison.software.bits)} ·{' '}
              {formatExactFP16(comparison.software.decoded.value)}
            </p>
            <p aria-label="RTL software flags">
              {Object.entries(comparison.software.flags)
                .map(([key, value]) => `${key}: ${value}`)
                .join('; ')}
            </p>
          </div>
          <div>
            <h4>Recorded RTL result</h4>
            <p className="rtl-value" aria-label="RTL recorded result">
              {fp16BitsToHex(sample.rtlBits)} · {formatExactFP16(rtl.value)}
            </p>
            <p aria-label="RTL recorded flags">
              {Object.entries(sample.rtlFlags)
                .map(([key, value]) => `${key}: ${value}`)
                .join('; ')}
            </p>
            <code tabIndex={0} className="rtl-binary">
              {formatFP16Bits(sample.rtlBits)}
            </code>
          </div>
        </div>
        {signals.special ? (
          <p>
            Special-value path: ordinary finite alignment, normalization, and
            rounding are bypassed. Their recorded zero signals are defaults, not
            fabricated arithmetic stages.
          </p>
        ) : (
          <p>
            Lossless fixed-grid alignment retains every bit through
            cancellation. G/R/S are taken from the exact 41-bit magnitude only
            after arithmetic. The wide datapath is deliberately unoptimized.
          </p>
        )}
        <ol className="rtl-stages">
          <li>Classify operands and apply B’s effective sign.</li>
          <li>Compare effective exponents.</li>
          <li>Align exactly to integer units of 2^-24.</li>
          <li>Add/subtract magnitudes and select the result sign.</li>
          <li>Find the leading bit and normalize.</li>
          <li>Extract retained significand and guard, round, sticky.</li>
          <li>Round nearest-even and handle carry.</li>
          <li>Pack zero/subnormal/normal/infinity/NaN and flags.</li>
        </ol>
        <dl className="mixed-data">
          {[
            ['Effective B sign', String(signals.effectiveBSign)],
            ['Exponent difference', String(signals.exponentDifference)],
            [
              'Aligned A · 40 bits',
              BigInt(`0x${signals.alignedA}`).toString(2).padStart(40, '0'),
            ],
            [
              'Aligned B · 40 bits',
              BigInt(`0x${signals.alignedB}`).toString(2).padStart(40, '0'),
            ],
            [
              'Arithmetic magnitude · 41 bits',
              BigInt(`0x${signals.magnitude}`).toString(2).padStart(41, '0'),
            ],
            ['Normalization left shift', String(signals.normalizationShift)],
            [
              'Normalized magnitude · 41 bits',
              BigInt(`0x${signals.normalizedMagnitude}`)
                .toString(2)
                .padStart(41, '0'),
            ],
            ['Discard count on fixed grid', String(signals.discardCount)],
            [
              'Retained significand',
              signals.retained.toString(2).padStart(11, '0'),
            ],
            [
              'Guard / round / sticky',
              `${signals.guard} / ${signals.round} / ${signals.sticky}`,
            ],
            ['Increment', String(Boolean(signals.increment))],
            ['Rounded raw exponent', String(signals.exponentRounded)],
          ].map(([label, text]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd
                tabIndex={text.length > 24 ? 0 : undefined}
                aria-label={`RTL ${label}`}
              >
                {text}
              </dd>
            </div>
          ))}
        </dl>
        <p className="muted text-small">
          These are combinational observation signals, not clocked stages.
          Waveform timestamps are transaction observation indices, not measured
          propagation delays or clock-cycle latency. Underflow is always clear
          for finite FP16 add/sub: tiny results are exact multiples of 2^-24.
        </p>
      </div>
    </details>
  )
}
