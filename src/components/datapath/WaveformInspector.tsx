import { useState } from 'react'
import { valueAt } from '../../lib/datapath/waves'
import type { RecordedWave } from '../../lib/datapath/waves'
import type { RecordedExample } from '../../lib/datapath/artifacts'
export function WaveformInspector({
  wave,
  record,
}: {
  wave: RecordedWave
  record: RecordedExample
}) {
  const defaultName =
    record.operation === 'multiply'
      ? 'multiply_result'
      : record.operation === 'fma'
        ? 'fused_result'
        : record.operation === 'nonfused'
          ? 'nonfused_result'
          : 'result'
  const [signal, setSignal] = useState(defaultName),
    [span, setSpan] = useState(8),
    [start, setStart] = useState(() =>
      Math.max(
        0,
        Math.min(
          wave.data.frames.length - 8,
          wave.data.frames.findIndex(
            (f) => f.time >= wave.sampleTimes[record.index],
          ) - 2,
        ),
      ),
    )
  const width = wave.data.signals.find((s) => s.name === signal)!.width,
    visible = wave.data.frames.slice(start, start + span),
    selectedTime = wave.sampleTimes[record.index],
    selectedValue = valueAt(wave, selectedTime, signal)
  const end = Math.max(0, wave.data.frames.length - span),
    clamp = (n: number) => Math.max(0, Math.min(end, n))
  return (
    <section aria-label="Recorded waveform inspector" className="datapath-wave">
      <h3>Recorded waveform subset</h3>
      <p>
        Actual VCD values, held until their next recorded transition. Reset and
        settled transactions alternate in the testbench. VCD ticks are
        observation indices; the {wave.data.timescale} header does not establish
        propagation delay, a hardware clock, or pipeline latency.
      </p>
      <label htmlFor="datapath-wave-signal">Waveform signal</label>
      <select
        id="datapath-wave-signal"
        value={signal}
        onChange={(e) => setSignal(e.target.value)}
      >
        {wave.data.signals.map((s) => (
          <option key={s.name} value={s.name}>
            {s.name} · {s.width} bits
          </option>
        ))}
      </select>
      <div className="datapath-controls">
        <button
          type="button"
          onClick={() => {
            const next = Math.max(2, Math.floor(span / 2))
            setSpan(next)
            setStart(
              Math.min(start, Math.max(0, wave.data.frames.length - next)),
            )
          }}
          disabled={span <= 2}
        >
          Zoom waveform in
        </button>
        <button
          type="button"
          onClick={() => {
            const next = Math.min(wave.data.frames.length, span * 2)
            setSpan(next)
            setStart(
              Math.min(start, Math.max(0, wave.data.frames.length - next)),
            )
          }}
          disabled={span >= wave.data.frames.length}
        >
          Zoom waveform out
        </button>
        <button
          type="button"
          onClick={() => setStart(clamp(start - 1))}
          disabled={start === 0}
        >
          Pan waveform left
        </button>
        <button
          type="button"
          onClick={() => setStart(clamp(start + 1))}
          disabled={start >= end}
        >
          Pan waveform right
        </button>
      </div>
      <label htmlFor="datapath-wave-pan">First visible waveform frame</label>
      <input
        id="datapath-wave-pan"
        type="range"
        min="0"
        max={end}
        value={Math.min(start, end)}
        onChange={(e) => setStart(Number(e.target.value))}
      />
      <svg
        viewBox="0 0 960 130"
        role="img"
        aria-label={`Recorded ${signal} waveform, frames ${start} to ${start + visible.length - 1}`}
      >
        {visible.map((frame, i) => {
          const x = 30 + (i * 900) / visible.length,
            w = 900 / visible.length,
            bits = frame.values[signal],
            v = /^[01]+$/.test(bits)
              ? '0x' + BigInt('0b' + bits).toString(16)
              : bits,
            nextBits = visible[i + 1]?.values[signal]
          return (
            <g key={frame.time}>
              <title>
                Observation {frame.time}: {bits}
              </title>
              {width === 1 && /^[01]$/.test(bits) ? (
                <path
                  d={`M${x} ${bits === '1' ? 25 : 70}h${w}${nextBits && /^[01]$/.test(nextBits) ? `V${nextBits === '1' ? 25 : 70}` : ''}`}
                  fill="none"
                />
              ) : (
                <>
                  <rect x={x} y="25" width={w} height="45" />
                  <text x={x + 3} y="53">
                    {w > 80 ? v : 'bus'}
                  </text>
                </>
              )}
              <text x={x} y="102">
                {frame.time}
              </text>
            </g>
          )
        })}
      </svg>
      <p>
        Selected example at observation {selectedTime}:{' '}
        <code
          tabIndex={0}
          className="rtl-binary"
          aria-label="Selected waveform binary"
        >
          {selectedValue ?? 'Unavailable'}
        </code>
      </p>
      <details>
        <summary>Exact visible waveform values</summary>
        <table>
          <thead>
            <tr>
              <th>Observation index</th>
              <th>Recorded bits</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((f) => (
              <tr key={f.time}>
                <td>{f.time}</td>
                <td>
                  <code tabIndex={0} className="rtl-binary">
                    {f.values[signal]}
                  </code>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
      <a href={`${import.meta.env.BASE_URL}${wave.download}`} download>
        Download raw {wave.family} VCD
      </a>
      <p className="text-small">
        Curated subset from {wave.tool}; raw file includes the full recorded
        signals. No browser RTL execution.
      </p>
    </section>
  )
}
