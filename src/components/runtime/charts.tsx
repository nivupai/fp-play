import type { ReactNode } from 'react'

/**
 * Charts for the runtime workspaces. Each one is paired with an exact table
 * by its caller; the bar charts are also text themselves (every bar's label
 * and value are written out), so nothing is conveyed by length or color alone.
 */

export function ScrollTable({
  label,
  children,
}: {
  label: string
  children: ReactNode
}) {
  return (
    <div
      className="table-scroll"
      tabIndex={0}
      role="group"
      aria-label={`Scrollable ${label}`}
    >
      {children}
    </div>
  )
}

export interface Bar {
  label: string
  value: number
  text: string
}

/**
 * Horizontal bars. With `log`, lengths follow log10 of the value between
 * the smallest and largest positive value (a zero gets no bar and is
 * labeled as zero in its text).
 */
export function BarList({
  caption,
  bars,
  log = false,
}: {
  caption: string
  bars: Bar[]
  log?: boolean
}) {
  const positive = bars.map((b) => b.value).filter((v) => v > 0)
  const max = Math.max(0, ...bars.map((b) => b.value))
  const lo = positive.length ? Math.log10(Math.min(...positive)) : 0
  const hi = positive.length ? Math.log10(Math.max(...positive)) : 0
  const width = (v: number) => {
    if (v <= 0) return 0
    if (!log) return max > 0 ? (100 * v) / max : 0
    // Keep the smallest positive value visible: 4% … 100%.
    return hi === lo ? 100 : 4 + (96 * (Math.log10(v) - lo)) / (hi - lo)
  }
  return (
    <figure className="runtime-bars">
      <figcaption>
        {caption}
        {log ? ' Logarithmic bar lengths.' : ' Linear bar lengths.'}
      </figcaption>
      <ul>
        {bars.map((b) => (
          <li key={b.label}>
            <span className="runtime-bar-label">{b.label}</span>
            <span className="runtime-bar-track" aria-hidden="true">
              <span
                className="runtime-bar"
                style={{ width: `${width(b.value)}%` }}
              />
            </span>
            <span className="runtime-bar-value">{b.text}</span>
          </li>
        ))}
      </ul>
    </figure>
  )
}

export interface Segment {
  key: string
  label: string
  share: number
}

const PATTERNS = ['solid', 'hatch', 'dots', 'cross', 'outline']

/** One row of stacked shares; segments are told apart by pattern and by the legend text. */
export function StackedShares({
  caption,
  rows,
  legend,
}: {
  caption: string
  rows: { label: string; segments: Segment[] }[]
  legend: { key: string; label: string }[]
}) {
  const pattern = (key: string) =>
    PATTERNS[legend.findIndex((l) => l.key === key) % PATTERNS.length]
  return (
    <figure className="runtime-stacked">
      <figcaption>{caption}</figcaption>
      <ul className="runtime-legend">
        {legend.map((l) => (
          <li key={l.key}>
            <span
              className={`runtime-swatch runtime-pattern-${pattern(l.key)}`}
              aria-hidden="true"
            />
            {l.label}
          </li>
        ))}
      </ul>
      <ul>
        {rows.map((r) => (
          <li key={r.label}>
            <span className="runtime-bar-label">{r.label}</span>
            <span className="runtime-stack" aria-hidden="true">
              {r.segments.map((s) => (
                <span
                  key={s.key}
                  className={`runtime-segment runtime-pattern-${pattern(s.key)}`}
                  style={{ width: `${100 * s.share}%` }}
                />
              ))}
            </span>
            <span className="runtime-bar-value">
              {r.segments
                .filter((s) => s.share > 0)
                .map((s) => `${s.label} ${(100 * s.share).toFixed(1)}%`)
                .join(' · ')}
            </span>
          </li>
        ))}
      </ul>
    </figure>
  )
}

export interface Series {
  name: string
  points: { x: number; y: number }[]
}

const MARKS = ['●', '■', '▲', '◆']

/** A line chart with a logarithmic y axis; series differ by marker and dash. */
export function LogLineChart({
  caption,
  ariaLabel,
  series,
  xLabel,
  yLabel,
}: {
  caption: string
  ariaLabel: string
  series: Series[]
  xLabel: string
  yLabel: string
}) {
  const all = series.flatMap((s) => s.points)
  const ys = all.map((p) => p.y).filter((y) => y > 0)
  const xs = all.map((p) => p.x)
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)]
  const [lo, hi] = [Math.log10(Math.min(...ys)), Math.log10(Math.max(...ys))]
  const X = (x: number) => 70 + ((x - x0) / (x1 - x0 || 1)) * 500
  const Y = (y: number) => 230 - ((Math.log10(y) - lo) / (hi - lo || 1)) * 200
  const step = Math.max(1, Math.floor(all.length / series.length / 12))
  return (
    <figure>
      <figcaption>{caption}</figcaption>
      <ScrollTable label="chart">
        <svg
          viewBox="0 0 620 290"
          width={620}
          height={290}
          role="img"
          aria-label={ariaLabel}
        >
          <path d="M60 20V240H580" fill="none" stroke="currentColor" />
          <text x={8} y={30} fontSize="11" fill="currentColor">
            {Math.max(...ys).toExponential(1)}
          </text>
          <text x={8} y={234} fontSize="11" fill="currentColor">
            {Math.min(...ys).toExponential(1)}
          </text>
          <text x={70} y={262} fontSize="12" fill="currentColor">
            {x0}
          </text>
          <text
            x={570}
            y={262}
            fontSize="12"
            textAnchor="end"
            fill="currentColor"
          >
            {x1}
          </text>
          <text
            x={320}
            y={282}
            fontSize="12"
            textAnchor="middle"
            fill="currentColor"
          >
            {xLabel}
          </text>
          <text x={8} y={14} fontSize="12" fill="currentColor">
            {yLabel} (log scale)
          </text>
          {series.map((s, i) => (
            <g key={s.name}>
              <polyline
                fill="none"
                stroke="currentColor"
                strokeDasharray={i === 0 ? undefined : `${2 + 2 * i} 3`}
                points={s.points
                  .filter((p) => p.y > 0)
                  .map((p) => `${X(p.x)},${Y(p.y)}`)
                  .join(' ')}
              />
              {s.points
                .filter((p, j) => p.y > 0 && j % step === 0)
                .map((p) => (
                  <text
                    key={p.x}
                    x={X(p.x)}
                    y={Y(p.y) + 4}
                    textAnchor="middle"
                    fontSize="10"
                    fill="currentColor"
                  >
                    {MARKS[i % MARKS.length]}
                  </text>
                ))}
            </g>
          ))}
        </svg>
      </ScrollTable>
      {series.length > 1 && (
        <ul className="muted">
          {series.map((s, i) => (
            <li key={s.name}>
              {MARKS[i % MARKS.length]} {s.name}
              {i === 0 ? ' (solid line)' : ' (dashed line)'}
            </li>
          ))}
        </ul>
      )}
    </figure>
  )
}
