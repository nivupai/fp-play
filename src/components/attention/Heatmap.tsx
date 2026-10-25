import { formatNumber } from '../../lib/explorer/presentation'
import { floatHex, floatBinary, decodeFloat } from '../../lib/numerics/formats'
import type { FloatValue } from '../../lib/numerics/formats'
import type { HeatValues } from '../../lib/attention/presentation'
/** Color intensity supplements full text; same-quantity comparisons share a symmetric scale. */
export function Heatmap({
  title,
  values,
  encodings,
  scale,
  onSelect,
  selected,
}: {
  title: string
  values: HeatValues
  encodings?: readonly (readonly FloatValue[])[]
  scale: number
  onSelect?: (row: number, column: number) => void
  selected?: [number, number]
}) {
  return (
    <div className="attention-map">
      <div
        className="mixed-scroll"
        role="group"
        tabIndex={0}
        aria-label={`${title} scroll area`}
      >
        <table className="attention-table">
          <caption>{title}</caption>
          <thead>
            <tr>
              <th scope="col">Row / column</th>
              {values[0].map((_, j) => (
                <th scope="col" key={j}>
                  {j + 1}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {values.map((row, i) => (
              <tr key={i}>
                <th scope="row">{i + 1}</th>
                {row.map((value, j) => {
                  const finite = value !== null && Number.isFinite(value),
                    text = value === null ? 'Undefined' : formatNumber(value)
                  const encoding = encodings?.[i][j],
                    d = encoding && decodeFloat(encoding.format, encoding.bits)
                  return (
                    <td
                      key={j}
                      data-sign={
                        finite
                          ? value < 0
                            ? 'negative'
                            : 'positive'
                          : 'special'
                      }
                      style={{
                        backgroundImage: finite
                          ? `linear-gradient(rgb(127 127 127 / ${0.03 + (Math.abs(value) / scale) * 0.12}), rgb(127 127 127 / ${0.03 + (Math.abs(value) / scale) * 0.12}))`
                          : undefined,
                      }}
                    >
                      {onSelect ? (
                        <button
                          aria-label={`${title} row ${i + 1} column ${j + 1}: ${text}`}
                          aria-pressed={
                            selected?.[0] === i && selected?.[1] === j
                          }
                          onClick={() => onSelect(i, j)}
                        >
                          {text}
                        </button>
                      ) : (
                        <span>{text}</span>
                      )}
                      {encoding && (
                        <small
                          title={`${encoding.format} ${floatBinary(encoding.format, encoding.bits)}; ${d!.classification}`}
                        >
                          {encoding.format}{' '}
                          {floatHex(encoding.format, encoding.bits)}
                        </small>
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted text-small">
        Shared symmetric scale: −{formatNumber(scale)}…{formatNumber(scale)}.
        Tint shows magnitude; sign and nonfinite states are written explicitly.
        Decimal labels are shortest binary64 displays.
      </p>
    </div>
  )
}
