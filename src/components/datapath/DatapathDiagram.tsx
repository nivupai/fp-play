import { STAGES } from '../../lib/datapath/artifacts'
import type { RecordedExample } from '../../lib/datapath/artifacts'
export function DatapathDiagram({
  record,
  selected,
  onSelect,
}: {
  record: RecordedExample
  selected: number
  onSelect: (value: number) => void
}) {
  return (
    <div
      className="datapath-diagram"
      role="region"
      aria-label="Scrollable logical datapath diagram"
      tabIndex={0}
    >
      <svg
        viewBox={`0 0 1120 ${record.phases.length * 110 + 25}`}
        role="img"
        aria-label={`Logical block diagram for ${record.operation}; use the stage buttons below to select a block`}
      >
        <title>
          Conceptual datapath, not physical placement or an extracted gate-level
          topology
        </title>
        {record.phases.map((phase, p) => (
          <g key={p}>
            <text x="12" y={p * 110 + 20}>
              {phase.label}
            </text>
            {STAGES.map((stage, i) => {
              const x = i * 139 + 4,
                y = p * 110 + 35,
                active = selected === p * 8 + i
              return (
                <g
                  key={stage.id}
                  className={
                    active ? 'datapath-svg-active' : 'datapath-svg-block'
                  }
                >
                  <rect x={x} y={y} width="128" height="48" rx="6" />
                  <text x={x + 6} y={y + 20}>
                    {i + 1}.{' '}
                    {stage.title.length > 20
                      ? stage.title.split(' / ')[0]
                      : stage.title}
                  </text>
                  <text x={x + 6} y={y + 37}>
                    {phase.bypassed && !['unpack', 'pack'].includes(stage.id)
                      ? 'Bypassed'
                      : stage.id === 'round'
                        ? record.operation === 'fma'
                          ? 'Single final round'
                          : record.operation === 'nonfused'
                            ? `Round ${p + 1} of 2`
                            : 'RNE'
                        : phase.bypassed
                          ? 'Bypassed'
                          : 'Logical function'}
                  </text>
                  {i < 7 && (
                    <path
                      d={`M${x + 130} ${y + 24}h7m-4-3 4 3-4 3`}
                      fill="none"
                    />
                  )}
                </g>
              )
            })}
          </g>
        ))}
      </svg>
      {record.operation === 'nonfused' && (
        <p>
          First row’s packed, rounded product feeds the second row’s adder with
          the original C operand.
        </p>
      )}
      <p className="text-small">
        Logical block diagram. Blocks are explanatory functions, not physical
        placement, extracted gates, clock cycles, or measured stage delays.
      </p>
      <ol className="datapath-stage-buttons">
        {record.phases.flatMap((_, p) =>
          STAGES.map((stage, i) => (
            <li key={`${p}:${stage.id}`}>
              <button
                type="button"
                aria-current={selected === p * 8 + i ? 'step' : undefined}
                onClick={() => onSelect(p * 8 + i)}
              >
                {record.phases.length > 1
                  ? `${p === 0 ? 'Multiply' : 'Add'} · `
                  : ''}
                {i + 1}. {stage.title}
              </button>
            </li>
          )),
        )}
      </ol>
    </div>
  )
}
