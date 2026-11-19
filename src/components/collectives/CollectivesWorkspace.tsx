import { useEffect, useRef, useState } from 'react'
import {
  assocText,
  cellText,
  collectives,
  errorSeries,
  layoutTree,
  roundCells,
  schedulesFor,
  wires,
  worlds,
  type Assoc,
} from '../../lib/collectives/data'

const repo = 'https://github.com/nivupai/fp-play/blob/main/'
const STEP_MS = 900

function prefersReducedMotion(): boolean {
  return (
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  )
}

function AssociationTree({ assoc, leaf }: { assoc: Assoc; leaf: string }) {
  const { nodes, edges, columns, depth } = layoutTree(assoc, leaf)
  const w = 60 * columns + 20,
    h = 56 * (depth + 1) + 20
  const px = (x: number) => 40 + 60 * x
  const py = (y: number) => 30 + 56 * y
  return (
    <div
      className="table-scroll"
      tabIndex={0}
      role="group"
      aria-label="Scrollable association tree"
    >
      <svg
        viewBox={`0 0 ${w} ${h}`}
        width={w}
        height={h}
        role="img"
        aria-label={`Association tree: ${assocText(assoc, leaf)}`}
      >
        {edges.map(([a, b]) => (
          <line
            key={`${a}-${b}`}
            x1={px(nodes[a].x)}
            y1={py(nodes[a].y)}
            x2={px(nodes[b].x)}
            y2={py(nodes[b].y)}
            stroke="currentColor"
          />
        ))}
        {nodes.map((n) => (
          <g key={n.id}>
            {n.leaf ? (
              <rect
                x={px(n.x) - 18}
                y={py(n.y) - 13}
                width={36}
                height={26}
                fill="var(--surface, white)"
                stroke="currentColor"
              />
            ) : (
              <circle
                cx={px(n.x)}
                cy={py(n.y)}
                r={14}
                fill="var(--surface, white)"
                stroke="currentColor"
              />
            )}
            <text
              x={px(n.x)}
              y={py(n.y) + 5}
              textAnchor="middle"
              fontSize="13"
              fill="currentColor"
            >
              {n.label}
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}

function ErrorChart({ wire }: { wire: string }) {
  const series = errorSeries(wire)
  const all = [...series.values()].flat()
  const ws = [...new Set(all.map((p) => p.world))].sort((a, b) => a - b)
  const positive = all.filter((p) => p.value > 0).map((p) => p.value)
  if (!positive.length) return <p>No nonzero errors recorded for {wire}.</p>
  const lo = Math.log10(Math.min(...positive)),
    hi = Math.log10(Math.max(...positive))
  const span = hi - lo || 1
  const x = (w: number) =>
    70 + (ws.indexOf(w) / Math.max(1, ws.length - 1)) * 460
  const y = (v: number) =>
    v > 0 ? 230 - ((Math.log10(v) - lo) / span) * 200 : 245
  const marks = ['circle', 'square', 'triangle', 'diamond', 'cross']
  return (
    <figure>
      <figcaption>
        Relative RMS error against the exact sum versus P ({wire} wire;
        logarithmic axis, labels rounded). Exact values are in the table.
      </figcaption>
      <div
        className="table-scroll"
        tabIndex={0}
        role="group"
        aria-label="Scrollable error chart"
      >
        <svg
          viewBox="0 0 600 280"
          width={600}
          height={280}
          role="img"
          aria-label={`Error versus P for the ${wire} wire; exact values in the following table`}
        >
          <path d="M60 20V240H560" fill="none" stroke="currentColor" />
          {ws.map((w) => (
            <text
              key={w}
              x={x(w)}
              y={262}
              textAnchor="middle"
              fill="currentColor"
            >
              P = {w}
            </text>
          ))}
          <text x={8} y={30} fill="currentColor" fontSize="11">
            {Math.max(...positive).toExponential(1)}
          </text>
          <text x={8} y={234} fill="currentColor" fontSize="11">
            {Math.min(...positive).toExponential(1)}
          </text>
          {[...series.entries()].map(([name, pts], i) => (
            <g key={name}>
              <polyline
                fill="none"
                stroke="currentColor"
                strokeDasharray={i === 0 ? undefined : `${2 + i * 2} 3`}
                points={pts.map((p) => `${x(p.world)},${y(p.value)}`).join(' ')}
              />
              {pts.map((p) => (
                <text
                  key={p.world}
                  x={x(p.world)}
                  y={y(p.value) + 4}
                  textAnchor="middle"
                  fontSize="12"
                  fill="currentColor"
                >
                  {['●', '■', '▲', '◆', '✚'][i % marks.length]}
                </text>
              ))}
            </g>
          ))}
        </svg>
      </div>
      <ul className="muted">
        {[...series.keys()].map((name, i) => (
          <li key={name}>
            {['●', '■', '▲', '◆', '✚'][i % marks.length]} {name}
            {i === 0 ? ' (solid line)' : ' (dashed line)'}
          </li>
        ))}
      </ul>
      <div
        className="table-scroll"
        tabIndex={0}
        role="group"
        aria-label="Scrollable error table"
      >
        <table className="collectives-table">
          <caption>
            Error against the exact sum ({wire} wire): relative RMS error,
            elements that differ from the exact result, and mean ulps, copied
            from the recorded tp-allreduce run.
          </caption>
          <thead>
            <tr>
              <th scope="col">Schedule</th>
              <th scope="col">P</th>
              <th scope="col">Relative RMS error</th>
              <th scope="col">Elements differing</th>
              <th scope="col">Mean ulps</th>
            </tr>
          </thead>
          <tbody>
            {collectives.errors.runs
              .filter((e) => e.wire === wire)
              .map((e) => (
                <tr key={`${e.algorithm}-${e.world}`}>
                  <th scope="row">{e.algorithm}</th>
                  <td>{e.world}</td>
                  <td>{e.relative_rms_error.toExponential(3)}</td>
                  <td>
                    {e.mismatched} of {collectives.errors.problem.elements}
                  </td>
                  <td>{e.mean_ulp.toFixed(3)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </figure>
  )
}

export function CollectivesWorkspace() {
  const [world, setWorld] = useState(4)
  const options = schedulesFor(world)
  const [algorithm, setAlgorithm] = useState('ring')
  const schedule = options.find((s) => s.algorithm === algorithm) ?? options[0]
  const [round, setRound] = useState(0)
  const [element, setElement] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [wire, setWire] = useState(wires.includes('bf16') ? 'bf16' : wires[0])
  const [treeWorld, setTreeWorld] = useState(4)
  const timer = useRef<number | null>(null)
  const last = schedule.rounds.length - 1
  const current = Math.min(round, Math.max(0, last))

  useEffect(() => {
    if (!playing) return
    timer.current = window.setInterval(
      () => {
        setRound((r) => {
          if (r >= last) {
            setPlaying(false)
            return r
          }
          return r + 1
        })
      },
      prefersReducedMotion() ? STEP_MS * 2 : STEP_MS,
    )
    return () => {
      if (timer.current !== null) window.clearInterval(timer.current)
    }
  }, [playing, last])

  const cells = schedule.rounds.map((r) => roundCells(schedule.world, r))
  const tree = collectives.shard_trees.find((t) => t.world === treeWorld)

  return (
    <section className="collectives" aria-labelledby="collectives-heading">
      <div className="section-heading">
        <p className="eyebrow">Runtime R1 / recorded schedules</p>
        <h2 id="collectives-heading">
          How each AllReduce schedule adds the same partial sums.
        </h2>
        <p>
          Every round, transfer and association below was recorded from the Rust
          runtime's executed simulator (
          <a href={`${repo}runtime/results/r6-collectives.md`}>
            r6-collectives
          </a>
          ). The browser replays the recorded data; it does not run a
          collective. Rounds are synchronous steps of the simulation, not
          hardware timing.
        </p>
      </div>

      <div className="control-grid">
        <label htmlFor="collectives-world">Ranks (P)</label>
        <select
          id="collectives-world"
          value={world}
          onChange={(e) => {
            setWorld(Number(e.target.value))
            setElement(0)
            setRound(0)
            setPlaying(false)
          }}
        >
          {worlds.map((w) => (
            <option key={w} value={w}>
              {w}
            </option>
          ))}
        </select>
        <label htmlFor="collectives-algorithm">Schedule</label>
        <select
          id="collectives-algorithm"
          value={schedule.algorithm}
          onChange={(e) => {
            setAlgorithm(e.target.value)
            setRound(0)
            setPlaying(false)
          }}
        >
          {options.map((s) => (
            <option key={s.algorithm} value={s.algorithm}>
              {s.algorithm}
            </option>
          ))}
        </select>
      </div>
      <p className="muted">{schedule.description}.</p>

      <h3>Schedule: rank × round</h3>
      {schedule.rounds.length === 0 ? (
        <p>One rank: nothing is sent.</p>
      ) : (
        <>
          <div
            className="button-row"
            role="group"
            aria-label="Round controls"
            onKeyDown={(e) => {
              if (e.key === 'ArrowRight') setRound(Math.min(last, current + 1))
              else if (e.key === 'ArrowLeft') setRound(Math.max(0, current - 1))
              else if (e.key === 'Home') setRound(0)
              else if (e.key === 'End') setRound(last)
              else return
              e.preventDefault()
            }}
          >
            <button
              type="button"
              onClick={() => setRound(0)}
              disabled={current === 0}
            >
              First round
            </button>
            <button
              type="button"
              onClick={() => setRound(Math.max(0, current - 1))}
              disabled={current === 0}
            >
              Previous round
            </button>
            <button
              type="button"
              onClick={() => setRound(Math.min(last, current + 1))}
              disabled={current === last}
            >
              Next round
            </button>
            <button
              type="button"
              onClick={() => setRound(last)}
              disabled={current === last}
            >
              Last round
            </button>
            <button
              type="button"
              aria-pressed={playing}
              onClick={() => {
                if (!playing && current === last) setRound(0)
                setPlaying(!playing)
              }}
            >
              {playing ? 'Stop' : 'Play rounds'}
            </button>
          </div>
          <p aria-live="polite" className="collectives-status">
            Round {current + 1} of {schedule.rounds.length}:{' '}
            {schedule.rounds[current].phase} (
            {schedule.rounds[current].kind === 'reduce'
              ? 'received values are added'
              : 'received values are copied'}
            ).
          </p>
          <div
            className="table-scroll"
            tabIndex={0}
            role="group"
            aria-label="Scrollable schedule table"
          >
            <table className="collectives-table">
              <caption>
                What each rank does in each round, over {schedule.elements}{' '}
                elements. The current round is marked “current”.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Rank</th>
                  {schedule.rounds.map((r, i) => (
                    <th
                      scope="col"
                      key={i}
                      aria-current={i === current ? 'step' : undefined}
                    >
                      Round {i + 1}
                      {i === current ? ' (current)' : ''}
                      <br />
                      <span className="muted">{r.phase}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: schedule.world }, (_, rank) => (
                  <tr key={rank}>
                    <th scope="row">Rank {rank}</th>
                    {cells.map((col, i) => (
                      <td
                        key={i}
                        className={
                          i === current ? 'collectives-current' : undefined
                        }
                      >
                        {cellText(col[rank])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <h3>Association of one output element</h3>
      <label htmlFor="collectives-element">Element</label>
      <select
        id="collectives-element"
        value={element}
        onChange={(e) => setElement(Number(e.target.value))}
      >
        {schedule.association.map((_, i) => (
          <option key={i} value={i}>
            {i}
          </option>
        ))}
      </select>
      <p>
        Element {element} on every rank is{' '}
        <code>{assocText(schedule.association[element])}</code>, where x
        <sub>r</sub> is rank r's partial sum and each “+” is one rounded
        addition in the accumulator format
        {schedule.ranks_agree
          ? '; every rank ends with this association.'
          : '.'}
      </p>
      <AssociationTree assoc={schedule.association[element]} leaf="x" />

      {tree && (
        <>
          <h3>The shard tree: the same tree at every P</h3>
          <label htmlFor="collectives-tree-world">Shard-tree ranks (P)</label>
          <select
            id="collectives-tree-world"
            value={treeWorld}
            onChange={(e) => setTreeWorld(Number(e.target.value))}
          >
            {collectives.shard_trees.map((t) => (
              <option key={t.world} value={t.world}>
                {t.world}
              </option>
            ))}
          </select>
          <p>
            {tree.shards} shards (s<sub>i</sub>) are combined by one fixed tree:{' '}
            <code>{assocText(tree.association, 's')}</code>. With P ={' '}
            {tree.world}, shard owners are{' '}
            {tree.owners.map((o, s) => `s${s} → rank ${o}`).join(', ')};{' '}
            {tree.rounds.length} recorded rounds. The tree, and therefore the
            bits, do not change with P.
          </p>
          <AssociationTree assoc={tree.association} leaf="s" />
        </>
      )}

      <h3>Error against the exact sum</h3>
      <label htmlFor="collectives-wire">Wire format</label>
      <select
        id="collectives-wire"
        value={wire}
        onChange={(e) => setWire(e.target.value)}
      >
        {wires.map((w) => (
          <option key={w} value={w}>
            {w}
          </option>
        ))}
      </select>
      <ErrorChart wire={wire} />
      <p className="muted">
        Problem: {collectives.errors.problem.tokens} tokens ×{' '}
        {collectives.errors.problem.hidden} hidden ×{' '}
        {collectives.errors.problem.outputs} outputs, BF16 inputs, FP32 local
        accumulation. {collectives.note}
      </p>
    </section>
  )
}
