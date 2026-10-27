import { useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { loadCatalog } from '../../lib/datapath/catalog'
import {
  STAGES,
  expectedFields,
  signalBinary,
  flagMask,
} from '../../lib/datapath/artifacts'
import type { Operation, Stage } from '../../lib/datapath/artifacts'
import {
  compareExample,
  exactReference,
  exampleLabel,
  softwareTrace,
} from '../../lib/datapath/reference'
import {
  decodeFP16,
  formatFP16Bits,
  fp16BitsToHex,
  fp16BitsToFloat,
} from '../../lib/numerics/fp16'
import { formatExactFP16 } from '../../lib/explorer/presentation'
import { DatapathDiagram } from './DatapathDiagram'
import { GRSExplorer } from './GRSExplorer'
import { DatapathSynthesis } from './DatapathSynthesis'
import { WaveformInspector } from './WaveformInspector'
function Inspect({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false)
  return (
    <details onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>{label}</summary>
      {open && children}
    </details>
  )
}
const operations: { id: Operation; label: string }[] = [
  { id: 'add', label: 'Addition' },
  { id: 'subtract', label: 'Subtraction' },
  { id: 'multiply', label: 'Multiplication' },
  { id: 'fma', label: 'True FMA (single rounding)' },
  { id: 'nonfused', label: 'Nonfused multiply-add (two roundings)' },
]
const stageKinds: Record<Stage, string[]> = {
  unpack: [
    'decode',
    'decode-multiply',
    'decode-fma',
    'special',
    'multiply-special',
  ],
  exponent: ['compare', 'product-sign', 'product-exponent'],
  align: ['align', 'align-product'],
  arithmetic: ['multiply', 'arithmetic'],
  normalize: ['normalize'],
  grs: ['grs'],
  round: ['round'],
  pack: ['encode'],
}
const flags = (f: Parameters<typeof flagMask>[0]) =>
  Object.entries(f)
    .map(([k, v]) => `${k}: ${v}`)
    .join('; ')
export function HardwareDatapathExplorer({
  inputs,
}: {
  inputs?: Parameters<typeof loadCatalog>[0]
}) {
  const loaded = useMemo(() => loadCatalog(inputs), [inputs]),
    [operation, setOperation] = useState<Operation>('add'),
    [exampleId, setExampleId] = useState(''),
    [selected, setSelected] = useState(0)
  if (!loaded.ok)
    return (
      <section className="panel" aria-label="Hardware Datapath Explorer">
        <h2>Hardware Datapath Explorer</h2>
        <p role="alert">
          Recorded hardware data could not be loaded: {loaded.error}. No
          fabricated example is substituted.
        </p>
      </section>
    )
  const { catalog } = loaded,
    options = catalog.examples.filter((r) => r.operation === operation),
    record = options.find((r) => r.id === exampleId) ?? options[0]
  const changeOperation = (value: Operation) => {
    setOperation(value)
    setExampleId('')
    setSelected(0)
  }
  if (!record)
    return (
      <section className="panel" aria-label="Hardware Datapath Explorer">
        <h2>Hardware Datapath Explorer</h2>
        <p>No recorded examples are available for this operation.</p>
        <label htmlFor="datapath-operation-empty">Datapath operation</label>
        <select
          id="datapath-operation-empty"
          value={operation}
          onChange={(e) => changeOperation(e.target.value as Operation)}
        >
          {operations.map((op) => (
            <option key={op.id} value={op.id}>
              {op.label}
            </option>
          ))}
        </select>
      </section>
    )
  const comparison = compareExample(record),
    phaseIndex = Math.min(record.phases.length - 1, Math.floor(selected / 8)),
    phase = record.phases[phaseIndex],
    stageIndex = selected % 8,
    stage = STAGES[stageIndex],
    reference = comparison.phases[phaseIndex].software,
    trace = softwareTrace(reference),
    stageTrace = trace.filter((s) => stageKinds[stage.id].includes(s.kind)),
    fields = expectedFields(phase, stage.id),
    wave = catalog.waves.find((w) => w.family === record.family)
  const extras =
    stage.id === 'unpack' && phase.kind !== 'add' && phase.operands.length === 3
      ? ['significandC']
      : stage.id === 'normalize' && phase.kind !== 'add'
        ? ['normalizedMagnitude']
        : stage.id === 'exponent' && phase.kind !== 'add'
          ? ['effectiveExponentA', 'effectiveExponentB']
          : []
  return (
    <section
      className="panel datapath-explorer"
      id="datapath-results"
      aria-labelledby="datapath-heading"
    >
      <div className="section-heading">
        <p className="eyebrow">
          Recorded hardware · exact mathematics · controlled synthesis
        </p>
        <h2 id="datapath-heading">Hardware Datapath Explorer</h2>
        <p>
          Follow a logical floating-point datapath using genuine recorded
          examples. Your browser runs only the labeled numerical reference, not
          RTL simulation or synthesis.
        </p>
      </div>
      <div className="datapath-selection">
        <div>
          <label htmlFor="datapath-operation">Datapath operation</label>
          <select
            id="datapath-operation"
            value={operation}
            onChange={(e) => changeOperation(e.target.value as Operation)}
          >
            {operations.map((op) => (
              <option key={op.id} value={op.id}>
                {op.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="datapath-example">Verified hardware example</label>
          <select
            id="datapath-example"
            value={record.id}
            onChange={(e) => {
              setExampleId(e.target.value)
              setSelected(0)
            }}
          >
            {options.map((r) => (
              <option key={r.id} value={r.id}>
                {exampleLabel(r)}
              </option>
            ))}
          </select>
        </div>
      </div>
      {record.operation === 'multiply' && (
        <p>
          Standalone multiplication uses A and B only. Shared triple titles
          describe the original verification scenario; C is not an operand here.
        </p>
      )}
      {record.operation === 'fma' && (
        <p className="datapath-emphasis">
          True FMA: exact A × B + C, with one final rounding. No FP16
          intermediate product rounding or product overflow/underflow flags.
        </p>
      )}
      {record.operation === 'nonfused' && (
        <p className="datapath-emphasis">
          Nonfused: rounded FP16 product → FP16 adder → second rounding. Final
          flags union both operations. Detailed adder signals are unavailable in
          this artifact; they are not reconstructed as RTL observations.
        </p>
      )}
      <div className="datapath-operands">
        {record.operands.map((bits, i) => (
          <div key={i}>
            <h3>Recorded operand {['A', 'B', 'C'][i]}</h3>
            <code className="rtl-binary" tabIndex={0}>
              {fp16BitsToHex(bits)} · {formatFP16Bits(bits)}
            </code>
            <p>
              Software decoding: {formatExactFP16(fp16BitsToFloat(bits))} ·{' '}
              {decodeFP16(bits).classification}
            </p>
          </div>
        ))}
      </div>
      <p aria-label="Datapath agreement" aria-live="polite">
        {comparison.agrees
          ? 'Agreement: raw result bits, every exception flag, and all available checked signals match.'
          : 'DISAGREEMENT: current software differs from recorded hardware bits, flags, or internal signals. Inspect phase results and provenance.'}
      </p>
      <div className="datapath-results">
        <section aria-label="Exact software reference">
          <h3>Exact numerical reference · software</h3>
          {comparison.phases.map(({ software, recorded }, i) => (
            <div key={i}>
              <h4>{recorded.label}</h4>
              <code className="rtl-binary" tabIndex={0}>
                {exactReference(software)}
              </code>
              <p>
                Software rounded encoding:{' '}
                <code>{fp16BitsToHex(software.bits)}</code>
              </p>
            </div>
          ))}
          {operation === 'nonfused' && (
            <p>
              The second exact dyadic uses the already-rounded product. It is
              not the exact unrounded A × B + C expression.
            </p>
          )}
        </section>
        <section aria-label="Recorded RTL output">
          <h3>Recorded RTL output</h3>
          <p aria-label="Datapath RTL result">
            <strong>{fp16BitsToHex(record.bits)}</strong> ·{' '}
            {formatExactFP16(fp16BitsToFloat(record.bits))}
          </p>
          <code className="rtl-binary" tabIndex={0}>
            {formatFP16Bits(record.bits)}
          </code>
          <p aria-label="Datapath RTL flags">{flags(record.flags)}</p>
          <p>
            Raw-bit agreement: {String(comparison.bitsAgree)}; exception-flag
            agreement: {String(comparison.flagsAgree)}.
          </p>
          {comparison.phases.map(
            ({ software, recorded, bitsAgree, flagsAgree }, i) => (
              <p key={i}>
                {recorded.label}: observed {fp16BitsToHex(recorded.bits)} /
                software {fp16BitsToHex(software.bits)}; bits match{' '}
                {String(bitsAgree)}; flags match {String(flagsAgree)}. Recorded
                flags: {flags(recorded.flags)}.
              </p>
            ),
          )}
        </section>
      </div>
      <DatapathDiagram
        record={record}
        selected={selected}
        onSelect={setSelected}
      />
      <div
        className="datapath-controls"
        tabIndex={0}
        role="group"
        aria-label="Datapath stage navigation"
        onKeyDown={(e) => {
          if (e.target !== e.currentTarget) return
          const max = record.phases.length * 8 - 1,
            next =
              e.key === 'ArrowRight'
                ? Math.min(max, selected + 1)
                : e.key === 'ArrowLeft'
                  ? Math.max(0, selected - 1)
                  : e.key === 'Home'
                    ? 0
                    : e.key === 'End'
                      ? max
                      : null
          if (next !== null) {
            e.preventDefault()
            setSelected(next)
          }
        }}
      >
        <button
          type="button"
          disabled={selected === 0}
          onClick={() => setSelected((s) => s - 1)}
        >
          Previous datapath stage
        </button>
        <button
          type="button"
          disabled={selected === record.phases.length * 8 - 1}
          onClick={() => setSelected((s) => s + 1)}
        >
          Next datapath stage
        </button>
        <span aria-live="polite">
          {selected + 1} / {record.phases.length * 8} · {phase.label} ·{' '}
          {stage.title}
        </span>
        <p className="text-small">
          Focus this group for Left/Right/Home/End. Arrow keys inside a long
          code value scroll that value.
        </p>
      </div>
      <section aria-label="Selected logical stage">
        <h3>
          {phase.label} · {stage.title}
        </h3>
        <p>{stage.description}</p>
        {phase.bypassed === true && !['unpack', 'pack'].includes(stage.id) && (
          <p className="datapath-emphasis">
            Recorded special-value bypass: these finite-stage defaults are not
            executed rounding/arithmetic.
          </p>
        )}
        {phase.bypassed === null && (
          <p>
            Recorded finite-path/bypass status is unavailable for this phase.
          </p>
        )}
        <div className="datapath-results">
          <section aria-label="Observed stage signals">
            <h4>Observed RTL signals</h4>
            {stage.id === 'pack' ? (
              <>
                <p>
                  Recorded output: {fp16BitsToHex(phase.bits)} ·{' '}
                  {formatFP16Bits(phase.bits)}
                </p>
                <p>{flags(phase.flags)}</p>
              </>
            ) : (
              <dl className="mixed-data">
                {[...fields, ...extras].map((name) => {
                  const signal = phase.signals[name]
                  return (
                    <div key={name}>
                      <dt>{name}</dt>
                      <dd>
                        {signal ? (
                          <>
                            <span>
                              Recorded {signal.width}-bit{' '}
                              {signal.signed ? 'signed ' : ''}signal:{' '}
                              {String(signal.value)}
                            </span>
                            <code
                              aria-label={`Recorded ${name} bits`}
                              className="rtl-binary"
                              tabIndex={0}
                            >
                              {signalBinary(signal)}
                            </code>
                          </>
                        ) : (
                          <span>
                            Unavailable · not captured in this RTL record.
                          </span>
                        )}
                      </dd>
                    </div>
                  )
                })}
              </dl>
            )}
          </section>
          <section aria-label="Software stage explanation">
            <h4>Software trace explanation · not hardware observations</h4>
            {stageTrace.length ? (
              stageTrace.map((s, i) => (
                <div key={i}>
                  <h5>{s.title}</h5>
                  <p>{s.explanation}</p>
                  <dl className="mixed-data">
                    {s.rows.map(([name, value], i) => (
                      <div key={i}>
                        <dt>{name}</dt>
                        <dd>
                          <code className="rtl-binary" tabIndex={0}>
                            {value}
                          </code>
                        </dd>
                      </div>
                    ))}
                  </dl>
                </div>
              ))
            ) : (
              <p>
                The software engine has no separate trace stage for this
                conceptual block. No artificial stage values are supplied.
              </p>
            )}
          </section>
        </div>
      </section>
      <p>
        Available internal signal checks:{' '}
        {comparison.phases.reduce((sum, p) => sum + p.signalChecks, 0)}. Missing
        fields are not replaced by software.{' '}
        {comparison.phases.flatMap((p) => p.signalMismatches).join('; ')}
      </p>
      <GRSExplorer
        key={`${record.id}:${phaseIndex}`}
        phase={phase}
        software={reference}
      />
      <Inspect label="Full software computation trace">
        <ol>
          {trace.map((s, i) => (
            <li key={i}>
              <h4>{s.title}</h4>
              <p>{s.explanation}</p>
              <dl className="mixed-data">
                {s.rows.map(([k, v], j) => (
                  <div key={j}>
                    <dt>{k}</dt>
                    <dd>
                      <code tabIndex={0} className="rtl-binary">
                        {v}
                      </code>
                    </dd>
                  </div>
                ))}
              </dl>
            </li>
          ))}
        </ol>
      </Inspect>
      <Inspect label="Recorded RTL provenance and source hashes">
        <p>
          Original artifact title: {record.name}. Artifact:{' '}
          <code>{record.provenance.artifact}</code>, sample {record.index}.
          Simulator: {record.provenance.tool}. Full verified corpus:{' '}
          {record.provenance.cases.toLocaleString('en-US')}; seed{' '}
          {record.provenance.seed}. These hashes are stored provenance;
          repository tests check actual files. The browser does not rehash HDL
          files.
        </p>
        <dl className="mixed-data">
          {Object.entries(record.provenance.sourceHashes).map(([p, h]) => (
            <div key={p}>
              <dt>{p}</dt>
              <dd>
                <code tabIndex={0} className="rtl-binary">
                  {h}
                </code>
              </dd>
            </div>
          ))}
        </dl>
      </Inspect>
      <DatapathSynthesis
        record={record}
        context={catalog.synthesis}
        error={catalog.synthesisError}
      />
      <Inspect label="Inspect recorded waveforms">
        {wave ? (
          <WaveformInspector key={record.id} wave={wave} record={record} />
        ) : (
          <p>
            Waveforms unavailable:{' '}
            {catalog.waveError ?? 'No correlated waveform for this record'}.
          </p>
        )}
      </Inspect>
    </section>
  )
}
