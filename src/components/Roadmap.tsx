const stages = [
  {
    number: '01',
    title: 'Project foundation',
    detail:
      'Application shell, tests, documentation, and continuous integration.',
    status: 'Complete',
  },
  {
    number: '02',
    title: 'Binary16 numerical engine',
    detail:
      'Exact encoding, decoding, introspection, and ties-to-even rounding.',
    status: 'Complete',
  },
  {
    number: '03',
    title: 'Interactive FP16 Explorer',
    detail:
      'Bit manipulation, decimal quantization, numerical breakdown, and adjacent values.',
    status: 'Complete',
  },
  {
    number: '04',
    title: 'Precision microscope',
    detail:
      'Interactive number line, rounding cells, quantization error, and ULP spacing.',
    status: 'Complete',
  },
  {
    number: '05',
    title: 'Trace arithmetic',
    detail:
      'Exact FP16 addition/subtraction, alignment, normalization, rounding, and exception traces.',
    status: 'Complete',
  },
  {
    number: '06',
    title: 'Multiply & fuse',
    detail:
      'Exact FP16 multiplication, single-rounding FMA, and nonfused comparison with verified counterexamples.',
    status: 'Complete',
  },
  {
    number: '07',
    title: 'Public release preparation',
    detail:
      'Polished entry points, keyboard access, release documentation, and manual Pages readiness for v0.1.0.',
    status: 'Complete',
  },
  {
    number: '08',
    title: 'Mixed-precision lab',
    detail:
      'FP16/BF16/FP32 representation, exact arithmetic, mixed-precision accumulation, and reduction order.',
    status: 'Complete',
  },
  {
    number: '09',
    title: 'Attention precision',
    detail:
      'Bounded attention, stable/naive softmax, precision-stage traces, and reproducible error experiments.',
    status: 'Complete',
  },
  {
    number: '10',
    title: 'FP16 RTL arithmetic',
    detail:
      'Original combinational FP16 add/subtract, Verilator differential checks, waveforms, and recorded signal inspection.',
    status: 'Current milestone',
  },
]

export function Roadmap() {
  return (
    <section id="roadmap" aria-labelledby="roadmap-heading" className="roadmap">
      <h2 id="roadmap-heading">A path from bits to behavior</h2>
      <p className="muted">
        Built incrementally, with numerical correctness at every step.
      </p>
      <ol className="roadmap-grid">
        {stages.map((stage) => (
          <li key={stage.number} className="roadmap-card">
            <div className="roadmap-card-heading">
              <span aria-hidden="true" className="font-mono muted">
                {stage.number}
              </span>
              <span className={stage.number === '10' ? 'accent-text' : 'muted'}>
                {stage.status}
              </span>
            </div>
            <h3>{stage.title}</h3>
            <p className="muted text-small">{stage.detail}</p>
          </li>
        ))}
      </ol>
    </section>
  )
}
