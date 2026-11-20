import { version } from '../../package.json'
import {
  fusedComparison,
  numericalExamples,
  formatMetric,
} from '../lib/showcase/evidence'
import { fp16BitsToHex } from '../lib/numerics/fp16'
const repository = 'https://github.com/nivupai/fp-play'
const experiences = [
  {
    id: 'explorer',
    number: '01',
    title: 'Bit Explorer',
    detail:
      'Flip sixteen bits. Decode signs, exponents, subnormals, and special values.',
    caption: 'Start with representation',
  },
  {
    id: 'microscope',
    number: '02',
    title: 'Precision Microscope',
    detail:
      'Zoom into the number line. See representable values, rounding cells, and changing ULP spacing.',
    caption: 'Explore the gaps',
  },
  {
    id: 'arithmetic',
    number: '03',
    title: 'Arithmetic Pipeline',
    detail:
      'Follow exact addition, subtraction, and multiplication through alignment, normalization, and rounding.',
    caption: 'Trace a calculation',
  },
  {
    id: 'fma',
    number: '04',
    title: 'FMA Comparison',
    detail:
      'Compare one rounding with two. Discover lost residuals and avoided intermediate overflow.',
    caption: 'Try a verified counterexample',
  },
]

export function Landing() {
  const measured = fusedComparison()
  return (
    <section className="landing" aria-labelledby="landing-heading">
      <div className="hero">
        <div>
          <p className="eyebrow">
            FLOATING-POINT ARITHMETIC / AN OPEN LABORATORY
          </p>
          <h1 id="landing-heading">
            FP Play<span className="hero-dot">.</span>
          </h1>
          <p className="hero-tagline">Make precision visible.</p>
          <nav className="landing-parts" aria-label="FP Play in two parts">
            <a className="landing-part" href="#explorer">
              <span className="eyebrow">PART 1 / ONE OPERATION</span>
              <strong>Bits, rounding, FMA and verified RTL</strong>
              <span className="text-small">
                Exact FP16, BF16 and FP32 arithmetic, mixed precision,
                attention, and the hardware that computes it.
              </span>
            </a>
            <div className="landing-part landing-part-runtime">
              <span className="eyebrow">PART 2 / MANY DEVICES</span>
              <strong>The inference runtime</strong>
              <span className="text-small">
                A deterministic Rust runtime: what sharding, batching and
                precision do to a model’s numbers, from recorded runs.
              </span>
              <span className="landing-part-links">
                <a href="#collectives">Collectives</a>
                <a href="#serving">Serving</a>
                <a href="#decode">Decode</a>
                <a href="#precision">Precision</a>
                <a href="#dataflow">Dataflow</a>
              </span>
            </div>
          </nav>
          <p className="hero-description">
            A hands-on laboratory for understanding how bits become numbers, and
            how representation and accumulation precision shape a result. Follow
            exact software calculations into verified RTL and measured synthesis
            experiments, then across many devices in the runtime.
          </p>
          <div className="landing-links">
            <a className="primary-button" href="#explorer">
              Explore the bits <span aria-hidden="true">↗</span>
            </a>
            <a href={repository}>
              GitHub repository <span aria-hidden="true">↗</span>
            </a>
            <a href={`${repository}/blob/main/src/lib/numerics/README.md`}>
              Technical documentation <span aria-hidden="true">↗</span>
            </a>
          </div>
        </div>
        <aside className="landing-note" aria-label="About this release">
          <p className="release-badge">
            v{version} <span>Release candidate · not published</span>
          </p>
          <p>
            Built for curious engineers, students, and researchers in numerical
            computing and AI hardware.
          </p>
          <p className="muted text-small">
            Exact arithmetic models, independently checked results, and
            inspectable hardware evidence. Public standards and open-source
            code.
          </p>
        </aside>
      </div>
      <div
        className="showcase-proof-strip"
        aria-label="Selected verification evidence"
      >
        <p>
          <strong>65,536</strong>
          <span>FP16 encodings exhaustively checked</span>
        </p>
        <p>
          <strong>536,888</strong>
          <span>recorded independent SoftFloat operations</span>
        </p>
        <p>
          <strong>7</strong>
          <span>controlled Nangate45 experiments</span>
        </p>
      </div>
      <div className="landing-section-heading">
        <h2 id="experiences-heading">Four ways to see the same sixteen bits</h2>
        <p className="muted text-small">
          Start anywhere. Each experiment explains its mathematical limits.
        </p>
      </div>
      <nav className="experience-grid" aria-labelledby="experiences-heading">
        {experiences.map((experience) => (
          <a
            className="experience-card"
            href={`#${experience.id}`}
            onClick={() => {
              // A repeated hash link still opens the FMA example after other edits.
              if (experience.id === 'fma' && window.location.hash === '#fma')
                window.dispatchEvent(new Event('hashchange'))
            }}
            key={experience.id}
          >
            <span className="eyebrow">{experience.number} / INTERACTIVE</span>
            <h3 id={`experience-${experience.id}`}>{experience.title}</h3>
            <p>{experience.detail}</p>
            <span className="experience-caption">
              {experience.caption} <span aria-hidden="true">↗</span>
            </span>
          </a>
        ))}
      </nav>
      <nav className="landing-links" aria-label="Multi-precision experiments">
        <a href="#formats">Compare FP16, BF16, and FP32 →</a>
        <a href="#dot-product">Dot Product Precision Lab →</a>
        <a href="#attention">Attention Precision Lab →</a>
      </nav>
      <nav className="landing-links" aria-label="Runtime (Part 2)">
        <a href="#collectives">AllReduce schedules, round by round →</a>
        <a href="#serving">Serving simulator →</a>
        <a href="#decode">Does sharding change the tokens? →</a>
        <a href="#precision">Per-layer precision and a trained model →</a>
        <a href="#dataflow">A layer on a toy accelerator →</a>
        <a href={`${repository}/blob/main/docs/RUNTIME_CASE_STUDY.md`}>
          Runtime case study →
        </a>
      </nav>
      <nav className="landing-links" aria-label="Hardware and research">
        <a href="#research">Measured hardware showcase →</a>
        <a href="#datapath">RTL datapaths and waveforms →</a>
        <a href={`${repository}/blob/main/docs/CASE_STUDY.md`}>
          Technical case study →
        </a>
        <a href={`${repository}/blob/main/docs/REPRODUCIBILITY.md`}>
          Reproduce the results →
        </a>
      </nav>
      <div className="showcase-examples" aria-label="Verified examples">
        <div>
          <p className="eyebrow">A small input, a visible rounding decision</p>
          <p>
            <code>
              {numericalExamples.conversion.original} →{' '}
              {numericalExamples.conversion.value}
            </code>{' '}
            in FP16 ·{' '}
            <code>{fp16BitsToHex(numericalExamples.conversion.bits)}</code>
          </p>
          <a href="#microscope">Explore the spacing near one →</a>
        </div>
        <div>
          <p className="eyebrow">One rounding preserves a lost residual</p>
          <p>
            <code>(1 + 2⁻¹⁰)² − (1 + 2⁻⁹)</code>: fused{' '}
            <code>
              2⁻²⁰ ({fp16BitsToHex(numericalExamples.fma.fused.bits)})
            </code>
            , nonfused{' '}
            <code>
              0 ({fp16BitsToHex(numericalExamples.fma.nonfused.bits)})
            </code>
            .
          </p>
          <a
            href="#fma"
            onClick={() => {
              if (window.location.hash === '#fma')
                window.dispatchEvent(new Event('hashchange'))
            }}
          >
            Trace this verified FMA example →
          </a>
        </div>
      </div>
      <p className="showcase-hardware-preview">
        Measured baseline FMA:{' '}
        <strong>{formatMetric(measured!.fused.area)} µm²</strong> /{' '}
        <strong>{formatMetric(measured!.fused.delay!)} ns</strong>, versus
        nonfused <strong>{formatMetric(measured!.nonfused.area)} µm²</strong> /{' '}
        <strong>{formatMetric(measured!.nonfused.delay!)} ns</strong>. One
        Nangate45 experiment; cell area and constrained timing estimates, not
        physical chip measurements.
      </p>
      <p className="showcase-architecture" aria-label="Architecture overview">
        Bits and formats <span aria-hidden="true">→</span> exact BigInt
        arithmetic <span aria-hidden="true">→</span> verified FP16 SystemVerilog{' '}
        <span aria-hidden="true">→</span> mapped cells and timing
      </p>
      <p className="landing-limits text-small muted">
        Educational floating-point models with nearest-even rounding. No full
        IEEE 754 conformance claim. The original FMA workspace produces FP16
        output; the mixed-precision labs explicitly model FP32 accumulation.
      </p>
    </section>
  )
}
