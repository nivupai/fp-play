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
          <p className="hero-description">
            A hands-on laboratory for understanding how bits become numbers, and
            how representation and accumulation precision shape a result.
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
            v0.1.0 <span>Release preparation</span>
          </p>
          <p>
            Built for curious engineers, students, and researchers in numerical
            computing and AI hardware.
          </p>
          <p className="muted text-small">
            Exact FP16, BF16, and FP32 models. Computation-derived traces.
            Public standards and open-source code.
          </p>
        </aside>
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
            aria-labelledby={`experience-${experience.id}`}
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
      <p className="landing-context">
        Why this matters: accelerator workloads combine many small numerical
        decisions. Explore how representation, spacing, cancellation, and
        rounding influence a result—without assuming a particular chip design.
      </p>
      <p className="landing-limits text-small muted">
        Educational floating-point models with nearest-even rounding. No full
        IEEE 754 conformance claim. The original FMA workspace produces FP16
        output; the mixed-precision labs explicitly model FP32 accumulation.
      </p>
    </section>
  )
}
