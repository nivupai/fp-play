import { useState } from 'react'
import { version } from '../package.json'
import { ResearchShowcaseEntry } from './components/showcase/ResearchShowcaseEntry'
import { AttentionLab } from './components/attention/AttentionLab'
import { FormatComparison } from './components/mixed/FormatComparison'
import { DotProductLab } from './components/mixed/DotProductLab'
import { ArithmeticExplorer } from './components/arithmetic/ArithmeticExplorer'
import { PrecisionMicroscope } from './components/microscope/PrecisionMicroscope'
import { FP16Explorer } from './components/FP16Explorer'
import { Header } from './components/Header'
import type { ColorTheme } from './components/Header'
import { Landing } from './components/Landing'
import { Roadmap } from './components/Roadmap'
import { HardwareDatapathEntry } from './components/datapath/HardwareDatapathEntry'
import { HardwareAnalysisEntry } from './components/hardware/HardwareAnalysisEntry'

export default function App() {
  const [theme, setTheme] = useState<ColorTheme>('system')
  return (
    <div className="app-shell" data-theme={theme}>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <Header theme={theme} onThemeChange={setTheme} />
      <main id="main" tabIndex={-1} className="page-width main-content">
        <Landing />
        <ResearchShowcaseEntry />
        <FP16Explorer />
        <PrecisionMicroscope />
        <ArithmeticExplorer />
        <FormatComparison />
        <DotProductLab />
        <AttentionLab />
        <HardwareAnalysisEntry />
        <HardwareDatapathEntry />
        <Roadmap />
        <section
          id="about"
          aria-labelledby="about-heading"
          className="about-section"
        >
          <h2 id="about-heading">Precision should be understandable.</h2>
          <p className="muted">
            FP Play connects numerical representations to their consequences
            through accessible, testable experiments. Explore IEEE 754
            representation, precision, and exact add/subtract/multiply/FMA
            traces, BF16/FP32 representations, and mixed-precision accumulation.
            Explore bounded transformer attention with explicit softmax
            policies. All work uses public standards and original, open-source
            implementations.
          </p>
        </section>
      </main>
      <footer className="site-footer">
        <span>FP Play · v{version}</span>
        <a href="https://github.com/nivupai/fp-play">Source on GitHub</a>
        <a href="https://github.com/nivupai/fp-play/blob/main/LICENSE">
          MIT License
        </a>
        <a href="https://github.com/nivupai/fp-play/blob/main/docs/README.md">
          Documentation
        </a>
        <a href={`${import.meta.env.BASE_URL}THIRD_PARTY_NOTICES.txt`}>
          Third-party notices
        </a>
        <a href="#hardware">Hardware analysis</a>
        <a href="#datapath">Datapath explorer</a>
        <a href="#main">Back to top ↑</a>
      </footer>
    </div>
  )
}
