import addArtifact from '../../../rtl/vectors/simulation.json'
import productArtifact from '../../../rtl/vectors/mul-fma-simulation.json'
import synthesisArtifact from '../../../rtl/vectors/hardware-analysis.json'
import waveArtifact from '../../../rtl/vectors/datapath-waveforms.json'
import { parseSimulationArtifact } from './artifacts'
import { parseSynthesis } from './synthesis'
import { parseWaves, validateCorrelation } from './waves'
import type { RecordedExample } from './artifacts'
import type { SynthesisContext } from './synthesis'
import type { RecordedWave } from './waves'
export interface DatapathCatalog {
  examples: RecordedExample[]
  synthesis: SynthesisContext | null
  waves: RecordedWave[]
  synthesisError: string | null
  waveError: string | null
}
const message = (e: unknown) =>
  e instanceof Error ? e.message : 'Invalid artifact'
export function loadCatalog(
  inputs: {
    add?: unknown
    product?: unknown
    synthesis?: unknown
    waves?: unknown
  } = {},
): { ok: true; catalog: DatapathCatalog } | { ok: false; error: string } {
  try {
    const examples = [
      ...parseSimulationArtifact(
        inputs.add === undefined ? addArtifact : inputs.add,
        'add',
      ),
      ...parseSimulationArtifact(
        inputs.product === undefined ? productArtifact : inputs.product,
        'product',
      ),
    ]
    let synthesis: SynthesisContext | null = null,
      synthesisError: string | null = null,
      waves: RecordedWave[] = [],
      waveError: string | null = null
    try {
      synthesis = parseSynthesis(
        inputs.synthesis === undefined ? synthesisArtifact : inputs.synthesis,
      )
    } catch (e) {
      synthesisError = message(e)
    }
    try {
      waves = parseWaves(
        inputs.waves === undefined ? waveArtifact : inputs.waves,
      )
      validateCorrelation(waves, examples)
    } catch (e) {
      waves = []
      waveError = message(e)
    }
    return {
      ok: true,
      catalog: { examples, synthesis, waves, synthesisError, waveError },
    }
  } catch (e) {
    return { ok: false, error: message(e) }
  }
}
