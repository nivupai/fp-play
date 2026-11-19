import { render } from '@testing-library/react'
import axe from 'axe-core'
import { describe, expect, it } from 'vitest'
import App from '../App'
import { ServingWorkspace } from '../components/runtime/ServingWorkspace'
import { DecodeWorkspace } from '../components/runtime/DecodeWorkspace'
import { PrecisionWorkspace } from '../components/runtime/PrecisionWorkspace'
import { DataflowWorkspace } from '../components/runtime/DataflowWorkspace'
import { CollectivesWorkspace } from '../components/collectives/CollectivesWorkspace'

/*
 * Automated axe-core audit in jsdom. jsdom does no layout or painting, so
 * color contrast (and anything else needing rendering) is not checked here;
 * docs/RELEASE_VERIFICATION.md records the same audit run in headless
 * Chromium, where contrast is included.
 */
const options: axe.RunOptions = {
  rules: { 'color-contrast': { enabled: false } },
}

async function violations(node: Element) {
  const r = await axe.run(node, options)
  return r.violations.map(
    (v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(', ')}`,
  )
}

describe('axe-core audit (jsdom, contrast excluded)', () => {
  it('the whole app with the landing page has no violations', async () => {
    const { container } = render(<App />)
    expect(await violations(container)).toEqual([])
  }, 30_000)

  it.each([
    ['serving', ServingWorkspace],
    ['decode', DecodeWorkspace],
    ['precision', PrecisionWorkspace],
    ['dataflow', DataflowWorkspace],
    ['collectives', CollectivesWorkspace],
  ] as const)(
    'the %s workspace has no violations',
    async (_, Workspace) => {
      const { container } = render(
        <main>
          <Workspace />
        </main>,
      )
      expect(await violations(container)).toEqual([])
    },
    30_000,
  )
})
