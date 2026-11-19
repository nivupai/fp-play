import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import {
  DataflowEntry,
  DecodeEntry,
  PrecisionEntry,
  ServingEntry,
} from '../components/runtime/RuntimeEntries'
import { ServingWorkspace } from '../components/runtime/ServingWorkspace'
import { DecodeWorkspace } from '../components/runtime/DecodeWorkspace'
import { PrecisionWorkspace } from '../components/runtime/PrecisionWorkspace'
import { DataflowWorkspace } from '../components/runtime/DataflowWorkspace'

afterEach(() => window.history.replaceState(null, '', '#main'))

describe('runtime workspace entries', () => {
  it.each([
    ['serving', ServingEntry, 'Scenario group'],
    ['decode', DecodeEntry, 'Model and prompts'],
    ['precision', PrecisionEntry, 'Model and prompts'],
    ['dataflow', DataflowEntry, 'Machine'],
  ] as const)(
    '#%s loads only on request, including by hash navigation',
    async (id, Entry, control) => {
      render(<Entry />)
      expect(screen.queryByLabelText(control)).not.toBeInTheDocument()
      act(() => {
        window.history.replaceState(null, '', `#${id}`)
        fireEvent(window, new Event('hashchange'))
      })
      expect(await screen.findByLabelText(control)).toBeInTheDocument()
    },
  )
})

/** Every figure in a workspace is followed by a table (consecutive charts may share one). */
function expectTablesForCharts(container: HTMLElement) {
  const figures = container.querySelectorAll('figure')
  expect(figures.length).toBeGreaterThan(0)
  for (const f of figures) {
    let next = f.nextElementSibling
    while (next && !next.querySelector('table')) next = next.nextElementSibling
    expect(next?.querySelector('table'), f.textContent ?? '').toBeTruthy()
  }
}

describe('serving workspace', () => {
  it('switches groups and metrics; the chart and the table show the same scenarios', async () => {
    const user = userEvent.setup()
    const { container } = render(<ServingWorkspace />)
    const table = () => screen.getByRole('table', { name: /every scenario/ })
    expect(within(table()).getAllByRole('row')).toHaveLength(1 + 12)
    expect(screen.getByText('26.43 ms')).toBeInTheDocument()
    await user.selectOptions(
      screen.getByLabelText('Scenario group'),
      'prefix_sharing',
    )
    expect(within(table()).getAllByRole('row')).toHaveLength(1 + 2)
    expect(within(table()).getByText('42.9%')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Metric'), 'throughput')
    expect(
      screen.getByText(/^Output tokens per second per scenario/),
    ).toBeInTheDocument()
    expectTablesForCharts(container)
  })
})

describe('dataflow workspace', () => {
  it('shows every shape and the ops of the chosen shape', async () => {
    const user = userEvent.setup()
    const { container } = render(<DataflowWorkspace />)
    const runs = screen.getByRole('table', { name: /Every shape/ })
    expect(within(runs).getAllByRole('row')).toHaveLength(1 + 7)
    expect(within(runs).getByText('95.7%')).toBeInTheDocument()
    await user.selectOptions(
      screen.getByLabelText('Shape'),
      'decode batch 1, context 1024',
    )
    expect(
      screen.getByRole('table', { name: /Ops of decode batch 1/ }),
    ).toBeInTheDocument()
    await user.selectOptions(
      screen.getByLabelText('Machine'),
      screen.getAllByRole('option', { name: /4\.179512/ })[0],
    )
    expect(screen.getByText(/not an achievable clock/)).toBeInTheDocument()
    expectTablesForCharts(container)
  })
})

describe('decode workspace', () => {
  it('compares random and trained weights and shows the batch-invariance verdict', async () => {
    const user = userEvent.setup()
    const { container } = render(<DecodeWorkspace />)
    const margins = screen.getByRole('table', { name: /Top-2 margins/ })
    expect(
      within(margins).getByText(/5\.5726 \/ 8\.6619 \/ 9\.3781/),
    ).toBeInTheDocument()
    expect(within(margins).getByText('160 / 12527')).toBeInTheDocument()
    expect(screen.getByText('The prediction held exactly.')).toBeInTheDocument()
    await user.selectOptions(
      screen.getByLabelText('Model and prompts'),
      'trained-trained',
    )
    const grid = screen.getByRole('table', { name: /Every grid run/ })
    expect(within(grid).getAllByText('0 of 32')).toHaveLength(19)
    await user.selectOptions(
      screen.getByLabelText('Wire policy'),
      'fp32-rs/bf16-ag',
    )
    expect(
      screen.getByRole('table', { name: /fp32-rs\/bf16-ag wire\)\. At-risk/ }),
    ).toBeInTheDocument()
    expectTablesForCharts(container)
  })
})

describe('precision workspace', () => {
  it('shows every level of the chosen model and the training run', async () => {
    const user = userEvent.setup()
    const { container } = render(<PrecisionWorkspace />)
    const uniform = () =>
      screen.getByRole('table', { name: /Every site at one level/ })
    expect(within(uniform()).getAllByRole('row')).toHaveLength(1 + 8)
    expect(within(uniform()).getByText('9.24e-5')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Model and prompts'), 'r4')
    expect(within(uniform()).getAllByRole('row')).toHaveLength(1 + 5)
    expect(within(uniform()).getByText('5.05e-5')).toBeInTheDocument()
    expect(
      screen.getByRole('img', { name: /Training loss from 8\.12/ }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole('table', { name: /Held-out accuracy/ }),
    ).toHaveTextContent('64 of 64')
    expectTablesForCharts(container)
  })
})
