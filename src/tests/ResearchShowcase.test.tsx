import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import { version } from '../../package.json'
import { Landing } from '../components/Landing'
import { ResearchShowcase } from '../components/showcase/ResearchShowcase'
import { ResearchShowcaseEntry } from '../components/showcase/ResearchShowcaseEntry'
afterEach(() => window.history.replaceState(null, '', '#main'))
describe('research showcase and compact landing', () => {
  it('preserves the four main entries while introducing verified evidence, hardware and technical guides', () => {
    render(<Landing />)
    expect(
      within(
        screen.getByRole('navigation', {
          name: 'Four ways to see the same sixteen bits',
        }),
      ).getAllByRole('link'),
    ).toHaveLength(4)
    expect(screen.getByText('v' + version)).toBeInTheDocument()
    expect(
      screen.getByLabelText('Selected verification evidence'),
    ).toHaveTextContent('536,888')
    expect(screen.getByLabelText('Verified examples')).toHaveTextContent(
      '1.0001 → 1',
    )
    expect(screen.getByLabelText('Verified examples')).toHaveTextContent(
      '0x3c00',
    )
    expect(screen.getByLabelText('Architecture overview')).toHaveTextContent(
      'verified FP16 SystemVerilog',
    )
    expect(
      screen.getByRole('link', { name: 'Measured hardware showcase →' }),
    ).toHaveAttribute('href', '#research')
    expect(
      screen.getByRole('link', { name: 'Technical case study →' }),
    ).toHaveAttribute(
      'href',
      'https://github.com/nivupai/fp-play/blob/main/docs/CASE_STUDY.md',
    )
    expect(
      screen.getByRole('link', { name: 'Reproduce the results →' }),
    ).toHaveAttribute(
      'href',
      'https://github.com/nivupai/fp-play/blob/main/docs/REPRODUCIBILITY.md',
    )
  })
  it('loads detailed evidence only on request and supports repeated direct hash navigation', async () => {
    window.history.replaceState(null, '', '#main')
    render(<ResearchShowcaseEntry />)
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    act(() => {
      window.history.replaceState(null, '', '#research')
      fireEvent(window, new Event('hashchange'))
    })
    expect(await screen.findByLabelText('Compare hardware units')).toHaveValue(
      'all',
    )
    const summary = screen.getByText(
      'Hardware research showcase · compare the seven measured experiments',
    )
    await userEvent.click(summary)
    expect(document.getElementById('research')).not.toHaveAttribute('open')
    act(() => fireEvent(window, new Event('hashchange')))
    expect(
      await screen.findByLabelText('Compare hardware units'),
    ).toBeInTheDocument()
  })
  it('shows all seven actual measurements with accessible shape and exact table alternatives', () => {
    render(<ResearchShowcase />)
    const table = within(
      screen.getByRole('region', {
        name: 'Showcase exact hardware measurements',
      }),
    )
    expect(table.getAllByRole('row')).toHaveLength(8)
    expect(table.getByText('3,588.074')).toBeInTheDocument()
    expect(table.getByText('4.179512')).toBeInTheDocument()
    expect(
      screen.getByRole('img', { name: /Hardware area and delay scatter plot/ }),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Circles: baseline. Diamonds: tree/),
    ).toBeInTheDocument()
    expect(screen.getByText(/9.89% less mapped cell area/)).toHaveTextContent(
      '26.06%',
    )
    expect(screen.getByText(/Hypothesis:/)).toBeInTheDocument()
    expect(screen.getByText(/not isolated by ablation/)).toBeInTheDocument()
  })
  it('filters baseline and tree pairs with a native keyboard-accessible selector', async () => {
    const user = userEvent.setup()
    render(<ResearchShowcase />)
    const select = screen.getByLabelText('Compare hardware units')
    select.focus()
    expect(select).toHaveFocus()
    await user.selectOptions(select, 'mul')
    const table = within(
      screen.getByRole('region', {
        name: 'Showcase exact hardware measurements',
      }),
    )
    expect(table.getAllByRole('row')).toHaveLength(3)
    await user.click(
      table.getByRole('button', { name: 'Multiply / tree-leading-bit' }),
    )
    expect(
      screen.getByRole('button', { name: 'Multiply / tree-leading-bit' }),
    ).toHaveAttribute('aria-pressed', 'true')
    expect(
      screen.getByLabelText('Showcase selected experiment'),
    ).toHaveTextContent('92 cell arcs')
    expect(
      screen.getByLabelText('Showcase selected experiment'),
    ).toHaveTextContent('result[8]')
    await user.selectOptions(select, 'add')
    expect(table.getAllByRole('row')).toHaveLength(2)
    expect(
      screen.getByLabelText('Showcase selected experiment'),
    ).toHaveTextContent('Add/subtract · baseline')
  })
  it('exposes real critical endpoints, full conditions and original dirty-state provenance', async () => {
    render(<ResearchShowcase />)
    await userEvent.click(
      screen.getByRole('button', { name: 'True FMA / tree-leading-bit' }),
    )
    const selected = screen.getByLabelText('Showcase selected experiment')
    expect(selected).toHaveTextContent(
      'flags[1] (output port clocked by virtual_io)',
    )
    expect(selected).toHaveTextContent('141 cell arcs')
    expect(screen.getByText(/Input slew 0.05 ns/)).toHaveTextContent(
      'virtual IO clock 100 ns',
    )
    expect(screen.getByText(/Yosys 0.70\+219/)).toBeInTheDocument()
    await userEvent.click(
      screen.getByText('Measurement provenance and snapshot source'),
    )
    expect(screen.getByText(/dirty working tree: true/)).toBeInTheDocument()
    expect(
      screen.getByRole('link', { name: 'rtl/vectors/hardware-analysis.json' }),
    ).toBeInTheDocument()
  })
})
