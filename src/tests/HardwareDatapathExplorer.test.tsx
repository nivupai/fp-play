import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import add from '../../rtl/vectors/simulation.json'
import product from '../../rtl/vectors/mul-fma-simulation.json'
import synthesis from '../../rtl/vectors/hardware-analysis.json'
import { HardwareDatapathExplorer } from '../components/datapath/HardwareDatapathExplorer'
import { HardwareDatapathEntry } from '../components/datapath/HardwareDatapathEntry'

afterEach(() => vi.unstubAllEnvs())

const operation = (value: string) =>
  fireEvent.change(screen.getByLabelText('Datapath operation'), {
    target: { value },
  })
const example = (value: string) =>
  fireEvent.change(screen.getByLabelText('Verified hardware example'), {
    target: { value },
  })
const region = (name: string) => within(screen.getByRole('region', { name }))
async function open(label: string) {
  await userEvent.click(screen.getByText(label, { selector: 'summary' }))
}

describe('Hardware Datapath Explorer integration with real records and numerical engines', () => {
  it('loads lazily and follows direct and repeated hash navigation', async () => {
    window.history.replaceState(null, '', '#main')
    render(<HardwareDatapathEntry />)
    expect(
      screen.queryByLabelText('Datapath operation'),
    ).not.toBeInTheDocument()
    act(() => {
      window.history.replaceState(null, '', '#datapath')
      fireEvent(window, new Event('hashchange'))
    })
    expect(await screen.findByLabelText('Datapath operation')).toHaveValue(
      'add',
    )
    expect(document.getElementById('datapath')).toHaveAttribute('open')
    await open(
      'Hardware Datapath Explorer · open logical stages and recorded signals',
    )
    expect(document.getElementById('datapath')).not.toHaveAttribute('open')
    act(() => {
      fireEvent(window, new Event('hashchange'))
    })
    expect(
      await screen.findByLabelText('Datapath operation'),
    ).toBeInTheDocument()
    window.history.replaceState(null, '', '#main')
  })
  it('shows observed bits, exact software, missing signals and actual provenance separately', async () => {
    render(<HardwareDatapathExplorer />)
    expect(screen.getByLabelText('Datapath RTL result')).toHaveTextContent(
      '0x3e00',
    )
    expect(screen.getByLabelText('Datapath agreement')).toHaveTextContent(
      'Agreement:',
    )
    expect(
      region('Exact software reference').getByText(/1.5 =/),
    ).toBeInTheDocument()
    expect(
      region('Observed stage signals').getAllByText(/Unavailable/),
    ).toHaveLength(2)
    await open('Recorded RTL provenance and source hashes')
    expect(screen.getByText(/462,172; seed/)).toHaveTextContent('Verilator')
    expect(screen.getAllByText('rtl/src/fp16_addsub.sv')).toHaveLength(1)
    expect(
      screen.getByRole('img', { name: /Logical block diagram/ }),
    ).toBeInTheDocument()
  })
  it('selects real subtraction and negative-zero examples', () => {
    render(<HardwareDatapathExplorer />)
    example('add:17:add')
    expect(screen.getByLabelText('Datapath RTL result')).toHaveTextContent(
      '0x8000 · -0',
    )
    operation('subtract')
    example('add:11:subtract')
    expect(screen.getByLabelText('Datapath RTL result')).toHaveTextContent(
      '0x03ff',
    )
    expect(screen.getByLabelText('Datapath agreement')).toHaveTextContent(
      'Agreement:',
    )
  })
  it('distinguishes true FMA from the two actual nonfused phases and their unavailable adder signals', () => {
    render(<HardwareDatapathExplorer />)
    operation('fma')
    example('product:1:fma')
    expect(screen.getByLabelText('Datapath RTL result')).toHaveTextContent(
      '0x0010',
    )
    expect(screen.getByText(/True FMA: exact/)).toBeInTheDocument()
    expect(
      region('Exact software reference').getByText(/0.00000095367431640625/),
    ).toBeInTheDocument()
    operation('nonfused')
    example('product:1:nonfused')
    expect(screen.getByLabelText('Datapath RTL result')).toHaveTextContent(
      '0x0000',
    )
    const nav = screen.getByRole('group', { name: 'Datapath stage navigation' })
    fireEvent.keyDown(nav, { key: 'End' })
    expect(nav).toHaveTextContent('16 / 16')
    fireEvent.click(screen.getByRole('button', { name: /Add · 6\./ }))
    expect(region('GRS explorer').getByText(/not recorded/)).toBeInTheDocument()
    expect(
      region('Observed stage signals').getAllByText(/Unavailable/).length,
    ).toBeGreaterThan(0)
  })
  it('supports bounded keyboard and button navigation without intercepting code scrolling', async () => {
    const user = userEvent.setup()
    render(<HardwareDatapathExplorer />)
    const nav = screen.getByRole('group', { name: 'Datapath stage navigation' })
    nav.focus()
    await user.keyboard('{ArrowRight}')
    expect(nav).toHaveFocus()
    expect(nav).toHaveTextContent('2 / 8')
    await user.keyboard('{End}{ArrowRight}')
    expect(nav).toHaveTextContent('8 / 8')
    expect(
      screen.getByRole('button', { name: 'Next datapath stage' }),
    ).toBeDisabled()
    await user.keyboard('{Home}{ArrowLeft}')
    expect(
      screen.getByRole('button', { name: 'Previous datapath stage' }),
    ).toBeDisabled()
    const bits = screen
      .getByRole('region', { name: 'Exact software reference' })
      .querySelector('code')!
    fireEvent.keyDown(bits, { key: 'End' })
    expect(nav).toHaveTextContent('1 / 8')
    await user.click(
      screen.getByRole('button', { name: 'Next datapath stage' }),
    )
    expect(
      screen.getByRole('button', { name: /2\. Exponent/ }),
    ).toHaveAttribute('aria-current', 'step')
  })
  it('explains observed odd and even midpoint rounding step by step', async () => {
    render(<HardwareDatapathExplorer />)
    operation('multiply')
    example('product:20:multiply')
    expect(
      screen.getByLabelText('Verified hardware example'),
    ).toHaveDisplayValue('Product midpoint · retained odd · increment')
    expect(screen.getByLabelText('GRS conclusion')).toHaveTextContent(
      'Exact midpoint tie; increment',
    )
    const nav = screen.getByRole('group', { name: 'GRS step navigation' })
    nav.focus()
    await userEvent.keyboard('{End}')
    expect(nav).toHaveTextContent('annotation, not a recorded carry wire')
    expect(screen.getByRole('button', { name: 'Next GRS step' })).toBeDisabled()
    example('product:21:multiply')
    expect(screen.getByLabelText('GRS conclusion')).toHaveTextContent(
      'retain the significand because its LSB is even',
    )
    expect(
      screen.getByRole('group', { name: 'GRS step navigation' }),
    ).toHaveTextContent('1 / 6')
  })
  it('explains carry, subnormal transition and underflow using genuine examples', () => {
    render(<HardwareDatapathExplorer />)
    example('add:6:add')
    fireEvent.click(screen.getByRole('button', { name: 'Carry / transition' }))
    expect(
      region('GRS explorer').getByText(/Carry inferred.*true/),
    ).toBeInTheDocument()
    operation('multiply')
    example('product:22:multiply')
    expect(
      region('GRS explorer').getByText(/Software trace: subnormal-to-normal/),
    ).toBeInTheDocument()
    example('product:4:multiply')
    expect(screen.getByLabelText('Datapath RTL flags')).toHaveTextContent(
      'underflow: true; inexact: true',
    )
  })
  it('preserves NaN payloads and infinity while marking finite debug defaults bypassed', () => {
    render(<HardwareDatapathExplorer />)
    operation('fma')
    example('product:10:fma')
    expect(screen.getByLabelText('Datapath RTL result')).toHaveTextContent(
      '0xfe11 · NaN',
    )
    expect(screen.getByLabelText('Datapath RTL flags')).toHaveTextContent(
      'invalid: true',
    )
    fireEvent.click(screen.getByRole('button', { name: /6\. Guard/ }))
    expect(
      region('Selected logical stage').getByText(/special-value bypass/),
    ).toBeInTheDocument()
    expect(
      region('GRS explorer').getByText(/do not describe executed rounding/),
    ).toBeInTheDocument()
    operation('add')
    example('add:20:add')
    expect(screen.getByLabelText('Datapath RTL result')).toHaveTextContent(
      '0x7c00 · Infinity',
    )
  })
  it('ties whole-unit metrics to actual matching sources, tools and constraints', () => {
    render(<HardwareDatapathExplorer />)
    operation('fma')
    const table = region('Synthesis comparison table')
    expect(table.getAllByRole('row')).toHaveLength(3)
    expect(table.getByText('3221')).toBeInTheDocument()
    expect(table.getByText('3205')).toBeInTheDocument()
    expect(
      region('Datapath synthesis comparison').getByText(/Nangate45 timing/),
    ).toBeInTheDocument()
    expect(screen.getByText(/Yosys 0.70/)).toBeInTheDocument()
    expect(screen.getByText(/1.1.*25/)).toBeInTheDocument()
  })
  it('exposes actual VCD values with accessible signal selection, zoom, pan and download', async () => {
    vi.stubEnv('BASE_URL', '/fp-play/')
    render(<HardwareDatapathExplorer />)
    await open('Inspect recorded waveforms')
    expect(screen.getByLabelText('Selected waveform binary')).toHaveTextContent(
      '0011111000000000',
    )
    const image = () =>
      screen.getByRole('img', { name: /Recorded result waveform/ })
    expect(image()).toHaveAttribute(
      'aria-label',
      'Recorded result waveform, frames 0 to 7',
    )
    await userEvent.click(
      screen.getByRole('button', { name: 'Zoom waveform in' }),
    )
    expect(image()).toHaveAttribute(
      'aria-label',
      'Recorded result waveform, frames 0 to 3',
    )
    await userEvent.click(
      screen.getByRole('button', { name: 'Pan waveform right' }),
    )
    expect(image()).toHaveAttribute(
      'aria-label',
      'Recorded result waveform, frames 1 to 4',
    )
    fireEvent.change(screen.getByLabelText('First visible waveform frame'), {
      target: { value: '3' },
    })
    expect(image()).toHaveAttribute(
      'aria-label',
      'Recorded result waveform, frames 3 to 6',
    )
    await userEvent.selectOptions(
      screen.getByLabelText('Waveform signal'),
      'guard_bit',
    )
    expect(screen.getByLabelText('Selected waveform binary')).toHaveTextContent(
      '0',
    )
    expect(
      screen.getByRole('link', { name: 'Download raw add VCD' }),
    ).toHaveAttribute('href', '/fp-play/artifacts/datapath-add.vcd')
  })
  it('reports invalid mandatory data without substituting a fabricated example', () => {
    render(<HardwareDatapathExplorer inputs={{ add: null }} />)
    expect(screen.getByRole('alert')).toHaveTextContent('could not be loaded')
    expect(
      screen.queryByLabelText('Datapath RTL result'),
    ).not.toBeInTheDocument()
  })
  it('handles empty datasets and missing optional synthesis/waves', async () => {
    const view = render(
      <HardwareDatapathExplorer inputs={{ add: { ...add, samples: [] } }} />,
    )
    expect(screen.getByText(/No recorded examples/)).toBeInTheDocument()
    view.rerender(
      <HardwareDatapathExplorer inputs={{ synthesis: null, waves: null }} />,
    )
    expect(
      screen.getByText('Synthesis comparison unavailable'),
    ).toBeInTheDocument()
    await open('Inspect recorded waveforms')
    expect(screen.getByText(/Waveforms unavailable/)).toBeInTheDocument()
    expect(screen.getByLabelText('Datapath agreement')).toHaveTextContent(
      'Agreement:',
    )
  })
  it('reports valid-shape data disagreements and source mismatches visibly', () => {
    const corrupt = structuredClone(add)
    corrupt.samples[0].rtlBits ^= 1
    const changed = structuredClone(synthesis)
    changed.sourceHashes['rtl/src/fp16_addsub.sv'] = 'f'.repeat(64)
    render(
      <HardwareDatapathExplorer
        inputs={{ add: corrupt, synthesis: changed, product }}
      />,
    )
    expect(screen.getByLabelText('Datapath agreement')).toHaveTextContent(
      'DISAGREEMENT',
    )
    expect(
      screen.getByText(/No automatic area\/delay comparison/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole('region', { name: 'Synthesis comparison table' }),
    ).not.toBeInTheDocument()
  })
})
