import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { RTLInspection } from '../components/arithmetic/RTLInspection'
import { RTL_SIMULATIONS } from '../lib/arithmetic/rtl'

describe('recorded RTL inspection', () => {
  it('labels precomputed results and compares their actual software reference', async () => {
    const user = userEvent.setup()
    render(<RTLInspection />)
    await user.click(
      screen.getByText('Inspect precomputed FP16 RTL simulations'),
    )
    expect(
      screen.getByText(/browser does not run Verilator/),
    ).toBeInTheDocument()
    expect(screen.getByLabelText('RTL agreement')).toHaveTextContent(
      'Agreement: result bits and all exception flags match',
    )
    expect(screen.getByLabelText('RTL software result')).toHaveTextContent(
      '0x3e00',
    )
    expect(screen.getByLabelText('RTL recorded result')).toHaveTextContent(
      '0x3e00',
    )
    expect(screen.getByText(/462,172 cases/)).toBeInTheDocument()
  })
  it('uses a keyboard-accessible selector for GRS tie decisions and inspectable long signals', async () => {
    const user = userEvent.setup()
    render(<RTLInspection />)
    const toggle = screen.getByText('Inspect precomputed FP16 RTL simulations')
    toggle.focus()
    await user.keyboard('{Enter}')
    await user.selectOptions(screen.getByLabelText('Recorded RTL example'), '1')
    expect(
      screen.getByLabelText('RTL Guard / round / sticky'),
    ).toHaveTextContent('1 / 0 / 0')
    expect(screen.getByLabelText('RTL Increment').textContent).toBe('false')
    expect(screen.getByLabelText('RTL software flags')).toHaveTextContent(
      'inexact: true',
    )
    await user.selectOptions(screen.getByLabelText('Recorded RTL example'), '2')
    expect(screen.getByLabelText('RTL Increment').textContent).toBe('true')
    expect(screen.getByLabelText('RTL recorded result')).toHaveTextContent(
      '0x3c02',
    )
    expect(screen.getByLabelText('RTL Aligned A · 40 bits')).toHaveAttribute(
      'tabindex',
      '0',
    )
  })
  it('explains special paths and preserves signaling-NaN metadata in recorded results', async () => {
    const user = userEvent.setup()
    render(<RTLInspection />)
    await user.click(
      screen.getByText('Inspect precomputed FP16 RTL simulations'),
    )
    await user.selectOptions(
      screen.getByLabelText('Recorded RTL example'),
      '25',
    )
    expect(screen.getByLabelText('RTL recorded result')).toHaveTextContent(
      '0xfe11',
    )
    expect(screen.getByLabelText('RTL recorded flags')).toHaveTextContent(
      'invalid: true',
    )
    expect(
      screen.getByText(/Special-value path: ordinary finite/),
    ).toBeInTheDocument()
  })
  it('reports disagreement for incorrect bits and for flag-only differences', async () => {
    const user = userEvent.setup(),
      sample = RTL_SIMULATIONS[0]
    const { unmount } = render(
      <RTLInspection samples={[{ ...sample, rtlBits: 0 }]} />,
    )
    await user.click(
      screen.getByText('Inspect precomputed FP16 RTL simulations'),
    )
    expect(screen.getByLabelText('RTL agreement')).toHaveTextContent(
      'Disagreement',
    )
    unmount()
    render(
      <RTLInspection
        samples={[
          { ...sample, rtlFlags: { ...sample.rtlFlags, inexact: true } },
        ]}
      />,
    )
    await user.click(
      screen.getByText('Inspect precomputed FP16 RTL simulations'),
    )
    expect(screen.getByLabelText('RTL agreement')).toHaveTextContent(
      'Disagreement',
    )
  })
})
