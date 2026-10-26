import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { MulFmaRTLInspection } from '../components/arithmetic/MulFmaRTLInspection'
describe('recorded multiplier and FMA inspection', () => {
  it('exposes real records and a keyboard-accessible comparison selector', async () => {
    const user = userEvent.setup()
    render(<MulFmaRTLInspection />)
    screen.getByText('Inspect multiplier and true FMA RTL').focus()
    await user.keyboard('{Enter}')
    const select = screen.getByLabelText('Recorded multiply/FMA example')
    select.focus()
    expect(select).toHaveFocus()
    await user.selectOptions(select, '1')
    expect(screen.getByLabelText('True FMA hardware result')).toHaveTextContent(
      '0x0010',
    )
    expect(screen.getByLabelText('Nonfused hardware result')).toHaveTextContent(
      '0x0000',
    )
    expect(screen.getByText(/Agreement: all three/)).toBeInTheDocument()
    expect(
      screen.getByText(/genuine precomputed Verilator/),
    ).toBeInTheDocument()
    expect(
      BigInt(screen.getByLabelText('FMA RTL arithmeticMagnitude').textContent!),
    ).toBe(1n << 28n)
    expect(
      screen.getByLabelText('FMA RTL arithmeticMagnitude'),
    ).toHaveAttribute('tabindex', '0')
  })
  it('shows signed zero, special bypasses and intermediate overflow', async () => {
    const user = userEvent.setup()
    render(<MulFmaRTLInspection />)
    await user.click(screen.getByText('Inspect multiplier and true FMA RTL'))
    const select = screen.getByLabelText('Recorded multiply/FMA example')
    await user.selectOptions(select, '5')
    expect(screen.getByLabelText('True FMA hardware result')).toHaveTextContent(
      '0x8000',
    )
    await user.selectOptions(select, '2')
    expect(screen.getByLabelText('True FMA hardware result')).toHaveTextContent(
      '0x7bff',
    )
    expect(screen.getByLabelText('Nonfused hardware result')).toHaveTextContent(
      '0x7c00',
    )
    await user.selectOptions(select, '10')
    expect(screen.getByLabelText('True FMA hardware result')).toHaveTextContent(
      '0xfe11',
    )
    expect(screen.getAllByText(/Special-value bypass/)).toHaveLength(2)
  })
})
