import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import App from '../App'
import { ArithmeticExplorer } from '../components/arithmetic/ArithmeticExplorer'

afterEach(() => window.history.replaceState(null, '', '/'))

describe('public release entry points and accessibility', () => {
  it('introduces four experiences with named, existing hash targets and source documentation', () => {
    render(<App />)
    const navigation = screen.getByRole('navigation', {
      name: 'Four ways to see the same sixteen bits',
    })
    const links = within(navigation).getAllByRole('link')
    expect(links).toHaveLength(4)
    for (const [title, target] of [
      ['Bit Explorer', 'explorer'],
      ['Precision Microscope', 'microscope'],
      ['Arithmetic Pipeline', 'arithmetic'],
      ['FMA Comparison', 'fma'],
    ]) {
      const link = within(navigation).getByRole('link', { name: title })
      expect(link).toHaveAttribute('href', `#${target}`)
      expect(document.getElementById(target)).toHaveAccessibleName()
    }
    expect(
      screen.getByRole('link', { name: 'GitHub repository' }),
    ).toHaveAttribute('href', 'https://github.com/nivupai/fp-play')
    expect(
      screen.getByRole('link', { name: 'Technical documentation' }),
    ).toHaveAttribute(
      'href',
      'https://github.com/nivupai/fp-play/blob/main/src/lib/numerics/README.md',
    )
    expect(
      screen.getByText(/No full IEEE 754 conformance claim/),
    ).toBeInTheDocument()
  })

  it('opens an independently verified FMA example through a hash link without resetting the bit explorer', async () => {
    const user = userEvent.setup()
    render(<App />)
    // Keep role/label queries inside their workspace instead of traversing every lab.
    const explorer = within(
      screen.getByRole('region', { name: 'FP16 Explorer' }),
    )
    const arithmetic = within(
      screen.getByRole('region', { name: 'FP16 Arithmetic Explorer' }),
    )
    await user.click(explorer.getByRole('button', { name: 'Sign bit 15' }))
    act(() => {
      window.history.replaceState(null, '', '#fma')
      fireEvent(window, new Event('hashchange'))
    })
    expect(arithmetic.getByLabelText('Fused output hex')).toHaveTextContent(
      '0x0010',
    )
    expect(arithmetic.getByLabelText('Nonfused output hex')).toHaveTextContent(
      '0x0000',
    )
    expect(explorer.getByLabelText('Hexadecimal')).toHaveTextContent('0xbe00')
    await user.click(
      arithmetic.getByRole('button', { name: /^Exact addition/ }),
    )
    expect(arithmetic.queryByLabelText('Fused output')).not.toBeInTheDocument()
    const landing = screen.getByRole('navigation', {
      name: 'Four ways to see the same sixteen bits',
    })
    await user.click(
      within(landing).getByRole('link', { name: 'FMA Comparison' }),
    )
    expect(arithmetic.getByLabelText('Fused output hex')).toHaveTextContent(
      '0x0010',
    )
  })

  it('supports direct FMA hash loading, with all three operands and the correct comparison', () => {
    window.history.replaceState(null, '', '#fma')
    render(<ArithmeticExplorer />)
    expect(screen.getByRole('textbox', { name: 'Operand C' })).toHaveValue(
      '0xbc02',
    )
    expect(screen.getByLabelText('Fused output').textContent).toBe(
      '0.00000095367431640625',
    )
    expect(screen.getByLabelText('Nonfused output').textContent).toBe('0')
  })

  it('offers a keyboard-activated FMA entry when no comparison has been applied', async () => {
    const user = userEvent.setup()
    render(<ArithmeticExplorer />)
    const button = screen.getByRole('button', { name: 'Start FMA comparison' })
    button.focus()
    await user.keyboard('{Enter}')
    expect(screen.getByLabelText('Fused output hex')).toHaveTextContent(
      '0x0010',
    )
    expect(screen.getByRole('status')).toHaveTextContent('Stage 1 of 8')
  })

  it('makes long binary data focusable without intercepting its arrow keys as stage navigation', async () => {
    const user = userEvent.setup()
    render(<ArithmeticExplorer />)
    await user.click(
      screen.getByRole('button', { name: /^Large exponent difference/ }),
    )
    await user.click(
      screen.getByRole('button', { name: '3. Lossless significand alignment' }),
    )
    const binary = screen.getByLabelText('B aligned binary')
    expect(binary).toHaveAttribute('tabindex', '0')
    binary.focus()
    await user.keyboard('{ArrowRight}')
    expect(screen.getByRole('status')).toHaveTextContent('Stage 3 of 8')
    const panel = screen.getByRole('group', { name: 'Arithmetic stage detail' })
    panel.focus()
    await user.keyboard('{ArrowRight}')
    expect(screen.getByRole('status')).toHaveTextContent('Stage 4 of 8')
  })

  it('provides keyboard focus for plots and a named point table in addition to chart shortcuts', async () => {
    const user = userEvent.setup()
    render(<App />)
    for (const name of ['Scrollable number line plot', 'Scrollable ULP plot'])
      expect(screen.getByRole('group', { name })).toHaveAttribute(
        'tabindex',
        '0',
      )
    await user.click(screen.getByText(/Visible point data/))
    expect(
      screen.getByRole('group', { name: 'Scrollable FP16 point table' }),
    ).toHaveAttribute('tabindex', '0')
  })
})
