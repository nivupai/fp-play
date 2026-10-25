import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { FormatComparison } from '../components/mixed/FormatComparison'
import { DotProductLab } from '../components/mixed/DotProductLab'

describe('format comparison with the actual numerical engine', () => {
  it('compares three labeled bit layouts and quantization errors', () => {
    render(<FormatComparison />)
    expect(screen.getByLabelText('FP16 quantized value')).toHaveTextContent(
      '1.0009765625',
    )
    expect(screen.getByLabelText('BF16 quantized value')).toHaveTextContent('1')
    expect(screen.getByLabelText('FP32 bit layout')).toHaveTextContent(
      'F 00000000010000000000000',
    )
    expect(
      within(
        screen.getByRole('article', { name: 'BF16 representation' }),
      ).getByText('0.0009765625'),
    ).toBeInTheDocument()
  })
  it('applies negative zero and rejects invalid drafts without losing the previous result', async () => {
    const user = userEvent.setup()
    render(<FormatComparison />)
    const input = screen.getByLabelText('Compare a binary64 value')
    await user.clear(input)
    await user.type(input, '-0{Enter}')
    for (const name of ['FP16', 'BF16', 'FP32'])
      expect(screen.getByLabelText(`${name} quantized value`).textContent).toBe(
        '-0',
      )
    await user.clear(input)
    await user.type(input, 'oops{Enter}')
    expect(screen.getByRole('alert')).toHaveTextContent('Enter a decimal')
    expect(screen.getByLabelText('FP16 quantized value').textContent).toBe('-0')
  })
  it('handles overflow, infinities, and canonical NaNs', async () => {
    const user = userEvent.setup()
    render(<FormatComparison />)
    const input = screen.getByLabelText('Compare a binary64 value')
    for (const value of ['65536', 'Infinity', 'NaN']) {
      fireEvent.change(input, { target: { value } })
      await user.click(screen.getByRole('button', { name: 'Compare formats' }))
      expect(screen.getByLabelText('FP16 quantized value')).toHaveTextContent(
        value === 'NaN' ? 'NaN' : 'Infinity',
      )
      if (value === '65536')
        expect(screen.getByLabelText('BF16 quantized value')).toHaveTextContent(
          '65536',
        )
    }
  })
})
describe('dot-product lab integration and keyboard controls', () => {
  it('compares separately rounded and fused accumulation with exact references', async () => {
    const user = userEvent.setup()
    render(<DotProductLab />)
    expect(screen.getByLabelText('Mode B result').textContent).toBe('0')
    expect(screen.getByLabelText('Mode C result').textContent).toBe(
      '0.00000095367431640625',
    )
    expect(
      screen.getByLabelText('Exact quantized-input dot'),
    ).toHaveTextContent('0.00000095367431640625')
    const preset = screen.getByRole('button', { name: 'FP32 FMA accumulation' })
    preset.focus()
    await user.keyboard('{Enter}')
    expect(screen.getByLabelText('Precision configuration')).toHaveValue('C')
    expect(screen.getByLabelText('Accumulator result').textContent).toBe(
      '0.00000095367431640625',
    )
    await user.click(screen.getByRole('button', { name: 'Next step' }))
    expect(screen.getByText('Step 2 of 2: fma; terms 1, 2')).toBeInTheDocument()
    expect(screen.getByLabelText('Step binary')).toHaveTextContent(
      '00110101100000000000000000000000',
    )
  })
  it('changes reduction order and steps through the tree using keyboard-accessible controls', async () => {
    const user = userEvent.setup()
    render(<DotProductLab />)
    await user.click(
      screen.getByRole('button', { name: 'Pairwise saves increments' }),
    )
    expect(screen.getByLabelText('Accumulator result').textContent).toBe('2050')
    await user.selectOptions(
      screen.getByLabelText('Reduction order'),
      'sequential',
    )
    expect(screen.getByLabelText('Accumulator result').textContent).toBe('2048')
    const slider = screen.getByLabelText('Selected computational step')
    slider.focus()
    // jsdom does not implement native range ArrowRight, so verify the input path and native keyboard-select alternative.
    fireEvent.change(slider, { target: { value: '1' } })
    expect(screen.getByText('Step 2 of 6: add; terms 1')).toBeInTheDocument()
    await user.selectOptions(
      screen.getByLabelText('Inspect error point (keyboard or text control)'),
      '5',
    )
    expect(
      screen.getByText('Step 6 of 6: add; terms 1, 2, 3'),
    ).toBeInTheDocument()
    expect(
      screen.getByRole('group', { name: 'Scrollable dot error plot' }),
    ).toHaveAttribute('tabindex', '0')
  })
  it('validates equal lengths and applies vectors atomically', async () => {
    const user = userEvent.setup()
    render(<DotProductLab />)
    fireEvent.change(screen.getByLabelText('Vector A'), {
      target: { value: '1, 2, 3' },
    })
    await user.click(screen.getByRole('button', { name: 'Apply vectors' }))
    expect(screen.getByRole('alert')).toHaveTextContent('equal-length')
    expect(screen.getByLabelText('Mode C result').textContent).toBe(
      '0.00000095367431640625',
    )
    fireEvent.change(screen.getByLabelText('Vector B'), {
      target: { value: '1, 1, 1' },
    })
    await user.click(screen.getByRole('button', { name: 'Apply vectors' }))
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Accumulator result').textContent).toBe('6')
  })
  it('shows nonfinite arithmetic and separates quantization loss', async () => {
    const user = userEvent.setup()
    render(<DotProductLab />)
    await user.click(
      screen.getByRole('button', { name: 'Input loss is irreversible' }),
    )
    expect(
      screen.getByLabelText('Signed error versus quantized').textContent,
    ).toBe('0')
    expect(
      screen.getByLabelText('Signed error versus original').textContent,
    ).toBe('-0.000244140625')
    fireEvent.change(screen.getByLabelText('Vector A'), {
      target: { value: 'Infinity' },
    })
    fireEvent.change(screen.getByLabelText('Vector B'), {
      target: { value: '0' },
    })
    await user.click(screen.getByRole('button', { name: 'Apply vectors' }))
    expect(screen.getByLabelText('Accumulator result').textContent).toBe('NaN')
    expect(screen.getByLabelText('Exception indicators')).toHaveTextContent(
      'invalid: raised',
    )
    expect(
      screen.getByLabelText('Exact original binary64 dot'),
    ).toHaveTextContent('Undefined')
  })
  it('exposes every preset, input encoding, and the subnormal boundary', async () => {
    const user = userEvent.setup()
    render(<DotProductLab />)
    await user.click(
      screen.getByRole('button', { name: 'Subnormal / normal boundary' }),
    )
    expect(screen.getByLabelText('Accumulator result').textContent).toBe(
      '0.00006103515625',
    )
    await user.click(screen.getByText('Quantized input encodings'))
    expect(
      screen.getByRole('group', { name: 'Quantized vector data' }),
    ).toHaveTextContent('0x03ff')
    expect(
      screen.getByRole('group', { name: 'Quantized vector data' }),
    ).toHaveTextContent('0x0001')
    expect(
      screen.getByRole('table', { name: 'All modes · sequential reduction' }),
    ).toBeInTheDocument()
  })
})
