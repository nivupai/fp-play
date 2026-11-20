import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { FP16Explorer } from '../components/FP16Explorer'
import { decodeFP16, floatToFP16Bits, fp16BitsToHex } from '../lib/numerics'
import { FP16_PRESETS } from '../lib/explorer/presets'
import { formatExactFP16 } from '../lib/explorer/presentation'

function setup() {
  const user = userEvent.setup()
  render(<FP16Explorer />)
  return {
    user,
    input: screen.getByRole('textbox', { name: 'Decimal / special value' }),
    convert: async (value: string) => {
      const input = screen.getByRole('textbox', {
        name: 'Decimal / special value',
      })
      await user.clear(input)
      if (value) await user.type(input, value)
      await user.click(screen.getByRole('button', { name: 'Convert to FP16' }))
    },
  }
}

function expectEncoding(bits: number) {
  expect(screen.getByLabelText('Hexadecimal')).toHaveTextContent(
    fp16BitsToHex(bits),
  )
  expect(screen.getByLabelText('Exact decoded value').textContent).toBe(
    formatExactFP16(decodeFP16(bits).value),
  )
  expect(screen.getByLabelText('Classification')).toHaveTextContent(
    decodeFP16(bits).classification,
  )
}

describe('FP16 Explorer integration with the numerical engine', () => {
  it('shows all sixteen named, pressed-state bit switches and the decoded breakdown', () => {
    setup()
    expect(
      screen.getAllByRole('button', {
        name: /^[01], (Sign|Exponent|Fraction) bit \d+$/,
      }),
    ).toHaveLength(16)
    expect(
      screen.getByRole('group', { name: 'Sign 1 bit' }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole('group', { name: 'Exponent 5 bits' }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole('group', { name: 'Fraction 10 bits' }),
    ).toBeInTheDocument()
    expectEncoding(0x3e00)
    expect(screen.getByLabelText('Raw exponent')).toHaveTextContent(
      '01111₂ = 15',
    )
    expect(screen.getByLabelText('Exponent bias')).toHaveTextContent('15')
    expect(screen.getByLabelText('Unbiased exponent')).toHaveTextContent('0')
    expect(screen.getByLabelText('Fraction')).toHaveTextContent(
      '1000000000₂ = 512',
    )
    expect(screen.getByLabelText('Implicit leading bit')).toHaveTextContent(
      '1 (present)',
    )
  })

  it('updates the value and sign immediately when bits change', async () => {
    const { user } = setup()
    await user.click(
      screen.getByRole('button', { name: /^[01], Fraction bit 9$/ }),
    )
    expectEncoding(0x3c00)
    await user.click(
      screen.getByRole('button', { name: /^[01], Sign bit 15$/ }),
    )
    expectEncoding(0xbc00)
    expect(screen.getByLabelText('Sign bit')).toHaveTextContent('1 (negative)')
    expect(
      screen.getByRole('status', { name: 'FP16 selection' }),
    ).toHaveTextContent('0xbc00 · normal · -1')
    expect(
      screen.getByRole('button', { name: /^[01], Sign bit 15$/ }),
    ).toHaveAttribute('aria-pressed', 'true')
  })

  it('supports Tab, Space, and Enter on native bit buttons without losing focus', async () => {
    const { user } = setup()
    await user.tab()
    expect(screen.getByRole('textbox')).toHaveFocus()
    await user.tab()
    expect(
      screen.getByRole('button', { name: 'Convert to FP16' }),
    ).toHaveFocus()
    await user.tab()
    const sign = screen.getByRole('button', { name: /^[01], Sign bit 15$/ })
    expect(sign).toHaveFocus()
    await user.keyboard(' ')
    expectEncoding(0xbe00)
    expect(sign).toHaveFocus()
    await user.keyboard('{Enter}')
    expectEncoding(0x3e00)
    await user.tab()
    expect(
      screen.getByRole('button', { name: /^[01], Exponent bit 14$/ }),
    ).toHaveFocus()
    await user.keyboard('{Enter}')
    expectEncoding(0x7e00)
  })

  it('quantizes decimal input while showing the original binary64 value separately', async () => {
    const { convert } = setup()
    await convert('0.1')
    expectEncoding(floatToFP16Bits(0.1))
    expect(screen.getByLabelText('Original input').textContent).toBe('0.1')
    expect(screen.getByLabelText('Quantized FP16 value').textContent).toBe(
      '0.0999755859375',
    )
    expect(
      screen.getByText('Rounded · nearest, ties to even'),
    ).toBeInTheDocument()
  })

  it('supports scientific notation, exact values, and submission with Enter', async () => {
    const { user, input } = setup()
    await user.clear(input)
    await user.type(input, '1.5e0{Enter}')
    expectEncoding(0x3e00)
    expect(
      screen.getByText('Exactly representable in FP16'),
    ).toBeInTheDocument()
  })

  it('preserves negative zero in both the original input and the decoded value', async () => {
    const { convert } = setup()
    await convert('-0')
    expectEncoding(0x8000)
    expect(screen.getByLabelText('Original input').textContent).toBe('-0')
    expect(screen.getByLabelText('Quantized FP16 value').textContent).toBe('-0')
    expect(screen.getByLabelText('Value explanation')).toHaveTextContent(
      'negative zero',
    )
    expect(
      screen.getByLabelText('Previous representable value').textContent,
    ).toBe('-0.000000059604644775390625')
    expect(screen.getByLabelText('Next representable value').textContent).toBe(
      '0.000000059604644775390625',
    )
    expect(
      screen.getByText(/Numerical stepping skips the other zero encoding/),
    ).toBeInTheDocument()
  })

  it.each([
    ['Positive zero', 0x0000],
    ['Negative zero', 0x8000],
    ['1.0', 0x3c00],
    ['1.5', 0x3e00],
    ['Smallest positive subnormal', 0x0001],
    ['Largest positive subnormal', 0x03ff],
    ['Smallest positive normal', 0x0400],
    ['Maximum finite positive', 0x7bff],
    ['Positive infinity', 0x7c00],
    ['Quiet NaN · payload 1', 0x7e01],
  ] as const)('selects the explicit %s preset', async (label, bits) => {
    const { user } = setup()
    const preset = within(
      screen.getByRole('region', { name: 'Explore the edge cases' }),
    ).getByRole('button', {
      name: new RegExp(`^${label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`),
    })
    await user.click(preset)
    expectEncoding(bits)
    expect(preset).toHaveAttribute('aria-pressed', 'true')
    expect(FP16_PRESETS.find((entry) => entry.label === label)?.bits).toBe(bits)
  })

  it('crosses the subnormal/normal boundary through bit edits and neighboring values', async () => {
    const { user } = setup()
    await user.click(
      screen.getByRole('button', { name: /^Largest positive subnormal/ }),
    )
    expect(screen.getByLabelText('Implicit leading bit')).toHaveTextContent(
      '0 (no implicit 1)',
    )
    expect(screen.getByLabelText('Value explanation')).toHaveTextContent(
      'No implicit leading 1',
    )
    await user.click(screen.getByRole('button', { name: 'Select next' }))
    expectEncoding(0x0400)
    expect(screen.getByLabelText('Unbiased exponent')).toHaveTextContent('-14')
    await user.click(
      screen.getByRole('button', { name: /^[01], Exponent bit 10$/ }),
    )
    expectEncoding(0x0000)
    await user.click(
      screen.getByRole('button', { name: /^[01], Fraction bit 0$/ }),
    )
    expectEncoding(0x0001)
    expect(screen.getByLabelText('Exact decoded value').textContent).toBe(
      '0.000000059604644775390625',
    )
  })

  it('shows neighbors and binade ULP at 1, then steps numerically', async () => {
    const { user } = setup()
    await user.click(screen.getByRole('button', { name: /^1\.0 / }))
    expect(
      screen.getByLabelText('Previous representable value').textContent,
    ).toBe('0.99951171875')
    expect(
      screen.getByLabelText('Current representable value').textContent,
    ).toBe('1')
    expect(screen.getByLabelText('Next representable value').textContent).toBe(
      '1.0009765625',
    )
    expect(screen.getByLabelText('ULP value').textContent).toBe('0.0009765625')
    await user.click(screen.getByRole('button', { name: 'Select previous' }))
    expectEncoding(0x3bff)
  })

  it('uses numerical rather than raw unsigned adjacency for negative values', async () => {
    const { convert, user } = setup()
    await convert('-1')
    expect(screen.getByLabelText('Next representable value').textContent).toBe(
      '-0.99951171875',
    )
    expect(
      screen.getByLabelText('Previous representable value').textContent,
    ).toBe('-1.0009765625')
    await user.click(screen.getByRole('button', { name: 'Select next' }))
    expectEncoding(0xbbff)
    expect(screen.queryByLabelText('Original input')).not.toBeInTheDocument()
  })

  it.each([
    ['Infinity', 0x7c00],
    ['-Infinity', 0xfc00],
    ['NaN', 0x7e00],
    ['infinity', 0x7c00],
  ] as const)(
    'accepts %s with appropriate special-value explanations',
    async (input, bits) => {
      const { convert } = setup()
      await convert(input)
      expectEncoding(bits)
      expect(screen.getByLabelText('Unbiased exponent')).toHaveTextContent(
        'Not applicable',
      )
      expect(screen.getByLabelText('Implicit leading bit')).toHaveTextContent(
        'Not applicable',
      )
      if (input === 'NaN') {
        expect(
          screen.getByText('Canonical quiet NaN · 0x7e00'),
        ).toBeInTheDocument()
        expect(
          screen.getByLabelText('Next representable value'),
        ).toHaveTextContent('Not ordered')
        expect(
          screen.getByRole('button', { name: 'Select next' }),
        ).toBeDisabled()
        expect(
          screen.getByRole('button', { name: 'Select previous' }),
        ).toBeDisabled()
        expect(screen.getByLabelText('ULP value')).toHaveTextContent('NaN')
      } else {
        expect(screen.getByLabelText('ULP value')).toHaveTextContent('Infinity')
        expect(
          screen.getByText(/Infinity has no finite ULP/),
        ).toBeInTheDocument()
      }
    },
  )

  it('preserves a NaN payload during raw edits and canonicalizes only on decimal conversion', async () => {
    const { user, convert } = setup()
    await user.click(
      screen.getByRole('button', { name: /^Quiet NaN · payload 1/ }),
    )
    expect(screen.getByLabelText('Value explanation')).toHaveTextContent(
      'Quiet NaN · payload 1',
    )
    await user.click(
      screen.getByRole('button', { name: /^[01], Fraction bit 9$/ }),
    )
    expectEncoding(0x7c01)
    expect(screen.getByLabelText('Value explanation')).toHaveTextContent(
      'Signaling NaN · payload 1',
    )
    await user.click(
      screen.getByRole('button', { name: /^[01], Sign bit 15$/ }),
    )
    expectEncoding(0xfc01)
    expect(screen.getByLabelText('Value explanation')).toHaveTextContent(
      'Raw sign 1',
    )
    await convert('NaN')
    expectEncoding(0x7e00)
  })

  it('handles exact ties, underflow, and overflow using the actual encoder', async () => {
    const { convert, user } = setup()
    await convert('1.00048828125')
    expectEncoding(0x3c00)
    await convert('-1e-20')
    expectEncoding(0x8000)
    await convert('65520')
    expectEncoding(0x7c00)
    expect(screen.getByRole('button', { name: 'Select next' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Select previous' }))
    expectEncoding(0x7bff)
  })

  it.each(['', 'hello', '1.2.3', '0x3c00', '1e', '1_000', '1e999'])(
    'rejects invalid input %s without replacing the current pattern',
    async (input) => {
      const { convert } = setup()
      await convert(input)
      expect(screen.getByRole('alert')).toBeInTheDocument()
      expect(screen.getByRole('textbox')).toHaveAttribute(
        'aria-invalid',
        'true',
      )
      expectEncoding(0x3e00)
    },
  )

  it('clears stale errors and conversion comparisons after selecting a raw pattern', async () => {
    const { convert, user, input } = setup()
    await convert('0.1')
    await user.clear(input)
    await user.type(input, 'bad')
    await user.click(screen.getByRole('button', { name: 'Convert to FP16' }))
    expect(screen.getByRole('alert')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /^Negative zero/ }))
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Original input')).not.toBeInTheDocument()
    expect(input).toHaveValue('-0')
    expect(input).toHaveAttribute('aria-invalid', 'false')
    expect(screen.getByText('Raw encoding selected')).toBeInTheDocument()
  })
})
