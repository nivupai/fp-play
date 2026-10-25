import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { AttentionLab } from '../components/attention/AttentionLab'
import { Heatmap } from '../components/attention/Heatmap'
import { sharedHeatScale } from '../lib/attention/presentation'

describe('attention interface with actual numerical models', () => {
  it('compares accumulation policies with accessible heatmaps, encodings, and reference data', () => {
    render(<AttentionLab />)
    expect(
      screen.getByRole('heading', { name: 'Attention Precision Lab' }),
    ).toBeInTheDocument()
    const left = screen.getByRole('table', { name: 'Left QKᵀ scores' }),
      right = screen.getByRole('table', { name: 'Right QKᵀ scores' })
    expect(
      within(left).getByRole('button', {
        name: 'Left QKᵀ scores row 1 column 1: 0',
      }),
    ).toBeInTheDocument()
    expect(right).toHaveTextContent('0x3f800000')
    expect(
      screen.getByRole('table', { name: 'Stable FP32 QKᵀ scores' }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole('table', { name: 'Binary64 QKᵀ scores' }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole('table', { name: 'Left absolute error' }),
    ).toHaveTextContent('1')
    expect(
      screen.getByText(
        /correctly rounded FP16\/BF16\/FP32 exponentiation is not claimed/,
      ),
    ).toBeInTheDocument()
  })
  it('edits matrices atomically and retains the applied calculation after errors', async () => {
    const user = userEvent.setup()
    render(<AttentionLab />)
    fireEvent.change(screen.getByLabelText('Q matrix'), {
      target: { value: '1 2' },
    })
    await user.click(
      screen.getByRole('button', { name: 'Apply attention matrices' }),
    )
    expect(screen.getByRole('alert')).toHaveTextContent(
      'feature dimensions must match',
    )
    expect(
      screen.getByRole('table', { name: 'Right QKᵀ scores' }),
    ).toHaveTextContent('0x3f800000')
    fireEvent.change(screen.getByLabelText('Q matrix'), {
      target: { value: '1' },
    })
    fireEvent.change(screen.getByLabelText('K matrix'), {
      target: { value: '1;0' },
    })
    fireEvent.change(screen.getByLabelText('V matrix'), {
      target: { value: '2;0' },
    })
    await user.click(
      screen.getByRole('button', { name: 'Apply attention matrices' }),
    )
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(
      screen.getByRole('table', { name: 'Left QKᵀ scores' }),
    ).toHaveTextContent('0x3c00')
  })
  it('switches stable/naive softmax and exposes exp overflow and NaN diagnostics', async () => {
    const user = userEvent.setup()
    render(<AttentionLab />)
    const preset = screen.getByRole('button', {
      name: 'FP16 exponential overflow',
    })
    preset.focus()
    await user.keyboard('{Enter}')
    await user.selectOptions(
      screen.getByLabelText('Attention stage view'),
      'output',
    )
    expect(
      screen.getByRole('button', {
        name: 'Left Attention output row 1 column 1: NaN',
      }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole('list', { name: 'left attention diagnostics' }),
    ).toHaveTextContent('nonfinite')
    await user.selectOptions(
      screen.getByLabelText('Attention softmax method'),
      'stable',
    )
    expect(
      screen.queryByRole('button', {
        name: 'Left Attention output row 1 column 1: NaN',
      }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole('table', { name: 'Left Attention output' }),
    ).toHaveTextContent('0.73095703125')
  })
  it('supports keyboard cell selection, stage navigation, and precision controls', async () => {
    const user = userEvent.setup()
    render(<AttentionLab />)
    const cell = screen.getByRole('button', {
      name: 'Left QKᵀ scores row 1 column 2: 0',
    })
    cell.focus()
    await user.keyboard('{Enter}')
    expect(cell).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByLabelText('Attention selected bits')).toHaveTextContent(
      '0x0000',
    )
    const next = screen.getByRole('button', { name: 'Next attention step' })
    next.focus()
    await user.keyboard('{Enter}')
    expect(screen.getByText(/Attention step 2 of/)).toBeInTheDocument()
    await user.selectOptions(
      screen.getByLabelText('Left precision'),
      'bf16-wide',
    )
    expect(screen.getByText(/Attention step 1 of/)).toBeInTheDocument()
    const step = screen.getByLabelText(
      'Attention computation step',
    ) as HTMLSelectElement
    const index = [...step.options].find((o) =>
      o.textContent?.includes('normalize'),
    )!.value
    await user.selectOptions(step, index)
    expect(screen.getByLabelText('Attention stage result')).toHaveTextContent(
      'FP32',
    )
    expect(
      screen.getByRole('group', { name: 'Left QKᵀ scores scroll area' }),
    ).toHaveAttribute('tabindex', '0')
  })
  it('reveals probability loss, output differences, relative error, and verification methods', async () => {
    const user = userEvent.setup()
    render(<AttentionLab />)
    await user.click(
      screen.getByRole('button', { name: 'Probability quantization loss' }),
    )
    await user.selectOptions(
      screen.getByLabelText('Attention stage view'),
      'probabilities',
    )
    expect(
      screen.getByRole('button', {
        name: 'Left Softmax probabilities row 1 column 2: 0',
      }),
    ).toBeInTheDocument()
    expect(
      screen.getByLabelText('Attention probability row'),
    ).toHaveTextContent('not renormalized')
    await user.click(screen.getByText('Relative error heatmaps'))
    expect(
      screen.getByRole('table', { name: 'Left relative error' }),
    ).toHaveTextContent('1')
    await user.click(screen.getByText('Experiment verification method'))
    expect(
      screen.getByText(/Independent Fraction rounding/),
    ).toBeInTheDocument()
  })
  it('retains signed-zero raw input in its computation trace and supports multi-row probabilities', async () => {
    const user = userEvent.setup()
    render(<AttentionLab />)
    fireEvent.change(screen.getByLabelText('Q matrix'), {
      target: { value: '-0;1' },
    })
    fireEvent.change(screen.getByLabelText('K matrix'), {
      target: { value: '1;0' },
    })
    fireEvent.change(screen.getByLabelText('V matrix'), {
      target: { value: '1;0' },
    })
    await user.click(
      screen.getByRole('button', { name: 'Apply attention matrices' }),
    )
    expect(screen.getByLabelText('Attention stage result')).toHaveTextContent(
      '0x8000 = -0',
    )
    await user.selectOptions(
      screen.getByLabelText('Inspect probability row'),
      '1',
    )
    expect(
      screen.getByRole('heading', { name: 'Probability row 2' }),
    ).toBeInTheDocument()
  })
})
describe('heatmap text alternatives and comparison scale', () => {
  it('labels undefined/nonfinite values and keeps a shared scale independent of sign', () => {
    expect(sharedHeatScale([[1, -2, NaN, null]], [[4, Infinity]])).toBe(4)
    render(
      <Heatmap
        title="Special values"
        values={[[null, NaN, Infinity, -Infinity, -0]]}
        scale={1}
      />,
    )
    const table = screen.getByRole('table', { name: 'Special values' })
    for (const label of ['Undefined', 'NaN', 'Infinity', '-Infinity', '-0'])
      expect(within(table).getByText(label)).toBeInTheDocument()
  })
})
