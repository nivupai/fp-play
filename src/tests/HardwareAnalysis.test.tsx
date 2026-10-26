import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import artifact from '../../rtl/vectors/hardware-analysis.json'
import { HardwareAnalysis } from '../components/hardware/HardwareAnalysis'
import { HardwareAnalysisEntry } from '../components/hardware/HardwareAnalysisEntry'
import type { HardwareArtifact } from '../lib/hardware/types'
describe('recorded hardware analysis', () => {
  it('loads the dataset on request and follows the hardware hash accessibly', async () => {
    window.history.replaceState(null, '', '#main')
    render(<HardwareAnalysisEntry />)
    expect(
      screen.queryByRole('region', { name: 'Scrollable hardware comparison' }),
    ).not.toBeInTheDocument()
    act(() => {
      window.history.replaceState(null, '', '#hardware')
      fireEvent(window, new Event('hashchange'))
    })
    expect(
      await screen.findByRole('region', {
        name: 'Scrollable hardware comparison',
      }),
    ).toBeInTheDocument()
    expect(document.getElementById('hardware')).toHaveAttribute('open')
    window.history.replaceState(null, '', '#main')
  })
  it('displays genuine mapped comparisons and numerical verification status', () => {
    render(<HardwareAnalysis />)
    const comparison = screen.getByRole('region', {
      name: 'Scrollable hardware comparison',
    })
    expect(within(comparison).getAllByRole('row')).toHaveLength(8)
    expect(within(comparison).getByText('1,685.908')).toBeInTheDocument()
    expect(
      screen.getByText(/your browser does not run synthesis/),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Mapped cell area is not chip area/),
    ).toBeInTheDocument()
  })
  it('selects a verified alternative using native keyboard-accessible controls', async () => {
    const user = userEvent.setup()
    render(<HardwareAnalysis />)
    const select = screen.getByLabelText('Inspect hardware experiment')
    select.focus()
    expect(select).toHaveFocus()
    await user.selectOptions(select, 'tree-leading-bit-fma')
    const details = screen.getByRole('region', {
      name: 'Selected hardware experiment',
    })
    expect(within(details).getByRole('heading')).toHaveTextContent(
      'fma / tree-leading-bit',
    )
    expect(within(details).getByText(/301,544 cases/)).toBeInTheDocument()
    await user.click(
      screen.getByRole('button', { name: 'nonfused / baseline' }),
    )
    expect(select).toHaveValue('baseline-nonfused')
  })
  it('explains unavailable timing without inventing chart points', () => {
    const unavailable: HardwareArtifact = {
      ...artifact,
      records: artifact.records.map((r) => ({
        ...r,
        timing: null,
        timingUnavailable: 'OpenSTA executable unavailable',
      })),
    }
    render(<HardwareAnalysis artifact={unavailable} />)
    expect(
      screen.getByText(/No area-delay points are fabricated/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole('img', { name: /Area versus delay/ }),
    ).not.toBeInTheDocument()
    expect(screen.getAllByText('Unavailable')).toHaveLength(7)
  })
  it('mounts the long recorded timing path only when requested', async () => {
    const user = userEvent.setup()
    render(<HardwareAnalysis />)
    const region = screen.getByRole('region', {
      name: 'Selected hardware experiment',
    })
    expect(within(region).queryByRole('list')).not.toBeInTheDocument()
    await user.click(
      within(region).getByText('Inspect reported critical-path cell sequence'),
    )
    const list = await within(region).findByRole('list')
    expect(within(list).getAllByRole('listitem')).toHaveLength(
      artifact.records[0].timing!.cells.length,
    )
  })
  it('provides an explicit empty measurement state', () => {
    render(<HardwareAnalysis artifact={{ ...artifact, records: [] }} />)
    expect(
      screen.getByText(/No synthesis measurements are available/),
    ).toBeInTheDocument()
    expect(
      screen.queryByLabelText('Inspect hardware experiment'),
    ).not.toBeInTheDocument()
  })
})
