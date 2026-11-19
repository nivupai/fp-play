import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import { CollectivesEntry } from '../components/collectives/CollectivesEntry'
import { CollectivesWorkspace } from '../components/collectives/CollectivesWorkspace'

afterEach(() => window.history.replaceState(null, '', '#main'))

describe('collectives workspace', () => {
  it('loads only on request, including by direct hash navigation', async () => {
    render(<CollectivesEntry />)
    expect(screen.queryByLabelText('Schedule')).not.toBeInTheDocument()
    act(() => {
      window.history.replaceState(null, '', '#collectives')
      fireEvent(window, new Event('hashchange'))
    })
    expect(await screen.findByLabelText('Schedule')).toHaveValue('ring')
  })

  it('steps through rounds with buttons and arrow keys', async () => {
    const user = userEvent.setup()
    render(<CollectivesWorkspace />)
    const status = () => screen.getByText(/^Round \d+ of \d+/)
    expect(status()).toHaveTextContent('Round 1 of 6')
    await user.click(screen.getByRole('button', { name: 'Next round' }))
    expect(status()).toHaveTextContent('Round 2 of 6')
    const group = screen.getByRole('group', { name: 'Round controls' })
    fireEvent.keyDown(group, { key: 'End' })
    expect(status()).toHaveTextContent('Round 6 of 6')
    expect(screen.getByRole('button', { name: 'Next round' })).toBeDisabled()
    fireEvent.keyDown(group, { key: 'Home' })
    expect(status()).toHaveTextContent('Round 1 of 6')
    const current = screen
      .getAllByRole('columnheader')
      .filter((h) => h.getAttribute('aria-current') === 'step')
    expect(current).toHaveLength(1)
    expect(current[0]).toHaveTextContent('Round 1 (current)')
  })

  it('shows the recorded association of the chosen element and schedule', async () => {
    const user = userEvent.setup()
    render(<CollectivesWorkspace />)
    await user.selectOptions(screen.getByLabelText('Ranks (P)'), '3')
    expect(screen.getByLabelText('Schedule')).toHaveValue('ring')
    expect(
      screen.getByRole('img', { name: 'Association tree: (x2 + (x1 + x0))' }),
    ).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Element'), '7')
    expect(
      screen.getByRole('img', { name: 'Association tree: (x1 + (x0 + x2))' }),
    ).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Schedule'), 'exact-gather')
    expect(screen.getByText('exact(x0, x1, x2)')).toBeInTheDocument()
  })

  it('pairs the error chart with an exact table', async () => {
    const user = userEvent.setup()
    render(<CollectivesWorkspace />)
    await user.selectOptions(screen.getByLabelText('Wire format'), 'fp32')
    const table = screen.getByRole('table', { name: /fp32 wire/ })
    // 5 schedules at P = 2, 4 and 8; 4 at P = 3 (no halving-doubling).
    expect(within(table).getAllByRole('row')).toHaveLength(1 + 19)
  })
})
