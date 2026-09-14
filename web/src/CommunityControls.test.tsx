import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { expect, test, vi } from 'vitest'

import CommunityControls from './CommunityControls'


test('switches and adds isolated Chess.com players', async () => {
  const changed = vi.fn()
  const fetchMock = vi.fn(async (input: unknown, init?: RequestInit) => {
    const url = String(input)
    if (url === '/api/profiles' && (!init || init.method !== 'POST')) {
      return {
        ok: true,
        json: async () => ({
          active_username: 'srbmaury',
          profiles: [
            { username: 'srbmaury', display_username: 'srbmaury' },
            { username: 'other_player', display_username: 'Other_Player' },
          ],
        }),
      } as Response
    }
    if (url === '/api/pipeline/status') {
      return { ok: true, json: async () => ({ status: 'idle', stage: null }) } as Response
    }
    if (url === '/api/profiles/other_player/activate') {
      return {
        ok: true,
        json: async () => ({
          active_username: 'other_player',
          profiles: [
            { username: 'srbmaury', display_username: 'srbmaury' },
            { username: 'other_player', display_username: 'Other_Player' },
          ],
        }),
      } as Response
    }
    if (url === '/api/profiles' && init?.method === 'POST') {
      return { ok: true, json: async () => ({ username: 'new_player', active_username: 'new_player' }) } as Response
    }
    throw new Error(`Unexpected fetch: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)

  render(<CommunityControls onProfileChanged={changed} />)

  const selector = await screen.findByRole('combobox', { name: 'Active Chess.com player' })
  fireEvent.change(selector, { target: { value: 'other_player' } })
  await waitFor(() => expect(changed).toHaveBeenCalledTimes(1))

  fireEvent.change(screen.getByRole('textbox', { name: 'Chess.com username' }), { target: { value: 'new_player' } })
  fireEvent.click(screen.getByRole('button', { name: 'Add player' }))
  await waitFor(() => expect(changed).toHaveBeenCalledTimes(2))
  expect(fetchMock.mock.calls.some(([url, init]) => String(url) === '/api/profiles' && (init as RequestInit | undefined)?.method === 'POST')).toBe(true)
})


test('shows a safe stop control while Stockfish analysis is running', async () => {
  const fetchMock = vi.fn(async (input: unknown, init?: RequestInit) => {
    const url = String(input)
    if (url === '/api/profiles') {
      return {
        ok: true,
        json: async () => ({ active_username: 'srbmaury', profiles: [{ username: 'srbmaury', display_username: 'srbmaury' }] }),
      } as Response
    }
    if (url === '/api/pipeline/status') {
      return { ok: true, json: async () => ({ status: 'running', stage: 'analyze', username: 'srbmaury' }) } as Response
    }
    if (url === '/api/pipeline/stop' && init?.method === 'POST') {
      return { ok: true, json: async () => ({ status: 'stopping', stage: 'analyze', username: 'srbmaury' }) } as Response
    }
    throw new Error(`Unexpected fetch: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)

  render(<CommunityControls onProfileChanged={() => undefined} />)

  const stop = await screen.findByRole('button', { name: 'Stop analysis' })
  fireEvent.click(stop)

  expect(await screen.findByRole('button', { name: 'Stopping analysis…' })).toBeTruthy()
  expect(fetchMock.mock.calls.some(([url, init]) => String(url) === '/api/pipeline/stop' && (init as RequestInit | undefined)?.method === 'POST')).toBe(true)
})
