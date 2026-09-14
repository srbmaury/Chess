import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

vi.mock('react-chessboard', () => ({ Chessboard: () => <div /> }))

import App from './App'
import CommunityControls from './CommunityControls'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

const profiles = {
  active_username: 'srbmaury',
  profiles: [
    { username: 'srbmaury', display_username: 'srbmaury' },
    { username: 'other_player', display_username: 'Other_Player' },
  ],
}

test('player controls are disabled while any pipeline stage is running', async () => {
  vi.stubGlobal('fetch', vi.fn(async (input: unknown) => {
    const url = String(input)
    if (url === '/api/profiles') {
      return { ok: true, json: async () => profiles } as Response
    }
    if (url === '/api/pipeline/status') {
      return {
        ok: true,
        json: async () => ({ status: 'running', stage: 'sync', username: 'srbmaury' }),
      } as Response
    }
    throw new Error(`Unexpected fetch: ${url}`)
  }))

  render(<CommunityControls onProfileChanged={() => undefined} />)

  const selector = await screen.findByRole('combobox', { name: 'Active Chess.com player' })
  const input = screen.getByRole('textbox', { name: 'Chess.com username' })
  const add = screen.getByRole('button', { name: 'Add player' })

  await waitFor(() => {
    expect((selector as HTMLSelectElement).disabled).toBe(true)
    expect((input as HTMLInputElement).disabled).toBe(true)
    expect((add as HTMLButtonElement).disabled).toBe(true)
  })
})

test('pipeline stream closes on terminal state and reconnects for a new run', async () => {
  window.history.pushState({}, '', '/pipeline')

  class FakeEventSource {
    static instances: FakeEventSource[] = []
    onmessage: ((event: MessageEvent) => void) | null = null
    closed = false

    constructor(_url: string) {
      FakeEventSource.instances.push(this)
    }

    close() {
      this.closed = true
    }
  }
  vi.stubGlobal('EventSource', FakeEventSource)

  const fetchMock = vi.fn(async (input: unknown, init?: RequestInit) => {
    const url = String(input)
    if (url === '/api/profiles') {
      return { ok: true, json: async () => profiles } as Response
    }
    if (url === '/api/pipeline/status') {
      return { ok: true, json: async () => ({ status: 'idle', stage: null }) } as Response
    }
    if (url === '/api/pipeline/sync' && init?.method === 'POST') {
      return {
        ok: true,
        json: async () => ({ status: 'running', stage: 'sync', username: 'srbmaury' }),
      } as Response
    }
    throw new Error(`Unexpected fetch: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)

  render(<App />)
  expect(await screen.findByText('Build your coach')).toBeTruthy()
  await waitFor(() => expect(FakeEventSource.instances.length).toBe(1))

  const first = FakeEventSource.instances[0]
  await act(async () => {
    first.onmessage?.({
      data: JSON.stringify({ status: 'succeeded', stage: 'analyze', sequence: 3 }),
    } as MessageEvent)
  })
  expect(first.closed).toBe(true)

  fireEvent.click(screen.getAllByRole('button', { name: 'Run' })[0])
  await waitFor(() => expect(FakeEventSource.instances.length).toBe(2))
})
