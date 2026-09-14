import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

vi.mock('react-chessboard', () => ({
  Chessboard: () => null,
}))

import App from './App'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

const activeProfiles = {
  active_username: 'srbmaury',
  profiles: [{ username: 'srbmaury', display_username: 'srbmaury' }],
}

test('labels sync archive progress as months', async () => {
  window.history.pushState({}, '', '/pipeline')

  let source: FakeEventSource | null = null
  class FakeEventSource {
    onmessage: ((event: MessageEvent) => void) | null = null
    constructor(_url: string) {
      source = this
    }
    close() {}
  }
  vi.stubGlobal('EventSource', FakeEventSource)

  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: unknown) => {
      const url = String(input)
      if (url === '/api/profiles') {
        return { ok: true, json: async () => activeProfiles } as Response
      }
      if (url === '/api/pipeline/status') {
        return {
          ok: true,
          json: async () => ({ status: 'running', stage: 'sync', username: 'srbmaury' }),
        } as Response
      }
      throw new Error(`Unexpected fetch: ${url}`)
    }),
  )

  render(<App />)
  expect(await screen.findByText('Build your coach')).toBeTruthy()

  source?.onmessage?.({
    data: JSON.stringify({
      stage: 'sync',
      status: 'running',
      current: 7,
      total: 25,
      archive: 'https://api.chess.com/pub/player/srbmaury/games/2026/03',
      game_count: 321,
      skipped: false,
    }),
  } as MessageEvent)

  expect(await screen.findByText('Current month')).toBeTruthy()
  expect(screen.getByText('Total months')).toBeTruthy()
  expect(screen.getByText('7')).toBeTruthy()
  expect(screen.getByText('25')).toBeTruthy()
})

test('renders sync result fields instead of an object string', async () => {
  window.history.pushState({}, '', '/pipeline')

  class FakeEventSource {
    onmessage: ((event: MessageEvent) => void) | null = null
    constructor(_url: string) {}
    close() {}
  }
  vi.stubGlobal('EventSource', FakeEventSource)

  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: unknown) => {
      const url = String(input)
      if (url === '/api/profiles') {
        return { ok: true, json: async () => activeProfiles } as Response
      }
      if (url === '/api/pipeline/status') {
        return {
          ok: true,
          json: async () => ({
            status: 'succeeded',
            stage: 'sync',
            username: 'srbmaury',
            result: {
              downloaded: 18,
              total: 339,
              pgn_path: '/tmp/srbmaury_all_games.pgn',
            },
          }),
        } as Response
      }
      throw new Error(`Unexpected fetch: ${url}`)
    }),
  )

  render(<App />)

  expect(await screen.findByText('Sync result')).toBeTruthy()
  expect(screen.getByText('Downloaded games')).toBeTruthy()
  expect(screen.getByText('18')).toBeTruthy()
  expect(screen.getByText('Total games')).toBeTruthy()
  expect(screen.getByText('339')).toBeTruthy()
  expect(screen.queryByText('[object Object]')).toBeNull()
})
