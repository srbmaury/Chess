import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import App from './App'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

test('shows coverage when Features is built from partial analysis', async () => {
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
        return {
          ok: true,
          json: async () => ({
            active_username: 'player',
            profiles: [{ username: 'player', display_username: 'player' }],
          }),
        } as Response
      }
      if (url === '/api/pipeline/status') {
        return {
          ok: true,
          json: async () => ({
            status: 'succeeded',
            stage: 'features',
            result: {
              rows: 121537,
              analyzed_user_moves: 121537,
              total_user_moves: 211639,
              analysis_complete: false,
            },
          }),
        } as Response
      }
      throw new Error(`Unexpected fetch: ${url}`)
    }),
  )

  render(<App />)

  expect(await screen.findByText('Features built from partial analysis')).toBeTruthy()
  expect(screen.getByText(/121,537 of 211,639 analyzed user moves/)).toBeTruthy()
  expect(screen.getByText(/resume Analyze later and rebuild Features/)).toBeTruthy()
})
