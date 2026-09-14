import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

vi.mock('react-chessboard', () => ({
  Chessboard: ({ options }: { options: { onPieceDrop?: (move: { sourceSquare: string; targetSquare: string }) => boolean } }) => (
    <button aria-label="Play test move" onClick={() => options.onPieceDrop?.({ sourceSquare: 'd2', targetSquare: 'd4' })}>
      Play test move
    </button>
  ),
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

function commonResponse(url: string) {
  if (url === '/api/profiles') {
    return { ok: true, json: async () => activeProfiles } as Response
  }
  if (url === '/api/pipeline/status') {
    return { ok: true, json: async () => ({ status: 'idle', stage: null }) } as Response
  }
  return null
}

test('renders dashboard metrics from the local API', async () => {
  window.history.pushState({}, '', '/')
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: unknown) => {
      const url = String(input)
      const common = commonResponse(url)
      if (common) return common
      if (url === '/api/dashboard') {
        return {
          ok: true,
          json: async () => ({
            analyzed_moves: 6434,
            training: {
              total_puzzles: 100,
              due_puzzles: 12,
              reviewed_puzzles: 40,
              mastered_puzzles: 7,
              total_reviews: 80,
              accuracy: 0.75,
            },
            artifacts: {
              analysis: { exists: true, rows: 6434 },
              features: { exists: true, rows: 6434 },
            },
          }),
        } as Response
      }
      throw new Error(`Unexpected fetch: ${url}`)
    }),
  )

  render(<App />)

  expect(await screen.findByText('Your training cockpit')).toBeTruthy()
  expect(await screen.findByText('12')).toBeTruthy()
  expect(await screen.findByText('75.0%')).toBeTruthy()
  expect(await screen.findByText('6,434')).toBeTruthy()
})

test('reveals why the best move is best only after a practice attempt', async () => {
  window.history.pushState({}, '', '/practice')
  const fetchMock = vi.fn(async (input: unknown) => {
    const url = String(input)
    const common = commonResponse(url)
    if (common) return common
    if (url.endsWith('/api/practice/next')) {
      return {
        ok: true,
        json: async () => ({
          puzzle: {
            puzzle_id: 'p1',
            fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            orientation: 'white',
            game: 'srbmaury vs opponent',
            move: '1.e4',
            opening: "Queen's Pawn Opening",
            eco: 'D00',
            phase: 'opening',
            motif: 'forcing check',
            difficulty: 3,
          },
        }),
      } as Response
    }
    if (url.includes('/attempt')) {
      return {
        ok: true,
        json: async () => ({
          correct: true,
          best_move_san: 'd4',
          best_move_uci: 'd2d4',
          your_game_move: 'e4',
          evaluation_loss_pawns: 2.5,
          next_interval_days: 3,
          next_review_at: '2026-09-17T00:00:00Z',
        }),
      } as Response
    }
    if (url.includes('/explanation')) {
      return {
        ok: true,
        json: async () => ({
          idea: 'Forcing check',
          why: 'This move gives check and keeps the initiative.',
          best_line: ['d4', 'd5', 'Nc3'],
          why_your_move_was_worse: 'e4 missed the stronger continuation and lost 2.50 pawns.',
          engine_grounded: true,
          depth: 14,
          cached: false,
        }),
      } as Response
    }
    throw new Error(`Unexpected fetch: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)

  render(<App />)

  expect(await screen.findByText('White to move')).toBeTruthy()
  expect(screen.queryByText('Why is this best?')).toBeNull()

  fireEvent.click(screen.getByRole('button', { name: 'Play test move' }))

  expect(await screen.findByText('Best move:')).toBeTruthy()
  const whyButton = await screen.findByRole('button', { name: 'Why is this best?' })
  fireEvent.click(whyButton)

  expect(await screen.findByText('Forcing check')).toBeTruthy()
  expect(await screen.findByText('d4 d5 Nc3')).toBeTruthy()
  expect(await screen.findByText(/lost 2.50 pawns/)).toBeTruthy()
  expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/explanation'))).toBe(true)
})

test('requires a Chess.com username before loading a fresh community workspace', async () => {
  window.history.pushState({}, '', '/')
  const fetchMock = vi.fn(async (input: unknown) => {
    const url = String(input)
    if (url === '/api/profiles') {
      return { ok: true, json: async () => ({ active_username: null, profiles: [] }) } as Response
    }
    if (url === '/api/pipeline/status') {
      return { ok: true, json: async () => ({ status: 'idle', stage: null }) } as Response
    }
    throw new Error(`Unexpected fetch: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)

  render(<App />)

  expect(await screen.findByText('Choose your Chess.com player')).toBeTruthy()
  expect(screen.getByRole('textbox', { name: 'Chess.com username' })).toBeTruthy()
  expect(screen.queryByText('Your training cockpit')).toBeNull()
  expect(fetchMock.mock.calls.some(([url]) => String(url) === '/api/dashboard')).toBe(false)
})

test('shows Stop inside Pipeline while Stockfish analyze is running', async () => {
  window.history.pushState({}, '', '/pipeline')
  class FakeEventSource {
    onmessage: ((event: MessageEvent) => void) | null = null
    constructor(_url: string) {}
    close() {}
  }
  vi.stubGlobal('EventSource', FakeEventSource)

  const fetchMock = vi.fn(async (input: unknown, init?: RequestInit) => {
    const url = String(input)
    if (url === '/api/profiles') {
      return { ok: true, json: async () => activeProfiles } as Response
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

  render(<App />)

  const stop = await screen.findByRole('button', { name: 'Stop' })
  fireEvent.click(stop)

  await waitFor(() => expect(fetchMock.mock.calls.some(([url, init]) => String(url) === '/api/pipeline/stop' && (init as RequestInit | undefined)?.method === 'POST')).toBe(true))
})
