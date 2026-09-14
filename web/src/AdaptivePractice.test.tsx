import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

vi.mock('react-chessboard', () => ({
  Chessboard: ({ options }: { options: { position: string; onPieceDrop?: (move: { sourceSquare: string; targetSquare: string }) => boolean } }) => (
    <div>
      <span data-testid="board-position">{options.position}</span>
      <button aria-label="Play d4" onClick={() => options.onPieceDrop?.({ sourceSquare: 'd2', targetSquare: 'd4' })}>Play d4</button>
    </div>
  ),
}))

import LegacyApp from './LegacyApp'

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
const AFTER_REPLY = 'rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2'

const puzzle = {
  puzzle_id: 'p1',
  fen: START,
  orientation: 'white',
  game: 'srbmaury vs opponent',
  move: '1.e4',
  opening: "Queen's Pawn Opening",
  eco: 'D00',
  phase: 'opening',
  motif: 'positional / calculation',
  difficulty: 3,
}

const activeSession = {
  session_id: 's1',
  puzzle_id: 'p1',
  status: 'active',
  current_fen: START,
  orientation: 'white',
  user_moves_attempted: 0,
  user_moves_accepted: 0,
  current_ply: 0,
  max_eval_loss_cp: 0,
  max_user_decisions: 4,
  steps: [],
  review: null,
}

function response(body: unknown, ok = true) {
  return { ok, json: async () => body } as Response
}

function baseFetch(extra: (url: string, init?: RequestInit) => Response | null | Promise<Response | null>) {
  return vi.fn(async (input: unknown, init?: RequestInit) => {
    const url = String(input)
    if (url === '/api/practice/next') return response({ puzzle })
    const result = await extra(url, init)
    if (result) return result
    throw new Error(`Unexpected fetch: ${url}`)
  })
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  window.history.pushState({}, '', '/')
})

test('Adaptive is the default practice mode and resumes the server session', async () => {
  window.history.pushState({}, '', '/practice')
  const fetchMock = baseFetch((url) => {
    if (url === '/api/practice/p1/adaptive/start') return response(activeSession)
    return null
  })
  vi.stubGlobal('fetch', fetchMock)

  render(<LegacyApp />)

  expect(await screen.findByText('White to move')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Adaptive' }).getAttribute('aria-pressed')).toBe('true')
  expect(screen.getByTestId('board-position').textContent).toBe(START)
  expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith('/adaptive/start'))).toBe(true)
})

test('accepted adaptive move applies the engine reply and continues', async () => {
  window.history.pushState({}, '', '/practice')
  const fetchMock = baseFetch((url) => {
    if (url === '/api/practice/p1/adaptive/start') return response(activeSession)
    if (url === '/api/practice/adaptive/s1/move') {
      return response({
        ...activeSession,
        status: 'active',
        accepted: true,
        move_uci: 'd2d4',
        move_san: 'd4',
        eval_loss_cp: 0,
        engine_reply_uci: 'd7d5',
        engine_reply_san: 'd5',
        current_fen: AFTER_REPLY,
        user_moves_attempted: 1,
        user_moves_accepted: 1,
        current_ply: 2,
      })
    }
    return null
  })
  vi.stubGlobal('fetch', fetchMock)

  render(<LegacyApp />)
  await screen.findByText('White to move')
  fireEvent.click(screen.getByRole('button', { name: 'Play d4' }))

  expect(await screen.findByText('Strong. Continuing the line…')).toBeTruthy()
  expect(screen.getByText(/Engine replied d5/)).toBeTruthy()
  expect(screen.getByTestId('board-position').textContent).toBe(AFTER_REPLY)
  expect(screen.getByText(/1 \/ up to 4 decisions/)).toBeTruthy()
})

test('terminal adaptive sequence shows conversion result and explanation action', async () => {
  window.history.pushState({}, '', '/practice')
  const fetchMock = baseFetch((url) => {
    if (url === '/api/practice/p1/adaptive/start') return response(activeSession)
    if (url === '/api/practice/adaptive/s1/move') {
      return response({
        ...activeSession,
        status: 'succeeded',
        accepted: true,
        move_uci: 'd2d4',
        move_san: 'd4',
        eval_loss_cp: 0,
        engine_reply_uci: null,
        engine_reply_san: null,
        current_fen: AFTER_REPLY,
        user_moves_attempted: 2,
        user_moves_accepted: 2,
        current_ply: 3,
        review: {
          next_interval_days: 3,
          next_review_at: '2026-09-17T00:00:00Z',
          consecutive_correct: 1,
          mastered: false,
        },
      })
    }
    return null
  })
  vi.stubGlobal('fetch', fetchMock)

  render(<LegacyApp />)
  await screen.findByText('White to move')
  fireEvent.click(screen.getByRole('button', { name: 'Play d4' }))

  expect(await screen.findByText('Converted')).toBeTruthy()
  expect(screen.getByText(/2\/2 strong decisions/)).toBeTruthy()
  expect(screen.getByText(/Calculation depth: 3 plies/)).toBeTruthy()
  expect(screen.getByText('Next review: +3 days')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Why is this best?' })).toBeTruthy()
})

test('switching to Quick abandons an active adaptive drill and uses one-move attempt', async () => {
  window.history.pushState({}, '', '/practice')
  const fetchMock = baseFetch((url) => {
    if (url === '/api/practice/p1/adaptive/start') return response(activeSession)
    if (url === '/api/practice/adaptive/s1/abandon') return response({ ...activeSession, status: 'abandoned' })
    if (url === '/api/practice/p1/attempt') {
      return response({
        correct: true,
        best_move_san: 'd4',
        best_move_uci: 'd2d4',
        your_game_move: 'e4',
        evaluation_loss_pawns: 2.5,
        next_interval_days: 3,
        next_review_at: '2026-09-17T00:00:00Z',
      })
    }
    return null
  })
  vi.stubGlobal('fetch', fetchMock)

  render(<LegacyApp />)
  await screen.findByText('White to move')
  fireEvent.click(screen.getByRole('button', { name: 'Quick' }))
  await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith('/adaptive/s1/abandon'))).toBe(true))
  fireEvent.click(screen.getByRole('button', { name: 'Play d4' }))

  expect(await screen.findByText('Best move:')).toBeTruthy()
  expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith('/practice/p1/attempt'))).toBe(true)
})
