import { fireEvent, render, screen } from '@testing-library/react'
import { expect, test, vi } from 'vitest'

vi.mock('react-chessboard', () => ({
  Chessboard: ({ options }: { options: { onPieceDrop?: (move: { sourceSquare: string; targetSquare: string }) => boolean } }) => (
    <button aria-label="Play test move" onClick={() => options.onPieceDrop?.({ sourceSquare: 'd2', targetSquare: 'd4' })}>
      Play test move
    </button>
  ),
}))

import App from './App'


test('renders dashboard metrics from the local API', async () => {
  window.history.pushState({}, '', '/')
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
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
