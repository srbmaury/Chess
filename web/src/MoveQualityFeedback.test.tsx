import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

vi.mock('react-chessboard', () => ({
  Chessboard: ({ options }: { options: { onPieceDrop?: (move: { sourceSquare: string; targetSquare: string }) => boolean } }) => (
    <button aria-label="Play d4" onClick={() => options.onPieceDrop?.({ sourceSquare: 'd2', targetSquare: 'd4' })}>
      Play d4
    </button>
  ),
}))

import LegacyApp from './LegacyApp'

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

function response(body: unknown) {
  return { ok: true, json: async () => body } as Response
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  window.history.pushState({}, '', '/')
})

test('quick feedback shows a mate-aware label instead of a fake pawn loss', async () => {
  window.history.pushState({}, '', '/practice')
  vi.stubGlobal('fetch', vi.fn(async (input: unknown) => {
    const url = String(input)
    if (url === '/api/practice/next') {
      return response({
        puzzle: {
          puzzle_id: 'p1',
          fen: START,
          orientation: 'white',
          game: 'player vs opponent',
          move: '1.e4',
          opening: 'Opening',
          eco: 'A00',
          phase: 'opening',
          motif: 'missed mate',
          difficulty: 5,
        },
      })
    }
    if (url === '/api/practice/p1/adaptive/start') {
      return response({
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
      })
    }
    if (url === '/api/practice/adaptive/s1/abandon') {
      return response({ status: 'abandoned' })
    }
    if (url === '/api/practice/p1/attempt') {
      return response({
        correct: true,
        best_move_san: 'd4',
        best_move_uci: 'd2d4',
        your_game_move: 'e4',
        evaluation_loss_pawns: null,
        quality: 'miss',
        quality_reason: 'forced mate was available',
        next_interval_days: 3,
        next_review_at: '2026-09-17T00:00:00Z',
      })
    }
    throw new Error(`Unexpected fetch: ${url}`)
  }))

  render(<LegacyApp />)
  await screen.findByText('White to move')
  fireEvent.click(screen.getByRole('button', { name: 'Quick' }))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Quick' }).getAttribute('aria-pressed')).toBe('true'))

  fireEvent.click(screen.getByRole('button', { name: 'Play d4' }))

  expect(await screen.findByText('Miss')).toBeTruthy()
  expect(await screen.findByText('Forced mate was available')).toBeTruthy()
  expect(screen.queryByText(/998\.08/)).toBeNull()
  expect(screen.queryByText(/Evaluation loss:/)).toBeNull()
})

test('mistakes list uses the quality reason when pawn loss is unavailable', async () => {
  window.history.pushState({}, '', '/mistakes')
  vi.stubGlobal('fetch', vi.fn(async (input: unknown) => {
    const url = String(input)
    if (url === '/api/puzzles') {
      return response({
        items: [{
          puzzle_id: 'p1',
          fen: START,
          orientation: 'white',
          game: 'player vs opponent',
          move: '1.e4',
          your_move_san: 'e4',
          your_move_uci: 'e2e4',
          best_move_san: 'd4',
          best_move_uci: 'd2d4',
          evaluation_loss_pawns: null,
          quality: 'miss',
          quality_reason: 'forced mate was available',
          opening: 'Opening',
          eco: 'A00',
          phase: 'opening',
          motif: 'missed mate',
          difficulty: 5,
          attempts: 0,
          correct_attempts: 0,
          accuracy: null,
          consecutive_correct: 0,
          next_review_at: '2026-09-14T00:00:00Z',
          mastered: false,
        }],
        total: 1,
        limit: 50,
        offset: 0,
      })
    }
    throw new Error(`Unexpected fetch: ${url}`)
  }))

  render(<LegacyApp />)

  expect(await screen.findByText('forced mate was available')).toBeTruthy()
  expect(screen.queryByText(/998\.08/)).toBeNull()
  expect(screen.getByRole('option', { name: 'Miss' })).toBeTruthy()
})
