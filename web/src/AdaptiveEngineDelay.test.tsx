import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { Chess } from 'chess.js'

vi.mock('react-chessboard', () => ({
  Chessboard: ({ options }: { options: { position: string; allowDragging?: boolean; onPieceDrop?: (move: { sourceSquare: string; targetSquare: string }) => boolean } }) => (
    <div>
      <span data-testid="board-position">{options.position}</span>
      <span data-testid="board-dragging">{String(options.allowDragging)}</span>
      <button aria-label="Play d4" onClick={() => options.onPieceDrop?.({ sourceSquare: 'd2', targetSquare: 'd4' })}>Play d4</button>
    </div>
  ),
}))

import LegacyApp from './LegacyApp'

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

function fenAfter(...moves: string[]) {
  const board = new Chess(START)
  for (const move of moves) {
    board.move({ from: move.slice(0, 2), to: move.slice(2, 4), promotion: move[4] })
  }
  return board.fen()
}

const puzzle = {
  puzzle_id: 'p1',
  fen: START,
  orientation: 'white',
  game: 'player vs opponent',
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

function response(body: unknown) {
  return { ok: true, json: async () => body } as Response
}

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.unstubAllGlobals()
  window.history.pushState({}, '', '/')
})

test('user move appears before the brief computer reply delay', async () => {
  window.history.pushState({}, '', '/practice')
  vi.stubGlobal('fetch', vi.fn(async (input: unknown) => {
    const url = String(input)
    if (url === '/api/practice/next') return response({ puzzle })
    if (url === '/api/practice/p1/adaptive/start') return response(activeSession)
    if (url === '/api/practice/adaptive/s1/move') {
      return response({
        session_id: 's1',
        puzzle_id: 'p1',
        status: 'active',
        accepted: true,
        move_uci: 'd2d4',
        move_san: 'd4',
        eval_loss_cp: 0,
        engine_reply_uci: 'd7d5',
        engine_reply_san: 'd5',
        current_fen: fenAfter('d2d4', 'd7d5'),
        user_moves_attempted: 1,
        user_moves_accepted: 1,
        current_ply: 2,
        max_eval_loss_cp: 0,
        review: null,
      })
    }
    throw new Error(`Unexpected fetch: ${url}`)
  }))

  render(<LegacyApp />)
  await screen.findByText('White to move')
  vi.useFakeTimers()

  fireEvent.click(screen.getByRole('button', { name: 'Play d4' }))
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })

  const userFen = fenAfter('d2d4')
  const finalFen = fenAfter('d2d4', 'd7d5')
  expect(screen.getByTestId('board-position').textContent).toBe(userFen)
  expect(screen.getByTestId('board-dragging').textContent).toBe('false')
  expect(screen.queryByText((_, element) => element?.textContent === 'Engine replied d5')).toBeNull()

  await act(async () => {
    await vi.advanceTimersByTimeAsync(349)
  })
  expect(screen.getByTestId('board-position').textContent).toBe(userFen)
  expect(screen.queryByText((_, element) => element?.textContent === 'Engine replied d5')).toBeNull()

  await act(async () => {
    await vi.advanceTimersByTimeAsync(1)
  })
  expect(screen.getByTestId('board-position').textContent).toBe(finalFen)
  expect(screen.getByText((_, element) => element?.textContent === 'Engine replied d5')).toBeTruthy()
  expect(screen.getByTestId('board-dragging').textContent).toBe('true')
})
