import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { Chess } from 'chess.js'

vi.mock('react-chessboard', () => ({
  Chessboard: ({ options }: { options: { position: string; allowDragging?: boolean; onPieceDrop?: (move: { sourceSquare: string; targetSquare: string }) => boolean } }) => (
    <div>
      <span data-testid="board-position">{options.position}</span>
      <span data-testid="board-dragging">{String(options.allowDragging)}</span>
      <button aria-label="Castle kingside" onClick={() => options.onPieceDrop?.({ sourceSquare: 'e1', targetSquare: 'g1' })}>Castle kingside</button>
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

const preCastleFen = fenAfter('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5')

const puzzle = {
  puzzle_id: 'p1',
  fen: preCastleFen,
  orientation: 'white',
  game: 'player vs opponent',
  move: '4.O-O',
  opening: "Italian Game",
  eco: 'C50',
  phase: 'opening',
  motif: 'positional / calculation',
  difficulty: 3,
}

const activeSession = {
  session_id: 's1',
  puzzle_id: 'p1',
  status: 'active',
  current_fen: preCastleFen,
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

test('rejected castling move stays previewed until the animation settle delay, then reverts', async () => {
  window.history.pushState({}, '', '/practice')
  vi.stubGlobal('fetch', vi.fn(async (input: unknown) => {
    const url = String(input)
    if (url === '/api/practice/next') return response({ puzzle })
    if (url === '/api/practice/p1/adaptive/start') return response(activeSession)
    if (url === '/api/practice/adaptive/s1/move') {
      return response({
        ...activeSession,
        accepted: false,
        move_uci: 'e1g1',
        move_san: 'O-O',
        eval_loss_cp: 80,
        current_fen: preCastleFen,
        user_moves_attempted: 1,
        user_moves_accepted: 0,
        max_eval_loss_cp: 80,
      })
    }
    throw new Error(`Unexpected fetch: ${url}`)
  }))

  render(<LegacyApp />)
  await screen.findByText('White to move')
  vi.useFakeTimers()

  fireEvent.click(screen.getByRole('button', { name: 'Castle kingside' }))
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })

  const castledPreviewFen = fenAfter('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5', 'e1g1')
  // Preview still shows the castled position immediately after the rejected
  // move resolves, giving react-chessboard's own (uncancelled) rook-move
  // animation time to finish before we swap the position back.
  expect(screen.getByTestId('board-position').textContent).toBe(castledPreviewFen)

  await act(async () => {
    await vi.advanceTimersByTimeAsync(349)
  })
  expect(screen.getByTestId('board-position').textContent).toBe(castledPreviewFen)

  await act(async () => {
    await vi.advanceTimersByTimeAsync(1)
  })
  expect(screen.getByTestId('board-position').textContent).toBe(preCastleFen)
})
