import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { Chess } from 'chess.js'

vi.mock('react-chessboard', () => ({
  Chessboard: ({ options }: { options: { position: string; allowDragging?: boolean; onPieceDrop?: (move: { sourceSquare: string; targetSquare: string }) => boolean } }) => (
    <div>
      <span data-testid="board-position">{options.position}</span>
      <span data-testid="board-dragging">{String(options.allowDragging)}</span>
      <button aria-label="Try board move" onClick={() => options.onPieceDrop?.({ sourceSquare: 'c2', targetSquare: 'c4' })}>Try board move</button>
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

const steps = [
  { step_index: 0, side: 'user', move_uci: 'd2d4', move_san: 'd4', accepted: true },
  { step_index: 1, side: 'engine', move_uci: 'd7d5', move_san: 'd5', accepted: true },
  { step_index: 2, side: 'user', move_uci: 'g1f3', move_san: 'Nf3', accepted: true },
  { step_index: 3, side: 'engine', move_uci: 'g8f6', move_san: 'Nf6', accepted: true },
]

const activeSession = {
  session_id: 's1',
  puzzle_id: 'p1',
  status: 'active',
  current_fen: fenAfter('d2d4', 'd7d5', 'g1f3', 'g8f6'),
  orientation: 'white',
  user_moves_attempted: 2,
  user_moves_accepted: 2,
  current_ply: 4,
  max_eval_loss_cp: 0,
  max_user_decisions: 4,
  steps,
  review: null,
}

function response(body: unknown) {
  return { ok: true, json: async () => body } as Response
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  window.history.pushState({}, '', '/')
})

function renderActiveSession() {
  window.history.pushState({}, '', '/practice')
  const fetchMock = vi.fn(async (input: unknown) => {
    const url = String(input)
    if (url === '/api/practice/next') return response({ puzzle })
    if (url === '/api/practice/p1/adaptive/start') return response(activeSession)
    throw new Error(`Unexpected fetch: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)
  render(<LegacyApp />)
  return fetchMock
}

test('adaptive history navigates backward and forward one ply at a time', async () => {
  renderActiveSession()
  await screen.findByText('Line so far')

  expect(screen.getByTestId('board-position').textContent).toBe(fenAfter('d2d4', 'd7d5', 'g1f3', 'g8f6'))
  expect(screen.getByTestId('board-dragging').textContent).toBe('true')
  expect(screen.getByText('Ply 4 / 4')).toBeTruthy()

  fireEvent.click(screen.getByRole('button', { name: 'Previous position' }))
  expect(screen.getByTestId('board-position').textContent).toBe(fenAfter('d2d4', 'd7d5', 'g1f3'))
  expect(screen.getByText('Ply 3 / 4')).toBeTruthy()
  expect(screen.getByTestId('board-dragging').textContent).toBe('false')
  expect(screen.getByText('Reviewing earlier position')).toBeTruthy()

  fireEvent.click(screen.getByRole('button', { name: 'Previous position' }))
  expect(screen.getByTestId('board-position').textContent).toBe(fenAfter('d2d4', 'd7d5'))
  expect(screen.getByText('Ply 2 / 4')).toBeTruthy()

  fireEvent.click(screen.getByRole('button', { name: 'Next position' }))
  expect(screen.getByTestId('board-position').textContent).toBe(fenAfter('d2d4', 'd7d5', 'g1f3'))

  fireEvent.click(screen.getByRole('button', { name: 'Next position' }))
  expect(screen.getByTestId('board-position').textContent).toBe(activeSession.current_fen)
  expect(screen.getByText('Ply 4 / 4')).toBeTruthy()
  expect(screen.getByTestId('board-dragging').textContent).toBe('true')
  expect(screen.queryByText('Reviewing earlier position')).toBeNull()
})

test('clicking a played move jumps to that ply and return to current restores live play', async () => {
  const fetchMock = renderActiveSession()
  await screen.findByText('Line so far')

  fireEvent.click(screen.getByRole('button', { name: '… d5' }))

  expect(screen.getByTestId('board-position').textContent).toBe(fenAfter('d2d4', 'd7d5'))
  expect(screen.getByTestId('board-dragging').textContent).toBe('false')
  expect(screen.getByRole('button', { name: 'Show next best move' }).hasAttribute('disabled')).toBe(true)
  expect(screen.getByRole('button', { name: 'Skip' }).hasAttribute('disabled')).toBe(true)
  expect(screen.getByRole('button', { name: '… d5' }).getAttribute('aria-current')).toBe('step')

  fireEvent.click(screen.getByRole('button', { name: 'Try board move' }))
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))

  fireEvent.click(screen.getByRole('button', { name: 'Return to current' }))
  expect(screen.getByTestId('board-position').textContent).toBe(activeSession.current_fen)
  expect(screen.getByTestId('board-dragging').textContent).toBe('true')
  expect(screen.getByRole('button', { name: 'Show next best move' }).hasAttribute('disabled')).toBe(false)
})

test('completed adaptive drills still allow browsing every played ply', async () => {
  window.history.pushState({}, '', '/practice')
  const completed = {
    ...activeSession,
    status: 'succeeded',
    review: { next_interval_days: 3, next_review_at: '2026-09-17T00:00:00Z', consecutive_correct: 1, mastered: false },
  }
  vi.stubGlobal('fetch', vi.fn(async (input: unknown) => {
    const url = String(input)
    if (url === '/api/practice/next') return response({ puzzle })
    if (url === '/api/practice/p1/adaptive/start') return response(completed)
    throw new Error(`Unexpected fetch: ${url}`)
  }))

  render(<LegacyApp />)
  await screen.findByText('Converted')

  fireEvent.click(screen.getByRole('button', { name: 'Previous position' }))
  expect(screen.getByTestId('board-position').textContent).toBe(fenAfter('d2d4', 'd7d5', 'g1f3'))
  expect(screen.getByText('Reviewing earlier position')).toBeTruthy()
  expect(screen.getByTestId('board-dragging').textContent).toBe('false')
})
