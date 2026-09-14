import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

vi.mock('react-chessboard', () => ({
  Chessboard: ({ options }: { options: { position: string; allowDragging?: boolean } }) => (
    <div>
      <span data-testid="board-position">{options.position}</span>
      <span data-testid="board-dragging">{String(options.allowDragging)}</span>
    </div>
  ),
}))

import LegacyApp from './LegacyApp'

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

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
  vi.unstubAllGlobals()
  window.history.pushState({}, '', '/')
})

test('line so far shows only accepted moves and hides rejected guesses', async () => {
  window.history.pushState({}, '', '/practice')
  const resumed = {
    ...activeSession,
    current_ply: 4,
    user_moves_attempted: 5,
    user_moves_accepted: 2,
    steps: [
      { step_index: 0, side: 'user', move_uci: 'c2c4', move_san: 'c4', accepted: false },
      { step_index: 1, side: 'user', move_uci: 'd2d4', move_san: 'd4', accepted: true },
      { step_index: 2, side: 'engine', move_uci: 'd7d5', move_san: 'd5', accepted: true },
      { step_index: 3, side: 'user', move_uci: 'e2e4', move_san: 'e4', accepted: false },
      { step_index: 4, side: 'user', move_uci: 'g1f3', move_san: 'Nf3', accepted: true },
      { step_index: 5, side: 'engine', move_uci: 'g8f6', move_san: 'Nf6', accepted: true },
    ],
  }
  vi.stubGlobal('fetch', vi.fn(async (input: unknown) => {
    const url = String(input)
    if (url === '/api/practice/next') return response({ puzzle })
    if (url === '/api/practice/p1/adaptive/start') return response(resumed)
    throw new Error(`Unexpected fetch: ${url}`)
  }))

  render(<LegacyApp />)

  expect(await screen.findByText('Line so far')).toBeTruthy()
  expect(screen.getByText('1. d4')).toBeTruthy()
  expect(screen.getByText('… d5')).toBeTruthy()
  expect(screen.getByText('2. Nf3')).toBeTruthy()
  expect(screen.getByText('… Nf6')).toBeTruthy()
  expect(document.body.textContent).not.toContain('c4')
  expect(document.body.textContent).not.toContain('e4')
})

test('show next best move reveals one move while keeping the drill active', async () => {
  window.history.pushState({}, '', '/practice')
  let hintCalls = 0
  const fetchMock = vi.fn(async (input: unknown) => {
    const url = String(input)
    if (url === '/api/practice/next') return response({ puzzle })
    if (url === '/api/practice/p1/adaptive/start') return response(activeSession)
    if (url === '/api/practice/adaptive/s1/hint') {
      hintCalls += 1
      return response({
        session_id: 's1',
        puzzle_id: 'p1',
        status: 'active',
        move_uci: 'd2d4',
        move_san: 'd4',
        current_fen: START,
        user_moves_attempted: 1,
        user_moves_accepted: 0,
        current_ply: 0,
      })
    }
    throw new Error(`Unexpected fetch: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)

  render(<LegacyApp />)
  await screen.findByText('White to move')

  fireEvent.click(screen.getByRole('button', { name: 'Show next best move' }))

  expect(await screen.findByText((_, element) => element?.textContent === 'Next best move: d4')).toBeTruthy()
  expect(screen.getByTestId('board-position').textContent).toBe(START)
  expect(screen.getByTestId('board-dragging').textContent).toBe('true')
  expect(screen.getByRole('button', { name: 'Skip' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Move shown' }).hasAttribute('disabled')).toBe(true)
  fireEvent.click(screen.getByRole('button', { name: 'Move shown' }))
  await waitFor(() => expect(hintCalls).toBe(1))
  expect(document.body.textContent).not.toContain('d7d5')
  expect(document.body.textContent).not.toContain('g1f3')
})
