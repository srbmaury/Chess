import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

vi.mock('react-chessboard', () => ({
  Chessboard: ({ options }: { options: { onPieceDrop?: (move: { sourceSquare: string; targetSquare: string }) => boolean } }) => (
    <button aria-label="Play test move" onClick={() => options.onPieceDrop?.({ sourceSquare: 'h6', targetSquare: 'h5' })}>
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

// Real position from a srbmaury game (puzzle ed4eb398...): White to move at
// move 13. The user's actual Qh5+ hangs the queen to ...Qxh5; the engine
// line starts with the retreat Qh3. Confirms best_line move numbers/side
// alternation render correctly when the line does NOT start at move 1.
const FEN = 'r3kr2/pp2b2p/5p1Q/3q4/8/8/PPP2PPP/RNB1KB1R w KQq - 1 13'

test('best line shows move numbers and side alternation starting mid-game', async () => {
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
            puzzle_id: 'ed4eb398743ffc6af0deb674',
            fen: FEN,
            orientation: 'white',
            game: 'srbmaury vs opponent',
            move: '13.Qh5+',
            opening: 'Some Opening',
            eco: 'A00',
            phase: 'middlegame',
            motif: 'positional / calculation',
            difficulty: 3,
          },
        }),
      } as Response
    }
    if (url.includes('/attempt')) {
      return {
        ok: true,
        json: async () => ({
          correct: false,
          best_move_san: 'Qh3',
          best_move_uci: 'h6h3',
          your_game_move: 'Qh5+',
          evaluation_loss_pawns: 13.15,
          next_interval_days: 1,
          next_review_at: '2026-09-17T00:00:00Z',
        }),
      } as Response
    }
    if (url.includes('/explanation')) {
      return {
        ok: true,
        json: async () => ({
          idea: 'Best continuation',
          why: 'Stockfish evaluates this move as the strongest continuation in the position.',
          best_line: ['Qh3', 'f5', 'Be2', 'Rg8', 'Bh5+', 'Kd8'],
          why_your_move_was_worse:
            'Your game move Qh5+ missed this continuation and lost about 13.15 pawns of evaluation compared with the best move.',
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

  fireEvent.click(await screen.findByRole('button', { name: 'Quick' }))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Quick' }).getAttribute('aria-pressed')).toBe('true'))

  fireEvent.click(screen.getByRole('button', { name: 'Play test move' }))

  const whyButton = await screen.findByRole('button', { name: 'Why is this best?' })
  fireEvent.click(whyButton)

  expect(await screen.findByText('13.Qh3 f5 14.Be2 Rg8 15.Bh5+ Kd8')).toBeTruthy()
})
