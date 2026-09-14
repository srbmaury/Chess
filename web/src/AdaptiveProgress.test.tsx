import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

vi.mock('react-chessboard', () => ({ Chessboard: () => <div /> }))

import LegacyApp from './LegacyApp'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  window.history.pushState({}, '', '/')
})

test('Progress renders adaptive continuation metrics separately from review accuracy', async () => {
  window.history.pushState({}, '', '/progress')
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: unknown) => {
      const url = String(input)
      if (url === '/api/progress') {
        return {
          ok: true,
          json: async () => ({
            total_puzzles: 20,
            due_puzzles: 4,
            reviewed_puzzles: 10,
            mastered_puzzles: 2,
            total_reviews: 18,
            accuracy: 0.5,
            by_motif: [],
            by_opening: [],
            daily_reviews: [],
            adaptive: {
              sessions_completed: 8,
              success_rate: 0.625,
              continuation_accuracy: 0.75,
              average_accepted_decisions: 2.5,
              average_calculation_depth_plies: 4.25,
            },
          }),
        } as Response
      }
      throw new Error(`Unexpected fetch: ${url}`)
    }),
  )

  render(<LegacyApp />)

  expect(await screen.findByText('Adaptive calculation')).toBeTruthy()
  expect(screen.getByText('Adaptive drills')).toBeTruthy()
  expect(screen.getByText('8')).toBeTruthy()
  expect(screen.getByText('Conversion rate')).toBeTruthy()
  expect(screen.getByText('62.5%')).toBeTruthy()
  expect(screen.getByText('Continuation accuracy')).toBeTruthy()
  expect(screen.getByText('75.0%')).toBeTruthy()
  expect(screen.getByText('Avg strong decisions')).toBeTruthy()
  expect(screen.getByText('2.5')).toBeTruthy()
  expect(screen.getByText('Avg calculation depth')).toBeTruthy()
  expect(screen.getByText('4.3 plies')).toBeTruthy()
  expect(screen.getByText('50.0%')).toBeTruthy()
})
