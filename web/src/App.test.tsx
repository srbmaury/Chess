import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'

import App from './App'


test('renders dashboard metrics from the local API', async () => {
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
