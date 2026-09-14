import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

vi.mock('react-chessboard', () => ({ Chessboard: () => <div /> }))
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  XAxis: (props: Record<string, any>) => <div data-testid="progress-x-axis" data-tick-fill={props.tick?.fill} data-axis-stroke={props.axisLine?.stroke} />,
  YAxis: (props: Record<string, any>) => <div data-testid="progress-y-axis" data-tick-fill={props.tick?.fill} data-axis-stroke={props.axisLine?.stroke} />,
  Tooltip: (props: Record<string, any>) => <div data-testid="progress-tooltip" data-background={props.contentStyle?.background} data-border={props.contentStyle?.border} data-color={props.contentStyle?.color} />,
  Line: (props: Record<string, any>) => <div data-testid="progress-line" data-stroke={props.stroke} />,
}))

import App from './App'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

test('progress chart uses the dark application theme', async () => {
  window.history.pushState({}, '', '/progress')
  vi.stubGlobal('fetch', vi.fn(async (input: unknown) => {
    const url = String(input)
    if (url === '/api/profiles') {
      return { ok: true, json: async () => ({ active_username: 'srbmaury', profiles: [{ username: 'srbmaury', display_username: 'srbmaury' }] }) } as Response
    }
    if (url === '/api/pipeline/status') {
      return { ok: true, json: async () => ({ status: 'idle', stage: null }) } as Response
    }
    if (url === '/api/progress') {
      return { ok: true, json: async () => ({
        total_puzzles: 10,
        due_puzzles: 2,
        reviewed_puzzles: 8,
        mastered_puzzles: 3,
        total_reviews: 20,
        accuracy: 0.75,
        by_motif: [],
        by_opening: [],
        daily_reviews: [{ date: '2026-09-13', reviews: 5, correct: 4, accuracy: 0.8 }],
      }) } as Response
    }
    throw new Error(`Unexpected fetch: ${url}`)
  }))

  render(<App />)
  expect(await screen.findByText('Review activity')).toBeTruthy()

  expect(screen.getByTestId('progress-x-axis').getAttribute('data-tick-fill')).toBe('var(--muted)')
  expect(screen.getByTestId('progress-y-axis').getAttribute('data-tick-fill')).toBe('var(--muted)')
  expect(screen.getByTestId('progress-x-axis').getAttribute('data-axis-stroke')).toBe('var(--line)')
  expect(screen.getByTestId('progress-tooltip').getAttribute('data-background')).toBe('var(--panel)')
  expect(screen.getByTestId('progress-tooltip').getAttribute('data-border')).toBe('1px solid var(--line)')
  expect(screen.getByTestId('progress-tooltip').getAttribute('data-color')).toBe('var(--text)')
  expect(screen.getByTestId('progress-line').getAttribute('data-stroke')).toBe('var(--accent)')
})
