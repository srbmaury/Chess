import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

vi.mock('react-chessboard', () => ({
  Chessboard: () => null,
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

function installPipeline(status: Record<string, unknown>) {
  let source: FakeEventSource | null = null
  class FakeEventSource {
    onmessage: ((event: MessageEvent) => void) | null = null
    constructor(_url: string) {
      source = this
    }
    close() {}
  }
  vi.stubGlobal('EventSource', FakeEventSource)
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: unknown) => {
      const url = String(input)
      if (url === '/api/profiles') {
        return { ok: true, json: async () => activeProfiles } as Response
      }
      if (url === '/api/pipeline/status') {
        return { ok: true, json: async () => status } as Response
      }
      throw new Error(`Unexpected fetch: ${url}`)
    }),
  )
  return () => source
}

function emit(source: { onmessage: ((event: MessageEvent) => void) | null } | null, payload: Record<string, unknown>) {
  source?.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent)
}

test('labels sync archive progress as months', async () => {
  window.history.pushState({}, '', '/pipeline')
  const getSource = installPipeline({ status: 'running', stage: 'sync', username: 'srbmaury' })

  render(<App />)
  expect(await screen.findByText('Build your coach')).toBeTruthy()

  emit(getSource(), {
    stage: 'sync',
    status: 'running',
    current: 7,
    total: 25,
    archive: 'https://api.chess.com/pub/player/srbmaury/games/2026/03',
    game_count: 321,
    skipped: false,
  })

  expect(await screen.findByText('Current month')).toBeTruthy()
  expect(screen.getByText('Total months')).toBeTruthy()
  expect(screen.getByText('Games synced')).toBeTruthy()
  expect(screen.getByText('7')).toBeTruthy()
  expect(screen.getByText('25')).toBeTruthy()
})

test('labels analyze and puzzles progress with meaningful units', async () => {
  window.history.pushState({}, '', '/pipeline')
  const getSource = installPipeline({ status: 'running', stage: 'analyze', username: 'srbmaury' })

  render(<App />)
  expect(await screen.findByText('Build your coach')).toBeTruthy()

  emit(getSource(), {
    stage: 'analyze',
    status: 'running',
    completed: 121537,
    total: 211639,
    reused: 120000,
    analyzed: 1537,
  })

  expect(await screen.findByText('Completed moves')).toBeTruthy()
  expect(screen.getByText('Total user moves')).toBeTruthy()
  expect(screen.getByText('Reused analyses')).toBeTruthy()
  expect(screen.getByText('Newly analyzed moves')).toBeTruthy()

  emit(getSource(), {
    stage: 'puzzles',
    status: 'running',
    current: 1000,
    total: 121537,
    eligible: 213,
  })

  expect(await screen.findByText('Processed feature rows')).toBeTruthy()
  expect(screen.getByText('Total feature rows')).toBeTruthy()
  expect(screen.getByText('Eligible puzzles')).toBeTruthy()
})

test('renders pipeline result fields recursively instead of object strings', async () => {
  window.history.pushState({}, '', '/pipeline')
  installPipeline({
    status: 'succeeded',
    stage: 'train',
    username: 'srbmaury',
    result: {
      model_path: '/tmp/mistake_model.joblib',
      metadata_path: '/tmp/mistake_model.metadata.json',
      metrics: {
        roc_auc: 0.82,
        pr_auc: 0.64,
      },
    },
  })

  render(<App />)

  expect(await screen.findByText('Train result')).toBeTruthy()
  expect(screen.getByText('Model path')).toBeTruthy()
  expect(screen.getByText('ROC AUC')).toBeTruthy()
  expect(screen.getByText('0.82')).toBeTruthy()
  expect(screen.getByText('PR AUC')).toBeTruthy()
  expect(screen.getByText('0.64')).toBeTruthy()
  expect(screen.queryByText('[object Object]')).toBeNull()
})

test('uses stage-specific result labels for sync', async () => {
  window.history.pushState({}, '', '/pipeline')
  installPipeline({
    status: 'succeeded',
    stage: 'sync',
    username: 'srbmaury',
    result: {
      downloaded: 18,
      total: 339,
      pgn_path: '/tmp/srbmaury_all_games.pgn',
    },
  })

  render(<App />)

  expect(await screen.findByText('Sync result')).toBeTruthy()
  expect(screen.getByText('Downloaded games')).toBeTruthy()
  expect(screen.getByText('18')).toBeTruthy()
  expect(screen.getByText('Total games')).toBeTruthy()
  expect(screen.getByText('339')).toBeTruthy()
  expect(screen.queryByText('[object Object]')).toBeNull()
})
