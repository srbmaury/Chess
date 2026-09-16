import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
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
      if (url === '/api/pipeline/report') {
        return { ok: true, json: async () => ({ stage: 'report', status: 'running' }) } as Response
      }
      throw new Error(`Unexpected fetch: ${url}`)
    }),
  )
  return () => source
}

function emit(source: { onmessage: ((event: MessageEvent) => void) | null } | null, payload: Record<string, unknown>) {
  source?.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent)
}

function fakeWindow() {
  return {
    closed: false,
    close: vi.fn(),
    document: { title: '', body: { textContent: '' } },
    location: { href: '' },
  }
}

test('clicking Run for report opens a tab synchronously and fills it in on success', async () => {
  window.history.pushState({}, '', '/pipeline')
  const getSource = installPipeline({ status: 'idle', stage: null, username: 'srbmaury' })
  const win = fakeWindow()
  const openSpy = vi.spyOn(window, 'open').mockReturnValue(win as unknown as Window)

  render(<App />)
  expect(await screen.findByText('Build your coach')).toBeTruthy()

  const reportStage = screen.getByText('Report').closest('article')!
  fireEvent.click(within(reportStage).getByRole('button', { name: 'Run' }))

  // The tab must be opened synchronously inside the click handler, before
  // any async work, or browsers treat it as a blocked popup.
  expect(openSpy).toHaveBeenCalledWith('about:blank', '_blank')
  expect(win.document.body.textContent).toBe('Generating report…')
  expect(win.location.href).toBe('')

  emit(getSource(), { stage: 'report', status: 'succeeded', report_file: '/tmp/coaching_report.md' })

  expect(win.location.href).toBe('/api/report')
})

test('a failed report run shows the error in the opened tab instead of the report', async () => {
  window.history.pushState({}, '', '/pipeline')
  const getSource = installPipeline({ status: 'idle', stage: null, username: 'srbmaury' })
  const win = fakeWindow()
  vi.spyOn(window, 'open').mockReturnValue(win as unknown as Window)

  render(<App />)
  expect(await screen.findByText('Build your coach')).toBeTruthy()

  const reportStage = screen.getByText('Report').closest('article')!
  fireEvent.click(within(reportStage).getByRole('button', { name: 'Run' }))

  emit(getSource(), { stage: 'report', status: 'failed', error: 'Puzzle bank not found.' })

  expect(win.location.href).toBe('')
  expect(win.document.body.textContent).toBe('Report generation failed: Puzzle bank not found.')
})
