import { useCallback, useEffect, useState } from 'react'
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { Chess } from 'chess.js'
import { Chessboard } from 'react-chessboard'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts'

type Training = { total_puzzles: number; due_puzzles: number; reviewed_puzzles: number; mastered_puzzles: number; total_reviews: number; accuracy: number | null }
type Dashboard = { analyzed_moves: number; training: Training; artifacts: Record<string, { exists: boolean; updated_at?: string | null; rows?: number | null }> }
type PracticePuzzle = { puzzle_id: string; fen: string; orientation: string; game: string; move: string; opening: string; eco: string; phase: string; motif: string; difficulty: number; source_url?: string | null }
type Attempt = { correct: boolean; best_move_san: string; best_move_uci: string; your_game_move: string; evaluation_loss_pawns: number; next_interval_days: number; next_review_at: string; source_url?: string | null }
type PuzzleItem = PracticePuzzle & { your_move_san: string; your_move_uci: string; best_move_san: string; best_move_uci: string; evaluation_loss_pawns: number; quality: string; attempts: number; correct_attempts: number; accuracy: number | null; consecutive_correct: number; next_review_at: string; mastered: boolean }
type GroupRow = { label: string; puzzles: number; attempts: number; correct: number; accuracy: number | null }
type Progress = Training & { by_motif: GroupRow[]; by_opening: GroupRow[]; daily_reviews: Array<{ date: string; reviews: number; correct: number; accuracy: number }> }

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...init })
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const body = await response.json()
      message = body.detail || message
    } catch {
      // Keep generic message when the response is not JSON.
    }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

const api = {
  dashboard: () => request<Dashboard>('/api/dashboard'),
  nextPuzzle: () => request<{ puzzle: PracticePuzzle | null }>('/api/practice/next'),
  attempt: (id: string, move_uci: string) => request<Attempt>(`/api/practice/${id}/attempt`, { method: 'POST', body: JSON.stringify({ move_uci }) }),
  skip: (id: string) => request<{ skipped: boolean }>(`/api/practice/${id}/skip`, { method: 'POST' }),
  progress: () => request<Progress>('/api/progress'),
  puzzles: (query = '') => request<{ items: PuzzleItem[]; total: number; limit: number; offset: number }>(`/api/puzzles${query}`),
  pipelineStatus: () => request<Record<string, unknown>>('/api/pipeline/status'),
  startPipeline: (stage: string, options?: Record<string, unknown>) => request<Record<string, unknown>>(`/api/pipeline/${stage}`, { method: 'POST', body: JSON.stringify(options || {}) }),
}

const percentage = (value: number | null) => value == null ? '—' : `${(value * 100).toFixed(1)}%`

function Shell() {
  const nav = [['/', 'Dashboard'], ['/practice', 'Practice'], ['/mistakes', 'Mistakes'], ['/progress', 'Progress'], ['/pipeline', 'Pipeline']]
  return <div className="shell"><aside><div className="brand"><b>♞</b><div><strong>Chess ML Coach</strong><small>Personal training lab</small></div></div><nav>{nav.map(([to, label]) => <NavLink key={to} to={to} end={to === '/'}>{label}</NavLink>)}</nav></aside><main><Routes><Route path="/" element={<DashboardPage />} /><Route path="/practice" element={<PracticePage />} /><Route path="/mistakes" element={<MistakesPage />} /><Route path="/progress" element={<ProgressPage />} /><Route path="/pipeline" element={<PipelinePage />} /></Routes></main></div>
}

function Metric({ label, value }: { label: string; value: string | number }) { return <article className="metric"><span>{label}</span><strong>{value}</strong></article> }
function Heading({ kicker, title, copy, action }: { kicker: string; title: string; copy: string; action?: React.ReactNode }) { return <div className="heading"><div><p className="kicker">{kicker}</p><h1>{title}</h1><p>{copy}</p></div>{action}</div> }
function ErrorBox({ message }: { message: string }) { return <div className="error">{message}</div> }

function DashboardPage() {
  const [data, setData] = useState<Dashboard | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { api.dashboard().then(setData).catch((x: Error) => setError(x.message)) }, [])
  if (error) return <ErrorBox message={error} />
  if (!data) return <p className="muted">Loading dashboard…</p>
  return <section><Heading kicker="TODAY" title="Your training cockpit" copy="Use your own mistakes as the next study plan." action={<NavLink className="button" to="/practice">Start practice</NavLink>} /><div className="metrics"><Metric label="Due puzzles" value={data.training.due_puzzles} /><Metric label="Mastered" value={data.training.mastered_puzzles} /><Metric label="Review accuracy" value={percentage(data.training.accuracy)} /><Metric label="Analyzed moves" value={data.analyzed_moves.toLocaleString()} /></div><div className="panel"><h2>Pipeline readiness</h2><div className="artifacts">{Object.entries(data.artifacts).map(([key, value]) => <div className="artifact" key={key}><i className={value.exists ? 'ready' : ''} /><span><strong>{key}</strong><small>{value.exists ? 'Ready' : 'Not built yet'}</small></span></div>)}</div></div></section>
}

function PracticePage() {
  const [puzzle, setPuzzle] = useState<PracticePuzzle | null | undefined>(undefined)
  const [feedback, setFeedback] = useState<Attempt | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const load = useCallback(() => { setFeedback(null); setSubmitting(false); setError(''); setPuzzle(undefined); api.nextPuzzle().then((r) => setPuzzle(r.puzzle)).catch((x: Error) => setError(x.message)) }, [])
  useEffect(load, [load])

  async function submitMove(moveUci: string) {
    if (!puzzle || feedback || submitting) return
    setSubmitting(true)
    try {
      setFeedback(await api.attempt(puzzle.puzzle_id, moveUci))
    } catch (x) {
      setError((x as Error).message)
      setSubmitting(false)
    }
  }

  function onDrop(sourceSquare: string, targetSquare: string | null) {
    if (!puzzle || feedback || submitting || !targetSquare) return false
    let moveUci = `${sourceSquare}${targetSquare}`
    try {
      const position = new Chess(puzzle.fen)
      const piece = position.get(sourceSquare as never)
      if (piece?.type === 'p' && (targetSquare.endsWith('1') || targetSquare.endsWith('8'))) {
        moveUci += 'q'
      }
    } catch {
      // The backend remains authoritative; submit the basic UCI move if local parsing fails.
    }
    void submitMove(moveUci)
    return true
  }

  async function skip() { if (!puzzle || submitting) return; await api.skip(puzzle.puzzle_id); load() }
  if (error) return <section><Heading kicker="PRACTICE" title="Puzzle trainer" copy="Solve positions from your own games." /><ErrorBox message={error} /><button onClick={load}>Retry</button></section>
  if (puzzle === undefined) return <p className="muted">Loading next puzzle…</p>
  if (!puzzle) return <section className="empty"><h1>You're caught up</h1><p>No puzzles are due right now.</p></section>
  return <section><Heading kicker={`${puzzle.phase} · ${puzzle.motif}`} title={`${puzzle.orientation === 'white' ? 'White' : 'Black'} to move`} copy={`${puzzle.game} · ${puzzle.move}`} action={<span className="pill">Difficulty ${puzzle.difficulty}/5</span>} /><div className="practice"><div className="board"><Chessboard options={{ position: puzzle.fen, boardOrientation: puzzle.orientation === 'black' ? 'black' : 'white', onPieceDrop: ({ sourceSquare, targetSquare }) => onDrop(sourceSquare, targetSquare) }} /></div><div className="panel practice-info"><h2>{puzzle.opening} <small>({puzzle.eco})</small></h2>{!feedback ? <><p className="muted">{submitting ? 'Checking your move…' : 'Find the strongest move. The engine answer stays hidden until you commit.'}</p><button className="ghost" disabled={submitting} onClick={skip}>Skip</button></> : <div className={feedback.correct ? 'feedback good' : 'feedback bad'}><h3>{feedback.correct ? 'Correct' : 'Not quite'}</h3><p>Best move: <strong>{feedback.best_move_san}</strong></p>{!feedback.correct && <p>Your game move: <strong>{feedback.your_game_move}</strong></p>}<p>Evaluation loss: {feedback.evaluation_loss_pawns.toFixed(2)} pawns</p><p>Next review: +{feedback.next_interval_days} days</p>{feedback.source_url && <a href={feedback.source_url} target="_blank" rel="noreferrer">Open source game ↗</a>}<button onClick={load}>Next puzzle</button></div>}</div></div></section>
}

function MistakesPage() {
  const [items, setItems] = useState<PuzzleItem[]>([])
  const [quality, setQuality] = useState('')
  const [error, setError] = useState('')
  useEffect(() => { api.puzzles(quality ? `?quality=${quality}` : '').then((r) => setItems(r.items)).catch((x: Error) => setError(x.message)) }, [quality])
  return <section><Heading kicker="EXPLORE" title="Your mistakes" copy="Browse training positions without waiting for them to become due." /><div className="toolbar"><select value={quality} onChange={(event) => setQuality(event.target.value)}><option value="">All quality</option><option value="blunder">Blunders</option><option value="mistake">Mistakes</option></select></div>{error && <ErrorBox message={error} />}<div className="list">{items.map((item) => <article className="row" key={item.puzzle_id}><div><span className={`tag ${item.quality}`}>{item.quality}</span><h3>{item.game} · {item.move}</h3><p>{item.opening} ({item.eco}) · {item.phase} · {item.motif}</p></div><div className="moves"><span>You <b>{item.your_move_san}</b></span><span>Best <b>{item.best_move_san}</b></span><span>Loss <b>{item.evaluation_loss_pawns.toFixed(2)} pawns</b></span></div>{item.source_url && <a href={item.source_url} target="_blank" rel="noreferrer">Game ↗</a>}</article>)}</div></section>
}

function ProgressPage() {
  const [progress, setProgress] = useState<Progress | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { api.progress().then(setProgress).catch((x: Error) => setError(x.message)) }, [])
  if (error) return <ErrorBox message={error} />
  if (!progress) return <p className="muted">Loading progress…</p>
  return <section><Heading kicker="TRAINING" title="Progress" copy="Measure puzzle mastery and practice consistency." /><div className="metrics"><Metric label="Reviewed" value={progress.reviewed_puzzles} /><Metric label="Mastered" value={progress.mastered_puzzles} /><Metric label="Total reviews" value={progress.total_reviews} /><Metric label="Accuracy" value={percentage(progress.accuracy)} /></div><div className="panel"><h2>Review activity</h2><ResponsiveContainer width="100%" height={250}><LineChart data={progress.daily_reviews}><XAxis dataKey="date" /><YAxis allowDecimals={false} /><Tooltip /><Line type="monotone" dataKey="reviews" stroke="currentColor" strokeWidth={2} /></LineChart></ResponsiveContainer></div><div className="columns"><Ranking title="By motif" rows={progress.by_motif} /><Ranking title="By opening" rows={progress.by_opening} /></div></section>
}
function Ranking({ title, rows }: { title: string; rows: GroupRow[] }) { return <div className="panel"><h2>{title}</h2>{rows.slice(0, 8).map((row) => <div className="rank" key={row.label}><span>{row.label}</span><b>{percentage(row.accuracy)}</b></div>)}</div> }

function PipelinePage() {
  const [status, setStatus] = useState<Record<string, any> | null>(null)
  const [depth, setDepth] = useState(14)
  const [error, setError] = useState('')
  const refresh = useCallback(() => api.pipelineStatus().then(setStatus).catch((x: Error) => setError(x.message)), [])
  useEffect(() => { refresh(); const stream = new EventSource('/api/pipeline/events'); stream.onmessage = (event) => { const progress = JSON.parse(event.data); setStatus((old) => ({ ...old, progress })) }; return () => stream.close() }, [refresh])
  async function start(stage: string) { try { setError(''); setStatus(await api.startPipeline(stage, stage === 'analyze' ? { depth } : undefined)); window.setTimeout(refresh, 250) } catch (x) { setError((x as Error).message) } }
  const running = status?.status === 'running'
  return <section><Heading kicker="PIPELINE" title="Build your coach" copy="Run expensive operations here and watch them progress." />{error && <ErrorBox message={error} />}<div className="list">{['sync', 'analyze', 'features', 'puzzles', 'train', 'report'].map((stage) => <article className="stage" key={stage}><div><strong>{stage[0].toUpperCase() + stage.slice(1)}</strong><small>{stage === 'analyze' ? 'Stockfish evaluation' : stage === 'train' ? 'Personalized LightGBM model' : 'Pipeline stage'}</small></div>{stage === 'analyze' && <input aria-label="Stockfish depth" type="number" min={1} value={depth} onChange={(event) => setDepth(Number(event.target.value))} />}<button disabled={running} onClick={() => start(stage)}>{running && status?.stage === stage ? 'Running…' : 'Run'}</button></article>)}</div>{status?.progress && <div className="panel"><h2>Live progress</h2><div className="progress-grid">{Object.entries(status.progress).filter(([key]) => !['created_at', 'sequence', 'stage'].includes(key)).map(([key, value]) => <div key={key}><span>{key.replaceAll('_', ' ')}</span><b>{String(value)}</b></div>)}</div></div>}</section>
}

export default function App() { return <BrowserRouter><Shell /></BrowserRouter> }
