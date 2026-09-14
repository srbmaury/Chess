import { useCallback, useEffect, useRef, useState } from 'react'
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { Chess } from 'chess.js'
import { Chessboard } from 'react-chessboard'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts'

type Training = { total_puzzles: number; due_puzzles: number; reviewed_puzzles: number; mastered_puzzles: number; total_reviews: number; accuracy: number | null }
type Dashboard = { analyzed_moves: number; training: Training; artifacts: Record<string, { exists: boolean; updated_at?: string | null; rows?: number | null }> }
type PracticePuzzle = { puzzle_id: string; fen: string; orientation: string; game: string; move: string; opening: string; eco: string; phase: string; motif: string; difficulty: number; source_url?: string | null }
type Attempt = { correct: boolean; best_move_san: string; best_move_uci: string; your_game_move: string; evaluation_loss_pawns: number; next_interval_days: number; next_review_at: string; source_url?: string | null }
type Explanation = { idea: string; why: string; best_line: string[]; why_your_move_was_worse: string; engine_grounded: boolean; depth: number; cached: boolean }
type PuzzleItem = PracticePuzzle & { your_move_san: string; your_move_uci: string; best_move_san: string; best_move_uci: string; evaluation_loss_pawns: number; quality: string; attempts: number; correct_attempts: number; accuracy: number | null; consecutive_correct: number; next_review_at: string; mastered: boolean }
type GroupRow = { label: string; puzzles: number; attempts: number; correct: number; accuracy: number | null }
type AdaptiveReview = { next_interval_days: number; next_review_at: string; consecutive_correct: number; mastered: boolean }
type AdaptiveSafeStep = { step_index: number; side: string; move_uci: string; move_san: string; accepted: boolean }
type AdaptiveSession = {
  session_id: string
  puzzle_id: string
  status: string
  current_fen: string
  orientation: string
  user_moves_attempted: number
  user_moves_accepted: number
  current_ply: number
  max_eval_loss_cp: number
  max_user_decisions: number
  steps: AdaptiveSafeStep[]
  review: AdaptiveReview | null
}
type AdaptiveMoveResult = {
  session_id: string
  puzzle_id: string
  status: string
  accepted: boolean | null
  move_uci?: string | null
  move_san?: string | null
  eval_loss_cp?: number | null
  engine_reply_uci?: string | null
  engine_reply_san?: string | null
  current_fen: string
  user_moves_attempted: number
  user_moves_accepted: number
  current_ply: number
  max_eval_loss_cp: number
  review: AdaptiveReview | null
}
type AdaptiveProgress = {
  sessions_completed: number
  success_rate: number | null
  continuation_accuracy: number | null
  average_accepted_decisions: number | null
  average_calculation_depth_plies: number | null
}
type Progress = Training & { by_motif: GroupRow[]; by_opening: GroupRow[]; daily_reviews: Array<{ date: string; reviews: number; correct: number; accuracy: number }>; adaptive?: AdaptiveProgress }
type PracticeMode = 'adaptive' | 'quick'

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
  explanation: (id: string) => request<Explanation>(`/api/practice/${id}/explanation`),
  skip: (id: string) => request<{ skipped: boolean }>(`/api/practice/${id}/skip`, { method: 'POST' }),
  adaptiveStart: (id: string) => request<AdaptiveSession>(`/api/practice/${id}/adaptive/start`, { method: 'POST' }),
  adaptiveMove: (sessionId: string, move_uci: string) => request<AdaptiveMoveResult>(`/api/practice/adaptive/${sessionId}/move`, { method: 'POST', body: JSON.stringify({ move_uci }) }),
  adaptiveAbandon: (sessionId: string) => request<AdaptiveSession>(`/api/practice/adaptive/${sessionId}/abandon`, { method: 'POST' }),
  progress: () => request<Progress>('/api/progress'),
  puzzles: (query = '') => request<{ items: PuzzleItem[]; total: number; limit: number; offset: number }>(`/api/puzzles${query}`),
  pipelineStatus: () => request<Record<string, unknown>>('/api/pipeline/status'),
  startPipeline: (stage: string, options?: Record<string, unknown>) => request<Record<string, unknown>>(`/api/pipeline/${stage}`, { method: 'POST', body: JSON.stringify(options || {}) }),
  stopPipeline: () => request<Record<string, unknown>>('/api/pipeline/stop', { method: 'POST' }),
}

const percentage = (value: number | null) => value == null ? '—' : `${(value * 100).toFixed(1)}%`
const decimal = (value: number | null) => value == null ? '—' : value.toFixed(1)

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
  const [mode, setMode] = useState<PracticeMode>('adaptive')
  const modeRef = useRef<PracticeMode>('adaptive')
  const [puzzle, setPuzzle] = useState<PracticePuzzle | null | undefined>(undefined)
  const [feedback, setFeedback] = useState<Attempt | null>(null)
  const [adaptive, setAdaptive] = useState<AdaptiveSession | null>(null)
  const [lastAdaptiveMove, setLastAdaptiveMove] = useState<AdaptiveMoveResult | null>(null)
  const [explanation, setExplanation] = useState<Explanation | null>(null)
  const [explaining, setExplaining] = useState(false)
  const [explanationError, setExplanationError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => { modeRef.current = mode }, [mode])

  const load = useCallback(async () => {
    setFeedback(null)
    setAdaptive(null)
    setLastAdaptiveMove(null)
    setExplanation(null)
    setExplaining(false)
    setExplanationError('')
    setSubmitting(false)
    setError('')
    setPuzzle(undefined)
    try {
      const result = await api.nextPuzzle()
      setPuzzle(result.puzzle)
      if (result.puzzle && modeRef.current === 'adaptive') {
        setAdaptive(await api.adaptiveStart(result.puzzle.puzzle_id))
      }
    } catch (x) {
      setError((x as Error).message)
    }
  }, [])

  useEffect(() => { void load() }, [load])

  const adaptiveTerminal = adaptive != null && adaptive.status !== 'active'
  const finished = feedback != null || adaptiveTerminal
  const boardFen = mode === 'adaptive' && adaptive ? adaptive.current_fen : puzzle?.fen

  async function submitMove(moveUci: string) {
    if (!puzzle || submitting || finished) return
    setSubmitting(true)
    setError('')
    try {
      if (mode === 'quick') {
        setFeedback(await api.attempt(puzzle.puzzle_id, moveUci))
      } else {
        let session = adaptive
        if (!session) {
          session = await api.adaptiveStart(puzzle.puzzle_id)
          setAdaptive(session)
        }
        const result = await api.adaptiveMove(session.session_id, moveUci)
        setLastAdaptiveMove(result)
        setAdaptive((previous) => ({
          session_id: result.session_id,
          puzzle_id: result.puzzle_id,
          status: result.status,
          current_fen: result.current_fen,
          orientation: previous?.orientation ?? puzzle.orientation,
          user_moves_attempted: result.user_moves_attempted,
          user_moves_accepted: result.user_moves_accepted,
          current_ply: result.current_ply,
          max_eval_loss_cp: result.max_eval_loss_cp,
          max_user_decisions: previous?.max_user_decisions ?? 4,
          steps: previous?.steps ?? [],
          review: result.review,
        }))
      }
    } catch (x) {
      setError((x as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  async function explainBestMove() {
    if (!puzzle || !finished || explaining || explanation) return
    setExplaining(true)
    setExplanationError('')
    try {
      setExplanation(await api.explanation(puzzle.puzzle_id))
    } catch (x) {
      setExplanationError((x as Error).message)
    } finally {
      setExplaining(false)
    }
  }

  async function changeMode(next: PracticeMode) {
    if (next === mode || submitting || finished) return
    setError('')
    try {
      if (mode === 'adaptive' && adaptive?.status === 'active') {
        await api.adaptiveAbandon(adaptive.session_id)
      }
      modeRef.current = next
      setMode(next)
      setFeedback(null)
      setLastAdaptiveMove(null)
      setExplanation(null)
      setExplanationError('')
      if (next === 'adaptive' && puzzle) {
        setAdaptive(await api.adaptiveStart(puzzle.puzzle_id))
      } else {
        setAdaptive(null)
      }
    } catch (x) {
      setError((x as Error).message)
    }
  }

  function onDrop(sourceSquare: string, targetSquare: string | null) {
    if (!puzzle || submitting || finished || !targetSquare || !boardFen) return false
    let moveUci = `${sourceSquare}${targetSquare}`
    try {
      const position = new Chess(boardFen)
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

  async function skip() {
    if (!puzzle || submitting) return
    try {
      setSubmitting(true)
      if (mode === 'adaptive' && adaptive?.status === 'active') {
        await api.adaptiveAbandon(adaptive.session_id)
      }
      await api.skip(puzzle.puzzle_id)
      await load()
    } catch (x) {
      setError((x as Error).message)
      setSubmitting(false)
    }
  }

  if (error && puzzle === undefined) return <section><Heading kicker="PRACTICE" title="Puzzle trainer" copy="Solve positions from your own games." /><ErrorBox message={error} /><button onClick={() => { void load() }}>Retry</button></section>
  if (puzzle === undefined) return <p className="muted">Loading next puzzle…</p>
  if (!puzzle) return <section className="empty"><h1>You're caught up</h1><p>No puzzles are due right now.</p></section>

  const adaptiveReview = adaptive?.review
  const adaptiveActive = mode === 'adaptive' && adaptive?.status === 'active'
  const maxDecisions = adaptive?.max_user_decisions ?? 4
  const outcomeClass = adaptive?.status === 'failed' ? 'feedback bad' : 'feedback good'

  return <section>
    <Heading kicker={`${puzzle.phase} · ${puzzle.motif}`} title={`${puzzle.orientation === 'white' ? 'White' : 'Black'} to move`} copy={`${puzzle.game} · ${puzzle.move}`} action={<span className="pill">Difficulty {puzzle.difficulty}/5</span>} />
    <div className="mode-toggle" aria-label="Practice mode">
      <button className={mode === 'adaptive' ? 'active' : ''} aria-pressed={mode === 'adaptive'} disabled={submitting || finished} onClick={() => { void changeMode('adaptive') }}>Adaptive</button>
      <button className={mode === 'quick' ? 'active' : ''} aria-pressed={mode === 'quick'} disabled={submitting || finished} onClick={() => { void changeMode('quick') }}>Quick</button>
      <span>{mode === 'adaptive' ? 'Continue until the idea is converted.' : 'One best move, then review.'}</span>
    </div>
    {error && <ErrorBox message={error} />}
    <div className="practice">
      <div className="board"><Chessboard options={{ position: boardFen ?? puzzle.fen, boardOrientation: puzzle.orientation === 'black' ? 'black' : 'white', onPieceDrop: ({ sourceSquare, targetSquare }) => onDrop(sourceSquare, targetSquare) }} /></div>
      <div className="panel practice-info">
        <h2>{puzzle.opening} <small>({puzzle.eco})</small></h2>
        {mode === 'quick' ? (!feedback ? <><p className="muted">{submitting ? 'Checking your move…' : 'Find the strongest move. The engine answer stays hidden until you commit.'}</p><button className="ghost" disabled={submitting} onClick={() => { void skip() }}>Skip</button></> : <div className={feedback.correct ? 'feedback good' : 'feedback bad'}><h3>{feedback.correct ? 'Correct' : 'Not quite'}</h3><p>Best move: <strong>{feedback.best_move_san}</strong></p>{!feedback.correct && <p>Your game move: <strong>{feedback.your_game_move}</strong></p>}<p>Evaluation loss: {feedback.evaluation_loss_pawns.toFixed(2)} pawns</p><p>Next review: +{feedback.next_interval_days} days</p>{feedback.source_url && <a href={feedback.source_url} target="_blank" rel="noreferrer">Open source game ↗</a>}<button className="ghost" disabled={explaining} onClick={() => { void explainBestMove() }}>{explaining ? 'Analyzing why…' : 'Why is this best?'}</button>{explanationError && <ErrorBox message={explanationError} />}{explanation && <ExplanationView explanation={explanation} />}<button onClick={() => { void load() }}>Next puzzle</button></div>) : (
          !adaptive ? <p className="muted">Starting adaptive drill…</p> : adaptiveActive ? <div className="adaptive-live"><p className="muted">{submitting ? 'Checking the continuation…' : lastAdaptiveMove?.accepted ? 'Strong. Continuing the line…' : 'Find the strongest move. Strong alternatives are accepted.'}</p>{lastAdaptiveMove?.engine_reply_san && <p>Engine replied <strong>{lastAdaptiveMove.engine_reply_san}</strong></p>}<p className="sequence-progress">{adaptive.user_moves_accepted} / up to {maxDecisions} decisions</p><button className="ghost" disabled={submitting} onClick={() => { void skip() }}>Skip</button></div> : <div className={outcomeClass}><h3>{adaptive.status === 'succeeded' ? 'Converted' : adaptive.status === 'failed' ? 'Continuation missed' : 'Drill ended'}</h3><p>{adaptive.user_moves_accepted}/{adaptive.user_moves_attempted} strong decisions</p><p>Calculation depth: {adaptive.current_ply} plies</p><p>Maximum evaluation loss: {(adaptive.max_eval_loss_cp / 100).toFixed(2)} pawns</p>{adaptiveReview && <p>Next review: +{adaptiveReview.next_interval_days} days</p>}{puzzle.source_url && <a href={puzzle.source_url} target="_blank" rel="noreferrer">Open source game ↗</a>}<button className="ghost" disabled={explaining} onClick={() => { void explainBestMove() }}>{explaining ? 'Analyzing why…' : 'Why is this best?'}</button>{explanationError && <ErrorBox message={explanationError} />}{explanation && <ExplanationView explanation={explanation} />}<button onClick={() => { void load() }}>Next puzzle</button></div>
        )}
      </div>
    </div>
  </section>
}

function ExplanationView({ explanation }: { explanation: Explanation }) {
  return <div className="explanation"><p className="kicker">{explanation.engine_grounded ? `STOCKFISH · DEPTH ${explanation.depth}` : 'BOARD-ONLY FALLBACK'}</p><h3>{explanation.idea}</h3><p>{explanation.why}</p><p><strong>Best line:</strong> {explanation.best_line.join(' ')}</p><p><strong>Why your move was worse:</strong> {explanation.why_your_move_was_worse}</p></div>
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
  const adaptive = progress.adaptive ?? { sessions_completed: 0, success_rate: null, continuation_accuracy: null, average_accepted_decisions: null, average_calculation_depth_plies: null }
  return <section><Heading kicker="TRAINING" title="Progress" copy="Measure puzzle mastery and practice consistency." /><div className="metrics"><Metric label="Reviewed" value={progress.reviewed_puzzles} /><Metric label="Mastered" value={progress.mastered_puzzles} /><Metric label="Total reviews" value={progress.total_reviews} /><Metric label="Accuracy" value={percentage(progress.accuracy)} /></div><div className="panel"><h2>Review activity</h2><ResponsiveContainer width="100%" height={250}><LineChart data={progress.daily_reviews}><XAxis dataKey="date" tick={{ fill: 'var(--muted)' }} axisLine={{ stroke: 'var(--line)' }} tickLine={{ stroke: 'var(--line)' }} /><YAxis allowDecimals={false} tick={{ fill: 'var(--muted)' }} axisLine={{ stroke: 'var(--line)' }} tickLine={{ stroke: 'var(--line)' }} /><Tooltip contentStyle={{ background: 'var(--panel)', border: '1px solid var(--line)', borderRadius: 10, color: 'var(--text)' }} labelStyle={{ color: 'var(--text)' }} itemStyle={{ color: 'var(--text)' }} /><Line type="monotone" dataKey="reviews" stroke="var(--accent)" strokeWidth={2} /></LineChart></ResponsiveContainer></div><div className="panel"><h2>Adaptive calculation</h2><div className="adaptive-metrics"><Metric label="Adaptive drills" value={adaptive.sessions_completed} /><Metric label="Conversion rate" value={percentage(adaptive.success_rate)} /><Metric label="Continuation accuracy" value={percentage(adaptive.continuation_accuracy)} /><Metric label="Avg strong decisions" value={decimal(adaptive.average_accepted_decisions)} /><Metric label="Avg calculation depth" value={adaptive.average_calculation_depth_plies == null ? '—' : `${adaptive.average_calculation_depth_plies.toFixed(1)} plies`} /></div></div><div className="columns"><Ranking title="By motif" rows={progress.by_motif} /><Ranking title="By opening" rows={progress.by_opening} /></div></section>
}
function Ranking({ title, rows }: { title: string; rows: GroupRow[] }) { return <div className="panel"><h2>{title}</h2>{rows.slice(0, 8).map((row) => <div className="rank" key={row.label}><span>{row.label}</span><b>{percentage(row.accuracy)}</b></div>)}</div> }

function PipelinePage() {
  const [status, setStatus] = useState<Record<string, any> | null>(null)
  const [depth, setDepth] = useState(14)
  const [error, setError] = useState('')
  const [streamGeneration, setStreamGeneration] = useState(0)
  const refresh = useCallback(() => api.pipelineStatus().then(setStatus).catch((x: Error) => setError(x.message)), [])
  useEffect(() => {
    refresh()
    const stream = new EventSource('/api/pipeline/events')
    stream.onmessage = (event) => {
      const progress = JSON.parse(event.data)
      setStatus((old) => ({ ...old, ...progress, progress }))
      if (['succeeded', 'failed', 'cancelled'].includes(progress.status)) {
        stream.close()
      }
    }
    return () => stream.close()
  }, [refresh, streamGeneration])

  async function start(stage: string) {
    try {
      setError('')
      setStatus(await api.startPipeline(stage, stage === 'analyze' ? { depth } : undefined))
      setStreamGeneration((value) => value + 1)
      window.setTimeout(refresh, 250)
    } catch (x) {
      setError((x as Error).message)
    }
  }

  async function stop() {
    try {
      setError('')
      setStatus(await api.stopPipeline())
    } catch (x) {
      setError((x as Error).message)
    }
  }

  const active = ['running', 'stopping'].includes(status?.status)
  const analyzeActive = active && status?.stage === 'analyze'
  return <section><Heading kicker="PIPELINE" title="Build your coach" copy="Run expensive operations here and watch them progress." />{error && <ErrorBox message={error} />}<div className="list">{['sync', 'analyze', 'features', 'puzzles', 'train', 'report'].map((stage) => <article className="stage" key={stage}><div><strong>{stage[0].toUpperCase() + stage.slice(1)}</strong><small>{stage === 'analyze' ? 'Stockfish evaluation' : stage === 'train' ? 'Personalized LightGBM model' : 'Pipeline stage'}</small></div>{stage === 'analyze' && <input aria-label="Stockfish depth" type="number" min={1} value={depth} disabled={analyzeActive} onChange={(event) => setDepth(Number(event.target.value))} />}{stage === 'analyze' && analyzeActive ? <button className="ghost" disabled={status?.status === 'stopping'} onClick={() => { void stop() }}>{status?.status === 'stopping' ? 'Stopping…' : 'Stop'}</button> : <button disabled={active} onClick={() => { void start(stage) }}>{active && status?.stage === stage ? 'Running…' : 'Run'}</button>}</article>)}</div>{status?.progress && <div className="panel"><h2>Live progress</h2><div className="progress-grid">{Object.entries(status.progress).filter(([key]) => !['created_at', 'sequence', 'stage'].includes(key)).map(([key, value]) => <div key={key}><span>{key.replaceAll('_', ' ')}</span><b>{String(value)}</b></div>)}</div></div>}</section>
}

export default function App() { return <BrowserRouter><Shell /></BrowserRouter> }
