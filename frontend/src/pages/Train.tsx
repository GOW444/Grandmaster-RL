import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, THEMES, type AttemptResult, type NextPuzzle, type Theme } from '../api'
import { useAuth } from '../auth'
import PuzzleBoard, { type BoardStatus } from '../components/PuzzleBoard'

type Mode = 'coach' | 'rated' | 'theme' | 'daily' | 'review'

const MODES: { id: Mode; label: string; desc: string }[] = [
  { id: 'coach', label: '🤖 AI Coach', desc: 'RL policy picks theme + difficulty' },
  { id: 'rated', label: '⚖️ Rated', desc: 'Any theme at your rating' },
  { id: 'theme', label: '🎯 Theme drill', desc: 'Focus on one motif' },
  { id: 'daily', label: '📅 Daily', desc: 'Same puzzle for everyone' },
  { id: 'review', label: '🔁 Review', desc: 'Puzzles you missed' },
]

const fmt = (ms: number) => `${Math.floor(ms / 60000)}:${String(Math.floor(ms / 1000) % 60).padStart(2, '0')}`
const signed = (n: number) => `${n >= 0 ? '+' : ''}${n.toFixed(1)}`

export default function Train() {
  const { refresh } = useAuth()
  const [params, setParams] = useSearchParams()
  const mode = (params.get('mode') as Mode) || 'coach'
  const theme = (params.get('theme') as Theme) || 'fork'
  const selector = params.get('selector') || 'ab'
  const replayId = params.get('puzzle')

  const [data, setData] = useState<NextPuzzle | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<BoardStatus>('intro')
  const [toMove, setToMove] = useState<'white' | 'black'>('white')
  const [result, setResult] = useState<AttemptResult | null>(null)
  const [saved, setSaved] = useState(false)
  const [hintSignal, setHintSignal] = useState(0)
  const [solutionSignal, setSolutionSignal] = useState(0)
  const [hints, setHints] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const startRef = useRef(0)

  const load = useCallback(async () => {
    setError(null)
    setResult(null)
    setData(null)
    setHints(0)
    try {
      let d: NextPuzzle
      if (replayId) d = await api.puzzle(replayId)
      else if (mode === 'daily') d = await api.daily()
      else if (mode === 'review') d = await api.review()
      else d = await api.next(mode, { theme: mode === 'theme' ? theme : undefined, selector: mode === 'coach' ? selector : undefined })
      setData(d)
      setSaved(d.saved)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [mode, theme, selector, replayId])

  useEffect(() => {
    load()
  }, [load])

  // Timer
  useEffect(() => {
    if (status !== 'playing') return
    startRef.current = performance.now()
    setElapsed(0)
    const t = setInterval(() => setElapsed(performance.now() - startRef.current), 250)
    return () => clearInterval(t)
  }, [status, data])

  const onStatus = useCallback((s: BoardStatus, info: { hints: number; toMove: 'white' | 'black' }) => {
    setStatus(s)
    setHints(info.hints)
    setToMove(info.toMove)
  }, [])

  const onFinish = async (solved: boolean, timeMs: number, hintsUsed: number) => {
    if (!data) return
    try {
      const r = await api.attempt({
        puzzle_id: data.puzzle.id,
        solved,
        time_ms: timeMs,
        hints_used: hintsUsed,
        mode: replayId ? 'review' : mode,
        selector: data.selector,
      })
      setResult(r)
      refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const toggleSave = async () => {
    if (!data) return
    if (saved) await api.unsave(data.puzzle.id)
    else await api.save(data.puzzle.id)
    setSaved(!saved)
  }

  const setMode = (m: Mode) => setParams(m === 'theme' ? { mode: m, theme } : m === 'coach' ? { mode: m, selector } : { mode: m })
  const done = status === 'solved' || status === 'failed'

  return (
    <div className="space-y-4">
      <div className="flex gap-2 overflow-x-auto pb-1">
        {MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={`card px-4 py-2 text-left shrink-0 transition-colors ${
              mode === m.id && !replayId ? 'border-brand bg-ink-800' : 'hover:border-ink-700'
            }`}
          >
            <div className="text-sm font-semibold">{m.label}</div>
            <div className="text-xs text-ink-500">{m.desc}</div>
          </button>
        ))}
      </div>

      {mode === 'theme' && !replayId && (
        <div className="flex gap-2 flex-wrap">
          {THEMES.map((t) => (
            <button
              key={t}
              onClick={() => setParams({ mode: 'theme', theme: t })}
              className={`px-3 py-1 rounded-full text-sm capitalize ${
                t === theme ? 'bg-brand text-ink-950 font-semibold' : 'bg-ink-800 text-ink-300'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      )}
      {mode === 'coach' && !replayId && (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-ink-500">Policy:</span>
          {[
            ['ab', 'A/B (random split)'],
            ['ppo', 'PPO only'],
            ['rating_match', 'Rating-match only'],
          ].map(([id, label]) => (
            <button
              key={id}
              onClick={() => setParams({ mode: 'coach', selector: id })}
              className={`px-3 py-1 rounded-full ${selector === id ? 'bg-ink-700 text-ink-100' : 'text-ink-500 hover:text-ink-300'}`}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      <div className="grid lg:grid-cols-[minmax(0,560px)_1fr] gap-6 items-start">
        <div>
          {data ? (
            <PuzzleBoard
              puzzle={data.puzzle}
              onFinish={onFinish}
              onStatus={onStatus}
              hintSignal={hintSignal}
              showSolutionSignal={solutionSignal}
            />
          ) : (
            <div className="w-full aspect-square card flex items-center justify-center text-ink-500 p-6 text-center">
              {error ?? 'Loading puzzle…'}
            </div>
          )}
        </div>

        <aside className="space-y-4">
          {data && (
            <div className="card p-5 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <div className="label">Puzzle {data.puzzle.id}</div>
                  <div className="text-2xl font-bold">
                    {done || replayId ? data.puzzle.rating : '????'}{' '}
                    <span className="text-sm font-medium text-ink-500 capitalize">· {data.puzzle.primary_theme}</span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="label">Time</div>
                  <div className="text-2xl font-mono tabular-nums">{fmt(elapsed)}</div>
                </div>
              </div>

              <div
                className={`rounded-lg px-4 py-3 font-semibold ${
                  status === 'solved'
                    ? 'bg-brand/20 text-brand'
                    : status === 'failed'
                      ? 'bg-bad/20 text-bad'
                      : 'bg-ink-800 text-ink-100'
                }`}
              >
                {status === 'intro' && 'Watch the opponent’s move…'}
                {status === 'playing' && (
                  <span className="flex items-center gap-2">
                    <span className={`inline-block w-3 h-3 rounded-sm ${toMove === 'white' ? 'bg-white' : 'bg-black border border-ink-500'}`} />
                    Your turn — find the best move for {toMove}
                  </span>
                )}
                {status === 'solved' && '✓ Solved!'}
                {status === 'failed' && '✗ Not quite.'}
              </div>

              {data.reason && (
                <p className="text-sm text-ink-300">
                  {data.selector && (
                    <span
                      className={`mr-2 text-xs px-2 py-0.5 rounded-full font-semibold ${
                        data.selector === 'ppo' ? 'bg-brand/20 text-brand' : 'bg-gold/20 text-gold'
                      }`}
                    >
                      {data.selector === 'ppo' ? 'PPO' : 'Rating-match'}
                    </span>
                  )}
                  {data.reason}
                </p>
              )}

              <div className="flex flex-wrap gap-2">
                {status === 'playing' && (
                  <button className="btn-ghost" onClick={() => setHintSignal((h) => h + 1)} disabled={hints >= 2}>
                    💡 Hint {hints > 0 && `(${hints}/2)`}
                  </button>
                )}
                {status === 'failed' && (
                  <button className="btn-ghost" onClick={() => setSolutionSignal((s) => s + 1)}>
                    👁 Show solution
                  </button>
                )}
                <button className="btn-ghost" onClick={toggleSave}>
                  {saved ? '★ Saved' : '☆ Save'}
                </button>
                {done && data.puzzle.game_url && (
                  <a className="btn-ghost" href={data.puzzle.game_url} target="_blank" rel="noreferrer">
                    ↗ Game
                  </a>
                )}
                {done && (
                  <button
                    className="btn-primary ml-auto"
                    onClick={() => (replayId ? setParams({ mode: 'coach' }) : load())}
                  >
                    Next puzzle →
                  </button>
                )}
              </div>
              {status === 'playing' && hints > 0 && (
                <p className="text-xs text-ink-500">Hints used — this attempt counts as a miss for your rating.</p>
              )}
            </div>
          )}

          {result && (
            <div className="card p-5 space-y-3">
              <div className="label">Result</div>
              <div className="grid grid-cols-3 gap-3 text-center">
                <Stat
                  label="Rating"
                  value={Math.round(result.rating_after).toString()}
                  delta={result.rating_after - result.rating_before}
                />
                <Stat
                  label={`${result.theme}`}
                  value={Math.round(result.theme_rating_after).toString()}
                  delta={result.theme_rating_after - result.theme_rating_before}
                />
                <Stat label="XP" value={`+${result.xp_gained}`} />
              </div>
              {result.rating_after === result.rating_before && (
                <p className="text-xs text-ink-500">Unrated — you’ve seen this puzzle before.</p>
              )}
              {result.new_badges.map((b) => (
                <div key={b.code} className="flex items-center gap-3 rounded-lg bg-gold/10 border border-gold/30 px-3 py-2">
                  <span className="text-2xl">{b.icon}</span>
                  <div>
                    <div className="text-sm font-semibold text-gold">Badge unlocked: {b.name}</div>
                    <div className="text-xs text-ink-300">{b.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}

function Stat({ label, value, delta }: { label: string; value: string; delta?: number }) {
  return (
    <div className="rounded-lg bg-ink-800 py-3">
      <div className="text-xs text-ink-500 capitalize">{label}</div>
      <div className="text-xl font-bold">{value}</div>
      {delta !== undefined && (
        <div className={`text-xs font-semibold ${delta >= 0 ? 'text-brand' : 'text-bad'}`}>{signed(delta)}</div>
      )}
    </div>
  )
}
