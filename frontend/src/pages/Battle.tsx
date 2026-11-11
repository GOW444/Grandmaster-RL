import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, type Puzzle, type RaceState } from '../api'
import PuzzleBoard from '../components/PuzzleBoard'
import { play as sfx } from '../sounds'

const DURATIONS = [
  { s: 180, label: '3 min' },
  { s: 300, label: '5 min' },
]

/* ----------------------------------------------------------------------------------------------
 * Lobby
 * --------------------------------------------------------------------------------------------*/

export function BattleLobby() {
  const nav = useNavigate()
  const [duration, setDuration] = useState(180)
  const [code, setCode] = useState('')
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [board, setBoard] = useState<{ username: string; score: number }[]>([])

  useEffect(() => {
    api.races.leaderboard(duration).then(setBoard).catch(() => setBoard([]))
  }, [duration])

  const go = async (what: string, fn: () => Promise<RaceState>) => {
    setBusy(what)
    setError(null)
    try {
      const r = await fn()
      nav(`/battle/${r.id}`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="grid lg:grid-cols-[1fr_340px] gap-6 items-start">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold">⚔️ Puzzle Battle</h1>
          <p className="text-ink-300 mt-1">
            Solve as many puzzles as you can before the clock runs out. Three mistakes and you’re out. Puzzles get harder as you go —
            everyone in a race gets the same ones.
          </p>
        </div>

        <div className="flex gap-2">
          {DURATIONS.map((d) => (
            <button
              key={d.s}
              onClick={() => setDuration(d.s)}
              className={`px-4 py-2 rounded-lg font-semibold ${duration === d.s ? 'bg-brand text-ink-950' : 'bg-ink-800 text-ink-300'}`}
            >
              ⏱ {d.label}
            </button>
          ))}
          <span className="self-center text-sm text-ink-500 ml-2">· 3 strikes</span>
        </div>

        <div className="grid sm:grid-cols-3 gap-4">
          <ModeCard
            icon="🌍"
            title="Quick match"
            desc="Race a random player online right now."
            cta={busy === 'quick' ? 'Finding…' : 'Find opponent'}
            onClick={() => go('quick', () => api.races.quick(duration))}
            disabled={!!busy}
            primary
          />
          <ModeCard
            icon="🤝"
            title="Challenge friends"
            desc="Private room — share the code, up to 8 players."
            cta={busy === 'private' ? 'Creating…' : 'Create room'}
            onClick={() => go('private', () => api.races.create('private', duration))}
            disabled={!!busy}
          />
          <ModeCard
            icon="🏃"
            title="Solo rush"
            desc="Just you vs the clock. Chase the leaderboard."
            cta={busy === 'solo' ? 'Starting…' : 'Start'}
            onClick={() => go('solo', () => api.races.create('solo', duration))}
            disabled={!!busy}
          />
        </div>

        <form
          className="card p-5 flex flex-wrap items-end gap-3"
          onSubmit={(e) => {
            e.preventDefault()
            if (code.trim()) go('join', () => api.races.join(code.trim()))
          }}
        >
          <div className="flex-1 min-w-48 space-y-1">
            <label className="label">Have a room code?</label>
            <input
              className="input font-mono uppercase tracking-widest"
              placeholder="ABC123"
              maxLength={8}
              value={code}
              onChange={(e) => setCode(e.target.value)}
            />
          </div>
          <button className="btn-ghost" disabled={!!busy || !code.trim()}>
            Join room
          </button>
        </form>
        {error && <p className="text-bad text-sm">{error}</p>}
      </div>

      <aside className="card p-5">
        <div className="label mb-3">Best scores · {duration / 60} min</div>
        {board.length === 0 ? (
          <p className="text-sm text-ink-500">No finished races yet. Be the first!</p>
        ) : (
          <ol className="space-y-2">
            {board.map((r, i) => (
              <li key={r.username} className="flex items-center gap-3 text-sm">
                <span className="w-6 text-ink-500 tabular-nums">{i < 3 ? ['🥇', '🥈', '🥉'][i] : i + 1}</span>
                <span className="flex-1 font-medium">{r.username}</span>
                <span className="font-bold tabular-nums">{r.score}</span>
              </li>
            ))}
          </ol>
        )}
      </aside>
    </div>
  )
}

function ModeCard(props: {
  icon: string
  title: string
  desc: string
  cta: string
  onClick: () => void
  disabled: boolean
  primary?: boolean
}) {
  return (
    <div className={`card p-5 flex flex-col ${props.primary ? 'border-brand/50' : ''}`}>
      <div className="text-3xl">{props.icon}</div>
      <div className="font-semibold mt-2">{props.title}</div>
      <div className="text-sm text-ink-500 flex-1 mt-1">{props.desc}</div>
      <button onClick={props.onClick} disabled={props.disabled} className={`${props.primary ? 'btn-primary' : 'btn-ghost'} mt-4`}>
        {props.cta}
      </button>
    </div>
  )
}

/* ----------------------------------------------------------------------------------------------
 * Join via shared link: /battle/join/:code
 * --------------------------------------------------------------------------------------------*/

export function BattleJoin() {
  const { code = '' } = useParams()
  const nav = useNavigate()
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    api.races
      .join(code)
      .then((r) => nav(`/battle/${r.id}`, { replace: true }))
      .catch((e) => setError((e as Error).message))
  }, [code, nav])
  return (
    <div className="card p-8 text-center">
      {error ? (
        <>
          <p className="text-bad">{error}</p>
          <Link to="/battle" className="btn-ghost mt-4">
            Back to lobby
          </Link>
        </>
      ) : (
        <p className="text-ink-300">Joining room {code.toUpperCase()}…</p>
      )}
    </div>
  )
}

/* ----------------------------------------------------------------------------------------------
 * Race room
 * --------------------------------------------------------------------------------------------*/

export function BattleRoom() {
  const { id } = useParams()
  const raceId = Number(id)
  const nav = useNavigate()
  const [race, setRace] = useState<RaceState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [puzzle, setPuzzle] = useState<Puzzle | null>(null)
  const [flash, setFlash] = useState<'good' | 'bad' | null>(null)
  const [now, setNow] = useState(Date.now())
  const offset = useRef(0) // server clock - local clock
  const submitting = useRef(false)

  const apply = useCallback((r: RaceState) => {
    offset.current = new Date(r.server_now).getTime() - Date.now()
    setRace(r)
    // Only swap the board when the puzzle actually changes (polling returns fresh objects).
    setPuzzle((cur) => (r.puzzle ? (cur?.id === r.puzzle.id ? cur : r.puzzle) : null))
  }, [])

  // Poll: fast while live, stop when finished.
  useEffect(() => {
    let alive = true
    let timer = 0
    const tick = async () => {
      try {
        const r = await api.races.get(raceId)
        if (!alive) return
        if (!submitting.current) apply(r)
        if (r.status !== 'finished') timer = window.setTimeout(tick, r.status === 'running' ? 1000 : 1500)
      } catch (e) {
        if (alive) setError((e as Error).message)
      }
    }
    tick()
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [raceId, apply])

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 200)
    return () => clearInterval(t)
  }, [])

  const serverNow = now + offset.current
  const startsIn = race?.started_at ? Math.ceil((new Date(race.started_at).getTime() - serverNow) / 1000) : null
  const left = race?.ends_at ? Math.max(0, new Date(race.ends_at).getTime() - serverNow) : null

  // Countdown beeps
  const lastBeep = useRef<number | null>(null)
  useEffect(() => {
    if (startsIn !== null && startsIn > 0 && startsIn <= 3 && lastBeep.current !== startsIn) {
      lastBeep.current = startsIn
      sfx('hint')
    }
  }, [startsIn])

  const onFinish = async (_solved: boolean, _ms: number, _hints: number, played: string[]) => {
    if (!race?.me || submitting.current) return
    submitting.current = true
    try {
      const r = await api.races.answer(race.id, race.me.index, played)
      setFlash(r.correct ? 'good' : 'bad')
      window.setTimeout(() => setFlash(null), 450)
      // brief pause so the result registers, then next puzzle
      window.setTimeout(() => {
        submitting.current = false
        apply(r)
      }, r.correct ? 250 : 650)
    } catch (e) {
      submitting.current = false
      setError((e as Error).message)
    }
  }

  const leave = async () => {
    await api.races.leave(raceId).catch(() => undefined)
    nav('/battle')
  }

  if (error && !race) return <div className="card p-8 text-bad">{error}</div>
  if (!race) return <div className="text-ink-500">Loading race…</div>

  const shareUrl = `${window.location.origin}/battle/join/${race.code}`

  /* Waiting room */
  if (race.status === 'waiting') {
    const searching = race.visibility === 'public'
    return (
      <div className="max-w-xl mx-auto card p-8 text-center space-y-5">
        {searching ? (
          <>
            <div className="text-5xl animate-pulse">🌍</div>
            <h2 className="text-2xl font-bold">Looking for an opponent…</h2>
            <p className="text-ink-300 text-sm">
              {race.duration_s / 60}-minute race. We’ll start as soon as someone joins.
            </p>
            <div className="flex gap-2 justify-center">
              <button className="btn-ghost" onClick={leave}>
                Cancel
              </button>
              <button
                className="btn-primary"
                onClick={async () => {
                  await api.races.leave(raceId).catch(() => undefined)
                  const r = await api.races.create('solo', race.duration_s)
                  nav(`/battle/${r.id}`, { replace: true })
                }}
              >
                Play solo instead
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="label">Room code</div>
            <div className="text-5xl font-mono font-bold tracking-[0.3em] text-brand">{race.code}</div>
            <div className="flex gap-2 justify-center">
              <button className="btn-ghost" onClick={() => navigator.clipboard?.writeText(shareUrl)}>
                📋 Copy invite link
              </button>
            </div>
            <div className="text-left">
              <div className="label mb-2">Players ({race.players.length}/8)</div>
              <ul className="space-y-1">
                {race.players.map((p) => (
                  <li key={p.username} className="flex items-center gap-2 rounded-lg bg-ink-800 px-3 py-2">
                    <span>👤</span>
                    <span className="font-medium">{p.username}</span>
                    {p.username === race.host && <span className="text-xs text-gold ml-auto">host</span>}
                  </li>
                ))}
              </ul>
            </div>
            <div className="flex gap-2 justify-center">
              <button className="btn-ghost" onClick={leave}>
                Leave
              </button>
              {race.is_host ? (
                <button className="btn-primary" onClick={() => api.races.start(raceId).then(apply)}>
                  ▶ Start race
                </button>
              ) : (
                <span className="self-center text-sm text-ink-500">Waiting for {race.host} to start…</span>
              )}
            </div>
          </>
        )}
      </div>
    )
  }

  /* Results */
  if (race.status === 'finished') {
    const me = race.players.find((p) => p.you)
    const rank = race.players.findIndex((p) => p.you) + 1
    return (
      <div className="max-w-xl mx-auto card p-8 space-y-6">
        <div className="text-center">
          <div className="text-5xl">{race.players.length > 1 ? (rank === 1 ? '🏆' : rank === 2 ? '🥈' : '🎯') : '🏁'}</div>
          <h2 className="text-2xl font-bold mt-2">
            {race.players.length > 1 ? (rank === 1 ? 'You won!' : `You placed #${rank}`) : 'Time!'}
          </h2>
          <p className="text-ink-300 mt-1">
            <span className="text-3xl font-bold text-ink-100">{me?.score ?? 0}</span> solved
            {me?.best_rating ? <> · hardest {me.best_rating}</> : null}
          </p>
        </div>
        <Scoreboard race={race} />
        <div className="flex gap-2 justify-center">
          <Link to="/battle" className="btn-ghost">
            Lobby
          </Link>
          <button
            className="btn-primary"
            onClick={async () => {
              const r =
                race.visibility === 'public'
                  ? await api.races.quick(race.duration_s)
                  : await api.races.create(race.visibility === 'solo' ? 'solo' : 'private', race.duration_s)
              nav(`/battle/${r.id}`)
            }}
          >
            Play again
          </button>
        </div>
      </div>
    )
  }

  /* Running */
  const finishedMe = race.me?.finished
  return (
    <div className="grid lg:grid-cols-[minmax(0,560px)_1fr] gap-6 items-start">
      <div className="relative">
        {startsIn !== null && startsIn > 0 ? (
          <div className="w-full aspect-square card flex flex-col items-center justify-center">
            <div className="label">Get ready</div>
            <div className="text-8xl font-bold text-brand tabular-nums">{startsIn}</div>
          </div>
        ) : finishedMe ? (
          <div className="w-full aspect-square card flex flex-col items-center justify-center text-center p-6">
            <div className="text-5xl">{race.me!.errors >= race.max_errors ? '💥' : '✅'}</div>
            <div className="text-xl font-bold mt-2">
              {race.me!.errors >= race.max_errors ? 'Three strikes!' : 'Done!'} You solved {race.me!.score}.
            </div>
            <div className="text-ink-500 mt-1">Waiting for the others to finish…</div>
          </div>
        ) : puzzle ? (
          <>
            <PuzzleBoard key={puzzle.id} puzzle={puzzle} onFinish={onFinish} introMs={250} />
            {flash && (
              <div
                className={`pointer-events-none absolute inset-0 rounded-lg ring-8 ${flash === 'good' ? 'ring-brand/70' : 'ring-bad/70'}`}
              />
            )}
          </>
        ) : (
          <div className="w-full aspect-square card flex items-center justify-center text-ink-500">Loading…</div>
        )}
      </div>

      <aside className="space-y-4">
        <div className="card p-5">
          <div className="flex items-end justify-between">
            <div>
              <div className="label">Time left</div>
              <div className={`text-5xl font-mono font-bold tabular-nums ${left !== null && left < 30000 ? 'text-bad' : ''}`}>
                {left === null ? '--:--' : `${Math.floor(left / 60000)}:${String(Math.floor(left / 1000) % 60).padStart(2, '0')}`}
              </div>
            </div>
            <div className="text-right">
              <div className="label">Score</div>
              <div className="text-5xl font-bold tabular-nums text-brand">{race.me?.score ?? 0}</div>
            </div>
          </div>
          <div className="flex items-center gap-2 mt-4">
            <span className="label mr-1">Strikes</span>
            {Array.from({ length: race.max_errors }, (_, i) => (
              <span
                key={i}
                className={`w-8 h-8 rounded-md flex items-center justify-center font-bold ${
                  i < (race.me?.errors ?? 0) ? 'bg-bad text-ink-950' : 'bg-ink-800 text-ink-700'
                }`}
              >
                ✕
              </span>
            ))}
          </div>
          {puzzle && !finishedMe && (startsIn ?? 0) <= 0 && (
            <div className="text-xs text-ink-500 mt-3">
              Puzzle #{(race.me?.index ?? 0) + 1} · {puzzle.rating}
            </div>
          )}
        </div>

        {race.players.length > 1 && (
          <div className="card p-5">
            <div className="label mb-3">Live standings</div>
            <Scoreboard race={race} />
          </div>
        )}
        <button className="btn-ghost w-full" onClick={leave}>
          Resign
        </button>
      </aside>
    </div>
  )
}

function Scoreboard({ race }: { race: RaceState }) {
  const top = Math.max(1, ...race.players.map((p) => p.score))
  return (
    <ul className="space-y-2">
      {race.players.map((p, i) => (
        <li key={p.username} className={`rounded-lg px-3 py-2 ${p.you ? 'bg-brand/10 border border-brand/30' : 'bg-ink-800'}`}>
          <div className="flex items-center gap-2 text-sm">
            <span className="w-5 text-ink-500 tabular-nums">{i + 1}</span>
            <span className="font-medium flex-1">
              {p.username}
              {p.you && <span className="text-ink-500"> (you)</span>}
            </span>
            <span className="text-bad text-xs">{'✕'.repeat(p.errors)}</span>
            {p.finished && <span className="text-xs text-ink-500">done</span>}
            <span className="font-bold tabular-nums w-8 text-right">{p.score}</span>
          </div>
          <div className="h-1.5 mt-2 rounded-full bg-ink-950 overflow-hidden">
            <div className="h-full bg-brand transition-all" style={{ width: `${(100 * p.score) / top}%` }} />
          </div>
        </li>
      ))}
    </ul>
  )
}
