import { useEffect, useState } from 'react'
import { api } from '../api'
import { useAuth } from '../auth'

type By = 'rating' | 'xp' | 'streak'

export default function Leaderboard() {
  const { user } = useAuth()
  const [by, setBy] = useState<By>('rating')
  const [rows, setRows] = useState<Awaited<ReturnType<typeof api.leaderboard>>>([])

  useEffect(() => {
    api.leaderboard(by).then(setRows)
  }, [by])

  return (
    <div className="space-y-4 max-w-2xl">
      <h1 className="text-2xl font-bold">Leaderboard</h1>
      <div className="flex gap-2">
        {(['rating', 'xp', 'streak'] as By[]).map((b) => (
          <button
            key={b}
            onClick={() => setBy(b)}
            className={`px-3 py-1 rounded-full text-sm capitalize ${b === by ? 'bg-brand text-ink-950 font-semibold' : 'bg-ink-800 text-ink-300'}`}
          >
            {b}
          </button>
        ))}
      </div>
      <div className="card divide-y divide-ink-800">
        {rows.map((r, i) => (
          <div
            key={r.username}
            className={`flex items-center gap-4 px-5 py-3 ${r.username === user?.username ? 'bg-brand/10' : ''}`}
          >
            <span className="w-8 text-lg font-bold text-ink-500 tabular-nums">
              {i < 3 ? ['🥇', '🥈', '🥉'][i] : i + 1}
            </span>
            <span className="flex-1 font-semibold">{r.username}</span>
            <span className={`tabular-nums ${by === 'rating' ? 'text-ink-100 font-semibold' : 'text-ink-500'}`}>{r.rating}</span>
            <span className={`tabular-nums w-20 text-right ${by === 'xp' ? 'text-ink-100 font-semibold' : 'text-ink-500'}`}>{r.xp} XP</span>
            <span className={`tabular-nums w-12 text-right ${by === 'streak' ? 'text-ink-100 font-semibold' : 'text-ink-500'}`}>🔥{r.streak}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
