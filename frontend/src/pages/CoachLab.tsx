import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type ABStats } from '../api'

const NAMES: Record<string, string> = { ppo: 'PPO policy', rating_match: 'Rating-match baseline' }

// Offline results from the `engine` branch (simulated learner, 20 episodes; reproduced from checkpoints).
const OFFLINE = [
  { agent: 'PPO', lei: 0.0016, success: 0.697, drho: 4.63 },
  { agent: 'FixedProgression', lei: 0.0013, success: 0.536, drho: 5.57 },
  { agent: 'RatingMatch', lei: 0.0012, success: 0.419, drho: 6.48 },
  { agent: 'SAC', lei: 0.0005, success: 0.133, drho: 11.34 },
  { agent: 'Random', lei: 0.0, success: 0.233, drho: 2.57 },
]

export default function CoachLab() {
  const [ab, setAb] = useState<{ you: Record<string, ABStats>; global: Record<string, ABStats> } | null>(null)
  useEffect(() => {
    api.coachAB().then(setAb)
  }, [])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Coach Lab</h1>
        <p className="text-ink-300 mt-1 max-w-3xl">
          The AI Coach is a PPO agent trained to build puzzle curricula for a simulated learner (IRT model). Here it
          runs on <i>real</i> people: your six theme ratings and last-10 results become its observation, and it outputs
          a theme + target difficulty. In A/B mode each puzzle is randomly assigned to PPO or a Lichess-style
          rating-match baseline so we can compare them on real attempts.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {(['you', 'global'] as const).map((scope) => (
          <div key={scope} className="card p-5">
            <div className="label mb-3">{scope === 'you' ? 'Your A/B results' : 'All users'}</div>
            <ABTable data={ab?.[scope]} />
          </div>
        ))}
      </div>

      <div className="card p-5">
        <div className="label mb-1">Offline evaluation (simulated learner, held-out fatigue + noise env)</div>
        <p className="text-xs text-ink-500 mb-3">LEI = (rating gain / steps) × success rate × difficulty consistency. Higher is better.</p>
        <div className="space-y-2">
          {OFFLINE.map((r) => (
            <div key={r.agent} className="grid grid-cols-[140px_1fr_70px_90px] items-center gap-3 text-sm">
              <span className={r.agent === 'PPO' ? 'font-semibold text-brand' : 'text-ink-300'}>{r.agent}</span>
              <div className="h-2.5 bg-ink-800 rounded-full overflow-hidden">
                <div className={`h-full ${r.agent === 'PPO' ? 'bg-brand' : 'bg-ink-500'}`} style={{ width: `${(r.lei / 0.0016) * 100}%` }} />
              </div>
              <span className="tabular-nums text-right">{r.lei.toFixed(4)}</span>
              <span className="tabular-nums text-right text-ink-500">{(r.success * 100).toFixed(0)}% solve</span>
            </div>
          ))}
        </div>
      </div>

      <Link to="/train?mode=coach&selector=ab" className="btn-primary">
        ▶ Train with A/B coach
      </Link>
    </div>
  )
}

function ABTable({ data }: { data?: Record<string, ABStats> }) {
  if (!data) return <div className="text-ink-500 text-sm">Loading…</div>
  if (Object.keys(data).length === 0)
    return <div className="text-ink-500 text-sm">No coach attempts yet — play some AI Coach puzzles.</div>
  return (
    <table className="w-full text-sm">
      <thead className="text-ink-500 text-left">
        <tr>
          <th className="py-1 font-medium">Selector</th>
          <th className="font-medium">n</th>
          <th className="font-medium">Solve rate</th>
          <th className="font-medium">Avg Δ theme</th>
          <th className="font-medium" title="Puzzle rating minus your theme rating">Stretch</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(data).map(([k, v]) => (
          <tr key={k} className="border-t border-ink-800">
            <td className="py-2 font-medium">{NAMES[k] ?? k}</td>
            <td className="tabular-nums">{v.attempts}</td>
            <td className="tabular-nums">{(v.success_rate * 100).toFixed(0)}%</td>
            <td className={`tabular-nums ${v.avg_rating_gain >= 0 ? 'text-brand' : 'text-bad'}`}>
              {v.avg_rating_gain >= 0 ? '+' : ''}
              {v.avg_rating_gain.toFixed(1)}
            </td>
            <td className="tabular-nums">{v.avg_stretch > 0 ? '+' : ''}{Math.round(v.avg_stretch)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
