import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type AttemptRow } from '../api'

export default function History() {
  const [rows, setRows] = useState<AttemptRow[] | null>(null)
  useEffect(() => {
    api.history(200).then(setRows)
  }, [])

  if (!rows) return <div className="text-ink-500">Loading…</div>
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">History</h1>
      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-ink-500 text-left">
            <tr>
              {['When', 'Puzzle', 'Theme', 'Rating', 'Mode', 'Result', 'Time', 'Your rating'].map((h) => (
                <th key={h} className="px-4 py-3 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center text-ink-500">
                  No attempts yet.
                </td>
              </tr>
            )}
            {rows.map((r) => (
              <tr key={r.id} className="border-t border-ink-800 hover:bg-ink-800/40">
                <td className="px-4 py-2 text-ink-500 whitespace-nowrap">{new Date(r.created_at).toLocaleString()}</td>
                <td className="px-4">
                  <Link to={`/train?puzzle=${r.puzzle_id}`} className="text-brand font-mono">
                    {r.puzzle_id}
                  </Link>
                </td>
                <td className="px-4 capitalize">{r.theme}</td>
                <td className="px-4 tabular-nums">{r.puzzle_rating}</td>
                <td className="px-4 text-ink-300">
                  {r.mode}
                  {r.selector && <span className="text-ink-500"> ({r.selector === 'ppo' ? 'PPO' : 'RM'})</span>}
                </td>
                <td className={`px-4 font-semibold ${r.solved ? 'text-brand' : 'text-bad'}`}>{r.solved ? '✓' : '✗'}</td>
                <td className="px-4 tabular-nums">{(r.time_ms / 1000).toFixed(1)}s</td>
                <td className="px-4 tabular-nums">{Math.round(r.rating_after)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
