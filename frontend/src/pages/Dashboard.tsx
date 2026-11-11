import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, type Badge, type Overview } from '../api'
import ActivityHeatmap from '../components/ActivityHeatmap'

const tooltipStyle = {
  contentStyle: { background: '#141a22', border: '1px solid #2a3443', borderRadius: 8, fontSize: 12 },
  labelStyle: { color: '#aab6c8' },
}

export default function Dashboard() {
  const [ov, setOv] = useState<Overview | null>(null)
  const [history, setHistory] = useState<{ i: number; rating: number }[]>([])
  const [activity, setActivity] = useState<{ day: string; attempts: number; solved: number }[]>([])
  const [buckets, setBuckets] = useState<{ label: string; attempts: number; accuracy: number }[]>([])
  const [badges, setBadges] = useState<Badge[]>([])

  useEffect(() => {
    api.overview().then(setOv)
    api.ratingHistory().then((h) => setHistory([{ i: 0, rating: 1000 }, ...h.map((r, i) => ({ i: i + 1, rating: r.rating }))]))
    api.activity().then(setActivity)
    api.timeBuckets().then(setBuckets)
    api.badges().then(setBadges)
  }, [])

  if (!ov) return <div className="text-ink-500">Loading…</div>
  const { user, totals } = ov
  const accuracy = totals.attempts ? totals.solved / totals.attempts : 0

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="grid md:grid-cols-[1.4fr_1fr_1fr_1fr] gap-4">
        <div className="card p-5">
          <div className="label">Welcome back</div>
          <div className="text-2xl font-bold">{user.username}</div>
          <div className="mt-3 flex items-center gap-3">
            <span className="text-sm font-semibold text-gold">Lv {user.level}</span>
            <div className="flex-1 h-2 rounded-full bg-ink-800 overflow-hidden">
              <div
                className="h-full bg-gold"
                style={{ width: `${(100 * user.xp_into_level) / Math.max(user.xp_for_next, 1)}%` }}
              />
            </div>
            <span className="text-xs text-ink-500 tabular-nums">
              {user.xp_into_level}/{user.xp_for_next} XP
            </span>
          </div>
          <Link to="/train" className="btn-primary mt-4 w-full">
            ▶ Continue training
          </Link>
        </div>
        <BigStat label="Puzzle rating" value={Math.round(user.rating)} sub="mean of 6 theme ratings" />
        <BigStat label="Streak" value={`🔥 ${user.current_streak}`} sub={`best ${user.longest_streak} days`} />
        <BigStat
          label="Solved"
          value={totals.solved}
          sub={`${(accuracy * 100).toFixed(0)}% acc · ${(totals.avg_time_ms / 1000).toFixed(0)}s avg`}
        />
      </div>

      {/* Focus areas */}
      <div className="card p-5">
        <div className="label mb-3">Areas to improve</div>
        {ov.focus.length === 0 && ov.untried.length === 6 ? (
          <p className="text-ink-300 text-sm">Solve a few puzzles and we’ll show where you’re losing points.</p>
        ) : (
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {ov.focus.map((f) => (
              <FocusCard key={f.theme + f.why} theme={f.theme} why={f.why} />
            ))}
            {ov.untried.slice(0, 4 - ov.focus.length).map((t) => (
              <FocusCard key={t} theme={t} why="Not tried yet" />
            ))}
          </div>
        )}
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <div className="card p-5">
          <div className="label mb-2">Theme ratings</div>
          <div className="h-72">
            <ResponsiveContainer>
              <RadarChart data={ov.themes} outerRadius="75%">
                <PolarGrid stroke="#2a3443" />
                <PolarAngleAxis dataKey="theme" tick={{ fill: '#aab6c8', fontSize: 12 }} />
                <PolarRadiusAxis
                  domain={[Math.min(800, ...ov.themes.map((t) => t.rating - 50)), 'auto']}
                  tick={false}
                  axisLine={false}
                />
                <Radar dataKey="rating" stroke="#7fa650" fill="#7fa650" fillOpacity={0.35} />
                <Tooltip {...tooltipStyle} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card p-5">
          <div className="label mb-2">Rating progress</div>
          <div className="h-72">
            {history.length > 1 ? (
              <ResponsiveContainer>
                <LineChart data={history}>
                  <CartesianGrid stroke="#1c2430" />
                  <XAxis dataKey="i" tick={{ fill: '#6b7a90', fontSize: 11 }} />
                  <YAxis domain={['auto', 'auto']} tick={{ fill: '#6b7a90', fontSize: 11 }} width={40} />
                  <Tooltip {...tooltipStyle} labelFormatter={(i) => `Puzzle #${i}`} />
                  <Line dataKey="rating" stroke="#7fa650" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <Empty />
            )}
          </div>
        </div>
      </div>

      <div className="card p-5 overflow-x-auto">
        <div className="label mb-3">By theme</div>
        <table className="w-full text-sm">
          <thead className="text-ink-500 text-left">
            <tr>
              <th className="py-2 font-medium">Theme</th>
              <th className="font-medium">Rating</th>
              <th className="font-medium">Attempts</th>
              <th className="font-medium">Accuracy</th>
              <th className="font-medium">Avg time</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {ov.themes.map((t) => (
              <tr key={t.theme} className="border-t border-ink-800">
                <td className="py-2 capitalize font-medium">{t.theme}</td>
                <td className="tabular-nums">{Math.round(t.rating)}</td>
                <td className="tabular-nums">{t.attempts}</td>
                <td>
                  {t.accuracy === null ? (
                    <span className="text-ink-500">—</span>
                  ) : (
                    <div className="flex items-center gap-2">
                      <div className="w-20 h-1.5 bg-ink-800 rounded-full overflow-hidden">
                        <div className="h-full bg-brand" style={{ width: `${t.accuracy * 100}%` }} />
                      </div>
                      <span className="tabular-nums">{Math.round(t.accuracy * 100)}%</span>
                    </div>
                  )}
                </td>
                <td className="tabular-nums">{t.avg_time_ms === null ? '—' : `${(t.avg_time_ms / 1000).toFixed(1)}s`}</td>
                <td className="text-right">
                  <Link to={`/train?mode=theme&theme=${t.theme}`} className="text-brand text-xs font-semibold">
                    Drill →
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid lg:grid-cols-[1.5fr_1fr] gap-6">
        <div className="card p-5">
          <div className="label mb-3">Activity (last 26 weeks)</div>
          <ActivityHeatmap data={activity} />
        </div>
        <div className="card p-5">
          <div className="label mb-2">Accuracy vs time spent</div>
          <div className="h-48">
            {buckets.length ? (
              <ResponsiveContainer>
                <BarChart data={buckets}>
                  <XAxis dataKey="label" tick={{ fill: '#6b7a90', fontSize: 10 }} />
                  <YAxis domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} tick={{ fill: '#6b7a90', fontSize: 10 }} width={36} />
                  <Tooltip {...tooltipStyle} formatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                  <Bar dataKey="accuracy" fill="#e3b341" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Empty />
            )}
          </div>
          <p className="text-xs text-ink-500 mt-2">Low accuracy on fast solves usually means you’re rushing.</p>
        </div>
      </div>

      <div className="card p-5">
        <div className="label mb-3">Badges</div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {badges.map((b) => (
            <div
              key={b.code}
              className={`rounded-lg border p-3 ${b.earned_at ? 'border-gold/40 bg-gold/5' : 'border-ink-800 opacity-45'}`}
              title={b.desc}
            >
              <div className="text-2xl">{b.icon}</div>
              <div className="text-sm font-semibold mt-1">{b.name}</div>
              <div className="text-xs text-ink-500">{b.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function BigStat({ label, value, sub }: { label: string; value: string | number; sub: string }) {
  return (
    <div className="card p-5">
      <div className="label">{label}</div>
      <div className="text-3xl font-bold mt-1 tabular-nums">{value}</div>
      <div className="text-xs text-ink-500 mt-1">{sub}</div>
    </div>
  )
}

function FocusCard({ theme, why }: { theme: string; why: string }) {
  return (
    <Link
      to={`/train?mode=theme&theme=${theme}`}
      className="rounded-lg border border-ink-800 hover:border-brand p-3 transition-colors block"
    >
      <div className="font-semibold capitalize">{theme}</div>
      <div className="text-xs text-ink-500">{why}</div>
      <div className="text-xs text-brand mt-2 font-semibold">Drill now →</div>
    </Link>
  )
}

function Empty() {
  return <div className="h-full flex items-center justify-center text-sm text-ink-500">No data yet</div>
}
