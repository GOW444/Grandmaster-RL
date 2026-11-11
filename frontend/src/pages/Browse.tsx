import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api, THEMES, type BrowseItem } from '../api'
import MiniBoard from '../components/MiniBoard'
import RatingRange from '../components/RatingRange'

const STATUS = [
  ['all', 'All'],
  ['unseen', 'Unseen'],
  ['solved', 'Solved'],
  ['failed', 'Missed'],
] as const

export default function Browse() {
  const [params, setParams] = useSearchParams()
  const min = Number(params.get('min') ?? 1200)
  const max = Number(params.get('max') ?? 1600)
  const theme = params.get('theme') ?? ''
  const status = params.get('status') ?? 'all'
  const sort = params.get('sort') ?? 'popular'
  const page = Number(params.get('page') ?? 1)

  const [data, setData] = useState<{ total: number; items: BrowseItem[] } | null>(null)
  const [error, setError] = useState<string | null>(null)

  const update = (patch: Record<string, string | number>) => {
    const next = new URLSearchParams(params)
    for (const [k, v] of Object.entries(patch)) next.set(k, String(v))
    if (!('page' in patch)) next.set('page', '1')
    setParams(next, { replace: true })
  }

  useEffect(() => {
    setError(null)
    const t = setTimeout(() => {
      api
        .browse({ min, max, theme: theme || undefined, status, sort, page })
        .then(setData)
        .catch((e) => setError((e as Error).message))
    }, 200) // debounce slider drags
    return () => clearTimeout(t)
  }, [min, max, theme, status, sort, page])

  const pages = data ? Math.max(1, Math.ceil(data.total / 24)) : 1
  const trainHref = `/train?mode=custom&min=${min}&max=${max}${theme ? `&theme=${theme}` : ''}`

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">Browse puzzles</h1>
          <p className="text-ink-500 text-sm">Pick any rating range and theme, then play a specific puzzle or train the whole range.</p>
        </div>
        <Link to={trainHref} className="btn-primary">
          ▶ Train this range
        </Link>
      </div>

      <div className="card p-5 grid lg:grid-cols-[1.4fr_1fr] gap-6">
        <RatingRange min={min} max={max} onChange={(lo, hi) => update({ min: lo, max: hi })} />
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="label">Theme</div>
            <div className="flex flex-wrap gap-2">
              {['', ...THEMES].map((t) => (
                <button
                  key={t || 'all'}
                  onClick={() => update({ theme: t })}
                  className={`px-3 py-1 rounded-full text-sm capitalize ${
                    theme === t ? 'bg-brand text-ink-950 font-semibold' : 'bg-ink-800 text-ink-300'
                  }`}
                >
                  {t || 'All'}
                </button>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap gap-6">
            <div className="space-y-2">
              <div className="label">Status</div>
              <div className="flex gap-1">
                {STATUS.map(([v, l]) => (
                  <button
                    key={v}
                    onClick={() => update({ status: v })}
                    className={`px-3 py-1 rounded-md text-sm ${status === v ? 'bg-ink-700 text-ink-100' : 'text-ink-500 hover:text-ink-300'}`}
                  >
                    {l}
                  </button>
                ))}
              </div>
            </div>
            <div className="space-y-2">
              <div className="label">Sort</div>
              <select className="input py-1" value={sort} onChange={(e) => update({ sort: e.target.value })}>
                <option value="popular">Most popular</option>
                <option value="rating">Rating ↑</option>
                <option value="-rating">Rating ↓</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {error && <p className="text-bad text-sm">{error}</p>}
      <div className="text-sm text-ink-500">{data ? `${data.total.toLocaleString()} puzzles` : 'Loading…'}</div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        {data?.items.map((p) => (
          <Link key={p.id} to={`/train?puzzle=${p.id}`} className="card p-2 hover:border-brand transition-colors block">
            <MiniBoard puzzle={p} />
            <div className="flex items-center justify-between mt-2 px-1 text-sm">
              <span className="font-semibold tabular-nums">{p.rating}</span>
              <span
                className={`text-xs ${p.status === 'solved' ? 'text-brand' : p.status === 'failed' ? 'text-bad' : 'text-ink-500'}`}
              >
                {p.status === 'solved' ? '✓ solved' : p.status === 'failed' ? '✗ missed' : 'new'}
              </span>
            </div>
            <div className="text-xs text-ink-500 capitalize px-1">{p.primary_theme}</div>
          </Link>
        ))}
      </div>

      {data && data.total > 24 && (
        <div className="flex items-center justify-center gap-3">
          <button className="btn-ghost" disabled={page <= 1} onClick={() => update({ page: page - 1 })}>
            ← Prev
          </button>
          <span className="text-sm text-ink-300 tabular-nums">
            Page {page} / {pages}
          </span>
          <button className="btn-ghost" disabled={page >= pages} onClick={() => update({ page: page + 1 })}>
            Next →
          </button>
        </div>
      )}
    </div>
  )
}

