interface Props {
  data: { day: string; attempts: number; solved: number }[]
  weeks?: number
}

const color = (n: number) =>
  n === 0 ? '#1c2430' : n < 3 ? '#3b5228' : n < 8 ? '#5d7d38' : n < 15 ? '#7fa650' : '#a6d06f'

const iso = (d: Date) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`

export default function ActivityHeatmap({ data, weeks = 26 }: Props) {
  const byDay = new Map(data.map((d) => [d.day, d]))
  const today = new Date()
  const start = new Date(today)
  start.setDate(today.getDate() - (weeks * 7 - 1) - today.getDay())

  const cols: { date: Date; n: number; s: number }[][] = []
  const cur = new Date(start)
  while (cur <= today) {
    const week: { date: Date; n: number; s: number }[] = []
    for (let i = 0; i < 7 && cur <= today; i++) {
      const rec = byDay.get(iso(cur))
      week.push({ date: new Date(cur), n: rec?.attempts ?? 0, s: rec?.solved ?? 0 })
      cur.setDate(cur.getDate() + 1)
    }
    cols.push(week)
  }

  return (
    <div className="overflow-x-auto">
      <div className="flex gap-[3px] w-max">
        {cols.map((w, i) => (
          <div key={i} className="flex flex-col gap-[3px]">
            {w.map((d) => (
              <div
                key={d.date.toISOString()}
                className="w-3 h-3 rounded-[3px]"
                style={{ background: color(d.n) }}
                title={`${d.date.toDateString()}: ${d.n} puzzles, ${d.s} solved`}
              />
            ))}
          </div>
        ))}
      </div>
      <div className="flex items-center gap-1 text-xs text-ink-500 mt-3">
        Less
        {[0, 2, 5, 10, 20].map((n) => (
          <span key={n} className="w-3 h-3 rounded-[3px] inline-block" style={{ background: color(n) }} />
        ))}
        More
      </div>
    </div>
  )
}
