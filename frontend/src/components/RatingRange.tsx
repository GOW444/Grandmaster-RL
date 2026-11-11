// eslint-disable-next-line react-refresh/only-export-components
export const RANGE_PRESETS: [string, number, number][] = [
  ['Beginner', 400, 1000],
  ['Club', 1000, 1400],
  ['Intermediate', 1400, 1800],
  ['Advanced', 1800, 2200],
  ['Expert', 2200, 3000],
]

const LO = 400
const HI = 3000

export default function RatingRange({
  min,
  max,
  onChange,
}: {
  min: number
  max: number
  onChange: (min: number, max: number) => void
}) {
  const pct = (v: number) => ((v - LO) / (HI - LO)) * 100
  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <div className="label">Rating range</div>
        <div className="font-semibold tabular-nums">
          {min} – {max}
        </div>
      </div>
      <div className="relative h-6">
        <div className="absolute top-1/2 -translate-y-1/2 h-1.5 w-full rounded-full bg-ink-800" />
        <div
          className="absolute top-1/2 -translate-y-1/2 h-1.5 rounded-full bg-brand"
          style={{ left: `${pct(min)}%`, width: `${pct(max) - pct(min)}%` }}
        />
        <input
          type="range"
          aria-label="Minimum rating"
          min={LO}
          max={HI}
          step={50}
          value={min}
          onChange={(e) => onChange(Math.min(Number(e.target.value), max - 50), max)}
          className="range-thumb absolute inset-0 w-full"
        />
        <input
          type="range"
          aria-label="Maximum rating"
          min={LO}
          max={HI}
          step={50}
          value={max}
          onChange={(e) => onChange(min, Math.max(Number(e.target.value), min + 50))}
          className="range-thumb absolute inset-0 w-full"
        />
      </div>
      <div className="flex flex-wrap gap-2">
        {RANGE_PRESETS.map(([label, lo, hi]) => (
          <button
            key={label}
            onClick={() => onChange(lo, hi)}
            className={`px-3 py-1 rounded-full text-xs ${
              min === lo && max === hi ? 'bg-brand text-ink-950 font-semibold' : 'bg-ink-800 text-ink-300'
            }`}
          >
            {label} <span className="opacity-70">{lo}–{hi}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
