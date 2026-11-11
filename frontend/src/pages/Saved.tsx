import { Chess } from 'chess.js'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Chessboard } from 'react-chessboard'
import { api, type Puzzle } from '../api'

function startPosition(p: Puzzle) {
  const g = new Chess(p.fen)
  const m = p.moves[0]
  g.move({ from: m.slice(0, 2), to: m.slice(2, 4), promotion: m[4] })
  return { fen: g.fen(), orientation: (g.turn() === 'w' ? 'white' : 'black') as 'white' | 'black' }
}

export default function Saved() {
  const [items, setItems] = useState<Awaited<ReturnType<typeof api.saved>> | null>(null)

  useEffect(() => {
    api.saved().then(setItems)
  }, [])

  const remove = async (id: string) => {
    await api.unsave(id)
    setItems((xs) => xs?.filter((x) => x.puzzle.id !== id) ?? null)
  }

  if (!items) return <div className="text-ink-500">Loading…</div>
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Saved puzzles</h1>
      {items.length === 0 && (
        <div className="card p-8 text-center text-ink-300">
          Nothing saved yet. Hit <b>☆ Save</b> on any puzzle to build your collection.
        </div>
      )}
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {items.map(({ puzzle, created_at }) => {
          const { fen, orientation } = startPosition(puzzle)
          return (
            <div key={puzzle.id} className="card p-3 space-y-2">
              <Link to={`/train?puzzle=${puzzle.id}`} className="block">
                <Chessboard
                  id={`saved-${puzzle.id}`}
                  position={fen}
                  boardOrientation={orientation}
                  arePiecesDraggable={false}
                  customDarkSquareStyle={{ backgroundColor: '#769656' }}
                  customLightSquareStyle={{ backgroundColor: '#eeeed2' }}
                  customBoardStyle={{ borderRadius: '6px' }}
                />
              </Link>
              <div className="flex items-center justify-between text-sm">
                <div>
                  <span className="font-semibold">{puzzle.rating}</span>
                  <span className="text-ink-500 capitalize"> · {puzzle.primary_theme}</span>
                </div>
                <button onClick={() => remove(puzzle.id)} className="text-ink-500 hover:text-bad text-xs">
                  Remove
                </button>
              </div>
              <div className="text-xs text-ink-500">Saved {new Date(created_at).toLocaleDateString()}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
