import { Chess } from 'chess.js'
import type { Puzzle } from './api'

/** Position the solver actually sees (after the opponent's setup move), and their orientation. */
export function startPosition(p: Pick<Puzzle, 'fen' | 'moves'>) {
  const g = new Chess(p.fen)
  const m = p.moves[0]
  g.move({ from: m.slice(0, 2), to: m.slice(2, 4), promotion: m[4] })
  return { fen: g.fen(), orientation: (g.turn() === 'w' ? 'white' : 'black') as 'white' | 'black' }
}
