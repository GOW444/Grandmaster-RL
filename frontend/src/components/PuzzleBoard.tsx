import { Chess, type Square } from 'chess.js'
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { Chessboard } from 'react-chessboard'
import type { Puzzle } from '../api'
import { play as sfx, playMove } from '../sounds'

export type BoardStatus = 'intro' | 'playing' | 'solved' | 'failed'

interface Props {
  puzzle: Puzzle
  onFinish?: (solved: boolean, timeMs: number, hints: number) => void
  hintSignal?: number // increment to request a hint
  showSolutionSignal?: number // increment to animate the remaining solution
  onStatus?: (s: BoardStatus, info: { hints: number; toMove: 'white' | 'black' }) => void
}

const LAST = { background: 'rgba(227,179,65,0.35)' }
const SEL = { background: 'rgba(127,166,80,0.55)' }
const HINT = { boxShadow: 'inset 0 0 0 4px rgba(127,166,80,0.9)' }
const BAD = { background: 'rgba(229,83,75,0.55)' }
const GOOD = { background: 'rgba(127,166,80,0.55)' }

const uciParts = (uci: string) => ({
  from: uci.slice(0, 2) as Square,
  to: uci.slice(2, 4) as Square,
  promotion: uci.length > 4 ? uci[4] : undefined,
})

export default function PuzzleBoard({ puzzle, onFinish, hintSignal = 0, showSolutionSignal = 0, onStatus }: Props) {
  const game = useRef(new Chess(puzzle.fen))
  const [fen, setFen] = useState(puzzle.fen)
  const [idx, setIdx] = useState(0) // index of the next expected move in puzzle.moves
  const [status, setStatus] = useState<BoardStatus>('intro')
  const [last, setLast] = useState<{ from: Square; to: Square; bad?: boolean; good?: boolean } | null>(null)
  const [selected, setSelected] = useState<Square | null>(null)
  const [hints, setHints] = useState(0)
  const startedAt = useRef(0)
  const failedAtFen = useRef<string | null>(null)
  const timers = useRef<number[]>([])

  const toMove: 'white' | 'black' = puzzle.fen.split(' ')[1] === 'w' ? 'black' : 'white'

  const later = (fn: () => void, ms: number) => {
    timers.current.push(window.setTimeout(fn, ms))
  }

  const play = useCallback((uci: string) => {
    const m = game.current.move(uciParts(uci))
    playMove(m)
    setFen(game.current.fen())
    setLast({ from: m.from, to: m.to })
  }, [])

  // Reset on new puzzle, then play the opponent's setup move.
  useEffect(() => {
    timers.current.forEach(clearTimeout)
    timers.current = []
    game.current = new Chess(puzzle.fen)
    setFen(puzzle.fen)
    setIdx(0)
    setStatus('intro')
    setLast(null)
    setSelected(null)
    setHints(0)
    failedAtFen.current = null
    later(() => {
      play(puzzle.moves[0])
      setIdx(1)
      setStatus('playing')
      startedAt.current = performance.now()
    }, 600)
    return () => timers.current.forEach(clearTimeout)
  }, [puzzle, play])

  useEffect(() => {
    onStatus?.(status, { hints, toMove })
  }, [status, hints, toMove, onStatus])

  const finish = (solved: boolean) => {
    window.setTimeout(() => sfx(solved ? 'success' : 'fail'), 150)
    setStatus(solved ? 'solved' : 'failed')
    onFinish?.(solved, Math.round(performance.now() - startedAt.current), hints)
  }

  const tryMove = (from: Square, to: Square, promotion?: string): boolean => {
    if (status !== 'playing') return false
    const before = game.current.fen()
    let m
    try {
      m = game.current.move({ from, to, promotion: promotion ?? 'q' })
    } catch {
      return false
    }
    setSelected(null)
    playMove(m)
    const expected = puzzle.moves[idx]
    const uci = m.from + m.to + (m.promotion ?? '')
    // Lichess accepts any mating move as correct, even if it differs from the stored line.
    const correct = uci === expected || game.current.isCheckmate()
    setFen(game.current.fen())
    if (!correct) {
      setLast({ from: m.from, to: m.to, bad: true })
      failedAtFen.current = before
      finish(false)
      return true
    }
    setLast({ from: m.from, to: m.to, good: true })
    if (idx + 1 >= puzzle.moves.length || game.current.isCheckmate()) {
      finish(true)
      return true
    }
    const reply = puzzle.moves[idx + 1]
    setIdx(idx + 2)
    later(() => play(reply), 350)
    return true
  }

  // Hints: 1st highlights the piece to move, 2nd shows the full move as an arrow.
  useEffect(() => {
    if (hintSignal > 0 && status === 'playing') {
      sfx('hint')
      setHints((h) => Math.min(h + 1, 2))
    }
  }, [hintSignal]) // eslint-disable-line react-hooks/exhaustive-deps

  // After a miss: rewind to before the wrong move and animate the rest of the line.
  useEffect(() => {
    if (showSolutionSignal === 0 || status !== 'failed' || !failedAtFen.current) return
    game.current = new Chess(failedAtFen.current)
    setFen(failedAtFen.current)
    puzzle.moves.slice(idx).forEach((mv, i) => later(() => play(mv), 500 + i * 700))
  }, [showSolutionSignal]) // eslint-disable-line react-hooks/exhaustive-deps

  const onSquareClick = (sq: Square) => {
    if (status !== 'playing') return
    const piece = game.current.get(sq)
    const mine = piece && piece.color === toMove[0]
    if (selected && !mine) {
      const isPromo =
        game.current.get(selected)?.type === 'p' && (sq[1] === '8' || sq[1] === '1')
      tryMove(selected, sq, isPromo ? 'q' : undefined)
      setSelected(null)
    } else {
      setSelected(mine ? sq : null)
    }
  }

  const squareStyles = useMemo(() => {
    const s: Record<string, CSSProperties> = {}
    if (last) {
      const st = last.bad ? BAD : last.good ? GOOD : LAST
      s[last.from] = st
      s[last.to] = st
    }
    if (selected) {
      s[selected] = SEL
      for (const m of game.current.moves({ square: selected, verbose: true })) {
        s[m.to] = {
          background: game.current.get(m.to as Square)
            ? 'radial-gradient(transparent 58%, rgba(0,0,0,0.25) 60%)'
            : 'radial-gradient(rgba(0,0,0,0.25) 22%, transparent 24%)',
        }
      }
    }
    if (status === 'playing' && hints >= 1) {
      const { from } = uciParts(puzzle.moves[idx])
      s[from] = { ...(s[from] ?? {}), ...HINT }
    }
    return s
  }, [last, selected, hints, status, idx, puzzle.moves, fen]) // eslint-disable-line react-hooks/exhaustive-deps

  const arrows = useMemo(() => {
    if (status !== 'playing' || hints < 2) return []
    const { from, to } = uciParts(puzzle.moves[idx])
    return [[from, to, 'rgba(127,166,80,0.85)']] as [Square, Square, string][]
  }, [status, hints, idx, puzzle.moves])

  return (
    <div className="w-full aspect-square">
      <Chessboard
        id="puzzle"
        position={fen}
        boardOrientation={toMove}
        onPieceDrop={(from, to, piece) => {
          const isPromo = piece[1] !== 'P' && game.current.get(from)?.type === 'p'
          return tryMove(from, to, isPromo ? piece[1].toLowerCase() : undefined)
        }}
        onSquareClick={onSquareClick}
        arePiecesDraggable={status === 'playing'}
        isDraggablePiece={({ piece }) => piece[0] === toMove[0]}
        customSquareStyles={squareStyles}
        customArrows={arrows}
        animationDuration={200}
        customDarkSquareStyle={{ backgroundColor: '#769656' }}
        customLightSquareStyle={{ backgroundColor: '#eeeed2' }}
        customBoardStyle={{ borderRadius: '8px', boxShadow: '0 10px 30px rgba(0,0,0,0.45)' }}
      />
    </div>
  )
}
