import { memo } from 'react'
import { Chessboard } from 'react-chessboard'
import type { Puzzle } from '../api'
import { startPosition } from '../chessutil'

function MiniBoard({ puzzle }: { puzzle: Pick<Puzzle, 'id' | 'fen' | 'moves'> }) {
  const { fen, orientation } = startPosition(puzzle)
  return (
    <Chessboard
      id={`mini-${puzzle.id}`}
      position={fen}
      boardOrientation={orientation}
      arePiecesDraggable={false}
      animationDuration={0}
      customDarkSquareStyle={{ backgroundColor: '#769656' }}
      customLightSquareStyle={{ backgroundColor: '#eeeed2' }}
      customBoardStyle={{ borderRadius: '6px' }}
    />
  )
}

export default memo(MiniBoard)
