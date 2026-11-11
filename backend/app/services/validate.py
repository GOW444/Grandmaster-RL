"""Server-side puzzle answer checking (so competitive scores can't be forged)."""

import chess


def check_solution(fen: str, solution: list[str], played: list[str]) -> bool:
    """True if `played` (the solver's UCI moves, in order) solves the puzzle.

    solution[0] is the opponent's setup move; the solver plays solution[1], [3], ...
    As on Lichess, any move that delivers checkmate is accepted even if it differs
    from the stored line."""
    board = chess.Board(fen)
    board.push_uci(solution[0])
    i = 1
    for uci in played:
        if i >= len(solution):
            return False
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return False
        if move not in board.legal_moves:
            return False
        board.push(move)
        if uci != solution[i]:
            return board.is_checkmate()
        if board.is_checkmate():
            return True
        i += 1
        if i < len(solution):
            board.push_uci(solution[i])  # opponent's reply
            i += 1
    return i >= len(solution)
