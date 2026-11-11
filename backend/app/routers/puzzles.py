import hashlib
from datetime import date

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, exists, func, select
from sqlalchemy.orm import Session, aliased

from app.db import get_db
from app.models import THEMES, Attempt, Puzzle, SavedPuzzle, User
from app.routers.common import puzzle_out, skills_by_theme
from app.schemas import NextPuzzleOut, SavedOut, SaveIn
from app.security import get_current_user
from app.services.coach import CoachState, coach, recommend_rating_match
from app.services.puzzles import nearest_puzzle

router = APIRouter(prefix="/api/puzzles", tags=["puzzles"])
_rng = np.random.default_rng()


def _is_saved(db: Session, user_id: int, puzzle_id: str) -> bool:
    return db.get(SavedPuzzle, (user_id, puzzle_id)) is not None


def _coach_state(db: Session, user: User) -> CoachState:
    skills = skills_by_theme(db, user.id)
    recent = db.execute(
        select(Attempt.solved, Attempt.puzzle_rating)
        .where(Attempt.user_id == user.id)
        .order_by(Attempt.created_at.desc())
        .limit(10)
    ).all()[::-1]
    return CoachState(
        theme_ratings={t: s.rating for t, s in skills.items()},
        recent_solved=[r.solved for r in recent],
        recent_difficulty=[r.puzzle_rating for r in recent],
    )


@router.get("/next", response_model=NextPuzzleOut)
def next_puzzle(
    mode: str = Query("coach", pattern=r"^(coach|rated|theme)$"),
    theme: str | None = Query(None),
    selector: str | None = Query(None, pattern=r"^(ppo|rating_match|ab)$"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> NextPuzzleOut:
    """coach: AI-selected (PPO, or rating-match baseline; `ab` = random 50/50 split).
    rated: your overall rating, any theme. theme: drill one theme at your level."""
    state = _coach_state(db, user)
    reason, sel, target, pick_theme = None, None, user.rating, None

    if mode == "coach":
        want = selector or "ab"
        if want == "ab":
            want = "ppo" if _rng.random() < 0.5 else "rating_match"
        if want == "ppo" and coach.available:
            rec = coach.recommend_ppo(state)
        else:
            rec = recommend_rating_match(state, _rng)
        sel, target, pick_theme, reason = rec.selector, rec.target_rating, rec.theme, rec.reason
    elif mode == "theme":
        if theme not in THEMES:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"theme must be one of {THEMES}")
        pick_theme, target = theme, state.theme_ratings[theme]
        reason = f"Theme drill: {theme} at your {theme} rating."

    puzzle = nearest_puzzle(db, user.id, target, pick_theme)
    if puzzle is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No unseen puzzles left for this selection")
    db.commit()  # persists lazily-seeded skill rows, if any
    return NextPuzzleOut(
        puzzle=puzzle_out(puzzle), mode=mode, selector=sel, target_rating=target,
        reason=reason, saved=_is_saved(db, user.id, puzzle.id),
    )


@router.get("/daily", response_model=NextPuzzleOut)
def daily_puzzle(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> NextPuzzleOut:
    """Same puzzle for everyone today: deterministic pick from popular 1400-1800 puzzles."""
    seed = int(hashlib.sha256(date.today().isoformat().encode()).hexdigest(), 16)
    q = select(Puzzle).where(Puzzle.rating.between(1400, 1800), Puzzle.popularity >= 90)
    count = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    if count == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No puzzles imported")
    puzzle = db.scalars(q.order_by(Puzzle.id).offset(seed % count).limit(1)).one()
    return NextPuzzleOut(puzzle=puzzle_out(puzzle), mode="daily", reason="Puzzle of the day",
                         saved=_is_saved(db, user.id, puzzle.id))


@router.get("/review", response_model=NextPuzzleOut)
def review_puzzle(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> NextPuzzleOut:
    """Spaced-review: oldest puzzle you failed and haven't solved since."""
    solve = aliased(Attempt)
    later_solve = exists().where(solve.user_id == user.id, solve.puzzle_id == Puzzle.id, solve.solved)
    failed = (
        select(Puzzle, func.max(Attempt.created_at).label("last"))
        .join(Attempt, and_(Attempt.puzzle_id == Puzzle.id, Attempt.user_id == user.id))
        .where(~Attempt.solved, ~later_solve)
        .group_by(Puzzle.id)
        .order_by("last")
        .limit(1)
    )
    row = db.execute(failed).first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Nothing to review — no unsolved misses!")
    p = row[0]
    return NextPuzzleOut(puzzle=puzzle_out(p), mode="review", reason="You missed this one earlier.",
                         saved=_is_saved(db, user.id, p.id))


@router.get("/saved", response_model=list[SavedOut])
def list_saved(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[SavedOut]:
    rows = db.scalars(
        select(SavedPuzzle).where(SavedPuzzle.user_id == user.id).order_by(SavedPuzzle.created_at.desc())
    ).all()
    return [SavedOut(puzzle=puzzle_out(r.puzzle), note=r.note, created_at=r.created_at) for r in rows]


@router.get("/{puzzle_id}", response_model=NextPuzzleOut)
def get_puzzle(puzzle_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> NextPuzzleOut:
    p = db.get(Puzzle, puzzle_id)
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Puzzle not found")
    return NextPuzzleOut(puzzle=puzzle_out(p), mode="review", saved=_is_saved(db, user.id, p.id))


@router.put("/{puzzle_id}/save", status_code=status.HTTP_204_NO_CONTENT)
def save_puzzle(puzzle_id: str, body: SaveIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> None:
    if db.get(Puzzle, puzzle_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Puzzle not found")
    row = db.get(SavedPuzzle, (user.id, puzzle_id))
    if row is None:
        db.add(SavedPuzzle(user_id=user.id, puzzle_id=puzzle_id, note=body.note))
    else:
        row.note = body.note
    db.commit()


@router.delete("/{puzzle_id}/save", status_code=status.HTTP_204_NO_CONTENT)
def unsave_puzzle(puzzle_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> None:
    row = db.get(SavedPuzzle, (user.id, puzzle_id))
    if row:
        db.delete(row)
        db.commit()
