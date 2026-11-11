from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Attempt, Puzzle, User
from app.routers.common import skills_by_theme
from app.schemas import AttemptHistoryOut, AttemptIn, AttemptOut
from app.security import get_current_user
from app.services.gamification import BADGES, award_badges, level_for_xp, touch_streak
from app.services.skill import overall_rating, update_theme_rating, xp_for

router = APIRouter(prefix="/api/attempts", tags=["attempts"])

RATED_MODES = {"coach", "rated", "theme", "daily"}


@router.post("", response_model=AttemptOut, status_code=status.HTTP_201_CREATED)
def record_attempt(body: AttemptIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> AttemptOut:
    puzzle = db.get(Puzzle, body.puzzle_id)
    if puzzle is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Puzzle not found")

    # Lock the user row so concurrent submissions can't race the rating update.
    user = db.scalar(select(User).where(User.id == user.id).with_for_update())
    seen_before = db.scalar(
        select(exists().where(Attempt.user_id == user.id, Attempt.puzzle_id == puzzle.id))
    )
    # Only the first attempt at a puzzle moves ratings; hints count as a miss for rating purposes.
    rated = body.mode in RATED_MODES and not seen_before
    rating_solved = body.solved and body.hints_used == 0

    skills = skills_by_theme(db, user.id)
    skill = skills[puzzle.primary_theme]
    theme_before, overall_before = skill.rating, user.rating

    if rated:
        skill.rating = update_theme_rating(skill.rating, puzzle.rating, rating_solved, skill.attempts)
        user.rating = overall_rating([s.rating for s in skills.values()])
    skill.attempts += 1
    skill.solved += int(body.solved)
    skill.total_time_ms += body.time_ms

    xp = xp_for(body.solved, puzzle.rating, theme_before, body.time_ms, body.hints_used)
    if not rated:
        xp = min(xp, 3)
    user.xp += xp
    touch_streak(user, date.today())

    attempt = Attempt(
        user_id=user.id, puzzle_id=puzzle.id, theme=puzzle.primary_theme, mode=body.mode,
        selector=body.selector if body.mode == "coach" else None,
        solved=body.solved, time_ms=body.time_ms, hints_used=body.hints_used,
        puzzle_rating=puzzle.rating, theme_rating_before=theme_before, theme_rating_after=skill.rating,
        rating_before=overall_before, rating_after=user.rating, xp_gained=xp,
    )
    db.add(attempt)
    db.flush()
    new_badges = award_badges(db, user, attempt, {t: s.rating for t, s in skills.items()})
    db.commit()

    return AttemptOut(
        solved=body.solved, theme=puzzle.primary_theme,
        theme_rating_before=theme_before, theme_rating_after=skill.rating,
        rating_before=overall_before, rating_after=user.rating,
        xp_gained=xp, level=level_for_xp(user.xp)[0],
        new_badges=[{"code": c, **BADGES[c]} for c in new_badges],
        streak=user.current_streak,
    )


@router.get("", response_model=list[AttemptHistoryOut])
def history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Attempt]:
    return db.scalars(
        select(Attempt).where(Attempt.user_id == user.id)
        .order_by(Attempt.created_at.desc()).limit(limit).offset(offset)
    ).all()
