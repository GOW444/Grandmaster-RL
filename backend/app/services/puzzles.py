"""Puzzle retrieval. Replaces the engine's per-theme KD-tree with an equivalent
B-tree lookup on (primary_theme, rating): nearest-rating search is two index
range scans (above / below the target) instead of a full sort."""

import random

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.models import Attempt, Puzzle


def _unseen(user_id: int):
    return ~exists().where(Attempt.user_id == user_id, Attempt.puzzle_id == Puzzle.id)


def nearest_puzzle(db: Session, user_id: int, target: float, theme: str | None = None, pool: int = 8) -> Puzzle | None:
    """Closest unseen puzzle to `target` rating (optionally within a theme).

    Picks randomly among the `pool` nearest so the same user doesn't always
    get the identical puzzle for the same target."""
    base = select(Puzzle).where(_unseen(user_id))
    if theme:
        base = base.where(Puzzle.primary_theme == theme)
    t = int(round(target))
    above = db.scalars(base.where(Puzzle.rating >= t).order_by(Puzzle.rating.asc()).limit(pool)).all()
    below = db.scalars(base.where(Puzzle.rating < t).order_by(Puzzle.rating.desc()).limit(pool)).all()
    candidates = sorted([*above, *below], key=lambda p: abs(p.rating - t))[:pool]
    return random.choice(candidates) if candidates else None
