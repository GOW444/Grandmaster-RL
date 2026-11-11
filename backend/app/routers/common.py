from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import START_RATING, THEMES, Puzzle, ThemeSkill, User
from app.schemas import PuzzleOut, UserOut
from app.services.gamification import level_for_xp


def puzzle_out(p: Puzzle) -> PuzzleOut:
    return PuzzleOut(
        id=p.id,
        fen=p.fen,
        moves=p.moves.split(),
        rating=p.rating,
        themes=p.themes,
        primary_theme=p.primary_theme,
        game_url=p.game_url,
    )


def user_out(u: User) -> UserOut:
    level, into, need = level_for_xp(u.xp)
    return UserOut(
        id=u.id,
        username=u.username,
        email=u.email,
        rating=round(u.rating, 1),
        xp=u.xp,
        level=level,
        xp_into_level=into,
        xp_for_next=need,
        current_streak=u.current_streak,
        longest_streak=u.longest_streak,
        created_at=u.created_at,
    )


def skills_by_theme(db: Session, user_id: int) -> dict[str, ThemeSkill]:
    rows = db.scalars(select(ThemeSkill).where(ThemeSkill.user_id == user_id)).all()
    by_theme = {r.theme: r for r in rows}
    for t in THEMES:
        if t not in by_theme:
            by_theme[t] = ThemeSkill(user_id=user_id, theme=t, rating=START_RATING, attempts=0, solved=0, total_time_ms=0)
            db.add(by_theme[t])
    return by_theme
