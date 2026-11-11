from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Integer, cast, func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import THEMES, Attempt, User, UserBadge
from app.routers.common import skills_by_theme, user_out
from app.security import get_current_user
from app.services.gamification import BADGES

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/overview")
def overview(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    skills = skills_by_theme(db, user.id)
    db.commit()

    themes = []
    for t in THEMES:
        s = skills[t]
        themes.append({
            "theme": t,
            "rating": round(s.rating, 1),
            "attempts": s.attempts,
            "solved": s.solved,
            "accuracy": round(s.solved / s.attempts, 3) if s.attempts else None,
            "avg_time_ms": s.total_time_ms // s.attempts if s.attempts else None,
        })

    # Areas to improve: low rating, low accuracy, or slow -- only once there's enough signal.
    tried = [x for x in themes if x["attempts"] >= 3]
    focus = []
    if tried:
        weakest = min(tried, key=lambda x: x["rating"])
        focus.append({"theme": weakest["theme"], "why": f"Lowest rated theme ({weakest['rating']:.0f})"})
        worst_acc = min(tried, key=lambda x: x["accuracy"] or 0)
        if worst_acc["theme"] != weakest["theme"]:
            focus.append({"theme": worst_acc["theme"], "why": f"Lowest accuracy ({worst_acc['accuracy']:.0%})"})
        slowest = max(tried, key=lambda x: x["avg_time_ms"] or 0)
        if slowest["theme"] not in {f["theme"] for f in focus}:
            focus.append({"theme": slowest["theme"], "why": f"Slowest solves ({slowest['avg_time_ms'] / 1000:.0f}s avg)"})
    untried = [x["theme"] for x in themes if x["attempts"] == 0]

    totals = db.execute(
        select(func.count(), func.coalesce(func.sum(cast(Attempt.solved, Integer)), 0),
               func.coalesce(func.avg(Attempt.time_ms), 0))
        .where(Attempt.user_id == user.id)
    ).one()

    return {
        "user": user_out(user),
        "themes": themes,
        "focus": focus,
        "untried": untried,
        "totals": {"attempts": totals[0], "solved": int(totals[1]), "avg_time_ms": int(totals[2])},
    }


@router.get("/rating-history")
def rating_history(
    limit: int = Query(300, ge=1, le=2000),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    rows = db.execute(
        select(Attempt.created_at, Attempt.rating_after, Attempt.theme, Attempt.theme_rating_after)
        .where(Attempt.user_id == user.id).order_by(Attempt.created_at.desc()).limit(limit)
    ).all()[::-1]
    return [{"t": r.created_at, "rating": round(r.rating_after, 1), "theme": r.theme,
             "theme_rating": round(r.theme_rating_after, 1)} for r in rows]


@router.get("/activity")
def activity(days: int = Query(182, ge=7, le=366), user: User = Depends(get_current_user),
             db: Session = Depends(get_db)) -> list[dict]:
    since = date.today() - timedelta(days=days - 1)
    day = func.date(Attempt.created_at)
    rows = db.execute(
        select(day.label("day"), func.count(), func.sum(cast(Attempt.solved, Integer)))
        .where(Attempt.user_id == user.id, Attempt.created_at >= since)
        .group_by(day).order_by(day)
    ).all()
    return [{"day": r[0], "attempts": r[1], "solved": int(r[2] or 0)} for r in rows]


@router.get("/time-buckets")
def time_buckets(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict]:
    """Accuracy by how long you spent: are you rushing or overthinking?"""
    bucket = func.width_bucket(Attempt.time_ms, 0, 120_000, 8)
    rows = db.execute(
        select(bucket.label("b"), func.count(), func.avg(cast(Attempt.solved, Integer)))
        .where(Attempt.user_id == user.id).group_by("b").order_by("b")
    ).all()
    out = []
    for b, n, acc in rows:
        lo = (b - 1) * 15
        out.append({"label": f"{lo}-{lo + 15}s" if b <= 8 else "120s+", "attempts": n, "accuracy": round(float(acc), 3)})
    return out


@router.get("/coach-ab")
def coach_ab(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    """PPO vs rating-match on real attempts: yours and across all users."""
    def agg(where):
        rows = db.execute(
            select(Attempt.selector, func.count(), func.avg(cast(Attempt.solved, Integer)),
                   func.avg(Attempt.theme_rating_after - Attempt.theme_rating_before),
                   func.avg(Attempt.puzzle_rating - Attempt.theme_rating_before))
            .where(Attempt.mode == "coach", Attempt.selector.is_not(None), *where)
            .group_by(Attempt.selector)
        ).all()
        return {r[0]: {"attempts": r[1], "success_rate": round(float(r[2]), 3),
                       "avg_rating_gain": round(float(r[3]), 2), "avg_stretch": round(float(r[4]), 1)} for r in rows}

    return {"you": agg([Attempt.user_id == user.id]), "global": agg([])}


@router.get("/badges")
def badges(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict]:
    earned = {b.code: b.earned_at for b in db.scalars(select(UserBadge).where(UserBadge.user_id == user.id))}
    return [{"code": c, **meta, "earned_at": earned.get(c)} for c, meta in BADGES.items()]


@router.get("/leaderboard")
def leaderboard(by: str = Query("rating", pattern=r"^(rating|xp|streak)$"), db: Session = Depends(get_db),
                _: User = Depends(get_current_user)) -> list[dict]:
    col = {"rating": User.rating, "xp": User.xp, "streak": User.current_streak}[by]
    rows = db.scalars(select(User).order_by(col.desc()).limit(25)).all()
    return [{"username": u.username, "rating": round(u.rating), "xp": u.xp, "streak": u.current_streak} for u in rows]
