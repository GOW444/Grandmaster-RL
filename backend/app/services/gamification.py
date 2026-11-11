from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Attempt, User, UserBadge

BADGES: dict[str, dict[str, str]] = {
    "first_blood": {"name": "First Blood", "desc": "Solve your first puzzle", "icon": "🩸"},
    "ten_solved": {"name": "Getting Warm", "desc": "Solve 10 puzzles", "icon": "🔥"},
    "hundred_solved": {"name": "Centurion", "desc": "Solve 100 puzzles", "icon": "💯"},
    "streak_3": {"name": "Habit Forming", "desc": "3-day training streak", "icon": "📅"},
    "streak_7": {"name": "Week Warrior", "desc": "7-day training streak", "icon": "🗓️"},
    "giant_slayer": {"name": "Giant Slayer", "desc": "Solve a puzzle 300+ above your theme rating", "icon": "🗡️"},
    "speed_demon": {"name": "Speed Demon", "desc": "Solve a 1500+ puzzle in under 10 seconds", "icon": "⚡"},
    "theme_expert": {"name": "Specialist", "desc": "Reach 1500 in any theme", "icon": "🎯"},
    "all_rounder": {"name": "All-Rounder", "desc": "Solve at least one puzzle in every theme", "icon": "🌐"},
}


def level_for_xp(xp: int) -> tuple[int, int, int]:
    """Level n needs 50 * n^2 total XP. Returns (level, xp_into_level, xp_for_next_level)."""
    level = 1
    while 50 * level**2 <= xp:
        level += 1
    prev = 50 * (level - 1) ** 2
    return level, xp - prev, 50 * level**2 - prev


def touch_streak(user: User, today: date) -> None:
    if user.last_active_on == today:
        return
    if user.last_active_on == today - timedelta(days=1):
        user.current_streak += 1
    else:
        user.current_streak = 1
    user.longest_streak = max(user.longest_streak, user.current_streak)
    user.last_active_on = today


def award_badges(db: Session, user: User, attempt: Attempt, theme_ratings: dict[str, float]) -> list[str]:
    owned = set(db.scalars(select(UserBadge.code).where(UserBadge.user_id == user.id)).all())
    solved_total = db.scalar(
        select(func.count()).select_from(Attempt).where(Attempt.user_id == user.id, Attempt.solved)
    ) or 0
    themes_solved = db.scalar(
        select(func.count(func.distinct(Attempt.theme))).where(Attempt.user_id == user.id, Attempt.solved)
    ) or 0

    earned = []
    checks = {
        "first_blood": solved_total >= 1,
        "ten_solved": solved_total >= 10,
        "hundred_solved": solved_total >= 100,
        "streak_3": user.current_streak >= 3,
        "streak_7": user.current_streak >= 7,
        "giant_slayer": attempt.solved and attempt.puzzle_rating - attempt.theme_rating_before >= 300,
        "speed_demon": attempt.solved and attempt.puzzle_rating >= 1500 and attempt.time_ms < 10_000,
        "theme_expert": max(theme_ratings.values()) >= 1500,
        "all_rounder": themes_solved >= len(theme_ratings),
    }
    for code, ok in checks.items():
        if ok and code not in owned:
            db.add(UserBadge(user_id=user.id, code=code))
            earned.append(code)
    return earned
