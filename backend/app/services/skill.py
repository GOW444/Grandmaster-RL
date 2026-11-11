"""Skill tracking for real users.

The expected-score curve is the same IRT logistic the RL engine's simulated
learner uses (env/learner_model.py: P = sigmoid((phi - delta) / tau), tau=200),
so a user's per-theme ratings live on the same scale the PPO coach was trained on.
The update itself is Elo-style (K * (score - expected)) because the engine's
alpha=0.05 rule is tuned for 100-step simulated episodes and moves far too
slowly for a real person.
"""

import math

from app.models import START_RATING

TAU = 200.0
RATING_MIN, RATING_MAX = 400.0, 3000.0


def expected_score(skill: float, puzzle_rating: float, tau: float = TAU) -> float:
    return 1.0 / (1.0 + math.exp(-(skill - puzzle_rating) / tau))


def k_factor(attempts_in_theme: int) -> float:
    # Provisional period: move fast for the first ~20 puzzles in a theme, then settle.
    return max(16.0, 48.0 - 1.6 * attempts_in_theme)


def update_theme_rating(skill: float, puzzle_rating: float, solved: bool, attempts_in_theme: int) -> float:
    score = 1.0 if solved else 0.0
    new = skill + k_factor(attempts_in_theme) * (score - expected_score(skill, puzzle_rating))
    return float(min(max(new, RATING_MIN), RATING_MAX))


def overall_rating(theme_ratings: list[float]) -> float:
    # Same definition as the engine: overall rating = mean of per-theme skills.
    return sum(theme_ratings) / len(theme_ratings) if theme_ratings else START_RATING


def xp_for(solved: bool, puzzle_rating: int, skill_before: float, time_ms: int, hints: int) -> int:
    if not solved:
        return 2
    xp = 10 + max(0, int((puzzle_rating - skill_before) / 25))
    if time_ms < 15_000:
        xp += 5
    return max(3, xp - 4 * hints)
