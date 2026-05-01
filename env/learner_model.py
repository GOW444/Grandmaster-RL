"""
Learner Model
=============
IRT-based logistic success probability and piecewise skill update rule.
All functions are pure (no side effects) and deterministic.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Global constants — match the report exactly
# ---------------------------------------------------------------------------
THEMES: list[str] = ["fork", "pin", "mate", "endgame", "skewer", "discovery"]

RATING_MIN: float = 400.0
RATING_MAX: float = 3000.0

TAU: float = 200.0    # IRT temperature: governs sharpness of solve curve
ALPHA: float = 0.05   # Learning rate for skill update
LAMBDA: float = 0.5   # Scale of success gain relative to skill gap
MU: float = 5.0       # Fixed penalty on failure (in rating points)


# ---------------------------------------------------------------------------
# Core model functions
# ---------------------------------------------------------------------------

def solve_prob(
    skill_theme: float,
    puzzle_rating: float,
    tau: float = TAU,
) -> float:
    """Compute the probability that a learner solves a puzzle (IRT logistic model).

    Uses the Item Response Theory logistic function:
        P(solve) = σ((φ - δ) / τ)
    where φ is the learner's current skill for the relevant theme,
    δ is the puzzle's Glicko-2 rating, and τ is the temperature parameter.

    Args:
        skill_theme:   Learner's current skill estimate for the puzzle's theme (φ).
        puzzle_rating: Actual Glicko-2 rating of the selected puzzle (δ).
        tau:           IRT temperature; higher τ → softer probability curve.

    Returns:
        Probability in (0, 1) that the learner solves the puzzle.
    """
    logit = (skill_theme - puzzle_rating) / tau
    # Use numerically stable sigmoid
    return float(1.0 / (1.0 + np.exp(-logit)))


def update_skill(
    phi: float,
    delta: float,
    solved: bool,
    alpha: float = ALPHA,
    lam: float = LAMBDA,
    mu: float = MU,
) -> float:
    """Apply the piecewise skill update rule after a puzzle attempt.

    On success the gain is proportional to how much harder the puzzle was
    relative to the learner's current skill (negative if the puzzle was too
    easy, naturally discouraging trivially easy selections).
    On failure a fixed penalty ``-mu`` is applied.

    Update rule:
        if solved:  gain = λ * (δ - φ) / φ
        else:       gain = -μ
        φ_new = φ + α * gain
        φ_new = clamp(φ_new, RATING_MIN, RATING_MAX)

    Args:
        phi:    Current theme skill estimate (φ), in [RATING_MIN, RATING_MAX].
        delta:  Actual puzzle rating (δ), in [RATING_MIN, RATING_MAX].
        solved: Whether the learner solved the puzzle.
        alpha:  Learning rate (α).
        lam:    Success gain scale factor (λ).
        mu:     Failure penalty in rating points (μ).

    Returns:
        Updated skill estimate clamped to [RATING_MIN, RATING_MAX].
    """
    if solved:
        gain = lam * (delta - phi) / max(phi, 1.0)  # guard against division by ~0
    else:
        gain = -mu

    phi_new = phi + alpha * gain
    return float(np.clip(phi_new, RATING_MIN, RATING_MAX))


# ---------------------------------------------------------------------------
# Convenience vectorised wrappers (used by the environment)
# ---------------------------------------------------------------------------

def solve_probs_batch(
    skills: np.ndarray,
    ratings: np.ndarray,
    tau: float = TAU,
) -> np.ndarray:
    """Vectorised version of :func:`solve_prob` over arrays.

    Args:
        skills:  Array of per-theme skill values, shape ``(N,)``.
        ratings: Array of puzzle ratings, shape ``(N,)``.
        tau:     IRT temperature.

    Returns:
        Array of probabilities, shape ``(N,)``.
    """
    logits = (skills - ratings) / tau
    return 1.0 / (1.0 + np.exp(-logits))
