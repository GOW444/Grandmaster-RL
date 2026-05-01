"""
Baseline Agents
===============
Three reference agents that serve as performance lower bounds:

* :class:`RandomAgent`           — purely random theme + difficulty selection.
* :class:`RatingMatchAgent`      — matches difficulty to the learner's current rating.
* :class:`FixedProgressionAgent` — ramps difficulty up linearly over the episode.

All agents implement a unified interface:
    ``predict(obs: np.ndarray) -> np.ndarray``  # action in [0,1]² space

A :func:`rollout` utility runs any agent (or SB3 model) for multiple episodes
and collects metrics used by the evaluation module.
"""

import logging
from typing import Any

import numpy as np

from env.chess_env import ChessPuzzleEnv, _denormalize_rating, _normalize_rating
from env.learner_model import THEMES

log = logging.getLogger(__name__)

# Shorthand for action shape
_ACTION_DIM = 2
_N_THEMES = len(THEMES)


# ---------------------------------------------------------------------------
# Random Agent
# ---------------------------------------------------------------------------

class RandomAgent:
    """Selects a uniformly random theme and difficulty each step.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return a random action in [0, 1]² action space.

        Args:
            obs: Current environment observation (unused).

        Returns:
            Action array of shape ``(2,)``:
            ``[theme_float ∈ [0, N_THEMES), difficulty ∈ [0, 1]]``.
        """
        theme = self._rng.uniform(0.0, float(_N_THEMES))
        difficulty = self._rng.uniform(0.0, 1.0)
        return np.array([theme, difficulty], dtype=np.float32)

    def reset(self) -> None:
        """No persistent state; present for API symmetry."""
        pass


# ---------------------------------------------------------------------------
# Rating Match Agent
# ---------------------------------------------------------------------------

class RatingMatchAgent:
    """Matches target difficulty to the learner's current overall rating.

    The agent reads the learner's current overall rating from ``obs[0]``
    (which is normalized to [0, 1]), adds uniform noise in ±100 rating
    points, and selects a random theme.

    Args:
        noise_scale: Half-width of the rating noise window (default: 100).
        seed:        Random seed.
    """

    def __init__(self, noise_scale: float = 100.0, seed: int = 0) -> None:
        self.noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Choose a difficulty near the learner's current rating ± noise.

        Args:
            obs: Current normalized observation, shape ``(9,)``.
                 ``obs[0]`` is the normalized overall rating ρ.

        Returns:
            Action array of shape ``(2,)``.
        """
        rho = _denormalize_rating(float(obs[0]))
        noise = self._rng.uniform(-self.noise_scale, self.noise_scale)
        target = np.clip(rho + noise, 400.0, 3000.0)
        diff_norm = float(_normalize_rating(target))

        theme = self._rng.uniform(0.0, float(_N_THEMES))
        return np.array([theme, diff_norm], dtype=np.float32)

    def reset(self) -> None:
        """No persistent state; present for API symmetry."""
        pass


# ---------------------------------------------------------------------------
# Fixed Progression Agent
# ---------------------------------------------------------------------------

class FixedProgressionAgent:
    """Slowly ramps difficulty upward on a fixed schedule.

    Difficulty starts at ``start_rating`` and increases by ``delta_step``
    rating points every ``step_every`` steps. Theme is random.

    Args:
        start_rating: Initial target difficulty (in absolute rating units).
        delta_step:   Rating increment applied every ``step_every`` steps.
        step_every:   How often (in steps) the difficulty is incremented.
        seed:         Random seed.
    """

    def __init__(
        self,
        start_rating: float = 800.0,
        delta_step: float = 25.0,
        step_every: int = 10,
        seed: int = 0,
    ) -> None:
        self.start_rating = start_rating
        self.delta_step = delta_step
        self.step_every = step_every
        self._rng = np.random.default_rng(seed)
        self._t: int = 0
        self._current_rating: float = start_rating

    def reset(self) -> None:
        """Reset the step counter and difficulty back to initial values."""
        self._t = 0
        self._current_rating = self.start_rating

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return action with a linearly increasing difficulty.

        Args:
            obs: Current normalized observation (used for theme only; ignored).

        Returns:
            Action array of shape ``(2,)``.
        """
        if self._t > 0 and self._t % self.step_every == 0:
            self._current_rating = min(self._current_rating + self.delta_step, 3000.0)

        diff_norm = float(_normalize_rating(self._current_rating))
        theme = self._rng.uniform(0.0, float(_N_THEMES))
        self._t += 1
        return np.array([theme, diff_norm], dtype=np.float32)


# ---------------------------------------------------------------------------
# Rollout utility
# ---------------------------------------------------------------------------

def rollout(
    agent: Any,
    env: ChessPuzzleEnv,
    n_episodes: int = 100,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Run *agent* for *n_episodes* and collect diagnostic metrics.

    Compatible with baseline agents (``predict(obs) -> action``) as well as
    SB3 models (``model.predict(obs, deterministic=True) -> (action, state)``).

    Args:
        agent:       A baseline agent or SB3 model.
        env:         A :class:`~env.chess_env.ChessPuzzleEnv` (or subclass).
        n_episodes:  Number of episodes to roll out.
        seed_offset: Episode seeds start at ``seed_offset``.

    Returns:
        Dictionary containing:

        * ``"delta_rho_per_episode"``  — list of (ρ_T - ρ_0) per episode
        * ``"success_history"``        — flattened list of 0/1 outcomes
        * ``"difficulty_history"``     — flattened list of actual puzzle ratings
        * ``"per_theme_skill_gain"``   — array shape ``(6,)``, mean Δφ per theme
        * ``"success_rate_trajectories"`` — list of per-episode p̄_t arrays
        * ``"difficulty_trajectories"``   — list of per-episode δ*_t arrays
    """
    # Detect SB3 model vs baseline agent
    is_sb3 = hasattr(agent, "predict") and hasattr(agent, "policy")

    delta_rhos: list[float] = []
    all_successes: list[float] = []
    all_difficulties: list[float] = []
    all_theme_gains: list[np.ndarray] = []
    success_traj: list[list[float]] = []
    difficulty_traj: list[list[float]] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)

        if hasattr(agent, "reset"):
            agent.reset()

        rho_0 = _denormalize_rating(float(obs[0]))
        init_skills = env._skills.copy()

        ep_successes: list[float] = []
        ep_difficulties: list[float] = []
        done = False

        while not done:
            if is_sb3:
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = agent.predict(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_successes.append(1.0 if info["solved"] else 0.0)
            ep_difficulties.append(float(info["puzzle_rating"]))

        rho_T = _denormalize_rating(float(obs[0]))
        delta_rho = rho_T - rho_0
        theme_gains = env._skills - init_skills

        delta_rhos.append(delta_rho)
        all_successes.extend(ep_successes)
        all_difficulties.extend(ep_difficulties)
        all_theme_gains.append(theme_gains)
        success_traj.append(ep_successes)
        difficulty_traj.append(ep_difficulties)

    mean_theme_gains = np.mean(all_theme_gains, axis=0)

    return {
        "delta_rho_per_episode": delta_rhos,
        "success_history": all_successes,
        "difficulty_history": all_difficulties,
        "per_theme_skill_gain": mean_theme_gains,
        "success_rate_trajectories": success_traj,
        "difficulty_trajectories": difficulty_traj,
    }
