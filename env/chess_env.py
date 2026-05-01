"""
Chess Puzzle Training Environment
===================================
A Gymnasium-compliant MDP environment that simulates a chess learner
receiving adaptively selected puzzles from an RL agent.

State  (9-dim, float32, all in [0, 1]):
    [0]   overall rating ρ (normalized)
    [1-6] per-theme skill φ^(k) for k in {fork, pin, mate, endgame, skewer, discovery}
    [7]   rolling success rate p̄ over the last W puzzles
    [8]   mean difficulty δ̄ over the last W puzzles (normalized)

Action (2-dim, float32):
    [0]  theme index as float in [0, 6) → rounded to int in {0,…,5}
    [1]  target difficulty δ* in [0, 1] → denormalized to [400, 3000]
"""

import logging
import pickle
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial import KDTree

from env.learner_model import (
    THEMES,
    RATING_MIN,
    RATING_MAX,
    TAU,
    ALPHA,
    LAMBDA,
    MU,
    solve_prob,
    update_skill,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
_RATING_RANGE = RATING_MAX - RATING_MIN  # 2600


def _normalize_rating(r: float) -> float:
    """Map a rating in [400, 3000] to [0, 1]."""
    return (r - RATING_MIN) / _RATING_RANGE


def _denormalize_rating(r: float) -> float:
    """Map a value in [0, 1] back to a rating in [400, 3000]."""
    return r * _RATING_RANGE + RATING_MIN


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ChessPuzzleEnv(gym.Env):
    """Training environment for adaptive chess puzzle selection.

    The RL agent selects (theme, difficulty) pairs; the environment simulates
    the learner's response via an IRT logistic model, updates the learner's
    skill estimates, and returns a reward equal to the change in overall rating.

    Args:
        indices_dir: Path to the directory containing ``{theme}.pkl`` KD-tree files.
        T:           Episode length (number of puzzle attempts per episode).
        mu0:         Mean of the initial skill Normal distribution.
        sigma0:      Std-dev of the initial skill Normal distribution.
        window:      Rolling window size W for success-rate and mean-difficulty.
        tau:         IRT temperature (passed to the learner model).
        alpha:       Learner model learning rate.
        lam:         Success gain scale factor.
        mu:          Failure penalty in rating points.
        seed:        Base random seed.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        indices_dir: str | Path = "data/indices",
        T: int = 100,
        mu0: float = 1000.0,
        sigma0: float = 150.0,
        window: int = 10,
        tau: float = TAU,
        alpha: float = ALPHA,
        lam: float = LAMBDA,
        mu: float = MU,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self._indices_dir = Path(indices_dir)
        self.T = T
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.window = window

        # Learner model params
        self.tau = tau
        self.alpha = alpha
        self.lam = lam
        self.mu = mu

        # Action / observation spaces
        n_themes = len(THEMES)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_themes + 3,), dtype=np.float32
        )
        # action[0]: theme float in [0, n_themes)
        # action[1]: normalized difficulty in [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([float(n_themes), 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Load KD-tree indices
        self._trees: list[KDTree] = []
        self._dfs: list[Any] = []  # list of pd.DataFrame
        self._load_indices()

        # Internal state (initialized properly in reset())
        self._skills: np.ndarray = np.zeros(n_themes, dtype=np.float64)
        self._overall: float = 0.0
        self._success_buf: deque[float] = deque(maxlen=window)
        self._diff_buf: deque[float] = deque(maxlen=window)
        self._step_count: int = 0

        # Seeding
        self._np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_indices(self) -> None:
        """Load all per-theme KD-tree + DataFrame pairs from disk."""
        for theme in THEMES:
            pkl_path = self._indices_dir / f"{theme}.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(
                    f"KD-tree index not found: {pkl_path}\n"
                    "Run 'python scripts/build_dataset.py' first."
                )
            with open(pkl_path, "rb") as fh:
                tree, df = pickle.load(fh)
            self._trees.append(tree)
            self._dfs.append(df)
        log.debug("Loaded %d KD-tree indices.", len(THEMES))

    def _build_state(self) -> np.ndarray:
        """Construct and return the normalized (9,) state vector."""
        norm_skills = np.array(
            [_normalize_rating(s) for s in self._skills], dtype=np.float32
        )
        norm_overall = float(_normalize_rating(self._overall))

        if len(self._success_buf) == 0:
            rolling_success = 0.0
            rolling_diff = _normalize_rating(self.mu0)
        else:
            rolling_success = float(np.mean(list(self._success_buf)))
            rolling_diff = float(
                np.mean([_normalize_rating(d) for d in self._diff_buf])
            )

        state = np.concatenate(
            [[norm_overall], norm_skills, [rolling_success, rolling_diff]],
            dtype=np.float32,
        )
        return np.clip(state, 0.0, 1.0)

    def _query_puzzle(self, theme_idx: int, target_rating: float) -> tuple[float, int]:
        """Query the KD-tree for the nearest puzzle to ``target_rating``.

        Args:
            theme_idx:     Index into ``THEMES``.
            target_rating: Desired difficulty in [400, 3000].

        Returns:
            A tuple ``(actual_rating, row_index_in_df)``.
        """
        tree = self._trees[theme_idx]
        df = self._dfs[theme_idx]
        _, idx = tree.query([[target_rating]])
        row = df.iloc[int(idx)]
        return float(row["Rating"]), int(idx)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to a new learner episode.

        Samples initial per-theme skills from N(mu0, sigma0²) clipped to
        [600, 1800] and initialises all rolling buffers.

        Args:
            seed:    Optional seed for reproducibility.
            options: Unused; present for API compliance.

        Returns:
            A tuple ``(observation, info_dict)``.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        n_themes = len(THEMES)
        self._skills = np.clip(
            self._np_rng.normal(self.mu0, self.sigma0, size=n_themes),
            600.0,
            1800.0,
        )
        self._overall = float(np.mean(self._skills))
        self._success_buf = deque(maxlen=self.window)
        self._diff_buf = deque(maxlen=self.window)
        self._step_count = 0

        return self._build_state(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the environment by one puzzle attempt.

        Args:
            action: Array of shape ``(2,)`` where
                    ``action[0]`` selects the theme (float, rounded to int)
                    ``action[1]`` is the normalized target difficulty.

        Returns:
            Tuple of ``(obs, reward, terminated, truncated, info)``.
        """
        # --- Decode action ---
        theme_idx = int(np.clip(round(float(action[0])), 0, len(THEMES) - 1))
        norm_diff = float(np.clip(action[1], 0.0, 1.0))
        target_rating = _denormalize_rating(norm_diff)

        # --- Query nearest puzzle ---
        puzzle_rating, _ = self._query_puzzle(theme_idx, target_rating)

        # --- Learner dynamics (call-able overrides in EvalEnv) ---
        effective_skill, effective_tau = self._get_effective_skill_and_tau(theme_idx)

        p_solve = solve_prob(effective_skill, puzzle_rating, tau=effective_tau)
        solved = bool(self._np_rng.random() < p_solve)

        # Skill update uses the REAL skill (not effective_skill)
        phi_before = float(self._skills[theme_idx])
        self._skills[theme_idx] = update_skill(
            phi=phi_before,
            delta=puzzle_rating,
            solved=solved,
            alpha=self.alpha,
            lam=self.lam,
            mu=self.mu,
        )
        phi_after = float(self._skills[theme_idx])

        old_overall = self._overall
        self._overall = float(np.mean(self._skills))

        # --- Update rolling buffers ---
        self._success_buf.append(1.0 if solved else 0.0)
        self._diff_buf.append(puzzle_rating)

        # --- Reward ---
        reward = self._overall - old_overall

        # --- Termination ---
        self._step_count += 1
        terminated = self._step_count >= self.T
        truncated = False

        info: dict[str, Any] = {
            "theme_idx": theme_idx,
            "theme_name": THEMES[theme_idx],
            "puzzle_rating": puzzle_rating,
            "target_rating": target_rating,
            "solved": solved,
            "p_solve": p_solve,
            "skill_before": phi_before,
            "skill_after": phi_after,
            "overall_rating": self._overall,
            "step": self._step_count,
        }

        return self._build_state(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Hook for subclasses (EvalChessPuzzleEnv overrides this)
    # ------------------------------------------------------------------

    def _get_effective_skill_and_tau(
        self, theme_idx: int
    ) -> tuple[float, float]:
        """Return the effective skill and tau used for solve_prob.

        Subclasses override this to inject fatigue or jitter.

        Args:
            theme_idx: Index of the selected theme.

        Returns:
            Tuple ``(effective_skill, effective_tau)``.
        """
        return float(self._skills[theme_idx]), self.tau

    # ------------------------------------------------------------------
    # Gymnasium compliance
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Rendering is not supported; present for API compliance."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        self._trees.clear()
        self._dfs.clear()
