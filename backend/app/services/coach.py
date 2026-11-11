"""AI Coach: runs the trained PPO curriculum policy on a real user's state.

The observation is built exactly like env/chess_env.py::_build_state, but from
the user's live database records instead of a simulated learner:

    [overall_rating, 6 theme skills, rolling success (last W), rolling mean difficulty (last W)]

all min-max normalised from [400, 3000] to [0, 1]. No retraining: the
checkpoint from the `engine` branch is loaded read-only.
"""

from __future__ import annotations

import logging
import pickle
import sys
import threading
from dataclasses import dataclass

import numpy as np

from app.config import REPO_ROOT, get_settings
from app.models import THEMES
from app.services.policy import NumpyPolicy

log = logging.getLogger(__name__)

RATING_MIN, RATING_MAX = 400.0, 3000.0
WINDOW = 10  # rolling window W used during training


def _norm(r: float) -> float:
    return float(np.clip((r - RATING_MIN) / (RATING_MAX - RATING_MIN), 0.0, 1.0))


def _denorm(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0)) * (RATING_MAX - RATING_MIN) + RATING_MIN


@dataclass
class CoachState:
    theme_ratings: dict[str, float]
    recent_solved: list[bool]  # most recent last
    recent_difficulty: list[int]


@dataclass
class Recommendation:
    theme: str
    target_rating: float
    selector: str  # "ppo" | "rating_match"
    reason: str


def build_observation(state: CoachState) -> np.ndarray:
    skills = [state.theme_ratings[t] for t in THEMES]
    overall = float(np.mean(skills))
    succ = state.recent_solved[-WINDOW:]
    diff = state.recent_difficulty[-WINDOW:]
    rolling_success = float(np.mean(succ)) if succ else 0.0
    rolling_diff = float(np.mean([_norm(d) for d in diff])) if diff else _norm(1000.0)
    obs = np.array([_norm(overall), *[_norm(s) for s in skills], rolling_success, rolling_diff], dtype=np.float32)
    return np.clip(obs, 0.0, 1.0)


class PPOCoach:
    def __init__(self) -> None:
        self._model = None
        self._obs_rms = None
        self._lock = threading.Lock()
        self.available = False
        self.error: str | None = None

    def load(self) -> None:
        """Prefer the torch-free NumPy export; fall back to the SB3 checkpoint if it's missing."""
        s = get_settings()
        try:
            if s.policy_npz.exists():
                self._model = NumpyPolicy(s.policy_npz)
                log.info("PPO coach loaded (numpy) from %s", s.policy_npz)
            else:
                if str(REPO_ROOT) not in sys.path:
                    sys.path.insert(0, str(REPO_ROOT))  # so the custom HybridPolicy unpickles
                from stable_baselines3 import PPO

                self._model = PPO.load(str(s.ppo_checkpoint), device="cpu")
                log.info("PPO coach loaded (SB3) from %s", s.ppo_checkpoint)
            if s.coach_use_vecnormalize and s.ppo_vecnormalize.exists():
                with open(s.ppo_vecnormalize, "rb") as fh:
                    self._obs_rms = pickle.load(fh).obs_rms
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on local artifacts
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("PPO coach unavailable, falling back to rating-match: %s", self.error)

    def _predict(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_rms is not None:
            obs = np.clip((obs - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + 1e-8), -10, 10).astype(np.float32)
        if isinstance(self._model, NumpyPolicy):
            return self._model.act(obs)
        with self._lock:
            action, _ = self._model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).ravel()

    def recommend_ppo(self, state: CoachState) -> Recommendation:
        action = self._predict(build_observation(state))
        theme_idx = int(np.clip(round(float(action[0])), 0, len(THEMES) - 1))
        theme = THEMES[theme_idx]
        target = _denorm(float(action[1]))
        skill = state.theme_ratings[theme]
        gap = target - skill
        weakest = min(THEMES, key=lambda t: state.theme_ratings[t])
        if gap > 75:
            pitch = f"a stretch puzzle ~{gap:.0f} above your {theme} rating"
        elif gap < -75:
            pitch = f"a confidence builder ~{-gap:.0f} below your {theme} rating"
        else:
            pitch = f"a puzzle right at your {theme} level"
        why = f"PPO policy picked {pitch}"
        if theme == weakest:
            why += f" ({theme} is currently your weakest theme)"
        return Recommendation(theme, target, "ppo", why + ".")


def recommend_rating_match(state: CoachState, rng: np.random.Generator) -> Recommendation:
    """Lichess-style baseline: random theme, difficulty = your theme rating +/- 100."""
    theme = THEMES[int(rng.integers(len(THEMES)))]
    target = state.theme_ratings[theme] + float(rng.uniform(-100, 100))
    return Recommendation(theme, target, "rating_match", f"Rating-match baseline: {theme} near your {theme} rating.")


coach = PPOCoach()
