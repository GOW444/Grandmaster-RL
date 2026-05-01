"""
Evaluation Environment
======================
Subclass of ChessPuzzleEnv that injects two robustness-testing perturbations:

1. **Fatigue**: The learner's effective skill degrades linearly over the
   episode:  f(t) = 1 - β*(t/T),  β = 0.15.
   Only affects the solve probability — skill updates still use real φ.

2. **Temperature jitter**: Per-step τ' = τ + N(0, σ_τ²),  σ_τ = 20.0.
   This alters the curvature of the IRT solve curve each step.

This environment is **never used during training** — only during the final
robustness evaluation after training completes.
"""

import numpy as np

from env.chess_env import ChessPuzzleEnv
from env.learner_model import TAU


class EvalChessPuzzleEnv(ChessPuzzleEnv):
    """Held-out evaluation environment with fatigue and temperature jitter.

    Inherits all behaviour from :class:`~env.chess_env.ChessPuzzleEnv` and
    overrides only the hook that computes the effective skill and tau used in
    :func:`~env.learner_model.solve_prob`.

    Args:
        beta:    Fatigue slope (β). Effective skill = φ * (1 - β * t/T).
        sigma_tau: Standard deviation of the per-step temperature jitter (σ_τ).
        **kwargs: Forwarded unchanged to :class:`ChessPuzzleEnv`.
    """

    def __init__(
        self,
        beta: float = 0.15,
        sigma_tau: float = 20.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.sigma_tau = sigma_tau

    # ------------------------------------------------------------------
    # Override the hook that supplies skill + tau to solve_prob
    # ------------------------------------------------------------------

    def _get_effective_skill_and_tau(
        self, theme_idx: int
    ) -> tuple[float, float]:
        """Inject fatigue and temperature jitter before solve probability computation.

        Fatigue scalar:       f(t) = 1 − β * (t / T)
        Temperature jitter:  τ' = τ + N(0, σ_τ²)

        The fatigued skill and jittered tau are used **only** inside
        ``solve_prob``; the :func:`~env.learner_model.update_skill` call in
        :meth:`ChessPuzzleEnv.step` always receives the real φ.

        Args:
            theme_idx: Index of the selected theme.

        Returns:
            Tuple ``(effective_skill, effective_tau)``.
        """
        # Fatigue: linearly degrades skill over the episode
        fatigue_factor = 1.0 - self.beta * (self._step_count / self.T)
        effective_skill = float(self._skills[theme_idx]) * fatigue_factor

        # Temperature jitter: per-step noise on τ
        jitter = self._np_rng.normal(0.0, self.sigma_tau)
        effective_tau = max(self.tau + jitter, 1.0)  # keep tau strictly positive

        return effective_skill, effective_tau
