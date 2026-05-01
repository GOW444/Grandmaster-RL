"""
Evaluation Module
=================
Computes the Learning Efficiency Index (LEI), robustness score, and all
diagnostic metrics used in the paper's result tables and plots.

Key functions:
    :func:`compute_lei`        — scalar LEI from one agent's episode data
    :func:`compute_robustness` — ratio of eval-env LEI to train-env LEI
    :func:`evaluate_agent`     — full per-agent metrics dict
    :func:`evaluate_all`       — run all agents and return a results DataFrame
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from agents.baselines import rollout
from env.chess_env import ChessPuzzleEnv, _denormalize_rating
from env.learner_model import THEMES

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_lei(
    rho_0: float,
    rho_T: float,
    success_history: list[float] | np.ndarray,
    difficulty_history: list[float] | np.ndarray,
    sigma2_ref: float = 40_000.0,
) -> float:
    """Compute the Learning Efficiency Index (LEI) for one episode.

    Definition (from the paper, Eq. 6):
        LEI = (Δρ / T) × p̄ × (1 / (1 + Var(δ*) / σ²_ref))

    where:
        Δρ   = ρ_T − ρ_0   (total rating gain, in raw rating units)
        T    = len(success_history)  (number of puzzle attempts)
        p̄    = mean(success_history)  (success rate)
        Var(δ*) = variance of puzzle ratings presented (difficulty_history)
        σ²_ref  = reference variance (~200² = 40000) for normalising spread

    Higher LEI → faster skill gain at a moderate, stable difficulty level.

    Args:
        rho_0:              Initial overall rating (absolute, in [400, 3000]).
        rho_T:              Final overall rating (absolute, in [400, 3000]).
        success_history:    Sequence of 0/1 outcomes, length T.
        difficulty_history: Sequence of actual puzzle ratings (δ) seen, length T.
                            Values should be in [400, 3000] (denormalised).
        sigma2_ref:         Reference difficulty variance (default: 40 000).

    Returns:
        LEI as a float. Can be negative if Δρ < 0.
    """
    success_history = np.asarray(success_history, dtype=float)
    difficulty_history = np.asarray(difficulty_history, dtype=float)

    T = max(len(success_history), 1)
    delta_rho = rho_T - rho_0
    p_bar = float(np.mean(success_history))
    var_delta = float(np.var(difficulty_history))

    consistency_term = 1.0 / (1.0 + var_delta / sigma2_ref)
    lei = (delta_rho / T) * p_bar * consistency_term
    return float(lei)


def compute_robustness(lei_train: float, lei_eval: float) -> float:
    """Compute robustness score as the ratio of eval to train LEI.

    A value close to 1.0 indicates the policy transfers well to the unseen
    evaluation environment. Values below 0.5 suggest over-fitting to the
    simulated training dynamics.

    Args:
        lei_train: LEI measured on the training environment.
        lei_eval:  LEI measured on the held-out evaluation environment.

    Returns:
        Robustness score ∈ (0, ∞). Values > 1.0 are possible if the agent
        happens to perform better under fatigue (unusual but theoretically valid).
    """
    return lei_eval / (lei_train + 1e-8)


# ---------------------------------------------------------------------------
# Per-agent evaluation
# ---------------------------------------------------------------------------

def _run_rollout(
    agent: Any,
    env: ChessPuzzleEnv,
    n_episodes: int,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Thin wrapper around :func:`~agents.baselines.rollout`.

    Handles both SB3 models (which use ``.predict(obs, deterministic=True)``)
    and baseline agents (which use ``.predict(obs)``).
    """
    return rollout(agent, env, n_episodes=n_episodes, seed_offset=seed_offset)


def evaluate_agent(
    model_or_agent: Any,
    train_env: ChessPuzzleEnv,
    eval_env: ChessPuzzleEnv,
    n_episodes: int = 50,
) -> dict[str, Any]:
    """Evaluate a single agent on both training and evaluation environments.

    Args:
        model_or_agent: An SB3 model (``PPO`` / ``SAC``) or a baseline agent
                        with a ``.predict(obs)`` interface.
        train_env:      The training :class:`~env.chess_env.ChessPuzzleEnv`.
        eval_env:       The held-out :class:`~env.eval_env.EvalChessPuzzleEnv`.
        n_episodes:     Number of rollout episodes per environment.

    Returns:
        Dictionary with the following keys:

        ``lei_train``, ``lei_eval``, ``robustness``,
        ``mean_delta_rho_train``, ``mean_delta_rho_eval``,
        ``mean_success_rate_train``, ``mean_success_rate_eval``,
        ``per_theme_skill_gain_train``, ``per_theme_skill_gain_eval``,
        ``difficulty_variance_train``, ``difficulty_variance_eval``,
        ``success_rate_trajectories_train``, ``difficulty_trajectories_train``,
        ``success_rate_trajectories_eval``, ``difficulty_trajectories_eval``.
    """
    log.info("  Rolling out on TRAIN env (%d episodes) …", n_episodes)
    train_data = _run_rollout(model_or_agent, train_env, n_episodes, seed_offset=0)

    log.info("  Rolling out on EVAL env (%d episodes) …", n_episodes)
    eval_data = _run_rollout(model_or_agent, eval_env, n_episodes, seed_offset=1000)

    # Compute LEI from pooled episode data
    def _pool_lei(data: dict[str, Any]) -> tuple[float, float, float, float]:
        delta_rhos = data["delta_rho_per_episode"]
        rho_0 = 1000.0  # reference start (environment resets around mu0)
        rho_T_mean = rho_0 + float(np.mean(delta_rhos))
        lei = compute_lei(
            rho_0=rho_0,
            rho_T=rho_T_mean,
            success_history=data["success_history"],
            difficulty_history=data["difficulty_history"],
        )
        mean_drho = float(np.mean(delta_rhos))
        mean_sr = float(np.mean(data["success_history"]))
        diff_var = float(np.var(data["difficulty_history"]))
        return lei, mean_drho, mean_sr, diff_var

    lei_train, mean_drho_train, mean_sr_train, diff_var_train = _pool_lei(train_data)
    lei_eval, mean_drho_eval, mean_sr_eval, diff_var_eval = _pool_lei(eval_data)
    robustness = compute_robustness(lei_train, lei_eval)

    return {
        "lei_train": lei_train,
        "lei_eval": lei_eval,
        "robustness": robustness,
        "mean_delta_rho_train": mean_drho_train,
        "mean_delta_rho_eval": mean_drho_eval,
        "mean_success_rate_train": mean_sr_train,
        "mean_success_rate_eval": mean_sr_eval,
        "per_theme_skill_gain_train": train_data["per_theme_skill_gain"],
        "per_theme_skill_gain_eval": eval_data["per_theme_skill_gain"],
        "difficulty_variance_train": diff_var_train,
        "difficulty_variance_eval": diff_var_eval,
        "success_rate_trajectories_train": train_data["success_rate_trajectories"],
        "difficulty_trajectories_train": train_data["difficulty_trajectories"],
        "success_rate_trajectories_eval": eval_data["success_rate_trajectories"],
        "difficulty_trajectories_eval": eval_data["difficulty_trajectories"],
    }


# ---------------------------------------------------------------------------
# Multi-agent comparison table
# ---------------------------------------------------------------------------

def evaluate_all(
    models: dict[str, Any],
    train_env: ChessPuzzleEnv,
    eval_env: ChessPuzzleEnv,
    n_episodes: int = 50,
) -> pd.DataFrame:
    """Evaluate all agents and aggregate results into a tidy DataFrame.

    Args:
        models:      Mapping of agent name → model/agent instance, e.g.
                     ``{"PPO": ppo_model, "SAC": sac_model, "Random": rand}``.
        train_env:   Training environment.
        eval_env:    Held-out evaluation environment.
        n_episodes:  Episodes per agent per environment.

    Returns:
        :class:`pandas.DataFrame` with one row per agent and columns:

        ``agent``, ``lei_train``, ``lei_eval``, ``robustness``,
        ``mean_delta_rho_train``, ``mean_delta_rho_eval``,
        ``mean_success_rate_train``, ``mean_success_rate_eval``,
        ``difficulty_variance_train``, ``difficulty_variance_eval``,
        plus per-theme gain columns ``theme_{t}_gain_train`` and
        ``theme_{t}_gain_eval`` for each t in THEMES.
    """
    rows: list[dict[str, Any]] = []
    for name, agent in models.items():
        log.info("Evaluating agent: %s", name)
        metrics = evaluate_agent(agent, train_env, eval_env, n_episodes=n_episodes)

        row: dict[str, Any] = {"agent": name}
        row["lei_train"] = metrics["lei_train"]
        row["lei_eval"] = metrics["lei_eval"]
        row["robustness"] = metrics["robustness"]
        row["mean_delta_rho_train"] = metrics["mean_delta_rho_train"]
        row["mean_delta_rho_eval"] = metrics["mean_delta_rho_eval"]
        row["mean_success_rate_train"] = metrics["mean_success_rate_train"]
        row["mean_success_rate_eval"] = metrics["mean_success_rate_eval"]
        row["difficulty_variance_train"] = metrics["difficulty_variance_train"]
        row["difficulty_variance_eval"] = metrics["difficulty_variance_eval"]

        for i, theme in enumerate(THEMES):
            row[f"theme_{theme}_gain_train"] = float(
                metrics["per_theme_skill_gain_train"][i]
            )
            row[f"theme_{theme}_gain_eval"] = float(
                metrics["per_theme_skill_gain_eval"][i]
            )

        rows.append(row)

    df = pd.DataFrame(rows)
    return df
