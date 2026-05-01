"""
Visualization Module
====================
Generates all publication-quality plots for the Grandmaster-RL paper.
All figures are saved to ``results/plots/`` at 300 DPI.

Functions:
    :func:`plot_lei_comparison`       — Grouped bar: LEI_train vs LEI_eval per agent
    :func:`plot_skill_improvement`    — Bar chart of mean Δρ per agent
    :func:`plot_theme_heatmap`        — Per-theme skill gain heatmap
    :func:`plot_success_trajectory`   — Rolling-mean success rate over episode
    :func:`plot_difficulty_progression` — Difficulty δ* over episode steps
    :func:`plot_robustness`           — Horizontal bar chart of robustness scores
    :func:`plot_all`                  — Convenience wrapper calling all of the above
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from env.learner_model import THEMES

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
sns.set_theme(style="darkgrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
})

_RL_PALETTE = ["#3A86FF", "#8338EC"]   # PPO blue, SAC purple
_BASELINE_PALETTE = ["#888888", "#AAAAAA", "#CCCCCC"]  # muted greys
_DIVERGING_CMAP = "RdYlGn"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _rl_vs_baseline_colors(agents: list[str]) -> list[str]:
    """Assign blue shades to RL agents and greys to baselines."""
    rl_names = {"PPO", "SAC"}
    rl_idx = 0
    bl_idx = 0
    colors: list[str] = []
    for name in agents:
        if name in rl_names:
            colors.append(_RL_PALETTE[rl_idx % len(_RL_PALETTE)])
            rl_idx += 1
        else:
            colors.append(_BASELINE_PALETTE[bl_idx % len(_BASELINE_PALETTE)])
            bl_idx += 1
    return colors


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_lei_comparison(results_df: pd.DataFrame, save_path: Path) -> None:
    """Grouped bar chart of LEI_train and LEI_eval for each agent.

    RL agents are shown in blue shades; baselines in grey.

    Args:
        results_df: DataFrame from :func:`~evaluation.evaluate.evaluate_all`.
        save_path:  File path (including filename) to save the PNG.
    """
    _ensure_dir(save_path.parent)
    agents = results_df["agent"].tolist()
    x = np.arange(len(agents))
    width = 0.35

    colors = _rl_vs_baseline_colors(agents)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_train = ax.bar(
        x - width / 2,
        results_df["lei_train"],
        width,
        label="LEI (train)",
        color=colors,
        alpha=0.9,
        edgecolor="white",
    )
    bars_eval = ax.bar(
        x + width / 2,
        results_df["lei_eval"],
        width,
        label="LEI (eval)",
        color=colors,
        alpha=0.5,
        edgecolor="white",
        hatch="///",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha="right")
    ax.set_ylabel("Learning Efficiency Index (LEI)")
    ax.set_title("LEI Comparison: Train vs. Held-Out Eval Environment")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # Annotate bars
    for bar in list(bars_train) + list(bars_eval):
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    log.info("Saved: %s", save_path)


def plot_skill_improvement(results_df: pd.DataFrame, save_path: Path) -> None:
    """Bar chart of mean Δρ (overall rating gain) per agent.

    Args:
        results_df: DataFrame from :func:`~evaluation.evaluate.evaluate_all`.
        save_path:  Destination PNG path.
    """
    _ensure_dir(save_path.parent)
    agents = results_df["agent"].tolist()
    colors = _rl_vs_baseline_colors(agents)
    x = np.arange(len(agents))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width / 2,
        results_df["mean_delta_rho_train"],
        width,
        label="Train env",
        color=colors,
        alpha=0.9,
        edgecolor="white",
    )
    ax.bar(
        x + width / 2,
        results_df["mean_delta_rho_eval"],
        width,
        label="Eval env",
        color=colors,
        alpha=0.5,
        hatch="///",
        edgecolor="white",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha="right")
    ax.set_ylabel("Mean Rating Gain (Δρ, rating points)")
    ax.set_title("Overall Skill Improvement per Agent")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    log.info("Saved: %s", save_path)


def plot_theme_heatmap(results_df: pd.DataFrame, save_path: Path) -> None:
    """Heatmap of per-theme skill gain: rows = agents, columns = themes.

    Uses a diverging colormap centred at 0 so improvements and regressions
    are immediately visible.

    Args:
        results_df: DataFrame from :func:`~evaluation.evaluate.evaluate_all`.
        save_path:  Destination PNG path.
    """
    _ensure_dir(save_path.parent)
    agents = results_df["agent"].tolist()
    train_cols = [f"theme_{t}_gain_train" for t in THEMES]
    heat_data = results_df[train_cols].values  # shape (n_agents, 6)

    # Symmetric color range
    vmax = np.abs(heat_data).max()
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(10, len(agents) * 0.9 + 2))
    sns.heatmap(
        heat_data,
        ax=ax,
        annot=True,
        fmt=".2f",
        xticklabels=THEMES,
        yticklabels=agents,
        cmap=_DIVERGING_CMAP,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        cbar_kws={"label": "Skill gain Δφ (rating pts)"},
    )
    ax.set_title("Per-Theme Skill Gain Heatmap (Train Env)")
    ax.set_xlabel("Tactical Theme")
    ax.set_ylabel("Agent")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    log.info("Saved: %s", save_path)


def plot_success_trajectory(
    trajectories: dict[str, list[list[float]]],
    save_path: Path,
    smooth_window: int = 10,
) -> None:
    """Line plot of rolling-mean success rate over episode steps.

    Args:
        trajectories:  Dict mapping agent name → list of per-episode p̄_t arrays.
                       Each inner list has length T (e.g., 100).
        save_path:     Destination PNG path.
        smooth_window: Rolling average window for smoothing (default: 10).
    """
    _ensure_dir(save_path.parent)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(trajectories)))

    for (name, eps), color in zip(trajectories.items(), colors):
        # Pad episodes to the same length and average
        max_t = max(len(e) for e in eps)
        matrix = np.full((len(eps), max_t), np.nan)
        for i, ep in enumerate(eps):
            matrix[i, : len(ep)] = ep

        mean_traj = np.nanmean(matrix, axis=0)
        # Rolling mean
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(mean_traj, kernel, mode="valid")
        steps = np.arange(len(smoothed)) + smooth_window // 2

        ax.plot(steps, smoothed, label=name, color=color, linewidth=2)
        sem = np.nanstd(matrix, axis=0) / np.sqrt(len(eps))
        sem_smooth = np.convolve(sem[: len(smoothed) + smooth_window - 1], kernel, mode="valid")[
            : len(smoothed)
        ]
        ax.fill_between(steps, smoothed - sem_smooth, smoothed + sem_smooth, alpha=0.15, color=color)

    ax.set_xlabel("Episode Step")
    ax.set_ylabel("Success Rate (rolling mean)")
    ax.set_title("Success Rate Trajectory (averaged over episodes)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    log.info("Saved: %s", save_path)


def plot_difficulty_progression(
    trajectories: dict[str, list[list[float]]],
    save_path: Path,
    mean_initial_rating: float = 1000.0,
) -> None:
    """Line plot of puzzle difficulty δ* over episode steps (denormalized).

    Includes a horizontal reference line at the mean initial learner rating.

    Args:
        trajectories:        Dict mapping agent name → per-episode difficulty arrays
                             in absolute rating units [400, 3000].
        save_path:           Destination PNG path.
        mean_initial_rating: Reference horizontal line (default: 1000.0).
    """
    _ensure_dir(save_path.parent)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(trajectories)))

    for (name, eps), color in zip(trajectories.items(), colors):
        max_t = max(len(e) for e in eps)
        matrix = np.full((len(eps), max_t), np.nan)
        for i, ep in enumerate(eps):
            matrix[i, : len(ep)] = ep

        mean_traj = np.nanmean(matrix, axis=0)
        smooth_window = 5
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(mean_traj, kernel, mode="same")

        ax.plot(np.arange(len(smoothed)), smoothed, label=name, color=color, linewidth=2)

    ax.axhline(
        mean_initial_rating,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean initial rating ({mean_initial_rating:.0f})",
    )
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("Puzzle Difficulty (Glicko-2 rating)")
    ax.set_title("Difficulty Progression over Episode")
    ax.set_ylim(400, 3000)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    log.info("Saved: %s", save_path)


def plot_robustness(results_df: pd.DataFrame, save_path: Path) -> None:
    """Horizontal bar chart of robustness scores with reference line at 1.0.

    Args:
        results_df: DataFrame from :func:`~evaluation.evaluate.evaluate_all`.
        save_path:  Destination PNG path.
    """
    _ensure_dir(save_path.parent)
    agents = results_df["agent"].tolist()
    scores = results_df["robustness"].tolist()
    colors = _rl_vs_baseline_colors(agents)

    fig, ax = plt.subplots(figsize=(9, max(4, len(agents) * 0.8 + 1)))
    y = np.arange(len(agents))
    ax.barh(y, scores, color=colors, alpha=0.85, edgecolor="white", height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(agents)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.4, label="Perfect robustness (1.0)")
    ax.set_xlabel("Robustness Score (LEI_eval / LEI_train)")
    ax.set_title("Robustness to Unseen Learner Dynamics")
    ax.legend()

    for i, (score, bar_y) in enumerate(zip(scores, y)):
        ax.text(
            max(score + 0.01, 0.01),
            bar_y,
            f"{score:.3f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    log.info("Saved: %s", save_path)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def plot_all(
    results_df: pd.DataFrame,
    trajectories: dict[str, dict[str, list[list[float]]]],
    output_dir: str | Path = "results/plots",
) -> None:
    """Generate all plots and save them to *output_dir*.

    Args:
        results_df:   Evaluation results DataFrame.
        trajectories: Dict mapping agent name → dict with keys
                      ``"success"`` and ``"difficulty"``, each a list of
                      per-episode step arrays.
        output_dir:   Directory where all PNGs will be saved.
    """
    out = Path(output_dir)
    _ensure_dir(out)

    success_traj = {name: data["success"] for name, data in trajectories.items()}
    diff_traj = {name: data["difficulty"] for name, data in trajectories.items()}

    plot_lei_comparison(results_df, out / "lei_comparison.png")
    plot_skill_improvement(results_df, out / "skill_improvement.png")
    plot_theme_heatmap(results_df, out / "theme_heatmap.png")
    plot_success_trajectory(success_traj, out / "success_trajectory.png")
    plot_difficulty_progression(diff_traj, out / "difficulty_progression.png")
    plot_robustness(results_df, out / "robustness.png")

    log.info("All plots saved to %s", out)
