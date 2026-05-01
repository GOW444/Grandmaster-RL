"""
PPO Training Script
===================
Trains a PPO agent with the HybridPolicy on the ChessPuzzleEnv.

Usage:
    python training/train_ppo.py [--config ...] [--timesteps ...] [--seed ...] [--output_dir ...]

Flow:
    1. Load hyperparameter config from YAML
    2. Set global seeds (numpy, torch, random, Python)
    3. Instantiate training env (ChessPuzzleEnv) and eval env (EvalChessPuzzleEnv)
    4. Wrap training env with Monitor + VecNormalize
    5. Build PPO model with HybridPolicy
    6. Attach EvalCallback on the training env for mid-training checkpointing
    7. model.learn(total_timesteps=...)
    8. Save final model + VecNormalize stats
    9. Post-training evaluation with evaluate_agent(); print summary table
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.chess_env import ChessPuzzleEnv
from env.eval_env import EvalChessPuzzleEnv
from evaluation.evaluate import evaluate_agent
from networks.hybrid_policy import HybridPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info("Seeds set to %d (device: %s)", seed, DEVICE)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """Load a YAML hyperparameter config file.

    Args:
        path: Path to the ``.yaml`` config file.

    Returns:
        Dictionary of hyperparameters.
    """
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    log.info("Loaded config from %s: %s", path, cfg)
    return cfg


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_train_env(seed: int, indices_dir: str = "data/indices") -> VecNormalize:
    """Create, wrap, and normalise the training environment.

    Args:
        seed:        Random seed.
        indices_dir: Path to the directory containing KD-tree pkl files.

    Returns:
        A :class:`~stable_baselines3.common.vec_env.VecNormalize`-wrapped
        :class:`~stable_baselines3.common.vec_env.DummyVecEnv`.
    """
    def _make() -> ChessPuzzleEnv:
        env = ChessPuzzleEnv(indices_dir=indices_dir, seed=seed)
        return Monitor(env)

    vec_env = DummyVecEnv([_make])
    # Normalise observations + rewards for more stable training
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    return vec_env


def make_eval_env(seed: int, indices_dir: str = "data/indices") -> ChessPuzzleEnv:
    """Create the held-out evaluation environment (unwrapped, for accurate metrics).

    Args:
        seed:        Random seed.
        indices_dir: Path to the directory containing KD-tree pkl files.

    Returns:
        An :class:`~env.eval_env.EvalChessPuzzleEnv` instance.
    """
    return EvalChessPuzzleEnv(indices_dir=indices_dir, seed=seed + 999)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    config_path: Path,
    total_timesteps: int,
    seed: int,
    output_dir: Path,
    indices_dir: str = "data/indices",
) -> None:
    """Run the full PPO training pipeline.

    Args:
        config_path:      Path to ``configs/ppo.yaml``.
        total_timesteps:  Total environment steps to train for.
        seed:             Global random seed.
        output_dir:       Directory where checkpoints and the final model are saved.
        indices_dir:      Path to the KD-tree index directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(seed)
    cfg = load_config(config_path)

    # --- Environments ---
    log.info("Building environments …")
    train_env = make_train_env(seed, indices_dir)
    # A separate VecEnv is used inside EvalCallback to avoid contaminating stats
    eval_vec_env_for_callback = DummyVecEnv(
        [lambda: Monitor(ChessPuzzleEnv(indices_dir=indices_dir, seed=seed + 1))]
    )

    # --- Model ---
    log.info("Building PPO model with HybridPolicy …")
    model = PPO(
        policy=HybridPolicy,
        env=train_env,
        n_steps=cfg.get("n_steps", 2048),
        batch_size=cfg.get("batch_size", 256),
        n_epochs=cfg.get("n_epochs", 10),
        learning_rate=cfg.get("learning_rate", 3e-4),
        gamma=cfg.get("gamma", 0.99),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.01),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        max_grad_norm=cfg.get("max_grad_norm", 0.5),
        verbose=cfg.get("verbose", 1),
        tensorboard_log=str(output_dir / "tensorboard"),
        seed=seed,
        device=DEVICE,
    )
    log.info("Model: %s", model)

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=max(total_timesteps // 10, 10_000),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_chess",
    )
    eval_cb = EvalCallback(
        eval_env=eval_vec_env_for_callback,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # --- Training ---
    log.info("Starting PPO training for %d timesteps …", total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # --- Save final model ---
    final_model_path = output_dir / "ppo_final"
    model.save(str(final_model_path))
    # Save VecNormalize statistics so they can be loaded at inference time
    vec_norm_path = output_dir / "vec_normalize.pkl"
    train_env.save(str(vec_norm_path))
    log.info("Final model saved to %s", final_model_path)
    log.info("VecNormalize stats saved to %s", vec_norm_path)

    # --- Post-training evaluation ---
    log.info("Running post-training evaluation …")
    # For fair evaluation, load model onto a fresh unwrapped environment
    bare_train_env = ChessPuzzleEnv(indices_dir=indices_dir, seed=seed + 200)
    bare_eval_env = make_eval_env(seed, indices_dir)

    # Load the saved model for evaluation (uses the final checkpoint)
    loaded_model = PPO.load(str(final_model_path), device=DEVICE)

    metrics = evaluate_agent(
        loaded_model,
        bare_train_env,
        bare_eval_env,
        n_episodes=20,
    )

    print("\n" + "=" * 60)
    print("PPO EVALUATION RESULTS")
    print("=" * 60)
    print(f"  LEI (train env):          {metrics['lei_train']:.4f}")
    print(f"  LEI (eval env):           {metrics['lei_eval']:.4f}")
    print(f"  Robustness score:         {metrics['robustness']:.4f}")
    print(f"  Mean Δρ (train):          {metrics['mean_delta_rho_train']:.2f}")
    print(f"  Mean Δρ (eval):           {metrics['mean_delta_rho_eval']:.2f}")
    print(f"  Mean success rate (train):{metrics['mean_success_rate_train']:.3f}")
    print(f"  Mean success rate (eval): {metrics['mean_success_rate_eval']:.3f}")
    print("=" * 60 + "\n")

    train_env.close()
    eval_vec_env_for_callback.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on ChessPuzzleEnv.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/configs/ppo.yaml"),
        help="Path to ppo.yaml config file.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500 000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("checkpoints/ppo"),
        help="Directory for checkpoints and final model.",
    )
    parser.add_argument(
        "--indices_dir",
        type=str,
        default="data/indices",
        help="Directory containing KD-tree .pkl files.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if not args.config.exists():
        log.error("Config not found: %s", args.config)
        sys.exit(1)
    train(
        config_path=args.config,
        total_timesteps=args.timesteps,
        seed=args.seed,
        output_dir=args.output_dir,
        indices_dir=args.indices_dir,
    )
