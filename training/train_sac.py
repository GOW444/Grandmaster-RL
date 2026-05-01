"""
SAC Training Script
===================
Trains a SAC agent with the HybridPolicy on the ChessPuzzleEnv.

Usage:
    python training/train_sac.py [--config ...] [--timesteps ...] [--seed ...] [--output_dir ...]

Implementation note on SAC + discrete theme action:
----------------------------------------------------
SAC is designed for fully continuous action spaces. The theme component of
our hybrid action is treated as a continuous float in [0, N_THEMES) and
rounded to an integer at environment step time. This means:

  1. SAC's critic receives the continuous float theme value, which is
     slightly inconsistent with the actual discrete theme chosen. In
     practice, since the rounding introduces only a small quantization
     error and N_THEMES is small, training still converges.

  2. The theme head in HybridPolicy outputs a Categorical distribution,
     but SAC's policy gradient relies on the reparameterization trick.
     Categoricals are not reparameterizable via the standard trick.

  TODO: For a rigorous implementation, replace the Categorical theme head
        with Gumbel-Softmax reparameterization (temperature annealed from
        ~1.0 to ~0.1 over training). This would allow gradients to flow
        through the discrete theme selection similarly to SAC's usual
        squashed-Gaussian continuous head.
        Reference: Jang et al. (2017) "Categorical Reparameterization with
        Gumbel-Softmax", ICLR 2017.

  TODO: SAC's ContinuousCritic uses Q(s, a) where a is the full action
        vector. Rounding action[0] post-sampling before feeding to the
        critic would make Q-values more accurate for the discrete component.

Despite these approximations, SAC is included as a comparison point;
PPO is the primary recommended algorithm for this environment.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

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
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    config_path: Path,
    total_timesteps: int,
    seed: int,
    output_dir: Path,
    indices_dir: str = "data/indices",
) -> None:
    """Run the full SAC training pipeline.

    Args:
        config_path:      Path to ``configs/sac.yaml``.
        total_timesteps:  Total environment steps to train for.
        seed:             Global random seed.
        output_dir:       Directory where checkpoints and the final model are saved.
        indices_dir:      Path to the KD-tree index directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(seed)
    cfg = load_config(config_path)

    # --- Environment (SAC uses an unwrapped DummyVecEnv) ---
    log.info("Building training environment …")
    train_env = DummyVecEnv(
        [lambda: Monitor(ChessPuzzleEnv(indices_dir=indices_dir, seed=seed))]
    )

    # --- Model ---
    log.info("Building SAC model with HybridPolicy …")
    ent_coef = cfg.get("ent_coef", "auto")
    model = SAC(
        policy=HybridPolicy,
        env=train_env,
        learning_rate=cfg.get("learning_rate", 3e-4),
        buffer_size=cfg.get("buffer_size", 100_000),
        batch_size=cfg.get("batch_size", 256),
        gamma=cfg.get("gamma", 0.99),
        tau=cfg.get("tau", 0.005),
        ent_coef=ent_coef,
        train_freq=cfg.get("train_freq", 1),
        gradient_steps=cfg.get("gradient_steps", 1),
        learning_starts=cfg.get("learning_starts", 1_000),
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
        name_prefix="sac_chess",
    )

    # --- Training ---
    log.info("Starting SAC training for %d timesteps …", total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb],
        progress_bar=True,
    )

    # --- Save final model ---
    final_model_path = output_dir / "sac_final"
    model.save(str(final_model_path))
    log.info("Final model saved to %s", final_model_path)

    # --- Post-training evaluation ---
    log.info("Running post-training evaluation …")
    bare_train_env = ChessPuzzleEnv(indices_dir=indices_dir, seed=seed + 200)
    bare_eval_env = EvalChessPuzzleEnv(indices_dir=indices_dir, seed=seed + 999)

    loaded_model = SAC.load(str(final_model_path), device=DEVICE)
    metrics = evaluate_agent(
        loaded_model,
        bare_train_env,
        bare_eval_env,
        n_episodes=20,
    )

    print("\n" + "=" * 60)
    print("SAC EVALUATION RESULTS")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on ChessPuzzleEnv.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/configs/sac.yaml"),
        help="Path to sac.yaml config file.",
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
        default=Path("checkpoints/sac"),
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
