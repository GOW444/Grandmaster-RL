"""
Reproduce the FINAL_REPORT.md results table from the saved checkpoints.

Usage (after `python scripts/build_dataset.py --csv <csv>`):
    python scripts/verify_results.py [--episodes 50]

Loads PPO/SAC exactly as the README does (no retraining), runs the seeded
evaluation on the train and held-out eval envs, and prints the metrics next
to the numbers reported in FINAL_REPORT.md.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from stable_baselines3 import PPO, SAC

from agents.baselines import FixedProgressionAgent, RandomAgent, RatingMatchAgent
from env.chess_env import ChessPuzzleEnv
from env.eval_env import EvalChessPuzzleEnv
from evaluation.evaluate import evaluate_all

REPORTED = pd.DataFrame(
    [
        ["PPO", 0.0024, 0.0016, 0.6674, 6.38, 4.63, 0.754, 0.697],
        ["SAC", 0.0012, 0.0005, 0.4077, 16.65, 11.34, 0.190, 0.133],
        ["Random", 0.0000, 0.0000, 0.6413, 3.94, 2.57, 0.234, 0.233],
        ["RatingMatch", 0.0018, 0.0012, 0.6667, 8.56, 6.48, 0.485, 0.419],
        ["FixedProgression", 0.0019, 0.0013, 0.7149, 7.19, 5.57, 0.582, 0.536],
    ],
    columns=[
        "agent", "lei_train", "lei_eval", "robustness",
        "mean_delta_rho_train", "mean_delta_rho_eval",
        "mean_success_rate_train", "mean_success_rate_eval",
    ],
).set_index("agent")


def main() -> None:
    parser = argparse.ArgumentParser()
    # The report's table was produced from the Streamlit GUI, whose episode slider defaults to 20.
    # LEI pools all steps (T = episodes * 100), so it only reproduces at 20 episodes.
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--indices_dir", default=str(ROOT / "data" / "indices"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    ckpt = ROOT / "checkpoints"
    models = {
        "PPO": PPO.load(ckpt / "ppo" / "ppo_final", device="cpu"),
        "SAC": SAC.load(ckpt / "sac" / "sac_final", device="cpu"),
        "Random": RandomAgent(),
        "RatingMatch": RatingMatchAgent(),
        "FixedProgression": FixedProgressionAgent(),
    }
    train_env = ChessPuzzleEnv(indices_dir=args.indices_dir)
    eval_env = EvalChessPuzzleEnv(indices_dir=args.indices_dir)

    got = evaluate_all(models, train_env, eval_env, n_episodes=args.episodes)
    got = got.set_index("agent")[REPORTED.columns]

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== Reproduced ===")
    print(got)
    print("\n=== Reported (FINAL_REPORT.md) ===")
    print(REPORTED)
    print("\n=== Reproduced - Reported ===")
    print(got - REPORTED)


if __name__ == "__main__":
    main()
