# Grandmaster-RL

> **RL-Based Adaptive Chess Puzzle Training System**
> DSAI Project — IIIT Bangalore

## Overview

Grandmaster-RL trains a reinforcement learning agent to construct personalized chess puzzle curricula for simulated learners. Rather than the standard "rating-match" heuristic used by platforms like Lichess and Chess.com — where a player simply receives the next puzzle within ±Δ of their current rating — this system models the learner as a Markov Decision Process and optimizes long-term skill gain using PPO and SAC.

The agent selects a tactical theme (`fork`, `pin`, `mate`, `endgame`, `skewer`, `discovery`) and a target difficulty rating at each step. A logistic IRT model simulates whether the learner solves the puzzle, and their per-theme skill estimates are updated accordingly. The primary metric is the **Learning Efficiency Index (LEI)**, which jointly measures rating gain, success rate, and difficulty consistency. Robustness is assessed by evaluating trained policies on a held-out environment that injects learner fatigue and IRT temperature jitter — dynamics never seen during training.

## Project Structure

```
Grandmaster-RL/
├── data/
│   ├── processed/          # Filtered per-theme CSVs (generated)
│   └── indices/            # KD-tree .pkl files (generated)
├── env/
│   ├── chess_env.py        # Training Gymnasium environment
│   ├── eval_env.py         # Held-out eval env (fatigue + jitter)
│   └── learner_model.py    # IRT logistic model + skill update
├── agents/
│   └── baselines.py        # Random, RatingMatch, FixedProgression
├── networks/
│   └── hybrid_policy.py    # Custom SB3 ActorCriticPolicy (hybrid head)
├── training/
│   ├── train_ppo.py
│   ├── train_sac.py
│   └── configs/
│       ├── ppo.yaml
│       └── sac.yaml
├── evaluation/
│   ├── evaluate.py         # LEI, robustness, and multi-agent comparison
│   └── visualize.py        # All 6 publication plots
├── scripts/
│   └── build_dataset.py    # Phase 1: filter CSV + build KD-tree indices
├── tests/
│   └── test_env.py         # Pytest sanity checks
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Data Setup

Place `lichess_puzzles_reduced.csv` in the project root, then run:

```bash
python scripts/build_dataset.py --csv lichess_puzzles_reduced.csv
```

This will:
- Filter puzzles by rating quality (RatingDeviation ≤ 150, Rating ∈ [400, 3000])
- Assign each puzzle to one of 6 primary tactical themes
- Save per-theme CSVs to `data/processed/`
- Build and serialize 1D KD-trees to `data/indices/`

The script is idempotent — re-running it won't overwrite existing files unless you pass `--force`.

## Training

### PPO (recommended)

```bash
python training/train_ppo.py \
    --config training/configs/ppo.yaml \
    --timesteps 500000 \
    --seed 42 \
    --output_dir checkpoints/ppo
```

### SAC

```bash
python training/train_sac.py \
    --config training/configs/sac.yaml \
    --timesteps 500000 \
    --seed 42 \
    --output_dir checkpoints/sac
```

Checkpoints are saved every 10% of total timesteps. The best model (by training-env episodic reward) is saved to `checkpoints/{algo}/best_model/`. TensorBoard logs go to `checkpoints/{algo}/tensorboard/`.

```bash
tensorboard --logdir checkpoints/
```

## Evaluation

Run the comparison evaluation across all agents (PPO, SAC, and all 3 baselines):

```python
from stable_baselines3 import PPO, SAC
from env.chess_env import ChessPuzzleEnv
from env.eval_env import EvalChessPuzzleEnv
from agents.baselines import RandomAgent, RatingMatchAgent, FixedProgressionAgent
from evaluation.evaluate import evaluate_all
from evaluation.visualize import plot_all

train_env = ChessPuzzleEnv()
eval_env  = EvalChessPuzzleEnv()

models = {
    "PPO":              PPO.load("checkpoints/ppo/ppo_final"),
    "SAC":              SAC.load("checkpoints/sac/sac_final"),
    "Random":           RandomAgent(),
    "RatingMatch":      RatingMatchAgent(),
    "FixedProgression": FixedProgressionAgent(),
}

results_df = evaluate_all(models, train_env, eval_env, n_episodes=50)
print(results_df)

# Generate all 6 paper plots → results/plots/
# (trajectories dict must be collected during evaluate_all — see evaluate.py)
```

## Running Tests

```bash
# Requires data/indices/ to be built first
pytest tests/test_env.py -v
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Box action space (theme as float) | Most SB3-compatible; avoids custom rollout buffer |
| KD-tree per theme (1D on rating) | Full rating granularity, O(log n) lookup, no binning |
| IRT logistic solve model | Theoretically grounded; matches Elo/Glicko intuition |
| Eval env never seen during training | Proper robustness measurement |
| LEI metric | Captures gain, success rate, and difficulty consistency jointly |

## References

1. Glickman, M. E. (1995). The Glicko system. *Boston University*.
2. Lichess Open Puzzle Database. https://database.lichess.org/#puzzles
3. Schulman et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
4. Haarnoja et al. (2018). Soft Actor-Critic. *ICML 2018*.
5. Jang et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR 2017*.
