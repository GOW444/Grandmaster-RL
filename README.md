# Grandmaster-RL

> **RL-powered chess tactics trainer** — a full-stack puzzle platform whose AI coach is a PPO curriculum policy.
> Started as a DSAI course project at IIIT Bangalore (the RL engine, preserved on the [`engine`](../../tree/engine) branch) and extended into a web app.

## The platform

| | |
|---|---|
| **Train** | Real Lichess puzzles on an interactive board (drag or click, promotion, hints, show-solution, alternate mates accepted). Modes: AI Coach, Rated, Theme drill, Daily, Review-your-misses. |
| **AI Coach** | The trained PPO policy runs on a real user's state (6 theme ratings + last-10 results) and picks theme + difficulty. A/B mode randomly assigns each puzzle to PPO or a Lichess-style rating-match baseline and logs outcomes, so the simulated-learner result can be checked on real people (**Coach Lab** page). |
| **Stats** | Per-theme ratings (radar), rating history, accuracy & avg time per theme, accuracy-vs-time-spent, activity heatmap, auto-detected areas to improve. |
| **Puzzle Battle** | Timed (3/5 min), 3-strike puzzle races on an escalating ladder: solo rush, private rooms with a join code / invite link (up to 8), or quick-match a random player. Live standings via polling; answers are re-validated server-side with python-chess so scores can't be forged. |
| **Custom training** | Pick any rating range (+ optional theme) and train inside it, or browse/filter the catalogue (rating, theme, solved/missed/unseen, popularity) and play a specific puzzle. |
| **Gamification** | XP & levels, daily streaks, 9 badges, leaderboard (rating / XP / streak). |
| **Library** | Save puzzles, browse full attempt history, replay any puzzle. |

**Stack:** FastAPI · SQLAlchemy 2 · PostgreSQL · JWT auth (bcrypt) · Stable-Baselines3 (inference only) · React 19 + TypeScript · Vite · Tailwind v4 · react-chessboard + chess.js · Recharts · pytest · Playwright (e2e smoke).

### How the RL engine plugs in

```
user's DB rows ──► [overall, 6 theme ratings, rolling success, rolling difficulty] ──► PPO (engine checkpoint)
                    (identical to env/chess_env.py::_build_state — enforced by a test)      │
                                                                                            ▼
             Postgres B-tree on (primary_theme, rating)  ◄── nearest unseen puzzle ── (theme, target rating)
             (replaces the engine's per-theme KD-tree)
```

- Ratings use the engine's own IRT curve `P(solve) = σ((φ − δ)/200)` as the expected score, with an Elo-style K-factor update (the engine's α=0.05 rule is tuned for 100-step simulated episodes and is far too slow for humans).
- Only a user's first attempt at a puzzle is rated; hints count as a miss for rating purposes.
- No retraining: `checkpoints/ppo/ppo_final.zip` is loaded read-only.

### Data model

`users` · `races` / `race_players` · `theme_skills (user, theme → rating, attempts, solved, time)` · `puzzles (Lichess id, FEN, UCI moves, rating, themes[], primary_theme)` · `attempts (user, puzzle, mode, selector, solved, time_ms, hints, rating before/after, xp)` · `saved_puzzles` · `user_badges`

### Run it locally

Prereqs: Python 3.11+, Node 20+, PostgreSQL 15+.

```bash
# 0. Postgres
psql -U postgres -c "CREATE ROLE grandmaster LOGIN PASSWORD 'grandmaster';"                  -c "CREATE DATABASE grandmaster OWNER grandmaster;"

# 1. Python env (engine + backend)
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements-dev.txt

# 2. Puzzles: official Lichess dump (~310 MB) -> ~200k puzzles stratified over themes x ratings
mkdir -p data/raw && curl -o data/raw/lichess_db_puzzle.csv.zst https://database.lichess.org/lichess_db_puzzle.csv.zst
cd backend && python scripts/import_puzzles.py --src ../data/raw/lichess_db_puzzle.csv.zst --per-theme 40000

# 3. API  (http://localhost:8000/docs)
cp .env.example .env   # set JWT_SECRET
uvicorn app.main:app --reload

# 4. Web app (http://localhost:5173)
cd ../frontend && npm install && npm run dev
```

Tests: `cd backend && pytest` (needs a `grandmaster_test` database).

### Reproducing the engine's reported results

`python scripts/verify_results.py` re-runs the seeded evaluation from the saved checkpoints (after `scripts/build_dataset.py`). On the November 2026 Lichess dump, PPO and SAC match `FINAL_REPORT.md` exactly; baseline agents differ slightly because their internal RNG state depended on run order in the original GUI session. The report's table corresponds to **20** episodes (the Streamlit default), not 50.

---

# RL engine (original project)

> **RL-Based Adaptive Chess Puzzle Training System**
> DSAI Project — IIIT Bangalore

## Overview

Grandmaster-RL trains a reinforcement learning agent to construct personalized chess puzzle curricula for simulated learners. Rather than the standard "rating-match" heuristic used by platforms like Lichess and Chess.com — where a player simply receives the next puzzle within ±Δ of their current rating — this system models the learner as a Markov Decision Process and optimizes long-term skill gain using PPO, with SAC included as a comparison method.

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
├── gui/
│   ├── app.py              # Streamlit Interactive Dashboard
├── agents/
│   └── baselines.py        # Random, RatingMatch, FixedProgression
├── networks/
│   └── hybrid_policy.py    # Custom PPO ActorCriticPolicy (hybrid head)
├── training/
│   ├── train_ppo.py
│   ├── train_sac.py
│   └── configs/            # YAML hyperparameter configurations
├── evaluation/
│   ├── evaluate.py         # LEI, robustness, and multi-agent comparison
│   └── visualize.py        # All 6 publication plots
├── scripts/
│   └── build_dataset.py    # Phase 1: filter CSV + build KD-tree indices
├── tests/
│   └── test_env.py         # Environment and baseline sanity tests
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

SAC uses Stable-Baselines3's standard continuous `MlpPolicy`. The environment exposes the theme as a continuous float and rounds it to the nearest theme at step time, so SAC is a useful continuous-control comparison but PPO is the recommended model for the hybrid action structure.

```bash
python training/train_sac.py \
    --config training/configs/sac.yaml \
    --timesteps 500000 \
    --seed 42 \
    --output_dir checkpoints/sac
```

Checkpoints are saved every 10% of total timesteps. PPO also saves a callback best model in `checkpoints/ppo/best_model/`. TensorBoard logs go to `checkpoints/{algo}/tensorboard/`.

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

```

The Streamlit dashboard can run the same multi-agent evaluation and generate all six publication plots under `results/plots/`.

## Running Tests

```bash
# Requires data/indices/ to be built first
pytest tests/test_env.py -v
```

## GUI
```bash
# Launches the Streamlit dashboard
streamlit run gui/app.py
```

## Final Results

| Agent | LEI Eval | Robustness | Mean Δρ Eval | Success Eval |
|---|---:|---:|---:|---:|
| PPO | 0.0016 | 0.6674 | 4.63 | 0.697 |
| SAC | 0.0005 | 0.4077 | 11.34 | 0.133 |
| Random | 0.0000 | 0.6413 | 2.57 | 0.233 |
| RatingMatch | 0.0012 | 0.6667 | 6.48 | 0.419 |
| FixedProgression | 0.0013 | 0.7149 | 5.57 | 0.536 |

PPO is the recommended final policy because it achieves the strongest evaluation LEI while maintaining a high held-out success rate.

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
