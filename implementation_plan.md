# RL-Based Adaptive Chess Puzzle Training System
## Implementation Plan

---

## Suggested Repository Names

| Name | Vibe |
|---|---|
| `chesscurriculum-rl` | Descriptive, searchable |
| `tactiq` | Tactical + IQ, clean product feel |
| `puzzleforge` | Crafting a path through puzzles |
| `chessmind-rl` | Personalized intelligence angle |
| `adaptive-tactician` | Precise to what the agent does |
| `rook-and-learn` | Chess pun + learning, memorable |

**Recommended:** `tactiq` or `chesscurriculum-rl`

---

## Project Structure

```
tactiq/
├── data/
│   ├── raw/                    # Downloaded Lichess puzzle CSV
│   ├── processed/              # Filtered, theme-mapped puzzles
│   └── indices/                # Serialized KD-trees (one per theme)
├── env/
│   ├── chess_env.py            # Gymnasium MDP environment
│   ├── learner_model.py        # Logistic success model + skill update
│   └── eval_env.py             # Held-out evaluation environment (fatigue)
├── agents/
│   ├── ppo_agent.py            # PPO with hybrid action head
│   ├── sac_agent.py            # SAC with hybrid action head
│   └── baselines.py            # Random, RatingMatch, FixedProgression
├── training/
│   ├── train_ppo.py
│   ├── train_sac.py
│   └── configs/                # YAML hyperparameter configs
├── evaluation/
│   ├── evaluate.py             # LEI + diagnostic metrics
│   └── visualize.py            # Plots: skill curves, difficulty progression
├── notebooks/
│   └── exploratory.ipynb       # EDA on puzzle dataset
├── tests/
│   └── test_env.py
├── requirements.txt
└── README.md
```

---

## Phase 1 — Data Pipeline
**Goal:** Clean, index, and serve the Lichess puzzle dataset efficiently.

### 1.1 Download & Filter
- Download the Lichess Open Puzzle Database CSV (~3.5M rows).
- Keep only columns: `PuzzleId`, `FEN`, `Moves`, `Rating`, `RatingDeviation`, `Themes`.
- Drop puzzles with `RatingDeviation > 150` (unreliable ratings).
- Drop extreme outliers: keep ratings in `[400, 3000]`.

### 1.2 Theme Assignment
Define the 6 primary themes:
```python
THEMES = ["fork", "pin", "mate", "endgame", "skewer", "discovery"]
```
- Parse the space-separated `Themes` field.
- Assign each puzzle its **first matching** primary theme; discard if none match.
- Result: 6 filtered DataFrames, one per theme.

### 1.3 KD-Tree Indexing
For each theme `θ`:
```python
from scipy.spatial import KDTree
import numpy as np, pickle

ratings = df_theme["Rating"].values.reshape(-1, 1)
tree = KDTree(ratings)
pickle.dump((tree, df_theme), open(f"data/indices/{θ}.pkl", "wb"))
```
- At query time: `tree.query([[δ*]])` returns the nearest real puzzle.
- This preserves full rating granularity — no binning.

**Deliverable:** 6 `.pkl` files, each containing a `(KDTree, DataFrame)` pair.

---

## Phase 2 — MDP Environment
**Goal:** Implement a Gymnasium-compliant environment that faithfully simulates learner dynamics.

### 2.1 State Space
```python
# State vector: shape = (|Θ| + 3,) = (9,)
# [overall_rating, skill_fork, skill_pin, ..., rolling_success_rate, mean_difficulty]
# All normalized to [0, 1]
```

### 2.2 Learner Model (`learner_model.py`)
Implement the IRT logistic success probability:
```python
def solve_prob(skill_theme, puzzle_rating, tau=200.0):
    return 1 / (1 + np.exp(-(skill_theme - puzzle_rating) / tau))
```
Implement the piecewise skill update:
```python
def update_skill(phi, delta, solved, alpha=0.1, lam=0.5, mu=5.0):
    if solved:
        gain = lam * (delta - phi) / phi  # negative if too easy
    else:
        gain = -mu
    return phi + alpha * gain
```

### 2.3 Environment (`chess_env.py`)
```python
class ChessPuzzleEnv(gymnasium.Env):
    def reset(self):
        # Sample initial skills from N(µ₀, σ₀²I), e.g. µ₀=1000, σ₀=150
        ...
    def step(self, action):
        # action = [theme_idx (int), target_rating (float)]
        # 1. Query KD-tree → retrieve actual puzzle rating δ
        # 2. Sample Bernoulli(solve_prob(...))
        # 3. Update skill + rolling stats
        # 4. Compute reward: r = ρ_{t+1} - ρ_t
        # 5. Increment step counter; done = (t == T=100)
        ...
```

### 2.4 Evaluation Environment (`eval_env.py`)
Subclass `ChessPuzzleEnv` and override `step()` to inject:
- **Fatigue:** multiply effective skill by `f(t) = 1 - β*(t/T)`, e.g. `β=0.15`
- **Temperature jitter:** `τ' = τ + N(0, σ_τ²)` per episode

This environment is **never seen during training** — only used for robustness scoring.

**Deliverable:** `env/chess_env.py` and `env/eval_env.py`, passing `pytest` unit tests.

---

## Phase 3 — Policy Network Architecture
**Goal:** Design a shared network that handles the hybrid (discrete theme + continuous difficulty) action space.

### Shared Architecture
```
Input: state (dim=9, normalized)
  └─► Linear(9 → 128) → ReLU
      └─► Linear(128 → 128) → ReLU
            ├─► [Actor] Softmax head → P(theme) ∈ ℝ^6
            ├─► [Actor] Gaussian head → (µ_δ, log σ_δ) ∈ ℝ²
            └─► [Critic] Linear(128 → 1) → V(s) or Q(s,a)
```

- Theme is sampled categorically: `θ* ~ Categorical(softmax(logits))`
- Difficulty is sampled as: `δ* ~ N(µ_δ, σ_δ²)`, then clipped to [400, 3000]
- During evaluation, use the mode (argmax / µ_δ) for deterministic rollouts

---

## Phase 4 — RL Algorithm Implementation
**Goal:** Train PPO and SAC agents using Stable-Baselines3 with custom wrappers.

### 4.1 PPO (`train_ppo.py`)
Use SB3's `PPO` with a custom policy class that registers the hybrid head.

Key hyperparameters to tune:
```yaml
# configs/ppo.yaml
n_steps: 2048
batch_size: 256
n_epochs: 10
learning_rate: 3e-4
gamma: 0.99
clip_range: 0.2
ent_coef: 0.01          # encourage theme exploration
gae_lambda: 0.95
```

### 4.2 SAC (`train_sac.py`)
SAC handles continuous actions natively; discrete theme requires a Gumbel-Softmax reparameterization trick or a separate categorical head with log-prob correction.

Key hyperparameters:
```yaml
# configs/sac.yaml
learning_rate: 3e-4
buffer_size: 100000
batch_size: 256
gamma: 0.99
tau: 0.005              # soft target update
ent_coef: "auto"        # automatic entropy tuning
train_freq: 1
gradient_steps: 1
```

### 4.3 Training Loop
```python
model = PPO("MlpPolicy", env, **config, verbose=1,
            tensorboard_log="./runs/ppo/")
model.learn(total_timesteps=500_000,
            callback=EvalCallback(eval_env, eval_freq=5000))
model.save("checkpoints/ppo_final")
```
- Use `EvalCallback` on the **training** env for early stopping / checkpointing.
- Separately run the held-out eval env post-training for robustness scores.

---

## Phase 5 — Baselines
**Goal:** Implement the three baseline agents in `agents/baselines.py`.

```python
class RandomAgent:
    def predict(self, state):
        return (np.random.randint(6), np.random.uniform(400, 3000))

class RatingMatchAgent:
    def predict(self, state):
        rho = state[0] * 2600 + 400   # denormalize overall rating
        delta = rho + np.random.uniform(-100, 100)
        theme = np.random.randint(6)
        return (theme, np.clip(delta, 400, 3000))

class FixedProgressionAgent:
    def __init__(self, delta_step=5, step_every=5):
        self.t = 0; self.delta = 800
    def predict(self, state):
        if self.t % self.step_every == 0:
            self.delta += self.delta_step
        self.t += 1
        return (np.random.randint(6), self.delta)
```

---

## Phase 6 — Evaluation & Metrics
**Goal:** Compute LEI and all diagnostic metrics; produce publication-quality plots.

### 6.1 Learning Efficiency Index
```python
def compute_lei(rho_0, rho_T, success_history, difficulty_history, sigma2_ref=40000):
    delta_rho = rho_T - rho_0
    p_bar = np.mean(success_history)
    var_delta = np.var(difficulty_history)
    lei = (delta_rho / 100) * p_bar * (1 / (1 + var_delta / sigma2_ref))
    return lei
```

### 6.2 Robustness Score
```
robustness = LEI_eval_env / LEI_train_env
```
Values close to 1.0 → robust; values < 0.5 → likely overfitting to simulator.

### 6.3 Plots to Generate (`visualize.py`)
- **Skill improvement bar chart** — ∆ρ across all 5 methods
- **LEI comparison bar chart** — primary result figure
- **Per-theme skill gain heatmap** — rows = agents, columns = themes
- **Success rate trajectory** — p̄_t over 100 steps per agent
- **Difficulty progression curve** — δ*_t over 100 steps, smoothed
- **Robustness score comparison** — robustness bar chart (train vs eval LEI)

---

## Phase 7 — Ablation Study
**Goal:** Validate that continuous actions outperform discrete binning.

Create a `DiscreteEnv` variant that maps difficulty to 3 bins:
- Easy: [400, 1200), Medium: [1200, 2000), Hard: [2000, 3000]

Train a PPO agent on `DiscreteEnv` and compare LEI against full continuous PPO.
This directly tests the paper's Hypothesis 5.

---

## Implementation Order & Timeline

```
Week 1   ▸ Phase 1 (Data Pipeline) + Phase 2 (Environment)
Week 2   ▸ Phase 3 (Network) + Phase 4 (PPO training, get first numbers)
Week 3   ▸ Phase 4 (SAC training) + Phase 5 (Baselines)
Week 4   ▸ Phase 6 (Evaluation + all plots) + Phase 7 (Ablation)
Buffer   ▸ Hyperparameter tuning, report writing
```

---

## Dependencies

```
# requirements.txt
gymnasium>=0.29
stable-baselines3>=2.3
torch>=2.2
scipy
numpy
pandas
matplotlib
seaborn
tensorboard
pyyaml
pytest
```

---

## Key Implementation Gotchas

**KD-tree query is 1D** — ratings are scalars, so reshape to `(-1, 1)` before building the tree. Querying with `[[δ*]]` returns `(distance, index)`; use `index` to look up the row in the DataFrame.

**Hybrid action space in SB3** — SB3 does not natively support mixed discrete+continuous spaces. The cleanest approach is to use a `Box` action space for both outputs (theme as a float [0, 5], rounded to int at step time), and define a custom `ActorCriticPolicy` subclass.

**Reward scale matters** — since skill updates use ratings in [400, 3000], raw ∆ρ rewards can be large. Normalize rewards or use a small `alpha` (e.g. 0.05) to keep per-step rewards in roughly [-1, 1] for stable training.

**Reproducibility** — seed both `numpy`, `torch`, and the Gymnasium env in every training run. Log seeds in the config YAML.

**Evaluation env separation** — never pass `eval_env` to `EvalCallback` during training. Run a completely separate evaluation script post-training to compute robustness scores.

---

*This plan corresponds to the system described in "Reinforcement Learning for Adaptive Chess Puzzle Training Curriculum" (DSAI, IIIT Bangalore).*
