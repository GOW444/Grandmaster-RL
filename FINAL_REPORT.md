# Grandmaster-RL Final Report

## 1. Project Overview

Grandmaster-RL is an adaptive chess puzzle training system that uses reinforcement learning to select personalized puzzle curricula for simulated learners. At each step, the agent chooses:

- a tactical theme: `fork`, `pin`, `mate`, `endgame`, `skewer`, or `discovery`
- a target puzzle difficulty in the rating range `[400, 3000]`

The environment then retrieves the nearest real puzzle from prebuilt per-theme KD-tree indices and simulates whether the learner solves it using an Item Response Theory style logistic model. The learner's per-theme skill estimates are updated after every puzzle attempt.

The main goal is to maximize learning efficiency, measured by rating gain, success rate, and stability of selected puzzle difficulty.

## 2. Dataset and Preprocessing

The system uses a reduced Lichess puzzle dataset. The preprocessing pipeline:

1. Filters puzzles with `RatingDeviation <= 150`.
2. Keeps puzzles with ratings in `[400, 3000]`.
3. Assigns each puzzle to one primary tactical theme.
4. Saves one processed CSV per theme.
5. Builds a 1D KD-tree per theme using puzzle rating as the search coordinate.

One important correction was made during implementation: Lichess does not use the literal tag `discovery`. Instead, discovery tactics appear as `discoveredAttack` and `discoveredCheck`. The preprocessing script now maps those tags into the internal `discovery` theme.

Final theme counts after rebuilding:

| Theme | Puzzle Count |
|---|---:|
| fork | 743,786 |
| pin | 325,478 |
| mate | 1,685,685 |
| endgame | 1,576,401 |
| skewer | 28,223 |
| discovery | 162,344 |

## 3. Environment Design

The training environment is implemented as a Gymnasium environment.

Observation vector:

```text
[overall_rating, six_theme_skills, rolling_success_rate, rolling_mean_difficulty]
```

All observation values are normalized to `[0, 1]`.

Action vector:

```text
[theme_float, normalized_target_difficulty]
```

The theme component is represented as a continuous float for Stable-Baselines3 compatibility and rounded inside the environment. The difficulty component is denormalized to a rating in `[400, 3000]`.

The evaluation environment adds unseen learner variation through fatigue and IRT temperature jitter. This tests whether a trained policy is robust beyond the exact dynamics seen during training.

## 4. Algorithms

### PPO

PPO is the primary algorithm. It uses a custom `HybridPolicy` with:

- categorical theme selection
- Gaussian difficulty selection
- shared MLP feature trunk
- value head for actor-critic training

PPO is the best match for the mixed discrete-continuous action structure.

### SAC

SAC is included as a comparison model. Since Stable-Baselines3 SAC expects a fully continuous policy, the SAC trainer uses `MlpPolicy`. The action space is still compatible because the environment exposes both action dimensions as a continuous `Box`; the theme float is rounded at step time.

SAC is therefore less semantically exact than PPO, but it is useful as an aggressive continuous-control baseline.

## 5. Baselines

Three non-RL baselines were evaluated:

- `RandomAgent`: randomly selects theme and difficulty.
- `RatingMatchAgent`: selects puzzle difficulty near the learner's current rating.
- `FixedProgressionAgent`: starts at an easier rating and increases difficulty on a fixed schedule.

These baselines are important because a practical RL policy should beat or meaningfully differ from simple curriculum heuristics.

## 6. Metrics

The main metric is Learning Efficiency Index:

```text
LEI = (rating_gain / episode_length) * success_rate * difficulty_consistency
```

Additional metrics:

- mean rating change, `Mean Delta rho`
- mean success rate
- robustness score, computed as eval LEI divided by train LEI

Higher LEI is better. Higher robustness means the policy transfers better to the held-out evaluation dynamics.

## 7. Final Results

| Agent | LEI Train | LEI Eval | Robustness | Mean Delta rho Train | Mean Delta rho Eval | Success Train | Success Eval |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO | 0.0024 | 0.0016 | 0.6674 | 6.38 | 4.63 | 0.754 | 0.697 |
| SAC | 0.0012 | 0.0005 | 0.4077 | 16.65 | 11.34 | 0.190 | 0.133 |
| Random | 0.0000 | 0.0000 | 0.6413 | 3.94 | 2.57 | 0.234 | 0.233 |
| RatingMatch | 0.0018 | 0.0012 | 0.6667 | 8.56 | 6.48 | 0.485 | 0.419 |
| FixedProgression | 0.0019 | 0.0013 | 0.7149 | 7.19 | 5.57 | 0.582 | 0.536 |

## 8. Interpretation

PPO is the strongest overall policy by the main metric. It achieves the highest train and evaluation LEI while maintaining a high success rate. Its evaluation success rate of `69.7%` suggests that it selects puzzles that are challenging but still solvable for the simulated learner.

SAC produces the largest raw rating gains, especially on the training environment. However, its success rate is much lower: `19.0%` on train and `13.3%` on eval. This suggests that SAC learns a more aggressive curriculum, selecting harder puzzles that produce large gains when solved but many failures overall. Its lower robustness score also indicates weaker transfer to the held-out evaluation dynamics.

The RatingMatch and FixedProgression baselines are competitive, which is expected because matching difficulty to learner ability is already a strong tutoring heuristic. PPO still beats both baselines on evaluation LEI, which supports the value of RL-based curriculum optimization.

FixedProgression has the highest robustness score among the compared agents, but its evaluation LEI is lower than PPO's. This means it transfers consistently, but does not optimize learning efficiency as well as PPO.

## 9. Conclusion

The project successfully implements an RL-based adaptive chess puzzle curriculum system using real puzzle metadata and a simulated learner model.

The best final model is PPO. It provides the strongest balance between:

- learning efficiency
- rating improvement
- high solve rate
- robustness to unseen learner dynamics

SAC is useful as a comparison model and demonstrates a different high-risk, high-reward strategy, but PPO is the recommended policy for the final system.

## 10. Key Implementation Fixes

Several issues were found and fixed during development:

- Corrected KD-tree query index extraction for SciPy return shapes.
- Fixed empty `discovery.pkl` by mapping Lichess `discoveredAttack` and `discoveredCheck` tags.
- Added repo-root path bootstrapping so training scripts work when run as files.
- Disabled Stable-Baselines3 progress bars automatically when optional `rich` dependency is missing.
- Wrapped PPO callback evaluation env with `VecNormalize` to match the training env.
- Switched SAC from incompatible `HybridPolicy` to SB3's continuous `MlpPolicy`.
- Rebalanced the learner skill update so successful solves produce meaningful positive learning signal.

## 11. Recommended Next Steps

1. Use PPO as the primary reported model.
2. Include SAC as an aggressive comparison model.
3. Include RatingMatch and FixedProgression baselines in the discussion, since they are strong practical heuristics.
4. Generate plots for final presentation:
   - LEI comparison bar chart
   - rating gain comparison
   - success rate comparison
   - robustness comparison
5. In future work, replace SAC's rounded continuous theme action with a true hybrid or Gumbel-Softmax policy.

