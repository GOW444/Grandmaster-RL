"""The app's coach must feed PPO exactly what the engine's env fed it during training."""

import sys
from collections import deque

import numpy as np

from app.config import REPO_ROOT
from app.services.coach import CoachState, build_observation
from app.services.skill import expected_score

sys.path.insert(0, str(REPO_ROOT))
from env.chess_env import ChessPuzzleEnv  # noqa: E402
from env.learner_model import THEMES, solve_prob  # noqa: E402


def test_observation_matches_training_env():
    rng = np.random.default_rng(0)
    skills = rng.uniform(600, 2200, size=6)
    solved = [bool(x) for x in rng.integers(0, 2, size=14)]
    diffs = [int(x) for x in rng.integers(500, 2500, size=14)]

    env = ChessPuzzleEnv.__new__(ChessPuzzleEnv)  # skip KD-tree loading; only _build_state is exercised
    env.mu0, env.window = 1000.0, 10
    env._skills = skills.copy()
    env._overall = float(np.mean(skills))
    env._success_buf = deque([float(s) for s in solved], maxlen=10)
    env._diff_buf = deque([float(d) for d in diffs], maxlen=10)

    ours = build_observation(CoachState(dict(zip(THEMES, skills)), solved, diffs))
    np.testing.assert_allclose(ours, env._build_state(), atol=1e-6)


def test_empty_history_matches_env_reset_state():
    env = ChessPuzzleEnv.__new__(ChessPuzzleEnv)
    env.mu0, env.window = 1000.0, 10
    env._skills = np.full(6, 1000.0)
    env._overall = 1000.0
    env._success_buf, env._diff_buf = deque(maxlen=10), deque(maxlen=10)
    ours = build_observation(CoachState({t: 1000.0 for t in THEMES}, [], []))
    np.testing.assert_allclose(ours, env._build_state(), atol=1e-6)


def test_skill_model_uses_engine_irt_curve():
    for phi, delta in [(1000, 1000), (1200, 900), (800, 1700)]:
        assert abs(expected_score(phi, delta) - solve_prob(phi, delta)) < 1e-12


def test_numpy_policy_matches_sb3_checkpoint():
    from stable_baselines3 import PPO

    from app.config import get_settings
    from app.services.policy import NumpyPolicy

    s = get_settings()
    model = PPO.load(str(s.ppo_checkpoint), device="cpu")
    pol = NumpyPolicy(s.policy_npz)
    obs = np.random.default_rng(1).uniform(0, 1, size=(2000, 9)).astype(np.float32)
    ref, _ = model.predict(obs, deterministic=True)
    ours = np.stack([pol.act(o) for o in obs])
    assert np.array_equal(ref[:, 0], ours[:, 0])
    assert np.max(np.abs(ref[:, 1] - ours[:, 1])) < 1e-5
