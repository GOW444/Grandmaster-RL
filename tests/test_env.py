"""
Environment Tests
=================
Pytest unit tests covering the core environment, learner model, eval env,
and all three baseline agents.

Run with:
    pytest tests/test_env.py -v
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def indices_dir() -> str:
    """Return the path to built KD-tree indices."""
    return "data/indices"


@pytest.fixture(scope="module")
def train_env(indices_dir):
    """Create a shared training environment for all tests."""
    from env.chess_env import ChessPuzzleEnv
    env = ChessPuzzleEnv(indices_dir=indices_dir, seed=0)
    yield env
    env.close()


@pytest.fixture(scope="module")
def eval_env(indices_dir):
    """Create a shared evaluation environment for all tests."""
    from env.eval_env import EvalChessPuzzleEnv
    env = EvalChessPuzzleEnv(indices_dir=indices_dir, seed=99)
    yield env
    env.close()


# ---------------------------------------------------------------------------
# Test 1 — reset returns correct observation
# ---------------------------------------------------------------------------

def test_env_reset(train_env):
    """State shape must be (9,) with all values in [0, 1]."""
    obs, info = train_env.reset(seed=42)
    assert isinstance(obs, np.ndarray), "Observation must be a numpy array."
    assert obs.shape == (9,), f"Expected shape (9,), got {obs.shape}."
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}."
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), (
        f"Observation values out of [0, 1]: min={obs.min():.4f}, max={obs.max():.4f}"
    )
    assert isinstance(info, dict), "Info must be a dict."


# ---------------------------------------------------------------------------
# Test 2 — random steps don't crash; rewards are finite
# ---------------------------------------------------------------------------

def test_env_step_random(train_env):
    """100 random steps must not raise and must return finite rewards."""
    obs, _ = train_env.reset(seed=0)
    rng = np.random.default_rng(0)

    for step_i in range(100):
        action = rng.uniform(
            train_env.action_space.low, train_env.action_space.high
        ).astype(np.float32)
        obs, reward, terminated, truncated, info = train_env.step(action)

        assert obs.shape == (9,), f"Step {step_i}: bad obs shape."
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), (
            f"Step {step_i}: obs out of [0, 1]."
        )
        assert np.isfinite(reward), f"Step {step_i}: reward is not finite ({reward})."
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "solved" in info

        if terminated or truncated:
            obs, _ = train_env.reset(seed=step_i)


# ---------------------------------------------------------------------------
# Test 3 — full episode terminates exactly at T=100
# ---------------------------------------------------------------------------

def test_env_episode(indices_dir):
    """A fresh episode should reach done=True at exactly step T=100."""
    from env.chess_env import ChessPuzzleEnv
    T = 100
    env = ChessPuzzleEnv(indices_dir=indices_dir, T=T, seed=7)
    obs, _ = env.reset(seed=7)
    rng = np.random.default_rng(7)

    terminated = False
    step_count = 0
    while not terminated:
        action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        assert step_count <= T, f"Episode exceeded T={T} without terminating."

    assert step_count == T, (
        f"Episode terminated at step {step_count}, expected {T}."
    )
    env.close()


# ---------------------------------------------------------------------------
# Test 4 — learner model functions
# ---------------------------------------------------------------------------

def test_learner_model():
    """solve_prob must be in (0,1); update_skill must clamp to [400, 3000]."""
    from env.learner_model import RATING_MAX, RATING_MIN, solve_prob, update_skill

    # solve_prob: harder puzzle → lower probability
    p_easy = solve_prob(skill_theme=1500.0, puzzle_rating=800.0)
    p_hard = solve_prob(skill_theme=1500.0, puzzle_rating=2500.0)
    assert 0.0 < p_easy < 1.0, f"p_easy={p_easy} not in (0,1)."
    assert 0.0 < p_hard < 1.0, f"p_hard={p_hard} not in (0,1)."
    assert p_easy > p_hard, "Easier puzzle should have higher solve probability."

    # Equal skill and rating → ~50%
    p_equal = solve_prob(skill_theme=1500.0, puzzle_rating=1500.0)
    assert abs(p_equal - 0.5) < 0.01, f"Equal skill/rating should give ~0.5, got {p_equal}."

    # update_skill: result must be in [RATING_MIN, RATING_MAX]
    phi_new_success = update_skill(phi=1500.0, delta=2000.0, solved=True)
    assert RATING_MIN <= phi_new_success <= RATING_MAX

    phi_new_fail = update_skill(phi=1500.0, delta=2000.0, solved=False)
    assert RATING_MIN <= phi_new_fail <= RATING_MAX

    # Success with hard puzzle → slight gain
    assert phi_new_success > 1500.0, "Solving a harder puzzle should increase skill."
    # Failure → loss
    assert phi_new_fail < 1500.0, "Failing a puzzle should decrease skill."

    # Clamp test: extremely high delta should not push phi above RATING_MAX
    phi_extreme = update_skill(phi=2999.0, delta=3000.0, solved=True, alpha=1000.0)
    assert phi_extreme <= RATING_MAX, "update_skill should clamp to RATING_MAX."


# ---------------------------------------------------------------------------
# Test 5 — EvalChessPuzzleEnv runs a full episode without error
# ---------------------------------------------------------------------------

def test_eval_env(eval_env):
    """The evaluation env must complete a full episode without crashing."""
    obs, _ = eval_env.reset(seed=42)
    assert obs.shape == (9,)
    rng = np.random.default_rng(42)

    terminated = False
    step_count = 0
    while not terminated:
        action = rng.uniform(eval_env.action_space.low, eval_env.action_space.high).astype(np.float32)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        step_count += 1
        assert np.isfinite(reward), f"Eval env step {step_count}: non-finite reward."
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    assert step_count == eval_env.T, (
        f"Eval env terminated at step {step_count}, expected {eval_env.T}."
    )


# ---------------------------------------------------------------------------
# Test 6 — All 3 baselines produce valid actions
# ---------------------------------------------------------------------------

def test_baselines(train_env):
    """All baseline agents must return actions of shape (2,) with values in [0, 1]."""
    from agents.baselines import FixedProgressionAgent, RandomAgent, RatingMatchAgent
    from env.learner_model import THEMES

    obs, _ = train_env.reset(seed=0)
    n_themes = float(len(THEMES))

    agents = [
        RandomAgent(seed=0),
        RatingMatchAgent(seed=0),
        FixedProgressionAgent(seed=0),
    ]

    for agent in agents:
        name = type(agent).__name__
        for step in range(5):
            action = agent.predict(obs)
            assert isinstance(action, np.ndarray), f"{name}: action must be ndarray."
            assert action.shape == (2,), f"{name}: expected shape (2,), got {action.shape}."
            # Theme float in [0, n_themes)
            assert 0.0 <= action[0] < n_themes + 1e-6, (
                f"{name}: action[0]={action[0]} out of [0, {n_themes})."
            )
            # Difficulty in [0, 1]
            assert 0.0 <= action[1] <= 1.0, (
                f"{name}: action[1]={action[1]} out of [0, 1]."
            )
