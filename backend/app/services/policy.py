"""Pure-NumPy forward pass of the trained PPO actor (see scripts/export_policy.py).

Lets the API serve the RL coach without torch / stable-baselines3, which keeps the
deploy image small enough for free-tier hosts (512 MB RAM)."""

from pathlib import Path

import numpy as np


class NumpyPolicy:
    def __init__(self, path: Path) -> None:
        with np.load(path) as f:
            self.p = {k: f[k] for k in f.files}

    def act(self, obs: np.ndarray) -> np.ndarray:
        p = self.p
        x = np.asarray(obs, dtype=np.float64).ravel()
        h = np.maximum(p["W1"] @ x + p["w1_b"], 0.0)
        h = np.maximum(p["W2"] @ h + p["w2_b"], 0.0)
        z = np.tanh(p["P1"] @ h + p["p1_b"])
        z = np.tanh(p["P2"] @ z + p["p2_b"])
        theme = float(np.argmax(p["T"] @ z + p["t_b"]))
        mu = (p["D"] @ z + p["d_b"])[0]
        diff = float(np.clip(1.0 / (1.0 + np.exp(-mu)), 0.0, 1.0))
        return np.array([theme, diff], dtype=np.float32)
