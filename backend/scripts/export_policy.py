"""Export the trained PPO actor to a dependency-free .npz (no torch at serve time).

    python scripts/export_policy.py            # writes app/assets/ppo_policy.npz and checks parity

Deterministic HybridPolicy action, reproduced exactly:
    h  = relu(W1 x + b1); h = relu(W2 h + b2)            # SharedMLP trunk (shared extractor)
    z  = tanh(P1 h + c1); z = tanh(P2 z + c2)            # SB3 mlp_extractor.policy_net
    theme = argmax(T z + t)                              # _theme_head -> Categorical mode
    diff  = clip(sigmoid((D z + d)[0]), 0, 1)            # _diff_head  -> Normal mean
"""

import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path[:0] = [str(BACKEND), str(REPO)]

OUT = BACKEND / "app" / "assets" / "ppo_policy.npz"
KEYS = {
    "W1": "features_extractor.net.0", "W2": "features_extractor.net.2",
    "P1": "mlp_extractor.policy_net.0", "P2": "mlp_extractor.policy_net.2",
    "T": "_theme_head", "D": "_diff_head",
}


def main() -> None:
    from stable_baselines3 import PPO

    from app.services.policy import NumpyPolicy

    model = PPO.load(str(REPO / "checkpoints" / "ppo" / "ppo_final"), device="cpu")
    assert model.policy.share_features_extractor
    sd = {k: v.cpu().numpy().astype(np.float64) for k, v in model.policy.state_dict().items()}
    arrays = {}
    for name, prefix in KEYS.items():
        arrays[name] = sd[f"{prefix}.weight"]
        arrays[name.lower() + "_b"] = sd[f"{prefix}.bias"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **arrays)

    pol = NumpyPolicy(OUT)
    rng = np.random.default_rng(0)
    obs = rng.uniform(0, 1, size=(20_000, 9)).astype(np.float32)
    ref, _ = model.predict(obs, deterministic=True)
    ours = np.stack([pol.act(o) for o in obs])
    theme_match = float(np.mean(ref[:, 0] == ours[:, 0]))
    diff_err = float(np.max(np.abs(ref[:, 1] - ours[:, 1])))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"parity on 20k random obs: theme match {theme_match:.4%}, max |diff error| {diff_err:.2e}")
    assert theme_match == 1.0 and diff_err < 1e-5


if __name__ == "__main__":
    main()
