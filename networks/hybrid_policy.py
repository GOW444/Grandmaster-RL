"""
Hybrid Policy Network for SB3
==============================
A custom ActorCriticPolicy that supports the project's hybrid action space:
  - Discrete theme selection via a Categorical head
  - Continuous difficulty selection via a Gaussian head

Action encoding convention (matches ChessPuzzleEnv):
  action[0]: theme float in [0, n_themes), rounded to int at environment step time
  action[1]: normalized difficulty in [0, 1]

Both the PPO and SAC trainers point to this policy class. SAC has limited
support for discrete action components; see the TODO in train_sac.py.
"""

import logging
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch.distributions import Categorical, Normal

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Number of themes — must stay in sync with env/learner_model.py
# ---------------------------------------------------------------------------
N_THEMES: int = 6
LOG_STD_MIN: float = -4.0
LOG_STD_MAX: float = 1.0


# ---------------------------------------------------------------------------
# Shared MLP trunk (feature extractor)
# ---------------------------------------------------------------------------

class SharedMLP(BaseFeaturesExtractor):
    """Two-layer shared MLP trunk used by both actor and critic.

    Architecture:
        Linear(obs_dim → 128) → ReLU → Linear(128 → 128) → ReLU

    Args:
        observation_space: The Gymnasium observation space.
        features_dim:      Output dimensionality (latent_dim = 128).
    """

    def __init__(self, observation_space, features_dim: int = 128) -> None:
        super().__init__(observation_space, features_dim)
        obs_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through the shared trunk.

        Args:
            observations: Batch of observations, shape ``(B, obs_dim)``.

        Returns:
            Latent representation, shape ``(B, 128)``.
        """
        return self.net(observations)


# ---------------------------------------------------------------------------
# Hybrid action distribution
# ---------------------------------------------------------------------------

class HybridDistribution(Distribution):
    """Joint distribution over (discrete theme, continuous difficulty).

    Theme is modelled as a :class:`~torch.distributions.Categorical` over
    ``N_THEMES`` classes.
    Difficulty is modelled as a clipped :class:`~torch.distributions.Normal`
    in [0, 1] (sigmoid-squashed mean, learned log-std).

    The ``log_prob`` and ``entropy`` are the *sums* of the two marginals,
    which is exact under the assumption that the two components are
    independent (they share the same trunk but have separate heads, so
    at the distribution level they are treated as independent).
    """

    def __init__(self) -> None:
        super().__init__()
        self._categorical: Optional[Categorical] = None
        self._normal: Optional[Normal] = None

    # ------------------------------------------------------------------
    # SB3 Distribution interface
    # ------------------------------------------------------------------

    def proba_distribution_net(
        self, latent_dim: int, log_std_init: float = 0.0
    ) -> Tuple[nn.Module, nn.Module]:
        """Create the actor head modules.

        Args:
            latent_dim:    Dimensionality of the shared trunk output.
            log_std_init:  Initial value for log_std (unused — learned jointly).

        Returns:
            Tuple ``(theme_head, difficulty_head)``.
        """
        theme_head = nn.Linear(latent_dim, N_THEMES)
        difficulty_head = nn.Linear(latent_dim, 2)  # [mu, log_sigma]
        return theme_head, difficulty_head

    def proba_distribution(
        self,
        theme_logits: torch.Tensor,
        diff_params: torch.Tensor,
    ) -> "HybridDistribution":
        """Instantiate the distributions from the actor head outputs.

        Args:
            theme_logits: Raw logits for Categorical, shape ``(B, N_THEMES)``.
            diff_params:  ``[mu, log_sigma]`` for Normal, shape ``(B, 2)``.

        Returns:
            Self, with ``_categorical`` and ``_normal`` set.
        """
        self._categorical = Categorical(logits=theme_logits)

        # Squash mu to (0, 1) with sigmoid; clamp log_std to safe range
        mu = torch.sigmoid(diff_params[:, 0])
        log_std = torch.clamp(diff_params[:, 1], LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        self._normal = Normal(mu, std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute the joint log probability of a batch of actions.

        Args:
            actions: Shape ``(B, 2)`` where ``actions[:, 0]`` are theme floats
                     and ``actions[:, 1]`` are difficulty values in [0, 1].

        Returns:
            Log-probability, shape ``(B,)``.
        """
        assert self._categorical is not None and self._normal is not None
        theme_idx = actions[:, 0].long()
        diff_val = actions[:, 1]

        log_p_theme = self._categorical.log_prob(theme_idx)
        log_p_diff = self._normal.log_prob(torch.clamp(diff_val, 1e-6, 1 - 1e-6))
        return log_p_theme + log_p_diff

    def entropy(self) -> torch.Tensor:
        """Compute the sum of theme and difficulty entropies.

        Returns:
            Entropy, shape ``(B,)``.
        """
        assert self._categorical is not None and self._normal is not None
        return self._categorical.entropy() + self._normal.entropy()

    def sample(self) -> torch.Tensor:
        """Draw a sample action.

        Returns:
            Action tensor, shape ``(B, 2)``, where column 0 is the theme
            float and column 1 is the clamped difficulty in [0, 1].
        """
        assert self._categorical is not None and self._normal is not None
        theme = self._categorical.sample().float()  # (B,)
        diff = torch.clamp(self._normal.sample(), 0.0, 1.0)  # (B,)
        return torch.stack([theme, diff], dim=-1)

    def mode(self) -> torch.Tensor:
        """Return the most probable (deterministic) action.

        Returns:
            Action tensor, shape ``(B, 2)``.
        """
        assert self._categorical is not None and self._normal is not None
        theme = self._categorical.probs.argmax(dim=-1).float()
        diff = torch.clamp(self._normal.mean, 0.0, 1.0)
        return torch.stack([theme, diff], dim=-1)

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """Sample or use mode depending on ``deterministic``.

        Args:
            deterministic: If ``True`` return mode, else sample.

        Returns:
            Action tensor, shape ``(B, 2)``.
        """
        if deterministic:
            return self.mode()
        return self.sample()

    def actions_from_params(
        self,
        theme_logits: torch.Tensor,
        diff_params: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Parameterise the distribution and return actions in one call.

        Args:
            theme_logits:  Shape ``(B, N_THEMES)``.
            diff_params:   Shape ``(B, 2)``.
            deterministic: Whether to use the mode.

        Returns:
            Actions, shape ``(B, 2)``.
        """
        self.proba_distribution(theme_logits, diff_params)
        return self.get_actions(deterministic)

    def log_prob_from_params(
        self,
        theme_logits: torch.Tensor,
        diff_params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parameterise the distribution and return (actions, log_probs).

        Args:
            theme_logits: Shape ``(B, N_THEMES)``.
            diff_params:  Shape ``(B, 2)``.

        Returns:
            Tuple ``(actions, log_probs)``, each shape ``(B, 2)`` and ``(B,)``.
        """
        actions = self.actions_from_params(theme_logits, diff_params)
        log_prob = self.log_prob(actions)
        return actions, log_prob


# ---------------------------------------------------------------------------
# Custom ActorCriticPolicy
# ---------------------------------------------------------------------------

class HybridPolicy(ActorCriticPolicy):
    """Custom SB3 ActorCriticPolicy with a hybrid discrete+continuous actor.

    The shared MLP trunk produces a 128-dim latent vector which is fed to:
      - A theme head: ``Linear(128 → N_THEMES)`` → Categorical distribution
      - A difficulty head: ``Linear(128 → 2)`` → Normal distribution
      - A value head: ``Linear(128 → 1)`` → V(s)

    This class hooks into SB3's ``PPO`` (and experimentally ``SAC``).

    Args:
        See :class:`stable_baselines3.common.policies.ActorCriticPolicy`.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Suppress SB3's default ortho-init and its own distribution building;
        # we'll build everything ourselves after super().__init__().
        kwargs["features_extractor_class"] = SharedMLP
        kwargs["features_extractor_kwargs"] = {"features_dim": 128}
        # Tell SB3 not to build its own action distribution
        kwargs.setdefault("normalize_images", False)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """Build actor and critic heads after the trunk is initialized.

        Overrides SB3's ``_build`` to insert custom actor heads.

        Args:
            lr_schedule: Learning rate schedule callable.
        """
        super()._build(lr_schedule)
        latent_dim = self.mlp_extractor.latent_dim_pi

        # Actor heads
        self._theme_head = nn.Linear(latent_dim, N_THEMES).to(self.device)
        self._diff_head = nn.Linear(latent_dim, 2).to(self.device)

        # Replace the action_net with our dual head (theme + diff)
        # We'll override _get_action_dist_from_latent instead.
        self._hybrid_dist = HybridDistribution()

        # Critic head is already built by super()._build()
        log.debug(
            "HybridPolicy built: latent_dim=%d, theme_head=%s, diff_head=%s",
            latent_dim,
            self._theme_head,
            self._diff_head,
        )

    def _get_action_dist_from_latent(
        self, latent_pi: torch.Tensor
    ) -> HybridDistribution:
        """Compute the action distribution from the actor's latent vector.

        Args:
            latent_pi: Output of the actor MLP trunk, shape ``(B, latent_dim)``.

        Returns:
            A :class:`HybridDistribution` parameterised by the actor heads.
        """
        theme_logits = self._theme_head(latent_pi)
        diff_params = self._diff_head(latent_pi)
        return self._hybrid_dist.proba_distribution(theme_logits, diff_params)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass used by the rollout collector.

        Args:
            obs:           Batch of observations, shape ``(B, obs_dim)``.
            deterministic: Whether to use the mode action.

        Returns:
            Tuple ``(actions, values, log_probs)``.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        dist = self._get_action_dist_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate log-prob and entropy for collected actions (used in PPO update).

        Args:
            obs:     Batch of observations.
            actions: Batch of collected actions, shape ``(B, 2)``.

        Returns:
            Tuple ``(values, log_prob, entropy)``.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        dist = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(latent_vf)
        return values, log_prob, entropy

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """Compute state-value estimates V(s) for a batch of observations.

        Args:
            obs: Batch of observations.

        Returns:
            Value estimates, shape ``(B, 1)``.
        """
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)
