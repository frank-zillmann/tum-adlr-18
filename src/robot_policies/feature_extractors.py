"""Custom feature extractors for Stable Baselines 3."""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.spaces import utils as gym_utils
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CameraPoseExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Dict observations with camera_pose key.
    Extensible to handle additional observation keys.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 64,
        hidden_dims: List[int] = [64, 64],
    ):
        super().__init__(observation_space, features_dim)

        # Get input dim from camera_pose space
        input_dim = gym_utils.flatdim(observation_space["camera_pose"])
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim

        layers.extend([nn.Linear(prev_dim, features_dim), nn.ReLU()])
        self.net = nn.Sequential(*layers)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(observations["camera_pose"])


class MeshExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for mesh/point cloud observations.

    Can be combined with CameraPoseExtractor for multi-modal observations.
    Uses a simple PointNet-like architecture.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 64,
        n_points: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        # Simple PointNet-style encoder
        self.point_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
        )
        self.n_points = n_points

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, n_points * 3) -> (batch, n_points, 3)
        batch_size = observations.shape[0]
        points = observations.view(batch_size, -1, 3)

        # Per-point features
        point_features = self.point_net(points)  # (batch, n_points, features_dim)

        # Global max pooling
        global_features, _ = torch.max(point_features, dim=1)
        return global_features
