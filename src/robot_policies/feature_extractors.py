"""Custom feature extractors for Stable Baselines 3."""

from typing import Dict, List

import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.spaces import utils as gym_utils
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CameraPoseExtractor(BaseFeaturesExtractor):
    """Feature extractor for camera_pose only (no reconstruction render)."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int,
        hidden_dims: List[int] = [64, 64],
    ):
        super().__init__(observation_space, features_dim)
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


class CameraPoseMeshRenderingExtractor(BaseFeaturesExtractor):
    """
    Feature extractor combining camera_pose + reconstruction render.
    Uses simple CNN for the depth render, MLP for pose, then concatenates.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int,
    ):
        super().__init__(observation_space, features_dim)

        # CNN for reconstruction render (1, 64, 64) -> features
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> 16x32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 32x16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 32x8x8
            nn.ReLU(),
            nn.Flatten(),  # -> 32*8*8 = 2048
            nn.Linear(4096, 128),
            nn.ReLU(),
        )

        # MLP for camera pose (7) -> features
        self.pose_net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
        )

        # Combine: 64 + 32 -> features_dim
        self.combine = nn.Sequential(
            nn.Linear(128 + 32, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        render = observations["reconstruction_render"]
        # Downsample to 64x64 if needed
        if render.shape[-2:] != (64, 64):
            render = nn.functional.interpolate(
                render, size=(64, 64), mode="bilinear", align_corners=False
            )
            # print(
            #     f"Interpolated render from {observations['reconstruction_render'].shape[-2:]} to (64, 64)"
            # )
        render_features = self.cnn(render)

        pose_features = self.pose_net(observations["camera_pose"])
        combined = torch.cat([render_features, pose_features], dim=1)
        return self.combine(combined)
