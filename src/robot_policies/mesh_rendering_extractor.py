"""Mesh rendering feature extractor for Stable Baselines 3."""

from typing import Dict

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MeshRenderingExtractor(BaseFeaturesExtractor):
    """Feature extractor for reconstruction mesh renderings.

    Uses a CNN to process depth renderings of the current reconstruction.
    Expects input shape (batch, 1, H, W) and downsamples to 64x64 if needed.

    Args:
        observation_space: Gym observation space (must contain 'mesh_render')
        features_dim: Output dimension of the feature extractor
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
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 32x16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 64x8x8 = 4096
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        render = observations["mesh_render"]

        # Downsample to 64x64 if needed
        if render.shape[-2:] != (64, 64):
            render = nn.functional.interpolate(
                render, size=(64, 64), mode="bilinear", align_corners=False
            )

        return self.cnn(render)
