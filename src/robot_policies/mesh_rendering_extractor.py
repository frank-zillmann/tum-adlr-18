"""Image feature extractor for Stable Baselines 3."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ImageExtractor(BaseFeaturesExtractor):
    """Feature extractor for 2D image observations.

    Concatenates one or more image observations channel-wise and processes
    them through a CNN with BatchNorm.  Each input is resized to 64x64 in
    the forward pass if needed, so camera and render resolutions can differ.

    Supports any combination of single-channel (e.g. ``mesh_render``) and
    multi-channel (e.g. ``birdview_image`` RGB) observations.  The total
    number of input channels is inferred from the observation space.

    Args:
        observation_space: Gym observation space
        features_dim: Output dimension of the feature extractor
        image_keys: Observation keys to concatenate channel-wise.
            Defaults to ``["mesh_render"]`` for backward compatibility.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int,
        image_keys: Optional[List[str]] = None,
    ):
        super().__init__(observation_space, features_dim)
        if image_keys is None:
            image_keys = ["mesh_render"]
        self.image_keys = image_keys

        # Infer total input channels from observation space shapes
        in_channels = sum(
            observation_space[k].shape[0]  # type: ignore[index]
            for k in image_keys
        )

        print(
            f"ImageExtractor: using keys {image_keys} with total {in_channels} channels"
        )

        # CNN: (in_channels, 64, 64) -> features
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # -> 16x32x32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 32x16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 64x8x8 = 4096
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for key in self.image_keys:
            img = observations[key]
            if img.shape[-2:] != (64, 64):
                img = nn.functional.interpolate(
                    img, size=(64, 64), mode="bilinear", align_corners=False
                )
            parts.append(img)
        x = torch.cat(parts, dim=1)  # (B, C_total, 64, 64)
        return self.cnn(x)
