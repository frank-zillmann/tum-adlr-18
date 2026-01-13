"""Training configuration with YAML support."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Union
import yaml


@dataclass
class TrainConfig:
    """Training configuration with defaults."""

    # Reconstruction policy and metric (required - must be specified in config)
    reconstruction_policy: str = field()  # 'open3d' or 'nvblox'
    reconstruction_metric: str = field()  # 'chamfer_distance' or 'voxelwise_tsdf_error'

    # Robot Environment
    horizon: int = 40
    control_freq: int = 4
    camera_height: int = 128
    camera_width: int = 128
    render_height: int = 128
    render_width: int = 128

    # Observations to include (camera_pose, mesh_render, sdf_grid, weight_grid)
    observations: List[str] = field(default_factory=lambda: ["camera_pose"])

    # Reward settings
    sdf_gt_size: int = 32  # Size of the ground truth SDF grid along each dimension
    # Factor by which the SDF box is expanded on each side beyond the object bounds
    bbox_padding: float = 0.05

    reward_scale: float = 1.0
    characteristic_error: float = 0.5
    action_penalty_scale: float = 0.02

    # Network
    features_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])

    # PPO and training
    total_timesteps: int = 500_000
    n_envs: int = 1  # Number of parallel environments
    lr: float = 2e-4  # Learning rate
    n_steps: int = 2048  # Steps per env before update
    batch_size: int = 64  # Minibatch size for gradient updates
    n_epochs: int = 5  # Passes over rollout buffer per update, PPO default is 10
    gamma: float = 0.98  # Discount factor, PPO default is 0.99
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    clip_range: float = 0.2  # PPO clipping parameter
    ent_coef: float = 0.00  # Entropy bonus for exploration
    seed: int = 0  # Random seed for PPO (not for envs)

    checkpoint_freq: int = 10_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 1

    # Logging
    log_dir: str = "data/logs"

    # Benchmarking - collect timing stats during training (requires n_envs=1)
    benchmark: bool = False

    def save(self, path: str):
        Path(path).write_text(
            yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
        )

    @classmethod
    def load(cls, paths: Union[str, List[str]]) -> "TrainConfig":
        """Load config, merging with defaults (only override specified fields)."""
        if isinstance(paths, str):
            paths = [paths]

        data = {}
        for path in paths:
            data.update(yaml.safe_load(Path(path).read_text()) or {})
        return cls(**data)
