"""Training configuration with YAML support."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Union
import yaml


@dataclass
class TrainConfig:
    """Training configuration with sensible defaults."""

    # Environment
    horizon: int = 40
    camera_height: int = 128
    camera_width: int = 128
    render_height: int = 128
    render_width: int = 128

    # PPO hyperparameters
    lr: float = 2e-4
    n_steps: int = 1024  # Steps per env before update
    batch_size: int = 256  # Minibatch size for gradient updates
    n_epochs: int = 5  # Passes over rollout buffer per update
    gamma: float = 0.98  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    clip_range: float = 0.2  # PPO clipping parameter
    ent_coef: float = 0.01  # Entropy bonus for exploration

    # Network
    features_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])

    # Training
    total_timesteps: int = 100_000
    n_envs: int = 1  # Use 1 for safety, increase on powerful machines
    seed: int = 0

    # Logging
    log_dir: str = "data/logs"
    # Reconstruction policy: 'open3d' (default) or 'nvblox'
    reconstruction_policy: str = "open3d"
    # Reconstruction metric for reward computation:
    # - 'chamfer_distance': compute Chamfer distance between reconstructed and ground truth meshes
    # - 'voxelwise_tsdf_error': compute elementwise SDF error on a regular grid
    reconstruction_metric: str = "chamfer_distance"
    checkpoint_freq: int = 10_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5

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
