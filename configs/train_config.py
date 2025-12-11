"""Training configuration with YAML support."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List
import yaml


@dataclass
class TrainConfig:
    """Training configuration with sensible defaults for T4 GPU (4 cores, 32GB)."""
    # Environment
    horizon: int = 10
    camera_height: int = 128
    camera_width: int = 128
    
    # PPO hyperparameters (see docstring below for explanation)
    lr: float = 3e-4
    n_steps: int = 512        # Steps per env before update (512 * n_envs = rollout buffer)
    batch_size: int = 64      # Minibatch size for gradient updates
    n_epochs: int = 10        # Passes over rollout buffer per update
    gamma: float = 0.95       # Discount factor
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    clip_range: float = 0.2   # PPO clipping parameter
    ent_coef: float = 0.01    # Entropy bonus for exploration
    
    # Network
    features_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    
    # Training
    total_timesteps: int = 100_000
    n_envs: int = 2           # Parallel envs (2-4 good for T4)
    seed: int = 0
    
    # Logging
    log_dir: str = "data/logs"
    checkpoint_freq: int = 10_000
    eval_freq: int = 5_000
    n_eval_episodes: int = 5
    
    def save(self, path: str):
        Path(path).write_text(yaml.dump(asdict(self), default_flow_style=False, sort_keys=False))
    
    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        """Load config, merging with defaults (only override specified fields)."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)
