"""Train 3D reconstruction policy with PPO."""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.reconstruct3D_gym_wrapper import Reconstruct3DGymWrapper
from src.robot_policies.feature_extractors import (
    CameraPoseExtractor,
    CameraPoseMeshRenderingExtractor,
)
from configs.train_config import TrainConfig


def make_env(
    config: TrainConfig,
    seed: int,
    collect_timing: bool = False,
    eval_log_dir=None,
):

    def _init():
        # Select reconstruction policy
        if config.reconstruction_policy == "open3d":
            from src.reconstruction_policies.open3d_TSDF_generator import (
                Open3DTSDFGenerator,
            )

            reconstruction_policy = Open3DTSDFGenerator(
                bbox_min=np.array([-0.5, -0.5, 0.5]),
                bbox_max=np.array([0.5, 0.5, 1.5]),
                voxel_size=0.01,
                sdf_trunc=0.5,
            )
        elif config.reconstruction_policy == "nvblox":
            from src.reconstruction_policies.nvblox_reconstruction_policy import (
                NvbloxReconstructionPolicy,
            )

            reconstruction_policy = NvbloxReconstructionPolicy()
        else:
            raise ValueError(
                f"Unknown reconstruction policy: {config.reconstruction_policy}"
            )
        
        env = Reconstruct3DGymWrapper(
            reconstruction_policy=reconstruction_policy,
            horizon=config.horizon,
            camera_height=config.camera_height,
            camera_width=config.camera_width,
            render_height=config.render_height,
            render_width=config.render_width,
            collect_timing=collect_timing,
            eval_log_dir=eval_log_dir,
            reconstruction_metric=config.reconstruction_metric,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


class TimingCallback(BaseCallback):
    """Callback to save environment timing stats at the end of training."""

    def __init__(self, log_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir

    def _on_training_end(self) -> None:
        """Save timing stats when training ends."""
        # Access the unwrapped env to get timing stats
        # For DummyVecEnv, we can access envs[0]
        try:
            env = self.training_env.envs[0]
            # Unwrap Monitor to get the Reconstruct3DGymWrapper
            while hasattr(env, "env"):
                env = env.env

            stats = env.get_timing_stats()
            if stats and stats.n_steps > 0:
                summary = stats.summary()
                print(f"\n{'='*60}")
                print("TRAINING TIMING STATS")
                print(f"{'='*60}")
                print(summary)

                # Save to file
                benchmark_path = self.log_dir / "timing_stats.txt"
                with open(benchmark_path, "w") as f:
                    f.write("Training Timing Statistics\n")
                    f.write(f"{'='*40}\n\n")
                    f.write(summary)
                    f.write("\n")
                print(f"\nTiming stats saved to: {benchmark_path}")
                print(f"{'='*60}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not save timing stats: {e}")

    def _on_step(self) -> bool:
        return True


def train(config: TrainConfig, checkpoint: str = None):
    """Train PPO agent."""
    # Setup paths
    run_name = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(config.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(str(log_dir / "config.yaml"))

    # Create environments (with timing collection if benchmark enabled)
    # Note: timing only works with DummyVecEnv (n_envs=1), not SubprocVecEnv
    collect_timing = config.benchmark and config.n_envs == 1
    env_fn = lambda i: make_env(config, seed=i, collect_timing=collect_timing)
    train_env = (
        SubprocVecEnv([env_fn(i) for i in range(config.n_envs)])
        if config.n_envs > 1
        else DummyVecEnv([env_fn(0)])
    )
    # Eval env logs images and rewards to log_dir/eval_data/
    eval_log_dir = log_dir / "eval_data"
    eval_env = DummyVecEnv([make_env(config, seed=42, eval_log_dir=eval_log_dir)])

    # Policy kwargs
    policy_kwargs = {
        "features_extractor_class": CameraPoseMeshRenderingExtractor,
        "features_extractor_kwargs": {"features_dim": config.features_dim},
        "net_arch": config.hidden_dims,
    }

    # # Policy kwargs
    # policy_kwargs = {
    #     "features_extractor_class": CameraPoseExtractor,
    #     "features_extractor_kwargs": {"features_dim": config.features_dim},
    #     "net_arch": config.hidden_dims,
    # }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PyTorch/SB3] Using device: {device}")

    # Create or load model
    if checkpoint:
        print(f"Resuming from: {checkpoint}")
        model = PPO.load(
            checkpoint, env=train_env, tensorboard_log=str(log_dir), device=device
        )
    else:
        model = PPO(
            "MultiInputPolicy",  # Required for Dict observation space
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=config.lr,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=config.seed,
            device=device,
        )

    # Callbacks
    callback_list = [
        CheckpointCallback(
            save_freq=config.checkpoint_freq,
            save_path=str(log_dir / "checkpoints"),
            name_prefix="model",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(log_dir / "best"),
            eval_freq=config.eval_freq,
            n_eval_episodes=config.n_eval_episodes,
        ),
    ]

    # Add timing callback if benchmarking
    if collect_timing:
        callback_list.append(TimingCallback(log_dir))

    callbacks = CallbackList(callback_list)

    # Train
    print(f"Training for {config.total_timesteps} steps. Logs: {log_dir}")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not checkpoint,
    )
    print(f"Training complete.")
    model.save(str(log_dir / "final_model"))

    train_env.close()
    eval_env.close()
    print(f"Environments closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Resume from checkpoint"
    )
    args = parser.parse_args()

    config = TrainConfig.load(args.config)
    train(config, args.checkpoint)
