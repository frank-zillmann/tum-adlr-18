"""Train 3D reconstruction policy with PPO."""

import argparse
import os
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.robot_policies import (
    CameraPoseExtractor,
    CameraPoseHistoryExtractor,
    MeshRenderingExtractor,
    WeightGridExtractor,
    CombinedExtractor,
)
from src.utils.env_factory import make_env_fn
from configs.train_config import TrainConfig


import signal
import faulthandler

# # --- Crash diagnostics ------------------------------------------------
# # 1. Write crash traceback to a dedicated file (survives lost stderr)
# _fault_file = open("faulthandler_4.log", "w")
# faulthandler.enable(file=_fault_file, all_threads=True)
# # Also keep stderr output as backup
# faulthandler.enable(all_threads=True)

# # 2. Dump traceback on demand:  kill -SIGUSR1 <pid>
# faulthandler.register(signal.SIGUSR1, file=_fault_file, all_threads=True)

# # 3. Watchdog: dump traceback if process hangs for >300s without progress
# #    (resets each interval, repeats forever)
# faulthandler.dump_traceback_later(300, repeat=True, file=_fault_file)

# print(
#     f"[diag] PID={os.getpid()}  faulthandler → faulthandler.log"
#     f"  |  kill -SIGUSR1 {os.getpid()} to dump traceback"
# )


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


def _get_tb_writer(logger):
    """Get SB3's own SummaryWriter from its logger."""
    from stable_baselines3.common.logger import TensorBoardOutputFormat

    for fmt in logger.output_formats:
        if isinstance(fmt, TensorBoardOutputFormat):
            return fmt.writer
    return None


def _log_scalar_info(writer, info, prefix, step):
    """Write all scalar values from an info dict to TensorBoard."""
    for key, value in info.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            writer.add_scalar(f"{prefix}/{key}", value, step)


class TensorboardCallback(BaseCallback):
    """Callback to log custom metrics from environment info to Tensorboard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._tb_writer = None

    def _on_step(self) -> bool:
        try:
            if self._tb_writer is None:
                self._tb_writer = _get_tb_writer(self.logger)
            if self._tb_writer is None:
                return True

            infos = self.locals.get("infos")
            if infos:
                for i, info in enumerate(infos):
                    _log_scalar_info(
                        self._tb_writer, info, f"env_{i}", self.num_timesteps
                    )
        except Exception as e:
            print(f"Error in TensorboardCallback: {e}")

        return True


class LoggingEvalCallback(EvalCallback):
    """EvalCallback that also logs per-step eval info metrics to TensorBoard."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_step_count = 0
        self._tb_writer = None

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        try:
            if self._tb_writer is None:
                self._tb_writer = _get_tb_writer(self.logger)
            if self._tb_writer is None:
                return
            info = locals_.get("info")
            if info:
                _log_scalar_info(self._tb_writer, info, "eval", self._eval_step_count)
            self._eval_step_count += 1
        except Exception as e:
            print(f"Error in LoggingEvalCallback: {e}")


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
    env_fn = lambda i: make_env_fn(config, seed=i, collect_timing=collect_timing)
    train_env = (
        SubprocVecEnv([env_fn(i) for i in range(config.n_envs)])
        if config.n_envs > 1
        else DummyVecEnv([env_fn(0)])
    )
    # Eval env logs images and rewards to log_dir/eval_data/
    eval_log_dir = log_dir / "eval_data"
    eval_env = DummyVecEnv([make_env_fn(config, seed=42, eval_log_dir=eval_log_dir)])

    # Build extractors config based on configured observations
    extractors_config = []
    if "camera_pose" in config.observations:
        extractors_config.append(
            (CameraPoseExtractor, {"features_dim": 32, "hidden_dims": [64]})
        )
    if "camera_pose_history" in config.observations:
        extractors_config.append(
            (
                CameraPoseHistoryExtractor,
                {
                    "features_dim": 64,
                    "d_model": 64,
                    "n_heads": 4,
                    "n_layers": 2,
                    "max_steps": config.horizon,
                },
            )
        )
    if "mesh_render" in config.observations:
        extractors_config.append((MeshRenderingExtractor, {"features_dim": 128}))
    if "sdf_grid" in config.observations:
        # TODO: Add SdfGridExtractor when implemented
        print(
            "Warning: sdf_grid observation enabled but no SdfGridExtractor exists yet"
        )
    if "weight_grid" in config.observations:
        extractors_config.append((WeightGridExtractor, {"features_dim": 128}))

    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "features_extractor_kwargs": {
            "features_dim": config.features_dim,
            "extractors_config": extractors_config,
        },
        "net_arch": dict(pi=config.hidden_dims, vf=config.hidden_dims),
    }

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
        LoggingEvalCallback(
            eval_env,
            best_model_save_path=str(log_dir / "best"),
            eval_freq=config.eval_freq,
            n_eval_episodes=config.n_eval_episodes,
        ),
        TensorboardCallback(),
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
        "--config", type=str, nargs="+", default=[], help="Path(s) to config YAML"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Resume from checkpoint"
    )
    args = parser.parse_args()

    config = TrainConfig.load(args.config)
    train(config, args.checkpoint)
