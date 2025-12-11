"""Train 3D reconstruction policy with PPO."""

import argparse
from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.reconstruct3D_gym_wrapper import Reconstruct3DGymWrapper
from src.robot_policies.feature_extractors import CameraPoseExtractor
from configs.train_config import TrainConfig


def make_env(mode: str, seed: int, horizon: int):
    def _init():
        env = Monitor(Reconstruct3DGymWrapper(mode=mode, horizon=horizon))
        env.reset(seed=seed)
        return env

    return _init


def train(config: TrainConfig, checkpoint: str = None):
    """Train PPO agent."""
    # Setup paths
    run_name = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(config.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(str(log_dir / "config.yaml"))

    # Create environments
    env_fn = lambda i: make_env("train", i, config.horizon)
    train_env = (
        SubprocVecEnv([env_fn(i) for i in range(config.n_envs)])
        if config.n_envs > 1
        else DummyVecEnv([env_fn(0)])
    )
    eval_env = DummyVecEnv([make_env("val", 42, config.horizon)])

    # Policy kwargs
    policy_kwargs = {
        "features_extractor_class": CameraPoseExtractor,
        "features_extractor_kwargs": {
            "features_dim": config.features_dim,
            "hidden_dims": config.hidden_dims,
        },
        "net_arch": config.hidden_dims,
    }

    # Create or load model
    if checkpoint:
        print(f"Resuming from: {checkpoint}")
        model = PPO.load(checkpoint, env=train_env, tensorboard_log=str(log_dir))
    else:
        model = PPO(
            "MlpPolicy",
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
        )

    # Callbacks
    callbacks = CallbackList(
        [
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
    )

    # Train
    print(f"Training for {config.total_timesteps} steps. Logs: {log_dir}")
    model.learn(
        total_timesteps=config.total_timesteps, callback=callbacks, progress_bar=True
    )
    model.save(str(log_dir / "final_model"))

    train_env.close()
    eval_env.close()


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
