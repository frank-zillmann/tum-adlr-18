"""Gym-compatible wrapper for 3D reconstruction RL environment."""

import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
    get_real_depth_map,
)

from src.reconstruction_policies.base import BaseReconstructionPolicy
from src.reconstruction_policies.TSDF_generator_open3d import TSDF_generator_open3d


@dataclass
class TimingStats:
    """Accumulated timing statistics for steps and resets."""

    # Step timing
    n_steps: int = 0
    simulation_total: float = 0.0
    reconstruction_total: float = 0.0
    reward_total: float = 0.0

    # Reset timing
    n_resets: int = 0
    reset_env_total: float = 0.0
    reset_reconstruction_total: float = 0.0
    reset_mesh_total: float = 0.0

    def summary(self) -> str:
        if self.n_steps == 0:
            return "No timing data collected"

        step_total = (
            self.simulation_total + self.reconstruction_total + self.reward_total
        )
        avg_step = step_total / self.n_steps * 1000  # ms

        lines = [
            f"Timing over {self.n_steps} steps ({self.n_resets} resets):",
            f"",
            f"  Step breakdown (avg {avg_step:.1f} ms):",
            f"    Simulation:     {self.simulation_total/self.n_steps*1000:6.1f} ms ({100*self.simulation_total/step_total:5.1f}%)",
            f"    Reconstruction: {self.reconstruction_total/self.n_steps*1000:6.1f} ms ({100*self.reconstruction_total/step_total:5.1f}%)",
            f"    Reward:         {self.reward_total/self.n_steps*1000:6.1f} ms ({100*self.reward_total/step_total:5.1f}%)",
        ]

        if self.n_resets > 0:
            reset_total = (
                self.reset_env_total
                + self.reset_reconstruction_total
                + self.reset_mesh_total
            )
            avg_reset = reset_total / self.n_resets * 1000
            lines.extend(
                [
                    f"",
                    f"  Reset breakdown (avg {avg_reset:.1f} ms):",
                    f"    Env reset:      {self.reset_env_total/self.n_resets*1000:6.1f} ms ({100*self.reset_env_total/reset_total:5.1f}%)",
                    f"    Recon reset:    {self.reset_reconstruction_total/self.n_resets*1000:6.1f} ms ({100*self.reset_reconstruction_total/reset_total:5.1f}%)",
                    f"    GT mesh:        {self.reset_mesh_total/self.n_resets*1000:6.1f} ms ({100*self.reset_mesh_total/reset_total:5.1f}%)",
                ]
            )

        return "\n".join(lines)


class Reconstruct3DGymWrapper(gym.Env):
    """
    Gym-compatible wrapper combining robosuite environment with reconstruction policy.

    The RL agent controls camera trajectories to optimize 3D reconstruction quality.

    Observation (Dict):
        - camera_pose: (7,) position (3) + quaternion wxyz (4)

    Action:
        - Uses OSC_POSE controller (Operational Space Control)
        - 7D: position delta (3) + axis-angle rotation delta (3) + gripper (1)
        - Deltas are in robot base frame, scaled from [-1,1] to physical limits
        - Position: ±5cm, Rotation: ±0.5rad per step

    Reward: reconstruction quality (chamfer distance or SDF error)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        mode: str = "train",
        reconstruction_policy: BaseReconstructionPolicy = TSDF_generator_open3d(
            bbox_min=np.array([-0.5, -0.5, 0.5]),
            bbox_max=np.array([0.5, 0.5, 1.5]),
            voxel_size=0.05,
            sdf_trunc=0.1,
        ),
        horizon=10,
        camera_height=128,
        camera_width=128,
        collect_timing=False,
    ):
        self.mode = mode
        self._step_count = 0
        self.collect_timing = collect_timing
        self.timing_stats = TimingStats() if collect_timing else None

        # Use default Panda controller (BASIC with OSC_POSE)
        # This gives delta control in base frame: [dx, dy, dz, dax, day, daz, gripper]
        controller_config = load_composite_controller_config(
            controller=None,  # Load default for robot
            robot="Panda",
        )

        if self.mode == "train":
            camera_names = [
                "robot0_eye_in_hand",
            ]
        elif self.mode == "val" or self.mode == "test":
            camera_names = [
                "robot0_eye_in_hand",
                "frontview",
                "birdview",
                "sideview",
            ]
        else:
            raise ValueError(f"Unknown type: {self.mode}")

        self.robot_env = robosuite.make(
            env_name="Reconstruct3D",
            robots="Panda",
            controller_configs=controller_config,
            horizon=horizon,
            camera_names=camera_names,
            camera_heights=camera_height,
            camera_widths=camera_width,
        )

        # Initialize reconstruction policy
        self.reconstruction_policy = reconstruction_policy

        # Define observation space as Dict for flexibility
        self.observation_space = spaces.Dict(
            {
                "camera_pose": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
            }
        )

        # Action space: use robot's native action space
        low, high = self.robot_env.action_spec
        self.action_space = spaces.Box(
            low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32
        )

    def _get_camera_pose(self) -> np.ndarray:
        """Extract camera pose (position + quaternion) from environment."""
        extrinsic = get_camera_extrinsic_matrix(
            self.robot_env.sim, "robot0_eye_in_hand"
        )
        position = extrinsic[:3, 3]
        # Convert rotation matrix to quaternion (wxyz format)
        from robosuite.utils.transform_utils import mat2quat

        quat = mat2quat(extrinsic[:3, :3])
        return np.concatenate([position, quat]).astype(np.float32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation (camera pose)."""
        obs = {"camera_pose": self._get_camera_pose()}
        return obs

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        # Step robot environment (physics + rendering)
        t0 = time.perf_counter()
        obs_dict, _, done, info = self.robot_env.step(action)
        self._step_count += 1
        if self.collect_timing:
            self.timing_stats.simulation_total += time.perf_counter() - t0

        # Get depth observation and integrate into reconstruction
        rgb_image = obs_dict.get("robot0_eye_in_hand_image")
        depth_image = obs_dict.get("robot0_eye_in_hand_depth")
        if depth_image is not None:
            depth_image = get_real_depth_map(self.robot_env.sim, depth_image)

        # Get camera intrinsics and extrinsics
        intrinsic = get_camera_intrinsic_matrix(
            self.robot_env.sim,
            "robot0_eye_in_hand",
            self.robot_env.camera_heights[0],
            self.robot_env.camera_widths[0],
        )
        extrinsic = get_camera_extrinsic_matrix(
            self.robot_env.sim, "robot0_eye_in_hand"
        )

        # TSDF integration
        t0 = time.perf_counter()
        self.reconstruction_policy.add_obs(
            camera_intrinsic=intrinsic,
            camera_extrinsic=extrinsic,
            rgb_image=rgb_image,
            depth_image=depth_image,
        )
        if self.collect_timing:
            self.timing_stats.reconstruction_total += time.perf_counter() - t0

        # Compute reward based on reconstruction quality
        t0 = time.perf_counter()
        reconstruction = self.reconstruction_policy.reconstruct()
        reward = float(self.robot_env.reward(reconstruction=reconstruction))
        if self.collect_timing:
            self.timing_stats.reward_total += time.perf_counter() - t0
            self.timing_stats.n_steps += 1

        # Episode termination
        terminated = done
        truncated = self._step_count >= self.robot_env.horizon

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and return initial observation."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Reset robot environment (MuJoCo state)
        t0 = time.perf_counter()
        self.robot_env.reset()
        if self.collect_timing:
            self.timing_stats.reset_env_total += time.perf_counter() - t0

        # Reset reconstruction policy (TSDF volume)
        t0 = time.perf_counter()
        self.reconstruction_policy.reset()
        if self.collect_timing:
            self.timing_stats.reset_reconstruction_total += time.perf_counter() - t0

        self._step_count = 0

        # Compute ground truth mesh for reward calculation (chamfer distance)
        t0 = time.perf_counter()
        self.robot_env.compute_static_env_mesh()
        if self.collect_timing:
            self.timing_stats.reset_mesh_total += time.perf_counter() - t0

        # SDF computation (currently skipped - mesh2sdf can be slow and not needed yet)
        # t0 = time.perf_counter()
        # self.robot_env.compute_static_env_sdf()
        # if self.collect_timing:
        #     self.timing_stats.reset_sdf_total += time.perf_counter() - t0

        if self.collect_timing:
            self.timing_stats.n_resets += 1

        return self._get_obs(), {}

    def get_timing_stats(self) -> Optional[TimingStats]:
        """Get accumulated timing statistics (only if collect_timing=True)."""
        return self.timing_stats

    def render(self) -> Optional[np.ndarray]:
        """Render current camera view."""

        return self.robot_env.render()

    def close(self):
        """Clean up resources."""
        self.robot_env.close()
