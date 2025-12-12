"""Gym-compatible wrapper for 3D reconstruction RL environment."""

from typing import Optional, Tuple, Dict, Any

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
    ):
        self.mode = mode
        self._step_count = 0

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
        # Step robot environment
        obs_dict, _, done, info = self.robot_env.step(action)
        self._step_count += 1

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

        if self.mode == "val" or self.mode == "test":
            pass

        self.reconstruction_policy.add_obs(
            camera_intrinsic=intrinsic,
            camera_extrinsic=extrinsic,
            rgb_image=rgb_image,
            depth_image=depth_image,
        )

        # Compute reward based on reconstruction quality
        reconstruction = self.reconstruction_policy.reconstruct()
        reward = float(self.robot_env.reward(reconstruction=reconstruction))

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

        self.robot_env.reset()
        self.reconstruction_policy.reset()
        self._step_count = 0

        # Compute ground truth mesh and sdf for reward calculation
        self.robot_env.compute_static_env_mesh()
        self.robot_env.compute_static_env_sdf()

        return self._get_obs(), {}

    def render(self) -> Optional[np.ndarray]:
        """Render current camera view."""

        return self.robot_env.render()

    def close(self):
        """Clean up resources."""
        self.robot_env.close()
