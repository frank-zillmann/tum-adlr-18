"""Gym-compatible wrapper for 3D reconstruction RL environment."""

import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
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
from src.utils.render_mesh import render_mesh


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
    reset_sdf_total: float = 0.0

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
                + self.reset_sdf_total
            )
            avg_reset = reset_total / self.n_resets * 1000
            lines.extend(
                [
                    f"",
                    f"  Reset breakdown (avg {avg_reset:.1f} ms):",
                    f"    Env reset:      {self.reset_env_total/self.n_resets*1000:6.1f} ms ({100*self.reset_env_total/reset_total:5.1f}%)",
                    f"    Recon reset:    {self.reset_reconstruction_total/self.n_resets*1000:6.1f} ms ({100*self.reset_reconstruction_total/reset_total:5.1f}%)",
                    f"    GT mesh:        {self.reset_mesh_total/self.n_resets*1000:6.1f} ms ({100*self.reset_mesh_total/reset_total:5.1f}%)",
                    f"    GT SDF:         {self.reset_sdf_total/self.n_resets*1000:6.1f} ms ({100*self.reset_sdf_total/reset_total:5.1f}%)",
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
        reconstruction_policy: BaseReconstructionPolicy,
        horizon=40,
        camera_height=128,
        camera_width=128,
        render_height=64,
        render_width=64,
        collect_timing=False,
        eval_log_dir: Optional[Path] = None,
        reconstruction_metric: str = "chamfer_distance",
    ):
        self._step_count = 0
        self._episode_count = 0

        self.collect_timing = collect_timing
        self.camera_resolution = (camera_height, camera_width)
        self.render_resolution = (render_height, render_width)
        self.timing_stats = TimingStats() if collect_timing else None
        self.reconstruction_metric = reconstruction_metric

        # Evaluation logging (only active in val mode with log_dir set)
        self.eval_log_dir = Path(eval_log_dir) if eval_log_dir else None
        self.eval_mode = self.eval_log_dir is not None

        # Use default Panda controller (BASIC with OSC_POSE)
        # This gives delta control in base frame: [dx, dy, dz, dax, day, daz, gripper]
        controller_config = load_composite_controller_config(
            controller=None,  # Load default for robot
            robot="Panda",
        )

        # Increase action limits for camera exploration task
        # Default: position ±0.05m, orientation ±0.5rad
        controller_config["body_parts"]["right"]["output_max"] = [
            0.10,
            0.10,
            0.10,
            0.5,
            0.5,
            0.5,
        ]
        controller_config["body_parts"]["right"]["output_min"] = [
            -0.10,
            -0.10,
            -0.10,
            -0.5,
            -0.5,
            -0.5,
        ]

        print(f"Using controller config: {controller_config}")

        # Always include birdview for reconstruction rendering
        # TODO: try to remove birdview camera in train mode as only extrinsic and intrinsic but not the simulation renderings are needed
        if self.eval_mode:
            self.camera_names = [
                "robot0_eye_in_hand",
                "birdview",
                "frontview",
                "sideview",
            ]
        else:
            self.camera_names = ["robot0_eye_in_hand", "birdview"]

        self.robot_env = robosuite.make(
            env_name="Reconstruct3D",
            robots="Panda",
            controller_configs=controller_config,
            horizon=horizon,
            camera_names=self.camera_names,
            camera_heights=camera_height,
            camera_widths=camera_width,
        )

        # Initialize reconstruction policy
        self.reconstruction_policy = reconstruction_policy

        # Define observation space as Dict
        self.observation_space = spaces.Dict(
            {
                "camera_pose": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
                "reconstruction_render": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1, *self.render_resolution),
                    dtype=np.float32,  # TODO: Correct? Why extra dimension of size 1?
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

    def _get_obs(self, reconstruction) -> Dict[str, np.ndarray]:
        """Get current observation (camera pose + reconstruction render)."""
        # Get birdview camera matrices for rendering reconstruction
        # Pass render resolution to get intrinsic matrix scaled appropriately
        intrinsic = get_camera_intrinsic_matrix(
            self.robot_env.sim,
            "birdview",
            self.render_resolution[0],
            self.render_resolution[1],
        )
        extrinsic = get_camera_extrinsic_matrix(self.robot_env.sim, "birdview")

        # Render current reconstruction from birdview (grayscale for feature extraction)
        vertices, faces = reconstruction
        render = render_mesh(
            vertices,
            faces,
            extrinsic,
            intrinsic,
            self.render_resolution,
            grayscale=True,
        )

        return {
            "camera_pose": self._get_camera_pose(),
            "reconstruction_render": render[np.newaxis, :, :].astype(np.float32),
        }

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
            if (
                np.any(np.isnan(depth_image))
                or np.any(depth_image < 0.0)
                or np.any(depth_image > 1.0)
            ):
                # Log diagnostic info when problematic values are detected
                n_nan = np.sum(np.isnan(depth_image))
                n_invalid = np.sum((depth_image < 0.0) | (depth_image > 1.0))
                print(
                    f"[WARNING] Depth map has {n_nan} NaN values and {n_invalid} out-of-range values at step {self._step_count}"
                )
            depth_image = np.nan_to_num(depth_image, nan=1.0, posinf=1.0, neginf=0.0)
            depth_image = np.clip(depth_image, 0.0, 1.0)
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
        
        # Always get mesh reconstruction for observation/rendering
        mesh_reconstruction = self.reconstruction_policy.reconstruct(type="mesh")
        
        # Get appropriate reconstruction for reward computation based on metric
        if self.reconstruction_metric == "voxelwise_tsdf_error":
            # For dense TSDF-based metrics, get SDF grid for reward computation
            reward_reconstruction = self.reconstruction_policy.reconstruct(
                type="tsdf_dense",
                sdf_size=self.robot_env.sdf_size,
                sdf_bbox_center=self.robot_env.sdf_bbox_center,
                sdf_bbox_size=self.robot_env.sdf_bbox_size,
            )
            truncation_distance = self.reconstruction_policy.sdf_trunc

        elif self.reconstruction_metric == "chamfer_distance":
            reward_reconstruction = mesh_reconstruction
        
        if self.collect_timing:
            self.timing_stats.reconstruction_total += time.perf_counter() - t0

        # Compute reward based on reconstruction quality
        t0 = time.perf_counter()
        reward, error = self.robot_env.reward(
            reconstruction=reconstruction, output_error=True
        )
        if self.collect_timing:
            self.timing_stats.reward_total += time.perf_counter() - t0
            self.timing_stats.n_steps += 1

        # Save eval data if in eval mode
        if self.eval_mode:
            self._save_eval_data(
                reward=reward,
                error=error,
                obs_dict=obs_dict,
                reconstruction=mesh_reconstruction,
            )

        obs = self._get_obs(reconstruction=mesh_reconstruction)
        # TODO: include tsdf/weights as obs

        return obs, reward, done, self._step_count >= self.robot_env.horizon, info

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
        self._episode_count += 1

        # Compute ground truth mesh for reward calculation (chamfer distance)
        t0 = time.perf_counter()
        self.robot_env.compute_static_env_mesh()
        if self.collect_timing:
            self.timing_stats.reset_mesh_total += time.perf_counter() - t0

        # Compute SDF ground truth if using TSDF-based metric
        t0 = time.perf_counter()
        if self.reconstruction_metric in ("voxelwise_tsdf_error"):
            self.robot_env.compute_static_env_sdf()
        if self.collect_timing:
            self.timing_stats.reset_sdf_total += time.perf_counter() - t0
            self.timing_stats.n_resets += 1

        return (
            self._get_obs(
                reconstruction=(
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.int32),
                )
            ),
            {},
        )

    def get_timing_stats(self) -> Optional[TimingStats]:
        """Get accumulated timing statistics (only if collect_timing=True)."""
        return self.timing_stats

    def render(self) -> Optional[np.ndarray]:
        """Render current camera view."""
        return self.robot_env.render()

    def _save_eval_data(self, reward, error, obs_dict, reconstruction):
        """Save buffered evaluation data at end of episode."""
        episode_dir = self.eval_log_dir / f"episode_{self._episode_count:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save rewards CSV
        if not (episode_dir / "rewards.csv").exists():
            with open(episode_dir / "rewards.csv", "w") as f:
                f.write("step,reward,error\n")
        with open(episode_dir / "rewards.csv", "a") as f:
            f.write(f"{self._step_count},{reward},{error}\n")

        # Save images and mesh renders
        for camera_name, obs in obs_dict.items():
            # RGB image
            if "image" in camera_name:
                plt.imsave(
                    episode_dir / f"step_{self._step_count:03d}_{camera_name}.png",
                    obs,
                )

            # Depth image
            if "depth" in camera_name:
                depth_2d = np.squeeze(obs)
                plt.imsave(
                    episode_dir / f"step_{self._step_count:03d}_{camera_name}.png",
                    depth_2d,
                    cmap="viridis",
                )

        for camera_name in self.camera_names:
            # Camera extrinsics and intrinsics (use render resolution for intrinsic)
            extrinsic = get_camera_extrinsic_matrix(self.robot_env.sim, camera_name)
            intrinsic = get_camera_intrinsic_matrix(
                self.robot_env.sim,
                camera_name,
                self.render_resolution[0],
                self.render_resolution[1],
            )

            rendered_reconstruction = render_mesh(
                reconstruction[0],
                reconstruction[1],
                extrinsic,
                intrinsic,
                resolution=self.render_resolution,
                grayscale=False,
            )

            plt.imsave(
                episode_dir
                / f"step_{self._step_count:03d}_{camera_name}_reconstruction.png",
                rendered_reconstruction,
            )

            rendered_gt = render_mesh(
                self.robot_env.static_env_vertices,
                self.robot_env.static_env_faces,
                extrinsic,
                intrinsic,
                resolution=self.render_resolution,
                grayscale=False,
            )

            plt.imsave(
                episode_dir / f"step_{self._step_count:03d}_{camera_name}_gt.png",
                rendered_gt,
            )

    def close(self):
        """Clean up resources."""
        self.robot_env.close()
