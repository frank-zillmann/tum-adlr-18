"""
Test TSDF_generator_open3d with Reconstruct3D environment.

Integrates a single depth observation from robot0_eye_in_hand camera
and extracts the reconstructed mesh.
"""

import os
import sys
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
    get_real_depth_map,
)

from src.reconstruction_policies.TSDF_generator_open3d import TSDF_generator_open3d


def test_tsdf_single_observation(save_dir: str = "./data/test_TSDF_generator/"):
    """
    Test TSDF integration with a single observation from robot0_eye_in_hand.

    Args:
        save_dir: Directory to save output mesh
    """
    print("=" * 60)
    print("TEST: TSDF Generator with Reconstruct3D Environment")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # Camera settings
    camera_name = "robot0_eye_in_hand"
    camera_height = 128
    camera_width = 128

    # Create environment
    print("\nCreating Reconstruct3D environment...")
    controller_config = load_composite_controller_config(
        controller="WHOLE_BODY_MINK_IK",
        robot="Panda",
    )

    env = suite.make(
        env_name="Reconstruct3D",
        robots="Panda",
        controller_configs=controller_config,
        horizon=100,
        camera_names=[camera_name],
        camera_heights=camera_height,
        camera_widths=camera_width,
    )

    print("Environment created!")

    try:
        # Reset and get observation
        print("\nResetting environment and getting observation...")
        obs = env.reset()

        # Get depth observation
        depth_key = f"{camera_name}_depth"
        if depth_key not in obs:
            print(f"Available keys: {list(obs.keys())}")
            raise KeyError(f"Depth key '{depth_key}' not found in observations")

        # Get normalized depth and convert to real depth (meters)
        depth_normalized = obs[depth_key][:, :, 0]  # Remove channel dim
        depth_real = get_real_depth_map(env.sim, depth_normalized)

        print(f"Depth shape: {depth_real.shape}")
        print(f"Depth range: [{depth_real.min():.3f}, {depth_real.max():.3f}] meters")

        # Get camera intrinsic and extrinsic matrices
        intrinsic = get_camera_intrinsic_matrix(
            env.sim, camera_name, camera_height, camera_width
        )
        extrinsic = get_camera_extrinsic_matrix(env.sim, camera_name)

        print(f"\nCamera intrinsic matrix:\n{intrinsic}")
        print(f"\nCamera extrinsic (camera to world):\n{extrinsic}")

        # Define workspace bounding box (should cover the table area)
        bbox_min = np.array([-1.0, -1.0, -0.5])  # meters
        bbox_max = np.array([1.0, 1.0, 1.5])  # meters

        print(f"\nWorkspace bounds: min={bbox_min}, max={bbox_max}")

        # Create TSDF generator
        print("\nInitializing TSDF generator...")
        tsdf = TSDF_generator_open3d(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            voxel_size=0.02,  # 2cm voxels
            sdf_trunc=0.05,  # 5cm truncation
            # device auto-selected (CUDA if available, else CPU)
        )

        print(f"Device: {tsdf.device}")

        print(f"Volume size: {tsdf.volume_size}")
        print(f"Grid shape: {tsdf.grid_shape}")

        # Integrate depth observation
        print("\nIntegrating depth observation...")
        tsdf.integrate_depth(
            depth=depth_real,
            camera_intrinsic=intrinsic,
            camera_extrinsic=extrinsic,
            depth_trunc=1.0,  # Ignore depth beyond 1m
            depth_scale=1.0,  # Already in meters
        )
        print("Integration complete!")

        # Debug: Extract point cloud first to verify data
        print("\nExtracting point cloud...")
        pcd = tsdf.extract_point_cloud(weight_threshold=0.0)
        print(f"Point cloud points: {len(pcd.points)}")

        # Extract mesh
        print("\nExtracting mesh...")
        mesh = tsdf.extract_mesh(weight_threshold=0.0)

        print(f"Mesh vertices: {len(mesh.vertices)}")
        print(f"Mesh triangles: {len(mesh.triangles)}")

        if len(mesh.vertices) > 0:
            # Save mesh as OBJ
            mesh_path = os.path.join(save_dir, "tsdf_mesh_single_obs.obj")
            import open3d as o3d

            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f"\nMesh saved to: {mesh_path}")
        else:
            print("\nWarning: No mesh extracted (possibly no surface visible)")

        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    finally:
        env.close()


if __name__ == "__main__":
    test_tsdf_single_observation()
