"""
Test script to extract meshes from Reconstruct3D environment and convert to SDF using mesh2sdf.
"""

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import matplotlib.pyplot as plt
import os
import mesh2sdf


def test_mesh2sdf():
    """Test mesh extraction and SDF conversion"""

    # Create environment
    controller_config = load_composite_controller_config(
        controller="WHOLE_BODY_MINK_IK",
        robot="Panda",
    )

    env = suite.make(
        env_name="Reconstruct3D",
        robots="Panda",
        controller_configs=controller_config,
        horizon=100,
    )

    print("Environment created successfully!")

    # Reset environment
    obs = env.reset()
    print("Environment reset complete")

    # Extract static environment mesh
    print("\nExtracting static environment mesh...")
    vertices, faces = env.get_static_env_mesh()

    print(f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")
    print(f"Vertex bounds: min={vertices.min(axis=0)}, max={vertices.max(axis=0)}")

    # Normalize vertices to [-1, 1] for mesh2sdf
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = (bbox_max - bbox_min).max()

    vertices_normalized = (vertices - bbox_center) / (bbox_size / 2)
    print(
        f"Normalized vertices to range: [{vertices_normalized.min():.3f}, {vertices_normalized.max():.3f}]"
    )

    # Convert to SDF using mesh2sdf
    print("\nConverting mesh to SDF...")
    sdf_size = 64
    sdf_grid = mesh2sdf.compute(vertices_normalized, faces, size=sdf_size, fix=True)

    print(f"SDF grid shape: {sdf_grid.shape}")
    print(f"SDF value range: [{sdf_grid.min():.3f}, {sdf_grid.max():.3f}]")

    # Save SDF grid
    path_to_save = "./data/sdf_output/"
    os.makedirs(path_to_save, exist_ok=True)
    np.save(path_to_save + "sdf_grid.npy", sdf_grid)
    print(f"\nSaved SDF grid to '{path_to_save}sdf_grid.npy'")

    # Visualize SDF slices
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Show slices along different axes
    slice_indices = [sdf_size // 4, sdf_size // 2, 3 * sdf_size // 4]

    for i, slice_idx in enumerate(slice_indices):
        # XY slice (Z-axis)
        axes[0, i].imshow(sdf_grid[:, :, slice_idx], cmap="RdBu", vmin=-0.5, vmax=0.5)
        axes[0, i].set_title(f"XY slice (z={slice_idx})")
        axes[0, i].axis("off")

        # XZ slice (Y-axis)
        axes[1, i].imshow(sdf_grid[:, slice_idx, :], cmap="RdBu", vmin=-0.5, vmax=0.5)
        axes[1, i].set_title(f"XZ slice (y={slice_idx})")
        axes[1, i].axis("off")

    plt.suptitle("SDF Visualization (negative=inside, positive=outside)")
    plt.tight_layout()
    plt.savefig(path_to_save + "sdf_slices.png", dpi=150)
    print(f"Saved SDF visualization to '{path_to_save}sdf_slices.png'")
    plt.close()

    # Close environment
    env.close()
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_mesh2sdf()
