"""
Test function to compute and visualize SDF from environment mesh.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def test_compute_sdf(env, path_to_save="./data/sdf_output/"):
    """
    Compute SDF using the environment's built-in method and visualize slices.

    Args:
        env: Reconstruct3D environment instance
        sdf_size: Optional SDF grid resolution. If provided, temporarily overrides env.sdf_size.
        path_to_save: Directory to save SDF grid and visualization

    Returns:
        tuple: (sdf_grid, bbox_center, bbox_size) from the environment's object variables
    """
    print("\n" + "=" * 60)
    print("TEST 2: Computing SDF from Environment Mesh")
    print("=" * 60)

    os.makedirs(path_to_save, exist_ok=True)

    # Use the environment's built-in method to compute SDF
    print("\nComputing SDF using env.compute_static_env_sdf()...")
    env.compute_static_env_sdf(geom_groups=[1])

    # Access stored object variables
    sdf_grid = env.sdf_grid
    bbox_center = env.sdf_bbox_center
    bbox_size = env.sdf_bbox_size
    sdf_size = env.sdf_size

    print(f"SDF grid shape: {sdf_grid.shape}")
    print(f"SDF value range: [{sdf_grid.min():.3f}, {sdf_grid.max():.3f}]")
    print(f"Bounding box center: {bbox_center}, size: {bbox_size}")

    # Visualize SDF slices
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    # Show slices along different axes (edges and interior)
    slice_indices = [0, sdf_size // 4, sdf_size // 2, 3 * sdf_size // 4, sdf_size - 1]

    for i, slice_idx in enumerate(slice_indices):
        # XY slice (Z-axis)
        xy_slice = sdf_grid[:, :, slice_idx]
        axes[0, i].imshow(xy_slice, cmap="RdBu", vmin=-0.5, vmax=0.5)
        axes[0, i].set_title(f"XY slice (z={slice_idx})")
        axes[0, i].axis("off")
        # Add SDF values as text annotations
        for row in range(xy_slice.shape[0]):
            for col in range(xy_slice.shape[1]):
                axes[0, i].text(
                    col,
                    row,
                    f"{xy_slice[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black",
                )

        # XZ slice (Y-axis)
        xz_slice = sdf_grid[:, slice_idx, :]
        axes[1, i].imshow(xz_slice, cmap="RdBu", vmin=-0.5, vmax=0.5)
        axes[1, i].set_title(f"XZ slice (y={slice_idx})")
        axes[1, i].axis("off")
        # Add SDF values as text annotations
        for row in range(xz_slice.shape[0]):
            for col in range(xz_slice.shape[1]):
                axes[1, i].text(
                    col,
                    row,
                    f"{xz_slice[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black",
                )

        # YZ slice (X-axis)
        yz_slice = sdf_grid[slice_idx, :, :]
        axes[2, i].imshow(yz_slice, cmap="RdBu", vmin=-0.5, vmax=0.5)
        axes[2, i].set_title(f"YZ slice (x={slice_idx})")
        axes[2, i].axis("off")
        # Add SDF values as text annotations
        for row in range(yz_slice.shape[0]):
            for col in range(yz_slice.shape[1]):
                axes[2, i].text(
                    col,
                    row,
                    f"{yz_slice[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black",
                )

    plt.suptitle("SDF Visualization (negative=inside, positive=outside)")
    plt.tight_layout()
    plt.savefig(path_to_save + "sdf_slices.png", dpi=150)
    print(f"Saved SDF visualization to '{path_to_save}sdf_slices.png'")
    plt.close()

    print("\nSDF computation complete!")

    return
