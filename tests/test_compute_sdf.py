"""
Test function to compute and visualize SDF from environment mesh.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import mesh2sdf


def test_compute_sdf(env, path_to_save="./data/sdf_output/"):
    """
    Extract environment mesh, compute SDF, and visualize slices.

    Args:
        env: Reconstruct3D environment instance
        path_to_save: Directory to save SDF grid and visualization
    """
    print("\n" + "=" * 60)
    print("TEST 2: Computing SDF from Environment Mesh")
    print("=" * 60)

    os.makedirs(path_to_save, exist_ok=True)

    # Extract static environment mesh
    print("\nExtracting static environment mesh...")
    # Use collision geoms only (group 1+) to avoid duplicates
    vertices, faces = env.get_static_env_mesh(geom_groups=[1])

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

    # Save SDF grid and metadata
    np.save(path_to_save + "sdf_grid.npy", sdf_grid)
    np.savez(
        path_to_save + "sdf_metadata.npz",
        bbox_center=bbox_center,
        bbox_size=bbox_size,
        sdf_size=sdf_size,
    )
    print(f"\nSaved SDF grid to '{path_to_save}sdf_grid.npy'")
    print(f"Saved metadata to '{path_to_save}sdf_metadata.npz'")

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

    print("\nSDF computation complete!")

    return sdf_grid, bbox_center, bbox_size
