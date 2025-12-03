"""
Test function to generate mesh from SDF and render it for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import measure
import trimesh


def test_render_sdf_mesh(
    env,
    sdf_grid=None,
    bbox_center=None,
    bbox_size=None,
    path_to_save="./data/sdf_output/",
):
    """
    Extract mesh from SDF using marching cubes and render it.

    Args:
        env: Reconstruct3D environment instance
        sdf_grid: SDF grid array (N, N, N)
        bbox_center: Center of the bounding box used for normalization
        bbox_size: Size of the bounding box used for normalization
        path_to_save: Directory to save renderings
    """

    if sdf_grid is None:
        sdf_grid = env.sdf_grid
    if bbox_center is None:
        bbox_center = env.sdf_bbox_center
    if bbox_size is None:
        bbox_size = env.sdf_bbox_size

    print("\n" + "=" * 60)
    print("TEST 3: Rendering Mesh from SDF")
    print("=" * 60)

    os.makedirs(path_to_save, exist_ok=True)

    # Extract mesh from SDF using marching cubes
    print("\nExtracting mesh from SDF using marching cubes...")
    level = 0.0  # Surface is at SDF value 0

    vertices, faces, normals, values = measure.marching_cubes(sdf_grid, level=level)
    print(f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")

    # Transform vertices back to world coordinates
    sdf_size = sdf_grid.shape[0]
    vertices_normalized = vertices / sdf_size * 2 - 1  # Map to [-1, 1]
    vertices_world = vertices_normalized * (bbox_size / 2) + bbox_center

    print(
        f"Vertex bounds: min={vertices_world.min(axis=0)}, max={vertices_world.max(axis=0)}"
    )

    # Create 3D visualization
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(15, 5))

    # View 1: Front view
    ax1 = fig.add_subplot(131, projection="3d")
    mesh = Poly3DCollection(
        vertices_world[faces], alpha=0.7, edgecolor="k", linewidth=0.1
    )
    mesh.set_facecolor([0.5, 0.7, 1.0])
    ax1.add_collection3d(mesh)
    ax1.set_xlim(vertices_world[:, 0].min(), vertices_world[:, 0].max())
    ax1.set_ylim(vertices_world[:, 1].min(), vertices_world[:, 1].max())
    ax1.set_zlim(vertices_world[:, 2].min(), vertices_world[:, 2].max())
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Front View")
    ax1.view_init(elev=20, azim=0)

    # View 2: Top view
    ax2 = fig.add_subplot(132, projection="3d")
    mesh = Poly3DCollection(
        vertices_world[faces], alpha=0.7, edgecolor="k", linewidth=0.1
    )
    mesh.set_facecolor([0.5, 0.7, 1.0])
    ax2.add_collection3d(mesh)
    ax2.set_xlim(vertices_world[:, 0].min(), vertices_world[:, 0].max())
    ax2.set_ylim(vertices_world[:, 1].min(), vertices_world[:, 1].max())
    ax2.set_zlim(vertices_world[:, 2].min(), vertices_world[:, 2].max())
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Top View")
    ax2.view_init(elev=90, azim=0)

    # View 3: Side view
    ax3 = fig.add_subplot(133, projection="3d")
    mesh = Poly3DCollection(
        vertices_world[faces], alpha=0.7, edgecolor="k", linewidth=0.1
    )
    mesh.set_facecolor([0.5, 0.7, 1.0])
    ax3.add_collection3d(mesh)
    ax3.set_xlim(vertices_world[:, 0].min(), vertices_world[:, 0].max())
    ax3.set_ylim(vertices_world[:, 1].min(), vertices_world[:, 1].max())
    ax3.set_zlim(vertices_world[:, 2].min(), vertices_world[:, 2].max())
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("Side View")
    ax3.view_init(elev=20, azim=90)

    plt.suptitle("Reconstructed Mesh from SDF")
    plt.tight_layout()
    plt.savefig(path_to_save + "sdf_mesh_rendering.png", dpi=150)
    print(f"Saved mesh rendering to '{path_to_save}sdf_mesh_rendering.png'")
    plt.close()

    # Create trimesh object for professional mesh export
    mesh_trimesh = trimesh.Trimesh(
        vertices=vertices_world, faces=faces, vertex_normals=normals
    )

    # Save mesh in multiple formats
    obj_path = path_to_save + "sdf_mesh.obj"
    mesh_trimesh.export(obj_path)
    print(f"Saved mesh as OBJ to '{obj_path}'")

    stl_path = path_to_save + "sdf_mesh.stl"
    mesh_trimesh.export(stl_path)
    print(f"Saved mesh as STL to '{stl_path}'")

    print("\nSDF mesh rendering complete!")
