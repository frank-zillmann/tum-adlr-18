"""Utility for rendering 3D meshes to images using matplotlib."""

from pathlib import Path
from typing import Union, Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def save_mesh_rendering(
    vertices: np.ndarray,
    faces: np.ndarray,
    save_path: Union[str, Path],
    views: List[Tuple[float, float]] = [
        (20, 0),
        (90, 0),
        (20, 90),
    ],  # Front, Top, Side
    figsize: Tuple[int, int] = None,
    dpi: int = 150,
    title: str = None,
):
    """
    Render a triangle mesh and save directly to file.

    Args:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle face indices
        save_path: Path to save the image
        views: List of (elev, azim) tuples for multiple views. Default: single diagonal view
        figsize: Figure size in inches. Default: (5, 5) for single view, (15, 5) for 3 views
        dpi: Resolution of saved image
        title: Optional title for the figure
    """
    if len(vertices) == 0 or len(faces) == 0:
        return

    n_views = len(views)
    if figsize is None:
        figsize = (5 * n_views, 5)

    fig = plt.figure(figsize=figsize)

    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, n_views, i + 1, projection="3d")

        # Create mesh collection
        mesh = Poly3DCollection(
            vertices[faces], alpha=0.7, edgecolor="k", linewidth=0.1
        )
        mesh.set_facecolor([0.5, 0.7, 1.0])
        ax.add_collection3d(mesh)

        # Set axis limits
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)
