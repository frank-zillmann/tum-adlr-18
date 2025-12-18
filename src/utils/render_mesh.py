"""Mesh rendering using Open3D's offscreen renderer."""

import contextlib
import os
import sys
from typing import Tuple, Optional

import numpy as np
import open3d as o3d


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Temporarily suppress stdout and stderr at the C level (for Filament engine spam)."""
    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    old_stdout = os.dup(stdout_fd)
    old_stderr = os.dup(stderr_fd)

    # Redirect to /dev/null
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.dup2(devnull, stderr_fd)
    try:
        yield
    finally:
        # Restore original file descriptors
        os.dup2(old_stdout, stdout_fd)
        os.dup2(old_stderr, stderr_fd)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)


o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# Global renderer cache to avoid memory leak from repeated creation/destruction
# Key: (width, height), Value: OffscreenRenderer
_renderer_cache: dict = {}


def _get_or_create_renderer(
    width: int, height: int
) -> o3d.visualization.rendering.OffscreenRenderer:
    """Get cached renderer or create new one for given resolution."""
    key = (width, height)
    if key not in _renderer_cache:
        with _suppress_stdout_stderr():
            _renderer_cache[key] = o3d.visualization.rendering.OffscreenRenderer(
                width, height
            )
    return _renderer_cache[key]


def render_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    resolution: Tuple[int, int] = (64, 64),
    grayscale: bool = True,
    lighting: str = "ambient",
) -> np.ndarray:
    """
    Renders mesh using Open3D's offscreen renderer.

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) triangle indices
        extrinsic: (4, 4) camera extrinsic matrix (world-to-camera)
        intrinsic: (3, 3) camera intrinsic matrix
        resolution: (H, W) output resolution
        grayscale: If True, return grayscale image (H, W), else RGB (H, W, 3)
        lighting: Lighting mode, one of:
            - "unlit": Flat shading, no lighting (most consistent for NN input)
            - "ambient": Soft ambient lighting only (uniform, no harsh shadows)

    Returns:
        Image array with values in [0, 1], shape (H, W) or (H, W, 3)
    """
    H, W = resolution

    # Handle empty mesh
    if len(vertices) == 0 or len(faces) == 0:
        if grayscale:
            return np.zeros((H, W), dtype=np.float32)
        return np.zeros((H, W, 3), dtype=np.float32)

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()

    # Paint mesh light gray for visibility
    mesh.paint_uniform_color([0.7, 0.7, 0.7])

    # Get or create cached renderer (avoids memory leak from repeated creation)
    renderer = _get_or_create_renderer(W, H)

    # Clear previous scene content
    renderer.scene.clear_geometry()
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])  # Black background

    # Add mesh to scene with appropriate shader
    material = o3d.visualization.rendering.MaterialRecord()
    if lighting == "unlit":
        # Flat shading, no lighting (most consistent for neural network input)
        material.shader = "defaultUnlit"
    elif lighting == "ambient":
        # Soft ambient lighting only
        material.shader = "defaultLit"
    else:
        raise ValueError(
            f"Unknown lighting mode: {lighting}. Use 'unlit' or 'ambient'."
        )
    renderer.scene.add_geometry("mesh", mesh, material)

    # Configure lighting
    if lighting == "ambient":
        # Disable sun light and use soft ambient lighting profile
        renderer.scene.scene.enable_sun_light(False)
        renderer.scene.set_lighting(
            renderer.scene.LightingProfile.SOFT_SHADOWS, (0, 0, 0)
        )

    # Setup camera intrinsics (must match render resolution)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    # Setup camera extrinsics
    # Robosuite's get_camera_extrinsic_matrix returns camera-to-world (camera pose in world)
    # Open3D's setup_camera expects world-to-camera (view matrix), so we invert
    extrinsic_w2c = np.linalg.inv(extrinsic)
    renderer.setup_camera(intrinsic_o3d, extrinsic_w2c)

    # Render and copy result
    img = np.asarray(renderer.render_to_image()).copy()

    # Convert to float [0, 1]
    result = img.astype(np.float32) / 255.0

    if grayscale:
        return np.mean(result, axis=-1)
    return result
