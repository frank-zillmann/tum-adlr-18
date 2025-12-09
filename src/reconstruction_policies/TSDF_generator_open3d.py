from typing import Optional

import numpy as np
import open3d as o3d


def get_default_device() -> str:
    """Get the best available device (CUDA if available, else CPU)."""
    if o3d.core.cuda.is_available():
        return "CUDA:0"
    return "CPU:0"


class TSDF_generator_open3d:
    """
    Generate Truncated Signed Distance Fields from depth observations using Open3D.

    Uses the tensor-based VoxelBlockGrid API which supports depth-only integration
    without dummy color images.

    Note: bbox_min/bbox_max are only used for the `get_sdf_grid()` method to define
    the query region. The VoxelBlockGrid itself dynamically allocates voxel blocks
    based on where depth observations fall.

    Note: The `get_sdf_grid` method approximates SDF by computing signed distances
    to the extracted mesh, which may differ from the internally stored TSDF values.
    """

    def __init__(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        voxel_size: float,
        sdf_trunc: float,
        device: Optional[str] = None,
    ):
        """
        Initialize TSDF generator.

        Args:
            bbox_min: (3,) array, minimum corner of bounding box for SDF grid queries (meters)
            bbox_max: (3,) array, maximum corner of bounding box for SDF grid queries (meters)
            voxel_size: Size of voxel in meters
            sdf_trunc: Truncation distance for TSDF in meters. Controls how far from
                surfaces we track signed distance. Larger values = more computation but
                less truncation.
            device: Device to use for computation ("CPU:0" or "CUDA:0").
                If None, automatically selects CUDA if available, else CPU.
        """
        self.bbox_min = np.asarray(bbox_min)
        self.bbox_max = np.asarray(bbox_max)
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc

        # Auto-select device if not specified
        device_str = device if device is not None else get_default_device()
        self.device = o3d.core.Device(device_str)

        # Compute grid dimensions (informational, for get_sdf_grid)
        self.volume_size = self.bbox_max - self.bbox_min
        self.grid_shape = np.ceil(self.volume_size / voxel_size).astype(int)

        # Block size for VoxelBlockGrid (typically 8 or 16)
        self.block_resolution = 8

        # Initialize the voxel block grid
        self._init_volume()

    def _init_volume(self):
        """Initialize or reset the voxel block grid."""
        # VoxelBlockGrid with only TSDF and weight (no color)
        self.volume = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight"),
            attr_dtypes=(o3d.core.float32, o3d.core.float32),
            attr_channels=((1,), (1,)),
            voxel_size=self.voxel_size,
            block_resolution=self.block_resolution,
            block_count=50000,  # Initial block count, grows as needed
            device=self.device,
        )

    def integrate_depth(
        self,
        depth: np.ndarray,
        camera_intrinsic: np.ndarray,
        camera_extrinsic: np.ndarray,
        depth_trunc: float,
        depth_scale: float = 1.0,
    ):
        """
        Integrate a depth observation into the TSDF volume.

        Args:
            depth: Depth map (H, W) in meters (if depth_scale=1.0)
            camera_intrinsic: 3x3 camera intrinsic matrix
            camera_extrinsic: 4x4 camera pose (camera to world transform)
            depth_trunc: Maximum depth to integrate in meters. Depth pixels farther
                than this are ignored (useful for filtering noisy far-field depth).
            depth_scale: Scale factor to convert depth values to meters.
                Use 1.0 if depth is already in meters, 1000.0 if in millimeters.
        """
        # Convert to tensor format
        depth_t = o3d.t.geometry.Image(
            o3d.core.Tensor(depth.astype(np.float32), device=self.device)
        )

        # Create intrinsic tensor (3x3)
        intrinsic_t = o3d.core.Tensor(
            camera_intrinsic.astype(np.float64), device=self.device
        )

        # Create extrinsic tensor (4x4) - world to camera
        extrinsic_t = o3d.core.Tensor(
            np.linalg.inv(camera_extrinsic).astype(np.float64), device=self.device
        )

        # Compute truncation in voxels (trunc_voxel_multiplier)
        trunc_voxel_multiplier = self.sdf_trunc / self.voxel_size

        # Get frustum block coordinates to determine which voxel blocks to update
        frustum_block_coords = self.volume.compute_unique_block_coordinates(
            depth_t, intrinsic_t, extrinsic_t, depth_scale, depth_trunc
        )

        # Integrate depth into volume (no color)
        # Note: trunc_voxel_multiplier controls TSDF truncation distance in voxel units
        self.volume.integrate(
            frustum_block_coords,
            depth_t,
            intrinsic_t,
            extrinsic_t,
            depth_scale,
            depth_trunc,
            trunc_voxel_multiplier,
        )

    def extract_mesh(self, weight_threshold: float = 3.0) -> o3d.geometry.TriangleMesh:
        """
        Extract triangle mesh from the TSDF volume using marching cubes.

        Surface is extracted at SDF=0 (the zero-crossing).

        Args:
            weight_threshold: Minimum weight for a voxel to be included in mesh extraction.
                Higher values = more observations required = cleaner but potentially incomplete mesh.

        Returns:
            mesh: Open3D TriangleMesh (empty mesh if no surface extracted)
        """

        mesh = self.volume.extract_triangle_mesh(weight_threshold=weight_threshold)
        # Check if mesh has vertices before converting
        if mesh.vertex.positions.shape[0] == 0:
            print("Warning: No mesh vertices extracted from TSDF volume")
            return o3d.geometry.TriangleMesh()
        mesh_legacy = mesh.to_legacy()
        mesh_legacy.compute_vertex_normals()
        return mesh_legacy

    def extract_point_cloud(
        self, weight_threshold: float = 0.0
    ) -> o3d.geometry.PointCloud:
        """
        Extract point cloud from the TSDF volume.

        Args:
            weight_threshold: Minimum weight for a voxel to be included.

        Returns:
            pcd: Open3D PointCloud
        """
        pcd = self.volume.extract_point_cloud(weight_threshold=weight_threshold)
        return pcd.to_legacy()

    def get_sdf_grid(
        self, grid_resolution: int, weight_threshold: float = 0.0
    ) -> np.ndarray:
        """
        Get SDF values on a regular grid within the bounding box.

        Note: This computes signed distances to the extracted mesh surface,
        which approximates the TSDF but is not truncated.

        Args:
            grid_resolution: Number of points along each axis
            weight_threshold: Minimum weight for mesh extraction (default: 0.0)

        Returns:
            sdf_grid: (grid_resolution, grid_resolution, grid_resolution) array of SDF values
        """
        # Create grid of query points
        x = np.linspace(self.bbox_min[0], self.bbox_max[0], grid_resolution)
        y = np.linspace(self.bbox_min[1], self.bbox_max[1], grid_resolution)
        z = np.linspace(self.bbox_min[2], self.bbox_max[2], grid_resolution)

        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        query_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        # Extract mesh and compute signed distances
        mesh = self.extract_mesh(weight_threshold=weight_threshold)

        if len(mesh.vertices) == 0:
            # No mesh extracted yet, return large positive values (outside)
            return np.full(
                (grid_resolution, grid_resolution, grid_resolution),
                self.sdf_trunc,
                dtype=np.float32,
            )

        # Create raycasting scene for distance queries
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)

        # Compute signed distances
        query_points_t = o3d.core.Tensor(
            query_points.astype(np.float32), dtype=o3d.core.Dtype.Float32
        )
        signed_distances = scene.compute_signed_distance(query_points_t).numpy()

        # Reshape to grid
        sdf_grid = signed_distances.reshape(
            grid_resolution, grid_resolution, grid_resolution
        )

        return sdf_grid

    def get_native_tsdf_points(
        self, weight_threshold: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get native TSDF values at all integrated voxel positions.

        Returns sparse data (only voxels that have been observed).

        **WARNING**: The returned positions are in VoxelBlockGrid's internal coordinate
        system, NOT world coordinates. The voxel_coordinates_and_flattened_indices()
        method returns coordinates relative to an internal grid origin (appears to be
        near the first integrated observation), not the world origin or bbox_min.

        For correct world-space coordinates, use extract_mesh() or get_sdf_grid().

        Args:
            weight_threshold: Minimum weight for a voxel to be returned.

        Returns:
            positions: (N, 3) array of voxel center positions (VoxelBlockGrid internal coordinates!)
            tsdf_values: (N,) array of TSDF values
            weights: (N,) array of integration weights
        """
        # Extract voxel coordinates and indices
        voxel_coords, voxel_indices = (
            self.volume.voxel_coordinates_and_flattened_indices()
        )

        if voxel_coords.shape[0] == 0:
            return np.zeros((0, 3)), np.zeros(0), np.zeros(0)

        # Get attribute tensors
        tsdf_tensor = self.volume.attribute("tsdf")
        weight_tensor = self.volume.attribute("weight")

        # Flatten and convert to numpy
        tsdf_values = tsdf_tensor.reshape(-1).cpu().numpy()
        weight_values = weight_tensor.reshape(-1).cpu().numpy()

        # Use indices to get values
        flat_indices = voxel_indices.cpu().numpy()
        voxel_tsdf = tsdf_values[flat_indices]
        voxel_weights = weight_values[flat_indices]

        # Convert voxel coordinates to positions
        voxel_coords_np = voxel_coords.cpu().numpy()

        # NOTE: VoxelBlockGrid's voxel_coordinates_and_flattened_indices() returns
        # coordinates in an internal sparse hashmap coordinate system. Multiplying
        # by voxel_size gives metric positions, but they're relative to an arbitrary
        # origin, not world coordinates.
        positions = voxel_coords_np * self.voxel_size

        # Filter by weight threshold
        mask = voxel_weights >= weight_threshold

        return positions[mask], voxel_tsdf[mask], voxel_weights[mask]

    def reset(self):
        """Reset the TSDF volume to empty state."""
        self._init_volume()
