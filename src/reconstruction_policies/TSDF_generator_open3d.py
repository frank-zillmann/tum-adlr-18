import numpy as np
import open3d as o3d


class TSDF_generator_open3d:
    """
    Generate Truncated Signed Distance Fields from RGB-D observations using Open3D.

    Note: Open3D's ScalableTSDFVolume does NOT provide direct access to raw TSDF voxel values.
    The `get_sdf_grid` method approximates SDF by computing signed distances to the extracted mesh,
    which may differ from the internally stored TSDF values.
    """

    def __init__(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        voxel_size: float = 0.02,
        sdf_trunc: float = 0.10,
    ):
        """
        Initialize TSDF generator.

        Args:
            bbox_min: (3,) array, minimum corner of workspace bounding box in meters
            bbox_max: (3,) array, maximum corner of workspace bounding box in meters
            voxel_size: Size of voxel in meters (e.g., 0.02 = 2cm)
            sdf_trunc: Truncation distance for TSDF in meters. This determines how far
                from surfaces we track signed distance values. Values beyond this are clamped.
        """
        self.bbox_min = np.asarray(bbox_min)
        self.bbox_max = np.asarray(bbox_max)
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc

        # Compute grid dimensions
        self.volume_size = self.bbox_max - self.bbox_min
        self.grid_shape = np.ceil(self.volume_size / voxel_size).astype(int)

        # Initialize Open3D TSDF volume
        # Note: ScalableTSDFVolume doesn't use explicit bounds, it grows dynamically
        # but we track bounds for our grid extraction
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def integrate_rgbd(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_intrinsic: np.ndarray,
        camera_extrinsic: np.ndarray,
        depth_trunc: float = 3.0,
    ):
        """
        Integrate an RGB-D observation into the TSDF volume.

        Args:
            rgb: RGB image (H, W, 3) with values in [0, 255]. Used for colorizing the mesh.
            depth: Depth map (H, W) in meters
            camera_intrinsic: 3x3 camera intrinsic matrix
            camera_extrinsic: 4x4 camera pose (camera to world transform)
            depth_trunc: Maximum depth to integrate in meters (pixels farther are ignored)
        """
        h, w = depth.shape

        # Create Open3D intrinsic
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=w,
            height=h,
            fx=camera_intrinsic[0, 0],
            fy=camera_intrinsic[1, 1],
            cx=camera_intrinsic[0, 2],
            cy=camera_intrinsic[1, 2],
        )

        # Convert numpy arrays to Open3D images
        rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

        # Create RGBD image
        # depth_scale=1.0 means depth values are already in meters (no conversion needed)
        # depth_trunc clips pixels with depth > depth_trunc
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        # Integrate into volume
        # Open3D expects world-to-camera transform (inverse of camera_extrinsic)
        self.volume.integrate(
            rgbd,
            intrinsic_o3d,
            np.linalg.inv(camera_extrinsic),
        )

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        Extract triangle mesh from the TSDF volume using marching cubes.

        Returns:
            mesh: Open3D TriangleMesh with vertex colors from integrated RGB
        """
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        Extract colored point cloud from the TSDF volume.

        Returns:
            pcd: Open3D PointCloud
        """
        return self.volume.extract_point_cloud()

    def extract_voxel_grid(self) -> o3d.geometry.VoxelGrid:
        """
        Extract voxel grid from the TSDF volume.

        Returns:
            voxel_grid: Open3D VoxelGrid
        """
        return self.volume.extract_voxel_grid()

    def get_sdf_grid(self, grid_resolution: int) -> np.ndarray:
        """
        Get SDF values on a regular grid within the bounding box.

        Note: This is an APPROXIMATION. Open3D's ScalableTSDFVolume doesn't expose
        raw TSDF values, so we extract the mesh and compute signed distances to it.
        This may differ from the internally stored truncated SDF values.

        Args:
            grid_resolution: Number of points along each axis

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
        mesh = self.extract_mesh()

        if len(mesh.vertices) == 0:
            # No mesh extracted yet, return large positive values (outside)
            return (
                np.ones((grid_resolution, grid_resolution, grid_resolution))
                * self.sdf_trunc
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

    def reset(self):
        """Reset the TSDF volume to empty state."""
        self.volume.reset()
