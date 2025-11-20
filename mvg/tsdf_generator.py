import numpy as np
from typing import Optional, Tuple
import open3d as o3d


class TSDFGenerator:
    """
    Generate Truncated Signed Distance Fields from RGB-D observations with open3d.
    """

    def __init__(
        self,
        voxel_size: float = 0.02,
        sdf_trunc: float = 0.10, # TODO: What is a good default here?
        workspace_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Initialize TSDF generator.

        Args:
            voxel_size: Size of voxel in meters (e.g., 0.02 = 2cm)
            sdf_trunc: Truncation distance for TSDF in meters
            workspace_bounds: Optional (min_xyz, max_xyz) to limit volume
        """
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.workspace_bounds = workspace_bounds

        # Initialize Open3D TSDF volume
        if workspace_bounds is not None:
            min_bound, max_bound = workspace_bounds
            volume_size = max_bound - min_bound
            self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=voxel_size,
                sdf_trunc=sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )
        else:
            # Use scalable volume if no bounds specified
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
        depth_scale: float = 1.0,
        depth_trunc: float = 3.0,
    ):
        """
        Integrate an RGB-D observation into the TSDF volume.

        Args:
            rgb: RGB image (H, W, 3) with values in [0, 255]
            depth: Depth map (H, W) in meters
            camera_intrinsic: 3x3 camera intrinsic matrix
            camera_extrinsic: 4x4 camera pose (camera to world)
            depth_scale: Scale factor for depth (usually 1.0 if already in meters)
            depth_trunc: Maximum depth to integrate in meters
        """
        # Convert to Open3D format
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
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0 / depth_scale,  # Open3D expects inverse scale
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        # Integrate into volume
        self.volume.integrate(
            rgbd,
            intrinsic_o3d,
            np.linalg.inv(camera_extrinsic),  # Open3D expects world to camera
        )

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        Extract triangle mesh from the TSDF volume using marching cubes.

        Returns:
            mesh: Open3D TriangleMesh
        """
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        Extract point cloud from the TSDF volume.

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

    def get_sdf_values(self, query_points: np.ndarray) -> np.ndarray:
        """
        Query SDF values at specific 3D points.

        Note: Open3D's ScalableTSDFVolume doesn't directly expose SDF values,
        so this is an approximation using nearest mesh surface distance.

        Args:
            query_points: (N, 3) array of 3D points

        Returns:
            sdf_values: (N,) array of signed distance values
        """
        # Extract mesh first
        mesh = self.extract_mesh()

        # Create scene for distance queries
        mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_legacy)

        # Compute signed distances
        query_points_t = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
        signed_distances = scene.compute_signed_distance(query_points_t).numpy()

        return signed_distances

    def reset(self):
        """Reset the TSDF volume."""
        self.volume.reset()

