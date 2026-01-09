import numpy as np
import torch
from src.reconstruction_policies.base import BaseReconstructionPolicy
from nvblox_torch.mapper import Mapper, QueryType
from nvblox_torch.mapper_params import MapperParams


class NvbloxReconstructionPolicy(BaseReconstructionPolicy):
    """
    Simple reconstruction policy using nvblox_torch for TSDF and mesh extraction.
    """

    def __init__(self, voxel_size=0.02, sdf_trunc=0.08, **kwargs):
        super().__init__(**kwargs)
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        # sdf_trunc is in voxels for nvblox_torch
        self.trunc_voxels = sdf_trunc / voxel_size
        self.reset()

    def add_obs(
        self,
        camera_intrinsic,
        camera_extrinsic,
        rgb_image=None,
        depth_image=None,
        **kwargs,
    ):
        # nvblox_torch expects:
        # - depth_frame: torch.Tensor (H, W) on GPU in meters
        # - t_w_c: torch.Tensor (4, 4) on CPU - camera extrinsic (world to camera transform)
        # - intrinsics: torch.Tensor (3, 3) on CPU - camera intrinsic matrix
        
        # Convert numpy arrays to torch tensors if needed
        # Depth goes on GPU
        if isinstance(depth_image, np.ndarray):
            depth_tensor = torch.from_numpy(depth_image).float().cuda()
        else:
            depth_tensor = depth_image.float().cuda()
        
        # Squeeze to 2D if depth has a channel dimension (e.g., H, W, 1) -> (H, W)
        depth_tensor = depth_tensor.squeeze()
        
        # Intrinsics and extrinsics stay on CPU
        if isinstance(camera_intrinsic, np.ndarray):
            intrinsic_tensor = torch.from_numpy(camera_intrinsic).float()
        else:
            intrinsic_tensor = camera_intrinsic.float().cpu()
            
        if isinstance(camera_extrinsic, np.ndarray):
            extrinsic_tensor = torch.from_numpy(camera_extrinsic).float()
        else:
            extrinsic_tensor = camera_extrinsic.float().cpu()
        
        self.nvblox_mapper.add_depth_frame(
            depth_frame=depth_tensor,
            t_w_c=extrinsic_tensor,
            intrinsics=intrinsic_tensor,
        )

    def reconstruct(self, type="mesh", sdf_size=32, sdf_bbox_center=None, sdf_bbox_size=None, **kwargs):
        """
        Reconstruct the scene from integrated observations.
        
        Args:
            type: "mesh" returns (vertices, faces) tuple as numpy arrays
                  "tsdf" returns the raw TsdfLayer object
                  "sdf" returns a 3D numpy array of SDF values for reward computation
            sdf_size: Resolution of the SDF grid for type="sdf" (default 32)
            sdf_bbox_center: Center of the bounding box for SDF query (numpy array shape (3,))
            sdf_bbox_size: Size of the bounding box for SDF query (scalar, length of longest side)
        
        Returns:
            For "mesh": tuple of (vertices, faces) as numpy arrays
            For "tsdf": TsdfLayer object
            For "sdf": numpy array of shape (sdf_size, sdf_size, sdf_size)
        """
        if type == "mesh":
            self.nvblox_mapper.update_color_mesh()
            color_mesh = self.nvblox_mapper.get_color_mesh()
            
            # Extract vertices and faces as numpy arrays
            # vertices() and triangles() are methods that return torch tensors
            vertices = color_mesh.vertices().cpu().numpy()
            faces = color_mesh.triangles().cpu().numpy()
            
            return (vertices, faces)
        elif type == "tsdf":
            return self.nvblox_mapper.tsdf_layer_view()
        elif type == "sdf":
            return self._query_sdf_grid(sdf_size, sdf_bbox_center, sdf_bbox_size)
        else:
            raise ValueError(f"Unknown reconstruction type: {type}")

    def _query_sdf_grid(self, sdf_size, sdf_bbox_center, sdf_bbox_size):
        """
        Query the TSDF on a regular grid compatible with reconstruct3D reward function.
        
        The grid is constructed to match the same coordinate system used by mesh2sdf
        in reconstruct3D.compute_static_env_sdf():
        - The mesh is normalized to [-1, 1] centered at bbox_center with scale bbox_size/2
        - The SDF grid samples this [-1, 1]^3 space uniformly
        
        Args:
            sdf_size: Resolution of the grid (grid will be sdf_size^3)
            sdf_bbox_center: Center of bounding box in world coordinates, shape (3,)
            sdf_bbox_size: Size of bounding box (length of longest axis after padding)
            
        Returns:
            numpy array of shape (sdf_size, sdf_size, sdf_size) with SDF values.
            Unobserved regions are filled with a large positive value (100.0).
        """
        if sdf_bbox_center is None or sdf_bbox_size is None:
            raise ValueError("sdf_bbox_center and sdf_bbox_size must be provided for type='sdf'")
        
        sdf_bbox_center = np.asarray(sdf_bbox_center)
        
        # Create grid in normalized [-1, 1] space (matching mesh2sdf convention)
        # mesh2sdf uses uniform sampling in [-1, 1]^3
        coords_1d = np.linspace(-1, 1, sdf_size)
        
        # Create 3D meshgrid
        xx, yy, zz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        
        # Stack into query points (N, 3) in normalized space
        query_points_normalized = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        
        # Transform from normalized [-1, 1] to world coordinates
        # world = normalized * (bbox_size/2) + bbox_center
        query_points_world = query_points_normalized * (sdf_bbox_size / 2) + sdf_bbox_center
        
        # Convert to torch tensor on GPU for nvblox query
        query_points_tensor = torch.from_numpy(query_points_world).float().cuda()
        
        # Query TSDF layer
        # Returns [N, 2] where column 0 is SDF value, column 1 is weight
        tsdf_result = self.nvblox_mapper.query_layer(QueryType.TSDF, query_points_tensor)
        
        # Extract SDF values and weights
        sdf_values = tsdf_result[:, 0].cpu().numpy()
        weights = tsdf_result[:, 1].cpu().numpy()
        
        # Handle unobserved regions (weight == 0)
        # nvblox returns SDF=100.0 for unobserved
        # TODO: Fix this into two valid value only comparison
        
        # Reshape to 3D grid
        sdf_grid = sdf_values.reshape(sdf_size, sdf_size, sdf_size)
        
        return sdf_grid

    def reset(self, **kwargs):
        # Create mapper params and set truncation distance
        mapper_params = MapperParams()
        integrator_params = mapper_params.get_projective_integrator_params()
        integrator_params.projective_integrator_truncation_distance_vox = self.trunc_voxels
        mapper_params.set_projective_integrator_params(integrator_params)
        
        self.nvblox_mapper = Mapper(
            voxel_sizes_m=self.voxel_size,
            mapper_parameters=mapper_params,
        )
