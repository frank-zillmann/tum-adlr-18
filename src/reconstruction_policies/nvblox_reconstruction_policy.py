import numpy as np
import torch
from src.reconstruction_policies.base import BaseReconstructionPolicy
from nvblox_torch.mapper import Mapper
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

    def reconstruct(self, type="mesh", **kwargs):
        """
        Reconstruct the scene from integrated observations.
        
        Args:
            type: "mesh" returns (vertices, faces) tuple as numpy arrays
                  "tsdf" returns the raw TsdfLayer object
        
        Returns:
            For "mesh": tuple of (vertices, faces) as numpy arrays
            For "tsdf": TsdfLayer object
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
        else:
            raise ValueError(f"Unknown reconstruction type: {type}")

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
