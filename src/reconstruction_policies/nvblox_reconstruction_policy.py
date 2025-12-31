import numpy as np
from src.reconstruction_policies.base import BaseReconstructionPolicy
import nvblox


class NvbloxReconstructionPolicy(BaseReconstructionPolicy):
    """
    Simple reconstruction policy using nvblox for TSDF and mesh extraction.
    """

    def __init__(self, voxel_size=0.02, sdf_trunc=0.08, **kwargs):
        super().__init__(**kwargs)
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.reset()

    def add_obs(
        self,
        camera_intrinsic,
        camera_extrinsic,
        rgb_image=None,
        depth_image=None,
        **kwargs,
    ):
        # nvblox expects depth in meters, camera_intrinsic as 3x3, camera_extrinsic as 4x4
        self.nvblox_mapper.integrate_depth(
            depth_image, camera_intrinsic, camera_extrinsic
        )

    def reconstruct(self, type="tsdf", **kwargs):
        if type == "mesh":
            return self.nvblox_mapper.extract_mesh()
        elif type == "tsdf":
            return self.nvblox_mapper.extract_tsdf()
        else:
            raise ValueError(f"Unknown reconstruction type: {type}")

    def reset(self, **kwargs):
        self.nvblox_mapper = nvblox.NvbloxMapper(
            voxel_size=self.voxel_size, sdf_trunc=self.sdf_trunc
        )
