"""Custom feature extractors for Stable Baselines 3.

This module re-exports extractors from their individual modules for backward compatibility.
Consider importing directly from the specific module or from src.robot_policies.
"""

# Re-export all extractors for backward compatibility
from src.robot_policies.camera_pose_extractor import CameraPoseExtractor
from robot_policies.image_extractor import ImageExtractor, MeshRenderingExtractor
from robot_policies.weight_grid_extractor import SDFWeightExtractor
from src.robot_policies.combined_extractor import (
    CombinedExtractor,
    CameraPoseMeshRenderingExtractor,
)

__all__ = [
    "CameraPoseExtractor",
    "MeshRenderingExtractor",
    "SDFWeightExtractor",
    "CombinedExtractor",
    "CameraPoseMeshRenderingExtractor",
]
