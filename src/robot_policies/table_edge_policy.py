"""Scripted policy that moves TCP along table edges while looking at table center."""

import numpy as np
from typing import Tuple, Optional, Dict, Any


class TableEdgePolicy:
    """
    A hard-coded robot policy that moves the TCP along the edges of a table
    while the camera always points toward the table center.

    Compatible with StableBaselines3 interface (predict method).

    Default table geometry (based on robosuite Reconstruct3D environment):
        - Table surface at z ≈ 0.8
        - Object/BBox center at approximately (0, 0, 0.9)
        - TCP moves at z ≈ 1.1 (about 0.3m above table)
        - Corners at (±0.3, ±0.3) relative to center

    The policy moves along a rectangular path visiting the corners.

    Action space: Normalized [-1, 1] which the OSC_POSE controller scales to:
        - Position: ±0.1m per step
        - Orientation: ±0.5 rad per step
    """

    def __init__(
        self,
        table_center: Tuple[float, float, float] = (0.0, 0.0, 0.9),
        tcp_height: float = 1.1,
        corner_offset: float = 0.3,
        position_gain: float = 1.0,  # Normalized action gain
        orientation_gain: float = 0.5,  # Normalized action gain
    ):
        """
        Initialize the table edge policy.

        Args:
            table_center: (x, y, z) coordinates of the point to look at
            tcp_height: Height (z) at which TCP should move
            corner_offset: Distance from center to corners (in x and y)
            position_gain: Gain for position control (scales normalized action)
            orientation_gain: Gain for orientation control (scales normalized action)
        """
        self.table_center = np.array(table_center)
        self.tcp_height = tcp_height
        self.corner_offset = corner_offset
        self.position_gain = position_gain
        self.orientation_gain = orientation_gain

        # Define waypoints (corners at TCP height, centered on table)
        # Moving counter-clockwise when viewed from above
        cx, cy = table_center[0], table_center[1]
        self.waypoints = np.array(
            [
                [cx + corner_offset, cy + corner_offset, tcp_height],
                [cx - corner_offset, cy + corner_offset, tcp_height],
                [cx - corner_offset, cy - corner_offset, tcp_height],
                [cx + corner_offset, cy - corner_offset, tcp_height],
            ]
        )

        self.current_waypoint_idx = 0
        self.waypoint_threshold = 0.05  # Distance to consider waypoint reached

        # Physical limits (for computing normalized actions)
        # OSC_POSE controller scales [-1, 1] to these limits
        self.pos_output_max = 0.10  # meters
        self.rot_output_max = 0.5  # radians

    def reset(self):
        """Reset policy state."""
        self.current_waypoint_idx = 0

    def _get_current_position(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract current TCP position from observation."""
        # camera_pose is (7,): position (3) + quaternion wxyz (4)
        camera_pose = observation.get("camera_pose")
        if camera_pose is None:
            raise ValueError("TableEdgePolicy requires 'camera_pose' in observations")
        return camera_pose[:3]

    def _get_current_orientation(
        self, observation: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Extract current TCP orientation quaternion from observation."""
        camera_pose = observation.get("camera_pose")
        if camera_pose is None:
            raise ValueError("TableEdgePolicy requires 'camera_pose' in observations")
        return camera_pose[3:7]  # wxyz quaternion

    def _compute_look_at_rotation(self, current_pos: np.ndarray) -> np.ndarray:
        """
        Compute axis-angle rotation to make camera look at table center.

        The camera should look from current_pos toward table_center.
        Returns axis-angle representation for the action.
        """
        # Direction from camera to target
        direction = self.table_center - current_pos
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # Camera looks along negative z-axis in camera frame (OpenGL convention)
        # We want to align -z_camera with the direction vector

        # Compute desired rotation matrix
        # z-axis points away from target (since camera looks along -z)
        z_axis = -direction

        # Choose up vector (world z typically, but handle edge cases)
        world_up = np.array([0.0, 0.0, 1.0])

        # x-axis = up × z (right vector)
        x_axis = np.cross(world_up, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            # Looking straight up or down, use different up vector
            world_up = np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(world_up, z_axis)
            x_norm = np.linalg.norm(x_axis)
        x_axis = x_axis / (x_norm + 1e-8)

        # y-axis = z × x (up vector in camera frame)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

        # Desired rotation matrix (columns are x, y, z axes)
        R_desired = np.column_stack([x_axis, y_axis, z_axis])

        return R_desired

    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to axis-angle representation."""
        # Rodrigues formula inverse
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

        if theta < 1e-6:
            return np.zeros(3)

        if abs(theta - np.pi) < 1e-6:
            # Handle 180 degree rotation
            # Find axis from eigenvector with eigenvalue 1
            _, V = np.linalg.eig(R)
            axis = np.real(V[:, np.argmax(np.abs(np.diag(R) + 1))])
            return axis * theta

        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
            2 * np.sin(theta)
        )

        return axis * theta

    def _quat_to_rotation_matrix(self, quat_wxyz: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to rotation matrix."""
        w, x, y, z = quat_wxyz

        # Normalize quaternion
        norm = np.sqrt(w * w + x * x + y * y + z * z)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        return R

    def predict(
        self,
        observation: Dict[str, np.ndarray],
        state: Optional[Any] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Compute action to move along table edges while looking at center.

        Compatible with StableBaselines3 policy interface.

        Args:
            observation: Dict containing at least 'camera_pose'
            state: RNN hidden state (unused, for API compatibility)
            episode_start: Whether this is an episode start (resets waypoint index)
            deterministic: Whether to be deterministic (always True for scripted)

        Returns:
            action: 6D action [dx, dy, dz, dax, day, daz]
            state: None (no RNN state)
        """
        # Reset on episode start
        if episode_start is not None and (
            episode_start is True
            or (hasattr(episode_start, "__iter__") and any(episode_start))
        ):
            self.reset()

        # Handle vectorized observations (batch dimension)
        is_batched = False
        if isinstance(observation, dict):
            first_value = next(iter(observation.values()))
            if first_value.ndim > 1:
                is_batched = True
                # Extract single observation for processing
                observation = {k: v[0] for k, v in observation.items()}

        # Get current position and orientation
        current_pos = self._get_current_position(observation)
        current_quat = self._get_current_orientation(observation)
        current_R = self._quat_to_rotation_matrix(current_quat)

        # Get target waypoint
        target_pos = self.waypoints[self.current_waypoint_idx]

        # Check if we've reached the current waypoint
        distance_to_target = np.linalg.norm(current_pos - target_pos)
        if distance_to_target < self.waypoint_threshold:
            # Move to next waypoint
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(
                self.waypoints
            )
            target_pos = self.waypoints[self.current_waypoint_idx]

        # Compute position delta (proportional control)
        pos_error = target_pos - current_pos
        pos_delta = self.position_gain * pos_error

        # Normalize to [-1, 1] action space (controller scales by pos_output_max)
        pos_action = pos_delta / self.pos_output_max
        pos_action = np.clip(pos_action, -1.0, 1.0)

        # Compute desired orientation (look at table center)
        R_desired = self._compute_look_at_rotation(current_pos)

        # Compute rotation error (R_error = R_desired @ R_current.T)
        R_error = R_desired @ current_R.T

        # Convert to axis-angle
        rot_error_aa = self._rotation_matrix_to_axis_angle(R_error)

        # Apply gain and normalize to [-1, 1] action space
        rot_delta = self.orientation_gain * rot_error_aa
        rot_action = rot_delta / self.rot_output_max
        rot_action = np.clip(rot_action, -1.0, 1.0)

        # Combine into action (6D normalized: position + rotation)
        action = np.concatenate([pos_action, rot_action]).astype(np.float32)

        # Handle batched output
        if is_batched:
            action = action[np.newaxis, :]

        return action, None
