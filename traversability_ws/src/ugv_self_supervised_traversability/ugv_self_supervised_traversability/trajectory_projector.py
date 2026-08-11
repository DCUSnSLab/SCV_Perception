"""Project future traversed footprints into stored past observations."""

import numpy as np

from .observation_buffer import Observation
from .utils.camera_projection import depth_visibility_mask, rasterize_camera_polygon
from .utils.transforms import transform_points


class TrajectoryProjector:
    """Reusable delayed footprint-to-image projector."""

    def __init__(self, depth_check_enabled: bool, depth_tolerance: float) -> None:
        if depth_tolerance < 0.0:
            raise ValueError('depth_tolerance must be non-negative')
        self.depth_check_enabled = depth_check_enabled
        self.depth_tolerance = depth_tolerance

    def project_footprint(self, world_points: np.ndarray, observation: Observation) -> np.ndarray:
        """Return a boolean positive-evidence mask for one footprint."""
        camera_t_world = np.linalg.inv(observation.world_t_camera)
        camera_points = transform_points(world_points, camera_t_world)
        footprint_mask, expected_depth = rasterize_camera_polygon(
            camera_points, observation.intrinsic, observation.label.shape)
        valid = footprint_mask.astype(bool)
        if self.depth_check_enabled:
            if observation.depth_m is None:
                return np.zeros_like(valid)
            valid &= depth_visibility_mask(
                expected_depth, observation.depth_m, self.depth_tolerance)
        return valid
