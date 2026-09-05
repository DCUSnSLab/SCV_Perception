"""NumPy ground-plane fitting used by the parking perception node."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class GroundSegmentationConfig:
    """Geometry and RANSAC thresholds for one optical-frame cloud."""

    expected_up: np.ndarray
    ransac_iterations: int
    max_ransac_points: int
    ransac_distance_threshold_m: float
    max_ground_tilt_deg: float
    min_ground_inliers: int
    min_ground_inlier_ratio: float
    secondary_ransac_iterations: int
    secondary_min_ground_inlier_ratio: float
    secondary_min_normal_delta_deg: float
    secondary_max_plane_distance_from_origin_m: float
    ground_candidate_min_range_m: float
    ground_candidate_max_range_m: float
    ground_candidate_min_down_m: float
    ground_candidate_max_down_m: float
    min_plane_distance_from_origin_m: float
    max_plane_distance_from_origin_m: float
    obstacle_min_height_m: float
    obstacle_max_height_m: float
    obstacle_min_range_m: float
    obstacle_max_range_m: float
    obstacle_radius_filter_radius_m: float
    obstacle_radius_filter_min_neighbors: int
    obstacle_far_radius_filter_start_m: float
    obstacle_far_radius_filter_radius_m: float
    obstacle_far_radius_filter_min_neighbors: int
    obstacle_voxel_size_m: float
    ground_region_filter_enabled: bool
    ground_region_grid_size_m: float
    ground_region_max_step_m: float
    ground_region_max_slope_deg: float
    ground_region_max_plane_residual_m: float
    ground_region_point_tolerance_m: float
    ground_region_min_points: int

    def normalized(self) -> 'GroundSegmentationConfig':
        """Return a validated, normalized copy of this configuration."""
        up = np.asarray(self.expected_up, dtype=np.float32).reshape(-1)
        if up.size != 3:
            raise ValueError(
                'expected_up_vector must contain exactly 3 values')
        norm = float(np.linalg.norm(up))
        if norm < 1.0e-9:
            raise ValueError('expected_up_vector must be non-zero')

        return GroundSegmentationConfig(
            expected_up=up / norm,
            ransac_iterations=max(1, int(self.ransac_iterations)),
            max_ransac_points=max(3, int(self.max_ransac_points)),
            ransac_distance_threshold_m=max(
                0.001, float(self.ransac_distance_threshold_m)),
            max_ground_tilt_deg=min(
                89.0, max(0.0, float(self.max_ground_tilt_deg))),
            min_ground_inliers=max(3, int(self.min_ground_inliers)),
            min_ground_inlier_ratio=min(
                1.0, max(0.0, float(self.min_ground_inlier_ratio))),
            secondary_ransac_iterations=max(
                0, int(self.secondary_ransac_iterations)),
            secondary_min_ground_inlier_ratio=min(
                1.0,
                max(0.0, float(self.secondary_min_ground_inlier_ratio)),
            ),
            secondary_min_normal_delta_deg=min(
                89.0,
                max(0.0, float(self.secondary_min_normal_delta_deg)),
            ),
            secondary_max_plane_distance_from_origin_m=max(
                float(self.min_plane_distance_from_origin_m),
                float(self.secondary_max_plane_distance_from_origin_m),
            ),
            ground_candidate_min_range_m=max(
                0.0, float(self.ground_candidate_min_range_m)),
            ground_candidate_max_range_m=max(
                float(self.ground_candidate_min_range_m),
                float(self.ground_candidate_max_range_m)),
            ground_candidate_min_down_m=max(
                0.0, float(self.ground_candidate_min_down_m)),
            ground_candidate_max_down_m=max(
                float(self.ground_candidate_min_down_m),
                float(self.ground_candidate_max_down_m)),
            min_plane_distance_from_origin_m=max(
                0.0, float(self.min_plane_distance_from_origin_m)),
            max_plane_distance_from_origin_m=max(
                float(self.min_plane_distance_from_origin_m),
                float(self.max_plane_distance_from_origin_m)),
            obstacle_min_height_m=max(
                0.0, float(self.obstacle_min_height_m)),
            obstacle_max_height_m=max(
                float(self.obstacle_min_height_m),
                float(self.obstacle_max_height_m)),
            obstacle_min_range_m=max(
                0.0, float(self.obstacle_min_range_m)),
            obstacle_max_range_m=max(
                float(self.obstacle_min_range_m),
                float(self.obstacle_max_range_m)),
            obstacle_radius_filter_radius_m=max(
                0.0, float(self.obstacle_radius_filter_radius_m)),
            obstacle_radius_filter_min_neighbors=max(
                0, int(self.obstacle_radius_filter_min_neighbors)),
            obstacle_far_radius_filter_start_m=max(
                0.0, float(self.obstacle_far_radius_filter_start_m)),
            obstacle_far_radius_filter_radius_m=max(
                0.0, float(self.obstacle_far_radius_filter_radius_m)),
            obstacle_far_radius_filter_min_neighbors=max(
                0, int(self.obstacle_far_radius_filter_min_neighbors)),
            obstacle_voxel_size_m=max(
                0.0, float(self.obstacle_voxel_size_m)),
            ground_region_filter_enabled=bool(
                self.ground_region_filter_enabled),
            ground_region_grid_size_m=max(
                0.02, float(self.ground_region_grid_size_m)),
            ground_region_max_step_m=max(
                0.0, float(self.ground_region_max_step_m)),
            ground_region_max_slope_deg=min(
                45.0, max(0.0, float(self.ground_region_max_slope_deg))),
            ground_region_max_plane_residual_m=max(
                0.001, float(self.ransac_distance_threshold_m),
                float(self.ground_region_max_plane_residual_m),
            ),
            ground_region_point_tolerance_m=max(
                0.001, float(self.ransac_distance_threshold_m),
                float(self.ground_region_point_tolerance_m),
            ),
            ground_region_min_points=max(
                1, int(self.ground_region_min_points)),
        )


@dataclass(frozen=True)
class PlaneModel:
    """Oriented plane ``normal dot point + offset = 0``."""

    normal: np.ndarray
    offset: float
    inliers: int
    squared_error: float


def select_ground_candidates(
    points: np.ndarray,
    config: GroundSegmentationConfig,
) -> np.ndarray:
    """Return indices in the plausible ground range/down ROI."""
    squared_range = points[:, 0] ** 2 + points[:, 2] ** 2
    down = -(points @ config.expected_up)
    mask = (
        (squared_range >= config.ground_candidate_min_range_m ** 2)
        & (squared_range <= config.ground_candidate_max_range_m ** 2)
        & (down >= config.ground_candidate_min_down_m)
        & (down <= config.ground_candidate_max_down_m)
    )
    return np.flatnonzero(mask)


def _orient_plane(
    normal: np.ndarray,
    offset: float,
    config: GroundSegmentationConfig,
) -> Optional[Tuple[np.ndarray, float]]:
    norm = float(np.linalg.norm(normal))
    if norm < 1.0e-9:
        return None
    normal = np.asarray(normal, dtype=np.float32) / norm
    offset = float(offset) / norm
    if float(normal @ config.expected_up) < 0.0:
        normal = -normal
        offset = -offset
    minimum_alignment = math.cos(math.radians(config.max_ground_tilt_deg))
    if float(normal @ config.expected_up) < minimum_alignment:
        return None
    if not (
        config.min_plane_distance_from_origin_m
        <= offset
        <= config.max_plane_distance_from_origin_m
    ):
        return None
    return normal, offset


def _evaluate(
    candidates: np.ndarray,
    normal: np.ndarray,
    offset: float,
    config: GroundSegmentationConfig,
) -> Optional[PlaneModel]:
    distances = np.abs(candidates @ normal + offset)
    mask = distances <= config.ransac_distance_threshold_m
    count = int(np.count_nonzero(mask))
    ratio = count / len(candidates) if len(candidates) else 0.0
    if (
        count < config.min_ground_inliers
        or ratio < config.min_ground_inlier_ratio
    ):
        return None
    return PlaneModel(
        normal=normal,
        offset=offset,
        inliers=count,
        squared_error=float(np.square(distances[mask]).sum()),
    )


def fit_ground_plane(
    points: np.ndarray,
    config: GroundSegmentationConfig,
    rng: np.random.Generator,
    candidate_indices: Optional[np.ndarray] = None,
) -> Optional[PlaneModel]:
    """Fit the dominant valid road plane and refine it by PCA."""
    if candidate_indices is None:
        candidate_indices = select_ground_candidates(points, config)
    if candidate_indices.size < 3:
        return None
    candidates = points[candidate_indices]
    if candidate_indices.size > config.max_ransac_points:
        source = (
            np.arange(config.max_ransac_points, dtype=np.int64)
            * candidate_indices.size
            // config.max_ransac_points
        )
        sample = candidates[source]
    else:
        sample = candidates

    # Generate every 3-point hypothesis in one array, then score all of them in
    # one matrix operation instead of 240 Python/BLAS dispatches per frame.
    triples = np.empty((0, 3), dtype=np.int64)
    while len(triples) < config.ransac_iterations:
        proposed = rng.integers(
            0,
            len(candidates),
            size=(config.ransac_iterations * 2, 3),
        )
        unique = (
            (proposed[:, 0] != proposed[:, 1])
            & (proposed[:, 0] != proposed[:, 2])
            & (proposed[:, 1] != proposed[:, 2])
        )
        triples = np.vstack((triples, proposed[unique]))
    triples = triples[:config.ransac_iterations]
    point_a = candidates[triples[:, 0]]
    normals = np.cross(
        candidates[triples[:, 1]] - point_a,
        candidates[triples[:, 2]] - point_a,
    )
    norms = np.linalg.norm(normals, axis=1)
    nondegenerate = norms >= 1.0e-9
    normals[nondegenerate] /= norms[nondegenerate, None]
    offsets = -np.einsum('ij,ij->i', normals, point_a)
    alignment = normals @ config.expected_up
    flip = alignment < 0.0
    normals[flip] *= -1.0
    offsets[flip] *= -1.0
    alignment[flip] *= -1.0
    valid = (
        nondegenerate
        & (alignment >= math.cos(math.radians(config.max_ground_tilt_deg)))
        & (offsets >= config.min_plane_distance_from_origin_m)
        & (offsets <= config.max_plane_distance_from_origin_m)
    )
    normals = normals[valid]
    offsets = offsets[valid]

    if len(normals) == 0:
        return None
    distances = np.abs(sample @ normals.T + offsets)
    inliers = distances <= config.ransac_distance_threshold_m
    counts = np.count_nonzero(inliers, axis=0)
    best_count = int(counts.max())
    tied = np.flatnonzero(counts == best_count)
    if len(tied) == 1:
        best_index = int(tied[0])
    else:
        tied_distances = distances[:, tied]
        tied_inliers = inliers[:, tied]
        tied_errors = np.sum(
            np.square(tied_distances) * tied_inliers,
            axis=0,
        )
        best_index = int(tied[int(np.argmin(tied_errors))])
    best_normal = normals[best_index].copy()
    best_offset = float(offsets[best_index])

    scaled_minimum = min(config.min_ground_inliers, len(sample))
    if (
        best_count < scaled_minimum
        or best_count / len(sample) < config.min_ground_inlier_ratio
    ):
        return None

    ransac_model = _evaluate(
        candidates, best_normal, best_offset, config)
    distances = np.abs(candidates @ best_normal + best_offset)
    inliers = candidates[distances <= config.ransac_distance_threshold_m]
    if len(inliers) < 3:
        return ransac_model

    centroid = inliers.mean(axis=0)
    covariance = (inliers - centroid).T @ (inliers - centroid)
    _, eigenvectors = np.linalg.eigh(covariance)
    refined_normal = eigenvectors[:, 0]
    oriented = _orient_plane(
        refined_normal, -float(refined_normal @ centroid), config)
    refined_model = None
    if oriented is not None:
        refined_model = _evaluate(candidates, *oriented, config)

    if refined_model is None:
        return ransac_model
    if ransac_model is None:
        return refined_model
    if refined_model.inliers > ransac_model.inliers:
        return refined_model
    if (
        refined_model.inliers == ransac_model.inliers
        and refined_model.squared_error < ransac_model.squared_error
    ):
        return refined_model
    return ransac_model


def classify_points(
    points: np.ndarray,
    plane: PlaneModel | Sequence[PlaneModel],
    config: GroundSegmentationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return obstacle and ground masks for all finite input points."""
    planes = (plane,) if isinstance(plane, PlaneModel) else tuple(plane)
    if not planes:
        raise ValueError('at least one ground plane is required')
    squared_range = points[:, 0] ** 2 + points[:, 2] ** 2
    signed_heights = np.column_stack([
        points @ model.normal + model.offset for model in planes
    ])
    positive_heights = np.where(signed_heights >= 0.0, signed_heights, np.inf)
    height = np.min(positive_heights, axis=1)
    obstacle_mask = (
        (squared_range >= config.obstacle_min_range_m ** 2)
        & (squared_range <= config.obstacle_max_range_m ** 2)
        & (height >= config.obstacle_min_height_m)
        & (height <= config.obstacle_max_height_m)
    )
    ground_mask = np.any(
        np.abs(signed_heights) <= config.ransac_distance_threshold_m,
        axis=1,
    )
    obstacle_mask &= ~ground_mask
    return obstacle_mask, ground_mask


def recover_ground_from_plane_history(
    points: np.ndarray,
    ground_mask: np.ndarray,
    current_planes: Sequence[PlaneModel],
    previous_planes: Sequence[PlaneModel],
    distance_threshold_m: float,
    max_normal_delta_deg: float,
    max_offset_delta_m: float,
) -> np.ndarray:
    """Recover ground lost to a short-lived compatible plane wobble.

    Only previous planes close to the current primary plane are considered.
    They are used solely to add ground points, never to compute obstacle
    height.  This distinction keeps a curb above the current road plane from
    being lowered by a stale plane while allowing borderline road returns to
    survive a one-frame RANSAC offset change.
    """
    recovered = np.asarray(ground_mask, dtype=bool).copy()
    if (
        len(points) == 0
        or not current_planes
        or not previous_planes
        or distance_threshold_m <= 0.0
    ):
        return recovered

    current = current_planes[0]
    cosine_limit = math.cos(math.radians(max(0.0, max_normal_delta_deg)))
    compatible = [
        model for model in previous_planes
        if float(np.dot(current.normal, model.normal)) >= cosine_limit
        and abs(current.offset - model.offset) <= max_offset_delta_m
    ]
    if not compatible:
        return recovered

    residual = np.min(np.column_stack([
        np.abs(points @ model.normal + model.offset)
        for model in compatible
    ]), axis=1)
    recovered |= residual <= distance_threshold_m
    return recovered


def grow_ground_region(
    points: np.ndarray,
    initial_ground_mask: np.ndarray,
    planes: Sequence[PlaneModel],
    config: GroundSegmentationConfig,
) -> np.ndarray:
    """Extend plane inliers over locally continuous, gently varying road.

    A global plane (or even two planes) cannot follow road crown, shallow
    drainage transitions, and gradual vertical curvature. Those surfaces are
    dense, so a radius outlier filter cannot distinguish them from obstacles.

    This filter projects the cloud onto a grid perpendicular to vehicle-up,
    starts from RANSAC ground cells, and grows through four-connected cells
    whose representative surface height changes by no more than a small step
    plus the configured local slope. Only points close to the accepted local
    surface are added; obstacle points sharing a cell with road therefore stay
    obstacles. A curb or vertical face cannot bridge the step gate.
    """
    initial = np.asarray(initial_ground_mask, dtype=bool)
    if (
        not config.ground_region_filter_enabled
        or len(points) == 0
        or not planes
        or not np.any(initial)
    ):
        return initial.copy()

    up = config.expected_up
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    axis_u = np.cross(up, reference)
    if float(np.linalg.norm(axis_u)) < 1.0e-6:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        axis_u = np.cross(up, reference)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(up, axis_u)
    axis_v /= np.linalg.norm(axis_v)

    horizontal = np.column_stack((points @ axis_u, points @ axis_v))
    squared_range = points[:, 0] ** 2 + points[:, 2] ** 2
    max_filter_range = max(
        config.ground_candidate_max_range_m,
        config.obstacle_max_range_m,
    )
    eligible_indices = np.flatnonzero(
        squared_range <= max_filter_range ** 2)
    if len(eligible_indices) == 0:
        return initial.copy()

    height = points @ up
    plane_residual = np.min(np.column_stack([
        np.abs(points @ model.normal + model.offset) for model in planes
    ]), axis=1)
    active_horizontal = horizontal[eligible_indices]
    keys = np.floor(
        active_horizontal / config.ground_region_grid_size_m).astype(np.int64)
    minimum_key = keys.min(axis=0)
    keys -= minimum_key
    span = keys.max(axis=0) + 1
    width = int(span[0])
    height_cells = int(span[1])
    cell_count = width * height_cells
    linear = keys[:, 1] * width + keys[:, 0]
    counts = np.bincount(linear, minlength=cell_count)

    # Sort once by cell and vehicle-up height. The low quintile represents the
    # road when one cell contains both road and obstacle returns, without a
    # Python loop over thousands of cells.
    active_height = height[eligible_indices]
    order = np.lexsort((active_height, linear))
    starts = np.cumsum(counts) - counts
    populated = np.flatnonzero(
        counts >= config.ground_region_min_points)
    ranks = (counts[populated] - 1) // 5
    selected_order_positions = starts[populated] + ranks
    selected_active_indices = order[selected_order_positions]
    selected_point_indices = eligible_indices[selected_active_indices]

    cell_height = np.full(cell_count, np.nan, dtype=np.float32)
    cell_plane_residual = np.full(cell_count, np.inf, dtype=np.float32)
    cell_height[populated] = height[selected_point_indices]
    cell_plane_residual[populated] = plane_residual[selected_point_indices]

    active_ground = initial[eligible_indices]
    ground_linear = linear[active_ground]
    ground_counts = np.bincount(ground_linear, minlength=cell_count)
    seed = ground_counts > 0
    if np.any(seed):
        ground_height_sum = np.bincount(
            ground_linear,
            weights=active_height[active_ground],
            minlength=cell_count,
        )
        cell_height[seed] = ground_height_sum[seed] / ground_counts[seed]
        cell_plane_residual[seed] = 0.0

    valid = (
        np.isfinite(cell_height)
        & (cell_plane_residual
           <= config.ground_region_max_plane_residual_m)
    )
    accepted = (seed & valid).reshape(height_cells, width)
    valid_grid = valid.reshape(height_cells, width)
    height_grid = cell_height.reshape(height_cells, width)
    neighbor_limit = (
        config.ground_region_max_step_m
        + math.tan(math.radians(config.ground_region_max_slope_deg))
        * config.ground_region_grid_size_m
    )

    # Vectorized four-connected flood fill. The maximum useful path in a
    # rectangular grid is bounded by width + height, and convergence normally
    # occurs much sooner because RANSAC seeds cover the near road densely.
    for _ in range(width + height_cells):
        candidates = np.zeros_like(accepted)
        vertical_continuity = (
            np.abs(height_grid[1:, :] - height_grid[:-1, :])
            <= neighbor_limit
        )
        candidates[1:, :] |= accepted[:-1, :] & vertical_continuity
        candidates[:-1, :] |= accepted[1:, :] & vertical_continuity
        horizontal_continuity = (
            np.abs(height_grid[:, 1:] - height_grid[:, :-1])
            <= neighbor_limit
        )
        candidates[:, 1:] |= accepted[:, :-1] & horizontal_continuity
        candidates[:, :-1] |= accepted[:, 1:] & horizontal_continuity
        candidates &= valid_grid & ~accepted
        if not np.any(candidates):
            break
        accepted |= candidates

    accepted_cells = accepted.reshape(-1)
    active_delta = np.abs(active_height - cell_height[linear])
    active_expanded = (
        accepted_cells[linear]
        & np.isfinite(active_delta)
        & (active_delta <= config.ground_region_point_tolerance_m)
    )
    expanded = initial.copy()
    expanded[eligible_indices] |= active_expanded
    return expanded


def fit_ground_planes(
    points: np.ndarray,
    config: GroundSegmentationConfig,
    rng: np.random.Generator,
    candidate_indices: Optional[np.ndarray] = None,
) -> Tuple[PlaneModel, ...]:
    """Fit the dominant road plane and one distinct residual slope plane."""
    if candidate_indices is None:
        candidate_indices = select_ground_candidates(points, config)
    primary = fit_ground_plane(points, config, rng, candidate_indices)
    if primary is None:
        return ()
    planes = [primary]
    if config.secondary_ransac_iterations <= 0:
        return tuple(planes)

    candidates = points[candidate_indices]
    primary_distance = np.abs(candidates @ primary.normal + primary.offset)
    residual = candidates[
        primary_distance > config.ransac_distance_threshold_m]
    if len(residual) < config.min_ground_inliers:
        return tuple(planes)

    secondary_config = replace(
        config,
        ransac_iterations=config.secondary_ransac_iterations,
        min_ground_inlier_ratio=max(
            config.min_ground_inlier_ratio,
            config.secondary_min_ground_inlier_ratio,
        ),
        max_plane_distance_from_origin_m=(
            config.secondary_max_plane_distance_from_origin_m),
        secondary_ransac_iterations=0,
    )
    secondary = fit_ground_plane(
        residual,
        secondary_config,
        rng,
        np.arange(len(residual), dtype=np.int64),
    )
    if secondary is None:
        return tuple(planes)

    alignment = float(np.clip(
        primary.normal @ secondary.normal, -1.0, 1.0))
    normal_delta_deg = math.degrees(math.acos(alignment))
    if normal_delta_deg < config.secondary_min_normal_delta_deg:
        return tuple(planes)
    planes.append(secondary)
    return tuple(planes)


def voxel_first_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keep the first point in each 3-D voxel, preserving source order."""
    if voxel_size <= 0.0 or len(points) == 0:
        return np.arange(len(points), dtype=np.int64)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return np.sort(indices)


def radius_outlier_indices(
    points: np.ndarray,
    radius_m: float,
    min_neighbors: int,
) -> np.ndarray:
    """Return points with local support in either of two shifted 3-D grids."""
    if radius_m <= 0.0 or min_neighbors <= 0 or len(points) == 0:
        return np.arange(len(points), dtype=np.int64)
    if len(points) <= min_neighbors:
        return np.empty(0, dtype=np.int64)
    keep = np.zeros(len(points), dtype=bool)
    # ponytail: shifted support grids avoid a measured 159 ms exact-radius
    # search. Replace with native/CUDA radius search only if field data shows
    # grid-boundary misses; this path measured about 6 ms for 22k points.
    for offset in (0.0, radius_m * 0.5):
        quantized = np.floor((points + offset) / radius_m).astype(np.int64)
        quantized -= quantized.min(axis=0)
        spans = quantized.max(axis=0) + 1
        keys = (
            (quantized[:, 0] * spans[1] + quantized[:, 1]) * spans[2]
            + quantized[:, 2]
        )
        _, inverse, counts = np.unique(
            keys, return_inverse=True, return_counts=True)
        # Counts include the point itself, while the parameter does not.
        keep |= counts[inverse] > min_neighbors
    return np.flatnonzero(keep)


def range_adaptive_radius_outlier_indices(
    points: np.ndarray,
    near_radius_m: float,
    near_min_neighbors: int,
    far_start_range_m: float,
    far_radius_m: float,
    far_min_neighbors: int,
) -> np.ndarray:
    """Filter sparse returns with a wider support radius at long range.

    A fixed metric radius becomes progressively stricter in image space as
    depth increases. Splitting at a configured horizontal range preserves the
    tight near-field speckle rejection while allowing the wider point spacing
    of real distant objects.
    """
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    if (
        far_start_range_m <= 0.0
        or far_radius_m <= 0.0
        or far_min_neighbors <= 0
    ):
        return radius_outlier_indices(
            points, near_radius_m, near_min_neighbors)

    ranges = np.hypot(points[:, 0], points[:, 2])
    far_mask = ranges >= far_start_range_m
    near_indices = np.flatnonzero(~far_mask)
    far_indices = np.flatnonzero(far_mask)
    selected = []
    if len(near_indices):
        selected.append(near_indices[radius_outlier_indices(
            points[near_indices], near_radius_m, near_min_neighbors)])
    if len(far_indices):
        selected.append(far_indices[radius_outlier_indices(
            points[far_indices], far_radius_m, far_min_neighbors)])
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(selected))


def project_points_to_vehicle_bev(
    points: np.ndarray,
    vehicle_forward_vector: Optional[np.ndarray] = None,
    vehicle_left_vector: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project sensor-frame points onto vehicle forward/left axes.

    The old implementation assumed every input used an unpitched front
    optical frame (+Z forward, -X left).  That is not true for the pitched,
    rear-facing camera.  Supplying the two calibrated vehicle axes keeps the
    geometry common to both cameras without first transforming the full cloud.
    """
    values = np.asarray(points)
    if len(values) == 0:
        return np.empty((0, 2), dtype=np.float64)
    forward = np.asarray(
        [0.0, 0.0, 1.0]
        if vehicle_forward_vector is None else vehicle_forward_vector,
        dtype=np.float64,
    ).reshape(-1)
    left = np.asarray(
        [-1.0, 0.0, 0.0]
        if vehicle_left_vector is None else vehicle_left_vector,
        dtype=np.float64,
    ).reshape(-1)
    if forward.size != 3 or left.size != 3:
        raise ValueError('vehicle BEV vectors must contain exactly 3 values')
    forward_norm = float(np.linalg.norm(forward))
    left_norm = float(np.linalg.norm(left))
    if forward_norm < 1.0e-9 or left_norm < 1.0e-9:
        raise ValueError('vehicle BEV vectors must be non-zero')
    forward /= forward_norm
    left /= left_norm
    if abs(float(forward @ left)) > 0.05:
        raise ValueError('vehicle forward/left vectors must be orthogonal')
    return np.column_stack((values @ forward, values @ left))


def extrapolate_planar_odometry(
    odom_x_m: float,
    odom_y_m: float,
    odom_yaw_rad: float,
    body_velocity_x_mps: float,
    body_velocity_y_mps: float,
    yaw_rate_radps: float,
    delta_time_sec: float,
) -> Tuple[float, float, float]:
    """Extrapolate a planar odometry pose with a constant body-frame twist.

    RGB-D stamps can lead the odometry callback available to this CPU-heavy
    node.  Predicting the pose to cloud time avoids disabling persistence or
    projecting a turning vehicle's rear observations with a stale yaw.
    """
    delta = float(delta_time_sec)
    angular = float(yaw_rate_radps)
    velocity_x = float(body_velocity_x_mps)
    velocity_y = float(body_velocity_y_mps)
    angle = angular * delta
    if abs(angular) < 1.0e-9:
        body_x = velocity_x * delta
        body_y = velocity_y * delta
    else:
        sine = math.sin(angle)
        one_minus_cosine = 1.0 - math.cos(angle)
        body_x = (
            sine / angular * velocity_x
            - one_minus_cosine / angular * velocity_y
        )
        body_y = (
            one_minus_cosine / angular * velocity_x
            + sine / angular * velocity_y
        )
    cosine_yaw = math.cos(odom_yaw_rad)
    sine_yaw = math.sin(odom_yaw_rad)
    world_x = (
        float(odom_x_m) + cosine_yaw * body_x - sine_yaw * body_y)
    world_y = (
        float(odom_y_m) + sine_yaw * body_x + cosine_yaw * body_y)
    yaw = math.atan2(
        math.sin(odom_yaw_rad + angle),
        math.cos(odom_yaw_rad + angle),
    )
    return world_x, world_y, yaw


def obstacle_bev_keys(
    points: np.ndarray,
    odom_x_m: float,
    odom_y_m: float,
    odom_yaw_rad: float,
    cell_size_m: float,
    vehicle_forward_vector: Optional[np.ndarray] = None,
    vehicle_left_vector: Optional[np.ndarray] = None,
    sensor_origin_vehicle_xy_m: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return packed odom-frame BEV cells for calibrated sensor points.

    Camera translation must be included when the vehicle yaws.  Otherwise the
    omitted lever arm rotates with the vehicle and a stationary obstacle can
    jump by more than one 10 cm history cell, especially for the rear camera.
    """
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    cell_size = max(0.01, float(cell_size_m))
    cosine = math.cos(odom_yaw_rad)
    sine = math.sin(odom_yaw_rad)
    vehicle_bev = project_points_to_vehicle_bev(
        points, vehicle_forward_vector, vehicle_left_vector)
    origin = np.asarray(
        [0.0, 0.0]
        if sensor_origin_vehicle_xy_m is None
        else sensor_origin_vehicle_xy_m,
        dtype=np.float64,
    ).reshape(-1)
    if origin.size != 2:
        raise ValueError(
            'sensor_origin_vehicle_xy_m must contain exactly 2 values')
    forward = vehicle_bev[:, 0] + origin[0]
    left = vehicle_bev[:, 1] + origin[1]
    world_x = odom_x_m + cosine * forward - sine * left
    world_y = odom_y_m + sine * forward + cosine * left
    cell_x = np.floor(world_x / cell_size).astype(np.int64)
    cell_y = np.floor(world_y / cell_size).astype(np.int64)
    return ((cell_x << 32) ^ (cell_y & np.int64(0xFFFFFFFF))).astype(
        np.int64, copy=False)


def expand_bev_keys(keys: np.ndarray, neighbor_cells: int) -> np.ndarray:
    """Expand packed BEV cells to tolerate odometry and sampling jitter."""
    unique = np.unique(np.asarray(keys, dtype=np.int64))
    radius = max(0, int(neighbor_cells))
    if len(unique) == 0 or radius == 0:
        return unique
    cell_x = unique >> 32
    cell_y = (unique & np.int64(0xFFFFFFFF)).astype(
        np.uint32).view(np.int32).astype(np.int64)
    expanded = []
    for delta_x in range(-radius, radius + 1):
        for delta_y in range(-radius, radius + 1):
            expanded.append(
                ((cell_x + delta_x) << 32)
                ^ ((cell_y + delta_y) & np.int64(0xFFFFFFFF))
            )
    return np.unique(np.concatenate(expanded))


def temporal_obstacle_persistence_indices(
    points: np.ndarray,
    odom_x_m: float,
    odom_y_m: float,
    odom_yaw_rad: float,
    history: Sequence[np.ndarray],
    cell_size_m: float,
    min_previous_hits: int,
    near_bypass_range_m: float,
    vehicle_forward_vector: Optional[np.ndarray] = None,
    vehicle_left_vector: Optional[np.ndarray] = None,
    sensor_origin_vehicle_xy_m: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep near or odom-stabilized persistent obstacle returns.

    Far RGB-D road artefacts move in world coordinates as the vehicle turns,
    while a curb, cone, vehicle, or wall repeatedly occupies the same BEV
    cells.  Near returns bypass temporal confirmation to avoid adding latency
    to immediately hazardous obstacles.

    Returns the selected point indices and the unexpanded current-frame keys;
    callers store an expanded copy of the latter in their bounded history.
    """
    if len(points) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    keys = obstacle_bev_keys(
        points,
        odom_x_m,
        odom_y_m,
        odom_yaw_rad,
        cell_size_m,
        vehicle_forward_vector,
        vehicle_left_vector,
        sensor_origin_vehicle_xy_m,
    )
    required = max(0, int(min_previous_hits))
    if required == 0 or len(history) < required:
        return np.arange(len(points), dtype=np.int64), keys
    hits = np.zeros(len(points), dtype=np.uint8)
    for previous_keys in history:
        hits += np.isin(keys, previous_keys)
    sensor_bev = project_points_to_vehicle_bev(
        points, vehicle_forward_vector, vehicle_left_vector)
    near = np.linalg.norm(sensor_bev, axis=1) < max(
        0.0, float(near_bypass_range_m))
    return np.flatnonzero(near | (hits >= required)), keys


def horizontal_surface_filter_indices(
    obstacle_points: np.ndarray,
    source_points: np.ndarray,
    expected_up: np.ndarray,
    cell_size_m: float,
    min_component_points: int,
    min_up_alignment: float,
    max_plane_thickness_m: float,
    near_bypass_range_m: float,
    vertical_bin_size_m: float,
    vertical_min_points_per_bin: int,
    vertical_min_run_bins: int,
    vertical_min_occupied_bins: int,
    vertical_min_support_cells: int,
    vehicle_forward_vector: Optional[np.ndarray] = None,
    vehicle_left_vector: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reject far, flat obstacle components without a vertical face.

    Paint, shallow road undulation, and some stereo depth artefacts can form a
    dense and temporally persistent sheet above a global ground plane.  Such a
    sheet is not separable with a radius or history test.  This stage finds
    connected BEV components and suppresses only components that are strongly
    horizontal and thin.  A component is retained when the unclassified input
    cloud contains a vertically continuous face in multiple cells, which is
    how a curb top is distinguished from a painted line.

    Near points bypass the test.  Sparse or non-planar components also pass so
    uncertainty cannot erase a cone, pedestrian, vehicle, or wall.
    """
    count = len(obstacle_points)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    cell_size = max(0.02, float(cell_size_m))
    near_limit = max(0.0, float(near_bypass_range_m))
    obstacle_bev = project_points_to_vehicle_bev(
        obstacle_points, vehicle_forward_vector, vehicle_left_vector)
    ranges = np.linalg.norm(obstacle_bev, axis=1)
    keep = ranges < near_limit
    far_indices = np.flatnonzero(~keep)
    if len(far_indices) == 0:
        return np.arange(count, dtype=np.int64)

    far_points = obstacle_points[far_indices]
    cells = np.floor(obstacle_bev[far_indices] / cell_size).astype(np.int64)
    unique_cells, inverse = np.unique(
        cells, axis=0, return_inverse=True)
    cell_lookup = {
        (int(cell[0]), int(cell[1])): index
        for index, cell in enumerate(unique_cells)
    }
    unvisited = set(range(len(unique_cells)))
    components = []
    while unvisited:
        seed = unvisited.pop()
        stack = [seed]
        component = [seed]
        while stack:
            current = stack.pop()
            cell_x, cell_z = unique_cells[current]
            for delta_x in (-1, 0, 1):
                for delta_z in (-1, 0, 1):
                    neighbour = cell_lookup.get((
                        int(cell_x + delta_x),
                        int(cell_z + delta_z),
                    ))
                    if neighbour is not None and neighbour in unvisited:
                        unvisited.remove(neighbour)
                        stack.append(neighbour)
                        component.append(neighbour)
        components.append(np.asarray(component, dtype=np.int64))

    order = np.argsort(inverse, kind='stable')
    cell_counts = np.bincount(inverse, minlength=len(unique_cells))
    cell_starts = np.cumsum(cell_counts) - cell_counts
    up = np.asarray(expected_up, dtype=np.float64)
    up /= max(float(np.linalg.norm(up)), 1.0e-9)
    source_bev = project_points_to_vehicle_bev(
        source_points, vehicle_forward_vector, vehicle_left_vector)
    source_cells = np.floor(source_bev / cell_size).astype(np.int64)
    source_keys = (
        (source_cells[:, 0] << 32)
        ^ (source_cells[:, 1] & np.int64(0xFFFFFFFF))
    )
    source_heights = source_points @ up
    required_points = max(3, int(min_component_points))
    alignment_limit = min(1.0, max(0.0, float(min_up_alignment)))
    thickness_limit = max(0.001, float(max_plane_thickness_m))
    bin_size = max(0.005, float(vertical_bin_size_m))
    points_per_bin = max(1, int(vertical_min_points_per_bin))
    required_run = max(2, int(vertical_min_run_bins))
    required_occupied = max(2, int(vertical_min_occupied_bins))
    required_support_cells = max(1, int(vertical_min_support_cells))

    for component_cells in components:
        point_indices = np.concatenate([
            order[cell_starts[cell_index]:
                  cell_starts[cell_index] + cell_counts[cell_index]]
            for cell_index in component_cells
        ])
        if len(point_indices) < required_points:
            keep[far_indices[point_indices]] = True
            continue
        component_points = far_points[point_indices]
        centered = component_points - component_points.mean(axis=0)
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal_up_alignment = abs(float(eigenvectors[:, 0] @ up))
        thickness = math.sqrt(max(0.0, float(eigenvalues[0])))
        is_horizontal_sheet = (
            normal_up_alignment >= alignment_limit
            and thickness <= thickness_limit
        )
        if not is_horizontal_sheet:
            keep[far_indices[point_indices]] = True
            continue

        component_xy = unique_cells[component_cells]
        component_keys = (
            (component_xy[:, 0] << 32)
            ^ (component_xy[:, 1] & np.int64(0xFFFFFFFF))
        )
        source_mask = np.isin(source_keys, component_keys)
        selected_source_keys = source_keys[source_mask]
        selected_source_heights = source_heights[source_mask]
        support_cells = 0
        if len(selected_source_keys):
            _, source_inverse = np.unique(
                selected_source_keys, return_inverse=True)
            for source_cell_index in range(source_inverse.max() + 1):
                heights = selected_source_heights[
                    source_inverse == source_cell_index]
                height_bins, height_counts = np.unique(
                    np.floor(heights / bin_size).astype(np.int64),
                    return_counts=True,
                )
                occupied = height_bins[height_counts >= points_per_bin]
                if len(occupied) < required_occupied:
                    continue
                best_span = current_span = 1
                current_occupied = best_occupied = 1
                for difference in np.diff(occupied):
                    if difference <= 2:
                        current_span += int(difference)
                        current_occupied += 1
                    else:
                        current_span = 1
                        current_occupied = 1
                    if (
                        current_span > best_span
                        or (
                            current_span == best_span
                            and current_occupied > best_occupied
                        )
                    ):
                        best_span = current_span
                        best_occupied = current_occupied
                if (
                    best_span >= required_run
                    and best_occupied >= required_occupied
                ):
                    support_cells += 1
                    if support_cells >= required_support_cells:
                        break
        if support_cells >= required_support_cells:
            keep[far_indices[point_indices]] = True

    return np.flatnonzero(keep)


def range_residual_summary(
    points: np.ndarray,
    planes: Sequence[PlaneModel],
    max_range_m: float,
    bin_width_m: float = 2.0,
) -> str:
    """Summarize nearest-plane residuals in horizontal range bins."""
    if len(points) == 0 or not planes or max_range_m <= 0.0:
        return 'none'
    ranges = np.hypot(points[:, 0], points[:, 2])
    residuals = np.min(np.column_stack([
        np.abs(points @ model.normal + model.offset) for model in planes
    ]), axis=1)
    parts = []
    lower = 0.0
    while lower < max_range_m:
        upper = min(lower + bin_width_m, max_range_m)
        mask = (ranges >= lower) & (
            ranges <= upper if upper == max_range_m else ranges < upper)
        values = residuals[mask]
        if len(values):
            p50, p90 = np.percentile(values, [50.0, 90.0])
            parts.append(
                f'{lower:.0f}-{upper:.0f}m:n={len(values)},'
                f'p50={p50 * 100.0:.1f}cm,p90={p90 * 100.0:.1f}cm')
        lower = upper
    return '|'.join(parts) if parts else 'none'
