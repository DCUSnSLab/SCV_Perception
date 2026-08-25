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
    obstacle_voxel_size_m: float

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
            obstacle_voxel_size_m=max(
                0.0, float(self.obstacle_voxel_size_m)),
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
