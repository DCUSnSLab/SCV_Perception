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
    ransac_iterations: int = 160
    max_ransac_points: int = 30000
    ransac_distance_threshold_m: float = 0.05
    max_ground_tilt_deg: float = 25.0
    min_ground_inliers: int = 300
    min_ground_inlier_ratio: float = 0.03
    secondary_ransac_iterations: int = 60
    secondary_min_ground_inlier_ratio: float = 0.20
    secondary_min_normal_delta_deg: float = 4.0
    secondary_max_plane_distance_from_origin_m: float = 2.5
    ground_candidate_min_range_m: float = 0.4
    ground_candidate_max_range_m: float = 8.0
    ground_candidate_min_down_m: float = 0.15
    ground_candidate_max_down_m: float = 2.5
    min_plane_distance_from_origin_m: float = 0.15
    max_plane_distance_from_origin_m: float = 2.5
    obstacle_min_height_m: float = 0.10
    obstacle_max_height_m: float = 2.0
    obstacle_min_range_m: float = 0.25
    obstacle_max_range_m: float = 8.0
    obstacle_voxel_size_m: float = 0.0

    def normalized(self) -> 'GroundSegmentationConfig':
        """Return a validated, normalized copy of this configuration."""
        up = np.asarray(self.expected_up, dtype=np.float64).reshape(-1)
        if up.size != 3:
            raise ValueError('expected_up_vector must contain exactly 3 values')
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
    normal = np.asarray(normal, dtype=np.float64) / norm
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
    if count < config.min_ground_inliers or ratio < config.min_ground_inlier_ratio:
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
) -> Optional[PlaneModel]:
    """Fit the dominant valid road plane and refine it by PCA."""
    candidate_indices = select_ground_candidates(points, config)
    if candidate_indices.size < 3:
        return None
    if candidate_indices.size > config.max_ransac_points:
        source = (
            np.arange(config.max_ransac_points, dtype=np.int64)
            * candidate_indices.size
            // config.max_ransac_points
        )
        sample_indices = candidate_indices[source]
    else:
        sample_indices = candidate_indices
    sample = points[sample_indices]

    # Generate valid 3-point hypotheses in one array, then evaluate them in
    # matrix batches. This keeps the 240-hypothesis robustness of the C++ node
    # without 240 Python/BLAS dispatches per camera frame.
    triples = np.empty((0, 3), dtype=np.int64)
    while len(triples) < config.ransac_iterations:
        proposed = rng.integers(
            0,
            len(sample),
            size=(config.ransac_iterations * 2, 3),
        )
        unique = (
            (proposed[:, 0] != proposed[:, 1])
            & (proposed[:, 0] != proposed[:, 2])
            & (proposed[:, 1] != proposed[:, 2])
        )
        triples = np.vstack((triples, proposed[unique]))
    triples = triples[:config.ransac_iterations]
    point_a = sample[triples[:, 0]]
    normals = np.cross(
        sample[triples[:, 1]] - point_a,
        sample[triples[:, 2]] - point_a,
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

    best_normal = None
    best_offset = 0.0
    best_count = 0
    best_error = math.inf
    batch_size = 32
    for start in range(0, len(normals), batch_size):
        batch_normals = normals[start:start + batch_size]
        batch_offsets = offsets[start:start + batch_size]
        distances = np.abs(sample @ batch_normals.T + batch_offsets)
        inliers = distances <= config.ransac_distance_threshold_m
        counts = np.count_nonzero(inliers, axis=0)
        errors = np.sum(np.square(distances) * inliers, axis=0)
        for index in range(len(batch_normals)):
            count = int(counts[index])
            error = float(errors[index])
            if count > best_count or (
                count == best_count and error < best_error
            ):
                best_normal = batch_normals[index].copy()
                best_offset = float(batch_offsets[index])
                best_count = count
                best_error = error

    scaled_minimum = min(config.min_ground_inliers, len(sample))
    if (
        best_normal is None
        or best_count < scaled_minimum
        or best_count / len(sample) < config.min_ground_inlier_ratio
    ):
        return None

    candidates = points[candidate_indices]
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
) -> Tuple[PlaneModel, ...]:
    """Fit the dominant road plane and one distinct residual slope plane."""
    primary = fit_ground_plane(points, config, rng)
    if primary is None:
        return ()
    planes = [primary]
    if config.secondary_ransac_iterations <= 0:
        return tuple(planes)

    candidate_indices = select_ground_candidates(points, config)
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
    secondary = fit_ground_plane(residual, secondary_config, rng)
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
