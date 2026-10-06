"""Geometry-only terrain labels for gravity-aligned depth point clouds."""

from dataclasses import dataclass

import numpy as np


UNKNOWN = 0
GROUND = 1
ELEVATED_GROUND = 2
CURB = 3
OBSTACLE = 4

GEOMETRY_LABELS = {
    UNKNOWN: 'unknown',
    GROUND: 'ground',
    ELEVATED_GROUND: 'elevated_ground',
    CURB: 'curb',
    OBSTACLE: 'obstacle',
}

FUSED_LABELS = {
    0: 'unknown',
    1: 'road',
    2: 'sidewalk',
    3: 'curb',
    4: 'obstacle',
}


@dataclass(frozen=True)
class GeometryConfig:
    """Thresholds for a small-UGV depth geometry baseline."""

    resolution: float = 0.05
    x_min: float = -1.0
    x_max: float = 12.0
    y_min: float = -5.0
    y_max: float = 5.0
    z_min: float = -1.5
    z_max: float = 1.5
    expected_ground_z: float = -0.9
    expected_ground_tolerance: float = 0.45
    plane_tolerance: float = 0.05
    max_ground_slope_deg: float = 20.0
    curb_min_height: float = 0.05
    curb_max_height: float = 0.25
    obstacle_min_height: float = 0.28
    ransac_iterations: int = 64
    min_plane_points: int = 40


@dataclass
class GeometryResult:
    labels: np.ndarray
    confidence: np.ndarray
    plane: np.ndarray
    valid: np.ndarray


def classify_depth_geometry(points, config=None, seed=7):
    """Classify points using a ground plane and 5 cm height transitions.

    Input coordinates must be gravity aligned with +Z up. The returned labels
    are geometry evidence, not semantic road/sidewalk truth.
    """
    cfg = config or GeometryConfig()
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('points must have shape (N, 3)')
    labels = np.zeros(len(xyz), dtype=np.uint8)
    confidence = np.zeros(len(xyz), dtype=np.float32)
    roi = (np.isfinite(xyz).all(axis=1) &
           (xyz[:, 0] >= cfg.x_min) & (xyz[:, 0] < cfg.x_max) &
           (xyz[:, 1] >= cfg.y_min) & (xyz[:, 1] < cfg.y_max) &
           (xyz[:, 2] >= cfg.z_min) & (xyz[:, 2] <= cfg.z_max))
    roi_indices = np.flatnonzero(roi)
    if len(roi_indices) < cfg.min_plane_points:
        return GeometryResult(labels, confidence,
                              np.asarray((0.0, 0.0,
                                          cfg.expected_ground_z)), roi)

    roi_points = xyz[roi_indices]
    representatives = _lowest_grid_points(roi_points, cfg)
    plane = _fit_ground_plane(representatives, cfg, seed)
    height = xyz[:, 2] - (xyz[:, 0] * plane[0] +
                          xyz[:, 1] * plane[1] + plane[2])

    ground = roi & (height >= -cfg.plane_tolerance) & \
        (height <= cfg.plane_tolerance)
    elevated = roi & (height > cfg.plane_tolerance) & \
        (height <= cfg.curb_max_height)
    obstacle = roi & (height >= cfg.obstacle_min_height)
    labels[ground] = GROUND
    labels[elevated] = ELEVATED_GROUND
    labels[obstacle] = OBSTACLE
    confidence[ground] = np.clip(
        1.0 - np.abs(height[ground]) / cfg.plane_tolerance, 0.5, 1.0)
    confidence[elevated] = 0.65
    confidence[obstacle] = np.clip(
        0.65 + (height[obstacle] - cfg.obstacle_min_height), 0.65, 1.0)

    curb_cells = _curb_cells(roi_points, plane, cfg)
    if curb_cells:
        cell_x = np.floor((xyz[:, 0] - cfg.x_min) / cfg.resolution).astype(
            np.int64)
        cell_y = np.floor((xyz[:, 1] - cfg.y_min) / cfg.resolution).astype(
            np.int64)
        width = int(np.ceil((cfg.x_max - cfg.x_min) / cfg.resolution))
        cell_ids = cell_y * width + cell_x
        curb = (roi & np.isin(cell_ids, np.fromiter(
            curb_cells, dtype=np.int64)) &
            (height >= cfg.curb_min_height * 0.5) &
            (height <= cfg.curb_max_height + cfg.plane_tolerance))
        labels[curb] = CURB
        confidence[curb] = 0.8
    return GeometryResult(labels, confidence, plane, roi)


def fuse_cityscapes_geometry(cityscapes_labels, semantic_confidence,
                             geometry_labels, geometry_confidence):
    """Map Cityscapes road/sidewalk plus geometry overrides to custom IDs."""
    semantic = np.asarray(cityscapes_labels)
    fused = np.zeros(semantic.shape, dtype=np.uint8)
    fused[semantic == 0] = 1
    fused[semantic == 1] = 2
    fused_confidence = np.asarray(semantic_confidence, dtype=np.float32).copy()
    curb = geometry_labels == CURB
    obstacle = geometry_labels == OBSTACLE
    fused[curb] = 3
    fused[obstacle] = 4
    fused_confidence[curb | obstacle] = geometry_confidence[curb | obstacle]
    return fused, fused_confidence


def label_colors(labels):
    """Pack geometry/fused colors into PCL-compatible uint32 RGB."""
    palette = np.asarray([
        0x505050,  # unknown
        0x804080,  # road/ground
        0xF423E8,  # sidewalk/elevated ground
        0x00FFFF,  # curb
        0xFF0000,  # obstacle
    ], dtype=np.uint32)
    return palette[np.clip(np.asarray(labels), 0, len(palette) - 1)]


def _lowest_grid_points(points, cfg):
    width = int(np.ceil((cfg.x_max - cfg.x_min) / cfg.resolution))
    height = int(np.ceil((cfg.y_max - cfg.y_min) / cfg.resolution))
    ix = np.floor((points[:, 0] - cfg.x_min) / cfg.resolution).astype(int)
    iy = np.floor((points[:, 1] - cfg.y_min) / cfg.resolution).astype(int)
    flat = iy * width + ix
    lowest = np.full(width * height, np.inf, dtype=np.float32)
    np.minimum.at(lowest, flat, points[:, 2])
    occupied = np.flatnonzero(np.isfinite(lowest))
    cx = cfg.x_min + (occupied % width + 0.5) * cfg.resolution
    cy = cfg.y_min + (occupied // width + 0.5) * cfg.resolution
    return np.column_stack((cx, cy, lowest[occupied])).astype(np.float32)


def _fit_ground_plane(points, cfg, seed):
    fallback = np.asarray((0.0, 0.0, cfg.expected_ground_z), dtype=np.float32)
    # Seed the lower support surface. Including a broad elevated sidewalk can
    # make RANSAC explain a real step as a shallow ramp.
    near = ((points[:, 2] >=
             cfg.expected_ground_z - cfg.expected_ground_tolerance) &
            (points[:, 2] <=
             cfg.expected_ground_z + 2.0 * cfg.plane_tolerance))
    candidates = points[near]
    if len(candidates) < cfg.min_plane_points:
        return fallback
    rng = np.random.default_rng(seed)
    best_inliers = None
    best_score = -np.inf
    max_gradient = np.tan(np.radians(cfg.max_ground_slope_deg))
    design = np.column_stack((candidates[:, :2], np.ones(len(candidates))))
    for _ in range(cfg.ransac_iterations):
        chosen = rng.choice(len(candidates), 3, replace=False)
        sample = design[chosen]
        if np.linalg.matrix_rank(sample) < 3:
            continue
        plane = np.linalg.solve(sample, candidates[chosen, 2])
        if np.hypot(plane[0], plane[1]) > max_gradient:
            continue
        residual = np.abs(candidates[:, 2] - design @ plane)
        inliers = residual <= cfg.plane_tolerance
        count = int(inliers.sum())
        origin_error = abs(plane[2] - cfg.expected_ground_z)
        score = count - len(candidates) * 0.05 * (
            origin_error / cfg.expected_ground_tolerance)
        if count >= cfg.min_plane_points and score > best_score:
            best_score, best_inliers = score, inliers
    if best_inliers is None:
        return fallback
    refined, _, rank, _ = np.linalg.lstsq(
        design[best_inliers], candidates[best_inliers, 2], rcond=None)
    if rank < 3 or np.hypot(refined[0], refined[1]) > max_gradient:
        return fallback
    return refined.astype(np.float32)


def _curb_cells(points, plane, cfg):
    width = int(np.ceil((cfg.x_max - cfg.x_min) / cfg.resolution))
    height = int(np.ceil((cfg.y_max - cfg.y_min) / cfg.resolution))
    ix = np.floor((points[:, 0] - cfg.x_min) / cfg.resolution).astype(int)
    iy = np.floor((points[:, 1] - cfg.y_min) / cfg.resolution).astype(int)
    flat = iy * width + ix
    relative = points[:, 2] - (
        points[:, 0] * plane[0] + points[:, 1] * plane[1] + plane[2])
    low = np.full(width * height, np.inf, dtype=np.float32)
    high = np.full(width * height, -np.inf, dtype=np.float32)
    np.minimum.at(low, flat, relative)
    np.maximum.at(high, flat, relative)
    occupied = np.isfinite(low)
    surface = low.reshape(height, width)
    curb = (occupied & ((high - low) >= cfg.curb_min_height) &
            ((high - low) <= cfg.curb_max_height + cfg.plane_tolerance))
    curb = curb.reshape(height, width)
    for dy, dx in ((0, 1), (1, 0)):
        first = surface[:-dy or None, :-dx or None]
        second = surface[dy:, dx:]
        valid = np.isfinite(first) & np.isfinite(second)
        step = np.zeros_like(first)
        np.subtract(second, first, out=step, where=valid)
        step = np.abs(step)
        transition = valid & (step >= cfg.curb_min_height) & \
            (step <= cfg.curb_max_height)
        curb[:-dy or None, :-dx or None] |= transition
        curb[dy:, dx:] |= transition
    return set(np.flatnonzero(curb).tolist())
