"""Shared 3-D box and transform geometry for observation tracking."""

import math

import numpy as np


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def box_corners_3d(box):
    """Return eight corners for [x, y, z, length, width, height, yaw]."""
    x, y, z, length, width, height, yaw = map(float, box[:7])
    local = np.array([
        [sx * length * 0.5, sy * width * 0.5, sz * height * 0.5]
        for sz in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sx in (-1.0, 1.0)
    ], dtype=np.float32)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.array(
        [[cosine, -sine], [sine, cosine]], dtype=np.float32)
    local[:, :2] = local[:, :2] @ rotation.T
    local += np.array([x, y, z], dtype=np.float32)
    return local


def transform_points(points, target_from_source):
    points = np.asarray(points, dtype=np.float32)
    transform = np.asarray(target_from_source, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_box(box, camera_from_source, intrinsics, image_size,
                min_depth=0.10, min_roi_pixels=4.0):
    """Project a 3-D box and return its clipped image rectangle or None."""
    corners_camera = transform_points(
        box_corners_3d(box), camera_from_source)
    valid = corners_camera[:, 2] > float(min_depth)
    if not np.any(valid):
        return None
    points = corners_camera[valid]
    fx, fy, cx, cy = map(float, intrinsics)
    u = fx * points[:, 0] / points[:, 2] + cx
    v = fy * points[:, 1] / points[:, 2] + cy
    raw = np.array([u.min(), v.min(), u.max(), v.max()], dtype=np.float64)
    width, height = map(int, image_size)
    clipped = np.array([
        np.clip(raw[0], 0.0, width - 1.0),
        np.clip(raw[1], 0.0, height - 1.0),
        np.clip(raw[2], 0.0, width - 1.0),
        np.clip(raw[3], 0.0, height - 1.0),
    ])
    if (clipped[2] - clipped[0] < min_roi_pixels or
            clipped[3] - clipped[1] < min_roi_pixels):
        return None
    return tuple(float(value) for value in clipped)
