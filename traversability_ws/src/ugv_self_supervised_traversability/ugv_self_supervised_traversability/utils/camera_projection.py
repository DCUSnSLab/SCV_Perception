"""Pinhole projection, polygon rasterization, and depth visibility tests."""

from typing import Optional, Tuple

import numpy as np


def validate_intrinsics(k: np.ndarray) -> Tuple[float, float, float, float]:
    """Validate a CameraInfo K matrix and return fx, fy, cx, cy."""
    matrix = np.asarray(k, dtype=np.float64).reshape(3, 3)
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    if not np.all(np.isfinite([fx, fy, cx, cy])) or fx <= 0.0 or fy <= 0.0:
        raise ValueError('invalid camera intrinsics')
    return float(fx), float(fy), float(cx), float(cy)


def project_points(
    camera_points: np.ndarray,
    k: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project Nx3 optical-frame points and return pixels, depth, validity."""
    points = np.asarray(camera_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('camera_points must have shape Nx3')
    fx, fy, cx, cy = validate_intrinsics(k)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 0.0)
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    pixels[valid, 0] = fx * points[valid, 0] / z[valid] + cx
    pixels[valid, 1] = fy * points[valid, 1] / z[valid] + cy
    if image_shape is not None:
        height, width = image_shape
        valid &= ((pixels[:, 0] >= 0.0) & (pixels[:, 0] < width) &
                  (pixels[:, 1] >= 0.0) & (pixels[:, 1] < height))
    return pixels, z.copy(), valid


def rasterize_camera_polygon(
    camera_points: np.ndarray,
    k: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize a convex camera-space polygon and estimate per-pixel Z depth.

    The quadrilateral is split into triangles. Inverse depth is barycentrically
    interpolated, which is perspective-correct for a planar footprint.
    """
    points = np.asarray(camera_points, dtype=np.float64)
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    depth = np.full((height, width), np.nan, dtype=np.float32)
    if points.shape[0] < 3 or np.any(points[:, 2] <= 0.0):
        return mask, depth
    pixels, z_values, _ = project_points(points, k)
    triangles = [(0, index, index + 1) for index in range(1, points.shape[0] - 1)]
    for indices in triangles:
        uv = pixels[list(indices)]
        if not np.isfinite(uv).all():
            continue
        x0 = max(0, int(np.floor(np.min(uv[:, 0]))))
        x1 = min(width - 1, int(np.ceil(np.max(uv[:, 0]))))
        y0 = max(0, int(np.floor(np.min(uv[:, 1]))))
        y1 = min(height - 1, int(np.ceil(np.max(uv[:, 1]))))
        if x0 > x1 or y0 > y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        a, b, c = uv
        denominator = ((b[1] - c[1]) * (a[0] - c[0]) +
                       (c[0] - b[0]) * (a[1] - c[1]))
        if abs(denominator) < 1e-9:
            continue
        w0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / denominator
        w1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
        inverse_z = (w0 / z_values[indices[0]] + w1 / z_values[indices[1]] +
                     w2 / z_values[indices[2]])
        triangle_depth = np.where(inside & (inverse_z > 0.0), 1.0 / inverse_z, np.nan)
        view = depth[y0:y1 + 1, x0:x1 + 1]
        update = np.isfinite(triangle_depth) & (~np.isfinite(view) | (triangle_depth < view))
        view[update] = triangle_depth[update]
        mask[y0:y1 + 1, x0:x1 + 1][update] = 1
    return mask, depth


def depth_visibility_mask(
    projected_depth: np.ndarray,
    measured_depth: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Accept projected pixels whose measured and expected metric depths agree."""
    if tolerance < 0.0 or projected_depth.shape != measured_depth.shape:
        raise ValueError('invalid tolerance or depth image shape mismatch')
    valid = (np.isfinite(projected_depth) & (projected_depth > 0.0) &
             np.isfinite(measured_depth) & (measured_depth > 0.0))
    return valid & (np.abs(projected_depth - measured_depth) <= tolerance)


def convert_depth_to_meters(depth: np.ndarray, encoding: str) -> np.ndarray:
    """Convert common ROS depth encodings to float32 meters."""
    if encoding in ('16UC1', 'mono16'):
        return depth.astype(np.float32) * 0.001
    if encoding == '32FC1':
        return depth.astype(np.float32)
    raise ValueError('unsupported depth encoding: ' + encoding)
