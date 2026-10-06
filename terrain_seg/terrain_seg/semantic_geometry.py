"""Pure NumPy helpers for RGB-D semantic point-cloud generation."""

from typing import Iterable, Optional

import cv2
import numpy as np


CITYSCAPES_LABELS = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle',
)

# RGB colors from the Cityscapes train-id palette.
CITYSCAPES_PALETTE_RGB = np.asarray([
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32),
], dtype=np.uint8)


def depth_to_meters(depth: np.ndarray, encoding: str,
                    depth_scale: float = 0.001) -> np.ndarray:
    """Convert a ROS depth image array to float32 meters."""
    normalized = encoding.upper()
    if normalized in ('16UC1', 'MONO16'):
        return depth.astype(np.float32) * float(depth_scale)
    if normalized == '32FC1':
        return depth.astype(np.float32, copy=False)
    raise ValueError(
        f'unsupported depth encoding {encoding!r}; expected 16UC1 or 32FC1')


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Return a BGR visualization for a Cityscapes train-id image."""
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = (labels >= 0) & (labels < len(CITYSCAPES_PALETTE_RGB))
    output[valid] = CITYSCAPES_PALETTE_RGB[labels[valid]][:, ::-1]
    return output


def scale_intrinsics(camera_matrix: Iterable[float], source_size,
                     target_size):
    """Scale fx/fy/cx/cy when CameraInfo and depth resolutions differ."""
    matrix = np.asarray(tuple(camera_matrix), dtype=np.float64).reshape(3, 3)
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError('CameraInfo width and height must be positive')
    scale_x = target_width / float(source_width)
    scale_y = target_height / float(source_height)
    return (matrix[0, 0] * scale_x, matrix[1, 1] * scale_y,
            matrix[0, 2] * scale_x, matrix[1, 2] * scale_y)


def semantic_rgbd_points(
        depth_m: np.ndarray,
        labels: np.ndarray,
        confidence: np.ndarray,
        bgr: np.ndarray,
        intrinsics,
        stride: int = 2,
        min_depth: float = 0.25,
        max_depth: float = 12.0,
        min_confidence: float = 0.55,
        included_class_ids: Optional[Iterable[int]] = None):
    """Deproject selected semantic pixels into XYZ, RGB, label, confidence."""
    if depth_m.ndim != 2:
        raise ValueError('depth_m must be a two-dimensional array')
    height, width = depth_m.shape
    if labels.shape != (height, width) or confidence.shape != (height, width):
        raise ValueError('labels and confidence must match the depth shape')
    if bgr.shape != (height, width, 3):
        raise ValueError('bgr must match the depth height and width')
    if stride < 1:
        raise ValueError('stride must be at least one')

    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    z = depth_m[::stride, ::stride]
    sampled_labels = labels[::stride, ::stride]
    sampled_confidence = confidence[::stride, ::stride]
    sampled_bgr = bgr[::stride, ::stride]

    valid = (np.isfinite(z) & (z >= min_depth) & (z <= max_depth) &
             np.isfinite(sampled_confidence) &
             (sampled_confidence >= min_confidence))
    if included_class_ids is not None:
        class_ids = np.asarray(tuple(included_class_ids), dtype=np.int64)
        if class_ids.size:
            valid &= np.isin(sampled_labels, class_ids)

    z = z[valid].astype(np.float32, copy=False)
    u = cols[valid].astype(np.float32)
    v = rows[valid].astype(np.float32)
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError('camera focal lengths must be positive')

    xyz = np.empty((len(z), 3), dtype=np.float32)
    xyz[:, 0] = (u - cx) * z / fx
    xyz[:, 1] = (v - cy) * z / fy
    xyz[:, 2] = z

    colors = sampled_bgr[valid].astype(np.uint32)
    packed_rgb = ((colors[:, 2] << 16) |
                  (colors[:, 1] << 8) |
                  colors[:, 0]).astype(np.uint32)
    return (xyz, packed_rgb,
            sampled_labels[valid].astype(np.uint8),
            sampled_confidence[valid].astype(np.float32))


def resize_semantics(labels, confidence, bgr, target_shape):
    """Resize semantic/RGB arrays to the depth image shape."""
    height, width = target_shape
    if labels.shape != (height, width):
        labels = cv2.resize(
            labels, (width, height), interpolation=cv2.INTER_NEAREST)
        confidence = cv2.resize(
            confidence, (width, height), interpolation=cv2.INTER_LINEAR)
    if bgr.shape[:2] != (height, width):
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    return labels, confidence, bgr


def transform_xyz(xyz: np.ndarray, rotation: np.ndarray,
                  translation: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to an N-by-3 XYZ array."""
    return xyz @ np.asarray(rotation, dtype=np.float32).T + np.asarray(
        translation, dtype=np.float32)


def select_projected_semantics(
        xyz_camera: np.ndarray,
        labels: np.ndarray,
        confidence: np.ndarray,
        bgr: np.ndarray,
        intrinsics,
        min_confidence: float = 0.55,
        included_class_ids: Optional[Iterable[int]] = None,
        aligned_depth_m: Optional[np.ndarray] = None,
        occlusion_tolerance: float = 0.5,
        distortion_coeffs: Optional[Iterable[float]] = None):
    """Select 3D points whose camera projections have accepted semantics.

    Returns source-array indices plus sampled RGB, label and confidence. When
    aligned depth is supplied, points clearly behind or in front of the visible
    RGB-D surface are rejected; missing depth does not reject a point.
    """
    if xyz_camera.ndim != 2 or xyz_camera.shape[1] != 3:
        raise ValueError('xyz_camera must have shape (N, 3)')
    height, width = labels.shape
    if confidence.shape != (height, width) or bgr.shape != (height, width, 3):
        raise ValueError('semantic arrays must have matching dimensions')
    if aligned_depth_m is not None and aligned_depth_m.shape != (height, width):
        raise ValueError('aligned depth must match the semantic image shape')

    fx, fy, cx, cy = (float(value) for value in intrinsics)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError('camera focal lengths must be positive')
    z = xyz_camera[:, 2]
    finite_front = np.isfinite(xyz_camera).all(axis=1) & (z > 0.0)
    candidate_indices = np.flatnonzero(finite_front)
    if not len(candidate_indices):
        return _empty_projection_result()

    camera_points = xyz_camera[candidate_indices]
    u, v = project_pixels(
        camera_points, (fx, fy, cx, cy), distortion_coeffs)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    candidate_indices = candidate_indices[inside]
    u, v = u[inside], v[inside]
    if not len(candidate_indices):
        return _empty_projection_result()

    sampled_labels = labels[v, u]
    sampled_confidence = confidence[v, u]
    accepted = (np.isfinite(sampled_confidence) &
                (sampled_confidence >= min_confidence))
    if included_class_ids is not None:
        class_ids = np.asarray(tuple(included_class_ids), dtype=np.int64)
        if class_ids.size:
            accepted &= np.isin(sampled_labels, class_ids)
    if aligned_depth_m is not None:
        observed_depth = aligned_depth_m[v, u]
        has_depth = np.isfinite(observed_depth) & (observed_depth > 0.0)
        accepted &= (~has_depth | (
            np.abs(xyz_camera[candidate_indices, 2] - observed_depth) <=
            occlusion_tolerance))

    candidate_indices = candidate_indices[accepted]
    u, v = u[accepted], v[accepted]
    sampled_colors = bgr[v, u].astype(np.uint32)
    packed_rgb = ((sampled_colors[:, 2] << 16) |
                  (sampled_colors[:, 1] << 8) |
                  sampled_colors[:, 0]).astype(np.uint32)
    return (candidate_indices, packed_rgb,
            sampled_labels[accepted].astype(np.uint8),
            sampled_confidence[accepted].astype(np.float32))


def _empty_projection_result():
    return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.float32))


def project_pixels(xyz_camera: np.ndarray, intrinsics,
                   distortion_coeffs: Optional[Iterable[float]] = None):
    """Project optical-frame XYZ to integer image coordinates."""
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    distortion = tuple(distortion_coeffs or ())
    if distortion and any(abs(value) > 1e-12 for value in distortion):
        camera_matrix = np.asarray(
            ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)),
            dtype=np.float64)
        pixels, _ = cv2.projectPoints(
            xyz_camera.astype(np.float64), np.zeros(3), np.zeros(3),
            camera_matrix, np.asarray(distortion, dtype=np.float64))
        pixels = pixels.reshape(-1, 2)
        pixel_x = np.nan_to_num(
            pixels[:, 0], nan=-1.0, posinf=-1.0, neginf=-1.0)
        pixel_y = np.nan_to_num(
            pixels[:, 1], nan=-1.0, posinf=-1.0, neginf=-1.0)
        pixel_x = np.clip(pixel_x, -1.0e9, 1.0e9)
        pixel_y = np.clip(pixel_y, -1.0e9, 1.0e9)
        return (np.rint(pixel_x).astype(np.int64),
                np.rint(pixel_y).astype(np.int64))
    u = np.rint(fx * xyz_camera[:, 0] / xyz_camera[:, 2] + cx)
    v = np.rint(fy * xyz_camera[:, 1] / xyz_camera[:, 2] + cy)
    return u.astype(np.int64), v.astype(np.int64)


def rasterize_camera_depth(xyz_camera: np.ndarray, intrinsics, image_shape,
                           distortion_coeffs=None) -> np.ndarray:
    """Build a nearest-surface depth image from an unorganized camera cloud."""
    height, width = image_shape
    result = np.full(height * width, np.inf, dtype=np.float32)
    valid = np.isfinite(xyz_camera).all(axis=1) & (xyz_camera[:, 2] > 0.0)
    points = xyz_camera[valid]
    if not len(points):
        return np.zeros((height, width), dtype=np.float32)
    u, v = project_pixels(points, intrinsics, distortion_coeffs)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    flat = v[inside] * width + u[inside]
    np.minimum.at(result, flat, points[inside, 2])
    result[~np.isfinite(result)] = 0.0
    return result.reshape(height, width)
