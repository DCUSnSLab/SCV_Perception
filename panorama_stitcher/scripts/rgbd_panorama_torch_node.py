#!/usr/bin/env python3
"""Calibrated dual-RGB-D panorama implemented with rclpy and PyTorch.

The ROS process owns synchronization and message contracts; PyTorch owns the
projection, depth filtering, z-buffer and composition on CUDA.  The older C++
node remains available as an explicit launch-time rollback backend.
"""

from __future__ import annotations

import array
from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Deque, Dict, Iterable, Optional, Tuple

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
import torch
import torch.nn.functional as torch_functional

from panorama_triton_kernels import project_depth as project_depth_triton


@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    rotation_camera_to_rig: np.ndarray
    translation_camera_in_rig: np.ndarray

    def copy(self) -> "CameraModel":
        return CameraModel(
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.width,
            self.height,
            self.rotation_camera_to_rig.copy(),
            self.translation_camera_in_rig.copy(),
        )


@dataclass
class ProjectionState:
    width: int
    height: int
    focal_px: float
    minimum_angle: float
    maximum_angle: float
    minimum_vertical: float
    overlap_minimum_angle: float
    overlap_maximum_angle: float
    seam_x: int
    depth_color_minimum_x: int
    depth_color_maximum_x: int
    left_grid: torch.Tensor
    right_grid: torch.Tensor
    left_mask: torch.Tensor
    right_mask: torch.Tensor
    source_u: torch.Tensor
    source_v: torch.Tensor
    source_indices: torch.Tensor
    pointcloud_columns: torch.Tensor
    pointcloud_rows: torch.Tensor


@dataclass
class FrameOutputs:
    panorama_bgr: torch.Tensor
    validity: torch.Tensor
    range_m: torch.Tensor
    left_points: int
    right_points: int


def _rotation_from_yaw(yaw_rad: float) -> np.ndarray:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _stamp_ns(message: Image) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


class TorchPanoramaBackend:
    """GPU geometry/compositor independent of ROS subscription mechanics."""

    def __init__(self, parameters: Dict[str, object], device: torch.device) -> None:
        self.parameters = parameters
        self.device = device
        self.projection: Optional[ProjectionState] = None
        self.previous_left_depth: Optional[torch.Tensor] = None
        self.previous_right_depth: Optional[torch.Tensor] = None
        self.smoothed_gain = torch.ones(3, dtype=torch.float32, device=device)

        half_yaw = math.radians(float(parameters["camera_half_yaw_deg"]))
        baseline = float(parameters["camera_baseline_m"])
        self.left_model = self._camera_model(
            "left", -half_yaw, -0.5 * baseline
        )
        self.right_model = self._camera_model(
            "right", half_yaw, 0.5 * baseline
        )

    def _camera_model(
        self, prefix: str, yaw: float, translation_x: float
    ) -> CameraModel:
        rotation_values = self.parameters.get(
            f"{prefix}_rotation_camera_to_rig", []
        )
        translation_values = self.parameters.get(
            f"{prefix}_translation_camera_in_rig_m", []
        )
        rotation = (
            np.asarray(rotation_values, dtype=np.float64).reshape(3, 3)
            if len(rotation_values) == 9
            else _rotation_from_yaw(yaw)
        )
        translation = (
            np.asarray(translation_values, dtype=np.float64)
            if len(translation_values) == 3
            else np.asarray([translation_x, 0.0, 0.0], dtype=np.float64)
        )
        model = CameraModel(
            float(self.parameters[f"{prefix}_fx"]),
            float(self.parameters[f"{prefix}_fy"]),
            float(self.parameters[f"{prefix}_cx"]),
            float(self.parameters[f"{prefix}_cy"]),
            int(self.parameters[f"{prefix}_width"]),
            int(self.parameters[f"{prefix}_height"]),
            rotation,
            translation,
        )
        if bool(self.parameters[f"{prefix}_input_image_rotated_180"]):
            model.cx = float(model.width - 1) - model.cx
            model.cy = float(model.height - 1) - model.cy
        return model

    def update_camera_model(
        self, prefix: str, message: CameraInfo, image_rotated_180: bool
    ) -> bool:
        model = self.left_model if prefix == "left" else self.right_model
        updated = model.copy()
        updated.fx = float(message.k[0])
        updated.fy = float(message.k[4])
        updated.cx = float(message.k[2])
        updated.cy = float(message.k[5])
        updated.width = int(message.width)
        updated.height = int(message.height)
        if image_rotated_180:
            updated.cx = float(updated.width - 1) - updated.cx
            updated.cy = float(updated.height - 1) - updated.cy
        changed = any(
            [
                updated.width != model.width,
                updated.height != model.height,
                abs(updated.fx - model.fx) > 1.0e-6,
                abs(updated.fy - model.fy) > 1.0e-6,
                abs(updated.cx - model.cx) > 1.0e-6,
                abs(updated.cy - model.cy) > 1.0e-6,
            ]
        )
        if changed:
            if prefix == "left":
                self.left_model = updated
            else:
                self.right_model = updated
            self.projection = None
            self.previous_left_depth = None
            self.previous_right_depth = None
        return changed

    @staticmethod
    def _rig_ray(model: CameraModel, image_x: float, image_y: float) -> np.ndarray:
        camera_ray = np.asarray(
            [
                (image_x - model.cx) / model.fx,
                (image_y - model.cy) / model.fy,
                1.0,
            ],
            dtype=np.float64,
        )
        return model.rotation_camera_to_rig @ camera_ray

    def _camera_bounds(self, model: CameraModel) -> Tuple[float, float, float, float]:
        bounds = [math.inf, -math.inf, math.inf, -math.inf]
        reference_z = float(self.parameters["color_reference_plane_z_m"])
        for image_y in (0.0, float(model.height - 1)):
            for image_x in (0.0, float(model.width - 1)):
                ray = self._rig_ray(model, image_x, image_y)
                if ray[2] <= 1.0e-6:
                    continue
                if reference_z > 0.0:
                    scale = (reference_z - model.translation_camera_in_rig[2]) / ray[2]
                    if scale <= 0.0:
                        continue
                    ray = model.translation_camera_in_rig + scale * ray
                angle = math.atan2(float(ray[0]), float(ray[2]))
                vertical = float(ray[1]) / math.hypot(float(ray[0]), float(ray[2]))
                bounds[0] = min(bounds[0], angle)
                bounds[1] = max(bounds[1], angle)
                bounds[2] = min(bounds[2], vertical)
                bounds[3] = max(bounds[3], vertical)
        return tuple(bounds)

    def _build_inverse_grid(
        self,
        model: CameraModel,
        panorama_width: int,
        panorama_height: int,
        focal_px: float,
        minimum_angle: float,
        minimum_vertical: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, columns = np.indices(
            (panorama_height, panorama_width), dtype=np.float32
        )
        angles = minimum_angle + columns / np.float32(focal_px)
        ray_x = np.sin(angles)
        ray_y = (minimum_vertical + rows) / np.float32(focal_px)
        ray_z = np.cos(angles)

        reference_z = float(self.parameters["color_reference_plane_z_m"])
        if reference_z > 0.0:
            scale = np.float32(reference_z) / ray_z
            point_x = scale * ray_x - np.float32(model.translation_camera_in_rig[0])
            point_y = scale * ray_y - np.float32(model.translation_camera_in_rig[1])
            point_z = scale * ray_z - np.float32(model.translation_camera_in_rig[2])
        else:
            point_x, point_y, point_z = ray_x, ray_y, ray_z

        rotation_transpose = model.rotation_camera_to_rig.T.astype(np.float32)
        source_x_ray = (
            rotation_transpose[0, 0] * point_x
            + rotation_transpose[0, 1] * point_y
            + rotation_transpose[0, 2] * point_z
        )
        source_y_ray = (
            rotation_transpose[1, 0] * point_x
            + rotation_transpose[1, 1] * point_y
            + rotation_transpose[1, 2] * point_z
        )
        source_z_ray = (
            rotation_transpose[2, 0] * point_x
            + rotation_transpose[2, 1] * point_y
            + rotation_transpose[2, 2] * point_z
        )
        safe_source_z = np.where(
            np.abs(source_z_ray) > np.float32(1.0e-8),
            source_z_ray,
            np.float32(1.0),
        )
        source_x = (
            np.float32(model.fx) * source_x_ray / safe_source_z
            + np.float32(model.cx)
        )
        source_y = (
            np.float32(model.fy) * source_y_ray / safe_source_z
            + np.float32(model.cy)
        )
        valid = (
            (source_z_ray > 0.0)
            & (source_x >= 0.0)
            & (source_x <= model.width - 1.0)
            & (source_y >= 0.0)
            & (source_y <= model.height - 1.0)
        )
        normalized_x = 2.0 * source_x / np.float32(model.width - 1) - 1.0
        normalized_y = 2.0 * source_y / np.float32(model.height - 1) - 1.0
        normalized_x[~valid] = 2.0
        normalized_y[~valid] = 2.0
        grid = np.stack((normalized_x, normalized_y), axis=-1)[None]
        return (
            torch.from_numpy(grid).to(self.device),
            torch.from_numpy(valid).to(self.device),
        )

    def ensure_projection(self, source_width: int, source_height: int) -> ProjectionState:
        if (
            self.projection is not None
            and self.left_model.width == source_width
            and self.left_model.height == source_height
        ):
            return self.projection
        for model in (self.left_model, self.right_model):
            if model.width != source_width or model.height != source_height:
                raise RuntimeError(
                    "color image dimensions do not match calibrated CameraInfo"
                )

        focal_px = 0.5 * (self.left_model.fx + self.right_model.fx) * float(
            self.parameters["projection_scale"]
        )
        left_bounds = self._camera_bounds(self.left_model)
        right_bounds = self._camera_bounds(self.right_model)
        minimum_angle = min(left_bounds[0], right_bounds[0])
        maximum_angle = max(left_bounds[1], right_bounds[1])
        overlap_minimum = max(left_bounds[0], right_bounds[0])
        overlap_maximum = min(left_bounds[1], right_bounds[1])
        minimum_vertical_ratio = min(left_bounds[2], right_bounds[2])
        maximum_vertical_ratio = max(left_bounds[3], right_bounds[3])
        panorama_width = int(math.ceil((maximum_angle - minimum_angle) * focal_px)) + 1
        minimum_vertical = focal_px * minimum_vertical_ratio
        maximum_vertical = focal_px * maximum_vertical_ratio
        panorama_height = int(math.ceil(maximum_vertical - minimum_vertical)) + 1
        if panorama_width <= 0 or panorama_height <= 0 or overlap_maximum <= overlap_minimum:
            raise RuntimeError("invalid cylindrical panorama geometry")

        seam_angle = (
            0.5 * (overlap_minimum + overlap_maximum)
            if bool(self.parameters["auto_seam_center"])
            else math.radians(float(self.parameters["seam_angle_deg"]))
        )
        seam_x = int(round((seam_angle - minimum_angle) * focal_px))
        seam_x = max(0, min(panorama_width - 1, seam_x))
        margin = math.radians(float(self.parameters["depth_color_band_margin_deg"]))
        depth_color_minimum_x = int(
            math.ceil((overlap_minimum - margin - minimum_angle) * focal_px)
        )
        depth_color_maximum_x = int(
            math.floor((overlap_maximum + margin - minimum_angle) * focal_px)
        )
        depth_color_minimum_x = max(0, min(panorama_width - 1, depth_color_minimum_x))
        depth_color_maximum_x = max(0, min(panorama_width - 1, depth_color_maximum_x))

        left_grid, left_mask = self._build_inverse_grid(
            self.left_model,
            panorama_width,
            panorama_height,
            focal_px,
            minimum_angle,
            minimum_vertical,
        )
        right_grid, right_mask = self._build_inverse_grid(
            self.right_model,
            panorama_width,
            panorama_height,
            focal_px,
            minimum_angle,
            minimum_vertical,
        )
        source_rows, source_columns = torch.meshgrid(
            torch.arange(source_height, device=self.device, dtype=torch.float32),
            torch.arange(source_width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        stride = int(self.parameters["pointcloud_stride"])
        pointcloud_rows, pointcloud_columns = torch.meshgrid(
            torch.arange(0, panorama_height, stride, device=self.device),
            torch.arange(0, panorama_width, stride, device=self.device),
            indexing="ij",
        )
        self.projection = ProjectionState(
            panorama_width,
            panorama_height,
            focal_px,
            minimum_angle,
            maximum_angle,
            minimum_vertical,
            overlap_minimum,
            overlap_maximum,
            seam_x,
            depth_color_minimum_x,
            depth_color_maximum_x,
            left_grid,
            right_grid,
            left_mask,
            right_mask,
            source_columns.reshape(-1),
            source_rows.reshape(-1),
            torch.arange(source_width * source_height, device=self.device, dtype=torch.int64),
            pointcloud_columns,
            pointcloud_rows,
        )
        self.previous_left_depth = None
        self.previous_right_depth = None
        return self.projection

    def _valid_depth(self, depth: torch.Tensor) -> torch.Tensor:
        return (
            torch.isfinite(depth)
            & (depth >= float(self.parameters["min_depth_m"]))
            & (depth <= float(self.parameters["max_depth_m"]))
        )

    def _filter_depth(
        self, depth: torch.Tensor, previous: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = self._valid_depth(depth)
        if bool(self.parameters["cuda_depth_spatial_filter"]):
            padded_depth = torch_functional.pad(depth[None, None], (1, 1, 1, 1))
            padded_valid = torch_functional.pad(valid[None, None], (1, 1, 1, 1))
            threshold = torch.maximum(
                torch.full_like(depth, float(self.parameters["cuda_depth_spatial_delta_m"])),
                depth * float(self.parameters["cuda_depth_spatial_delta_relative"]),
            )
            total = torch.zeros_like(depth)
            count = torch.zeros_like(depth)
            for offset_y in range(3):
                for offset_x in range(3):
                    candidate = padded_depth[
                        0,
                        0,
                        offset_y:offset_y + depth.shape[0],
                        offset_x:offset_x + depth.shape[1],
                    ]
                    candidate_valid = padded_valid[
                        0,
                        0,
                        offset_y:offset_y + depth.shape[0],
                        offset_x:offset_x + depth.shape[1],
                    ]
                    accepted = candidate_valid & (torch.abs(candidate - depth) <= threshold)
                    total += torch.where(accepted, candidate, 0.0)
                    count += accepted
            depth = torch.where(valid, total / torch.clamp_min(count, 1.0), 0.0)
            valid = self._valid_depth(depth)
        if bool(self.parameters["cuda_depth_temporal_filter"]):
            if previous is None or previous.shape != depth.shape:
                previous = torch.zeros_like(depth)
            previous_valid = self._valid_depth(previous)
            stable = (
                valid
                & previous_valid
                & (
                    torch.abs(depth - previous)
                    <= float(self.parameters["cuda_depth_temporal_reset_m"])
                )
            )
            alpha = float(self.parameters["cuda_depth_temporal_alpha"])
            depth = torch.where(stable, (1.0 - alpha) * previous + alpha * depth, depth)
            depth = torch.where(valid, depth, 0.0)
        return depth, depth.clone()

    def _depth_edges(self, depth: torch.Tensor) -> torch.Tensor:
        valid = self._valid_depth(depth)
        threshold = torch.maximum(
            torch.full_like(depth, float(self.parameters["depth_discontinuity_abs_m"])),
            depth * float(self.parameters["depth_discontinuity_relative"]),
        )
        edge = torch.zeros_like(valid)
        comparisons = (
            (slice(None), slice(1, None), slice(None), slice(None, -1)),
            (slice(None), slice(None, -1), slice(None), slice(1, None)),
            (slice(1, None), slice(None), slice(None, -1), slice(None)),
            (slice(None, -1), slice(None), slice(1, None), slice(None)),
        )
        for target_y, target_x, neighbor_y, neighbor_x in comparisons:
            candidate_valid = valid[neighbor_y, neighbor_x]
            candidate = depth[neighbor_y, neighbor_x]
            edge[target_y, target_x] |= (~candidate_valid) | (
                torch.abs(candidate - depth[target_y, target_x])
                > threshold[target_y, target_x]
            )
        return edge & valid

    def _project_depth(
        self, depth: torch.Tensor, model: CameraModel
    ) -> Tuple[torch.Tensor, int]:
        if bool(self.parameters["use_triton_projection"]):
            keys = project_depth_triton(
                depth.contiguous(), model, self.projection, self.parameters
            )
            # The exact accepted source count is diagnostic-only. Avoid a
            # full-frame GPU reduction/synchronization in the hot path.
            return keys, int(depth.numel())
        projection = self.projection
        assert projection is not None
        flat_depth = depth.reshape(-1)
        valid = self._valid_depth(flat_depth)
        local_x = (projection.source_u - model.cx) / model.fx * flat_depth
        local_y = (projection.source_v - model.cy) / model.fy * flat_depth
        rotation = torch.as_tensor(
            model.rotation_camera_to_rig, dtype=torch.float32, device=self.device
        )
        translation = torch.as_tensor(
            model.translation_camera_in_rig,
            dtype=torch.float32,
            device=self.device,
        )
        rig_x = (
            rotation[0, 0] * local_x
            + rotation[0, 1] * local_y
            + rotation[0, 2] * flat_depth
            + translation[0]
        )
        rig_y = (
            rotation[1, 0] * local_x
            + rotation[1, 1] * local_y
            + rotation[1, 2] * flat_depth
            + translation[1]
        )
        rig_z = (
            rotation[2, 0] * local_x
            + rotation[2, 1] * local_y
            + rotation[2, 2] * flat_depth
            + translation[2]
        )
        horizontal_range = torch.hypot(rig_x, rig_z)
        projected_x_float = (
            (torch.atan2(rig_x, rig_z) - projection.minimum_angle)
            * projection.focal_px
        )
        projected_y_float = (
            projection.focal_px * rig_y / torch.clamp_min(horizontal_range, 1.0e-6)
            - projection.minimum_vertical
        )
        projected_x = torch.round(projected_x_float).to(torch.int64)
        projected_y = torch.round(projected_y_float).to(torch.int64)
        valid &= rig_z > 0.0
        edge = self._depth_edges(depth).reshape(-1)
        panorama_pixels = projection.width * projection.height
        minimum_range = torch.full(
            (panorama_pixels,), float("inf"), device=self.device
        )
        splat_radius = int(self.parameters["depth_splat_radius_px"])
        edge_radius = int(self.parameters["depth_edge_splat_radius_px"])
        offsets = range(-splat_radius, splat_radius + 1)
        for offset_y in offsets:
            for offset_x in offsets:
                target_x = projected_x + offset_x
                target_y = projected_y + offset_y
                selected = (
                    valid
                    & (target_x >= 0)
                    & (target_x < projection.width)
                    & (target_y >= 0)
                    & (target_y < projection.height)
                )
                if max(abs(offset_x), abs(offset_y)) > edge_radius:
                    selected &= ~edge
                indices = target_y[selected] * projection.width + target_x[selected]
                minimum_range.scatter_reduce_(
                    0,
                    indices,
                    horizontal_range[selected],
                    reduce="amin",
                    include_self=True,
                )

        projection_keys = torch.full(
            (panorama_pixels,), torch.iinfo(torch.int64).max, device=self.device, dtype=torch.int64
        )
        surface_margin = max(float(self.parameters["occlusion_switch_margin_m"]), 0.02)
        range_mm = torch.clamp(
            torch.round(horizontal_range * 1000.0), 0, 65534
        ).to(torch.int64)
        for offset_y in offsets:
            for offset_x in offsets:
                target_x = projected_x + offset_x
                target_y = projected_y + offset_y
                selected = (
                    valid
                    & (target_x >= 0)
                    & (target_x < projection.width)
                    & (target_y >= 0)
                    & (target_y < projection.height)
                )
                if max(abs(offset_x), abs(offset_y)) > edge_radius:
                    selected &= ~edge
                indices = target_y[selected] * projection.width + target_x[selected]
                selected_indices = torch.nonzero(selected, as_tuple=False).flatten()
                if selected_indices.numel() == 0:
                    continue
                on_surface = horizontal_range[selected] <= minimum_range[indices] + surface_margin
                indices = indices[on_surface]
                selected_indices = selected_indices[on_surface]
                distance_squared = (
                    (target_x[selected_indices].float() - projected_x_float[selected_indices]) ** 2
                    + (
                        target_y[selected_indices].float()
                        - projected_y_float[selected_indices]
                    ) ** 2
                )
                # PyTorch scatter_reduce currently operates on signed int64.
                # Keep the packed key below INT64_MAX; otherwise bit 63 would
                # make a farther candidate look negative and incorrectly win.
                distance_key = torch.clamp(
                    torch.round(distance_squared * 4096.0), 0, 32767
                ).to(torch.int64)
                keys = (
                    (distance_key << 48)
                    | (range_mm[selected_indices] << 32)
                    | projection.source_indices[selected_indices]
                )
                projection_keys.scatter_reduce_(
                    0, indices, keys, reduce="amin", include_self=True
                )
        return (
            projection_keys.reshape(projection.height, projection.width),
            int(valid.sum().item()),
        )

    def _remap_color(
        self, color_hwc: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        color_nchw = color_hwc.permute(2, 0, 1)[None].to(torch.float32)
        return torch_functional.grid_sample(
            color_nchw,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0].permute(1, 2, 0)

    def _estimate_gain(
        self,
        left_base: torch.Tensor,
        right_base: torch.Tensor,
        overlap: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(self.parameters["enable_exposure_compensation"]):
            return self.smoothed_gain
        stride = int(self.parameters["exposure_sample_stride"])
        sampled_mask = overlap[::stride, ::stride]
        if not bool(sampled_mask.any()):
            return self.smoothed_gain
        left_mean = left_base[::stride, ::stride][sampled_mask].mean(dim=0)
        right_mean = right_base[::stride, ::stride][sampled_mask].mean(dim=0)
        measured = torch.clamp(
            left_mean / torch.clamp_min(right_mean, 1.0),
            float(self.parameters["min_exposure_gain"]),
            float(self.parameters["max_exposure_gain"]),
        )
        alpha = float(self.parameters["exposure_smoothing"])
        self.smoothed_gain = (1.0 - alpha) * self.smoothed_gain + alpha * measured
        return self.smoothed_gain

    @staticmethod
    def _key_valid(keys: torch.Tensor) -> torch.Tensor:
        return keys != torch.iinfo(torch.int64).max

    @staticmethod
    def _key_range(keys: torch.Tensor) -> torch.Tensor:
        return ((keys >> 32) & 0xFFFF).to(torch.float32) * 0.001

    def process(
        self,
        left_color: torch.Tensor,
        left_depth: torch.Tensor,
        right_color: torch.Tensor,
        right_depth: torch.Tensor,
    ) -> FrameOutputs:
        projection = self.ensure_projection(left_color.shape[1], left_color.shape[0])
        left_depth, self.previous_left_depth = self._filter_depth(
            left_depth, self.previous_left_depth
        )
        right_depth, self.previous_right_depth = self._filter_depth(
            right_depth, self.previous_right_depth
        )
        left_keys, left_points = self._project_depth(left_depth, self.left_model)
        right_keys, right_points = self._project_depth(right_depth, self.right_model)

        # Inputs have already been normalized to BGR at the ROS boundary.
        left_bgr = left_color
        right_bgr = right_color
        left_base = self._remap_color(left_color, projection.left_grid)
        right_base = self._remap_color(right_color, projection.right_grid)
        overlap = projection.left_mask & projection.right_mask
        gain = self._estimate_gain(left_base, right_base, overlap)
        right_base = torch.clamp(right_base * gain, 0.0, 255.0)

        left_valid = self._key_valid(left_keys)
        right_valid = self._key_valid(right_keys)
        left_range = self._key_range(left_keys)
        right_range = self._key_range(right_keys)
        columns = torch.arange(projection.width, device=self.device)[None, :]
        owner_left = columns <= projection.seam_x
        use_left_range = left_valid & (~right_valid | owner_left)
        range_m = torch.where(
            use_left_range,
            left_range,
            torch.where(right_valid, right_range, 0.0),
        )
        validity = left_valid | right_valid

        panorama = torch.zeros_like(left_base)
        base_owner_left = projection.left_mask & (~projection.right_mask | owner_left)
        base_owner_right = projection.right_mask & ~base_owner_left
        panorama[base_owner_left] = left_base[base_owner_left]
        panorama[base_owner_right] = right_base[base_owner_right]

        if bool(self.parameters["depth_aware_color"]):
            left_source = left_bgr.reshape(-1, 3)
            right_source = right_bgr.reshape(-1, 3)
            band_start = projection.depth_color_minimum_x
            band_stop = projection.depth_color_maximum_x + 1
            left_keys_band = left_keys[:, band_start:band_stop]
            right_keys_band = right_keys[:, band_start:band_stop]
            left_indices = (left_keys_band & 0xFFFFFFFF).clamp_max(
                left_source.shape[0] - 1
            )
            right_indices = (right_keys_band & 0xFFFFFFFF).clamp_max(
                right_source.shape[0] - 1
            )
            left_projected_color = left_source[left_indices].to(torch.float32)
            right_projected_color = torch.clamp(
                right_source[right_indices].to(torch.float32) * gain, 0.0, 255.0
            )
            left_valid_band = left_valid[:, band_start:band_stop]
            right_valid_band = right_valid[:, band_start:band_stop]
            owner_left_band = owner_left[:, band_start:band_stop]
            depth_owner_left = left_valid_band & (
                ~right_valid_band | owner_left_band
            )
            depth_owner_right = right_valid_band & ~depth_owner_left
            panorama_band = panorama[:, band_start:band_stop]
            panorama_band[depth_owner_left] = left_projected_color[depth_owner_left]
            panorama_band[depth_owner_right] = right_projected_color[depth_owner_right]
        return FrameOutputs(
            panorama.to(torch.uint8),
            validity.to(torch.uint8) * 255,
            range_m,
            left_points,
            right_points,
        )

    def pointcloud_tensors(
        self, outputs: FrameOutputs
    ) -> torch.Tensor:
        projection = self.projection
        assert projection is not None
        rows = projection.pointcloud_rows
        columns = projection.pointcloud_columns
        ranges = outputs.range_m[rows, columns]
        valid = (outputs.validity[rows, columns] != 0) & torch.isfinite(ranges) & (ranges > 0.0)
        rows = rows[valid].to(torch.float32)
        columns = columns[valid].to(torch.float32)
        ranges = ranges[valid]
        angles = projection.minimum_angle + columns / projection.focal_px
        x = ranges * torch.sin(angles)
        y = ranges * (projection.minimum_vertical + rows) / projection.focal_px
        z = ranges * torch.cos(angles)
        colors = outputs.panorama_bgr[
            projection.pointcloud_rows[valid], projection.pointcloud_columns[valid]
        ].to(torch.int64)
        packed_rgb = (colors[:, 2] << 16) | (colors[:, 1] << 8) | colors[:, 0]
        # Match PointCloud2's 32-byte xyz/rgb layout on the GPU, then perform
        # one device-to-host transfer rather than four synchronizing copies.
        packed = torch.zeros((x.numel(), 8), dtype=torch.float32, device=self.device)
        packed[:, 0] = x
        packed[:, 1] = y
        packed[:, 2] = z
        packed.view(torch.int32)[:, 4] = packed_rgb.to(torch.int32)
        return packed


class RgbdPanoramaTorchNode(Node):
    def __init__(self) -> None:
        super().__init__("panorama_stitcher")
        defaults: Dict[str, object] = {
            "left_color_topic": "/front_left/front_left/color/image_raw",
            "left_depth_topic": "/front_left/front_left/aligned_depth_to_color/image_raw",
            "left_camera_info_topic": "/front_left/front_left/color/camera_info",
            "right_color_topic": "/front_right/front_right/color/image_raw",
            "right_depth_topic": "/front_right/front_right/aligned_depth_to_color/image_raw",
            "right_camera_info_topic": "/front_right/front_right/color/camera_info",
            "output_topic": "/panorama/image_raw",
            "validity_topic": "/panorama/validity",
            "range_topic": "/panorama/range",
            "pointcloud_topic": "/panorama/points",
            "output_frame_id": "panorama_optical_frame",
            "publish_auxiliary_outputs": False,
            "publish_validity_output": False,
            "publish_range_output": False,
            "publish_pointcloud": False,
            "publish_only_when_subscribed": True,
            "publisher_best_effort": False,
            "auxiliary_publisher_best_effort": True,
            "pointcloud_publisher_best_effort": True,
            "pointcloud_stride": 4,
            "max_output_rate_hz": 0.0,
            "auxiliary_output_rate_hz": 0.0,
            "input_best_effort": False,
            "sync_queue_size": 4,
            "sync_slop_ms": 35.0,
            "camera_half_yaw_deg": 30.397,
            "camera_baseline_m": 0.06339742196001438,
            # Non-empty float defaults force rclpy to declare DOUBLE_ARRAY;
            # an empty Python list is otherwise inferred as BYTE_ARRAY.
            "left_rotation_camera_to_rig": [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
            "right_rotation_camera_to_rig": [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
            "left_translation_camera_in_rig_m": [0.0, 0.0, 0.0],
            "right_translation_camera_in_rig_m": [0.0, 0.0, 0.0],
            "left_input_image_rotated_180": True,
            "right_input_image_rotated_180": True,
            "left_fx": 1375.93896484375,
            "left_fy": 1376.0078125,
            "left_cx": 962.9755859375,
            "left_cy": 539.9728393554688,
            "left_width": 1920,
            "left_height": 1080,
            "right_fx": 1369.7860107421875,
            "right_fy": 1369.6165771484375,
            "right_cx": 967.3739013671875,
            "right_cy": 566.1657104492188,
            "right_width": 1920,
            "right_height": 1080,
            # Match the existing native node: use the active sensor profile's
            # CameraInfo while keeping the calibrated rig extrinsics in YAML.
            "use_runtime_camera_info": True,
            "projection_scale": 1.0,
            "color_reference_plane_z_m": 0.0,
            "min_depth_m": 0.2,
            "max_depth_m": 15.0,
            "depth_scale_m": 0.001,
            "depth_color_band_margin_deg": 0.0,
            "auto_seam_center": True,
            "seam_angle_deg": 0.0,
            "depth_aware_color": True,
            "enable_exposure_compensation": True,
            "exposure_sample_stride": 4,
            "exposure_smoothing": 0.15,
            "min_exposure_gain": 0.75,
            "max_exposure_gain": 1.33,
            "cuda_depth_spatial_filter": True,
            "cuda_depth_spatial_delta_m": 0.03,
            "cuda_depth_spatial_delta_relative": 0.01,
            "cuda_depth_temporal_filter": True,
            "cuda_depth_temporal_alpha": 0.65,
            "cuda_depth_temporal_reset_m": 0.08,
            "depth_discontinuity_abs_m": 0.08,
            "depth_discontinuity_relative": 0.04,
            "depth_splat_radius_px": 1,
            "depth_edge_splat_radius_px": 0,
            "occlusion_switch_margin_m": 0.05,
            "diagnostics_period_sec": 2.0,
            "use_cuda": True,
            "use_triton_projection": True,
        }
        self.parameters: Dict[str, object] = {}
        for name, default in defaults.items():
            self.parameters[name] = self.declare_parameter(name, default).value
        if not bool(self.parameters["use_cuda"]):
            raise RuntimeError(
                "The Python panorama backend requires use_cuda:=true; use "
                "panorama_backend:=cpp use_cuda:=false for the CPU rollback path"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        torch.set_grad_enabled(False)
        self.device = torch.device("cuda:0")
        self.backend = TorchPanoramaBackend(self.parameters, self.device)
        self.projection_lock = threading.Lock()
        self.camera_info_received = {"left": False, "right": False}
        self.camera_info_changed = False
        self.backend_warmed = False
        # Start immediately from the persisted calibration. CameraInfo may be
        # advertised before the sensor actually emits it (for example while a
        # simulator is paused), so it must not gate panorama availability.
        self._warm_up_backend()

        self.image_publisher = self.create_publisher(
            Image,
            str(self.parameters["output_topic"]),
            self._qos(bool(self.parameters["publisher_best_effort"])),
        )
        self.validity_publisher = None
        self.range_publisher = None
        if bool(self.parameters["publish_validity_output"]):
            self.validity_publisher = self.create_publisher(
                Image,
                str(self.parameters["validity_topic"]),
                self._qos(bool(self.parameters["auxiliary_publisher_best_effort"])),
            )
        if bool(self.parameters["publish_range_output"]):
            self.range_publisher = self.create_publisher(
                Image,
                str(self.parameters["range_topic"]),
                self._qos(bool(self.parameters["auxiliary_publisher_best_effort"])),
            )
        self.pointcloud_publisher = None
        if bool(self.parameters["publish_pointcloud"]):
            self.pointcloud_publisher = self.create_publisher(
                PointCloud2,
                str(self.parameters["pointcloud_topic"]),
                self._qos(bool(self.parameters["pointcloud_publisher_best_effort"])),
            )

        input_qos = self._qos(bool(self.parameters["input_best_effort"]))
        self.queues: Dict[str, Deque[Image]] = {
            key: deque(maxlen=max(4, min(16, int(self.parameters["sync_queue_size"]))))
            for key in ("left_color", "left_depth", "right_color", "right_depth")
        }
        self.condition = threading.Condition()
        self.pending: Optional[Tuple[Image, Image, Image, Image, float, float]] = None
        self.pending_sequence = 0
        self.stop_worker = False
        self.input_counts = {key: 0 for key in self.queues}
        self.image_subscriptions = []
        topics = {
            "left_color": self.parameters["left_color_topic"],
            "left_depth": self.parameters["left_depth_topic"],
            "right_color": self.parameters["right_color_topic"],
            "right_depth": self.parameters["right_depth_topic"],
        }
        for key, topic in topics.items():
            self.image_subscriptions.append(
                self.create_subscription(
                    Image,
                    str(topic),
                    lambda message, queue_key=key: self._receive(queue_key, message),
                    input_qos,
                )
            )
        info_qos = self._qos(False)
        self.left_info_subscription = self.create_subscription(
            CameraInfo,
            str(self.parameters["left_camera_info_topic"]),
            lambda message: self._camera_info("left", message),
            info_qos,
        )
        self.right_info_subscription = self.create_subscription(
            CameraInfo,
            str(self.parameters["right_camera_info_topic"]),
            lambda message: self._camera_info("right", message),
            info_qos,
        )
        self.frame_count = 0
        self.diagnostic_frames = 0
        self.diagnostic_processing_ms = 0.0
        self.diagnostic_gpu_ms = 0.0
        self.last_diagnostic = time.monotonic()
        self.last_processing_start = 0.0
        self.last_depth_age = (0.0, 0.0)
        self.last_sync_span = 0.0
        self.last_points = (0, 0, 0)
        self.worker = threading.Thread(target=self._processing_loop, daemon=True)
        self.worker.start()
        self.get_logger().info(
            "Python/PyTorch CUDA panorama ready: "
            f"{torch.cuda.get_device_name(0)}, torch={torch.__version__} "
            f"CUDA={torch.version.cuda}"
        )

    def _warm_up_backend(self) -> None:
        """Compile Triton and initialize PyTorch kernels before live frames."""
        if self.backend_warmed:
            return
        height = int(self.parameters["left_height"])
        width = int(self.parameters["left_width"])
        color = torch.zeros((height, width, 3), dtype=torch.uint8, device=self.device)
        depth = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        started = time.monotonic()
        try:
            with torch.inference_mode():
                self.backend.process(color, depth, color, depth)
        except Exception as error:
            if not bool(self.parameters["use_triton_projection"]):
                raise
            self.parameters["use_triton_projection"] = False
            self.get_logger().error(
                f"Triton warm-up failed ({error}); using the slower PyTorch "
                "projection path"
            )
            with torch.inference_mode():
                self.backend.process(color, depth, color, depth)
        torch.cuda.synchronize()
        self.backend.previous_left_depth = None
        self.backend.previous_right_depth = None
        with torch.inference_mode():
            self.backend.smoothed_gain.fill_(1.0)
        self.backend_warmed = True
        self.get_logger().info(
            f"Python panorama GPU warm-up completed in "
            f"{(time.monotonic() - started) * 1000.0:.1f} ms"
        )

    @staticmethod
    def _qos(best_effort: bool) -> QoSProfile:
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=(
                ReliabilityPolicy.BEST_EFFORT
                if best_effort
                else ReliabilityPolicy.RELIABLE
            ),
            durability=DurabilityPolicy.VOLATILE,
        )

    def _camera_info(self, prefix: str, message: CameraInfo) -> None:
        if not bool(self.parameters["use_runtime_camera_info"]):
            return
        with self.projection_lock:
            changed = self.backend.update_camera_model(
                prefix,
                message,
                bool(self.parameters[f"{prefix}_input_image_rotated_180"]),
            )
            self.camera_info_received[prefix] = True
            self.camera_info_changed |= changed
            if all(self.camera_info_received.values()) and self.camera_info_changed:
                self.backend_warmed = False
                self._warm_up_backend()
                self.camera_info_changed = False
        if changed:
            self.get_logger().info(
                f"{prefix} CameraInfo: {message.width}x{message.height} "
                f"fx/fy={message.k[0]:.3f}/{message.k[4]:.3f} "
                f"cx/cy={message.k[2]:.3f}/{message.k[5]:.3f}"
            )

    def _receive(self, key: str, message: Image) -> None:
        with self.condition:
            self.input_counts[key] += 1
            self.queues[key].append(message)
            synchronized = self._try_synchronize()
            if synchronized is not None:
                self.pending = synchronized
                self.pending_sequence += 1
                self.condition.notify()

    @staticmethod
    def _nearest(queue: Iterable[Image], stamp: int) -> Tuple[int, int]:
        return min(
            enumerate(queue), key=lambda item: abs(_stamp_ns(item[1]) - stamp)
        )[0], min(abs(_stamp_ns(message) - stamp) for message in queue)

    def _try_synchronize(
        self,
    ) -> Optional[Tuple[Image, Image, Image, Image, float, float]]:
        if any(not queue for queue in self.queues.values()):
            return None
        left_colors = self.queues["left_color"]
        right_colors = self.queues["right_color"]
        pairs = [
            (
                abs(_stamp_ns(left) - _stamp_ns(right)),
                max(_stamp_ns(left), _stamp_ns(right)),
                li,
                ri,
            )
            for li, left in enumerate(left_colors)
            for ri, right in enumerate(right_colors)
        ]
        color_delta, _, left_index, right_index = min(
            pairs, key=lambda value: (value[0], -value[1])
        )
        slop_ns = int(float(self.parameters["sync_slop_ms"]) * 1.0e6)
        if color_delta > slop_ns:
            return None
        left_color = left_colors[left_index]
        right_color = right_colors[right_index]
        left_depth_index, left_age = self._nearest(
            self.queues["left_depth"], _stamp_ns(left_color)
        )
        right_depth_index, right_age = self._nearest(
            self.queues["right_depth"], _stamp_ns(right_color)
        )
        if left_age > slop_ns or right_age > slop_ns:
            return None
        left_depth = self.queues["left_depth"][left_depth_index]
        right_depth = self.queues["right_depth"][right_depth_index]
        for queue, index in (
            (left_colors, left_index),
            (right_colors, right_index),
            (self.queues["left_depth"], left_depth_index),
            (self.queues["right_depth"], right_depth_index),
        ):
            for _ in range(index + 1):
                queue.popleft()
        return (
            left_color,
            right_color,
            left_depth,
            right_depth,
            left_age / 1.0e6,
            right_age / 1.0e6,
        )

    @staticmethod
    def _image_numpy(message: Image, dtype: np.dtype, channels: int) -> np.ndarray:
        raw = np.frombuffer(message.data, dtype=dtype)
        row_elements = int(message.step) // np.dtype(dtype).itemsize
        rows = raw.reshape(int(message.height), row_elements)
        width_elements = int(message.width) * channels
        selected = rows[:, :width_elements]
        if channels == 1:
            return selected.reshape(int(message.height), int(message.width))
        return selected.reshape(int(message.height), int(message.width), channels)

    @staticmethod
    def _color_to_bgr(message: Image, image: torch.Tensor) -> torch.Tensor:
        encoding = message.encoding.lower()
        if encoding == "bgr8":
            return image
        if encoding == "rgb8":
            return image[..., [2, 1, 0]]
        raise RuntimeError(
            f"unsupported color encoding {message.encoding!r}; expected rgb8 or bgr8"
        )

    def _has_subscribers(self, publisher) -> bool:
        return (
            publisher is not None
            and (
                not bool(self.parameters["publish_only_when_subscribed"])
                or publisher.get_subscription_count() > 0
            )
        )

    def _processing_loop(self) -> None:
        processed_sequence = 0
        while rclpy.ok():
            with self.condition:
                self.condition.wait_for(
                    lambda: self.stop_worker or self.pending_sequence != processed_sequence
                )
                if self.stop_worker:
                    return
                rate = float(self.parameters["max_output_rate_hz"])
                if rate > 0.0 and self.last_processing_start > 0.0:
                    delay = self.last_processing_start + 1.0 / rate - time.monotonic()
                    if delay > 0.0:
                        self.condition.wait(timeout=delay)
                        if self.stop_worker:
                            return
                pending = self.pending
                processed_sequence = self.pending_sequence
                self.last_processing_start = time.monotonic()
            if pending is None:
                continue
            if not self.backend_warmed:
                self._maybe_log_diagnostics()
                continue
            image_demand = self._has_subscribers(self.image_publisher)
            range_demand = self._has_subscribers(self.range_publisher)
            validity_demand = self._has_subscribers(self.validity_publisher)
            cloud_demand = self._has_subscribers(self.pointcloud_publisher)
            if not (image_demand or range_demand or validity_demand or cloud_demand):
                self._maybe_log_diagnostics()
                continue
            try:
                self._process_frame(
                    *pending,
                    image_demand=image_demand,
                    range_demand=range_demand,
                    validity_demand=validity_demand,
                    cloud_demand=cloud_demand,
                )
            except Exception as error:  # keep camera callbacks alive on GPU failure
                if bool(self.parameters["use_triton_projection"]):
                    self.parameters["use_triton_projection"] = False
                    self.get_logger().error(
                        f"Triton projection failed ({error}); falling back to "
                        "the slower PyTorch reference path"
                    )
                else:
                    self.get_logger().error(f"Python panorama frame failed: {error}")
                time.sleep(0.05)

    def _process_frame(
        self,
        left_color_message: Image,
        right_color_message: Image,
        left_depth_message: Image,
        right_depth_message: Image,
        left_depth_age_ms: float,
        right_depth_age_ms: float,
        *,
        image_demand: bool,
        range_demand: bool,
        validity_demand: bool,
        cloud_demand: bool,
    ) -> None:
        start = time.monotonic()
        left_color_np = self._image_numpy(left_color_message, np.uint8, 3)
        right_color_np = self._image_numpy(right_color_message, np.uint8, 3)
        left_depth_np = self._image_numpy(left_depth_message, np.uint16, 1)
        right_depth_np = self._image_numpy(right_depth_message, np.uint16, 1)
        left_color = self._color_to_bgr(
            left_color_message, torch.from_numpy(left_color_np).to(self.device)
        )
        right_color = self._color_to_bgr(
            right_color_message, torch.from_numpy(right_color_np).to(self.device)
        )
        depth_scale = float(self.parameters["depth_scale_m"])
        left_depth = (
            torch.from_numpy(left_depth_np).to(self.device, dtype=torch.float32)
            * depth_scale
        )
        right_depth = (
            torch.from_numpy(right_depth_np).to(self.device, dtype=torch.float32)
            * depth_scale
        )

        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_stop = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
        with self.projection_lock, torch.inference_mode():
            outputs = self.backend.process(
                left_color, left_depth, right_color, right_depth
            )
            cloud_tensors = self.backend.pointcloud_tensors(outputs) if cloud_demand else None
        gpu_stop.record()
        gpu_stop.synchronize()
        gpu_ms = float(gpu_start.elapsed_time(gpu_stop))

        messages = [
            left_color_message,
            right_color_message,
            left_depth_message,
            right_depth_message,
        ]
        newest = max(messages, key=_stamp_ns)
        oldest_stamp = min(_stamp_ns(message) for message in messages)
        newest_stamp = max(_stamp_ns(message) for message in messages)
        header = Header()
        header.stamp.sec = newest.header.stamp.sec
        header.stamp.nanosec = newest.header.stamp.nanosec
        header.frame_id = str(self.parameters["output_frame_id"])
        if cloud_demand and cloud_tensors is not None:
            self.pointcloud_publisher.publish(
                self._pointcloud_message(header, cloud_tensors)
            )
        if range_demand:
            self.range_publisher.publish(
                self._image_message(
                    header,
                    outputs.range_m.cpu().numpy().astype(np.float32),
                    "32FC1",
                )
            )
        if validity_demand:
            self.validity_publisher.publish(
                self._image_message(header, outputs.validity.cpu().numpy(), "mono8")
            )
        if image_demand:
            self.image_publisher.publish(
                self._image_message(header, outputs.panorama_bgr.cpu().numpy(), "bgr8")
            )
        cloud_points = (
            int(cloud_tensors.shape[0]) if cloud_tensors is not None else 0
        )
        self.last_points = (outputs.left_points, outputs.right_points, cloud_points)
        self.last_depth_age = (left_depth_age_ms, right_depth_age_ms)
        self.last_sync_span = (newest_stamp - oldest_stamp) / 1.0e6
        elapsed_ms = (time.monotonic() - start) * 1000.0
        self.frame_count += 1
        self.diagnostic_frames += 1
        self.diagnostic_processing_ms += elapsed_ms
        self.diagnostic_gpu_ms += gpu_ms
        self._maybe_log_diagnostics()

    @staticmethod
    def _image_message(header, image: np.ndarray, encoding: str) -> Image:
        contiguous = np.ascontiguousarray(image)
        message = Image()
        message.header = header
        message.height = int(contiguous.shape[0])
        message.width = int(contiguous.shape[1])
        message.encoding = encoding
        message.is_bigendian = 0
        message.step = int(contiguous.strides[0])
        payload = array.array("B")
        payload.frombytes(memoryview(contiguous).cast("B"))
        message.data = payload
        return message

    @staticmethod
    def _pointcloud_message(header, packed_cloud: torch.Tensor) -> PointCloud2:
        storage = packed_cloud.cpu().numpy()
        message = PointCloud2()
        message.header = header
        message.height = 1
        message.width = int(storage.shape[0])
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = 32
        message.row_step = 32 * message.width
        message.is_dense = True
        payload = array.array("B")
        payload.frombytes(memoryview(storage).cast("B"))
        message.data = payload
        return message

    def _maybe_log_diagnostics(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_diagnostic
        if elapsed < float(self.parameters["diagnostics_period_sec"]):
            return
        projection = self.backend.projection
        if self.diagnostic_frames == 0 or projection is None:
            self.last_diagnostic = now
            self.input_counts = {key: 0 for key in self.input_counts}
            return
        count = float(self.diagnostic_frames)
        input_hz = [self.input_counts[key] / elapsed for key in self.queues]
        self.get_logger().info(
            f"output={projection.width}x{projection.height} "
            f"fps={count / elapsed:.1f} "
            f"processing={self.diagnostic_processing_ms / count:.1f} ms "
            f"backend=PYTORCH_CUDA gpu={self.diagnostic_gpu_ms / count:.1f} ms "
            f"input_hz(Lc/Ld/Rc/Rd)={input_hz[0]:.1f}/{input_hz[1]:.1f}/"
            f"{input_hz[2]:.1f}/{input_hz[3]:.1f} "
            f"depth_age(L/R)={self.last_depth_age[0]:.1f}/"
            f"{self.last_depth_age[1]:.1f} ms "
            f"sync_span(avg/max)={self.last_sync_span:.1f}/"
            f"{self.last_sync_span:.1f} ms "
            f"depth_points(left/right)={self.last_points[0]}/"
            f"{self.last_points[1]} panorama_points={self.last_points[2]} "
            f"total={self.frame_count}"
        )
        self.last_diagnostic = now
        self.diagnostic_frames = 0
        self.diagnostic_processing_ms = 0.0
        self.diagnostic_gpu_ms = 0.0
        self.input_counts = {key: 0 for key in self.input_counts}

    def destroy_node(self) -> bool:
        with self.condition:
            self.stop_worker = True
            self.condition.notify_all()
        if self.worker.is_alive():
            self.worker.join(timeout=3.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[RgbdPanoramaTorchNode] = None
    executor: Optional[MultiThreadedExecutor] = None
    try:
        node = RgbdPanoramaTorchNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
