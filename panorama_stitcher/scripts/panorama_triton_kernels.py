"""Triton kernels for the Python panorama's projection hot path.

The ROS node, calibration, synchronization and composition stay in Python.
Only the per-depth-pixel atomic z-buffer is expressed as a Python Triton JIT
kernel because nine separate PyTorch scatter passes cannot sustain 20 FPS at
dual 1920x1080 resolution.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _is_valid_depth(depth, minimum_depth, maximum_depth):
    return (depth >= minimum_depth) & (depth <= maximum_depth)


@triton.jit
def _project_geometry(
    source_index,
    depth,
    fx,
    fy,
    cx,
    cy,
    r00,
    r01,
    r02,
    r10,
    r11,
    r12,
    r20,
    r21,
    r22,
    tx,
    ty,
    tz,
    panorama_focal,
    panorama_minimum_angle,
    panorama_minimum_vertical,
    SOURCE_WIDTH: tl.constexpr,
):
    source_x = source_index % SOURCE_WIDTH
    source_y = source_index // SOURCE_WIDTH
    local_x = (source_x.to(tl.float32) - cx) / fx * depth
    local_y = (source_y.to(tl.float32) - cy) / fy * depth
    rig_x = r00 * local_x + r01 * local_y + r02 * depth + tx
    rig_y = r10 * local_x + r11 * local_y + r12 * depth + ty
    rig_z = r20 * local_x + r21 * local_y + r22 * depth + tz
    horizontal_range = libdevice.hypot(rig_x, rig_z)
    projected_x = (
        (libdevice.atan2(rig_x, rig_z) - panorama_minimum_angle)
        * panorama_focal
    )
    projected_y = (
        panorama_focal * rig_y / tl.maximum(horizontal_range, 1.0e-6)
        - panorama_minimum_vertical
    )
    return source_x, source_y, rig_z, horizontal_range, projected_x, projected_y


@triton.jit
def _depth_edge(
    depth_pointer,
    source_index,
    source_x,
    source_y,
    center_depth,
    minimum_depth,
    maximum_depth,
    discontinuity_absolute,
    discontinuity_relative,
    valid_program,
    SOURCE_WIDTH: tl.constexpr,
    SOURCE_HEIGHT: tl.constexpr,
):
    threshold = tl.maximum(
        discontinuity_absolute, discontinuity_relative * center_depth
    )
    edge = tl.zeros(source_index.shape, tl.int1)
    left_exists = source_x > 0
    right_exists = source_x + 1 < SOURCE_WIDTH
    up_exists = source_y > 0
    down_exists = source_y + 1 < SOURCE_HEIGHT
    left = tl.load(
        depth_pointer + source_index - 1,
        mask=valid_program & left_exists,
        other=center_depth,
    )
    right = tl.load(
        depth_pointer + source_index + 1,
        mask=valid_program & right_exists,
        other=center_depth,
    )
    up = tl.load(
        depth_pointer + source_index - SOURCE_WIDTH,
        mask=valid_program & up_exists,
        other=center_depth,
    )
    down = tl.load(
        depth_pointer + source_index + SOURCE_WIDTH,
        mask=valid_program & down_exists,
        other=center_depth,
    )
    left_valid = _is_valid_depth(left, minimum_depth, maximum_depth)
    right_valid = _is_valid_depth(right, minimum_depth, maximum_depth)
    up_valid = _is_valid_depth(up, minimum_depth, maximum_depth)
    down_valid = _is_valid_depth(down, minimum_depth, maximum_depth)
    edge |= left_exists & (
        (~left_valid) | (tl.abs(left - center_depth) > threshold)
    )
    edge |= right_exists & (
        (~right_valid) | (tl.abs(right - center_depth) > threshold)
    )
    edge |= up_exists & (
        (~up_valid) | (tl.abs(up - center_depth) > threshold)
    )
    edge |= down_exists & (
        (~down_valid) | (tl.abs(down - center_depth) > threshold)
    )
    return edge


@triton.jit
def project_minimum_range_kernel(
    depth_pointer,
    minimum_range_pointer,
    fx,
    fy,
    cx,
    cy,
    r00,
    r01,
    r02,
    r10,
    r11,
    r12,
    r20,
    r21,
    r22,
    tx,
    ty,
    tz,
    minimum_depth,
    maximum_depth,
    discontinuity_absolute,
    discontinuity_relative,
    panorama_focal,
    panorama_minimum_angle,
    panorama_minimum_vertical,
    SOURCE_PIXELS: tl.constexpr,
    SOURCE_WIDTH: tl.constexpr,
    SOURCE_HEIGHT: tl.constexpr,
    PANORAMA_WIDTH: tl.constexpr,
    PANORAMA_HEIGHT: tl.constexpr,
    SPLAT_RADIUS: tl.constexpr,
    EDGE_SPLAT_RADIUS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    source_index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_program = source_index < SOURCE_PIXELS
    depth = tl.load(depth_pointer + source_index, mask=valid_program, other=0.0)
    valid = valid_program & _is_valid_depth(depth, minimum_depth, maximum_depth)
    source_x, source_y, rig_z, horizontal_range, projected_x, projected_y = (
        _project_geometry(
            source_index,
            depth,
            fx,
            fy,
            cx,
            cy,
            r00,
            r01,
            r02,
            r10,
            r11,
            r12,
            r20,
            r21,
            r22,
            tx,
            ty,
            tz,
            panorama_focal,
            panorama_minimum_angle,
            panorama_minimum_vertical,
            SOURCE_WIDTH,
        )
    )
    valid &= rig_z > 0.0
    center_x = libdevice.rint(projected_x).to(tl.int32)
    center_y = libdevice.rint(projected_y).to(tl.int32)
    edge = _depth_edge(
        depth_pointer,
        source_index,
        source_x,
        source_y,
        depth,
        minimum_depth,
        maximum_depth,
        discontinuity_absolute,
        discontinuity_relative,
        valid_program,
        SOURCE_WIDTH,
        SOURCE_HEIGHT,
    )
    for offset_y in tl.static_range(-SPLAT_RADIUS, SPLAT_RADIUS + 1):
        target_y = center_y + offset_y
        for offset_x in tl.static_range(-SPLAT_RADIUS, SPLAT_RADIUS + 1):
            target_x = center_x + offset_x
            edge_allows = (~edge) | (
                (tl.abs(offset_x) <= EDGE_SPLAT_RADIUS)
                & (tl.abs(offset_y) <= EDGE_SPLAT_RADIUS)
            )
            target_valid = (
                valid
                & edge_allows
                & (target_x >= 0)
                & (target_x < PANORAMA_WIDTH)
                & (target_y >= 0)
                & (target_y < PANORAMA_HEIGHT)
            )
            target_index = target_y * PANORAMA_WIDTH + target_x
            tl.atomic_min(
                minimum_range_pointer + target_index,
                horizontal_range,
                mask=target_valid,
                sem="relaxed",
            )


@triton.jit
def project_selection_key_kernel(
    depth_pointer,
    minimum_range_pointer,
    selection_key_pointer,
    fx,
    fy,
    cx,
    cy,
    r00,
    r01,
    r02,
    r10,
    r11,
    r12,
    r20,
    r21,
    r22,
    tx,
    ty,
    tz,
    minimum_depth,
    maximum_depth,
    discontinuity_absolute,
    discontinuity_relative,
    surface_margin,
    panorama_focal,
    panorama_minimum_angle,
    panorama_minimum_vertical,
    SOURCE_PIXELS: tl.constexpr,
    SOURCE_WIDTH: tl.constexpr,
    SOURCE_HEIGHT: tl.constexpr,
    PANORAMA_WIDTH: tl.constexpr,
    PANORAMA_HEIGHT: tl.constexpr,
    SPLAT_RADIUS: tl.constexpr,
    EDGE_SPLAT_RADIUS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    source_index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_program = source_index < SOURCE_PIXELS
    depth = tl.load(depth_pointer + source_index, mask=valid_program, other=0.0)
    valid = valid_program & _is_valid_depth(depth, minimum_depth, maximum_depth)
    source_x, source_y, rig_z, horizontal_range, projected_x, projected_y = (
        _project_geometry(
            source_index,
            depth,
            fx,
            fy,
            cx,
            cy,
            r00,
            r01,
            r02,
            r10,
            r11,
            r12,
            r20,
            r21,
            r22,
            tx,
            ty,
            tz,
            panorama_focal,
            panorama_minimum_angle,
            panorama_minimum_vertical,
            SOURCE_WIDTH,
        )
    )
    valid &= rig_z > 0.0
    center_x = libdevice.rint(projected_x).to(tl.int32)
    center_y = libdevice.rint(projected_y).to(tl.int32)
    edge = _depth_edge(
        depth_pointer,
        source_index,
        source_x,
        source_y,
        depth,
        minimum_depth,
        maximum_depth,
        discontinuity_absolute,
        discontinuity_relative,
        valid_program,
        SOURCE_WIDTH,
        SOURCE_HEIGHT,
    )
    range_millimeters = tl.minimum(
        libdevice.rint(horizontal_range * 1000.0).to(tl.int64), 65534
    )
    for offset_y in tl.static_range(-SPLAT_RADIUS, SPLAT_RADIUS + 1):
        target_y = center_y + offset_y
        for offset_x in tl.static_range(-SPLAT_RADIUS, SPLAT_RADIUS + 1):
            target_x = center_x + offset_x
            edge_allows = (~edge) | (
                (tl.abs(offset_x) <= EDGE_SPLAT_RADIUS)
                & (tl.abs(offset_y) <= EDGE_SPLAT_RADIUS)
            )
            target_valid = (
                valid
                & edge_allows
                & (target_x >= 0)
                & (target_x < PANORAMA_WIDTH)
                & (target_y >= 0)
                & (target_y < PANORAMA_HEIGHT)
            )
            target_index = target_y * PANORAMA_WIDTH + target_x
            minimum_range = tl.load(
                minimum_range_pointer + target_index,
                mask=target_valid,
                other=float("inf"),
            )
            target_valid &= horizontal_range <= minimum_range + surface_margin
            delta_x = target_x.to(tl.float32) - projected_x
            delta_y = target_y.to(tl.float32) - projected_y
            distance_key = tl.minimum(
                libdevice.rint(
                    (delta_x * delta_x + delta_y * delta_y) * 4096.0
                ).to(tl.int64),
                32767,
            )
            key = (
                (distance_key << 48)
                | (range_millimeters << 32)
                | source_index.to(tl.int64)
            )
            tl.atomic_min(
                selection_key_pointer + target_index,
                key,
                mask=target_valid,
                sem="relaxed",
            )


def project_depth(
    depth: torch.Tensor,
    model,
    projection,
    parameters,
) -> torch.Tensor:
    """Project one aligned depth image into a packed panorama z-buffer."""
    if not depth.is_cuda or depth.dtype != torch.float32 or not depth.is_contiguous():
        raise ValueError("Triton depth input must be contiguous CUDA float32")
    source_height, source_width = depth.shape
    panorama_pixels = projection.width * projection.height
    minimum_ranges = torch.full(
        (panorama_pixels,), float("inf"), dtype=torch.float32, device=depth.device
    )
    selection_keys = torch.full(
        (panorama_pixels,),
        torch.iinfo(torch.int64).max,
        dtype=torch.int64,
        device=depth.device,
    )
    rotation = model.rotation_camera_to_rig.reshape(-1)
    translation = model.translation_camera_in_rig
    common = (
        depth,
        minimum_ranges,
        float(model.fx),
        float(model.fy),
        float(model.cx),
        float(model.cy),
        *[float(value) for value in rotation],
        *[float(value) for value in translation],
        float(parameters["min_depth_m"]),
        float(parameters["max_depth_m"]),
        float(parameters["depth_discontinuity_abs_m"]),
        float(parameters["depth_discontinuity_relative"]),
        float(projection.focal_px),
        float(projection.minimum_angle),
        float(projection.minimum_vertical),
    )
    block_size = 256
    grid = (triton.cdiv(depth.numel(), block_size),)
    compile_time = dict(
        SOURCE_PIXELS=depth.numel(),
        SOURCE_WIDTH=source_width,
        SOURCE_HEIGHT=source_height,
        PANORAMA_WIDTH=projection.width,
        PANORAMA_HEIGHT=projection.height,
        SPLAT_RADIUS=int(parameters["depth_splat_radius_px"]),
        EDGE_SPLAT_RADIUS=int(parameters["depth_edge_splat_radius_px"]),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    project_minimum_range_kernel[grid](*common, **compile_time)
    project_selection_key_kernel[grid](
        depth,
        minimum_ranges,
        selection_keys,
        *common[2:18],
        float(parameters["min_depth_m"]),
        float(parameters["max_depth_m"]),
        float(parameters["depth_discontinuity_abs_m"]),
        float(parameters["depth_discontinuity_relative"]),
        max(float(parameters["occlusion_switch_margin_m"]), 0.02),
        float(projection.focal_px),
        float(projection.minimum_angle),
        float(projection.minimum_vertical),
        **compile_time,
    )
    return selection_keys.reshape(projection.height, projection.width)
