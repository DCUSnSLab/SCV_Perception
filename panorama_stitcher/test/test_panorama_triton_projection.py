"""GPU equivalence test for the Python panorama projection hot path."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rgbd_panorama_torch_node import TorchPanoramaBackend  # noqa: E402


def synthetic_parameters(use_triton: bool):
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    values = {
        "camera_half_yaw_deg": 0.0,
        "camera_baseline_m": 0.0,
        "left_rotation_camera_to_rig": identity,
        "right_rotation_camera_to_rig": identity,
        "left_translation_camera_in_rig_m": [0.0, 0.0, 0.0],
        "right_translation_camera_in_rig_m": [0.0, 0.0, 0.0],
        "left_input_image_rotated_180": False,
        "right_input_image_rotated_180": False,
        "projection_scale": 1.0,
        "color_reference_plane_z_m": 0.0,
        "auto_seam_center": True,
        "seam_angle_deg": 0.0,
        "depth_color_band_margin_deg": 0.0,
        "depth_projection_stride": 1,
        "pointcloud_stride": 2,
        "min_depth_m": 0.2,
        "max_depth_m": 15.0,
        "depth_discontinuity_abs_m": 0.08,
        "depth_discontinuity_relative": 0.04,
        "depth_splat_radius_px": 1,
        "depth_edge_splat_radius_px": 0,
        "occlusion_switch_margin_m": 0.05,
        "pytorch_output_space_splat": False,
        "use_triton_projection": use_triton,
    }
    for prefix in ("left", "right"):
        values.update(
            {
                f"{prefix}_fx": 50.0,
                f"{prefix}_fy": 50.0,
                f"{prefix}_cx": 31.5,
                f"{prefix}_cy": 15.5,
                f"{prefix}_width": 64,
                f"{prefix}_height": 32,
            }
        )
    return values


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability(0)[0] < 7,
    reason="Triton projection requires CUDA compute capability 7.0 or newer",
)
def test_triton_projection_matches_pytorch_reference_mask_and_range():
    device = torch.device("cuda:0")
    triton_backend = TorchPanoramaBackend(synthetic_parameters(True), device)
    reference_backend = TorchPanoramaBackend(synthetic_parameters(False), device)
    triton_backend.ensure_projection(64, 32)
    reference_backend.ensure_projection(64, 32)

    rows = torch.arange(32, device=device, dtype=torch.float32)[:, None]
    columns = torch.arange(64, device=device, dtype=torch.float32)[None, :]
    depth = (2.0 + rows * 0.005 + columns * 0.001).contiguous()
    depth[:, 29:31] = 0.0

    triton_keys, _ = triton_backend._project_depth(
        depth, triton_backend.left_model
    )
    reference_keys, _ = reference_backend._project_depth(
        depth, reference_backend.left_model
    )
    torch.cuda.synchronize()

    invalid = torch.iinfo(torch.int64).max
    triton_valid = triton_keys != invalid
    reference_valid = reference_keys != invalid
    assert torch.equal(triton_valid, reference_valid)

    common = triton_valid & reference_valid
    triton_range_mm = (triton_keys[common] >> 32) & 0xFFFF
    reference_range_mm = (reference_keys[common] >> 32) & 0xFFFF
    assert int((triton_range_mm - reference_range_mm).abs().max()) <= 2
