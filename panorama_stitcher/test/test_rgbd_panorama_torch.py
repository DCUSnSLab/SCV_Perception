"""GPU equivalence test for the Python panorama projection hot path."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo, Image
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rgbd_panorama_torch_node import (  # noqa: E402
    LatestOnlyPublisher,
    RgbdPanoramaTorchNode,
    TorchPanoramaBackend,
    _edge_aware_spatial_filter,
)


def _image_at(milliseconds: int) -> Image:
    message = Image()
    nanoseconds = milliseconds * 1_000_000
    message.header.stamp.sec = nanoseconds // 1_000_000_000
    message.header.stamp.nanosec = nanoseconds % 1_000_000_000
    return message


def _depth_image(encoding: str, values: np.ndarray) -> Image:
    message = Image()
    message.height, message.width = values.shape
    message.encoding = encoding
    message.is_bigendian = False
    message.step = values.strides[0]
    message.data = values.tobytes()
    return message


def test_depth_numpy_uses_millimetre_scale_for_realsense_uint16():
    node = object.__new__(RgbdPanoramaTorchNode)
    message = _depth_image(
        "16UC1", np.array([[1000, 2500], [0, 9999]], dtype=np.uint16)
    )

    depth, scale = node._depth_numpy(message, 0.001)

    assert depth.dtype == np.uint16
    assert depth.tolist() == [[1000, 2500], [0, 9999]]
    assert scale == pytest.approx(0.001)


def test_depth_numpy_keeps_gazebo_float32_metres():
    node = object.__new__(RgbdPanoramaTorchNode)
    message = _depth_image(
        "32FC1", np.array([[1.0, 2.5], [0.0, 9.999]], dtype=np.float32)
    )

    depth, scale = node._depth_numpy(message, 0.001)

    assert depth.dtype == np.float32
    np.testing.assert_allclose(
        depth,
        np.array([[1.0, 2.5], [0.0, 9.999]], dtype=np.float32),
    )
    assert scale == pytest.approx(1.0)


def test_depth_numpy_rejects_unsupported_encoding():
    node = object.__new__(RgbdPanoramaTorchNode)
    message = _depth_image("8UC1", np.ones((2, 2), dtype=np.uint8))

    with pytest.raises(RuntimeError, match="unsupported depth encoding"):
        node._depth_numpy(message, 0.001)


def test_sync_statistics_count_success_stale_input_and_span():
    node = object.__new__(RgbdPanoramaTorchNode)
    node.parameters = {"sync_slop_ms": 35.0}
    node.queues = {
        "left_color": deque((_image_at(0), _image_at(100)), maxlen=4),
        "left_depth": deque((_image_at(99),), maxlen=4),
        "right_color": deque((_image_at(101),), maxlen=4),
        "right_depth": deque((_image_at(102),), maxlen=4),
    }
    node.sync_successes = 0
    node.sync_stale_drops = {key: 0 for key in node.queues}
    node.sync_span_sum_ms = 0.0
    node.sync_span_max_ms = 0.0

    synchronized = node._try_synchronize()

    assert synchronized is not None
    assert node.sync_successes == 1
    assert node.sync_stale_drops == {
        "left_color": 1,
        "left_depth": 0,
        "right_color": 0,
        "right_depth": 0,
    }
    assert node.sync_span_sum_ms == pytest.approx(3.0)
    assert node.sync_span_max_ms == pytest.approx(3.0)


def test_camera_profile_change_drops_old_frames():
    class Backend:
        @staticmethod
        def update_camera_model(*_args):
            return True

    class Logger:
        @staticmethod
        def info(*_args):
            pass

    node = object.__new__(RgbdPanoramaTorchNode)
    node.parameters = {
        "use_runtime_camera_info": True,
        "left_input_image_rotated_180": False,
    }
    node.backend = Backend()
    node.projection_lock = threading.Lock()
    node.condition = threading.Condition()
    node.camera_info_received = {"left": True, "right": True}
    node.camera_info_changed = False
    node.backend_warmed = True
    node.queues = {"left_color": deque((_image_at(0),), maxlen=4)}
    node.pending = (_image_at(0),)
    node.pending_ready = True
    node.get_logger = Logger

    node._camera_info("left", CameraInfo())

    assert not node.queues["left_color"]
    assert node.pending is None
    assert not node.pending_ready
    assert not node.backend_warmed
    assert node.camera_info_received == {"left": True, "right": False}


def test_latest_only_publisher_replaces_pending_stale_message():
    first_started = threading.Event()
    release_first = threading.Event()

    class FakePublisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)
            if len(self.messages) == 1:
                first_started.set()
                assert release_first.wait(timeout=2.0)

    publisher = FakePublisher()
    worker = LatestOnlyPublisher("test", publisher)
    try:
        assert worker.submit("first")
        assert first_started.wait(timeout=2.0)
        assert worker.submit("stale")
        assert worker.submit("latest")
        release_first.set()
        deadline = time.monotonic() + 2.0
        while len(publisher.messages) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert publisher.messages == ["first", "latest"]
        published, dropped, _, error = worker.take_statistics()
        assert published == 2
        assert dropped == 1
        assert error is None
    finally:
        worker.request_stop()
        worker.join(timeout=2.0)


def test_edge_aware_spatial_filter_preserves_depth_discontinuity():
    depth = torch.full((5, 5), 2.0, dtype=torch.float32)
    depth[2, 2] = 5.0
    depth[0, 0] = 0.0
    filtered = _edge_aware_spatial_filter(depth, 0.2, 15.0, 0.03, 0.01)
    assert filtered[2, 2] == pytest.approx(5.0)
    assert filtered[1, 1] == pytest.approx(2.0)
    assert filtered[0, 0] == pytest.approx(0.0)


def test_empty_overlap_does_not_poison_exposure_gain():
    backend = object.__new__(TorchPanoramaBackend)
    backend.parameters = {
        "enable_exposure_compensation": True,
        "exposure_sample_stride": 1,
        "min_exposure_gain": 0.75,
        "max_exposure_gain": 1.33,
        "exposure_smoothing": 0.15,
    }
    backend.smoothed_gain = torch.ones(3, dtype=torch.float32)
    image = torch.zeros((2, 2, 3), dtype=torch.float32)

    gain = backend._estimate_gain(
        image, image, torch.zeros((2, 2), dtype=torch.bool))

    assert torch.isfinite(gain).all()
    assert torch.equal(gain, torch.ones(3))


def synthetic_parameters():
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    values = {
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
        "far_seam_blend_width_deg": 0.0,
        "depth_color_band_margin_deg": 0.0,
        "pointcloud_stride": 2,
        "depth_projection_stride": 1,
        "min_depth_m": 0.2,
        "max_depth_m": 15.0,
        "depth_discontinuity_abs_m": 0.08,
        "depth_discontinuity_relative": 0.04,
        "depth_splat_radius_px": 1,
        "depth_edge_splat_radius_px": 0,
        "occlusion_switch_margin_m": 0.05,
        "pytorch_output_space_splat": False,
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
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)
def test_output_space_splat_preserves_reference_edges_and_ranges():
    device = torch.device("cuda:0")
    width = 640
    height = 360
    fast_parameters = synthetic_parameters()
    reference_parameters = synthetic_parameters()
    fast_parameters["pytorch_output_space_splat"] = True
    reference_parameters["pytorch_output_space_splat"] = False
    for parameters in (fast_parameters, reference_parameters):
        for prefix in ("left", "right"):
            parameters.update(
                {
                    f"{prefix}_fx": 458.0,
                    f"{prefix}_fy": 458.0,
                    f"{prefix}_cx": 319.5,
                    f"{prefix}_cy": 179.5,
                    f"{prefix}_width": width,
                    f"{prefix}_height": height,
                }
            )
    fast_backend = TorchPanoramaBackend(fast_parameters, device)
    reference_backend = TorchPanoramaBackend(reference_parameters, device)
    fast_backend.ensure_projection(width, height)
    reference_backend.ensure_projection(width, height)

    rows = torch.arange(height, device=device, dtype=torch.float32)[:, None]
    columns = torch.arange(width, device=device, dtype=torch.float32)[None, :]
    depth = (4.0 + rows * 0.001 + columns * 0.0002).contiguous()
    depth[70:290, 190:460] = 1.25
    depth[130:210, 290:350] = 0.0
    depth[::37, ::41] = 0.0

    fast_keys, _ = fast_backend._project_depth(depth, fast_backend.left_model)
    reference_keys, _ = reference_backend._project_depth(
        depth, reference_backend.left_model
    )
    torch.cuda.synchronize()

    invalid = torch.iinfo(torch.int64).max
    fast_valid = fast_keys != invalid
    reference_valid = reference_keys != invalid
    union = fast_valid | reference_valid
    validity_mismatch = (fast_valid ^ reference_valid).sum().float()
    assert float(validity_mismatch / union.sum().clamp_min(1)) <= 0.001

    common = fast_valid & reference_valid
    fast_range = (fast_keys[common] >> 32) & 0xFFFF
    reference_range = (reference_keys[common] >> 32) & 0xFFFF
    range_error_mm = (fast_range - reference_range).abs().float()
    assert float(torch.quantile(range_error_mm, 0.99)) <= 2.0

    fast_source = fast_keys[common] & 0xFFFFFFFF
    reference_source = reference_keys[common] & 0xFFFFFFFF
    exact_source_ratio = (fast_source == reference_source).float().mean()
    assert float(exact_source_ratio) >= 0.98


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)
def test_depth_projection_stride_keeps_original_source_indices():
    device = torch.device("cuda:0")
    parameters = synthetic_parameters()
    parameters["depth_projection_stride"] = 2
    backend = TorchPanoramaBackend(parameters, device)
    backend.ensure_projection(64, 32)
    depth = torch.full((32, 64), 2.0, device=device)
    keys, accepted = backend._project_depth(depth, backend.left_model)
    torch.cuda.synchronize()

    valid_keys = keys[keys != torch.iinfo(torch.int64).max]
    source_indices = valid_keys & 0xFFFFFFFF
    assert accepted == 32 * 64 // 4
    assert torch.all((source_indices % 64) % 2 == 0)
    assert torch.all((source_indices // 64) % 2 == 0)
