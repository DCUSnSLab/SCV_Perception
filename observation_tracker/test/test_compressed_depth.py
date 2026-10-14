import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

from observation_tracker.camera_observation import CameraDetection
from observation_tracker.depth_consistency import DepthConsistencyConfig
from observation_tracker.observation_tracker_node import (
    decode_compressed_depth, depth_detection_box)


def test_decode_16uc1_compressed_depth_png():
    expected = np.array([[0, 1000], [2500, 65535]], dtype=np.uint16)
    success, png = cv2.imencode(".png", expected)
    assert success
    message = CompressedImage()
    message.format = "16UC1; compressedDepth png"
    message.data = bytes(12) + png.tobytes()
    decoded = decode_compressed_depth(message)
    assert decoded.dtype == np.uint16
    assert np.array_equal(decoded, expected)


def test_depth_detection_box_back_projects_2d_extent():
    detection = CameraDetection((40, 30, 60, 70), "person", 0.9)
    depth = np.full((100, 100), 5000, np.uint16)
    source_from_camera = np.eye(4)
    source_from_camera[:3, :3] = np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    position, size, depth_m = depth_detection_box(
        detection, depth, (100, 100, 50, 50), source_from_camera,
        DepthConsistencyConfig())
    assert np.allclose(position, (5.25, 0.0, 0.0))
    assert np.allclose(size, (0.5, 1.0, 2.0))
    assert depth_m == 5.0
