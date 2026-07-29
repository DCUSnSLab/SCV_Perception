#!/usr/bin/env python3
"""One-shot RGB-D extrinsic calibration for the SSC dual RealSense rig.

The node observes one known ChArUco board in both cameras.  It combines:

* ArUco marker corners in each color image (2-D reprojection constraints)
* aligned depth sampled inside those markers (3-D metric constraints)

The resulting transform maps the native ``front`` color optical frame into the
native ``camera`` color optical frame.  The running panorama node is not
modified or stopped by this tool.  A calibration YAML is written only after
all geometry, reprojection, depth, and repeatability gates pass.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.optimize import least_squares
from sensor_msgs.msg import CameraInfo, Image


def stamp_seconds(message: Image) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def rotation_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> List[float]:
    """Return a normalized x, y, z, w quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            ) * 2.0
            quat = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            ) * 2.0
            quat = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            ) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    quat /= np.linalg.norm(quat)
    return [float(value) for value in quat]


def matrix_to_rpy_deg(rotation: np.ndarray) -> List[float]:
    """Fixed-axis XYZ roll, pitch, yaw in degrees."""
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return [math.degrees(value) for value in (roll, pitch, yaw)]


def as_float_list(array: np.ndarray) -> list:
    return np.asarray(array, dtype=np.float64).tolist()


def atomic_yaml_write(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".rig_calibration_", suffix=".yaml", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def fit_rigid_transform(
    object_points: np.ndarray, measured_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform from object_points to measured_points."""
    source = np.asarray(object_points, dtype=np.float64)
    target = np.asarray(measured_points, dtype=np.float64)
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T
    translation = target_center - rotation @ source_center
    return rotation, translation


@dataclass
class CameraSamples:
    name: str
    roi_min_x: float
    roi_max_x: float
    camera_matrix: Optional[np.ndarray] = None
    distortion: Optional[np.ndarray] = None
    info_width: int = 0
    info_height: int = 0
    latest_depth: Optional[np.ndarray] = None
    latest_depth_stamp: float = -math.inf
    corner_samples: Dict[int, List[Tuple[int, np.ndarray]]] = field(
        default_factory=dict
    )
    depth_samples: Dict[int, List[Tuple[int, float]]] = field(
        default_factory=dict
    )
    depth_inner_samples: Dict[int, List[Tuple[int, np.ndarray]]] = field(
        default_factory=dict
    )
    sampled_frames: int = 0
    detected_frames: int = 0
    last_sample_time: float = -math.inf


@dataclass
class AggregatedCamera:
    name: str
    camera_matrix: np.ndarray
    distortion: np.ndarray
    object_corners: np.ndarray
    image_corners: np.ndarray
    marker_ids: List[int]
    depth_object_centers: np.ndarray
    depth_camera_points: np.ndarray
    depth_marker_ids: List[int]
    shared_metric_points: Dict[Tuple[int, int], np.ndarray]


@dataclass
class SolveResult:
    label: str
    success: bool
    message: str
    rotation_camera_from_front: Optional[np.ndarray] = None
    translation_camera_from_front: Optional[np.ndarray] = None
    rotation_front_from_board: Optional[np.ndarray] = None
    translation_front_from_board: Optional[np.ndarray] = None
    metrics: dict = field(default_factory=dict)


class CharucoRigCalibrator(Node):
    def __init__(self) -> None:
        super().__init__("charuco_rig_calibrator")
        self._declare_parameters()
        self.bridge = CvBridge()
        self.finished = False
        self.capture_started_at: Optional[float] = None
        self.last_progress_log = -math.inf

        self.board_params_path = str(
            self.get_parameter("board_params_file").value
        )
        self.board_index = int(self.get_parameter("board_index").value)
        self.input_images_rotated_180 = bool(
            self.get_parameter("input_images_rotated_180").value
        )
        self.rotate_live_streams_180 = bool(
            self.get_parameter("rotate_live_streams_180").value
        )
        self.depth_scale_m = float(self.get_parameter("depth_scale_m").value)
        self.capture_duration_sec = float(
            self.get_parameter("capture_duration_sec").value
        )
        self.sample_period_sec = float(
            self.get_parameter("sample_period_sec").value
        )
        self.max_color_depth_delta_sec = (
            float(self.get_parameter("max_color_depth_delta_ms").value)
            * 1.0e-3
        )
        self.roi_scale = float(self.get_parameter("roi_scale").value)
        self.min_corner_observations = int(
            self.get_parameter("min_corner_observations").value
        )

        self._load_board()

        self.front = CameraSamples(
            name="front",
            roi_min_x=float(self.get_parameter("front_roi_min_x").value),
            roi_max_x=float(self.get_parameter("front_roi_max_x").value),
        )
        self.camera = CameraSamples(
            name="camera",
            roi_min_x=float(self.get_parameter("camera_roi_min_x").value),
            roi_max_x=float(self.get_parameter("camera_roi_max_x").value),
        )

        self._create_camera_subscriptions(self.front, "front")
        self._create_camera_subscriptions(self.camera, "camera")
        self.timer = self.create_timer(0.2, self._on_timer)

        self.get_logger().info(
            "SSC RGB-D rig calibration armed: board %d (IDs %d-%d), %.1f s"
            % (
                self.board_index,
                min(self.board_object_corners),
                max(self.board_object_corners),
                self.capture_duration_sec,
            )
        )
        self.get_logger().info(
            "This tool does not stop or modify the running panorama node."
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "front_color_topic": "/front/front/color/image_raw",
            "front_depth_topic": (
                "/front/front/aligned_depth_to_color/image_raw"
            ),
            "front_camera_info_topic": "/front/front/color/camera_info",
            "camera_color_topic": "/camera/camera/color/image_raw",
            "camera_depth_topic": (
                "/camera/camera/aligned_depth_to_color/image_raw"
            ),
            "camera_camera_info_topic": "/camera/camera/color/camera_info",
            "board_params_file": "/home/ssc/lidar_cam_calib/board_params.yaml",
            "board_index": 0,
            "capture_duration_sec": 20.0,
            "sample_period_sec": 0.10,
            "front_roi_min_x": 0.82,
            "front_roi_max_x": 1.0,
            "camera_roi_min_x": 0.0,
            "camera_roi_max_x": 0.18,
            "roi_scale": 3.0,
            "input_images_rotated_180": True,
            "rotate_live_streams_180": False,
            "depth_scale_m": 0.001,
            "max_color_depth_delta_ms": 80.0,
            "min_corner_observations": 5,
            "min_markers_per_camera": 4,
            "min_depth_markers_per_camera": 4,
            "min_common_depth_points": 6,
            "min_common_markers": 2,
            "min_common_second_axis_rms_m": 0.03,
            "marker_inner_fraction": 0.60,
            "depth_patch_radius_px": 2,
            "pixel_sigma": 1.5,
            "depth_xy_sigma_m": 0.03,
            "depth_z_sigma_m": 0.08,
            "nominal_native_relative_yaw_deg": 64.0,
            "max_rotation_prior_error_deg": 15.0,
            "housing_center_distance_m": 0.10,
            "min_optical_baseline_m": 0.02,
            "max_optical_baseline_m": 0.19,
            "max_reprojection_rms_px": 3.0,
            "max_depth_residual_median_m": 0.05,
            "max_depth_residual_p95_m": 0.12,
            "max_split_rotation_delta_deg": 2.0,
            "max_split_translation_delta_m": 0.03,
            "output_calibration_file": (
                "/home/ssc/SSC/src/perception/panorama_stitcher/config/"
                "rig_extrinsics.yaml"
            ),
            "output_report_file": (
                "/home/ssc/SSC/src/perception/panorama_stitcher/config/"
                "rig_calibration_report.yaml"
            ),
            "front_serial": "239122073045",
            "camera_serial": "239122071306",
            "device_model": "Intel RealSense D435if",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_board(self) -> None:
        with open(self.board_params_path, "r", encoding="utf-8") as stream:
            params = yaml.safe_load(stream)
        dictionary_name = str(params["aruco_dict"])
        dictionary_code = getattr(cv2.aruco, dictionary_name)
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(
            dictionary_code
        )
        squares_x = int(params["squares_x"])
        squares_y = int(params["squares_y"])
        square_length = float(params["square_length_m"])
        marker_length = float(params["marker_length_m"])
        id_stride = int(params["id_stride"])
        base_board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            self.aruco_dictionary,
        )
        marker_count = len(base_board.getIds())
        first_id = self.board_index * id_stride
        ids = np.arange(
            first_id, first_id + marker_count, dtype=np.int32
        ).reshape(-1, 1)
        self.board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            self.aruco_dictionary,
            ids,
        )
        board_ids = np.asarray(self.board.getIds()).reshape(-1)
        object_points = np.asarray(
            self.board.getObjPoints(), dtype=np.float64
        )
        self.board_object_corners = {
            int(marker_id): object_points[index]
            for index, marker_id in enumerate(board_ids)
        }
        self.board_metadata = {
            "params_file": os.path.abspath(self.board_params_path),
            "board_index": self.board_index,
            "aruco_dict": dictionary_name,
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_length_m": square_length,
            "marker_length_m": marker_length,
            "id_stride": id_stride,
            "marker_ids": [int(value) for value in board_ids],
        }
        detector_parameters = cv2.aruco.DetectorParameters()
        detector_parameters.minMarkerPerimeterRate = 0.005
        detector_parameters.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_SUBPIX
        )
        detector_parameters.cornerRefinementWinSize = 5
        detector_parameters.cornerRefinementMaxIterations = 50
        detector_parameters.cornerRefinementMinAccuracy = 0.01
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dictionary, detector_parameters
        )

    def _create_camera_subscriptions(
        self, samples: CameraSamples, prefix: str
    ) -> None:
        color_topic = str(
            self.get_parameter(f"{prefix}_color_topic").value
        )
        depth_topic = str(
            self.get_parameter(f"{prefix}_depth_topic").value
        )
        info_topic = str(
            self.get_parameter(f"{prefix}_camera_info_topic").value
        )
        self.create_subscription(
            CameraInfo,
            info_topic,
            lambda message, target=samples: self._on_camera_info(
                target, message
            ),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            depth_topic,
            lambda message, target=samples: self._on_depth(target, message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            color_topic,
            lambda message, target=samples: self._on_color(target, message),
            qos_profile_sensor_data,
        )

    def _on_camera_info(
        self, samples: CameraSamples, message: CameraInfo
    ) -> None:
        samples.camera_matrix = np.asarray(
            message.k, dtype=np.float64
        ).reshape(3, 3)
        samples.distortion = np.asarray(message.d, dtype=np.float64)
        samples.info_width = int(message.width)
        samples.info_height = int(message.height)

    def _on_depth(self, samples: CameraSamples, message: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(
                message, desired_encoding="passthrough"
            )
        except Exception as error:
            self.get_logger().warning(
                f"{samples.name}: depth conversion failed: {error}"
            )
            return
        if self.rotate_live_streams_180:
            depth = cv2.rotate(depth, cv2.ROTATE_180)
        samples.latest_depth = np.asarray(depth)
        samples.latest_depth_stamp = stamp_seconds(message)

    def _on_color(self, samples: CameraSamples, message: Image) -> None:
        if self.finished or samples.camera_matrix is None:
            return
        message_time = stamp_seconds(message)
        if (
            message_time - samples.last_sample_time
            < self.sample_period_sec
        ):
            return
        samples.last_sample_time = message_time
        samples.sampled_frames += 1
        try:
            image = self.bridge.imgmsg_to_cv2(
                message, desired_encoding="bgr8"
            )
        except Exception as error:
            self.get_logger().warning(
                f"{samples.name}: color conversion failed: {error}"
            )
            return
        if self.rotate_live_streams_180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        detections = self._detect_board_markers(image, samples)
        if not detections:
            return
        samples.detected_frames += 1
        sequence = samples.detected_frames - 1
        for marker_id, corners in detections:
            samples.corner_samples.setdefault(marker_id, []).append(
                (sequence, corners.copy())
            )
            depth_value, inner_depths = self._sample_marker_depths(
                samples, corners, image.shape[:2], message_time
            )
            if depth_value is not None:
                samples.depth_samples.setdefault(marker_id, []).append(
                    (sequence, depth_value)
                )
            if inner_depths is not None:
                samples.depth_inner_samples.setdefault(marker_id, []).append(
                    (sequence, inner_depths)
                )

    def _detect_board_markers(
        self, image: np.ndarray, samples: CameraSamples
    ) -> List[Tuple[int, np.ndarray]]:
        height, width = image.shape[:2]
        x_start = int(np.clip(round(samples.roi_min_x * width), 0, width - 1))
        x_end = int(np.clip(round(samples.roi_max_x * width), x_start + 1, width))
        roi = image[:, x_start:x_end]
        if self.roi_scale != 1.0:
            roi = cv2.resize(
                roi,
                None,
                fx=self.roi_scale,
                fy=self.roi_scale,
                interpolation=cv2.INTER_CUBIC,
            )
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return []
        results: List[Tuple[int, np.ndarray]] = []
        for detected_corners, marker_id_value in zip(corners, ids.reshape(-1)):
            marker_id = int(marker_id_value)
            if marker_id not in self.board_object_corners:
                continue
            full_corners = np.asarray(
                detected_corners, dtype=np.float64
            ).reshape(4, 2)
            full_corners /= self.roi_scale
            full_corners[:, 0] += x_start
            results.append((marker_id, full_corners))
        return results

    @staticmethod
    def _marker_inner_pixels(
        corners: np.ndarray, inner_fraction: float
    ) -> np.ndarray:
        margin = 0.5 * (1.0 - inner_fraction)
        canonical_corners = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        canonical_inner = np.array(
            [
                [margin, margin],
                [1.0 - margin, margin],
                [1.0 - margin, 1.0 - margin],
                [margin, 1.0 - margin],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(
            canonical_corners,
            np.asarray(corners, dtype=np.float32),
        )
        return cv2.perspectiveTransform(
            canonical_inner.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)

    @staticmethod
    def _depth_patch_median(
        depth: np.ndarray, pixel: np.ndarray, radius: int
    ) -> Optional[float]:
        height, width = depth.shape[:2]
        x_center = int(round(float(pixel[0])))
        y_center = int(round(float(pixel[1])))
        x_min = max(x_center - radius, 0)
        x_max = min(x_center + radius + 1, width)
        y_min = max(y_center - radius, 0)
        y_max = min(y_center + radius + 1, height)
        if x_min >= x_max or y_min >= y_max:
            return None
        values = np.asarray(
            depth[y_min:y_max, x_min:x_max], dtype=np.float64
        ).reshape(-1)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size < 3:
            return None
        return float(np.median(values))

    def _sample_marker_depths(
        self,
        samples: CameraSamples,
        color_corners: np.ndarray,
        color_shape: Sequence[int],
        color_stamp: float,
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        if samples.latest_depth is None:
            return None, None
        if (
            abs(samples.latest_depth_stamp - color_stamp)
            > self.max_color_depth_delta_sec
        ):
            return None, None
        depth = samples.latest_depth
        depth_height, depth_width = depth.shape[:2]
        color_height, color_width = color_shape
        scale = np.array(
            [depth_width / color_width, depth_height / color_height],
            dtype=np.float64,
        )
        polygon = color_corners * scale
        center = np.mean(polygon, axis=0)
        polygon = center + 0.45 * (polygon - center)
        polygon = np.round(polygon).astype(np.int32)
        x_min = max(int(np.min(polygon[:, 0])), 0)
        x_max = min(int(np.max(polygon[:, 0])) + 1, depth_width)
        y_min = max(int(np.min(polygon[:, 1])), 0)
        y_max = min(int(np.max(polygon[:, 1])) + 1, depth_height)
        if x_min >= x_max or y_min >= y_max:
            return None, None
        local_polygon = polygon - np.array([x_min, y_min], dtype=np.int32)
        mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_polygon, 255)
        values = depth[y_min:y_max, x_min:x_max][mask != 0]
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size < 5:
            center_depth_m = None
        else:
            center_depth_m = float(np.median(values) * self.depth_scale_m)
            if not 0.15 <= center_depth_m <= 20.0:
                center_depth_m = None

        inner_fraction = float(
            self.get_parameter("marker_inner_fraction").value
        )
        inner_pixels = (
            self._marker_inner_pixels(color_corners, inner_fraction) * scale
        )
        patch_radius = int(
            self.get_parameter("depth_patch_radius_px").value
        )
        inner_depths = np.full(4, np.nan, dtype=np.float64)
        for index, pixel in enumerate(inner_pixels):
            raw_depth = self._depth_patch_median(
                depth, pixel, patch_radius
            )
            if raw_depth is None:
                continue
            depth_m = raw_depth * self.depth_scale_m
            if 0.15 <= depth_m <= 20.0:
                inner_depths[index] = depth_m
        if np.count_nonzero(np.isfinite(inner_depths)) < 2:
            inner_depths_result = None
        else:
            inner_depths_result = inner_depths
        return center_depth_m, inner_depths_result

    def _on_timer(self) -> None:
        if self.finished:
            return
        now = time.monotonic()
        if (
            self.front.camera_matrix is None
            or self.camera.camera_matrix is None
        ):
            if now - self.last_progress_log > 2.0:
                self.get_logger().info("Waiting for both CameraInfo topics...")
                self.last_progress_log = now
            return
        if self.capture_started_at is None:
            self.capture_started_at = now
            self.get_logger().info("Both CameraInfo messages received; capture started.")
        elapsed = now - self.capture_started_at
        if now - self.last_progress_log > 2.0:
            self.get_logger().info(
                "capture %.1f/%.1f s | front frames=%d marker IDs=%s depth IDs=%s "
                "| camera frames=%d marker IDs=%s depth IDs=%s"
                % (
                    elapsed,
                    self.capture_duration_sec,
                    self.front.detected_frames,
                    sorted(self.front.corner_samples),
                    sorted(self.front.depth_samples),
                    self.camera.detected_frames,
                    sorted(self.camera.corner_samples),
                    sorted(self.camera.depth_samples),
                )
            )
            self.last_progress_log = now
        if elapsed < self.capture_duration_sec:
            return
        self.finished = True
        try:
            self._finalize()
        except Exception as error:
            self.get_logger().error(
                f"Calibration failed unexpectedly: {type(error).__name__}: {error}"
            )

    def _sequence_filter(
        self, split: Optional[int]
    ) -> Optional[Callable[[int], bool]]:
        if split is None:
            return None
        return lambda sequence: sequence % 2 == split

    def _aggregate_camera(
        self, samples: CameraSamples, split: Optional[int]
    ) -> AggregatedCamera:
        sequence_filter = self._sequence_filter(split)
        object_corner_blocks = []
        image_corner_blocks = []
        marker_ids = []
        depth_object_centers = []
        depth_camera_points = []
        depth_marker_ids = []
        shared_metric_points: Dict[Tuple[int, int], np.ndarray] = {}

        for marker_id in sorted(self.board_object_corners):
            corner_records = samples.corner_samples.get(marker_id, [])
            if sequence_filter is not None:
                corner_records = [
                    record
                    for record in corner_records
                    if sequence_filter(record[0])
                ]
            if len(corner_records) < self.min_corner_observations:
                continue
            stacked = np.stack([record[1] for record in corner_records])
            median_rotated = np.median(stacked, axis=0)
            median_native = self._to_native_pixels(
                median_rotated, samples.info_width, samples.info_height
            )
            object_corner_blocks.append(self.board_object_corners[marker_id])
            image_corner_blocks.append(median_native)
            marker_ids.append(marker_id)

            depth_records = samples.depth_samples.get(marker_id, [])
            if sequence_filter is not None:
                depth_records = [
                    record
                    for record in depth_records
                    if sequence_filter(record[0])
                ]
            if len(depth_records) < self.min_corner_observations:
                continue
            depth_m = float(np.median([record[1] for record in depth_records]))
            image_center = np.mean(median_native, axis=0)
            camera_point = self._back_project(
                image_center,
                depth_m,
                samples.camera_matrix,
                samples.distortion,
            )
            object_center = np.mean(
                self.board_object_corners[marker_id], axis=0
            )
            depth_object_centers.append(object_center)
            depth_camera_points.append(camera_point)
            depth_marker_ids.append(marker_id)

            inner_records = samples.depth_inner_samples.get(marker_id, [])
            if sequence_filter is not None:
                inner_records = [
                    record
                    for record in inner_records
                    if sequence_filter(record[0])
                ]
            if not inner_records:
                continue
            inner_native = self._to_native_pixels(
                self._marker_inner_pixels(
                    median_rotated,
                    float(
                        self.get_parameter("marker_inner_fraction").value
                    ),
                ),
                samples.info_width,
                samples.info_height,
            )
            inner_depth_matrix = np.stack(
                [record[1] for record in inner_records]
            )
            for corner_index in range(4):
                valid_depths = inner_depth_matrix[
                    np.isfinite(inner_depth_matrix[:, corner_index]),
                    corner_index,
                ]
                if valid_depths.size < self.min_corner_observations:
                    continue
                shared_metric_points[(marker_id, corner_index)] = (
                    self._back_project(
                        inner_native[corner_index],
                        float(np.median(valid_depths)),
                        samples.camera_matrix,
                        samples.distortion,
                    )
                )

        empty_corners = np.empty((0, 3), dtype=np.float64)
        empty_pixels = np.empty((0, 2), dtype=np.float64)
        return AggregatedCamera(
            name=samples.name,
            camera_matrix=np.asarray(samples.camera_matrix, dtype=np.float64),
            distortion=np.asarray(samples.distortion, dtype=np.float64),
            object_corners=(
                np.concatenate(object_corner_blocks, axis=0)
                if object_corner_blocks
                else empty_corners
            ),
            image_corners=(
                np.concatenate(image_corner_blocks, axis=0)
                if image_corner_blocks
                else empty_pixels
            ),
            marker_ids=marker_ids,
            depth_object_centers=(
                np.asarray(depth_object_centers, dtype=np.float64)
                if depth_object_centers
                else empty_corners
            ),
            depth_camera_points=(
                np.asarray(depth_camera_points, dtype=np.float64)
                if depth_camera_points
                else empty_corners
            ),
            depth_marker_ids=depth_marker_ids,
            shared_metric_points=shared_metric_points,
        )

    def _to_native_pixels(
        self, pixels: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        native = np.asarray(pixels, dtype=np.float64).copy()
        if self.input_images_rotated_180:
            native[:, 0] = (width - 1.0) - native[:, 0]
            native[:, 1] = (height - 1.0) - native[:, 1]
        return native

    @staticmethod
    def _back_project(
        pixel: np.ndarray,
        depth_m: float,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> np.ndarray:
        pixel_array = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
        normalized = cv2.undistortPoints(
            pixel_array, camera_matrix, distortion
        ).reshape(2)
        return np.array(
            [normalized[0] * depth_m, normalized[1] * depth_m, depth_m],
            dtype=np.float64,
        )

    def _solve_subset(
        self, label: str, split: Optional[int]
    ) -> SolveResult:
        front = self._aggregate_camera(self.front, split)
        camera = self._aggregate_camera(self.camera, split)
        minimum_markers = int(
            self.get_parameter("min_markers_per_camera").value
        )
        minimum_depth_markers = int(
            self.get_parameter("min_depth_markers_per_camera").value
        )
        # The two outward-looking cameras normally observe opposite halves of
        # the same board, so they need not share any marker IDs.  Every marker
        # ID still belongs to the same metric board coordinate frame.  Solve
        # board->camera for each side and derive camera<-front from those two
        # poses.  Common depth samples, when present, are only extra evidence.
        common_metric_keys = sorted(
            set(front.shared_metric_points)
            & set(camera.shared_metric_points)
        )
        common_marker_ids = sorted({key[0] for key in common_metric_keys})
        shared_board_marker_ids = sorted(
            set(front.marker_ids) & set(camera.marker_ids)
        )
        counts = {
            "calibration_mode": "shared_board_coordinate_frame",
            "front_markers": len(front.marker_ids),
            "camera_markers": len(camera.marker_ids),
            "front_depth_markers": len(front.depth_marker_ids),
            "camera_depth_markers": len(camera.depth_marker_ids),
            "front_marker_ids": front.marker_ids,
            "camera_marker_ids": camera.marker_ids,
            "front_depth_marker_ids": front.depth_marker_ids,
            "camera_depth_marker_ids": camera.depth_marker_ids,
            "shared_board_marker_ids": shared_board_marker_ids,
            "common_depth_point_count": len(common_metric_keys),
            "common_depth_marker_ids": common_marker_ids,
        }
        if (
            len(front.marker_ids) < minimum_markers
            or len(camera.marker_ids) < minimum_markers
            or len(front.depth_marker_ids) < minimum_depth_markers
            or len(camera.depth_marker_ids) < minimum_depth_markers
        ):
            return SolveResult(
                label=label,
                success=False,
                message=(
                    "insufficient board/depth observations: "
                    f"front={len(front.marker_ids)}/{len(front.depth_marker_ids)}, "
                    f"camera={len(camera.marker_ids)}/{len(camera.depth_marker_ids)}, "
                    f"same-ID overlap={len(shared_board_marker_ids)} markers"
                ),
                metrics=counts,
            )

        rotation_front_board, translation_front_board = fit_rigid_transform(
            front.depth_object_centers, front.depth_camera_points
        )
        rotation_camera_board, translation_camera_board = fit_rigid_transform(
            camera.depth_object_centers, camera.depth_camera_points
        )
        rotation_camera_front = (
            rotation_camera_board @ rotation_front_board.T
        )
        translation_camera_front = (
            translation_camera_board
            - rotation_camera_front @ translation_front_board
        )
        common_front_points = np.asarray(
            [front.shared_metric_points[key] for key in common_metric_keys],
            dtype=np.float64,
        ).reshape(-1, 3)
        common_camera_points = np.asarray(
            [camera.shared_metric_points[key] for key in common_metric_keys],
            dtype=np.float64,
        ).reshape(-1, 3)
        front_rvec = cv2.Rodrigues(rotation_front_board)[0].reshape(3)
        relative_rvec = cv2.Rodrigues(rotation_camera_front)[0].reshape(3)
        initial = np.concatenate(
            [
                front_rvec,
                translation_front_board,
                relative_rvec,
                translation_camera_front,
            ]
        )

        pixel_sigma = float(self.get_parameter("pixel_sigma").value)
        xy_sigma = float(self.get_parameter("depth_xy_sigma_m").value)
        z_sigma = float(self.get_parameter("depth_z_sigma_m").value)
        metric_sigma = np.array([xy_sigma, xy_sigma, z_sigma])

        def unpack(parameters: np.ndarray) -> Tuple[np.ndarray, ...]:
            rotation_fb = cv2.Rodrigues(parameters[0:3])[0]
            translation_fb = parameters[3:6]
            rotation_cf = cv2.Rodrigues(parameters[6:9])[0]
            translation_cf = parameters[9:12]
            rotation_cb = rotation_cf @ rotation_fb
            translation_cb = rotation_cf @ translation_fb + translation_cf
            return (
                rotation_fb,
                translation_fb,
                rotation_cf,
                translation_cf,
                rotation_cb,
                translation_cb,
            )

        def project(
            points: np.ndarray,
            rotation: np.ndarray,
            translation: np.ndarray,
            data: AggregatedCamera,
        ) -> np.ndarray:
            rotation_vector = cv2.Rodrigues(rotation)[0]
            projected, _ = cv2.projectPoints(
                points,
                rotation_vector,
                translation.reshape(3, 1),
                data.camera_matrix,
                data.distortion,
            )
            return projected.reshape(-1, 2)

        def residuals(parameters: np.ndarray) -> np.ndarray:
            (
                rotation_fb,
                translation_fb,
                rotation_cf,
                translation_cf,
                rotation_cb,
                translation_cb,
            ) = unpack(parameters)
            front_projected = project(
                front.object_corners,
                rotation_fb,
                translation_fb,
                front,
            )
            camera_projected = project(
                camera.object_corners,
                rotation_cb,
                translation_cb,
                camera,
            )
            front_metric = (
                (rotation_fb @ front.depth_object_centers.T).T
                + translation_fb
                - front.depth_camera_points
            ) / metric_sigma
            camera_metric = (
                (rotation_cb @ camera.depth_object_centers.T).T
                + translation_cb
                - camera.depth_camera_points
            ) / metric_sigma
            residual_blocks = [
                ((front_projected - front.image_corners) / pixel_sigma).ravel(),
                ((camera_projected - camera.image_corners) / pixel_sigma).ravel(),
                front_metric.ravel(),
                camera_metric.ravel(),
            ]
            if common_metric_keys:
                common_metric = (
                    (rotation_cf @ common_front_points.T).T
                    + translation_cf
                    - common_camera_points
                ) / metric_sigma
                residual_blocks.append(common_metric.ravel())
            return np.concatenate(residual_blocks)

        optimization = least_squares(
            residuals,
            initial,
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=1000,
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
        )
        (
            rotation_fb,
            translation_fb,
            rotation_cf,
            translation_cf,
            rotation_cb,
            translation_cb,
        ) = unpack(optimization.x)

        front_projected = project(
            front.object_corners, rotation_fb, translation_fb, front
        )
        camera_projected = project(
            camera.object_corners, rotation_cb, translation_cb, camera
        )
        front_pixel_errors = np.linalg.norm(
            front_projected - front.image_corners, axis=1
        )
        camera_pixel_errors = np.linalg.norm(
            camera_projected - camera.image_corners, axis=1
        )
        front_depth_errors = np.linalg.norm(
            (rotation_fb @ front.depth_object_centers.T).T
            + translation_fb
            - front.depth_camera_points,
            axis=1,
        )
        camera_depth_errors = np.linalg.norm(
            (rotation_cb @ camera.depth_object_centers.T).T
            + translation_cb
            - camera.depth_camera_points,
            axis=1,
        )
        all_depth_errors = np.concatenate(
            [front_depth_errors, camera_depth_errors]
        )
        if common_metric_keys:
            common_depth_errors = np.linalg.norm(
                (rotation_cf @ common_front_points.T).T
                + translation_cf
                - common_camera_points,
                axis=1,
            )
            common_centered = common_front_points - np.mean(
                common_front_points, axis=0
            )
            common_spread_axes = np.linalg.svd(
                common_centered, compute_uv=False
            ) / math.sqrt(len(common_front_points))
        else:
            common_depth_errors = np.empty((0,), dtype=np.float64)
            common_spread_axes = np.empty((0,), dtype=np.float64)
        expected_rotation = rotation_y(
            math.radians(
                float(
                    self.get_parameter(
                        "nominal_native_relative_yaw_deg"
                    ).value
                )
            )
        )
        rotation_prior_error = rotation_angle_deg(
            expected_rotation.T @ rotation_cf
        )
        singular_values = np.linalg.svd(
            optimization.jac, compute_uv=False
        )
        jacobian_condition = (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > 1.0e-12
            else math.inf
        )
        metrics = {
            **counts,
            "optimizer_success": bool(optimization.success),
            "optimizer_status": int(optimization.status),
            "optimizer_message": str(optimization.message),
            "optimizer_cost": float(optimization.cost),
            "optimizer_evaluations": int(optimization.nfev),
            "front_reprojection_rms_px": float(
                np.sqrt(np.mean(front_pixel_errors**2))
            ),
            "camera_reprojection_rms_px": float(
                np.sqrt(np.mean(camera_pixel_errors**2))
            ),
            "front_reprojection_p95_px": float(
                np.percentile(front_pixel_errors, 95)
            ),
            "camera_reprojection_p95_px": float(
                np.percentile(camera_pixel_errors, 95)
            ),
            "depth_residual_median_m": float(np.median(all_depth_errors)),
            "depth_residual_p95_m": float(
                np.percentile(all_depth_errors, 95)
            ),
            "front_depth_residuals_m": [
                float(value) for value in front_depth_errors
            ],
            "camera_depth_residuals_m": [
                float(value) for value in camera_depth_errors
            ],
            "common_depth_residual_median_m": (
                float(np.median(common_depth_errors))
                if common_depth_errors.size
                else None
            ),
            "common_depth_residual_p95_m": (
                float(np.percentile(common_depth_errors, 95))
                if common_depth_errors.size
                else None
            ),
            "common_depth_residuals_m": [
                float(value) for value in common_depth_errors
            ],
            "common_point_spread_rms_axes_m": [
                float(value) for value in common_spread_axes
            ],
            "optical_baseline_m": float(np.linalg.norm(translation_cf)),
            "rotation_prior_error_deg": float(rotation_prior_error),
            "jacobian_condition": jacobian_condition,
            "front_board_distance_m": float(translation_fb[2]),
            "camera_board_distance_m": float(translation_cb[2]),
        }
        return SolveResult(
            label=label,
            success=bool(optimization.success),
            message=str(optimization.message),
            rotation_camera_from_front=rotation_cf,
            translation_camera_from_front=translation_cf,
            rotation_front_from_board=rotation_fb,
            translation_front_from_board=translation_fb,
            metrics=metrics,
        )

    def _evaluate_gates(
        self, full: SolveResult, split_a: SolveResult, split_b: SolveResult
    ) -> Tuple[List[str], dict]:
        failures: List[str] = []
        repeatability = {
            "available": bool(split_a.success and split_b.success),
            "rotation_delta_deg": None,
            "translation_delta_m": None,
        }
        if not full.success:
            failures.append(f"full solve failed: {full.message}")
        if not split_a.success:
            failures.append(f"even-frame solve failed: {split_a.message}")
        if not split_b.success:
            failures.append(f"odd-frame solve failed: {split_b.message}")
        if not full.success:
            return failures, repeatability

        metrics = full.metrics
        max_reprojection = float(
            self.get_parameter("max_reprojection_rms_px").value
        )
        for camera_name in ("front", "camera"):
            value = float(
                metrics[f"{camera_name}_reprojection_rms_px"]
            )
            if value > max_reprojection:
                failures.append(
                    f"{camera_name} reprojection RMS {value:.3f}px "
                    f"> {max_reprojection:.3f}px"
                )

        max_depth_median = float(
            self.get_parameter("max_depth_residual_median_m").value
        )
        max_depth_p95 = float(
            self.get_parameter("max_depth_residual_p95_m").value
        )
        if metrics["depth_residual_median_m"] > max_depth_median:
            failures.append(
                "depth residual median %.3fm > %.3fm"
                % (
                    metrics["depth_residual_median_m"],
                    max_depth_median,
                )
            )
        if metrics["depth_residual_p95_m"] > max_depth_p95:
            failures.append(
                "depth residual p95 %.3fm > %.3fm"
                % (metrics["depth_residual_p95_m"], max_depth_p95)
            )
        if metrics["common_depth_point_count"] > 0:
            if metrics["common_depth_residual_median_m"] > max_depth_median:
                failures.append(
                    "common 3D residual median %.3fm > %.3fm"
                    % (
                        metrics["common_depth_residual_median_m"],
                        max_depth_median,
                    )
                )
            if metrics["common_depth_residual_p95_m"] > max_depth_p95:
                failures.append(
                    "common 3D residual p95 %.3fm > %.3fm"
                    % (
                        metrics["common_depth_residual_p95_m"],
                        max_depth_p95,
                    )
                )
            minimum_second_axis = float(
                self.get_parameter("min_common_second_axis_rms_m").value
            )
            spread_axes = metrics["common_point_spread_rms_axes_m"]
            if len(spread_axes) < 2:
                failures.append(
                    "common 3D points do not span two board axes"
                )
            elif float(spread_axes[1]) < minimum_second_axis:
                failures.append(
                    "common 3D points are nearly collinear: second-axis RMS "
                    "%.3fm < %.3fm"
                    % (float(spread_axes[1]), minimum_second_axis)
                )

        minimum_baseline = float(
            self.get_parameter("min_optical_baseline_m").value
        )
        maximum_baseline = float(
            self.get_parameter("max_optical_baseline_m").value
        )
        baseline = float(metrics["optical_baseline_m"])
        if not minimum_baseline <= baseline <= maximum_baseline:
            failures.append(
                f"optical baseline {baseline:.3f}m is outside "
                f"[{minimum_baseline:.3f}, {maximum_baseline:.3f}]m"
            )
        max_rotation_error = float(
            self.get_parameter("max_rotation_prior_error_deg").value
        )
        if metrics["rotation_prior_error_deg"] > max_rotation_error:
            failures.append(
                "relative rotation differs from +64deg native optical prior "
                "%.2fdeg > %.2fdeg"
                % (
                    metrics["rotation_prior_error_deg"],
                    max_rotation_error,
                )
            )
        if (
            metrics["front_board_distance_m"] <= 0.0
            or metrics["camera_board_distance_m"] <= 0.0
        ):
            failures.append("the solved board is behind at least one camera")

        if split_a.success and split_b.success:
            rotation_delta = rotation_angle_deg(
                split_a.rotation_camera_from_front.T
                @ split_b.rotation_camera_from_front
            )
            translation_delta = float(
                np.linalg.norm(
                    split_a.translation_camera_from_front
                    - split_b.translation_camera_from_front
                )
            )
            repeatability.update(
                {
                    "rotation_delta_deg": rotation_delta,
                    "translation_delta_m": translation_delta,
                }
            )
            max_rotation_delta = float(
                self.get_parameter("max_split_rotation_delta_deg").value
            )
            max_translation_delta = float(
                self.get_parameter("max_split_translation_delta_m").value
            )
            if rotation_delta > max_rotation_delta:
                failures.append(
                    "even/odd rotation delta %.2fdeg > %.2fdeg"
                    % (rotation_delta, max_rotation_delta)
                )
            if translation_delta > max_translation_delta:
                failures.append(
                    "even/odd translation delta %.3fm > %.3fm"
                    % (translation_delta, max_translation_delta)
                )
        return failures, repeatability

    def _camera_info_payload(self, samples: CameraSamples) -> dict:
        return {
            "width": int(samples.info_width),
            "height": int(samples.info_height),
            "camera_matrix": as_float_list(samples.camera_matrix),
            "distortion": as_float_list(samples.distortion),
        }

    def _transform_payload(
        self, rotation: np.ndarray, translation: np.ndarray
    ) -> dict:
        return {
            "rotation_matrix": as_float_list(rotation),
            "translation_m": as_float_list(translation),
            "quaternion_xyzw": matrix_to_quaternion_xyzw(rotation),
            "fixed_axis_rpy_deg": matrix_to_rpy_deg(rotation),
            "optical_horizontal_rotation_deg": math.degrees(
                math.atan2(rotation[0, 2], rotation[2, 2])
            ),
        }

    def _finalize(self) -> None:
        self.get_logger().info("Capture complete; solving full/even/odd datasets.")
        full = self._solve_subset("full", None)
        split_a = self._solve_subset("even_frames", 0)
        split_b = self._solve_subset("odd_frames", 1)
        failures, repeatability = self._evaluate_gates(
            full, split_a, split_b
        )
        accepted = len(failures) == 0
        report = {
            "schema_version": 1,
            "accepted": accepted,
            "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "purpose": "SSC dual D435if RGB-D rig extrinsic calibration",
            "evidence_boundary": (
                "The device model and approximately 0.10 m housing-center "
                "spacing are installation declarations. Camera intrinsics, "
                "images, and aligned depth are live measurements."
            ),
            "devices": {
                "front": {
                    "declared_model": str(
                        self.get_parameter("device_model").value
                    ),
                    "serial": str(self.get_parameter("front_serial").value),
                    **self._camera_info_payload(self.front),
                },
                "camera": {
                    "declared_model": str(
                        self.get_parameter("device_model").value
                    ),
                    "serial": str(self.get_parameter("camera_serial").value),
                    **self._camera_info_payload(self.camera),
                },
            },
            "board": self.board_metadata,
            "capture": {
                "duration_sec": self.capture_duration_sec,
                "input_images_rotated_180": self.input_images_rotated_180,
                "front_detected_frames": self.front.detected_frames,
                "camera_detected_frames": self.camera.detected_frames,
            },
            "installation_gates": {
                "nominal_native_relative_yaw_deg": float(
                    self.get_parameter(
                        "nominal_native_relative_yaw_deg"
                    ).value
                ),
                "housing_center_distance_m": float(
                    self.get_parameter("housing_center_distance_m").value
                ),
                "allowed_optical_baseline_m": [
                    float(
                        self.get_parameter("min_optical_baseline_m").value
                    ),
                    float(
                        self.get_parameter("max_optical_baseline_m").value
                    ),
                ],
                "note": (
                    "The 50 mm D435 stereo baseline is internal to one camera "
                    "and is not used as the inter-camera baseline."
                ),
            },
            "solves": {
                result.label: {
                    "success": result.success,
                    "message": result.message,
                    "metrics": result.metrics,
                }
                for result in (full, split_a, split_b)
            },
            "repeatability": repeatability,
            "gate_failures": failures,
        }

        if full.success:
            native_rotation = full.rotation_camera_from_front
            native_translation = full.translation_camera_from_front
            rotate_180 = np.diag([-1.0, -1.0, 1.0])
            rotated_rotation = (
                rotate_180 @ native_rotation @ rotate_180.T
            )
            rotated_translation = rotate_180 @ native_translation
            report["candidate_transform"] = {
                "convention": (
                    "p_camera_color_optical = R * "
                    "p_front_color_optical + t"
                ),
                "native_camera_info_optical_frames": self._transform_payload(
                    native_rotation, native_translation
                ),
                "rotation_180_image_optical_frames": self._transform_payload(
                    rotated_rotation, rotated_translation
                ),
            }

        report_path = str(
            self.get_parameter("output_report_file").value
        )
        atomic_yaml_write(report_path, report)
        self.get_logger().info(f"Audit report written: {report_path}")

        if not accepted:
            self.get_logger().error(
                "CALIBRATION REJECTED; rig_extrinsics.yaml was not written."
            )
            for failure in failures:
                self.get_logger().error(f"gate: {failure}")
            return

        calibration = {
            "schema_version": 1,
            "accepted": True,
            "created_local_time": report["created_local_time"],
            "front_serial": str(self.get_parameter("front_serial").value),
            "camera_serial": str(self.get_parameter("camera_serial").value),
            "board": self.board_metadata,
            "convention": (
                "p_camera_color_optical = R * p_front_color_optical + t"
            ),
            "native_camera_info_optical_frames": report[
                "candidate_transform"
            ]["native_camera_info_optical_frames"],
            "rotation_180_image_optical_frames": report[
                "candidate_transform"
            ]["rotation_180_image_optical_frames"],
            "validation": {
                "full_metrics": full.metrics,
                "repeatability": repeatability,
            },
        }
        calibration_path = str(
            self.get_parameter("output_calibration_file").value
        )
        atomic_yaml_write(calibration_path, calibration)
        self.get_logger().info(
            f"CALIBRATION ACCEPTED and written: {calibration_path}"
        )


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = CharucoRigCalibrator()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.2)
        if node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().warning(
            "Calibration interrupted; no calibration was accepted."
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
