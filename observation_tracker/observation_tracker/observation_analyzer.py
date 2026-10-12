"""Compose independent LiDAR, camera and depth evidence per object."""

import numpy as np

from pcdet_tracker.observation_analyzer import box_corners_3d, transform_points

from .camera_observation import CameraObservationAnalyzer
from .depth_consistency import DepthConsistencyAnalyzer
from .lidar_observation import LidarObservationAnalyzer
from .observation_state import (
    CameraState, DepthState, LidarState, ObservationClass, ObservationResult)
from .observation_region import expanded_box


class ObservationAnalyzer:
    def __init__(self, lidar=None, camera=None, depth=None,
                 enable_lidar=True, enable_camera=True, enable_depth=True,
                 enable_state=True):
        self.lidar = lidar or LidarObservationAnalyzer()
        self.camera = camera or CameraObservationAnalyzer()
        self.depth = depth or DepthConsistencyAnalyzer()
        self.enable_lidar = bool(enable_lidar)
        self.enable_camera = bool(enable_camera)
        self.enable_depth = bool(enable_depth)
        self.enable_state = bool(enable_state)

    def _classify(self, lidar, camera, depth):
        lidar_strong = lidar.state == LidarState.STRONG
        camera_supported = camera.state == CameraState.AVAILABLE_SUPPORTED
        depth_safe = depth.state in {DepthState.CONSISTENT, DepthState.UNAVAILABLE}
        if lidar_strong and camera_supported and depth_safe:
            return ObservationClass.MULTIMODAL_STRONG.value
        if lidar_strong:
            return ObservationClass.LIDAR_DOMINANT.value
        camera_depth_ok = (
            camera_supported and
            (not self.enable_depth or depth.state == DepthState.CONSISTENT))
        if camera_depth_ok:
            return ObservationClass.CAMERA_DOMINANT.value
        return ObservationClass.WEAK_OBSERVATION.value

    def analyze(self, boxes, class_names, lidar_points=None,
                camera_detections=None, camera_from_source=None,
                intrinsics=None, image_size=None, camera_available=False,
                depth_image=None):
        results = []
        for box, class_name in zip(np.asarray(boxes), class_names):
            lidar = self.lidar.analyze(
                box, class_name, lidar_points if self.enable_lidar else None)
            region = expanded_box(
                box, self.lidar.config.margin_xy, self.lidar.config.margin_z)
            camera = self.camera.analyze(
                region, class_name,
                camera_detections if self.enable_camera else (),
                camera_from_source if self.enable_camera else None,
                intrinsics, image_size,
                camera_available and self.enable_camera)
            depth = self.depth.analyze(
                region, camera.roi,
                depth_image if self.enable_depth else None,
                camera_from_source)
            camera_depth_center = None
            if (camera.detection_index >= 0 and
                    depth.state == DepthState.CONSISTENT and
                    depth.median_depth is not None and intrinsics is not None):
                detection = camera_detections[camera.detection_index]
                u = (detection.bbox[0] + detection.bbox[2]) * 0.5
                v = (detection.bbox[1] + detection.bbox[3]) * 0.5
                center_camera = transform_points(
                    np.asarray(region[:3]).reshape(1, 3),
                    camera_from_source)[0]
                corners_camera = transform_points(
                    box_corners_3d(region), camera_from_source)
                visible = corners_camera[corners_camera[:, 2] > 0.0, 2]
                if len(visible):
                    center_z = center_camera[2] + (
                        depth.median_depth - float(np.min(visible)))
                    fx, fy, cx, cy = intrinsics
                    point_camera = np.asarray([[
                        (u - cx) * center_z / fx,
                        (v - cy) * center_z / fy,
                        center_z]])
                    point_source = transform_points(
                        point_camera, np.linalg.inv(camera_from_source))[0]
                    camera_depth_center = tuple(map(float, point_source))
            raw_state = self._classify(lidar, camera, depth)
            state = raw_state if self.enable_state else (
                ObservationClass.LIDAR_DOMINANT.value
                if lidar.state == LidarState.STRONG
                else ObservationClass.WEAK_OBSERVATION.value)
            scores = []
            if lidar.available and lidar.visible:
                scores.append(lidar.score)
            if camera.available:
                scores.append(camera.score)
            if depth.consistency is not None:
                scores.append(depth.consistency)
            score = float(np.mean(scores)) if scores else 0.0
            results.append(ObservationResult(
                lidar_state=lidar.state,
                lidar_point_count=lidar.count,
                lidar_point_density=lidar.density,
                lidar_centroid_error=lidar.centroid_error,
                lidar_score=lidar.score,
                lidar_available=lidar.available,
                lidar_visible=lidar.visible,
                lidar_supported=lidar.state == LidarState.STRONG,
                camera_state=camera.state,
                camera_available=camera.available,
                camera_visible=camera.visible,
                camera_supported=camera.supported,
                camera_iou=camera.iou,
                camera_center_distance=camera.center_distance,
                camera_class_match=camera.class_match,
                camera_detection_score=camera.detection_score,
                camera_detection_index=camera.detection_index,
                camera_detection_class=camera.detection_class,
                camera_score=camera.score,
                depth_state=depth.state,
                depth_available=depth.available,
                depth_error=depth.error,
                depth_consistency=depth.consistency,
                depth_valid_pixel_count=depth.valid_pixel_count,
                depth_support_pixel_count=depth.support_pixel_count,
                depth_support_ratio=depth.support_ratio,
                camera_depth_center=camera_depth_center,
                object_range=lidar.object_range,
                state=state,
                raw_state=raw_state,
                observation_score=score,
                projected_roi=camera.roi,
            ))
        return results
