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
                 enable_state=True, pedestrian_depth_overlap_margin=0.35,
                 pedestrian_depth_overlap_threshold=0.05):
        self.lidar = lidar or LidarObservationAnalyzer()
        self.camera = camera or CameraObservationAnalyzer()
        self.depth = depth or DepthConsistencyAnalyzer()
        self.enable_lidar = bool(enable_lidar)
        self.enable_camera = bool(enable_camera)
        self.enable_depth = bool(enable_depth)
        self.enable_state = bool(enable_state)
        self.pedestrian_depth_overlap_margin = float(
            pedestrian_depth_overlap_margin)
        self.pedestrian_depth_overlap_threshold = float(
            pedestrian_depth_overlap_threshold)

    def _back_project_detection(self, detection, depth_image, intrinsics,
                                camera_from_source):
        if depth_image is None or intrinsics is None or camera_from_source is None:
            return None
        image = np.asarray(depth_image)
        height, width = image.shape[:2]
        x1, y1, x2, y2 = map(float, detection.bbox)
        scale = float(np.clip(self.depth.config.roi_inner_scale, 0.05, 1.0))
        u, v = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        half_width = (x2 - x1) * scale * 0.5
        half_height = (y2 - y1) * scale * 0.5
        left = max(0, min(width - 1, int(u - half_width)))
        right = max(0, min(width, int(u + half_width)))
        top = max(0, min(height - 1, int(v - half_height)))
        bottom = max(0, min(height, int(v + half_height)))
        if right <= left or bottom <= top:
            return None
        values = image[top:bottom, left:right].reshape(-1).astype(np.float32)
        if image.dtype == np.uint16:
            values *= self.depth.config.depth_scale
        values = values[
            np.isfinite(values) &
            (values >= self.depth.config.min_depth_m) &
            (values <= self.depth.config.max_depth_m)]
        if len(values) < self.depth.config.min_valid_pixels:
            return None
        z = float(np.median(values))
        fx, fy, cx, cy = intrinsics
        if fx <= 0.0 or fy <= 0.0:
            return None
        point_camera = np.asarray([[(u - cx) * z / fx, (v - cy) * z / fy, z]])
        return transform_points(point_camera, np.linalg.inv(camera_from_source))[0]

    def _point_box_overlap(self, point, box):
        delta = np.asarray(point, dtype=np.float32) - np.asarray(box[:3])
        yaw = float(box[6])
        local = np.asarray([
            np.cos(yaw) * delta[0] + np.sin(yaw) * delta[1],
            -np.sin(yaw) * delta[0] + np.cos(yaw) * delta[1],
            delta[2]])
        half_size = np.asarray(box[3:6], dtype=np.float32) * 0.5
        half_size += self.pedestrian_depth_overlap_margin
        normalized = np.abs(local) / np.maximum(half_size, 1e-3)
        return max(0.0, 1.0 - float(np.max(normalized)))

    @staticmethod
    def _box_center_from_depth_point(point, box, camera_from_source):
        point_camera = transform_points(
            np.asarray(point).reshape(1, 3), camera_from_source)[0]
        center_camera = transform_points(
            np.asarray(box[:3]).reshape(1, 3), camera_from_source)[0]
        corners_camera = transform_points(
            box_corners_3d(box), camera_from_source)
        visible = corners_camera[corners_camera[:, 2] > 0.0, 2]
        if not len(visible) or point_camera[2] <= 0.0:
            return point
        center_z = center_camera[2] + point_camera[2] - float(np.min(visible))
        centered_camera = point_camera * (center_z / point_camera[2])
        return transform_points(
            centered_camera.reshape(1, 3), np.linalg.inv(camera_from_source))[0]

    def _match_pedestrians(self, regions, class_names, detections, depth_image,
                           camera_from_source, intrinsics, image_size,
                           camera_available):
        candidates = []
        projected = set()
        for box_index, (region, class_name) in enumerate(zip(regions, class_names)):
            if str(class_name).lower() not in {"pedestrian", "person"}:
                continue
            for detection_index, detection in enumerate(detections or ()):
                if not self.camera.class_compatible(class_name, detection.class_name):
                    continue
                point = self._back_project_detection(
                    detection, depth_image, intrinsics, camera_from_source)
                if point is None:
                    continue
                evidence = self.camera.analyze(
                    region, class_name, [detection], camera_from_source,
                    intrinsics, image_size, camera_available)
                if not evidence.supported:
                    continue
                projected.add(box_index)
                overlap = self._point_box_overlap(point, region)
                if overlap < self.pedestrian_depth_overlap_threshold:
                    continue
                evidence.detection_index = detection_index
                candidates.append((
                    evidence.score + overlap, box_index, detection_index,
                    evidence, point, overlap))

        # One camera person cannot support two different 3-D pedestrians.
        matches, used_boxes, used_detections = {}, set(), set()
        for _, box_index, detection_index, evidence, point, overlap in sorted(
                candidates, key=lambda item: item[0], reverse=True):
            if box_index in used_boxes or detection_index in used_detections:
                continue
            matches[box_index] = (evidence, point, overlap)
            used_boxes.add(box_index)
            used_detections.add(detection_index)
        return matches, projected

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
        boxes = np.asarray(boxes)
        regions = [expanded_box(
            box, self.lidar.config.margin_xy, self.lidar.config.margin_z)
            for box in boxes]
        pedestrian_matches, depth_projected = {}, set()
        if self.enable_camera and self.enable_depth:
            pedestrian_matches, depth_projected = self._match_pedestrians(
                regions, class_names, camera_detections, depth_image,
                camera_from_source, intrinsics, image_size, camera_available)

        results = []
        for box_index, (box, class_name, region) in enumerate(zip(
                boxes, class_names, regions)):
            lidar = self.lidar.analyze(
                box, class_name, lidar_points if self.enable_lidar else None)
            pedestrian_match = pedestrian_matches.get(box_index)
            camera = (pedestrian_match[0] if pedestrian_match else
                      self.camera.analyze(
                          region, class_name,
                          camera_detections if self.enable_camera else (),
                          camera_from_source if self.enable_camera else None,
                          intrinsics, image_size,
                          camera_available and self.enable_camera))
            if (str(class_name).lower() in {"pedestrian", "person"} and
                    self.enable_depth and pedestrian_match is None and
                    box_index in depth_projected and camera.available):
                camera.supported = False
                camera.state = CameraState.AVAILABLE_UNSUPPORTED
                camera.class_match = False
                camera.detection_score = 0.0
                camera.detection_index = -1
                camera.detection_class = ""
            depth = self.depth.analyze(
                region, (camera_detections[camera.detection_index].bbox
                         if pedestrian_match else camera.roi),
                depth_image if self.enable_depth else None,
                camera_from_source)
            camera_depth_center = None
            camera_depth_overlap = 0.0
            if pedestrian_match:
                estimated_center = self._box_center_from_depth_point(
                    pedestrian_match[1], region, camera_from_source)
                camera_depth_center = tuple(map(float, estimated_center))
                camera_depth_overlap = float(pedestrian_match[2])
            elif (camera.detection_index >= 0 and
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
                camera_depth_overlap=camera_depth_overlap,
                object_range=lidar.object_range,
                state=state,
                raw_state=raw_state,
                observation_score=score,
                projected_roi=camera.roi,
            ))
        return results
