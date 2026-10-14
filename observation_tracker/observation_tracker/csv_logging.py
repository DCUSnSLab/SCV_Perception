"""Per-track CSV rows for threshold tuning and ablation studies."""

import csv
from pathlib import Path


FIELDS = [
    "timestamp", "frame_id", "track_id", "class",
    "detection_3d_available", "detection_3d_confidence",
    "detection_2d_available", "detection_2d_confidence",
    "object_distance", "lidar_point_count", "lidar_density",
    "lidar_centroid_error", "lidar_state", "static_dynamic_state",
    "camera_available", "camera_supported", "camera_iou",
    "camera_center_distance", "camera_class_match", "depth_state",
    "depth_error", "depth_valid_pixel_count", "depth_support_pixel_count",
    "depth_support_ratio", "camera_depth_overlap", "observation_state",
    "track_status", "hit_count",
    "miss_count", "track_confidence", "recovery_count",
    "occlusion_count", "occluded_by_track_id", "occlusion_ratio",
    "matched_detection_id", "association_stage", "association_cost",
    "observation_analysis_ms", "association_ms", "tracker_update_ms",
    "total_ms",
]


class ObservationCsvLogger:
    def __init__(self, output_dir, sequence_id, track_provider=None):
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{sequence_id}_observation_tracks.csv"
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        self._track_provider = track_provider

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def write_tracks(self, timestamp, frame_id, tracks, timings, expected_dt):
        del expected_dt
        if self._track_provider is not None:
            tracks = self._track_provider()
        for track in tracks:
            observation = track.get("observation")
            row = {
                "timestamp": f"{float(timestamp):.9f}",
                "frame_id": int(frame_id),
                "track_id": int(track["id"]),
                "class": int(track["label"]),
                "detection_3d_available": int(track["was_detected"]),
                "detection_3d_confidence": float(track["score"]),
                "detection_2d_available": int(
                    observation is not None and
                    observation.camera_detection_index >= 0),
                "detection_2d_confidence": (
                    observation.camera_detection_score if observation else 0.0),
                "object_distance": observation.object_range if observation else 0.0,
                "lidar_point_count": observation.lidar_point_count if observation else 0,
                "lidar_density": observation.lidar_point_density if observation else 0.0,
                "lidar_centroid_error": (
                    "" if observation is None or
                    observation.lidar_centroid_error is None
                    else observation.lidar_centroid_error),
                "lidar_state": (
                    observation.lidar_state.value if observation else "UNAVAILABLE"),
                "static_dynamic_state": (
                    "dynamic" if track["is_dynamic"] else "static"),
                "camera_available": int(
                    observation.camera_available if observation else False),
                "camera_supported": int(
                    observation.camera_supported if observation else False),
                "camera_iou": observation.camera_iou if observation else 0.0,
                "camera_center_distance": (
                    "" if observation is None or
                    observation.camera_center_distance is None
                    else observation.camera_center_distance),
                "camera_class_match": int(
                    observation.camera_class_match if observation else False),
                "depth_state": (
                    observation.depth_state.value if observation else "UNAVAILABLE"),
                "depth_error": (
                    "" if observation is None or observation.depth_error is None
                    else observation.depth_error),
                "depth_valid_pixel_count": (
                    observation.depth_valid_pixel_count if observation else 0),
                "depth_support_pixel_count": (
                    observation.depth_support_pixel_count if observation else 0),
                "depth_support_ratio": (
                    observation.depth_support_ratio if observation else 0.0),
                "camera_depth_overlap": (
                    observation.camera_depth_overlap if observation else 0.0),
                "observation_state": track["observation_state"],
                "track_status": track["track_status"],
                "hit_count": int(track["hit_count"]),
                "miss_count": int(track["miss_count"]),
                "track_confidence": float(track["track_confidence"]),
                "recovery_count": int(track["recovery_count"]),
                "occlusion_count": int(track.get("occlusion_count", 0)),
                "occluded_by_track_id": int(
                    track.get("occluded_by_track_id", -1)),
                "occlusion_ratio": float(track.get("occlusion_ratio", 0.0)),
                "matched_detection_id": int(track["matched_detection_id"]),
                "association_stage": track["association_stage"],
                "association_cost": track["association_cost"],
                "observation_analysis_ms": timings.get(
                    "observation_analysis_ms", 0.0),
                "association_ms": timings.get("association_ms", 0.0),
                "tracker_update_ms": timings.get("tracker_update_ms", 0.0),
                "total_ms": timings.get("total_ms", 0.0),
            }
            self._writer.writerow(row)
        self._file.flush()
