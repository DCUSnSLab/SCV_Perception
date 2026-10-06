"""CSV logging dedicated to observation-state ablation experiments."""

import csv
from pathlib import Path


FIELDS = [
    'timestamp', 'frame_id', 'track_id', 'class_id', 'x', 'y', 'z',
    'detector_confidence', 'lidar_point_count', 'lidar_point_density',
    'lidar_score', 'camera_available', 'camera_visible',
    'camera_supported', 'camera_iou', 'camera_score', 'depth_available',
    'depth_consistency', 'observation_state', 'observation_score',
    'association_cost', 'matched', 'track_age_sec', 'miss_count',
    'observation_analysis_ms', 'association_ms', 'tracker_update_ms',
    'total_ms',
]


class ObservationCsvWriter:
    def __init__(self, output_dir, sequence_id):
        path = Path(output_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        self.path = path / f'{sequence_id}_observability.csv'
        self._file = self.path.open('w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def write_tracks(self, timestamp, frame_id, tracks, timings,
                     expected_dt):
        for track in tracks:
            observation = track.get('observation')
            box = track['box']
            row = {
                'timestamp': f'{float(timestamp):.9f}',
                'frame_id': int(frame_id),
                'track_id': int(track['id']),
                'class_id': int(track['label']),
                'x': float(box[0]), 'y': float(box[1]), 'z': float(box[2]),
                'detector_confidence': float(track['score']),
                'lidar_point_count': (
                    observation.lidar_point_count if observation else 0),
                'lidar_point_density': (
                    observation.lidar_point_density if observation else 0.0),
                'lidar_score': observation.lidar_score if observation else 0.0,
                'camera_available': int(
                    observation.camera_available if observation else False),
                'camera_visible': int(
                    observation.camera_visible if observation else False),
                'camera_supported': int(
                    observation.camera_supported if observation else False),
                'camera_iou': observation.camera_iou if observation else 0.0,
                'camera_score': observation.camera_score if observation else 0.0,
                'depth_available': int(
                    observation.depth_available if observation else False),
                'depth_consistency': (
                    '' if observation is None or
                    observation.depth_consistency is None
                    else observation.depth_consistency),
                'observation_state': track['observation_state'],
                'observation_score': track['observation_confidence'],
                'association_cost': track['association_cost'],
                'matched': int(track['matched']),
                'track_age_sec': track['age_sec'],
                'miss_count': int(round(track['miss_age'] / expected_dt)),
                'observation_analysis_ms': timings.get(
                    'observation_analysis_ms', 0.0),
                'association_ms': timings.get('association_ms', 0.0),
                'tracker_update_ms': timings.get('tracker_update_ms', 0.0),
                'total_ms': timings.get('total_ms', 0.0),
            }
            self._writer.writerow(row)
        self._file.flush()
