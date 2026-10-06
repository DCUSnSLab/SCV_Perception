import csv
from pathlib import Path


def format_vector(values):
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


class TrackerCsvWriter:
    def __init__(self, sequence_id, detection_output_dir, track_output_dir):
        self.sequence_id = str(sequence_id)
        self.detection_output_path = Path(detection_output_dir) / f"{self.sequence_id}_detections.csv"
        self.track_output_path = Path(track_output_dir) / f"{self.sequence_id}_tracks.csv"

        self.detection_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.track_output_path.parent.mkdir(parents=True, exist_ok=True)

        self.detection_csv_file = self.detection_output_path.open("w", newline="", encoding="utf-8")
        self.track_csv_file = self.track_output_path.open("w", newline="", encoding="utf-8")
        self.detection_csv_writer = csv.writer(self.detection_csv_file)
        self.track_csv_writer = csv.writer(self.track_csv_file)

        self.detection_csv_writer.writerow([
            "sequence_id",
            "frame_id",
            "timestamp",
            "class_id",
            "score",
            "box_center",
            "box_size",
            "yaw",
        ])
        self.track_csv_writer.writerow([
            "sequence_id",
            "frame_id",
            "timestamp",
            "track_id",
            "class_id",
            "score",
            "box_center",
            "box_size",
            "yaw",
            "is_detected",
        ])

    def close(self):
        if getattr(self, "detection_csv_file", None):
            self.detection_csv_file.close()
            self.detection_csv_file = None
        if getattr(self, "track_csv_file", None):
            self.track_csv_file.close()
            self.track_csv_file = None

    def save_detections(self, frame_id, timestamp, detections):
        if len(detections) == 0:
            return

        for det in detections:
            self.detection_csv_writer.writerow([
                self.sequence_id,
                int(frame_id),
                f"{float(timestamp):.9f}",
                int(det[7]),
                f"{float(det[8]):.6f}",
                format_vector(det[0:3]),
                format_vector(det[3:6]),
                f"{float(det[6]):.6f}",
            ])
        self.detection_csv_file.flush()

    def save_tracks(self, frame_id, timestamp, tracked_objects):
        if len(tracked_objects) == 0:
            return

        for track in tracked_objects:
            box = track["box"]
            self.track_csv_writer.writerow([
                self.sequence_id,
                int(frame_id),
                f"{float(timestamp):.9f}",
                int(track["id"]),
                int(track["label"]),
                f"{float(track['score']):.6f}",
                format_vector(box[0:3]),
                format_vector(box[3:6]),
                f"{float(box[6]):.6f}",
                int(bool(track["was_detected"])),
            ])
        self.track_csv_file.flush()
