#!/usr/bin/env python3
"""Rank Jay tracker parameters from detector CSVs without ground truth."""

import ast
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from pcdet_tracker.jay_tracker import KittiHzTracker, TimeAwareTrack


def load_sequences(directory):
    sequences = {}
    for path in sorted(Path(directory).glob("*_detections.csv")):
        frames, times = defaultdict(list), {}
        with path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                frame = int(row["frame_id"])
                frames[frame].append(
                    ast.literal_eval(row["box_center"])
                    + ast.literal_eval(row["box_size"])
                    + [float(row["yaw"]), int(row["class_id"]), float(row["score"])]
                )
                times[frame] = float(row["timestamp"])
        sequences[path.stem.replace("_detections", "")] = [
            (frame, times[frame], np.asarray(detections, dtype=np.float32))
            for frame, detections in sorted(frames.items())
        ]
    return sequences


def evaluate(sequences, params):
    age, distance, spawn_score = params
    total = defaultdict(float)
    per_sequence = {}
    for name, frames in sequences.items():
        TimeAwareTrack._next_id = 1
        tracker = KittiHzTracker(
            expected_dt=0.1,
            max_age_frames=age,
            confirm_frames=2,
            match_distance=distance,
            min_spawn_score=spawn_score,
        )
        ended, max_hits = [], defaultdict(int)
        stats = defaultdict(float)
        for frame, timestamp, detections in frames:
            before = {
                track.id: (track.cls_id, track.box[:2].copy())
                for track in tracker.tracks
            }
            tracker.update(detections, timestamp=timestamp)
            active_ids = {track.id for track in tracker.tracks}
            ended.extend(
                (frame, cls, position)
                for track_id, (cls, position) in before.items()
                if track_id not in active_ids
            )
            ended = [item for item in ended if frame - item[0] <= 10]
            for event in tracker.last_events:
                if event.get("matched"):
                    stats["matched"] += 1
                    stats["high_cost"] += event["association_cost"] > 4.0
                if event.get("spawned"):
                    stats["spawns"] += 1
                    detection = detections[event["detection_index"]]
                    stats["handoffs"] += any(
                        cls == int(detection[7])
                        and np.linalg.norm(position - detection[:2]) < 3.0
                        for _, cls, position in ended
                    )
            stats["detections"] += len(detections)
            for track in tracker.tracks:
                max_hits[track.id] = max(max_hits[track.id], track.hits)
        stats["short"] = sum(hits <= 2 for hits in max_hits.values())
        stats["covered"] = stats["matched"] + stats["spawns"]
        per_sequence[name] = dict(stats)
        for key, value in stats.items():
            total[key] += value

    total["coverage"] = total["covered"] / max(total["detections"], 1)
    total["score"] = (
        6 * total["handoffs"]
        + 2 * total["short"]
        + total["high_cost"]
        + 8 * max(0, 0.98 * total["detections"] - total["covered"])
    )
    return dict(total), per_sequence


def main():
    directory = Path("results/tuning/detections")
    sequences = load_sequences(directory)
    rows = []
    for params in itertools.product(
        (3, 5, 7, 9),
        (1.8, 2.2, 2.4, 2.8, 3.2),
        (0.2, 0.4, 0.6),
    ):
        totals, per_sequence = evaluate(sequences, params)
        rows.append((totals["score"], params, totals, per_sequence))
    rows.sort(key=lambda item: item[0])

    for rank, (_, params, totals, _) in enumerate(rows[:10], 1):
        age, distance, spawn = params
        print(
            f"{rank:2d} age={age} distance={distance:.1f} spawn={spawn:.1f} "
            f"handoff={totals['handoffs']:.0f} short={totals['short']:.0f} "
            f"tracks={totals['spawns']:.0f} high_cost={totals['high_cost']:.0f} "
            f"coverage={totals['coverage']:.4f} score={totals['score']:.1f}"
        )

    baseline = next(row for row in rows if row[1] == (5, 2.4, 0.2))
    print("baseline", baseline[1], baseline[2])
    print("best per sequence", rows[0][3])


if __name__ == "__main__":
    main()
