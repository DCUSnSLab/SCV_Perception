"""Evaluate tracker CSV files against SUSTechPOINTS labels.

GT and prediction IDs are deliberately never compared by their literal value.
Boxes are associated by class and XY position first; identity metrics are then
computed from the resulting associations.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Box:
    frame: int
    identity: str
    class_name: str
    x: float
    y: float
    yaw: float
    z: float = 0.0
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0


def canonical_class(name: str) -> str:
    value = name.strip().lower()
    if value in {"car", "vehicle", "1"}:
        return "Car"
    if value in {"pedestrian", "ped", "person", "2"}:
        return "Pedestrian"
    return name.strip()


def load_sustech_labels(label_dir: Path) -> dict[int, list[Box]]:
    frames: dict[int, list[Box]] = {}
    for path in sorted(label_dir.glob("*.json")):
        frame = int(path.stem)
        boxes = []
        for item in json.loads(path.read_text()):
            psr = item["psr"]
            boxes.append(Box(
                frame=frame,
                identity=str(item["obj_id"]),
                class_name=canonical_class(str(item["obj_type"])),
                x=float(psr["position"]["x"]),
                y=float(psr["position"]["y"]),
                yaw=float(psr["rotation"]["z"]),
                z=float(psr["position"]["z"]),
                dx=float(psr["scale"]["x"]),
                dy=float(psr["scale"]["y"]),
                dz=float(psr["scale"]["z"]),
            ))
        frames[frame] = boxes
    return frames


def load_prediction_csv(path: Path) -> dict[int, list[Box]]:
    frames: dict[int, list[Box]] = defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            live_format = "frame_id" in row
            frame = int(row["frame_id"] if live_format else row["frame"])
            if live_format:
                center = ast.literal_eval(row["box_center"])
                size = ast.literal_eval(row["box_size"])
                class_name = row["class_id"]
                x, y = float(center[0]), float(center[1])
                z, dx, dy, dz = (float(center[2]), *(float(value) for value in size))
            else:
                class_name = row["class_name"]
                x, y = float(row["x"]), float(row["y"])
                z, dx, dy, dz = tuple(float(row[key]) for key in ("z", "dx", "dy", "dz"))
            frames[frame].append(Box(
                frame=frame,
                identity=str(row["track_id"]),
                class_name=canonical_class(class_name),
                x=x,
                y=y,
                yaw=float(row["yaw"]),
                z=z,
                dx=dx,
                dy=dy,
                dz=dz,
            ))
    return dict(frames)


def wrapped_angle_error(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def associate(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    gate: float,
) -> tuple[list[tuple[Box, Box, float]], int, int]:
    if not gt_boxes or not pred_boxes:
        return [], len(gt_boxes), len(pred_boxes)
    distances = np.array([
        [math.hypot(gt.x - pred.x, gt.y - pred.y) for pred in pred_boxes]
        for gt in gt_boxes
    ])
    rows, cols = linear_sum_assignment(distances)
    matches = [
        (gt_boxes[row], pred_boxes[col], float(distances[row, col]))
        for row, col in zip(rows, cols)
        if distances[row, col] <= gate
    ]
    return matches, len(gt_boxes) - len(matches), len(pred_boxes) - len(matches)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    sequence: str,
    gt_frames: dict[int, list[Box]],
    pred_frames: dict[int, list[Box]],
    gates: dict[str, float],
    class_filter: str | None = None,
) -> tuple[dict[str, float | int | str], list[dict], list[dict]]:
    gt_count = pred_count = false_negatives = false_positives = 0
    distances: list[float] = []
    yaw_errors: list[float] = []
    pair_counts: Counter[tuple[str, str, str]] = Counter()
    gt_track_lengths: Counter[tuple[str, str]] = Counter()
    pred_track_lengths: Counter[tuple[str, str]] = Counter()
    last_pred_for_gt: dict[tuple[str, str], str] = {}
    was_matched: dict[tuple[str, str], bool] = {}
    ever_matched: set[tuple[str, str]] = set()
    id_switches = fragments = 0
    switch_events: list[dict] = []

    for frame in sorted(set(gt_frames) | set(pred_frames)):
        frame_gt = [b for b in gt_frames.get(frame, []) if class_filter in (None, b.class_name)]
        frame_pred = [b for b in pred_frames.get(frame, []) if class_filter in (None, b.class_name)]
        gt_count += len(frame_gt)
        pred_count += len(frame_pred)
        for box in frame_gt:
            gt_track_lengths[(box.class_name, box.identity)] += 1
        for box in frame_pred:
            pred_track_lengths[(box.class_name, box.identity)] += 1

        matched_gt: set[tuple[str, str]] = set()
        for class_name in sorted({b.class_name for b in frame_gt + frame_pred}):
            class_gt = [b for b in frame_gt if b.class_name == class_name]
            class_pred = [b for b in frame_pred if b.class_name == class_name]
            matches, fn, fp = associate(class_gt, class_pred, gates.get(class_name, 2.0))
            false_negatives += fn
            false_positives += fp
            for gt, pred, distance in matches:
                gt_key = (gt.class_name, gt.identity)
                matched_gt.add(gt_key)
                distances.append(distance)
                yaw_errors.append(wrapped_angle_error(gt.yaw, pred.yaw))
                pair_counts[(gt.class_name, gt.identity, pred.identity)] += 1
                old_pred = last_pred_for_gt.get(gt_key)
                if old_pred is not None and old_pred != pred.identity:
                    id_switches += 1
                    switch_events.append({
                        "sequence": sequence,
                        "frame": frame,
                        "class_name": gt.class_name,
                        "gt_id": gt.identity,
                        "old_pred_id": old_pred,
                        "new_pred_id": pred.identity,
                        "center_distance_m": distance,
                    })
                if gt_key in ever_matched and was_matched.get(gt_key) is False:
                    fragments += 1
                last_pred_for_gt[gt_key] = pred.identity
                was_matched[gt_key] = True
                ever_matched.add(gt_key)

        for gt in frame_gt:
            key = (gt.class_name, gt.identity)
            if key not in matched_gt:
                was_matched[key] = False

    true_positives = len(distances)
    idtp = 0
    mappings: list[dict] = []
    for class_name in sorted({key[0] for key in gt_track_lengths} | {key[0] for key in pred_track_lengths}):
        gt_ids = sorted(key[1] for key in gt_track_lengths if key[0] == class_name)
        pred_ids = sorted(key[1] for key in pred_track_lengths if key[0] == class_name)
        if gt_ids and pred_ids:
            counts = np.array([
                [pair_counts[(class_name, gt_id, pred_id)] for pred_id in pred_ids]
                for gt_id in gt_ids
            ])
            rows, cols = linear_sum_assignment(-counts)
            idtp += int(counts[rows, cols].sum())
        for gt_id in gt_ids:
            candidates = [
                (count, pred_id)
                for (pair_class, pair_gt, pred_id), count in pair_counts.items()
                if pair_class == class_name and pair_gt == gt_id
            ]
            matched_frames, dominant_id = max(candidates, default=(0, ""))
            total_matches = sum(count for count, _ in candidates)
            mappings.append({
                "sequence": sequence,
                "class_name": class_name,
                "gt_id": gt_id,
                "dominant_pred_id": dominant_id,
                "dominant_matched_frames": matched_frames,
                "all_matched_frames": total_matches,
                "gt_frames": gt_track_lengths[(class_name, gt_id)],
                "id_consistency": safe_div(matched_frames, total_matches),
            })

    idfn = gt_count - idtp
    idfp = pred_count - idtp
    det_a = safe_div(true_positives, true_positives + false_negatives + false_positives)
    association_sum = sum(
        count * safe_div(
            count,
            gt_track_lengths[(class_name, gt_id)]
            + pred_track_lengths[(class_name, pred_id)] - count,
        )
        for (class_name, gt_id, pred_id), count in pair_counts.items()
    )
    ass_a = safe_div(association_sum, true_positives)
    metrics: dict[str, float | int | str] = {
        "sequence": sequence,
        "class_name": class_filter or "All",
        "gt_boxes": gt_count,
        "pred_boxes": pred_count,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "id_switches": id_switches,
        "fragments": fragments,
        "precision": safe_div(true_positives, pred_count),
        "recall": safe_div(true_positives, gt_count),
        "mota": 1.0 - safe_div(false_negatives + false_positives + id_switches, gt_count),
        "motp_center_m": safe_div(sum(distances), true_positives),
        "center_rmse_m": math.sqrt(safe_div(sum(d * d for d in distances), true_positives)),
        "yaw_mae_deg": math.degrees(safe_div(sum(yaw_errors), true_positives)),
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "id_precision": safe_div(idtp, idtp + idfp),
        "id_recall": safe_div(idtp, idtp + idfn),
        "idf1": safe_div(2 * idtp, 2 * idtp + idfp + idfn),
        "det_a": det_a,
        "ass_a": ass_a,
        "hota_gate": math.sqrt(det_a * ass_a),
    }
    return metrics, switch_events, mappings


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metrics(rows: list[dict]) -> list[dict]:
    aggregates = []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), str(row["class_name"]))].append(row)
    for (method, class_name), items in groups.items():
        summed = {
            key: sum(int(item[key]) for item in items)
            for key in (
                "gt_boxes", "pred_boxes", "tp", "fp", "fn", "id_switches",
                "fragments", "idtp", "idfp", "idfn",
            )
        }
        tp = summed["tp"]
        det_a = safe_div(tp, tp + summed["fn"] + summed["fp"])
        ass_a = safe_div(sum(float(item["ass_a"]) * int(item["tp"]) for item in items), tp)
        aggregates.append({
            "sequence": "ALL",
            "class_name": class_name,
            **summed,
            "precision": safe_div(tp, summed["pred_boxes"]),
            "recall": safe_div(tp, summed["gt_boxes"]),
            "mota": 1.0 - safe_div(
                summed["fn"] + summed["fp"] + summed["id_switches"],
                summed["gt_boxes"],
            ),
            "motp_center_m": safe_div(
                sum(float(item["motp_center_m"]) * int(item["tp"]) for item in items), tp,
            ),
            "center_rmse_m": math.sqrt(safe_div(
                sum(float(item["center_rmse_m"]) ** 2 * int(item["tp"]) for item in items), tp,
            )),
            "yaw_mae_deg": safe_div(
                sum(float(item["yaw_mae_deg"]) * int(item["tp"]) for item in items), tp,
            ),
            "id_precision": safe_div(summed["idtp"], summed["idtp"] + summed["idfp"]),
            "id_recall": safe_div(summed["idtp"], summed["idtp"] + summed["idfn"]),
            "idf1": safe_div(
                2 * summed["idtp"], 2 * summed["idtp"] + summed["idfp"] + summed["idfn"],
            ),
            "det_a": det_a,
            "ass_a": ass_a,
            "hota_gate": math.sqrt(det_a * ass_a),
            "method": method,
        })
    return aggregates


def format_report(rows: list[dict], gates: dict[str, float]) -> str:
    overall = [row for row in rows if row["class_name"] == "All"]
    lines = [
        "# SUSTechPOINTS tracking evaluation",
        "",
        "GT ID와 예측 ID의 숫자는 비교하지 않았다. 각 프레임에서 클래스와 XY 중심 거리로 먼저 대응한 후 ID 유지 여부를 계산했다.",
        f"거리 gate: Car {gates['Car']:.2f} m, Pedestrian {gates['Pedestrian']:.2f} m.",
        "",
        "| Method | Sequence | Recall | Precision | IDF1 | IDSW | MOTA | HOTA@gate | Center RMSE (m) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['method']} | {row['sequence']} | {row['recall']:.3f} | "
            f"{row['precision']:.3f} | {row['idf1']:.3f} | {row['id_switches']} | "
            f"{row['mota']:.3f} | {row['hota_gate']:.3f} | {row['center_rmse_m']:.3f} |"
        )
    lines.extend([
        "",
        "`HOTA@gate`는 거리 gate 한 지점에서 계산한 진단값이며, 여러 threshold를 평균하는 공식 HOTA와는 다르다.",
        "MOTA는 detector 오탐을 모두 벌점으로 포함하므로 입력 detector threshold의 영향을 크게 받는다.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=[f"tracking_test_{i}" for i in range(1, 5)])
    parser.add_argument("--methods", nargs="+", default=["ab3dmot", "eagermot", "fusiontrack", "observation_tracker"])
    parser.add_argument("--car-gate", type=float, default=2.0)
    parser.add_argument("--pedestrian-gate", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gates = {"Car": args.car_gate, "Pedestrian": args.pedestrian_gate}
    rows: list[dict] = []
    events: list[dict] = []
    mappings: list[dict] = []
    for sequence in args.sequences:
        gt_frames = load_sustech_labels(args.gt_root / sequence / "label")
        for method in args.methods:
            prediction_path = args.prediction_root / sequence / f"{method}.csv"
            pred_frames = load_prediction_csv(prediction_path)
            for class_name in (None, "Car", "Pedestrian"):
                metrics, method_events, method_mappings = evaluate(
                    sequence, gt_frames, pred_frames, gates, class_name,
                )
                metrics["method"] = method
                rows.append(metrics)
                if class_name is None:
                    events.extend({"method": method, **event} for event in method_events)
                    mappings.extend({"method": method, **mapping} for mapping in method_mappings)

    rows.extend(aggregate_metrics(rows))
    write_csv(args.output_dir / "metrics.csv", rows)
    write_csv(args.output_dir / "id_switch_events.csv", events)
    write_csv(args.output_dir / "gt_prediction_mapping.csv", mappings)
    (args.output_dir / "report.md").write_text(format_report(rows, gates))
    (args.output_dir / "config.json").write_text(json.dumps({
        "gt_root": str(args.gt_root),
        "prediction_root": str(args.prediction_root),
        "sequences": args.sequences,
        "methods": args.methods,
        "gates_m": gates,
        "association": "class-aware Hungarian matching on XY center distance",
    }, indent=2))


if __name__ == "__main__":
    main()
