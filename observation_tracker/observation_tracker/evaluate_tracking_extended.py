"""Extended 3-D MOT evaluation for the four SCV tracking scenarios."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .evaluate_sustech_tracking import (
    Box, associate, load_prediction_csv, load_sustech_labels, safe_div,
    write_csv,
)


ALPHAS = np.arange(0.05, 0.96, 0.05)


def _corners(box: Box) -> list[tuple[float, float]]:
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    return [
        (box.x + c * x - s * y, box.y + s * x + c * y)
        for x, y in ((-box.dx / 2, -box.dy / 2), (box.dx / 2, -box.dy / 2),
                     (box.dx / 2, box.dy / 2), (-box.dx / 2, box.dy / 2))
    ]


def _cross(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _polygon_clip(subject, clip):
    output = subject
    for edge_start, edge_end in zip(clip, clip[1:] + clip[:1]):
        edge = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])
        source, output = output, []
        if not source:
            break
        for previous, current in zip(source[-1:] + source[:-1], source):
            prev_rel = (previous[0] - edge_start[0], previous[1] - edge_start[1])
            curr_rel = (current[0] - edge_start[0], current[1] - edge_start[1])
            prev_inside = _cross(edge, prev_rel) >= -1e-9
            curr_inside = _cross(edge, curr_rel) >= -1e-9
            if curr_inside != prev_inside:
                segment = (current[0] - previous[0], current[1] - previous[1])
                denominator = _cross(segment, edge)
                if abs(denominator) > 1e-12:
                    t = _cross(
                        (edge_start[0] - previous[0], edge_start[1] - previous[1]), edge
                    ) / denominator
                    output.append((previous[0] + t * segment[0], previous[1] + t * segment[1]))
            if curr_inside:
                output.append(current)
    return output


def _polygon_area(points) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(
        x1 * y2 - y1 * x2
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )) / 2.0


def iou_3d(a: Box, b: Box) -> float:
    intersection_area = _polygon_area(_polygon_clip(_corners(a), _corners(b)))
    z_overlap = max(0.0, min(a.z + a.dz / 2, b.z + b.dz / 2)
                    - max(a.z - a.dz / 2, b.z - b.dz / 2))
    intersection = intersection_area * z_overlap
    union = a.dx * a.dy * a.dz + b.dx * b.dy * b.dz - intersection
    return safe_div(intersection, union)


def hota_sequence(gt_frames, pred_frames, class_filter=None):
    frames = sorted(set(gt_frames) | set(pred_frames))
    gt_keys = sorted({(b.class_name, b.identity) for boxes in gt_frames.values() for b in boxes
                      if class_filter in (None, b.class_name)})
    pred_keys = sorted({(b.class_name, b.identity) for boxes in pred_frames.values() for b in boxes
                        if class_filter in (None, b.class_name)})
    gt_index = {key: index for index, key in enumerate(gt_keys)}
    pred_index = {key: index for index, key in enumerate(pred_keys)}
    gt_ids, pred_ids, similarities = [], [], []
    potential = np.zeros((len(gt_keys), len(pred_keys)))
    gt_count = np.zeros((len(gt_keys), 1))
    pred_count = np.zeros((1, len(pred_keys)))

    for frame in frames:
        gt = [b for b in gt_frames.get(frame, []) if class_filter in (None, b.class_name)]
        pred = [b for b in pred_frames.get(frame, []) if class_filter in (None, b.class_name)]
        gids = np.array([gt_index[(b.class_name, b.identity)] for b in gt], dtype=int)
        pids = np.array([pred_index[(b.class_name, b.identity)] for b in pred], dtype=int)
        similarity = np.array([
            [iou_3d(a, b) if a.class_name == b.class_name else 0.0 for b in pred] for a in gt
        ], dtype=float).reshape(len(gt), len(pred))
        if len(gt) and len(pred):
            denom = similarity.sum(0)[None, :] + similarity.sum(1)[:, None] - similarity
            normalized = np.divide(similarity, denom, out=np.zeros_like(similarity), where=denom > 1e-12)
            potential[gids[:, None], pids[None, :]] += normalized
        if len(gids):
            gt_count[gids] += 1
        if len(pids):
            pred_count[0, pids] += 1
        gt_ids.append(gids)
        pred_ids.append(pids)
        similarities.append(similarity)

    alignment = np.divide(
        potential, gt_count + pred_count - potential,
        out=np.zeros_like(potential), where=(gt_count + pred_count - potential) > 1e-12,
    )
    tp = np.zeros(len(ALPHAS)); fn = np.zeros(len(ALPHAS)); fp = np.zeros(len(ALPHAS))
    match_counts = [np.zeros_like(potential) for _ in ALPHAS]
    for gids, pids, similarity in zip(gt_ids, pred_ids, similarities):
        if not len(gids):
            fp += len(pids); continue
        if not len(pids):
            fn += len(gids); continue
        rows, cols = linear_sum_assignment(-(alignment[gids[:, None], pids[None, :]] * similarity))
        for index, alpha in enumerate(ALPHAS):
            valid = similarity[rows, cols] >= alpha - np.finfo(float).eps
            matched_rows, matched_cols = rows[valid], cols[valid]
            count = len(matched_rows)
            tp[index] += count; fn[index] += len(gids) - count; fp[index] += len(pids) - count
            if count:
                match_counts[index][gids[matched_rows], pids[matched_cols]] += 1

    ass_a = np.zeros(len(ALPHAS))
    for index, matches in enumerate(match_counts):
        pair_accuracy = np.divide(
            matches, gt_count + pred_count - matches,
            out=np.zeros_like(matches), where=(gt_count + pred_count - matches) > 0,
        )
        ass_a[index] = safe_div(float(np.sum(matches * pair_accuracy)), tp[index])
    det_a = np.divide(tp, tp + fn + fp, out=np.zeros_like(tp), where=(tp + fn + fp) > 0)
    return {"tp": tp, "fn": fn, "fp": fp, "ass_a": ass_a,
            "det_a": det_a, "hota": np.sqrt(det_a * ass_a)}


def combine_hota(results):
    tp = sum(result["tp"] for result in results)
    fn = sum(result["fn"] for result in results)
    fp = sum(result["fp"] for result in results)
    ass_a = np.divide(
        sum(result["ass_a"] * result["tp"] for result in results), tp,
        out=np.zeros_like(tp), where=tp > 0,
    )
    det_a = np.divide(tp, tp + fn + fp, out=np.zeros_like(tp), where=(tp + fn + fp) > 0)
    return {"tp": tp, "fn": fn, "fp": fp, "ass_a": ass_a,
            "det_a": det_a, "hota": np.sqrt(det_a * ass_a)}


def _metric_frames(gt_frames, pred_frames):
    gt_keys = sorted({(b.class_name, b.identity) for boxes in gt_frames.values() for b in boxes})
    pred_keys = sorted({(b.class_name, b.identity) for boxes in pred_frames.values() for b in boxes})
    gt_index = {key: index for index, key in enumerate(gt_keys)}
    pred_index = {key: index for index, key in enumerate(pred_keys)}
    data = []
    for frame in sorted(set(gt_frames) | set(pred_frames)):
        gt, pred = gt_frames.get(frame, []), pred_frames.get(frame, [])
        similarity = np.array([
            [iou_3d(a, b) if a.class_name == b.class_name else 0.0 for b in pred] for a in gt
        ], dtype=float).reshape(len(gt), len(pred))
        data.append((
            np.array([gt_index[(b.class_name, b.identity)] for b in gt], dtype=int),
            np.array([pred_index[(b.class_name, b.identity)] for b in pred], dtype=int),
            similarity,
        ))
    return data, len(gt_keys), len(pred_keys)


def identity_3d_sequence(gt_frames, pred_frames, threshold=0.5):
    data, num_gt, num_pred = _metric_frames(gt_frames, pred_frames)
    potential = np.zeros((num_gt, num_pred))
    gt_count, pred_count = np.zeros(num_gt), np.zeros(num_pred)
    for gt_ids, pred_ids, similarity in data:
        rows, cols = np.nonzero(similarity >= threshold)
        if len(rows): potential[gt_ids[rows], pred_ids[cols]] += 1
        if len(gt_ids): gt_count[gt_ids] += 1
        if len(pred_ids): pred_count[pred_ids] += 1
    size = num_gt + num_pred
    fp_cost, fn_cost = np.zeros((size, size)), np.zeros((size, size))
    fp_cost[num_gt:, :num_pred] = 1e10
    fn_cost[:num_gt, num_pred:] = 1e10
    for gt_id in range(num_gt):
        fn_cost[gt_id, :num_pred] = gt_count[gt_id]
        fn_cost[gt_id, num_pred + gt_id] = gt_count[gt_id]
    for pred_id in range(num_pred):
        fp_cost[:num_gt, pred_id] = pred_count[pred_id]
        fp_cost[num_gt + pred_id, pred_id] = pred_count[pred_id]
    fn_cost[:num_gt, :num_pred] -= potential
    fp_cost[:num_gt, :num_pred] -= potential
    rows, cols = linear_sum_assignment(fn_cost + fp_cost)
    idfn, idfp = int(fn_cost[rows, cols].sum()), int(fp_cost[rows, cols].sum())
    idtp = int(gt_count.sum() - idfn)
    return {"idtp": idtp, "idfp": idfp, "idfn": idfn,
            "idf1": safe_div(2 * idtp, 2 * idtp + idfp + idfn)}


def clear_3d_sequence(gt_frames, pred_frames, threshold=0.5):
    data, num_gt, _ = _metric_frames(gt_frames, pred_frames)
    previous_match = np.full(num_gt, -1, dtype=int)
    previous_timestep = np.full(num_gt, -1, dtype=int)
    tracked_transitions = np.zeros(num_gt, dtype=int)
    tp = fp = fn = id_switches = 0
    for gt_ids, pred_ids, similarity in data:
        if not len(gt_ids):
            fp += len(pred_ids); previous_timestep[:] = -1; continue
        if not len(pred_ids):
            fn += len(gt_ids); previous_timestep[:] = -1; continue
        score = 1000.0 * (pred_ids[None, :] == previous_timestep[gt_ids, None]) + similarity
        score[similarity < threshold] = 0.0
        rows, cols = linear_sum_assignment(-score)
        valid = score[rows, cols] > 0
        rows, cols = rows[valid], cols[valid]
        matched_gt, matched_pred = gt_ids[rows], pred_ids[cols]
        id_switches += int(np.sum((previous_match[matched_gt] >= 0)
                                  & (previous_match[matched_gt] != matched_pred)))
        was_untracked = previous_timestep < 0
        previous_match[matched_gt] = matched_pred
        previous_timestep[:] = -1
        previous_timestep[matched_gt] = matched_pred
        tracked_transitions += was_untracked & (previous_timestep >= 0)
        tp += len(rows); fn += len(gt_ids) - len(rows); fp += len(pred_ids) - len(rows)
    return {"tp": tp, "fp": fp, "fn": fn, "idsw": id_switches,
            "fragmentation": int(np.sum(np.maximum(tracked_transitions - 1, 0)))}


def load_detections(path: Path):
    frames = defaultdict(list)
    with path.open(newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            frame = int(row["frame"])
            frames[frame].append(Box(
                frame, f"det-{frame}-{index}", row["class_name"],
                float(row["x"]), float(row["y"]), float(row["yaw"]),
                float(row["z"]), float(row["dx"]), float(row["dy"]), float(row["dz"]),
            ))
    return dict(frames)


def association_maps(gt_frames, object_frames, gates):
    gt_to_object, matched_objects = {}, set()
    for frame in sorted(set(gt_frames) | set(object_frames)):
        gt = gt_frames.get(frame, [])
        objects = object_frames.get(frame, [])
        for class_name in {b.class_name for b in gt + objects}:
            matches, _, _ = associate(
                [b for b in gt if b.class_name == class_name],
                [b for b in objects if b.class_name == class_name],
                gates.get(class_name, 2.0),
            )
            for gt_box, object_box, _ in matches:
                gt_key = (gt_box.class_name, gt_box.identity)
                object_key = (object_box.class_name, object_box.identity)
                gt_to_object[(frame, gt_key)] = object_key
                matched_objects.add((frame, object_key))
    return gt_to_object, matched_objects


def _runs(frames):
    runs = []
    for frame in sorted(frames):
        if not runs or frame != runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    return runs


def lifecycle_metrics(sequence, method, gt_frames, det_frames, pred_frames, gates, fps):
    gt_pred, matched_pred = association_maps(gt_frames, pred_frames, gates)
    gt_det, matched_det = association_maps(gt_frames, det_frames, gates)
    gt_tracks = defaultdict(list)
    pred_tracks = defaultdict(list)
    for frame, boxes in gt_frames.items():
        for box in boxes: gt_tracks[(box.class_name, box.identity)].append(frame)
    for frame, boxes in pred_frames.items():
        for box in boxes: pred_tracks[(box.class_name, box.identity)].append(frame)

    duration_frames, identities_per_gt = [], defaultdict(set)
    for gt_key, frames in gt_tracks.items():
        current_id, current_start, previous_frame = None, None, None
        for frame in sorted(frames):
            pred_key = gt_pred.get((frame, gt_key))
            if pred_key is not None:
                identities_per_gt[gt_key].add(pred_key)
            if pred_key != current_id or previous_frame is None or frame != previous_frame + 1:
                if current_id is not None:
                    duration_frames.append(previous_frame - current_start + 1)
                current_id, current_start = pred_key, frame
            previous_frame = frame
        if current_id is not None:
            duration_frames.append(previous_frame - current_start + 1)

    miss_rows, recovery_successes = [], 0
    for gt_key, frames in gt_tracks.items():
        frames = sorted(frames)
        detected = {frame: (frame, gt_key) in gt_det for frame in frames}
        missing = [frame for frame in frames if not detected[frame]]
        for run in _runs(missing):
            before, after = run[0] - 1, run[-1] + 1
            if before not in detected or after not in detected or not detected[before] or not detected[after]:
                continue
            old_id, new_id = gt_pred.get((before, gt_key)), gt_pred.get((after, gt_key))
            success = old_id is not None and old_id == new_id
            recovery_successes += int(success)
            miss_rows.append({
                "method": method, "sequence": sequence, "class_name": gt_key[0], "gt_id": gt_key[1],
                "start_frame": run[0], "end_frame": run[-1], "miss_frames": len(run),
                "old_pred_id": "" if old_id is None else old_id[1],
                "new_pred_id": "" if new_id is None else new_id[1], "recovered": int(success),
            })

    unmatched_by_track = defaultdict(list)
    for pred_key, frames in pred_tracks.items():
        for frame in frames:
            if (frame, pred_key) not in matched_pred:
                unmatched_by_track[pred_key].append(frame)
    ghost_rows, ghost_lengths = [], []
    for pred_key, frames in unmatched_by_track.items():
        for run in _runs(frames):
            ghost_lengths.append(len(run))
            ghost_rows.append({
                "method": method, "sequence": sequence, "class_name": pred_key[0],
                "pred_id": pred_key[1], "start_frame": run[0], "end_frame": run[-1],
                "frames": len(run), "lifetime_s": len(run) / fps,
            })

    detector_fp = [(frame, box) for frame, boxes in det_frames.items() for box in boxes
                   if (frame, (box.class_name, box.identity)) not in matched_det]
    unmatched_predictions = defaultdict(list)
    for frame, boxes in pred_frames.items():
        unmatched_predictions[frame] = [
            box for box in boxes if (frame, (box.class_name, box.identity)) not in matched_pred
        ]
    propagated = 0
    by_frame = defaultdict(list)
    for frame, box in detector_fp: by_frame[frame].append(box)
    for frame, detections in by_frame.items():
        predictions = unmatched_predictions[frame]
        for class_name in {b.class_name for b in detections + predictions}:
            matches, _, _ = associate(
                [b for b in detections if b.class_name == class_name],
                [b for b in predictions if b.class_name == class_name],
                gates.get(class_name, 2.0),
            )
            propagated += len(matches)

    termination_rows, fp_track_frames = [], 0
    sequence_end = max(gt_frames)
    for gt_key, frames in gt_tracks.items():
        end = max(frames)
        if end >= sequence_end:
            continue
        matched_history = [(frame, gt_pred[(frame, gt_key)]) for frame in frames if (frame, gt_key) in gt_pred]
        if not matched_history:
            continue
        pred_key = matched_history[-1][1]
        after = [frame for frame in pred_tracks[pred_key] if frame > end]
        delay_frames = max(after) - end if after else 0
        fp_track_frames += len(after)
        termination_rows.append({
            "method": method, "sequence": sequence, "class_name": gt_key[0], "gt_id": gt_key[1],
            "pred_id": pred_key[1], "gt_last_frame": end,
            "pred_last_frame": max(after, default=end), "delay_s": delay_frames / fps,
            "fp_track_frames": len(after),
        })

    ever_matched_pred = {pred_key for _, pred_key in matched_pred}
    metrics = {
        "false_track_count": sum(key not in ever_matched_pred for key in pred_tracks),
        "tracker_fp": sum(len(frames) for frames in unmatched_by_track.values()),
        "ghost_episodes": len(ghost_lengths),
        "ghost_lifetime_mean_s": safe_div(sum(ghost_lengths), len(ghost_lengths)) / fps,
        "ghost_lifetime_max_s": max(ghost_lengths, default=0) / fps,
        "detector_fp": len(detector_fp), "propagated_detector_fp": propagated,
        "fp_propagation_rate": safe_div(propagated, len(detector_fp)),
        "miss_episodes": len(miss_rows), "recovered_episodes": recovery_successes,
        "track_recovery_rate": safe_div(recovery_successes, len(miss_rows)),
        "id_preservation_rate": safe_div(
            sum(len(ids) == 1 for ids in identities_per_gt.values()), len(gt_tracks)
        ),
        "track_duration_mean_s": safe_div(sum(duration_frames), len(duration_frames)) / fps,
        "track_duration_max_s": max(duration_frames, default=0) / fps,
        "termination_events": len(termination_rows),
        "termination_delay_mean_s": safe_div(
            sum(row["delay_s"] for row in termination_rows), len(termination_rows)
        ),
        "fp_track_frames": fp_track_frames,
        "_duration_frames": duration_frames, "_ghost_lengths": ghost_lengths,
        "_preserved_tracks": sum(len(ids) == 1 for ids in identities_per_gt.values()),
        "_gt_tracks": len(gt_tracks),
    }
    return metrics, miss_rows, ghost_rows, termination_rows


def crossing_candidates(sequence, method, gt_frames, pred_frames, distance=2.0):
    gt_pred, _ = association_maps(gt_frames, pred_frames, {"Car": 2.0, "Pedestrian": 1.0})
    tracks = defaultdict(dict)
    for frame, boxes in gt_frames.items():
        for box in boxes: tracks[(box.class_name, box.identity)][frame] = box
    rows = []
    keys = sorted(tracks)
    for index, first in enumerate(keys):
        for second in keys[index + 1:]:
            if first[0] != second[0]:
                continue
            close = [frame for frame in sorted(set(tracks[first]) & set(tracks[second]))
                     if math.hypot(tracks[first][frame].x - tracks[second][frame].x,
                                   tracks[first][frame].y - tracks[second][frame].y) <= distance]
            for run in _runs(close):
                before, after = run[0] - 1, run[-1] + 1
                old = [gt_pred.get((before, key)) for key in (first, second)]
                new = [gt_pred.get((after, key)) for key in (first, second)]
                evaluable = all(old) and all(new)
                rows.append({
                    "method": method, "sequence": sequence, "class_name": first[0],
                    "gt_id_a": first[1], "gt_id_b": second[1], "start_frame": run[0],
                    "end_frame": run[-1], "min_center_distance_m": min(
                        math.hypot(tracks[first][f].x - tracks[second][f].x,
                                   tracks[first][f].y - tracks[second][f].y) for f in run),
                    "evaluable": int(evaluable),
                    "both_ids_preserved": int(evaluable and old == new),
                    "manual_validation_required": 1,
                })
    return rows


def clean_metrics(metrics):
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--sequences", nargs="+", default=[f"tracking_test_{i}" for i in range(1, 5)])
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gates = {"Car": 2.0, "Pedestrian": 1.0}
    rows, miss_rows, ghost_rows, termination_rows, crossing_rows = [], [], [], [], []
    hota_by_method = defaultdict(list)
    lifecycle_by_method = defaultdict(list)
    identity_by_method = defaultdict(list)
    clear_by_method = defaultdict(list)

    for method in args.methods:
        for sequence in args.sequences:
            gt = load_sustech_labels(args.gt_root / sequence / "label")
            pred = load_prediction_csv(args.prediction_root / sequence / f"{method}.csv")
            detections = load_detections(args.detection_root / f"{sequence}.csv")
            hota = hota_sequence(gt, pred)
            identity = identity_3d_sequence(gt, pred)
            clear = clear_3d_sequence(gt, pred)
            lifecycle, misses, ghosts, terminations = lifecycle_metrics(
                sequence, method, gt, detections, pred, gates, args.fps,
            )
            row = {"method": method, "sequence": sequence, "hota_3d": float(hota["hota"].mean()),
                   "assa_3d": float(hota["ass_a"].mean()), "deta_3d": float(hota["det_a"].mean()),
                   "idf1_3d": identity["idf1"], "idsw_3d": clear["idsw"],
                   "fragmentation_3d": clear["fragmentation"], "fn_3d": clear["fn"],
                   **clean_metrics(lifecycle)}
            rows.append(row)
            hota_by_method[method].append(hota)
            lifecycle_by_method[method].append(lifecycle)
            identity_by_method[method].append(identity)
            clear_by_method[method].append(clear)
            miss_rows += misses; ghost_rows += ghosts; termination_rows += terminations
            crossing_rows += crossing_candidates(sequence, method, gt, pred)

    for method in args.methods:
        hota = combine_hota(hota_by_method[method])
        items = lifecycle_by_method[method]
        identity_items = identity_by_method[method]
        clear_items = clear_by_method[method]
        durations = sum((item["_duration_frames"] for item in items), [])
        ghosts = sum((item["_ghost_lengths"] for item in items), [])
        total = lambda key: sum(item[key] for item in items)
        rows.append({
            "method": method, "sequence": "ALL", "hota_3d": float(hota["hota"].mean()),
            "assa_3d": float(hota["ass_a"].mean()), "deta_3d": float(hota["det_a"].mean()),
            "idf1_3d": safe_div(
                2 * sum(item["idtp"] for item in identity_items),
                2 * sum(item["idtp"] for item in identity_items)
                + sum(item["idfp"] for item in identity_items)
                + sum(item["idfn"] for item in identity_items),
            ),
            "idsw_3d": sum(item["idsw"] for item in clear_items),
            "fragmentation_3d": sum(item["fragmentation"] for item in clear_items),
            "fn_3d": sum(item["fn"] for item in clear_items),
            "false_track_count": total("false_track_count"), "tracker_fp": total("tracker_fp"),
            "ghost_episodes": len(ghosts),
            "ghost_lifetime_mean_s": safe_div(sum(ghosts), len(ghosts)) / args.fps,
            "ghost_lifetime_max_s": max(ghosts, default=0) / args.fps,
            "detector_fp": total("detector_fp"), "propagated_detector_fp": total("propagated_detector_fp"),
            "fp_propagation_rate": safe_div(total("propagated_detector_fp"), total("detector_fp")),
            "miss_episodes": total("miss_episodes"), "recovered_episodes": total("recovered_episodes"),
            "track_recovery_rate": safe_div(total("recovered_episodes"), total("miss_episodes")),
            "id_preservation_rate": safe_div(
                sum(item["_preserved_tracks"] for item in items),
                sum(item["_gt_tracks"] for item in items),
            ),
            "track_duration_mean_s": safe_div(sum(durations), len(durations)) / args.fps,
            "track_duration_max_s": max(durations, default=0) / args.fps,
            "termination_events": total("termination_events"),
            "termination_delay_mean_s": safe_div(
                sum(item["termination_delay_mean_s"] * item["termination_events"] for item in items),
                total("termination_events")),
            "fp_track_frames": total("fp_track_frames"),
        })

    write_csv(args.output_dir / "extended_metrics.csv", rows)
    write_csv(args.output_dir / "miss_episodes.csv", miss_rows)
    write_csv(args.output_dir / "ghost_episodes.csv", ghost_rows)
    write_csv(args.output_dir / "termination_events.csv", termination_rows)
    write_csv(args.output_dir / "crossing_candidates.csv", crossing_rows)


if __name__ == "__main__":
    main()
