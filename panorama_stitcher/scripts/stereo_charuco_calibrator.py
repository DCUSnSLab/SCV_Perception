#!/usr/bin/env python3
"""Calibrate the dual front RGB-D color cameras from pose-diverse views.

Unlike ``multiboard_bundle_calibrator.py``, every frame/board pair owns an
independent target pose.  The same ChArUco board may therefore be moved between
captures.  A calibration is accepted only when target pose diversity, a
disjoint-frame holdout, and an independent ArUco-corner solve all agree.
"""

import argparse
import datetime
import math
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


CAMERAS = ("front_left", "front_right")


def rotation_delta_deg(first, second):
    delta = first @ second.T
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def load_camera(capture_root, camera):
    path = os.path.join(capture_root, camera, "camera_info.yaml")
    with open(path, "r", encoding="utf-8") as stream:
        info = yaml.safe_load(stream)
    return {
        "K": np.asarray(info["K"], dtype=np.float64).reshape(3, 3),
        "D": np.asarray(info["D"], dtype=np.float64),
        "width": int(info["width"]),
        "height": int(info["height"]),
        "metadata": info,
    }


def load_boards(path, selected):
    with open(path, "r", encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, params["aruco_dict"]))
    base = cv2.aruco.CharucoBoard(
        (int(params["squares_x"]), int(params["squares_y"])),
        float(params["square_length_m"]),
        float(params["marker_length_m"]),
        dictionary,
    )
    marker_count = len(base.getIds())
    boards = {}
    for board_index in selected:
        ids = np.arange(
            board_index * int(params["id_stride"]),
            board_index * int(params["id_stride"]) + marker_count,
            dtype=np.int32,
        ).reshape(-1, 1)
        board = cv2.aruco.CharucoBoard(
            (int(params["squares_x"]), int(params["squares_y"])),
            float(params["square_length_m"]),
            float(params["marker_length_m"]),
            dictionary,
            ids,
        )
        boards[board_index] = {
            "board": board,
            "detector": cv2.aruco.CharucoDetector(board),
            "charuco_points": np.asarray(
                board.getChessboardCorners(), dtype=np.float32),
            "marker_points": {
                int(marker_id): np.asarray(points, dtype=np.float32)
                for marker_id, points in zip(
                    board.getIds().reshape(-1), board.getObjPoints())
            },
        }
    parameters = cv2.aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.005
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params, boards, cv2.aruco.ArucoDetector(dictionary, parameters)


def common_charuco_view(images, metadata):
    detections = []
    for image in images:
        corners, ids, _, _ = metadata["detector"].detectBoard(image)
        if ids is None:
            return None
        detections.append({
            int(corner_id): pixel.astype(np.float32)
            for corner_id, pixel in zip(
                ids.reshape(-1), corners.reshape(-1, 2))
        })
    common = sorted(set(detections[0]) & set(detections[1]))
    if len(common) < 6:
        return None
    return (
        metadata["charuco_points"][common].astype(np.float32),
        detections[0],
        detections[1],
        common,
    )


def common_aruco_view(images, metadata, detector):
    detections = []
    for image in images:
        corners, ids, _ = detector.detectMarkers(image)
        if ids is None:
            return None
        detections.append({
            int(marker_id): pixels.reshape(4, 2).astype(np.float32)
            for marker_id, pixels in zip(ids.reshape(-1), corners)
            if int(marker_id) in metadata["marker_points"]
        })
    common = sorted(set(detections[0]) & set(detections[1]))
    if len(common) < 2:
        return None
    objects = np.concatenate(
        [metadata["marker_points"][marker_id] for marker_id in common])
    left = np.concatenate([detections[0][marker_id] for marker_id in common])
    right = np.concatenate([detections[1][marker_id] for marker_id in common])
    return objects.astype(np.float32), left, right, common


def collect_views(capture_root, boards, aruco_detector, indices):
    result = {"charuco": [], "aruco": []}
    for frame_index in indices:
        paths = [
            os.path.join(capture_root, camera, f"img_{frame_index:03d}.png")
            for camera in CAMERAS
        ]
        if not all(os.path.isfile(path) for path in paths):
            continue
        images = [cv2.imread(path) for path in paths]
        if any(image is None for image in images):
            continue
        for board_index, metadata in boards.items():
            charuco = common_charuco_view(images, metadata)
            if charuco is not None:
                objects, left_map, right_map, common = charuco
                result["charuco"].append({
                    "frame": frame_index,
                    "board": board_index,
                    "object": objects,
                    "left": np.asarray(
                        [left_map[key] for key in common], dtype=np.float32),
                    "right": np.asarray(
                        [right_map[key] for key in common], dtype=np.float32),
                })
            aruco = common_aruco_view(images, metadata, aruco_detector)
            if aruco is not None:
                objects, left, right, _ = aruco
                result["aruco"].append({
                    "frame": frame_index,
                    "board": board_index,
                    "object": objects,
                    "left": left,
                    "right": right,
                })
    return result


def solve(views, cameras):
    if len(views) < 4:
        raise RuntimeError(f"only {len(views)} common target views")
    result = cv2.stereoCalibrateExtended(
        [view["object"] for view in views],
        [view["left"] for view in views],
        [view["right"] for view in views],
        cameras["front_left"]["K"].copy(),
        cameras["front_left"]["D"].copy(),
        cameras["front_right"]["K"].copy(),
        cameras["front_right"]["D"].copy(),
        (cameras["front_left"]["width"], cameras["front_left"]["height"]),
        None,
        None,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            200,
            1.0e-10,
        ),
    )
    rms, _, _, _, _, rotation, translation, _, _, _, _, per_view = result
    per_view = np.asarray(per_view, dtype=np.float64).reshape(-1)
    return {
        "rms_px": float(rms),
        "p95_view_rms_px": float(np.percentile(per_view, 95)),
        "rotation": np.asarray(rotation, dtype=np.float64),
        "translation": np.asarray(translation, dtype=np.float64).reshape(3),
        "view_count": len(views),
    }


def pose_diversity(views, left_camera):
    translations = []
    normals = []
    centroids = []
    for view in views:
        success, vector, translation = cv2.solvePnP(
            view["object"], view["left"], left_camera["K"], left_camera["D"])
        if not success:
            continue
        rotation = cv2.Rodrigues(vector)[0]
        translations.append(translation.reshape(3))
        normals.append(rotation[:, 2])
        centroids.append(np.mean(view["left"], axis=0))
    translations = np.asarray(translations)
    normals = np.asarray(normals)
    centroids = np.asarray(centroids)
    max_normal_angle = 0.0
    for first in normals:
        cosines = np.clip(normals @ first, -1.0, 1.0)
        max_normal_angle = max(
            max_normal_angle, float(np.degrees(np.arccos(cosines)).max()))
    centroid_span = np.ptp(centroids, axis=0)
    return {
        "depth_span_m": float(np.ptp(translations[:, 2])),
        "lateral_span_m": float(np.ptp(translations[:, 0])),
        "vertical_span_m": float(np.ptp(translations[:, 1])),
        "normal_span_deg": max_normal_angle,
        "centroid_span_px": centroid_span.tolist(),
    }


def panorama_geometry(rotation, translation):
    half = Rotation.from_matrix(rotation).as_rotvec() * 0.5
    left_to_rig = Rotation.from_rotvec(half).as_matrix()
    right_to_rig = left_to_rig @ rotation.T
    right_center_in_left = -rotation.T @ translation
    baseline_in_rig = left_to_rig @ right_center_in_left
    left_center = -0.5 * baseline_in_rig
    right_center = 0.5 * baseline_in_rig
    return left_to_rig, right_to_rig, baseline_in_rig, left_center, right_center


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True)
    parser.add_argument(
        "--board-params",
        default="/home/ssc/lidar_cam_calib/board_params.yaml")
    parser.add_argument("--boards", nargs="+", type=int, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cameras = {
        camera: load_camera(args.capture_root, camera) for camera in CAMERAS
    }
    params, boards, aruco_detector = load_boards(
        args.board_params, args.boards)
    indices = list(range(args.frame_start, args.frame_end + 1))
    observations = collect_views(
        args.capture_root, boards, aruco_detector, indices)
    charuco = observations["charuco"]
    train = [view for view in charuco if view["frame"] % 4 != 3]
    holdout = [view for view in charuco if view["frame"] % 4 == 3]

    full = solve(charuco, cameras)
    train_result = solve(train, cameras)
    holdout_result = solve(holdout, cameras)
    independent = solve(observations["aruco"], cameras)
    diversity = pose_diversity(charuco, cameras["front_left"])

    baseline = float(np.linalg.norm(full["translation"]))
    relative_angle = rotation_delta_deg(full["rotation"], np.eye(3))
    train_hold_rotation = rotation_delta_deg(
        train_result["rotation"], holdout_result["rotation"])
    train_hold_translation = float(np.linalg.norm(
        train_result["translation"] - holdout_result["translation"]))
    independent_rotation = rotation_delta_deg(
        full["rotation"], independent["rotation"])
    independent_translation = float(np.linalg.norm(
        full["translation"] - independent["translation"]))
    left_to_rig, right_to_rig, baseline_rig, left_center, right_center = (
        panorama_geometry(full["rotation"], full["translation"]))

    gates = []
    if len(charuco) < 12 or len(train) < 8 or len(holdout) < 4:
        gates.append("fewer than 12 pose-diverse views or insufficient holdout")
    if diversity["depth_span_m"] < 0.8:
        gates.append("target depth span is below 0.8 m")
    if diversity["normal_span_deg"] < 10.0:
        gates.append("target normal span is below 10 deg")
    if max(diversity["centroid_span_px"]) < 120.0:
        gates.append("target image-position span is below 120 px")
    if full["rms_px"] > 1.0 or full["p95_view_rms_px"] > 1.5:
        gates.append("ChArUco stereo reprojection error is too high")
    if not 0.10 <= baseline <= 0.14:
        gates.append("estimated baseline is outside 0.10..0.14 m")
    if not 54.0 <= relative_angle <= 62.0:
        gates.append("relative camera angle is outside 54..62 deg")
    if train_hold_rotation > 0.7 or train_hold_translation > 0.02:
        gates.append("disjoint-frame holdout transform is unstable")
    if independent_rotation > 0.7 or independent_translation > 0.02:
        gates.append("independent ArUco solve disagrees with ChArUco solve")
    if abs(float(baseline_rig[1])) > 0.03:
        gates.append("estimated vertical baseline exceeds 3 cm")

    report = {
        "schema_version": 3,
        "accepted": not gates,
        "created_local_time": datetime.datetime.now().astimezone().isoformat(),
        "method": "pose_diverse_stereo_charuco_with_aruco_holdout",
        "convention": (
            "X_right_topic_optical = R_right_from_left * "
            "X_left_topic_optical + t_right_from_left_m"),
        "inputs": {
            "capture_root": os.path.abspath(args.capture_root),
            "capture_indices": indices,
            "boards": args.boards,
            "square_length_m": float(params["square_length_m"]),
            "marker_length_m": float(params["marker_length_m"]),
            "left_topic_pixels_rotated_180": True,
            "right_topic_pixels_rotated_180": False,
        },
        "quality": {
            "charuco_view_count": len(charuco),
            "aruco_view_count": len(observations["aruco"]),
            "full_reprojection_rms_px": full["rms_px"],
            "full_p95_view_rms_px": full["p95_view_rms_px"],
            "train_reprojection_rms_px": train_result["rms_px"],
            "holdout_reprojection_rms_px": holdout_result["rms_px"],
            "train_hold_rotation_delta_deg": train_hold_rotation,
            "train_hold_translation_delta_m": train_hold_translation,
            "independent_aruco_rotation_delta_deg": independent_rotation,
            "independent_aruco_translation_delta_m": independent_translation,
            "pose_diversity": diversity,
        },
        "result": {
            "R_right_from_left": full["rotation"].tolist(),
            "t_right_from_left_m": full["translation"].tolist(),
            "estimated_color_lens_baseline_m": baseline,
            "relative_rotation_angle_deg": relative_angle,
            "R_left_topic_optical_to_panorama_optical": left_to_rig.tolist(),
            "R_right_topic_optical_to_panorama_optical": right_to_rig.tolist(),
            "baseline_left_to_right_in_panorama_optical_m": baseline_rig.tolist(),
            "left_lens_center_in_panorama_optical_m": left_center.tolist(),
            "right_lens_center_in_panorama_optical_m": right_center.tolist(),
        },
        "rejection_reasons": gates,
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    print(
        f"accepted={report['accepted']} views={len(charuco)} "
        f"RMS={full['rms_px']:.3f}px baseline={baseline:.5f}m "
        f"angle={relative_angle:.3f}deg")
    for gate in gates:
        print(f"gate: {gate}")
    print(f"report: {args.output}")
    if gates:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
