#!/usr/bin/env python3
"""Estimate dual-camera 6DoF extrinsics from multiple metric ChArUco boards.

The cameras and boards must stay fixed while capture.py records the selected
frame range for both cameras. The inter-camera baseline and all right-camera
angles are estimated, not imposed. ArUco marker corners provide an independent
validation solve after the primary ChArUco-corner solve.
"""

import argparse
import datetime
import math
import os

import cv2
import numpy as np
from scipy.optimize import least_squares
import yaml


CAMERAS = ("front_left", "front_right")


def rotation_angle_deg(rotation):
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rotation_delta_deg(first, second):
    return rotation_angle_deg(first @ second.T)


def rotation_y_deg(angle_deg):
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ])


def project(camera_matrix, points):
    pixels = (camera_matrix @ points.T).T
    return pixels[:, :2] / pixels[:, 2:3]


def average_rotations(rotations):
    u_matrix, _, vt_matrix = np.linalg.svd(np.sum(rotations, axis=0))
    result = u_matrix @ vt_matrix
    if np.linalg.det(result) < 0.0:
        u_matrix[:, -1] *= -1.0
        result = u_matrix @ vt_matrix
    return result


def load_problem(args):
    with open(args.board_params, "r", encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, params["aruco_dict"])
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
        dictionary,
    )
    marker_count = len(base_board.getIds())

    boards = {}
    for board_index in args.boards:
        marker_ids = np.arange(
            board_index * id_stride,
            board_index * id_stride + marker_count,
            dtype=np.int32,
        ).reshape(-1, 1)
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            dictionary,
            marker_ids,
        )
        boards[board_index] = {
            "board": board,
            "charuco_detector": cv2.aruco.CharucoDetector(board),
            "charuco_points": np.asarray(
                board.getChessboardCorners(), dtype=np.float64
            ).reshape(-1, 3),
            "marker_points": {
                int(marker_id): np.asarray(points, dtype=np.float64)
                for marker_id, points in zip(
                    board.getIds().reshape(-1), board.getObjPoints()
                )
            },
        }

    camera_matrices = {}
    for camera in CAMERAS:
        info_path = os.path.join(
            args.capture_root, camera, "camera_info.yaml"
        )
        with open(info_path, "r", encoding="utf-8") as stream:
            info = yaml.safe_load(stream)
        camera_matrices[camera] = np.asarray(
            info["K"], dtype=np.float64
        ).reshape(3, 3)

    detector_parameters = cv2.aruco.DetectorParameters()
    detector_parameters.minMarkerPerimeterRate = 0.005
    detector_parameters.cornerRefinementMethod = (
        cv2.aruco.CORNER_REFINE_SUBPIX
    )
    aruco_detector = cv2.aruco.ArucoDetector(
        dictionary, detector_parameters
    )

    observations = {"charuco": {}, "aruco": {}}
    frame_indices = list(range(args.frame_start, args.frame_end + 1))
    for camera in CAMERAS:
        for board_index, metadata in boards.items():
            charuco_rows = []
            marker_rows = []
            for frame_index in frame_indices:
                image_path = os.path.join(
                    args.capture_root,
                    camera,
                    f"img_{frame_index:03d}.png",
                )
                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(image_path)

                corners, corner_ids, _, _ = (
                    metadata["charuco_detector"].detectBoard(image)
                )
                if corner_ids is not None:
                    for pixel, corner_id in zip(
                        corners.reshape(-1, 2), corner_ids.reshape(-1)
                    ):
                        charuco_rows.append((
                            frame_index,
                            metadata["charuco_points"][int(corner_id)],
                            pixel.astype(np.float64),
                        ))

                marker_corners, marker_ids, _ = (
                    aruco_detector.detectMarkers(image)
                )
                if marker_ids is not None:
                    for detected_corners, marker_id in zip(
                        marker_corners, marker_ids.reshape(-1)
                    ):
                        marker_id = int(marker_id)
                        if marker_id not in metadata["marker_points"]:
                            continue
                        for point, pixel in zip(
                            metadata["marker_points"][marker_id],
                            detected_corners.reshape(4, 2),
                        ):
                            marker_rows.append((
                                frame_index,
                                point,
                                pixel.astype(np.float64),
                            ))
            observations["charuco"][(camera, board_index)] = charuco_rows
            observations["aruco"][(camera, board_index)] = marker_rows

    return params, boards, camera_matrices, observations, frame_indices


def initial_parameters(boards, camera_matrices, observations, baseline_hint):
    poses = {}
    for camera in CAMERAS:
        for board_index in boards:
            grouped = {}
            for _, point, pixel in observations[
                "charuco"
            ][(camera, board_index)]:
                key = tuple(np.round(point, 9))
                grouped.setdefault(key, []).append(pixel)
            if len(grouped) < 4:
                raise RuntimeError(
                    f"{camera} Board {board_index}: fewer than four corners"
                )
            object_points = np.asarray(list(grouped), dtype=np.float64)
            image_points = np.asarray([
                np.median(grouped[key], axis=0) for key in grouped
            ], dtype=np.float64)
            success, rotation_vector, translation = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrices[camera],
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                raise RuntimeError(
                    f"{camera} Board {board_index}: solvePnP failed"
                )
            poses[(camera, board_index)] = (
                rotation_vector.reshape(3),
                translation.reshape(3),
            )

    relative_rotations = []
    relative_translations = []
    for board_index in boards:
        left_rotation = cv2.Rodrigues(
            poses[("front_left", board_index)][0]
        )[0]
        right_rotation = cv2.Rodrigues(
            poses[("front_right", board_index)][0]
        )[0]
        relative_rotation = right_rotation @ left_rotation.T
        relative_translation = (
            poses[("front_right", board_index)][1]
            - relative_rotation @ poses[("front_left", board_index)][1]
        )
        relative_rotations.append(relative_rotation)
        relative_translations.append(relative_translation)

    relative_rotation = average_rotations(relative_rotations)
    relative_translation = np.mean(relative_translations, axis=0)
    relative_translation *= (
        baseline_hint / max(np.linalg.norm(relative_translation), 1.0e-9)
    )
    values = [
        *cv2.Rodrigues(relative_rotation)[0].reshape(3),
        *relative_translation,
    ]
    for board_index in boards:
        values.extend(poses[("front_left", board_index)][0])
        values.extend(poses[("front_left", board_index)][1])
    return np.asarray(values, dtype=np.float64)


def residuals(
    values,
    feature,
    boards,
    camera_matrices,
    observations,
    parity=None,
):
    relative_rotation = cv2.Rodrigues(values[:3])[0]
    relative_translation = values[3:6]
    result = []
    offset = 6
    for board_index in boards:
        board_rotation = cv2.Rodrigues(values[offset:offset + 3])[0]
        board_translation = values[offset + 3:offset + 6]
        offset += 6
        for camera in CAMERAS:
            for frame_index, object_point, observed_pixel in observations[
                feature
            ][(camera, board_index)]:
                if parity is not None and frame_index % 2 != parity:
                    continue
                camera_point = (
                    board_rotation @ object_point + board_translation
                )
                if camera == "front_right":
                    camera_point = (
                        relative_rotation @ camera_point
                        + relative_translation
                    )
                predicted_pixel = project(
                    camera_matrices[camera], camera_point.reshape(1, 3)
                )[0]
                result.extend(predicted_pixel - observed_pixel)
    return np.asarray(result, dtype=np.float64)


def solve(
    initial,
    feature,
    boards,
    camera_matrices,
    observations,
    parity=None,
):
    optimization = least_squares(
        lambda values: residuals(
            values,
            feature,
            boards,
            camera_matrices,
            observations,
            parity,
        ),
        initial,
        loss="soft_l1",
        f_scale=1.5,
        max_nfev=3000,
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
    )
    errors = np.linalg.norm(
        residuals(
            optimization.x,
            feature,
            boards,
            camera_matrices,
            observations,
            parity,
        ).reshape(-1, 2),
        axis=1,
    )
    return {
        "success": bool(optimization.success),
        "values": optimization.x,
        "rotation": cv2.Rodrigues(optimization.x[:3])[0],
        "translation": optimization.x[3:6],
        "rms_px": float(np.sqrt(np.mean(np.square(errors)))),
        "p95_px": float(np.percentile(errors, 95)),
        "observation_count": int(len(errors)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-root",
        default="/home/ssc/lidar_cam_calib/captures",
    )
    parser.add_argument(
        "--board-params",
        default="/home/ssc/lidar_cam_calib/board_params.yaml",
    )
    parser.add_argument("--boards", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--baseline-hint-m", type=float, default=0.11)
    parser.add_argument("--left-yaw-deg", type=float, default=-32.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    params, boards, camera_matrices, observations, frames = load_problem(
        args
    )
    initial = initial_parameters(
        boards,
        camera_matrices,
        observations,
        args.baseline_hint_m,
    )
    primary = solve(
        initial, "charuco", boards, camera_matrices, observations
    )
    primary_even = solve(
        primary["values"],
        "charuco",
        boards,
        camera_matrices,
        observations,
        parity=0,
    )
    primary_odd = solve(
        primary["values"],
        "charuco",
        boards,
        camera_matrices,
        observations,
        parity=1,
    )
    validation = solve(
        primary["values"], "aruco", boards, camera_matrices, observations
    )
    validation_even = solve(
        validation["values"],
        "aruco",
        boards,
        camera_matrices,
        observations,
        parity=0,
    )
    validation_odd = solve(
        validation["values"],
        "aruco",
        boards,
        camera_matrices,
        observations,
        parity=1,
    )

    split_rotation = rotation_delta_deg(
        primary_even["rotation"], primary_odd["rotation"]
    )
    split_translation = float(np.linalg.norm(
        primary_even["translation"] - primary_odd["translation"]
    ))
    validation_rotation = rotation_delta_deg(
        validation["rotation"], primary["rotation"]
    )
    validation_translation = float(np.linalg.norm(
        validation["translation"] - primary["translation"]
    ))
    baseline = float(np.linalg.norm(primary["translation"]))

    gates = []
    if not primary["success"] or not validation["success"]:
        gates.append("optimizer did not converge")
    if primary["rms_px"] > 1.0 or primary["p95_px"] > 2.0:
        gates.append("primary reprojection error is too high")
    if validation["rms_px"] > 3.0:
        gates.append("validation reprojection error is too high")
    if not 0.08 <= baseline <= 0.16:
        gates.append("estimated baseline is outside 0.08..0.16 m")
    if split_rotation > 1.0 or split_translation > 0.03:
        gates.append("primary even/odd split is unstable")
    if validation_rotation > 1.0 or validation_translation > 0.02:
        gates.append("independent ArUco solve disagrees with ChArUco solve")

    left_to_panorama = rotation_y_deg(args.left_yaw_deg)
    right_to_panorama = (
        left_to_panorama @ primary["rotation"].T
    )
    baseline_panorama = (
        -right_to_panorama @ primary["translation"]
    )
    right_axis = right_to_panorama[:, 2]
    right_yaw = math.degrees(math.atan2(
        right_axis[0], right_axis[2]
    ))
    right_pitch = math.degrees(math.atan2(
        -right_axis[1],
        math.hypot(right_axis[0], right_axis[2]),
    ))

    report = {
        "schema_version": 1,
        "accepted": not gates,
        "created_local_time": datetime.datetime.now().astimezone().isoformat(),
        "method": "unconstrained_metric_multiboard_bundle_adjustment",
        "convention": (
            "X_right_virtual_optical = R_right_from_left * "
            "X_left_virtual_optical + t_right_from_left_m"
        ),
        "inputs": {
            "capture_root": os.path.abspath(args.capture_root),
            "capture_indices": frames,
            "boards": list(boards),
            "square_length_m": float(params["square_length_m"]),
            "marker_length_m": float(params["marker_length_m"]),
            "baseline_hint_m": args.baseline_hint_m,
            "baseline_hint_use": "initialization_only",
        },
        "quality": {
            "primary_reprojection_rms_px": primary["rms_px"],
            "primary_reprojection_p95_px": primary["p95_px"],
            "validation_reprojection_rms_px": validation["rms_px"],
            "validation_reprojection_p95_px": validation["p95_px"],
            "validation_rotation_delta_deg": validation_rotation,
            "validation_translation_delta_m": validation_translation,
            "primary_even_odd_rotation_delta_deg": split_rotation,
            "primary_even_odd_translation_delta_m": split_translation,
            "validation_even_odd_rotation_delta_deg": rotation_delta_deg(
                validation_even["rotation"],
                validation_odd["rotation"],
            ),
            "validation_even_odd_translation_delta_m": float(np.linalg.norm(
                validation_even["translation"]
                - validation_odd["translation"]
            )),
        },
        "result": {
            "R_right_from_left": primary["rotation"].tolist(),
            "t_right_from_left_m": primary["translation"].tolist(),
            "estimated_color_lens_baseline_m": baseline,
            "relative_rotation_angle_deg": rotation_angle_deg(
                primary["rotation"]
            ),
            "left_yaw_deg": args.left_yaw_deg,
            "right_yaw_deg": right_yaw,
            "right_pitch_deg": right_pitch,
            "R_right_virtual_optical_to_panorama_optical":
                right_to_panorama.tolist(),
            "baseline_left_to_right_in_panorama_optical_m":
                baseline_panorama.tolist(),
            "left_lens_center_in_panorama_optical_m":
                (-0.5 * baseline_panorama).tolist(),
            "right_lens_center_in_panorama_optical_m":
                (0.5 * baseline_panorama).tolist(),
        },
        "rejection_reasons": gates,
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)

    print(
        f"accepted={report['accepted']} "
        f"RMS={primary['rms_px']:.3f}px "
        f"p95={primary['p95_px']:.3f}px "
        f"baseline={baseline:.5f}m "
        f"right yaw/pitch={right_yaw:.3f}/{right_pitch:.3f}deg"
    )
    print(f"report: {args.output}")
    if gates:
        for gate in gates:
            print(f"gate: {gate}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
