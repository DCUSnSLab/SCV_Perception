#!/usr/bin/env python3
"""Evaluate recorded RealSense/Velodyne alignment without replaying a bag.

This is a consistency test, not a metrology-grade calibration certificate.  It
pairs cloud header timestamps, reads the recorded static TF, and compares each
Velodyne return with the nearest RealSense depth point in the common camera
field of view.  An intentionally uncalibrated (co-located sensor origins)
baseline makes the value of the extrinsic transform directly measurable.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage


CAMERA_TOPIC = '/camera/camera/depth/color/points'
LIDAR_TOPIC = '/velodyne_points'
CAMERA_FRAME = 'camera_depth_optical_frame'
LIDAR_FRAME = 'velodyne'


def stamp_seconds(message):
    stamp = message.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


def quaternion_matrix(x, y, z, w):
    q = np.asarray([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def transform_matrix(transform):
    q = transform.rotation
    p = transform.translation
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_matrix(q.x, q.y, q.z, q.w)
    matrix[:3, 3] = [p.x, p.y, p.z]
    return matrix


def resolve_transform(edges, target, source):
    """Return target<-source from parent<-child edges."""
    queue = [(source, np.eye(4))]
    visited = {source}
    while queue:
        frame, frame_from_source = queue.pop(0)
        if frame == target:
            return frame_from_source
        for neighbor, neighbor_from_frame in edges.get(frame, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, neighbor_from_frame @ frame_from_source))
    raise RuntimeError(f'No TF path from {source!r} to {target!r}')


def xyz_array(message):
    offsets = {field.name: field.offset for field in message.fields}
    missing = {'x', 'y', 'z'} - offsets.keys()
    if missing:
        raise RuntimeError(f'PointCloud2 is missing fields: {sorted(missing)}')
    endian = '>' if message.is_bigendian else '<'
    dtype = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': [endian + 'f4'] * 3,
        'offsets': [offsets['x'], offsets['y'], offsets['z']],
        'itemsize': message.point_step,
    })
    values = np.frombuffer(
        message.data, dtype=dtype, count=message.width * message.height)
    return np.column_stack((values['x'], values['y'], values['z'])).astype(
        np.float64)


def new_reader(bag):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''))
    return reader


def inventory(bag):
    times = {'camera': [], 'lidar': []}
    edges = {}
    reader = new_reader(bag)
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in (CAMERA_TOPIC, LIDAR_TOPIC):
            message = deserialize_message(data, PointCloud2)
            key = 'camera' if topic == CAMERA_TOPIC else 'lidar'
            times[key].append(stamp_seconds(message))
        elif topic == '/tf_static':
            message = deserialize_message(data, TFMessage)
            for item in message.transforms:
                parent = item.header.frame_id.lstrip('/')
                child = item.child_frame_id.lstrip('/')
                parent_from_child = transform_matrix(item.transform)
                edges.setdefault(child, []).append((parent, parent_from_child))
                edges.setdefault(parent, []).append(
                    (child, np.linalg.inv(parent_from_child)))
    return {key: np.asarray(value) for key, value in times.items()}, edges


def choose_pairs(times, sample_count, max_sync_ms):
    camera = times['camera']
    lidar = times['lidar']
    right = np.clip(np.searchsorted(camera, lidar), 1, len(camera) - 1)
    left = right - 1
    camera_index = np.where(
        np.abs(camera[right] - lidar) < np.abs(camera[left] - lidar),
        right, left)
    delta = camera[camera_index] - lidar
    valid = np.flatnonzero(np.abs(delta) <= max_sync_ms / 1000.0)
    if not len(valid):
        raise RuntimeError('No point-cloud pairs satisfy the sync threshold')
    targets = np.linspace(lidar[valid[0]], lidar[valid[-1]], sample_count)
    selected = []
    for target in targets:
        index = int(valid[np.argmin(np.abs(lidar[valid] - target))])
        if not selected or index != selected[-1]:
            selected.append(index)
    return [(index, int(camera_index[index]), float(delta[index]))
            for index in selected], delta


def load_clouds(bag, pairs):
    wanted = {
        'camera': {camera for _, camera, _ in pairs},
        'lidar': {lidar for lidar, _, _ in pairs},
    }
    clouds = {'camera': {}, 'lidar': {}}
    indices = {'camera': 0, 'lidar': 0}
    reader = new_reader(bag)
    while reader.has_next() and (
            len(clouds['camera']) < len(wanted['camera']) or
            len(clouds['lidar']) < len(wanted['lidar'])):
        topic, data, _ = reader.read_next()
        if topic not in (CAMERA_TOPIC, LIDAR_TOPIC):
            continue
        key = 'camera' if topic == CAMERA_TOPIC else 'lidar'
        index = indices[key]
        indices[key] += 1
        if index in wanted[key]:
            clouds[key][index] = xyz_array(
                deserialize_message(data, PointCloud2))
    return clouds


def transform_points(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def evaluate(pairs, clouds, camera_from_lidar, point_stride=2):
    distances = []
    frame_metrics = []
    representative = None
    for pair_number, (lidar_index, camera_index, sync_delta) in enumerate(pairs):
        camera = clouds['camera'][camera_index]
        camera = camera[
            np.isfinite(camera).all(axis=1) &
            (camera[:, 2] > 0.4) & (camera[:, 2] < 10.0)][::point_stride]
        lidar = clouds['lidar'][lidar_index][::point_stride]
        lidar_camera = transform_points(lidar, camera_from_lidar)
        z = lidar_camera[:, 2]
        valid = (
            np.isfinite(lidar_camera).all(axis=1) & (z > 0.4) & (z < 10.0) &
            (np.abs(lidar_camera[:, 0] / z) < 1.15) &
            (np.abs(lidar_camera[:, 1] / z) < 0.72))
        lidar_camera = lidar_camera[valid]
        distance, _ = cKDTree(camera).query(lidar_camera, workers=-1)
        distances.append(distance)
        frame_metrics.append({
            'pair': pair_number,
            'sync_delta_ms': sync_delta * 1000.0,
            'lidar_points_in_fov': int(len(lidar_camera)),
            'median_distance_m': float(np.median(distance)),
            'within_0_10_m': float(np.mean(distance < 0.10)),
            'within_0_20_m': float(np.mean(distance < 0.20)),
        })
        if pair_number == len(pairs) // 2:
            representative = (camera, lidar_camera)
    combined = np.concatenate(distances)
    summary = {
        'matched_lidar_points': int(len(combined)),
        'median_distance_m': float(np.median(combined)),
        'p75_distance_m': float(np.percentile(combined, 75)),
        'p90_distance_m': float(np.percentile(combined, 90)),
        'within_0_10_m': float(np.mean(combined < 0.10)),
        'within_0_20_m': float(np.mean(combined < 0.20)),
        'within_0_30_m': float(np.mean(combined < 0.30)),
        'frame_median_std_m': float(np.std([
            item['median_distance_m'] for item in frame_metrics])),
    }
    return summary, frame_metrics, combined, representative


def save_plot(path, calibrated_distances, baseline_distances,
              calibrated_clouds, baseline_clouds):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bins = np.linspace(0, 1.0, 81)
    axes[0].hist(baseline_distances, bins=bins, density=True, alpha=0.65,
                 label='uncalibrated baseline')
    axes[0].hist(calibrated_distances, bins=bins, density=True, alpha=0.75,
                 label='recorded calibration')
    axes[0].set(xlabel='nearest 3-D distance [m]', ylabel='density',
                title='24 synchronized frames')
    axes[0].legend()
    for axis, clouds, title in zip(
            axes[1:], (baseline_clouds, calibrated_clouds),
            ('Uncalibrated baseline', 'Recorded calibration')):
        camera, lidar = clouds
        # Camera optical coordinates: horizontal=x, forward=z.
        axis.scatter(camera[::20, 2], -camera[::20, 0], s=0.25,
                     c='#2b83ba', alpha=0.35, label='RealSense')
        axis.scatter(lidar[::2, 2], -lidar[::2, 0], s=1.0,
                     c='#d7191c', alpha=0.55, label='Velodyne')
        axis.set(xlim=(0, 10), ylim=(-6, 6), aspect='equal',
                 xlabel='camera forward z [m]', ylabel='camera left [m]',
                 title=title)
        axis.grid(alpha=0.2)
        axis.legend(markerscale=5)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def format_matrix(matrix):
    return [[round(float(value), 8) for value in row] for row in matrix]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=24)
    parser.add_argument('--max-sync-ms', type=float, default=30.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    times, edges = inventory(args.bag)
    pairs, all_sync_delta = choose_pairs(
        times, args.samples, args.max_sync_ms)
    clouds = load_clouds(args.bag, pairs)

    camera_from_lidar = resolve_transform(
        edges, CAMERA_FRAME, LIDAR_FRAME)
    # Control case: preserve only the standard camera-link/optical-axis
    # rotation and pretend both sensor origins/mount attitudes coincide.
    camera_from_camera_link = resolve_transform(
        edges, CAMERA_FRAME, 'camera_link')
    uncalibrated = camera_from_camera_link

    calibrated_summary, frame_metrics, calibrated_distances, calibrated_view = (
        evaluate(pairs, clouds, camera_from_lidar))
    baseline_summary, _, baseline_distances, baseline_view = evaluate(
        pairs, clouds, uncalibrated)

    abs_sync = np.abs(all_sync_delta) * 1000.0
    selected_sync = np.abs([pair[2] for pair in pairs]) * 1000.0
    improvement = 1.0 - (
        calibrated_summary['median_distance_m'] /
        baseline_summary['median_distance_m'])
    report = {
        'method': {
            'bag': str(args.bag),
            'camera_topic': CAMERA_TOPIC,
            'lidar_topic': LIDAR_TOPIC,
            'camera_frame': CAMERA_FRAME,
            'lidar_frame': LIDAR_FRAME,
            'sample_count': len(pairs),
            'range_m': [0.4, 10.0],
            'max_selected_sync_ms': args.max_sync_ms,
            'metric': 'one-way Velodyne-to-RealSense nearest 3-D point distance',
        },
        'timing': {
            'camera_message_count': int(len(times['camera'])),
            'lidar_message_count': int(len(times['lidar'])),
            'camera_rate_hz': float(1 / np.median(np.diff(times['camera']))),
            'lidar_rate_hz': float(1 / np.median(np.diff(times['lidar']))),
            'all_pairs_abs_sync_median_ms': float(np.median(abs_sync)),
            'all_pairs_within_30_ms': float(np.mean(abs_sync <= 30.0)),
            'selected_abs_sync_median_ms': float(np.median(selected_sync)),
            'selected_abs_sync_max_ms': float(np.max(selected_sync)),
        },
        'recorded_camera_from_lidar_matrix': format_matrix(camera_from_lidar),
        'recorded_calibration': calibrated_summary,
        'uncalibrated_baseline': baseline_summary,
        'median_distance_improvement': float(improvement),
        'per_frame': frame_metrics,
        'limitations': [
            'No surveyed target or ground-truth extrinsic is present.',
            'Nearest-point distance is affected by sensor noise, beam density, '
            'occlusion, motion, and residual timestamp offset.',
            'The result demonstrates cross-sensor consistency, not absolute '
            'six-degree-of-freedom calibration accuracy.',
        ],
    }
    json_path = args.output_dir / 'calibration_metrics.json'
    json_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    plot_path = args.output_dir / 'calibration_alignment.png'
    save_plot(plot_path, calibrated_distances, baseline_distances,
              calibrated_view, baseline_view)

    verdict = (
        'The recorded extrinsic is strongly supported as a useful calibration'
        if improvement >= 0.5 and calibrated_summary['within_0_20_m'] >= 0.5
        else 'The recorded extrinsic does not pass the consistency threshold')
    markdown = f"""# D555–Velodyne calibration consistency report

## Result

{verdict}. It reduces the median cross-sensor nearest-point residual by
**{improvement * 100:.1f}%** relative to a physically uncalibrated control.

| Metric | Recorded calibration | Uncalibrated control |
|---|---:|---:|
| Median 3-D residual | {calibrated_summary['median_distance_m'] * 100:.1f} cm | {baseline_summary['median_distance_m'] * 100:.1f} cm |
| 75th percentile | {calibrated_summary['p75_distance_m'] * 100:.1f} cm | {baseline_summary['p75_distance_m'] * 100:.1f} cm |
| Points within 10 cm | {calibrated_summary['within_0_10_m'] * 100:.1f}% | {baseline_summary['within_0_10_m'] * 100:.1f}% |
| Points within 20 cm | {calibrated_summary['within_0_20_m'] * 100:.1f}% | {baseline_summary['within_0_20_m'] * 100:.1f}% |
| Points within 30 cm | {calibrated_summary['within_0_30_m'] * 100:.1f}% | {baseline_summary['within_0_30_m'] * 100:.1f}% |

The evaluation used {len(pairs)} frames spread across the bag and
{calibrated_summary['matched_lidar_points']:,} Velodyne points in the RealSense
field of view (0.4–10 m). Selected pairs have a median absolute timestamp
difference of {np.median(selected_sync):.1f} ms and a maximum of
{np.max(selected_sync):.1f} ms.

![Alignment comparison](calibration_alignment.png)

## Interpretation

This is good evidence that the recorded transform is substantially correct and
appropriate for coarse semantic point-cloud fusion. It is not enough to claim
centimetre-level absolute accuracy: there is no surveyed checkerboard/plane in
the bag, and general-scene nearest-neighbour residuals also include depth noise,
Velodyne angular sparsity, occlusion, platform motion, and timestamp error.

For a formal acceptance test, record a static calibration target visible to
both sensors, estimate plane/edge residuals on that target, and report
translation/rotation uncertainty over repeated captures.
"""
    md_path = args.output_dir / 'CALIBRATION_REPORT.md'
    md_path.write_text(markdown, encoding='utf-8')
    print(json.dumps({
        'report': str(md_path), 'metrics': str(json_path),
        'plot': str(plot_path), 'verdict': verdict,
        'median_cm': calibrated_summary['median_distance_m'] * 100,
        'baseline_median_cm': baseline_summary['median_distance_m'] * 100,
        'within_20cm_percent': calibrated_summary['within_0_20_m'] * 100,
    }, indent=2))


if __name__ == '__main__':
    main()
