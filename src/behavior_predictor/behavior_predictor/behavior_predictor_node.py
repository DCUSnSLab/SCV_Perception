#!/usr/bin/env python3

import argparse
import csv
import math
from collections import deque
from pathlib import Path

import numpy as np

import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from tracking_msgs.msg import DetectedObjectArray
from visualization_msgs.msg import Marker, MarkerArray


BEHAVIOR_COLORS = {
    "stationary": (0.70, 0.70, 0.70),
    "straight": (0.15, 0.85, 0.25),
    "turn_left": (0.20, 0.55, 1.00),
    "turn_right": (1.00, 0.50, 0.15),
}


def wrap_angle(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class BehaviorPredictorNode(Node):
    def __init__(self, args):
        super().__init__("behavior_predictor")
        self.args = args
        self.histories = {}
        self.last_object_meta = {}
        self.csv_file = None
        self.csv_writer = None
        self.class_colors = {
            "Vehicle": (0.15, 0.85, 0.25),
            "Pedestrian": (1.00, 0.55, 0.15),
            "Cyclist": (0.25, 0.70, 1.00),
        }
        self.sub = self.create_subscription(
            DetectedObjectArray,
            args.tracked_topic,
            self.tracked_objects_callback,
            10,
        )
        self.pub_markers = self.create_publisher(MarkerArray, args.prediction_marker_topic, 10)

        if args.enable_csv_logging:
            csv_dir = Path(args.prediction_output_dir).expanduser().resolve()
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / f"{args.sequence_id}_behavior_predictions.csv"
            self.csv_file = csv_path.open("w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "sequence_id",
                "timestamp",
                "track_id",
                "label",
                "score",
                "behavior",
                "speed_mps",
                "yaw_rate_rps",
                "history_len",
                "current_position",
                "predicted_path",
            ])
            self.get_logger().info(f"behavior prediction csv: {csv_path}")

        self.get_logger().info(f"listening tracked objects: {args.tracked_topic}")
        self.get_logger().info(f"publishing prediction markers: {args.prediction_marker_topic}")
        self.get_logger().info(
            f"history={args.history_seconds:.2f}s horizon={args.prediction_horizon:.2f}s "
            f"step={args.prediction_step:.2f}s"
        )

    def destroy_node(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
        super().destroy_node()

    def tracked_objects_callback(self, msg):
        timestamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        active_ids = set()

        for obj in msg.objects:
            if len(obj.pose) < 3:
                continue
            track_id = int(obj.id)
            active_ids.add(track_id)
            history = self.histories.setdefault(track_id, deque())
            history.append({
                "t": timestamp,
                "x": float(obj.pose[0]),
                "y": float(obj.pose[1]),
                "z": float(obj.pose[2]),
                "yaw": float(obj.yaw),
                "label": str(obj.label),
                "score": float(obj.score),
            })
            self.last_object_meta[track_id] = obj
            self._trim_history(history, timestamp)

        stale_ids = [
            track_id for track_id, history in self.histories.items()
            if not history or (timestamp - history[-1]["t"]) > self.args.max_track_gap_sec
        ]
        for track_id in stale_ids:
            self.histories.pop(track_id, None)
            self.last_object_meta.pop(track_id, None)

        self.publish_predictions(msg.header)

    def _trim_history(self, history, timestamp):
        min_time = timestamp - self.args.history_seconds
        while len(history) > 1 and history[0]["t"] < min_time:
            history.popleft()

    def estimate_state(self, history):
        if len(history) < 2:
            state = {
                "speed": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "yaw_rate": 0.0,
                "behavior": "stationary",
            }
            latest = history[-1]
            state.update({"x": latest["x"], "y": latest["y"], "z": latest["z"], "yaw": latest["yaw"]})
            return state

        first = history[0]
        last = history[-1]
        dt = max(last["t"] - first["t"], 1e-3)
        dx = last["x"] - first["x"]
        dy = last["y"] - first["y"]
        vx = dx / dt
        vy = dy / dt
        speed = float(math.hypot(vx, vy))

        yaw_rates = []
        for prev, curr in zip(history, list(history)[1:]):
            step_dt = max(curr["t"] - prev["t"], 1e-3)
            yaw_rates.append(wrap_angle(curr["yaw"] - prev["yaw"]) / step_dt)
        yaw_rate = float(sum(yaw_rates) / len(yaw_rates)) if yaw_rates else 0.0

        if speed < self.args.stationary_speed_thresh:
            behavior = "stationary"
        elif yaw_rate > self.args.turn_yaw_rate_thresh:
            behavior = "turn_left"
        elif yaw_rate < -self.args.turn_yaw_rate_thresh:
            behavior = "turn_right"
        else:
            behavior = "straight"

        return {
            "x": last["x"],
            "y": last["y"],
            "z": last["z"],
            "yaw": last["yaw"],
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "yaw_rate": yaw_rate,
            "behavior": behavior,
        }

    def rollout_prediction(self, state):
        points = []
        x = float(state["x"])
        y = float(state["y"])
        z = float(state["z"])
        yaw = float(state["yaw"])
        speed = float(state["speed"])
        vx = float(state["vx"])
        vy = float(state["vy"])
        step = float(self.args.prediction_step)
        horizon = float(self.args.prediction_horizon)
        yaw_rate = float(state["yaw_rate"])

        num_steps = max(1, int(round(horizon / step)))
        for _ in range(num_steps):
            if abs(yaw_rate) < 1e-3:
                x += vx * step
                y += vy * step
            else:
                yaw = wrap_angle(yaw + yaw_rate * step)
                x += speed * math.cos(yaw) * step
                y += speed * math.sin(yaw) * step
            points.append((x, y, z))
        return points

    def format_point_list(self, points):
        return "[" + ", ".join(
            f"[{float(px):.3f}, {float(py):.3f}, {float(pz):.3f}]"
            for px, py, pz in points
        ) + "]"

    def publish_predictions(self, header):
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        lifetime = Duration(seconds=self.args.marker_lifetime).to_msg()
        for track_id, history in self.histories.items():
            if len(history) < self.args.min_history_points:
                continue

            latest = history[-1]
            state = self.estimate_state(history)
            future_points = self.rollout_prediction(state)
            if not future_points:
                continue

            behavior = state["behavior"]
            color = BEHAVIOR_COLORS.get(behavior, self.class_colors.get(latest["label"], (0.8, 0.8, 0.8)))

            line_marker = Marker()
            line_marker.header = header
            line_marker.ns = "behavior_prediction_paths"
            line_marker.id = track_id
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = self.args.path_line_width
            line_marker.color.r = color[0]
            line_marker.color.g = color[1]
            line_marker.color.b = color[2]
            line_marker.color.a = 0.95
            line_marker.lifetime = lifetime

            start = Point()
            start.x = latest["x"]
            start.y = latest["y"]
            start.z = latest["z"] + 0.15
            line_marker.points.append(start)
            for px, py, pz in future_points:
                point = Point()
                point.x = px
                point.y = py
                point.z = pz + 0.15
                line_marker.points.append(point)
            marker_array.markers.append(line_marker)

            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "behavior_prediction_text"
            text_marker.id = track_id + 10000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = latest["x"]
            text_marker.pose.position.y = latest["y"]
            text_marker.pose.position.z = latest["z"] + self.args.text_height
            text_marker.scale.z = 0.45
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.lifetime = lifetime
            text_marker.text = (
                f"{latest['label']} {track_id}\n"
                f"{behavior} {state['speed']:.2f}m/s"
            )
            marker_array.markers.append(text_marker)

            if self.csv_writer is not None:
                self.csv_writer.writerow([
                    self.args.sequence_id,
                    f"{float(latest['t']):.9f}",
                    int(track_id),
                    str(latest["label"]),
                    f"{float(latest['score']):.6f}",
                    behavior,
                    f"{float(state['speed']):.6f}",
                    f"{float(state['yaw_rate']):.6f}",
                    int(len(history)),
                    self.format_point_list([(latest["x"], latest["y"], latest["z"])]),
                    self.format_point_list(future_points),
                ])

        self.pub_markers.publish(marker_array)
        if self.csv_file is not None:
            self.csv_file.flush()


def parse_config():
    parser = argparse.ArgumentParser(description="Basic behavior predictor using tracked trajectories")
    parser.add_argument(
        "--tracked_topic",
        type=str,
        default="/tracked_objects_3d",
        help="tracked object topic from jay_tracker",
    )
    parser.add_argument(
        "--prediction_marker_topic",
        type=str,
        default="/behavior/prediction_markers",
        help="MarkerArray topic for predicted trajectories and behavior labels",
    )
    parser.add_argument(
        "--history_seconds",
        type=float,
        default=1.5,
        help="seconds of tracked history to retain per object",
    )
    parser.add_argument(
        "--prediction_horizon",
        type=float,
        default=3.0,
        help="future horizon in seconds",
    )
    parser.add_argument(
        "--prediction_step",
        type=float,
        default=0.5,
        help="future rollout step in seconds",
    )
    parser.add_argument(
        "--min_history_points",
        type=int,
        default=3,
        help="minimum number of history points required before predicting",
    )
    parser.add_argument(
        "--max_track_gap_sec",
        type=float,
        default=0.8,
        help="drop a track if no updates arrive for this duration",
    )
    parser.add_argument(
        "--stationary_speed_thresh",
        type=float,
        default=0.35,
        help="speed threshold in m/s below which the object is treated as stationary",
    )
    parser.add_argument(
        "--turn_yaw_rate_thresh",
        type=float,
        default=0.20,
        help="absolute yaw-rate threshold in rad/s used to classify turning behavior",
    )
    parser.add_argument(
        "--path_line_width",
        type=float,
        default=0.12,
        help="line width for predicted path markers",
    )
    parser.add_argument(
        "--text_height",
        type=float,
        default=2.3,
        help="height offset for behavior text markers",
    )
    parser.add_argument(
        "--marker_lifetime",
        type=float,
        default=0.0,
        help="marker lifetime in seconds",
    )
    parser.add_argument(
        "--enable_csv_logging",
        type=lambda value: str(value).lower() in ("1", "true", "yes", "on"),
        default=False,
        help="write behavior predictions to CSV",
    )
    parser.add_argument(
        "--sequence_id",
        type=str,
        default="seq01",
        help="sequence id used in behavior prediction CSV filename",
    )
    parser.add_argument(
        "--prediction_output_dir",
        type=str,
        default="results/behavior_predictions",
        help="directory for behavior prediction CSV output",
    )
    return parser.parse_args()


def main(args=None):
    cli_args = parse_config()
    rclpy.init(args=args)
    node = BehaviorPredictorNode(cli_args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
