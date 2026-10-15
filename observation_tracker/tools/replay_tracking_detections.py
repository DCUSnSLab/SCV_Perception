#!/usr/bin/env python3
"""Publish saved 3-D detections once for every incoming LiDAR frame."""

import argparse
import csv
from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from tracking_msgs.msg import DetectedObject, DetectedObjectArray


def load_detections(path):
    frames = defaultdict(list)
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            frames[int(row["frame"])].append(row)
    return frames


class DetectionReplay(Node):
    def __init__(self, detections, input_topic, output_topic):
        super().__init__("detection_replay")
        self.detections = detections
        self.frame = 0
        self.publisher = self.create_publisher(
            DetectedObjectArray, output_topic, 10)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        self.subscription = self.create_subscription(
            PointCloud2, input_topic, self.callback, qos)

    def callback(self, message):
        output = DetectedObjectArray()
        output.header = message.header
        for row in self.detections.get(self.frame, ()):
            item = DetectedObject()
            item.header = message.header
            item.label = row["class_name"]
            item.score = float(row["score"])
            item.pose = [float(row[key]) for key in ("x", "y", "z")]
            item.dimensions = [float(row[key]) for key in ("dx", "dy", "dz")]
            item.yaw = float(row["yaw"])
            output.objects.append(item)
        self.publisher.publish(output)
        self.frame += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--input-topic", default="/velodyne_points")
    parser.add_argument("--output-topic", default="/detected_objects_3d")
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = DetectionReplay(
        load_detections(args.csv_path), args.input_topic, args.output_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
