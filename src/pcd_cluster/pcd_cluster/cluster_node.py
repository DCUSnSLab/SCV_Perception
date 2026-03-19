#!/usr/bin/env python3
import random
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from sklearn.neighbors import KDTree
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray


def pointcloud2_to_xyz(msg: PointCloud2):
    pts = []
    for p in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
        pts.append([p[0], p[1], p[2]])
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def xyz_to_pointcloud2(points_xyz, frame_id, stamp=None):
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    return point_cloud2.create_cloud(header, fields, points_xyz.tolist())


def voxel_downsample(points_xyz, voxel=0.15):
    """Simple voxel grid down-sampling."""
    if points_xyz.shape[0] == 0:
        return points_xyz
    vox = np.floor(points_xyz / voxel).astype(np.int32)
    _, keep_idx = np.unique(vox, axis=0, return_index=True)
    return points_xyz[keep_idx]


def euclidean_cluster(points_xyz, tol=0.4):
    """KDTree + BFS Euclidean clustering."""
    n = points_xyz.shape[0]
    if n == 0:
        return []

    tree = KDTree(points_xyz)
    visited = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        queue = deque([i])
        visited[i] = True
        comp = []
        while queue:
            idx = queue.popleft()
            comp.append(idx)
            nbrs = tree.query_radius(points_xyz[idx:idx + 1], r=tol)[0]
            for nb in nbrs:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        clusters.append(points_xyz[comp])
    return clusters


class ObstacleClusterNode(Node):
    """
    Subscribe to the ground-filtered point cloud, cluster obstacles, and publish
    the clustered cloud along with bounding box markers.
    """

    def __init__(self):
        super().__init__('obstacle_cluster_node')

        # Topics
        self.declare_parameter('input_topic', '/no_ground_points')
        self.declare_parameter('output_points_topic', '/obstacles')
        self.declare_parameter('output_boxes_topic', '/obstacle_boxes')

        # Pre-processing parameters
        self.declare_parameter('z_min', -2.0)
        self.declare_parameter('z_max', 3.0)
        self.declare_parameter('roi_x', 40.0)
        self.declare_parameter('roi_y', 30.0)
        self.declare_parameter('max_cluster_distance', 60.0)
        self.declare_parameter('voxel_size', 0.15)

        # Clustering parameters
        self.declare_parameter('cluster_tolerance', 0.4)
        self.declare_parameter('min_cluster_points', 15)
        self.declare_parameter('max_cluster_points', 25000)

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        points_topic = self.get_parameter('output_points_topic').get_parameter_value().string_value
        boxes_topic = self.get_parameter('output_boxes_topic').get_parameter_value().string_value

        self.sub_ = self.create_subscription(PointCloud2, input_topic, self.callback, 10)
        self.pub_points = self.create_publisher(PointCloud2, points_topic, 10)
        self.pub_boxes = self.create_publisher(MarkerArray, boxes_topic, 10)

        self.get_logger().info(
            f"✅ obstacle_cluster_node listening on {input_topic} -> {points_topic}, {boxes_topic}"
        )

    def callback(self, msg: PointCloud2):
        start = time.time()
        pts = pointcloud2_to_xyz(msg)
        if pts.shape[0] == 0:
            return

        filtered = self.apply_filters(
            pts,
            self.get_double('z_min'),
            self.get_double('z_max'),
            self.get_double('roi_x'),
            self.get_double('roi_y'),
            self.get_double('max_cluster_distance'),
        )
        if filtered.shape[0] == 0:
            self.publish_empty(msg)
            return

        voxel = self.get_double('voxel_size')
        downsampled = voxel_downsample(filtered, voxel)
        clusters = euclidean_cluster(downsampled, tol=self.get_double('cluster_tolerance'))
        valid_clusters = self.filter_clusters(
            clusters,
            self.get_int('min_cluster_points'),
            self.get_int('max_cluster_points'),
        )

        if not valid_clusters:
            self.publish_empty(msg)
            self.get_logger().info(
                f"0 clusters | raw={pts.shape[0]} filtered={filtered.shape[0]} down={downsampled.shape[0]} "
                f"time={(time.time() - start)*1000:.1f}ms"
            )
            return

        merged = np.concatenate(valid_clusters, axis=0).astype(np.float32)
        cloud_msg = xyz_to_pointcloud2(merged, msg.header.frame_id, msg.header.stamp)
        self.pub_points.publish(cloud_msg)
        self.pub_boxes.publish(self.build_markers(valid_clusters, msg.header))

        self.get_logger().info(
            f"clusters={len(valid_clusters)} pts_out={merged.shape[0]} "
            f"raw={pts.shape[0]} filtered={filtered.shape[0]} down={downsampled.shape[0]} "
            f"time={(time.time() - start)*1000:.1f}ms"
        )

    def apply_filters(self, points, z_min, z_max, roi_x, roi_y, max_dist):
        """Keep only usable points before clustering."""
        mask_z = (points[:, 2] > z_min) & (points[:, 2] < z_max)
        pts = points[mask_z]
        if pts.shape[0] == 0:
            return pts

        mask_roi = (np.abs(pts[:, 0]) < roi_x) & (np.abs(pts[:, 1]) < roi_y)
        pts = pts[mask_roi]
        if pts.shape[0] == 0:
            return pts

        dist_sq = pts[:, 0] * pts[:, 0] + pts[:, 1] * pts[:, 1]
        mask_dist = dist_sq < (max_dist * max_dist)
        return pts[mask_dist]

    def filter_clusters(self, clusters, min_points, max_points):
        valid = []
        for cluster in clusters:
            if cluster.shape[0] < min_points or cluster.shape[0] > max_points:
                continue

            min_pt = np.min(cluster, axis=0)
            max_pt = np.max(cluster, axis=0)
            size = max_pt - min_pt
            if size[2] < 0.15 or size[0] < 0.15 or size[1] < 0.15:
                continue
            valid.append(cluster)
        return valid

    def build_markers(self, clusters, header):
        markers = MarkerArray()
        for idx, cluster in enumerate(clusters):
            min_pt = np.min(cluster, axis=0)
            max_pt = np.max(cluster, axis=0)
            center = (min_pt + max_pt) / 2.0
            size = max_pt - min_pt

            marker = Marker()
            marker.header = Header()
            marker.header.frame_id = header.frame_id
            marker.header.stamp = header.stamp
            marker.ns = 'obstacle_boxes'
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])
            marker.pose.orientation.w = 1.0

            marker.scale.x = float(max(size[0], 0.1))
            marker.scale.y = float(max(size[1], 0.1))
            marker.scale.z = float(max(size[2], 0.1))

            rng = random.Random(idx)
            color = ColorRGBA()
            color.r = rng.random()
            color.g = rng.random()
            color.b = rng.random()
            color.a = 0.6
            marker.color = color

            marker.lifetime.sec = 0
            markers.markers.append(marker)
        return markers

    def publish_empty(self, msg: PointCloud2):
        empty = np.zeros((0, 3), dtype=np.float32)
        cloud_msg = xyz_to_pointcloud2(empty, msg.header.frame_id, msg.header.stamp)
        self.pub_points.publish(cloud_msg)
        self.pub_boxes.publish(MarkerArray())

    def get_double(self, name):
        return self.get_parameter(name).get_parameter_value().double_value

    def get_int(self, name):
        return self.get_parameter(name).get_parameter_value().integer_value


def main():
    rclpy.init()
    node = ObstacleClusterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
