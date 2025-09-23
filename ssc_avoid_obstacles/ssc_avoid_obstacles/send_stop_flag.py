#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Bool
from geometry_msgs.msg import Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

from sklearn.cluster import DBSCAN


# 좌표계 안내:
#             (높이) Z
#                   △   X (정면)
#                   |  ◁
#                   | /
#                   |/
#       Y ◁------- 차

TOPIC_PNT_SUB = '/zed/zed_node/point_cloud/cloud_registered'
TOPIC_ROI_PUB = '/roi_vis'
TOPIC_FLAG_PUB = '/stop_flag'
TOPIC_CLUSTER_PUB = '/clusters_vis'
STANDARD_FRAME = 'zed_camera_center'
PUB_HZ = 10
MAX_NUM = 0

# ROI 기본값
MIN_X, MAX_X = 4.0, 8.0
MIN_Y, MAX_Y = -1.0, 1.0
MIN_Z, MAX_Z = 1.5, 3.0

# DBSCAN 파라미터
DBSCAN_EPS = 0.2
DBSCAN_MIN_SAMPLES = 5
MIN_CLUSTER_SIZE = 1300

DEBUG = False


class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__('send_stop_flag')

        # 파라미터 선언
        self.declare_parameter('min_x', MIN_X)
        self.declare_parameter('max_x', MAX_X)
        self.declare_parameter('min_y', MIN_Y)
        self.declare_parameter('max_y', MAX_Y)
        self.declare_parameter('min_z', MIN_Z)
        self.declare_parameter('max_z', MAX_Z)
        self.declare_parameter('dbscan_eps', DBSCAN_EPS)
        self.declare_parameter('dbscan_min_samples', DBSCAN_MIN_SAMPLES)
        self.declare_parameter('min_cluster_size', MIN_CLUSTER_SIZE)
        self.declare_parameter('debug', DEBUG)
        self.declare_parameter('publish_cluster_markers', True)

        # 파라미터 적용
        self.min_x = float(self.get_parameter('min_x').value)
        self.max_x = float(self.get_parameter('max_x').value)
        self.min_y = float(self.get_parameter('min_y').value)
        self.max_y = float(self.get_parameter('max_y').value)
        self.min_z = float(self.get_parameter('min_z').value)
        self.max_z = float(self.get_parameter('max_z').value)
        self.dbscan_eps = float(self.get_parameter('dbscan_eps').value)
        self.dbscan_min_samples = int(self.get_parameter('dbscan_min_samples').value)
        self.min_cluster_size = int(self.get_parameter('min_cluster_size').value)
        self.debug = bool(self.get_parameter('debug').value)
        self._publish_cluster_markers = bool(self.get_parameter('publish_cluster_markers').value)

        # Pub/Sub
        self.flag_pub = self.create_publisher(Bool, TOPIC_FLAG_PUB, PUB_HZ)
        self.box_pub  = self.create_publisher(Marker, TOPIC_ROI_PUB, PUB_HZ)
        if self._publish_cluster_markers:
            self.cluster_pub = self.create_publisher(MarkerArray, TOPIC_CLUSTER_PUB, PUB_HZ)
        else:
            self.cluster_pub = None

        self.pnt_sub  = self.create_subscription(
            PointCloud2,
            TOPIC_PNT_SUB,
            self.pnt_callback,
            10
        )

        # ROI marker 초기화
        self._roi_marker = self._init_roi_marker()
        self._last_color_is_red = None

    def _init_roi_marker(self) -> Marker:
        now = self.get_clock().now().to_msg()
        m = Marker()
        m.header.frame_id = STANDARD_FRAME
        m.header.stamp = now
        m.ns = "roi"
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD

        cx = (self.min_x + self.max_x) / 2.0
        cy = (self.min_y + self.max_y) / 2.0
        cz = (self.min_z + self.max_z) / 2.0
        sx = max((self.max_x - self.min_x), 1e-6)
        sy = max((self.max_y - self.min_y), 1e-6)
        sz = max((self.max_z - self.min_z), 1e-6)

        m.pose.position.x = float(cx)
        m.pose.position.y = float(cy)
        m.pose.position.z = float(cz)
        m.pose.orientation.w = 1.0

        m.scale.x = float(sx)
        m.scale.y = float(sy)
        m.scale.z = float(sz)

        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 0.25
        m.lifetime = Duration(sec=0, nanosec=0)
        return m

    def publish_roi(self, is_red: bool) -> None:
        if self._last_color_is_red == is_red:
            return
        self._last_color_is_red = is_red

        m = self._roi_marker
        m.header.stamp = self.get_clock().now().to_msg()
        if is_red:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 0.25
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 0.25
        self.box_pub.publish(m)

    def build_cluster_markers(self, clusters_xyzi):
        ma = MarkerArray()
        if not clusters_xyzi:
            return ma

        now = self.get_clock().now().to_msg()
        for i, (centroid, size, bbox_min, bbox_max) in enumerate(clusters_xyzi):
            cx = float((bbox_min[0] + bbox_max[0]) / 2.0)
            cy = float((bbox_min[1] + bbox_max[1]) / 2.0)
            cz = float((bbox_min[2] + bbox_max[2]) / 2.0)
            sx = float(max(bbox_max[0] - bbox_min[0], 1e-3))
            sy = float(max(bbox_max[1] - bbox_min[1], 1e-3))
            sz = float(max(bbox_max[2] - bbox_min[2], 1e-3))

            m = Marker()
            m.header.frame_id = STANDARD_FRAME
            m.header.stamp = now
            m.ns = "cluster"
            m.id = int(i)
            m.type = Marker.CUBE
            m.action = Marker.ADD

            m.pose.position.x = cx
            m.pose.position.y = cy
            m.pose.position.z = cz
            m.pose.orientation.w = 1.0

            m.scale.x, m.scale.y, m.scale.z = sx, sy, sz
            alpha = min(0.9, 0.1 + float(size) / 1000.0)
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.2, float(alpha)
            ma.markers.append(m)

            mc = Marker()
            mc.header.frame_id = STANDARD_FRAME
            mc.header.stamp = now
            mc.ns = "cluster_centroid"
            mc.id = int(1000 + i)
            mc.type = Marker.SPHERE
            mc.action = Marker.ADD
            mc.pose.position.x = float(centroid[0])
            mc.pose.position.y = float(centroid[1])
            mc.pose.position.z = float(centroid[2])
            mc.pose.orientation.w = 1.0
            mc.scale.x = mc.scale.y = mc.scale.z = 0.05
            mc.color.r, mc.color.g, mc.color.b, mc.color.a = 1.0, 0.8, 0.0, 1.0
            ma.markers.append(mc)

        return ma

    def pnt_callback(self, msg: PointCloud2) -> None:
        start = self.get_clock().now()
        flag_bool, clusters_info = self.check_obstacles_with_dbscan(msg)

        flag = Bool()
        flag.data = bool(flag_bool)
        self.flag_pub.publish(flag)
        self.publish_roi(flag.data)

        if self.cluster_pub is not None and clusters_info:
            ma = self.build_cluster_markers(clusters_info)
            self.cluster_pub.publish(ma)

        if self.debug:
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e6
            self.get_logger().info(f"[Time] callback took {elapsed:.2f} ms | flag={flag.data}")
            

    def check_obstacles_with_dbscan(self, data: PointCloud2):
        global MAX_NUM
        clusters_info = []
        try:
            arr = pc2.read_points_numpy(data, field_names=("x","y","z"))
            arr = arr[::4]  # downsample

            if arr.size == 0:
                return False, []

            if arr.dtype.names is not None:
                pts = np.vstack([arr['x'], arr['y'], arr['z']]).T
            else:
                pts = arr.reshape(-1, 3)

            finite_mask = np.isfinite(pts).all(axis=1)
            pts = pts[finite_mask]
            if pts.shape[0] == 0:
                return False, []

            in_roi_mask = (
                (self.min_x <= pts[:,0]) & (pts[:,0] <= self.max_x) &
                (self.min_y <= pts[:,1]) & (pts[:,1] <= self.max_y) &
                (self.min_z <= pts[:,2]) & (pts[:,2] <= self.max_z)
            )
            roi_pts = pts[in_roi_mask]
            cnt = int(roi_pts.shape[0])
            if self.debug:
                self.get_logger().info(f"[Info] {cnt}")
                
            if cnt > MAX_NUM:
                MAX_NUM = cnt

            if cnt < self.dbscan_min_samples:
                return False, []

            X = roi_pts.astype(np.float32)
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, n_jobs=-1)
            labels = db.fit_predict(X)

            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)

            for lab in unique_labels:
                mask = (labels == lab)
                size = int(mask.sum())
                pts_cluster = X[mask]
                bbox_min = pts_cluster.min(axis=0)
                bbox_max = pts_cluster.max(axis=0)
                centroid = pts_cluster.mean(axis=0)
                clusters_info.append((centroid, size, bbox_min, bbox_max))

            for (_, size, _, _) in clusters_info:
                if size >= self.min_cluster_size:
                    return True, clusters_info

            return False, clusters_info

        except Exception as e:
            self.get_logger().error(f"[DBSCAN] Exception: {e}")
            return False, []


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
