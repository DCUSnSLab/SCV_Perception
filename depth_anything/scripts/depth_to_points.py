#!/usr/bin/env python3
import rospy, numpy as np, time
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from threading import Lock


class DepthToPointCloud:
    def __init__(self):
        # ---------- 파라미터 ----------
        depth_topic = rospy.get_param("~depth_topic",
                                      "/depth_anything/depth_registered/image_rect")
        rgb_topic   = rospy.get_param("~rgb_topic",
                                      "/zed_node/left/image_rect_color")
        info_topic  = rospy.get_param("~camera_info_topic",
                                      "/zed_node/left/camera_info")
        self.use_rgb = rospy.get_param("~use_rgb", True)
        self.debug   = rospy.get_param("~debug", False)
        # --------------------------------

        self.bridge = CvBridge()
        self.pub_pc = rospy.Publisher("/depth/points",
                                      PointCloud2, queue_size=1)

        # ---- 캐시용 변수 ----
        self.K_lock = Lock()
        self.K          = None
        self.rgb_img    = None
        self.rgb_stamp  = None

        # ---- Subscribers ----
        rospy.Subscriber(info_topic, CameraInfo,
                         self.info_cb, queue_size=1)
        if self.use_rgb:
            rospy.Subscriber(rgb_topic, Image,
                             self.rgb_cb, queue_size=1)
        rospy.Subscriber(depth_topic, Image,
                         self.depth_cb, queue_size=1)

        rospy.loginfo(f"[DepthToPC] ready (use_rgb={self.use_rgb}, debug={self.debug})")

    def info_cb(self, msg):
        with self.K_lock:
            self.K = np.asarray(msg.K, dtype=np.float32).reshape(3, 3)

    def rgb_cb(self, msg):
        try:
            self.rgb_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.rgb_stamp = msg.header.stamp
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, f"[bridge rgb] {e}")

    def depth_cb(self, msg):
        start_time = time.time()  # 타이머 시작

        # 필수 데이터 체크
        if self.K is None:
            rospy.logwarn_throttle(5.0, "Waiting for CameraInfo…")
            return
        if self.use_rgb and self.rgb_img is None:
            rospy.logwarn_throttle(5.0, "Waiting for RGB image…")
            return

        # Depth → meters
        try:
            depth = self._depth_to_meters(msg)
        except CvBridgeError as e:
            rospy.logerr(f"[bridge depth] {e}")
            return

        if self.debug:
            self._print_stats(msg, depth)

        # 포인트클라우드 생성 & 발행
        pc_msg = self._make_cloud(msg.header, depth, self.K,
                                  self.rgb_img if self.use_rgb else None)
        self.pub_pc.publish(pc_msg)

        # 타이머 종료 및 로그
        elapsed_ms = (time.time() - start_time) * 1000
        rospy.loginfo(f"[DepthToPC] processing time: {elapsed_ms:.2f} ms")

    @staticmethod
    def _depth_to_meters(msg):
        depth = CvBridge().imgmsg_to_cv2(msg)
        if msg.encoding == "16UC1":
            depth = depth.astype(np.float32) * 0.001
        elif msg.encoding != "32FC1":
            raise CvBridgeError(f"Unsupported encoding {msg.encoding}")
        return depth

    def _print_stats(self, msg, depth):
        finite = np.isfinite(depth)
        valid = depth[finite]
        rospy.logdebug(
            f"[{msg.header.stamp.to_sec():.3f}] "
            f"{depth.shape[::-1]} {msg.encoding} "
            f"min={valid.min():.2f}  max={valid.max():.2f}  "
            f"nan={np.isnan(depth).sum()}  inf={np.isinf(depth).sum()}"
        )

    def _make_cloud(self, header, depth, K, rgb=None):
        h, w = depth.shape
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        u = np.tile(np.arange(w), h)
        v = np.repeat(np.arange(h), w)
        z = depth.flatten()
        valid = np.isfinite(z) & (z > 0)

        x = (u[valid] - cx) * z[valid] / fx
        y = (v[valid] - cy) * z[valid] / fy

        points = []
        if rgb is not None:
            rgb_flat = rgb.reshape(-1, 3)[valid][:, ::-1]
            rgb_uint = (rgb_flat[:,0].astype(np.uint32) << 16) | \
                       (rgb_flat[:,1].astype(np.uint32) << 8)  | \
                        rgb_flat[:,2].astype(np.uint32)
            points = [(float(xi), float(yi), float(zi), int(rui))
                      for xi, yi, zi, rui in zip(x, y, z[valid], rgb_uint)]
            fields = [
                pc2.PointField("x", 0,  pc2.PointField.FLOAT32, 1),
                pc2.PointField("y", 4,  pc2.PointField.FLOAT32, 1),
                pc2.PointField("z", 8,  pc2.PointField.FLOAT32, 1),
                pc2.PointField("rgb", 12, pc2.PointField.UINT32, 1),
            ]
        else:
            points = [(float(xi), float(yi), float(zi))
                      for xi, yi, zi in zip(x, y, z[valid])]
            fields = [
                pc2.PointField("x", 0,  pc2.PointField.FLOAT32, 1),
                pc2.PointField("y", 4,  pc2.PointField.FLOAT32, 1),
                pc2.PointField("z", 8,  pc2.PointField.FLOAT32, 1),
            ]

        if self.debug:
            rospy.logdebug(f"[DepthToPC] publish {len(points)} pts")

        return pc2.create_cloud(header, fields, points)


if __name__ == "__main__":
    rospy.init_node("depth_to_pointcloud_node")
    DepthToPointCloud()
    rospy.spin()
