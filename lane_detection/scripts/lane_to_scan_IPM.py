#!/usr/bin/env python3
# coding: utf-8
"""
Subscribe : lane-line binary mask (mono8) + CameraInfo
Publish   : LaserScan (base_link 평면 기준, IPM 변환)

IPM 가정
- 지면은 하나의 평면(Z=0 in base_link, => y_cam = +h in optical frame)
- 카메라 외부 파라미터 : 고도 h[m], 피치 pitch_deg (카메라가 아래로 기운 각도, +40 = 40° 다운틸트)
- 카메라 yaw·roll = 0 (정면 장착), 필요하면 TF로 수정하면 됨
"""
import rospy, numpy as np, cv2
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from cv_bridge import CvBridge
from math import sin, cos, atan2, sqrt, radians, inf

class LaneMaskToScan:
    def __init__(self):
        rospy.init_node("lane_mask_to_scan_node")

        # ---------- Parameters ----------
        self.mask_topic        = rospy.get_param("~lane_mask_topic",  "/lane_line_node/lane_line")
        self.cinfo_topic       = rospy.get_param("~camera_info_topic","/camera/camera_info")
        self.h_cam             = rospy.get_param("~cam_height", 0.35)        # [m]
        self.pitch_deg         = rospy.get_param("~cam_pitch_deg", 40.0)     # [deg] (+ down)
        self.n_beams           = int(rospy.get_param("~scan_num_beams", 360))
        self.angle_min         = float(rospy.get_param("~scan_angle_min", -np.pi/4))  # –45°
        self.angle_max         = float(rospy.get_param("~scan_angle_max",  np.pi/4))  # +45°
        self.range_max         = rospy.get_param("~scan_range_max", 30.0)    # [m]
        self.scan_frame        = rospy.get_param("~scan_frame_id", "base_link")

        self.ang_inc = (self.angle_max - self.angle_min) / self.n_beams
        self.pitch_rad = radians(self.pitch_deg)

        # Camera intrinsics (filled once)
        self.fx = self.fy = self.cx = self.cy = None

        self.bridge = CvBridge()
        self.scan_pub = rospy.Publisher("~lane_scan", LaserScan, queue_size=1, latch=False)

        rospy.Subscriber(self.cinfo_topic, CameraInfo, self.cinfo_cb,   queue_size=1)
        rospy.Subscriber(self.mask_topic,  Image,      self.mask_cb,    queue_size=1)

        rospy.loginfo("lane_mask_to_scan_node ready – waiting for camera_info & lane masks")

    # ---------------------------------------
    def cinfo_cb(self, msg: CameraInfo):
        # 한 번만 읽어도 충분
        if self.fx is None:
            self.fx, self.fy = msg.K[0], msg.K[4]
            self.cx, self.cy = msg.K[2], msg.K[5]
            rospy.loginfo("CameraInfo received (fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f)",
                          self.fx, self.fy, self.cx, self.cy)

    # ---------------------------------------
    def mask_cb(self, msg: Image):
        # ───────── 0. 준비 검사 ─────────
        if self.fx is None:
            rospy.logwarn_once("Waiting for CameraInfo …")
            return

        # ───────── 1. 마스크 디코딩 ─────────
        mask = self.bridge.imgmsg_to_cv2(msg, "mono8")
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        ys, xs = np.where(mask > 0)
        rospy.loginfo("mask pixels: %d", xs.size)

        if xs.size == 0:
            return

        # ───────── 2. 카메라 레이 계산 ─────────
        X = (xs - self.cx) / self.fx
        Y = (ys - self.cy) / self.fy
        Z = np.ones_like(X)
        dirs = np.stack([X, Y, Z], axis=1)

        rospy.loginfo("dirs shape: %s; sample: %s", dirs.shape, dirs[0])

        # ───────── 3. pitch 회전 ─────────
        c, s = cos(self.pitch_rad), sin(self.pitch_rad)
        R = np.array([[1, 0, 0],
                      [0, c,-s],
                      [0, s, c]], dtype=np.float32)
        dirs = dirs @ R.T

        # ───────── 4. 지면 교차점 ─────────
        y_comp = dirs[:,1]
        valid_mask = y_comp > 1e-6
        rospy.loginfo("rays hitting ground: %d / %d", valid_mask.sum(), dirs.shape[0])

        if not valid_mask.any():
            rospy.logwarn("No valid ground intersections this frame")
            return

        dirs = dirs[valid_mask]
        y_comp = y_comp[valid_mask]
        t = self.h_cam / y_comp
        pts = dirs * t[:, None]           # (N,3)

        rospy.loginfo("ground pts sample (cam frame): %s", pts[0])

        # ───────── 5. camera→base 변환 ─────────
        xb =  pts[:,2]
        yb = -pts[:,0]
        dist = np.hypot(xb, yb)
        ang  = np.arctan2(yb, xb)

        # ───────── 6. LaserScan 채우기 ─────────
        ranges = np.full(self.n_beams, inf, dtype=np.float32)
        in_fov = (ang >= self.angle_min) & (ang <= self.angle_max) & (dist < self.range_max)
        rospy.loginfo("points in FOV: %d", in_fov.sum())

        if not in_fov.any():
            return

        sel_ang = ang[in_fov]
        sel_dst = dist[in_fov]
        idx = np.floor((sel_ang - self.angle_min) / self.ang_inc).astype(np.int32)
        for i, d in zip(idx, sel_dst):
            if d < ranges[i]:
                ranges[i] = d

        # ───────── 7. 퍼블리시 ─────────
        scan = LaserScan()
        scan.header.stamp    = msg.header.stamp
        scan.header.frame_id = self.scan_frame
        scan.angle_min       = self.angle_min
        scan.angle_max       = self.angle_max
        scan.angle_increment = self.ang_inc
        scan.range_min       = 0.0
        scan.range_max       = self.range_max
        scan.ranges          = ranges.tolist()

        rospy.loginfo("scan published; min range in frame: %.2f m",
                      np.min(ranges[np.isfinite(ranges)]) if np.isfinite(ranges).any() else -1)

        self.scan_pub.publish(scan)

# -------------------------------------------
if __name__ == "__main__":
    try:
        LaneMaskToScan()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
