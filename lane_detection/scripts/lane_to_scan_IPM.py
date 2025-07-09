#!/usr/bin/env python3
# coding: utf-8
"""
lane_mask_to_scan_node – GPU-accelerated (PyTorch ≥ 2.2)
  * mono8 lane-mask  → 360-beam LaserScan
  * 빈 마스크여도 항상 ranges=inf 로 발행
"""
import rospy, cv2, numpy as np, torch, time
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from cv_bridge import CvBridge
from math import radians

class LaneMaskToScan:
    # ────────────────────────────────────────────────────────────
    def __init__(self):
        rospy.init_node("lane_mask_to_scan_node")

        # ROS params ─────────────────────────────────────────────
        self.mask_topic  = rospy.get_param("~lane_mask_topic",  "/lane_line_node/lane_line")
        self.cinfo_topic = rospy.get_param("~camera_info_topic","/camera/camera_info")
        self.h_cam       = rospy.get_param("~cam_height", 0.35)          # [m]
        self.pitch_deg   = rospy.get_param("~cam_pitch_deg", 40.0)       # +down
        self.n_beams     = int(rospy.get_param("~scan_num_beams", 360))
        self.angle_min   = float(rospy.get_param("~scan_angle_min", -np.pi/4))
        self.angle_max   = float(rospy.get_param("~scan_angle_max",  np.pi/4))
        self.range_max   = rospy.get_param("~scan_range_max", 30.0)      # [m]
        self.range_scale = rospy.get_param("~range_scale", 1.0)
        self.scan_frame  = rospy.get_param("~scan_frame_id", "base_link")

        self.ang_inc   = (self.angle_max - self.angle_min) / self.n_beams
        self.pitch_rad = radians(self.pitch_deg)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Camera intrinsics & cached tensors (filled in cinfo_cb)
        self.fx = self.fy = self.cx = self.cy = None
        self.dir_tbl_gpu, self.R_pitch_gpu = None, None

        self.bridge   = CvBridge()
        self.scan_pub = rospy.Publisher("~lane_scan", LaserScan, queue_size=1)
        self.x_offset = -0.1  # +앞으로 0.30 m (−면 뒤로)
        self.y_offset = 0.00   # +왼쪽   0.00 m (−면 오른쪽)
        self.ang_inc   = (self.angle_max - self.angle_min) / self.n_beams
        rospy.Subscriber(self.cinfo_topic, CameraInfo, self.cinfo_cb, queue_size=1)
        rospy.Subscriber(self.mask_topic,  Image,      self.mask_cb,  queue_size=1)

        rospy.loginfo("lane_mask_to_scan_node (GPU) ready – waiting CameraInfo & masks.")

    # ────────────────────────────────────────────────────────────
    def cinfo_cb(self, msg: CameraInfo):
        if self.fx is not None:  # 이미 초기화됨
            return

        self.fx, self.fy = msg.K[0], msg.K[4]
        self.cx, self.cy = msg.K[2], msg.K[5]
        H, W = msg.height, msg.width

        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        dir_tbl = torch.stack(
            ((xs - self.cx) / self.fx,
             (ys - self.cy) / self.fy,
             torch.ones_like(xs, dtype=torch.float32)),
            dim=-1).float().to(self.device)

        c, s = np.cos(self.pitch_rad), np.sin(self.pitch_rad)
        R_pitch = torch.tensor([[1,0,0],[0,c,-s],[0,s,c]], dtype=torch.float32, device=self.device)

        self.dir_tbl_gpu, self.R_pitch_gpu = dir_tbl, R_pitch
        rospy.loginfo("CameraInfo received – LUT cached on %s.", self.device)

    # ────────────────────────────────────────────────────────────
    def mask_cb(self, msg: Image):
        if self.dir_tbl_gpu is None:
            return   # CameraInfo 아직

        tic = time.perf_counter()

        # 0) mask → uint8 numpy → torch CPU → GPU
        mask_np  = self.bridge.imgmsg_to_cv2(msg, "mono8")
        mask_cpu = torch.from_numpy(mask_np).pin_memory()
        mask_gpu = mask_cpu.to(self.device, non_blocking=True).bool()

        # ranges: inf 로 초기
        ranges_gpu = torch.full((self.n_beams,), float("inf"),
                                device=self.device, dtype=torch.float32)

        if mask_gpu.any():
            # 1) 방향 벡터 선택 + pitch 회전
            dirs = self.dir_tbl_gpu[mask_gpu]          # (N,3)
            dirs = torch.matmul(dirs, self.R_pitch_gpu.T)

            # 2) 지면 교차
            y = dirs[:, 1]
            valid = y > 1e-6
            if valid.any():
                dirs = dirs[valid]
                t    = self.h_cam / y[valid]
                pts  = dirs * t.unsqueeze(1)

                # xb =  pts[:, 2]
                # yb = -pts[:, 0]
                xb =  pts[:, 2] + self.x_offset   # forward/back
                yb = -pts[:, 0] + self.y_offset   # left/right
                dist = torch.hypot(xb, yb) * self.range_scale
                ang  = torch.atan2(yb, xb)

                in_fov = (ang >= self.angle_min) & (ang <= self.angle_max) & (dist < self.range_max)
                if in_fov.any():
                    ang, dist = ang[in_fov], dist[in_fov]
                    idx = torch.clamp(((ang - self.angle_min) / self.ang_inc).long(),
                                      0, self.n_beams - 1)
                    ranges_gpu.scatter_reduce_(0, idx, dist, reduce="amin")

        # 3) publish (always)
        self._publish_scan(msg.header.stamp, ranges_gpu)
        print(f"[scan] {(time.perf_counter()-tic)*1000:.2f} ms")

    # ────────────────────────────────────────────────────────────
    def _publish_scan(self, stamp, ranges_gpu):
        scan = LaserScan()
        scan.header.stamp    = stamp
        scan.header.frame_id = self.scan_frame
        scan.angle_min       = self.angle_min
        scan.angle_max       = self.angle_max
        scan.angle_increment = self.ang_inc
        scan.range_min       = 0.0
        scan.range_max       = self.range_max
        scan.ranges          = ranges_gpu.cpu().tolist()   # 360 floats
        self.scan_pub.publish(scan)

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        LaneMaskToScan()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
