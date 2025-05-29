#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, message_filters, numpy as np
import time
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from ultralytics_ros.msg import YoloResult           # ← ultralytics_ros 패키지
from filters import build as build_filter       

QUEUE_SIZE = 40
SLOP       = 0.06      # 60 ms

class ObjectDepthTracker:
    def __init__(self):
        self.bridge = CvBridge()

        # ── ★ 추적할 COCO class id 파라미터 ────────────────────────
        #   기본: person(0), car(2), bus(5), truck(7)
        self.track_ids = rospy.get_param("~track_class_ids", [0, 2, 5, 7])

        # ── message_filters 구독 ───────────────────────────────────
        sub_yolo  = message_filters.Subscriber("/yolo_result", YoloResult)
        sub_depth = message_filters.Subscriber(
            "/depth_anything/depth_registered/image_rect", Image)
        sub_cinfo = message_filters.Subscriber(
            "/zed_node/left/camera_info", CameraInfo)

        sync = message_filters.ApproximateTimeSynchronizer(
            [sub_yolo, sub_depth, sub_cinfo],
            queue_size=QUEUE_SIZE, slop=SLOP, allow_headerless=False)
        sync.registerCallback(self.sync_cb)

        self.marker_pub = rospy.Publisher(
            "tracked_points", MarkerArray, queue_size=2)
        filter_name = rospy.get_param("~filter_type", "centroid")  # 또는 kalman6d
        self.tracker = build_filter(filter_name)
        rospy.loginfo("Using filter: %s", filter_name)
        rospy.loginfo("Tracker ready  (track ids=%s)", self.track_ids)

    # ------------------------------------------------------------------
    def sync_cb(self, yolo_msg, depth_msg, caminfo_msg):
        t0 = time.perf_counter() 
        fx, fy = caminfo_msg.K[0], caminfo_msg.K[4]
        cx, cy = caminfo_msg.K[2], caminfo_msg.K[5]

        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        stamp = depth_msg.header.stamp
        frame = depth_msg.header.frame_id or "zed_left_camera"

        marker_arr = MarkerArray()
        measurements = []

        for det_id, det in enumerate(yolo_msg.detections.detections):
            # ── 1) 클래스 id 확인 -----------------------------------
            if not det.results:        # 안전체크
                continue
            cls_id = det.results[0].id
            if cls_id not in self.track_ids:
                continue                          # 추적 대상 아님

            if det_id >= len(yolo_msg.masks):     # 매칭 실패 방지
                continue
            mask = self.bridge.imgmsg_to_cv2(
                yolo_msg.masks[det_id], "mono8") > 0

            vv, uu = np.nonzero(mask)
            if vv.size < 30:
                continue
            z_vals = depth[vv, uu]
            good   = z_vals > 0
            if good.sum() < 30:
                continue

            # ── 2) 중심 픽셀·깊이 계산 -----------------------------
            u_bar = int(np.mean(uu[good]))
            v_bar = int(np.mean(vv[good]))
            Z     = float(np.median(z_vals[good]))

            X = (u_bar - cx) * Z / fx
            Y = (v_bar - cy) * Z / fy

            # ── 3) Marker -----------------------------------------
            mk = Marker()
            mk.header.stamp = stamp
            mk.header.frame_id = frame
            mk.ns, mk.id = "tracked_pts", det_id
            mk.type, mk.action = Marker.SPHERE, Marker.ADD
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = X, Y, Z
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = 0.15
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.0, 0.8, 1.0, 0.8
            marker_arr.markers.append(mk)
            measurements.append((np.array([X,Y,Z]), cls_id))
            
        tracks = self.tracker.update(measurements, stamp)
        marker_arr = self.to_markers(tracks, stamp, frame)
        self.marker_pub.publish(marker_arr)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0   # ⬅ 경과 시간(ms)
        rospy.loginfo_throttle(1.0, f"[obj_depth_tracker] 1 frame = {elapsed_ms:.1f} ms")

    def to_markers(self, tracks, stamp, frame):
        arr = MarkerArray()
        for trk in tracks:
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = frame
            m.ns = "tracked_pts"; m.id = trk.id
            m.type, m.action = Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = trk.xyz
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.r, m.color.g, m.color.b, m.color.a = 0.,.8,1.,.8
            arr.markers.append(m)
        return arr

# ----------------------------------------------------------------------
if __name__ == "__main__":
    rospy.init_node("object_depth_tracker_node")
    ObjectDepthTracker()
    rospy.spin()
