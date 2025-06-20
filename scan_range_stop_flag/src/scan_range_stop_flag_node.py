#!/usr/bin/env python3
# coding: utf-8
# file: scan_range_stop_flag_node.py
import math, rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
import time

class ScanRangeStopFlag:
    def __init__(self):
        rospy.init_node("scan_range_stop_flag_node")

        # -------- 파라미터 로드 --------
        self.x_min      = rospy.get_param("~x_min", 0.60)   # [m]
        self.x_max      = rospy.get_param("~x_max", 0.70)   # [m]
        self.y_abs      = rospy.get_param("~y_abs", 0.80)   # [m]  (|y| ≤ y_abs)
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        # --------------------------------

        # 퍼블리셔
        self.flag_pub   = rospy.Publisher("scan_range_stop_flag", Bool, queue_size=1)
        self.marker_pub = rospy.Publisher("scan_range_stop_roi", Marker,
                                          queue_size=1, latch=True)

        # ROI 시각화 마커 1회 발행
        self.publish_roi_marker()

        # 스캔 구독
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb, queue_size=1)

        rospy.loginfo("scan_range_stop_flag_node started "
                      f"(ROI: x={self.x_min:.2f}~{self.x_max:.2f}, |y|≤{self.y_abs:.2f})")
        rospy.spin()

    # ------------------------------------------------------------------
    def scan_cb(self, msg: LaserScan):
        """LaserScan 데이터에서 ROI 충돌 여부 판단"""
        t0 = time.perf_counter() 
        hit = False
        angle = msg.angle_min
        for r in msg.ranges:
            if math.isfinite(r):
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                if (self.x_min <= x <= self.x_max) and (abs(y) <= self.y_abs):
                    hit = True
                    break
            angle += msg.angle_increment
        self.flag_pub.publish(Bool(data=hit))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0   # ⬅ 경과 시간(ms)
        rospy.loginfo_throttle(1.0, f"[obj_depth_tracker] 1 frame = {elapsed_ms:.1f} ms")


    # ------------------------------------------------------------------
    def publish_roi_marker(self):
        """ROI를 나타내는 Marker(CUBE) 발행"""
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp    = rospy.Time.now()
        m.ns   = "scan_range_stop_roi"
        m.id   = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD

        # 중심 위치‧크기
        m.pose.position.x = (self.x_min + self.x_max) / 2.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = self.x_max - self.x_min
        m.scale.y = self.y_abs * 2.0
        m.scale.z = 0.05
        # 색 (빨간 반투명)
        m.color.r, m.color.a = 1.0, 0.4
        m.color.g = m.color.b = 0.0

        self.marker_pub.publish(m)

if __name__ == "__main__":
    try:
        ScanRangeStopFlag()
    except rospy.ROSInterruptException:
        pass
