#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from collections import deque

class LaserScanFilter:
    def __init__(self):
        rospy.init_node('laser_scan_filter', anonymous=True)

        # Low Pass Filter 설정
        self.alpha = 0.1  # 필터 계수
        self.filtered_ranges = None
        
        # 구독자 설정
        self.laser_scan_sub = rospy.Subscriber('/lane_scan', LaserScan, self.callback)
        
        # 퍼블리셔 설정
        self.filtered_pub = rospy.Publisher('/lane_scan_filtered', LaserScan, queue_size=10)

    def callback(self, msg):
        # LaserScan 메시지를 복사하여 필터링 적용
        filtered_msg = msg
        
        # Low Pass Filter 적용
        if self.filtered_ranges is None:
            self.filtered_ranges = np.array(msg.ranges)
        else:
            # 필터링된 값 계산
            self.filtered_ranges = (1 - self.alpha) * self.filtered_ranges + self.alpha * np.array(msg.ranges)

        # 필터링된 범위를 LaserScan 메시지에 적용
        filtered_msg.ranges = self.filtered_ranges.tolist()

        # 필터링된 메시지를 퍼블리시
        self.filtered_pub.publish(filtered_msg)

if __name__ == '__main__':
    try:
        laser_scan_filter = LaserScanFilter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
