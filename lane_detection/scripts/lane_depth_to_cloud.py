#!/usr/bin/env python3
"""
Convert masked depth (32FC1) + CameraInfo → PointCloud2.
Now uses **`rgb` (FLOAT32) field** that RViz understands, so the cloud is
visible. Lane pixels are painted solid green.
"""
import rospy, numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError
import message_filters, sensor_msgs.point_cloud2 as pc2
import time
GREEN_RGB_FLOAT = np.array([0x00FF00], dtype=np.uint32).view(np.float32)[0]  # 0x00FF00 as float32 bit pattern

class DepthToCloud:
    def __init__(self):
        rospy.init_node('lane_depth_to_cloud', anonymous=True)
        depth_topic = rospy.get_param('~depth_topic',  '/lane_depth_fusion/lane_depth')
        info_topic  = rospy.get_param('~camera_info_topic', '/lane_depth_fusion/camera_info')
        self.frame  = rospy.get_param('~frame_id', '')  # if empty → use depth_msg.frame_id

        depth_sub = message_filters.Subscriber(depth_topic, Image)
        info_sub  = message_filters.Subscriber(info_topic,  CameraInfo)
        sync = message_filters.ApproximateTimeSynchronizer([depth_sub, info_sub], 10, 0.1)
        sync.registerCallback(self.callback)

        self.pub = rospy.Publisher('~cloud', PointCloud2, queue_size=1)
        self.bridge = CvBridge()
        rospy.loginfo('DepthToCloud: depth="%s", info="%s"', depth_topic, info_topic)

    # ---------------------------------------------------------------
    def callback(self, depth_msg: Image, info_msg: CameraInfo):
        t0 = time.perf_counter() 
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        except CvBridgeError as e:
            rospy.logerr_throttle(5, 'CvBridge: %s', e); return

        h, w = depth.shape
        fx, fy = info_msg.K[0], info_msg.K[4]
        cx, cy = info_msg.K[2], info_msg.K[5]

        # pixel grid (vectorised)
        u = np.tile(np.arange(w), h)
        v = np.repeat(np.arange(h), w)
        z = depth.flatten()
        valid = z > 0
        if not np.any(valid):
            return

        temp_z = z[valid]
        u = u[valid]
        v = v[valid]
        temp_x = (u - cx) * temp_z / fx
        temp_y = (v - cy) * temp_z / fy
        x = temp_z
        y = -temp_x
        z = -temp_y
        # stack and append rgb float
        xyz = np.column_stack((x, y, z))
        rgb = np.full((xyz.shape[0], 1), GREEN_RGB_FLOAT, dtype=np.float32)
        pts = np.hstack((xyz.astype(np.float32), rgb))

        fields = [
            PointField('x', 0,  PointField.FLOAT32, 1),
            PointField('y', 4,  PointField.FLOAT32, 1),
            PointField('z', 8,  PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1)
        ]

        cloud = pc2.create_cloud(info_msg.header, fields, pts)
        cloud.header.frame_id = self.frame or depth_msg.header.frame_id
        self.pub.publish(cloud)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0   # ⬅ 경과 시간(ms)
        print(f"[yolop_lane_detection node] 1 frame = {elapsed_ms:.1f} ms")

    # ---------------------------------------------------------------
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        DepthToCloud().run()
    except rospy.ROSInterruptException:
        pass
