#!/usr/bin/env python3
"""
Fuse a mono-8 lane-line mask with a 32FC1 depth image to produce a
masked-depth image that contains depth only on lane pixels.
Publishes:
  • /lane_node/lane_depth   (sensor_msgs/Image, 32FC1)
  • /lane_node/overlay_rgb  (sensor_msgs/Image, bgr8) – visual debug
Usage notes:
  - Subscribes to (synchronised) depth image + lane mask topics
  - Accepts dynamic topic remapping via ROS launch.
"""
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import numpy as np
import cv2

class LaneDepthFusion:
    def __init__(self):
        rospy.init_node('lane_depth_fusion', anonymous=True)

        # -------- Parameters --------
        depth_topic = rospy.get_param('~depth_topic',  '/zed/depth/image')
        mask_topic  = rospy.get_param('~mask_topic',   '/lane_line_node/lane_line')
        overlay     = rospy.get_param('~publish_overlay', True)
        self.scale_depth = rospy.get_param('~depth_scale', 1.0)  # if depth needs metres conversion

        # -------- I/O --------
        self.bridge = CvBridge()
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        mask_sub  = message_filters.Subscriber(mask_topic,  Image)
        ats = message_filters.ApproximateTimeSynchronizer([depth_sub, mask_sub], queue_size=5, slop=0.05)
        ats.registerCallback(self.callback)

        self.depth_pub   = rospy.Publisher('~lane_depth',   Image, queue_size=1)
        self.overlay_pub = rospy.Publisher('~overlay_rgb', Image, queue_size=1) if overlay else None

        rospy.loginfo('LaneDepthFusion ready. depth="%s" mask="%s"', depth_topic, mask_topic)

    # ---------------------------------------------------------------
    def callback(self, depth_msg: Image, mask_msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)
            mask  = self.bridge.imgmsg_to_cv2(mask_msg,  desired_encoding='mono8')
        except CvBridgeError as e:
            rospy.logerr_throttle(5, 'CvBridge error: %s', e)
            return

        # normalise mask to {0,1}
        mask_bin = (mask > 0).astype(np.float32)
        fused    = depth * mask_bin * self.scale_depth

        # Publish fused depth
        try:
            fused_msg = self.bridge.cv2_to_imgmsg(fused, encoding='32FC1')
            fused_msg.header.stamp = depth_msg.header.stamp
            fused_msg.header.frame_id = depth_msg.header.frame_id
            self.depth_pub.publish(fused_msg)
        except CvBridgeError as e:
            rospy.logerr_throttle(5, 'CvBridge error publish: %s', e)

        # Optional overlay for visualisation
        if self.overlay_pub and self.overlay_pub.get_num_connections() > 0:
            overlay = cv2.cvtColor((mask_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            overlay[:,:,1] = 255  # make green for lanes
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            show = cv2.addWeighted(depth_vis, 1.0, overlay, 0.6, 0)
            show_msg = self.bridge.cv2_to_imgmsg(show, encoding='bgr8')
            show_msg.header = fused_msg.header
            self.overlay_pub.publish(show_msg)

    # ---------------------------------------------------------------
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = LaneDepthFusion()
        node.run()
    except rospy.ROSInterruptException:
        pass
