#!/usr/bin/env python3
import rospy
import pyzed.sl as sl
from zed_interfaces.msg import ObjectsStamped
from std_msgs.msg import Bool, Int32MultiArray

class ObjDet():
    def __init__(self, parent=None):
        rospy.init_node('object')

        self.last_obj_time = rospy.Time.now()

        self.stop_pub = rospy.Publisher('object_detect', Bool, queue_size=1)
        
        rospy.Subscriber('/zed_node/obj_det/objects', ObjectsStamped, self.callback, self.stop_pub)

        # 15Hz로 타임아웃 체크를 위한 타이머 추가
        rospy.Timer(rospy.Duration(1.0 / 15), self.check_timeout)

        rospy.spin()

    def callback(self, data: ObjectsStamped, pub):
        self.last_obj_time = rospy.Time.now()  
        detect = False
        for obj in data.objects:
            obj_label = obj.label
            obj_confidence = obj.confidence
            if obj_label == 'Person' and obj_confidence >= 65:
                obj_instance_id = obj.instance_id
                obj_position = obj.position
                x = obj_position[0]
                y = obj_position[1]
                z = obj_position[2]

                distance = (x**2 + y**2 + z**2)**0.5

                # 카메라 - 차: 0.93
                # 차 기준 오. 왼 (-0.37 ~ 0.33)
                if (distance > 0.95 and distance < 5) and (-0.57 <= y <= 0.53):
                    if y == 0:
                        rospy.loginfo(f'STOP, Object: {obj_instance_id}, Distance: {distance}')
                    elif y < 0:
                        rospy.loginfo(f'RIGHT, Object: {obj_instance_id}, Distance: {distance}')
                    elif y > 0:
                        rospy.loginfo(f'LEFT, Object: {obj_instance_id}, Distance: {distance}')
                    detect = True

        self.stop_pub.publish(detect)

    def check_timeout(self, event):
        current_time = rospy.Time.now()
        time_diff = current_time - self.last_obj_time

        if time_diff.to_sec() > 1.0:
            rospy.logwarn("No object detection data received for 3 seconds, assuming camera failure")
            dummy_detect = False  # 카메라가 작동하지 않는 경우 False
            self.stop_pub.publish(dummy_detect)

if __name__ == '__main__':
    ObjDet()
