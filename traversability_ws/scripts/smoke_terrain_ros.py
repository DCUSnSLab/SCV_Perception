"""Isolated-domain DDS smoke test with synthetic incline, IMU and odometry."""
import json
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import String
from ugv_self_supervised_traversability.terrain_node import TerrainAnalyzer
from ugv_self_supervised_traversability.utils.transforms import quaternion_from_euler, transform_points, transform_matrix


def main():
    rclpy.init()
    analyzer=TerrainAnalyzer()
    sensor=Node('terrain_smoke_sensor')
    received=[]
    sensor.create_subscription(String,'/traversability/terrain_status',lambda msg:received.append(json.loads(msg.data)),10)
    odom_pub=sensor.create_publisher(Odometry,'/odom',qos_profile_sensor_data)
    imu_pub=sensor.create_publisher(Imu,'/imu/data',qos_profile_sensor_data)
    cloud_pub=sensor.create_publisher(PointCloud2,'/velodyne_points',qos_profile_sensor_data)
    executor=SingleThreadedExecutor();executor.add_node(analyzer);executor.add_node(sensor)
    x,y=np.meshgrid(np.arange(-5.9,6,.08),np.arange(-5.9,6,.08))
    points=np.column_stack([x.ravel(),y.ravel(),x.ravel()*np.tan(np.radians(14))]).astype('<f4')
    cloud=PointCloud2();cloud.header.frame_id='base_link';cloud.height=1;cloud.width=len(points)
    cloud.point_step=12;cloud.row_step=12*len(points);cloud.is_dense=True;cloud.data=points.tobytes()
    cloud.fields=[PointField(name=name,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,name in enumerate(('x','y','z'))]
    imu=Imu();imu.header.frame_id='base_link'
    q=quaternion_from_euler(0,-np.radians(14),0)
    imu.orientation.x,imu.orientation.y,imu.orientation.z,imu.orientation.w=map(float,q)
    odom=Odometry();odom.header.frame_id='odom';odom.child_frame_id='base_link';odom.pose.pose.orientation.w=1.0
    begin=time.monotonic();last=0
    try:
        while time.monotonic()-begin<4.0:
            now=time.monotonic()
            if now-last>.1:
                stamp=sensor.get_clock().now().to_msg()
                imu.header.stamp=stamp;odom.header.stamp=stamp;cloud.header.stamp=stamp
                # Simulate a LiDAR rigidly mounted in the pitched chassis.
                cloud.data=transform_points(points, np.linalg.inv(transform_matrix(
                    [(now-begin)*.2,0,0], q))).astype('<f4').tobytes()
                odom.pose.pose.position.x=(now-begin)*.2
                imu_pub.publish(imu);odom_pub.publish(odom);cloud_pub.publish(cloud);last=now
            executor.spin_once(timeout_sec=.02)
        valid=[r for r in received if 'forward' in r]
        assert valid, 'No analysis arrived through DDS'
        assert any(r['forward']['status']=='GEOMETRY_PASS' and r['reverse']['status']=='GEOMETRY_BLOCKED' for r in valid), valid[-1]
        assert any(r['motion']['state']=='ASCENDING' for r in valid), valid[-1]
        assert all(r['cloud_orientation_source']=='imu_roll_pitch+odometry_yaw_translation' for r in valid)
        before=len(received);deadline=time.monotonic()+2
        while time.monotonic()<deadline:executor.spin_once(timeout_sec=.05)
        assert any(r.get('reason')=='pointcloud_missing_or_stale' for r in received[before:])
        # Regression: ROS 2 Humble TF Buffer must clear on simulated time rewind.
        from rclpy.parameter import Parameter
        from rclpy.time import Time
        from geometry_msgs.msg import TransformStamped
        analyzer.set_parameters([Parameter('use_sim_time', value=True)])
        analyzer.get_clock().set_ros_time_override(Time(seconds=100))
        transform=TransformStamped()
        transform.header.frame_id='rewind_world'
        transform.child_frame_id='rewind_base'
        transform.header.stamp=Time(seconds=100).to_msg()
        transform.transform.rotation.w=1.0
        analyzer.tf.set_transform(transform,'test')
        assert analyzer.tf.can_transform('rewind_world','rewind_base',Time(seconds=100))
        analyzer.get_clock().set_ros_time_override(Time(seconds=1))
        assert not analyzer.tf.can_transform('rewind_world','rewind_base',Time(seconds=100))
        print('PASS: DDS inputs -> IMU ascent, directional ramp, stale UNKNOWN, TF cleared on bag rewind')
    finally:
        executor.shutdown();analyzer.destroy_node();sensor.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
