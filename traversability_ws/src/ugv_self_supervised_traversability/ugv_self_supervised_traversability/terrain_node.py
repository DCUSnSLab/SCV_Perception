"""ROS adapter for gravity-aligned, directional LiDAR terrain assessment."""
from collections import deque
from dataclasses import asdict, fields
import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.clock import JumpThreshold
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .pose_fusion import PoseFusion, pointcloud_to_xyz, stamp_to_ns
from .terrain_analysis import TerrainConfig, analyze_terrain, MotionEstimator, attitude_motion
from .utils.transforms import transform_matrix, transform_points, quaternion_from_euler


class TerrainAnalyzer(Node):
    def __init__(self):
        super().__init__('terrain_analyzer')
        for name, default in dict(
            world_frame='odom', base_frame='base_link', odom_topic='/odom',
            imu_topic='/imu/data', pointcloud_topic='/velodyne_points',
            sensor_timeout_seconds=0.25, analysis_rate=2.0,
            odometry_z_valid=False, use_imu_cloud_leveling=True).items():
            self.declare_parameter(name, default)
        for field in fields(TerrainConfig):
            self.declare_parameter(field.name, getattr(TerrainConfig(),field.name))
        self.config = TerrainConfig(**{f.name:self.get_parameter(f.name).value for f in fields(TerrainConfig)})
        self.world = self.get_parameter('world_frame').value
        self.base = self.get_parameter('base_frame').value
        self.timeout = float(self.get_parameter('sensor_timeout_seconds').value)
        rate = float(self.get_parameter('analysis_rate').value)
        if self.timeout <= 0 or rate <= 0:
            raise ValueError('Timeout and analysis rate must be positive')
        self.period = 1.0/rate
        self.tf = Buffer(cache_time=Duration(seconds=30.0), node=self)
        self.listener = TransformListener(self.tf,self)
        self.fusion = PoseFusion(self.tf,self.world,base_frame=self.base,
            use_imu_orientation=True,imu_use_yaw=False,imu_timeout_seconds=self.timeout,
            use_lidar_ground_height=False,pointcloud_timeout_seconds=self.timeout,
            lidar_ground_radius=1.2,lidar_ground_min_points=20,lidar_ground_percentile=20)
        self.poses = deque(maxlen=500)
        self.motion = MotionEstimator()
        self.last_cloud_ns = None
        self.status_pub = self.create_publisher(String,'/traversability/terrain_status',10)
        self.marker_pub = self.create_publisher(MarkerArray,'/traversability/terrain_markers',10)
        self.odom_sub = self.create_subscription(Odometry,self.get_parameter('odom_topic').value,self.on_odom,qos_profile_sensor_data)
        self.imu_sub = self.create_subscription(Imu,self.get_parameter('imu_topic').value,self.fusion.update_imu,qos_profile_sensor_data)
        self.cloud_sub = self.create_subscription(PointCloud2,self.get_parameter('pointcloud_topic').value,self.on_cloud,qos_profile_sensor_data)
        self.last_receipt = None
        self.stale_sent = False
        self.watchdog = self.create_timer(0.5,self.check_stale)
        self.time_jump = self.get_clock().create_jump_callback(
            JumpThreshold(min_backward=Duration(nanoseconds=-1), on_clock_change=True),
            post_callback=self.on_time_jump)
        self.get_logger().info('Directional terrain geometry enabled; calibrated limits: %s' % self.config.limits_calibrated)

    def on_time_jump(self, jump):
        """Humble Buffer does not clear automatically when a bag loops."""
        self.tf.clear()
        self.poses.clear()
        self.motion.history.clear()
        self.fusion.clear_sensor_cache()
        self.last_cloud_ns = None
        self.last_receipt = None
        self.stale_sent = False

    def on_odom(self,msg):
        if msg.header.frame_id != self.world:
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        if not np.isfinite([p.x,p.y,p.z,q.x,q.y,q.z,q.w]).all() or np.linalg.norm([q.x,q.y,q.z,q.w]) < 1e-8:
            return
        t = stamp_to_ns(msg.header.stamp)
        if self.poses and t <= self.poses[-1][0]:
            self.poses.clear()
        fused = self.fusion.fuse_odometry(msg)
        odom_motion = self.motion.update(t/1e9,p.x,p.y,p.z)
        movement = dict(odom_motion)
        if not self.get_parameter('odometry_z_valid').value:
            movement = dict(state='UNKNOWN',source='none',reason='odometry_z_not_validated')
        if fused.orientation_source == 'odometry+imu' and len(self.motion.history)>1:
            old = self.motion.history[0]
            dx,dy = p.x-old[1],p.y-old[2]
            if t/1e9-old[0] >= self.motion.window*0.5:
                movement = attitude_motion(fused.roll, fused.pitch, fused.yaw, dx, dy,
                                           self.motion.min_distance, self.motion.threshold)
        self.poses.append((t,fused,movement,odom_motion))

    def unknown(self,reason):
        self.status_pub.publish(String(data=json.dumps(dict(
            status='UNKNOWN',reason=reason,calibrated=self.config.limits_calibrated))))
        clear = Marker(); clear.action = Marker.DELETEALL
        self.marker_pub.publish(MarkerArray(markers=[clear]))

    def check_stale(self):
        now = self.get_clock().now().nanoseconds
        if self.last_receipt is None or now-self.last_receipt > int(max(2*self.period,1.0)*1e9) or now < self.last_receipt:
            if not self.stale_sent:
                self.unknown('pointcloud_missing_or_stale')
                self.stale_sent = True

    def on_cloud(self,msg):
        timestamp = stamp_to_ns(msg.header.stamp)
        self.last_receipt = self.get_clock().now().nanoseconds
        self.stale_sent = False
        if self.last_cloud_ns is not None and timestamp < self.last_cloud_ns:
            self.poses.clear(); self.last_cloud_ns = None
            self.unknown('time_reversed'); return
        if self.last_cloud_ns is not None and timestamp-self.last_cloud_ns < self.period*1e9:
            return
        self.last_cloud_ns = timestamp
        if not self.poses or not msg.header.frame_id:
            self.unknown('odometry_or_cloud_frame_missing'); return
        t, pose, movement, odom_motion = min(self.poses,key=lambda p:abs(p[0]-timestamp))
        if abs(t-timestamp)>self.timeout*1e9:
            self.unknown('odometry_out_of_sync'); return
        imu_fresh = (pose.imu_timestamp_ns is not None and
                     abs(pose.imu_timestamp_ns-timestamp) <= self.timeout*1e9)
        if movement.get('source') == 'imu_attitude_and_odometry_direction' and not imu_fresh:
            movement = odom_motion if self.get_parameter('odometry_z_valid').value else dict(
                state='UNKNOWN', source='none', reason='imu_out_of_sync')
        try:
            points = pointcloud_to_xyz(msg)
            cloud_orientation_source = 'recorded_world_tf'
            if (self.get_parameter('use_imu_cloud_leveling').value and
                    imu_fresh):
                # Per-scan local map: use IMU roll/pitch even when odometry TF
                # is planar. Translation remains odometry's base height.
                if msg.header.frame_id != self.base:
                    tf = self.tf.lookup_transform(
                        self.base, msg.header.frame_id, Time.from_msg(msg.header.stamp),
                        timeout=Duration(seconds=0.0))
                    p, q = tf.transform.translation, tf.transform.rotation
                    points = transform_points(points, transform_matrix(
                        [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]))
                points = transform_points(points, transform_matrix(
                    [pose.x, pose.y, pose.base_z],
                    quaternion_from_euler(pose.roll, pose.pitch, pose.yaw)))
                cloud_orientation_source = 'imu_roll_pitch+odometry_yaw_translation'
            elif msg.header.frame_id != self.world:
                tf = self.tf.lookup_transform(
                    self.world, msg.header.frame_id, Time.from_msg(msg.header.stamp),
                    timeout=Duration(seconds=0.0))
                p, q = tf.transform.translation, tf.transform.rotation
                points = transform_points(points, transform_matrix(
                    [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]))
            result = analyze_terrain(points,pose.x,pose.y,pose.yaw,self.config)
        except (TransformException,ValueError,TypeError) as error:
            self.unknown(str(error)); return
        status = dict(timestamp_ns=timestamp,world_frame=self.world,
            evaluation='per_scan_geometry_only',limits=asdict(self.config),
            cloud_orientation_source=cloud_orientation_source,
            orientation_source=pose.orientation_source,imu_status=self.fusion.imu_status,
            imu_fresh=imu_fresh,
            roll_deg=math.degrees(pose.roll),pitch_deg=math.degrees(pose.pitch),
            motion=movement,odometry_motion=odom_motion,
            forward=result.corridor(1),reverse=result.corridor(-1))
        self.status_pub.publish(String(data=json.dumps(status,allow_nan=False)))
        self.publish_markers(result,pose,msg.header.stamp,status)

    def publish_markers(self,result,pose,stamp,status):
        markers = []
        palette = [(0.5,0.55,0.65),(0.15,0.85,0.45),(1.0,0.25,0.2)]
        for d,name in enumerate(('forward','reverse')):
            marker = Marker(); marker.header.frame_id=self.world; marker.header.stamp=stamp
            marker.ns=name; marker.id=d; marker.type=Marker.CUBE_LIST; marker.action=Marker.ADD
            marker.pose.orientation.w=1.0
            marker.scale.x=marker.scale.y=self.config.resolution*0.9; marker.scale.z=0.04
            marker.lifetime=Duration(seconds=max(2*self.period,1.0)).to_msg()
            # Offset the reverse layer slightly in z to allow selecting namespaces in RViz.
            for row,col in np.argwhere(result.observed):
                x,y,z=result.centers[row,col]
                marker.points.append(Point(x=float(pose.x+math.cos(pose.yaw)*x-math.sin(pose.yaw)*y),
                    y=float(pose.y+math.sin(pose.yaw)*x+math.cos(pose.yaw)*y),z=float(z+d*0.06)))
                r,g,b=palette[result.status[d,row,col]]
                marker.colors.append(ColorRGBA(r=r,g=g,b=b,a=0.8 if d==0 else 0.45))
            markers.append(marker)
        text=Marker(); text.header=markers[0].header; text.ns='status'; text.id=2
        text.type=Marker.TEXT_VIEW_FACING; text.action=Marker.ADD; text.pose.orientation.w=1.0
        text.pose.position=Point(x=pose.x,y=pose.y,z=pose.base_z+1.7)
        text.scale.z=0.25; text.color=ColorRGBA(r=1.0,g=1.0,b=1.0,a=1.0)
        text.lifetime=markers[0].lifetime
        text.text='Motion: %s\nForward: %s\nReverse: %s\nLimits calibrated: %s' % (
            status['motion']['state'],status['forward']['status'],status['reverse']['status'],self.config.limits_calibrated)
        markers.append(text)
        self.marker_pub.publish(MarkerArray(markers=markers))


def main(args=None):
    rclpy.init(args=args)
    node=TerrainAnalyzer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.try_shutdown()
