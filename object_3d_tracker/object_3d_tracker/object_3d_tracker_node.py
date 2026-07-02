#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import message_filters
import time

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import MarkerArray
from perception_interface.msg import DetectionArray

from .utils.mask_processor import MaskProcessor
from .utils.depth_processor import DepthProcessor  
from .utils.visualizer import Visualizer
from .utils.velocity_tracker import VelocityTracker
from .utils.object_history import ObjectHistory


class Object3DTracker(Node):
    def __init__(self):
        super().__init__('object_3d_tracker_node')
        
        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('detection_topic', '/detection/results'),
                ('depth_topic', '/camera/depth/image_raw'),
                ('camera_info_topic', '/camera/depth/camera_info'),
                ('output_topic', '/object_3d_tracker/markers'),
                ('queue_size', 10),
                ('slop', 0.1),  # seconds
                ('min_mask_pixels', 20),
                ('min_depth', 0.1),
                ('max_depth', 10.0),
                ('depth_filter_kernel_size', 3),
                ('marker_scale', 0.2),
                ('text_height', 0.3),
                ('use_morphology', True),
                ('publish_debug', False),
                # Velocity tracking parameters
                ('velocity_history_size', 10),
                ('velocity_smoothing_window', 3),
                ('min_velocity_threshold', 0.05),
                ('velocity_outlier_threshold', 10.0),
                ('publish_velocity_markers', True),
                ('velocity_arrow_scale', 1.0),
                # History visualization parameters
                ('enable_history', True),
                ('history_max_size', 1000),
                ('history_max_age', 300.0),
                ('show_all_points', True),
                ('show_trajectories', True),
                ('show_heatmap', False),
            ]
        )
        
        # Get parameters
        self.detection_topic = self.get_parameter('detection_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value  
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.queue_size = self.get_parameter('queue_size').value
        self.slop = self.get_parameter('slop').value
        self.publish_debug = self.get_parameter('publish_debug').value
        
        # Initialize processors
        self.mask_processor = MaskProcessor(
            min_mask_pixels=self.get_parameter('min_mask_pixels').value,
            use_morphology=self.get_parameter('use_morphology').value
        )
        
        self.depth_processor = DepthProcessor(
            min_depth=self.get_parameter('min_depth').value,
            max_depth=self.get_parameter('max_depth').value,
            filter_kernel_size=self.get_parameter('depth_filter_kernel_size').value
        )
        
        self.visualizer = Visualizer(
            marker_scale=self.get_parameter('marker_scale').value,
            text_height=self.get_parameter('text_height').value,
            velocity_arrow_scale=self.get_parameter('velocity_arrow_scale').value
        )
        
        self.velocity_tracker = VelocityTracker(
            history_size=self.get_parameter('velocity_history_size').value,
            smoothing_window=self.get_parameter('velocity_smoothing_window').value,
            min_velocity_threshold=self.get_parameter('min_velocity_threshold').value,
            velocity_outlier_threshold=self.get_parameter('velocity_outlier_threshold').value
        )
        
        self.publish_velocity_markers = self.get_parameter('publish_velocity_markers').value
        
        # Initialize object history
        self.enable_history = self.get_parameter('enable_history').value
        if self.enable_history:
            self.object_history = ObjectHistory(
                max_history_size=self.get_parameter('history_max_size').value,
                max_age_seconds=self.get_parameter('history_max_age').value
            )
            self.show_all_points = self.get_parameter('show_all_points').value
            self.show_trajectories = self.get_parameter('show_trajectories').value
            self.show_heatmap = self.get_parameter('show_heatmap').value
        
        # Setup QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        detection_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )
        
        # Setup synchronized subscribers
        self.detection_sub = message_filters.Subscriber(
            self, DetectionArray, self.detection_topic, qos_profile=detection_qos
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=sensor_qos
        )
        self.camera_info_sub = message_filters.Subscriber(
            self, CameraInfo, self.camera_info_topic, qos_profile=sensor_qos
        )
        
        # Setup synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.detection_sub, self.depth_sub, self.camera_info_sub],
            queue_size=self.queue_size,
            slop=self.slop,
            allow_headerless=False
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        # Setup publishers
        self.marker_pub = self.create_publisher(
            MarkerArray, self.output_topic, QoSProfile(depth=2)
        )
        
        if self.publish_debug:
            self.debug_pub = self.create_publisher(
                MarkerArray, '/object_3d_tracker/debug', QoSProfile(depth=2)
            )
        
        # History visualization publisher
        if self.enable_history:
            self.history_pub = self.create_publisher(
                MarkerArray, '/object_3d_tracker/history', QoSProfile(depth=2)
            )
        
        self.get_logger().info(
            f'Object 3D Tracker initialized:\n'
            f'  Detection topic: {self.detection_topic}\n' 
            f'  Depth topic: {self.depth_topic}\n'
            f'  Camera info topic: {self.camera_info_topic}\n'
            f'  Output topic: {self.output_topic}'
        )
    
    def synchronized_callback(self, detection_msg, depth_msg, camera_info_msg):
        """Main callback for synchronized messages"""
        start_time = time.perf_counter()
        
        # Process depth image
        depth_array = self.depth_processor.process_depth_image(depth_msg)
        if depth_array is None:
            self.get_logger().warning('Failed to process depth image')
            return
        
        # Process detections
        tracked_objects = []
        
        for detection in detection_msg.detections:
            # Skip detections without tracking ID
            if detection.track_id < 0:
                continue
            
            # Skip detections without segmentation mask
            if not hasattr(detection, 'mask') or detection.mask.width == 0:
                continue
            
            # Process segmentation mask
            processed_mask = self.mask_processor.process_mask(detection.mask)
            if processed_mask is None:
                continue
            
            # Convert mask to 3D points
            points_3d = self.depth_processor.mask_to_3d_points(
                processed_mask, depth_array, camera_info_msg
            )
            if points_3d is None:
                continue
            
            # Compute 3D centroid
            centroid = self.depth_processor.compute_centroid(points_3d)
            if centroid is None:
                continue
            
            # Update velocity tracking
            self.velocity_tracker.update_position(
                detection.track_id, centroid, depth_msg.header.stamp
            )
            
            # Get velocity information
            velocity = self.velocity_tracker.get_velocity(detection.track_id)
            speed = self.velocity_tracker.get_speed(detection.track_id)
            
            # Debug: log velocity tracking details
            if detection.track_id in self.velocity_tracker.tracked_objects:
                history = self.velocity_tracker.tracked_objects[detection.track_id]
                self.get_logger().info(
                    f"DEBUG ID:{detection.track_id} - positions:{len(history.positions)}, "
                    f"velocities:{len(history.velocities)}, is_stable:{history.is_stable()}",
                    throttle_duration_sec=1.0
                )
            
            # Store tracked object data
            tracked_object = {
                'track_id': detection.track_id,
                'class_name': detection.class_name,
                'confidence': detection.confidence,
                'centroid': centroid,
                'num_points': len(points_3d),
                'velocity': velocity,
                'speed': speed
            }
            tracked_objects.append(tracked_object)
        
        # Create and publish visualization markers
        if tracked_objects:
            frame_id = depth_msg.header.frame_id
            stamp = depth_msg.header.stamp
            
            marker_array = self.visualizer.create_marker_array(
                tracked_objects, frame_id, stamp, include_velocity=self.publish_velocity_markers
            )
            self.marker_pub.publish(marker_array)
            
            # Add objects to history
            if self.enable_history:
                self.object_history.add_objects(tracked_objects, depth_msg.header.stamp)
                
                # Publish history visualization
                history_markers = self.visualizer.create_history_markers(
                    self.object_history, frame_id, stamp,
                    show_all_points=self.show_all_points,
                    show_trajectories=self.show_trajectories,
                    show_heatmap=self.show_heatmap
                )
                self.history_pub.publish(history_markers)
        
        # Log performance and results
        processing_time = (time.perf_counter() - start_time) * 1000.0
        
        self.get_logger().info(
            f'Processed {len(detection_msg.detections)} detections -> '
            f'{len(tracked_objects)} valid 3D objects '
            f'({processing_time:.1f}ms)',
            throttle_duration_sec=1.0
        )
        
        # Cleanup old tracks
        active_track_ids = [obj['track_id'] for obj in tracked_objects]
        removed_count = self.velocity_tracker.cleanup_old_tracks(active_track_ids)
        
        # Log individual object info with velocity
        for obj in tracked_objects:
            velocity_str = ""
            if obj['velocity'] is not None and obj['speed'] > 0.005:  # Lower threshold
                vel = obj['velocity']
                velocity_str = f' vel:({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}) speed:{obj["speed"]:.2f}m/s'
            elif obj['velocity'] is None:
                velocity_str = ' [new track]'
            
            self.get_logger().info(
                f'  ID:{obj["track_id"]} {obj["class_name"]} at '
                f'({obj["centroid"][0]:.2f}, {obj["centroid"][1]:.2f}, {obj["centroid"][2]:.2f}) '
                f'[{obj["num_points"]} points]{velocity_str}',
                throttle_duration_sec=2.0
            )
        
        # Log velocity tracking stats periodically
        if len(tracked_objects) > 0:
            stats = self.velocity_tracker.get_tracking_stats()
            self.get_logger().info(
                f'Velocity tracking: {stats["stable_tracks"]}/{stats["total_tracks"]} stable, '
                f'{stats["moving_tracks"]} moving',
                throttle_duration_sec=5.0
            )
            
            # Log history stats periodically
            if self.enable_history:
                hist_stats = self.object_history.get_summary_stats()
                self.get_logger().info(
                    f'Object history: {hist_stats["total_detections"]} total detections, '
                    f'{hist_stats["total_unique_tracks"]} unique tracks, '
                    f'{hist_stats["recent_active_tracks"]} recent active',
                    throttle_duration_sec=10.0
                )


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    node = Object3DTracker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()