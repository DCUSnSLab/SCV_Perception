#!/usr/bin/env python3

import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class Visualizer:
    def __init__(self, marker_scale=0.2, text_height=0.3, velocity_arrow_scale=1.0):
        self.marker_scale = marker_scale
        self.text_height = text_height
        self.velocity_arrow_scale = velocity_arrow_scale
        
        # Color mapping for different classes
        self.class_colors = {
            'person': (1.0, 0.0, 0.0, 0.8),      # Red
            'car': (0.0, 1.0, 0.0, 0.8),         # Green  
            'truck': (0.0, 0.0, 1.0, 0.8),       # Blue
            'bus': (1.0, 1.0, 0.0, 0.8),         # Yellow
            'bicycle': (1.0, 0.0, 1.0, 0.8),     # Magenta
            'motorcycle': (0.0, 1.0, 1.0, 0.8),  # Cyan
            'default': (0.5, 0.5, 0.5, 0.8)      # Gray
        }
    
    def create_marker_array(self, tracked_objects, frame_id, stamp, include_velocity=False):
        """
        Create MarkerArray for visualization in RViz
        
        Args:
            tracked_objects: list of tracked object dictionaries
            frame_id: coordinate frame for markers
            stamp: timestamp for markers
            include_velocity: whether to include velocity arrows
            
        Returns:
            MarkerArray message
        """
        marker_array = MarkerArray()
        
        # Clear all previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        clear_marker.header.stamp = stamp
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # Create markers for each tracked object
        for obj in tracked_objects:
            # Create sphere marker for object center
            sphere_marker = self._create_sphere_marker(obj, frame_id, stamp)
            marker_array.markers.append(sphere_marker)
            
            # Create text marker for object label
            text_marker = self._create_text_marker(obj, frame_id, stamp)
            marker_array.markers.append(text_marker)
            
            # Create velocity arrow if requested and velocity is available
            if include_velocity and obj.get('velocity') is not None and obj.get('speed', 0) > 0.05:
                velocity_marker = self._create_velocity_arrow_marker(obj, frame_id, stamp)
                if velocity_marker:
                    marker_array.markers.append(velocity_marker)
                    
                # Create speed text marker
                speed_text_marker = self._create_speed_text_marker(obj, frame_id, stamp)
                if speed_text_marker:
                    marker_array.markers.append(speed_text_marker)
        
        return marker_array
    
    def _create_sphere_marker(self, obj, frame_id, stamp):
        """Create sphere marker for object center"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "object_centers"
        marker.id = obj['track_id']
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position.x = float(obj['centroid'][0])
        marker.pose.position.y = float(obj['centroid'][1])
        marker.pose.position.z = float(obj['centroid'][2])
        marker.pose.orientation.w = 1.0
        
        # Set scale
        marker.scale.x = self.marker_scale
        marker.scale.y = self.marker_scale
        marker.scale.z = self.marker_scale
        
        # Set color based on class
        color = self._get_class_color(obj['class_name'])
        marker.color.r = color[0]
        marker.color.g = color[1] 
        marker.color.b = color[2]
        marker.color.a = color[3]
        
        return marker
    
    def _create_text_marker(self, obj, frame_id, stamp):
        """Create text marker for object label"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "object_labels"
        marker.id = obj['track_id']
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Set position (slightly above the object)
        marker.pose.position.x = float(obj['centroid'][0])
        marker.pose.position.y = float(obj['centroid'][1])
        marker.pose.position.z = float(obj['centroid'][2]) + self.text_height
        marker.pose.orientation.w = 1.0
        
        # Set text content
        marker.text = f"ID:{obj['track_id']} {obj['class_name']}"
        if 'confidence' in obj:
            marker.text += f" ({obj['confidence']:.2f})"
        
        # Set text properties
        marker.scale.z = 0.15  # Text size
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        return marker
    
    def _get_class_color(self, class_name):
        """Get color for object class"""
        class_lower = class_name.lower()
        return self.class_colors.get(class_lower, self.class_colors['default'])
    
    def create_debug_marker_array(self, debug_data, frame_id, stamp):
        """
        Create debug markers for development/debugging
        
        Args:
            debug_data: dictionary with debug information
            frame_id: coordinate frame
            stamp: timestamp
            
        Returns:
            MarkerArray for debug visualization
        """
        marker_array = MarkerArray()
        
        # Add debug-specific markers here if needed
        # For example: point clouds, bounding boxes, etc.
        
        return marker_array
    
    def _create_velocity_arrow_marker(self, obj, frame_id, stamp):
        """Create arrow marker for velocity vector"""
        if obj.get('velocity') is None or obj.get('speed', 0) <= 0.05:
            return None
        
        velocity = obj['velocity']
        speed = obj['speed']
        
        # Create arrow marker
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "velocity_arrows"
        marker.id = obj['track_id']
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Set arrow start position (object center)
        marker.pose.position.x = float(obj['centroid'][0])
        marker.pose.position.y = float(obj['centroid'][1])
        marker.pose.position.z = float(obj['centroid'][2])
        
        # Calculate arrow orientation from velocity vector
        if speed > 0.01:  # Only if moving significantly
            # Normalize velocity for direction
            direction = velocity / speed
            
            # Calculate quaternion from direction vector
            # Arrow points in +X direction by default, so we need to rotate from (1,0,0) to direction
            import math
            
            # Calculate yaw angle
            yaw = math.atan2(direction[1], direction[0])
            
            # Calculate pitch angle
            pitch = math.atan2(-direction[2], math.sqrt(direction[0]**2 + direction[1]**2))
            
            # Convert to quaternion (simplified for arrow orientation)
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            cp = math.cos(pitch * 0.5)
            sp = math.sin(pitch * 0.5)
            
            marker.pose.orientation.w = cy * cp
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = sy * sp
            marker.pose.orientation.z = sy * cp
        else:
            marker.pose.orientation.w = 1.0
        
        # Set arrow scale based on speed
        arrow_length = min(speed * self.velocity_arrow_scale, 2.0)  # Max 2 meters
        marker.scale.x = max(arrow_length, 0.1)  # Length
        marker.scale.y = 0.05  # Width
        marker.scale.z = 0.05  # Height
        
        # Set color (blue for velocity)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.8
        
        return marker
    
    def _create_speed_text_marker(self, obj, frame_id, stamp):
        """Create text marker showing speed"""
        if obj.get('speed', 0) <= 0.01:
            return None
        
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "speed_text"
        marker.id = obj['track_id']
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position text below the main label
        marker.pose.position.x = float(obj['centroid'][0])
        marker.pose.position.y = float(obj['centroid'][1])
        marker.pose.position.z = float(obj['centroid'][2]) + self.text_height + 0.2
        marker.pose.orientation.w = 1.0
        
        # Set text content
        marker.text = f"{obj['speed']:.2f} m/s"
        
        # Set text properties (smaller than main label)
        marker.scale.z = 0.12
        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        return marker
    
    def create_history_markers(self, object_history, frame_id, stamp, 
                               show_all_points=True, show_trajectories=True, show_heatmap=False):
        """Create markers for object history visualization"""
        marker_array = MarkerArray()
        
        if show_all_points:
            all_points_marker = self._create_all_points_marker(object_history, frame_id, stamp)
            if all_points_marker:
                marker_array.markers.append(all_points_marker)
        
        if show_trajectories:
            trajectory_markers = self._create_trajectory_markers(object_history, frame_id, stamp)
            marker_array.markers.extend(trajectory_markers)
        
        if show_heatmap:
            heatmap_markers = self._create_heatmap_markers(object_history, frame_id, stamp)
            marker_array.markers.extend(heatmap_markers)
        
        return marker_array
    
    def _create_all_points_marker(self, object_history, frame_id, stamp):
        """Create point cloud of all detected positions"""
        positions = object_history.get_all_positions()
        
        if not positions:
            return None
        
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "all_detections"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        
        # Add all positions as points
        for pos in positions:
            point = Point()
            point.x = float(pos[0])
            point.y = float(pos[1])
            point.z = float(pos[2])
            marker.points.append(point)
        
        # Set point visualization
        marker.scale.x = 0.05  # Point width
        marker.scale.y = 0.05  # Point height
        
        # Semi-transparent white points
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.3
        
        return marker
    
    def _create_trajectory_markers(self, object_history, frame_id, stamp):
        """Create line markers for object trajectories"""
        markers = []
        
        active_trajectories = object_history.get_active_trajectories(min_points=3)
        
        for track_id, traj_data in active_trajectories.items():
            trajectory = traj_data['trajectory']
            class_name = traj_data['class_name']
            
            if len(trajectory) < 2:
                continue
            
            # Create line strip marker
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = "trajectories"
            marker.id = track_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            
            # Add trajectory points
            for pos in trajectory:
                point = Point()
                point.x = float(pos[0])
                point.y = float(pos[1]) 
                point.z = float(pos[2])
                marker.points.append(point)
            
            # Set line properties
            marker.scale.x = 0.02  # Line width
            
            # Color by class
            color = self._get_class_color(class_name)
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.7
            
            markers.append(marker)
        
        return markers
    
    def _create_heatmap_markers(self, object_history, frame_id, stamp, grid_size=0.5):
        """Create heatmap visualization of detection density"""
        markers = []
        
        positions = object_history.get_all_positions()
        if not positions:
            return markers
        
        # Get spatial bounds
        bounds = object_history.get_spatial_bounds()
        if not bounds:
            return markers
        
        positions_array = np.array(positions)
        
        # Create 3D grid for heatmap
        x_min, y_min, z_min = bounds['min']
        x_max, y_max, z_max = bounds['max']
        
        # Only create 2D heatmap (XY plane) for simplicity
        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)
        
        # Create histogram
        hist, x_edges, y_edges = np.histogram2d(
            positions_array[:, 0], positions_array[:, 1], 
            bins=[x_bins, y_bins]
        )
        
        # Create markers for high-density areas
        max_count = np.max(hist) if np.max(hist) > 0 else 1
        
        marker_id = 0
        for i in range(len(x_edges)-1):
            for j in range(len(y_edges)-1):
                count = hist[i, j]
                if count > 1:  # Only show areas with multiple detections
                    
                    marker = Marker()
                    marker.header.frame_id = frame_id
                    marker.header.stamp = stamp
                    marker.ns = "heatmap"
                    marker.id = marker_id
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    
                    # Position at grid center
                    marker.pose.position.x = (x_edges[i] + x_edges[i+1]) / 2.0
                    marker.pose.position.y = (y_edges[j] + y_edges[j+1]) / 2.0
                    marker.pose.position.z = float(bounds['min'][2])  # At ground level
                    marker.pose.orientation.w = 1.0
                    
                    # Size based on grid
                    marker.scale.x = grid_size
                    marker.scale.y = grid_size
                    marker.scale.z = 0.05  # Thin
                    
                    # Color based on density (red = high density)
                    intensity = count / max_count
                    marker.color.r = intensity
                    marker.color.g = 0.0
                    marker.color.b = 1.0 - intensity
                    marker.color.a = 0.5
                    
                    markers.append(marker)
                    marker_id += 1
        
        return markers