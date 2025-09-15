#!/usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge


class DepthProcessor:
    def __init__(self, min_depth=0.1, max_depth=10.0, filter_kernel_size=3, min_valid_points=10):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.filter_kernel_size = filter_kernel_size
        self.min_valid_points = min_valid_points
        self.bridge = CvBridge()
    
    def process_depth_image(self, depth_msg):
        """
        Process depth image from sensor_msgs/Image
        Returns: processed depth array as float32
        """
        try:
            # Convert ROS Image message to OpenCV format
            if depth_msg.encoding == "32FC1":
                depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            elif depth_msg.encoding == "16UC1":
                depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
                depth = depth.astype(np.float32) / 1000.0  # Convert mm to meters
            else:
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                depth = depth.astype(np.float32)
            
            # Filter invalid depths
            depth = self._filter_depth(depth)
            
            # Apply median filter to reduce noise
            if self.filter_kernel_size >= 3 and self.filter_kernel_size % 2 == 1:
                depth = cv2.medianBlur(depth, self.filter_kernel_size)
            
            return depth
            
        except Exception as e:
            print(f"Error processing depth image: {e}")
            return None
    
    def _filter_depth(self, depth):
        """Filter out invalid depth values"""
        # Set invalid depths to 0
        depth[(depth < self.min_depth) | (depth > self.max_depth) | (~np.isfinite(depth))] = 0.0
        return depth
    
    def mask_to_3d_points(self, mask, depth, camera_info):
        """
        Convert mask pixels to 3D points using depth and camera parameters
        Returns: numpy array of 3D points (N, 3) or None if insufficient points
        """
        try:
            # Get camera intrinsic parameters
            fx = camera_info.k[0]
            fy = camera_info.k[4] 
            cx = camera_info.k[2]
            cy = camera_info.k[5]
            
            # Get mask pixel coordinates
            v_coords, u_coords = np.nonzero(mask)
            
            if len(v_coords) < self.min_valid_points:
                return None
            
            # Get depth values at mask locations
            depth_values = depth[v_coords, u_coords]
            
            # Filter out invalid depths
            valid_mask = (depth_values >= self.min_depth) & (depth_values <= self.max_depth)
            
            if np.sum(valid_mask) < self.min_valid_points:
                return None
            
            # Keep only valid points
            u_valid = u_coords[valid_mask]
            v_valid = v_coords[valid_mask] 
            z_valid = depth_values[valid_mask]
            
            # Convert to 3D coordinates (camera frame)
            x_3d = (u_valid - cx) * z_valid / fx
            y_3d = (v_valid - cy) * z_valid / fy
            z_3d = z_valid
            
            points_3d = np.column_stack([x_3d, y_3d, z_3d])
            
            return points_3d
            
        except Exception as e:
            print(f"Error converting mask to 3D points: {e}")
            return None
    
    def compute_centroid(self, points_3d, outlier_removal=True):
        """
        Compute 3D centroid from point cloud
        Returns: centroid as numpy array (3,) or None
        """
        if points_3d is None or len(points_3d) < self.min_valid_points:
            return None
        
        try:
            if outlier_removal:
                points_3d = self._remove_outliers(points_3d)
                
                if len(points_3d) < self.min_valid_points:
                    return None
            
            # Compute centroid
            centroid = np.mean(points_3d, axis=0)
            # Set Z coordinate to 0 (ground level)
            centroid[1] = 0.0
            return centroid.astype(np.float32)
            
        except Exception as e:
            print(f"Error computing centroid: {e}")
            return None
    
    def _remove_outliers(self, points_3d, std_multiplier=2.0):
        """Remove outliers using statistical method"""
        if len(points_3d) < 10:
            return points_3d
        
        try:
            # Compute distances from median point
            median_point = np.median(points_3d, axis=0)
            distances = np.linalg.norm(points_3d - median_point, axis=1)
            
            # Remove points beyond threshold
            threshold = np.median(distances) + std_multiplier * np.std(distances)
            valid_mask = distances <= threshold
            
            return points_3d[valid_mask]
            
        except Exception as e:
            print(f"Error removing outliers: {e}")
            return points_3d