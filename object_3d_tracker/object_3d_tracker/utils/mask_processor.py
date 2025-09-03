#!/usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge


class MaskProcessor:
    def __init__(self, min_mask_pixels=20, use_morphology=True):
        self.min_mask_pixels = min_mask_pixels
        self.use_morphology = use_morphology
        self.bridge = CvBridge()
    
    def process_mask(self, mask_msg):
        """
        Process segmentation mask from sensor_msgs/Image
        Returns: processed binary mask as numpy array, or None if invalid
        """
        try:
            # Convert ROS Image message to OpenCV format
            mask = self.bridge.imgmsg_to_cv2(mask_msg, "mono8")
            
            # Convert to binary mask
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Check minimum pixel count
            if np.sum(binary_mask) < self.min_mask_pixels:
                return None
            
            # Apply morphological operations if enabled
            if self.use_morphology:
                binary_mask = self._apply_morphology(binary_mask)
                
                # Recheck pixel count after morphology
                if np.sum(binary_mask) < self.min_mask_pixels:
                    return None
            
            # Get largest connected component
            binary_mask = self._get_largest_component(binary_mask)
            
            return binary_mask
            
        except Exception as e:
            print(f"Error processing mask: {e}")
            return None
    
    def _apply_morphology(self, mask):
        """Apply morphological operations to clean up the mask"""
        kernel = np.ones((3, 3), np.uint8)
        
        # Opening: remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Closing: fill small gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _get_largest_component(self, mask):
        """Keep only the largest connected component"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:  # No components found
            return mask
        
        # Find largest component (excluding background label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = int(np.argmax(areas)) + 1
        
        return (labels == largest_label).astype(np.uint8)
    
    def get_mask_pixels(self, mask):
        """Get (y, x) coordinates of mask pixels"""
        return np.nonzero(mask)