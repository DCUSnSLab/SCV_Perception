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
            
            # Erode mask by 10% to use more stable center region
            binary_mask = self._erode_mask_by_percentage(binary_mask, erosion_percentage=0.1)
            
            # Final check for minimum pixels after erosion
            if np.sum(binary_mask) < self.min_mask_pixels:
                return None
            
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
    
    def _erode_mask_by_percentage(self, mask, erosion_percentage=0.1):
        """
        Erode mask by a percentage to focus on stable center region
        Args:
            mask: binary mask
            erosion_percentage: percentage to erode (0.1 = 10%)
        Returns:
            eroded binary mask
        """
        if np.sum(mask) < self.min_mask_pixels:
            return mask
        
        # Find bounding box of the mask
        coords = np.nonzero(mask)
        if len(coords[0]) == 0:
            return mask
            
        min_row, max_row = np.min(coords[0]), np.max(coords[0])
        min_col, max_col = np.min(coords[1]), np.max(coords[1])
        
        # Calculate erosion amount based on bounding box size
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        
        # Use smaller dimension for erosion calculation
        min_dimension = min(height, width)
        erosion_pixels = max(1, int(min_dimension * erosion_percentage / 2))
        
        # Create erosion kernel
        kernel_size = 2 * erosion_pixels + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply erosion
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        
        # If erosion removed too much, use smaller kernel
        if np.sum(eroded_mask) < self.min_mask_pixels and erosion_pixels > 1:
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            eroded_mask = cv2.erode(mask, kernel, iterations=1)
        
        return eroded_mask
    
    def get_mask_pixels(self, mask):
        """Get (y, x) coordinates of mask pixels"""
        return np.nonzero(mask)