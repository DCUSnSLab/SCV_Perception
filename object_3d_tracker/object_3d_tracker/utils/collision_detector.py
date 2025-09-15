#!/usr/bin/env python3

import numpy as np
import math


class TTCCalculator:
    """
    Time to Collision (TTC) calculator using vector-based approach
    """
    
    def __init__(self, collision_threshold=1.0, min_speed_threshold=0.1):
        """
        Args:
            collision_threshold: minimum distance considered as collision (meters)
            min_speed_threshold: minimum relative speed to consider (m/s)
        """
        self.collision_threshold = collision_threshold
        self.min_speed_threshold = min_speed_threshold
    
    def calculate_ttc(self, vehicle_pos, vehicle_vel, object_pos, object_vel):
        """
        Calculate Time to Collision using vector-based approach
        
        Args:
            vehicle_pos: (x, y, z) position of vehicle in odom frame
            vehicle_vel: (vx, vy, vz) velocity of vehicle in odom frame  
            object_pos: (x, y, z) position of object in odom frame
            object_vel: (vx, vy, vz) velocity of object in odom frame
            
        Returns:
            dict with TTC info or None if no collision expected
            {
                'ttc': float,  # time to collision in seconds
                'closest_distance': float,  # minimum distance at closest approach
                'closest_time': float,  # time when closest approach occurs
                'collision_point': (x, y, z),  # predicted collision point
                'relative_speed': float  # relative speed magnitude
            }
        """
        # Convert to numpy arrays
        vehicle_pos = np.array(vehicle_pos, dtype=np.float64)
        vehicle_vel = np.array(vehicle_vel, dtype=np.float64)
        object_pos = np.array(object_pos, dtype=np.float64)
        object_vel = np.array(object_vel, dtype=np.float64)
        
        # Calculate relative vectors
        relative_pos = object_pos - vehicle_pos  # vector from vehicle to object
        relative_vel = object_vel - vehicle_vel  # relative velocity vector
        
        # Check if relative speed is significant
        relative_speed = np.linalg.norm(relative_vel)
        if relative_speed < self.min_speed_threshold:
            return None  # Objects are essentially stationary relative to each other
        
        # Check if objects are approaching each other
        # If dot product > 0, they are moving away from each other
        dot_product = np.dot(relative_pos, relative_vel)
        if dot_product > 0:
            return None  # Objects are moving away from each other
        
        # Calculate time of closest approach
        # t = -(relative_pos · relative_vel) / |relative_vel|²
        relative_vel_squared = np.dot(relative_vel, relative_vel)
        if relative_vel_squared < 1e-10:  # Avoid division by zero
            return None
            
        closest_time = -dot_product / relative_vel_squared
        
        # Closest time should be in the future
        if closest_time < 0:
            return None
        
        # Calculate position vectors at closest approach
        vehicle_pos_at_closest = vehicle_pos + vehicle_vel * closest_time
        object_pos_at_closest = object_pos + object_vel * closest_time
        
        # Calculate minimum distance at closest approach
        closest_distance = np.linalg.norm(object_pos_at_closest - vehicle_pos_at_closest)
        
        # Check if this constitutes a collision
        if closest_distance > self.collision_threshold:
            return None  # Objects will pass by safely
        
        # Calculate collision point (midpoint between objects at closest approach)
        collision_point = (vehicle_pos_at_closest + object_pos_at_closest) / 2.0
        
        return {
            'ttc': closest_time,
            'closest_distance': closest_distance,
            'closest_time': closest_time,
            'collision_point': collision_point,
            'relative_speed': relative_speed,
            'severity': self._calculate_severity(closest_time, closest_distance, relative_speed)
        }
    
    def _calculate_severity(self, ttc, distance, relative_speed):
        """
        Calculate collision severity/urgency
        Returns: 'critical', 'warning', 'caution'
        """
        if ttc < 2.0 and distance < 0.5:
            return 'critical'
        elif ttc < 5.0 and distance < 1.0:
            return 'warning'
        else:
            return 'caution'
    
    def calculate_ttc_2d(self, vehicle_pos, vehicle_vel, object_pos, object_vel):
        """
        Calculate TTC using only X-Y coordinates (ignoring Z for ground vehicles)
        """
        # Use only X-Y components
        vehicle_pos_2d = vehicle_pos[:2]
        vehicle_vel_2d = vehicle_vel[:2]
        object_pos_2d = object_pos[:2]
        object_vel_2d = object_vel[:2]
        
        result = self.calculate_ttc(
            np.append(vehicle_pos_2d, 0),  # Add Z=0
            np.append(vehicle_vel_2d, 0),
            np.append(object_pos_2d, 0), 
            np.append(object_vel_2d, 0)
        )
        
        if result:
            # Update collision point to 2D
            result['collision_point'] = result['collision_point'][:2]
        
        return result


class CollisionDetector:
    """
    Manages collision detection for multiple objects
    """
    
    def __init__(self, collision_threshold=1.0, min_speed_threshold=0.1):
        self.ttc_calculator = TTCCalculator(collision_threshold, min_speed_threshold)
        self.collision_history = {}  # Store recent collision predictions
        self.max_history_size = 10
    
    def detect_collisions(self, vehicle_state, tracked_objects):
        """
        Detect potential collisions with all tracked objects
        
        Args:
            vehicle_state: dict with 'position' and 'velocity' 
            tracked_objects: list of tracked object dictionaries
            
        Returns:
            list of collision predictions sorted by urgency
        """
        collision_predictions = []
        
        vehicle_pos = vehicle_state['position']
        vehicle_vel = vehicle_state['velocity']
        
        for obj in tracked_objects:
            if obj['velocity'] is None:
                continue  # Skip objects without velocity info
            
            ttc_result = self.ttc_calculator.calculate_ttc(
                vehicle_pos, vehicle_vel,
                obj['centroid'], obj['velocity']
            )
            
            if ttc_result:
                collision_pred = {
                    'track_id': obj['track_id'],
                    'class_name': obj['class_name'],
                    'ttc': ttc_result['ttc'],
                    'closest_distance': ttc_result['closest_distance'],
                    'collision_point': ttc_result['collision_point'],
                    'severity': ttc_result['severity'],
                    'relative_speed': ttc_result['relative_speed'],
                    'object_position': obj['centroid'],
                    'object_velocity': obj['velocity']
                }
                collision_predictions.append(collision_pred)
        
        # Sort by TTC (most urgent first)
        collision_predictions.sort(key=lambda x: x['ttc'])
        
        # Store in history for trend analysis
        self._update_collision_history(collision_predictions)
        
        return collision_predictions
    
    def _update_collision_history(self, predictions):
        """Store collision predictions for trend analysis"""
        import time
        timestamp = time.time()
        
        self.collision_history[timestamp] = predictions
        
        # Remove old history
        if len(self.collision_history) > self.max_history_size:
            oldest_key = min(self.collision_history.keys())
            del self.collision_history[oldest_key]
    
    def get_most_urgent_collision(self, predictions):
        """Get the most urgent collision from predictions"""
        if not predictions:
            return None
        
        # Already sorted by TTC in detect_collisions
        return predictions[0]
    
    def filter_by_severity(self, predictions, min_severity='caution'):
        """
        Filter predictions by minimum severity
        Args:
            predictions: list of collision predictions
            min_severity: 'critical', 'warning', or 'caution'
        """
        severity_levels = {'caution': 0, 'warning': 1, 'critical': 2}
        min_level = severity_levels.get(min_severity, 0)
        
        return [pred for pred in predictions 
                if severity_levels.get(pred['severity'], 0) >= min_level]
    
    def get_collision_stats(self):
        """Get statistics about recent collision detections"""
        if not self.collision_history:
            return {'total_predictions': 0}
        
        recent_predictions = list(self.collision_history.values())[-5:]  # Last 5 frames
        all_preds = [pred for frame_preds in recent_predictions for pred in frame_preds]
        
        if not all_preds:
            return {'total_predictions': 0}
        
        severities = [pred['severity'] for pred in all_preds]
        ttcs = [pred['ttc'] for pred in all_preds]
        
        return {
            'total_predictions': len(all_preds),
            'critical_count': severities.count('critical'),
            'warning_count': severities.count('warning'),
            'caution_count': severities.count('caution'),
            'min_ttc': min(ttcs) if ttcs else float('inf'),
            'avg_ttc': sum(ttcs) / len(ttcs) if ttcs else 0
        }