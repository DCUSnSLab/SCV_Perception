#!/usr/bin/env python3

import numpy as np
from collections import deque
import time


class TrackingHistory:
    """Individual object tracking history"""
    def __init__(self, max_history_size=10):
        self.positions = deque(maxlen=max_history_size)  # (timestamp, x, y, z)
        self.velocities = deque(maxlen=max_history_size-1)  # (timestamp, vx, vy, vz, speed)
        self.track_id = None
        self.last_update = None
        self.creation_time = time.time()
    
    def add_position(self, position, timestamp):
        """Add new position measurement"""
        self.positions.append((timestamp, position[0], position[1], position[2]))
        self.last_update = timestamp
        
        # Calculate velocity if we have at least 2 positions
        if len(self.positions) >= 2:
            self._calculate_velocity()
    
    def _calculate_velocity(self):
        """Calculate velocity from recent positions"""
        if len(self.positions) < 2:
            print(f"DEBUG: Not enough positions: {len(self.positions)}")
            return
        
        # Get two most recent positions
        pos2 = self.positions[-1]  # (timestamp, x, y, z)
        pos1 = self.positions[-2]
        
        # Calculate time difference
        dt = pos2[0] - pos1[0]
        print(f"DEBUG: dt = {dt} (pos1_time={pos1[0]:.3f}, pos2_time={pos2[0]:.3f})")
        
        if dt <= 0.001:  # Change threshold to handle very small time differences
            print(f"DEBUG: dt too small: {dt}")
            return
        
        # Calculate velocity components
        vx = (pos2[1] - pos1[1]) / dt
        vy = (pos2[2] - pos1[2]) / dt
        vz = (pos2[3] - pos1[3]) / dt
        
        # Calculate speed (magnitude)
        speed = np.sqrt(vx*vx + vy*vy + vz*vz)
        
        print(f"DEBUG: Calculated velocity - vx:{vx:.3f}, vy:{vy:.3f}, vz:{vz:.3f}, speed:{speed:.3f}")
        
        self.velocities.append((pos2[0], vx, vy, vz, speed))
    
    def get_current_velocity(self):
        """Get current velocity vector"""
        if not self.velocities:
            return None
        
        latest_vel = self.velocities[-1]
        return np.array([latest_vel[1], latest_vel[2], latest_vel[3]], dtype=np.float32)
    
    def get_smoothed_velocity(self, window_size=3):
        """Get velocity smoothed over recent measurements"""
        if len(self.velocities) < 1:
            return None
        
        # Use recent velocities for smoothing
        recent_count = min(window_size, len(self.velocities))
        recent_vels = list(self.velocities)[-recent_count:]
        
        # If only one measurement, return it directly
        if len(recent_vels) == 1:
            vel = recent_vels[0]
            return np.array([vel[1], vel[2], vel[3]], dtype=np.float32)
        
        # Calculate weighted average (more weight to recent measurements)
        total_weight = 0
        weighted_vx = 0
        weighted_vy = 0
        weighted_vz = 0
        
        for i, vel in enumerate(recent_vels):
            weight = i + 1  # Linear weighting
            weighted_vx += vel[1] * weight
            weighted_vy += vel[2] * weight
            weighted_vz += vel[3] * weight
            total_weight += weight
        
        if total_weight > 0:
            return np.array([
                weighted_vx / total_weight,
                weighted_vy / total_weight,
                weighted_vz / total_weight
            ], dtype=np.float32)
        
        return None
    
    def get_current_speed(self):
        """Get current speed (scalar)"""
        velocity = self.get_smoothed_velocity()
        if velocity is not None:
            return float(np.linalg.norm(velocity))
        return 0.0
    
    def is_stable(self, min_tracking_time=0.1):
        """Check if tracking is stable enough for velocity calculation"""
        if len(self.positions) < 2:  # Just need 2 positions for velocity
            return False
        
        # Much more lenient - just need 2 consecutive measurements
        return True


class VelocityTracker:
    """Tracks velocities of multiple objects"""
    
    def __init__(self, history_size=10, smoothing_window=3, min_velocity_threshold=0.05, 
                 velocity_outlier_threshold=10.0, cleanup_timeout=5.0):
        self.history_size = history_size
        self.smoothing_window = smoothing_window
        self.min_velocity_threshold = min_velocity_threshold
        self.velocity_outlier_threshold = velocity_outlier_threshold
        self.cleanup_timeout = cleanup_timeout
        
        # Dictionary to store tracking history for each object
        self.tracked_objects = {}  # {track_id: TrackingHistory}
    
    def update_position(self, track_id, position, timestamp):
        """Update position for tracked object"""
        # Use system time instead of ROS time for more reliable dt calculation
        timestamp_sec = time.time()
        print(f"DEBUG: Using system time: {timestamp_sec:.3f}")
        
        # Create new tracking history if needed
        if track_id not in self.tracked_objects:
            self.tracked_objects[track_id] = TrackingHistory(self.history_size)
            self.tracked_objects[track_id].track_id = track_id
        
        # Add position to history
        self.tracked_objects[track_id].add_position(position, timestamp_sec)
    
    def get_velocity(self, track_id):
        """Get velocity for specific track ID"""
        if track_id not in self.tracked_objects:
            return None
        
        history = self.tracked_objects[track_id]
        if not history.is_stable():
            return None
        
        velocity = history.get_smoothed_velocity(self.smoothing_window)
        
        # Filter out very small velocities (likely noise)
        if velocity is not None:
            speed = np.linalg.norm(velocity)
            if speed < self.min_velocity_threshold:
                return np.zeros(3, dtype=np.float32)
            
            # Filter out unrealistic velocities
            if speed > self.velocity_outlier_threshold:
                return None
        
        return velocity
    
    def get_speed(self, track_id):
        """Get speed (scalar) for specific track ID"""
        velocity = self.get_velocity(track_id)
        if velocity is not None:
            return float(np.linalg.norm(velocity))
        return 0.0
    
    def get_all_velocities(self):
        """Get velocities for all tracked objects"""
        result = {}
        for track_id in self.tracked_objects.keys():
            velocity = self.get_velocity(track_id)
            if velocity is not None:
                result[track_id] = {
                    'velocity': velocity,
                    'speed': float(np.linalg.norm(velocity))
                }
        return result
    
    def cleanup_old_tracks(self, active_track_ids, current_time=None):
        """Remove tracking history for inactive objects"""
        if current_time is None:
            current_time = time.time()
        
        # Find tracks to remove
        tracks_to_remove = []
        for track_id, history in self.tracked_objects.items():
            # Remove if not in active list and hasn't been updated recently
            if (track_id not in active_track_ids and 
                history.last_update is not None and
                current_time - history.last_update > self.cleanup_timeout):
                tracks_to_remove.append(track_id)
        
        # Remove old tracks
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
        
        return len(tracks_to_remove)
    
    def get_tracking_stats(self):
        """Get statistics about current tracking"""
        stats = {
            'total_tracks': len(self.tracked_objects),
            'stable_tracks': 0,
            'moving_tracks': 0
        }
        
        for history in self.tracked_objects.values():
            if history.is_stable():
                stats['stable_tracks'] += 1
                speed = history.get_current_speed()
                if speed > self.min_velocity_threshold:
                    stats['moving_tracks'] += 1
        
        return stats