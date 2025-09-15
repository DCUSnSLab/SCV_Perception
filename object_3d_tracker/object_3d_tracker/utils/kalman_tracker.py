#!/usr/bin/env python3

import numpy as np
import time


class KalmanTracker:
    """
    3D Kalman Filter for object position and velocity tracking
    State vector: [x, y, z, vx, vy, vz] (6D)
    """
    
    def __init__(self, initial_position, process_noise=0.1, measurement_noise=0.5):
        """
        Initialize Kalman filter
        Args:
            initial_position: (x, y, z) initial position
            process_noise: process noise covariance
            measurement_noise: measurement noise covariance
        """
        # State vector [x, y, z, vx, vy, vz]
        self.state = np.array([
            initial_position[0], initial_position[1], initial_position[2],  # position
            0.0, 0.0, 0.0  # velocity (initially zero)
        ], dtype=np.float64)
        
        # State covariance matrix (6x6)
        self.P = np.eye(6, dtype=np.float64)
        self.P[:3, :3] *= 1.0  # position uncertainty
        self.P[3:, 3:] *= 10.0  # velocity uncertainty (higher initially)
        
        # Process noise covariance (6x6)
        self.Q = np.eye(6, dtype=np.float64) * process_noise
        self.Q[3:, 3:] *= 0.1  # lower process noise for velocity
        
        # Measurement noise covariance (3x3, only position is measured)
        self.R = np.eye(3, dtype=np.float64) * measurement_noise
        
        # Measurement matrix (3x6) - we only measure position
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[:3, :3] = np.eye(3)  # [1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0]
        
        self.last_update_time = time.time()
        self.track_id = None
        self.update_count = 0
        self.is_initialized = False
    
    def predict(self, dt):
        if dt <= 0:
            return
        F = np.eye(6, dtype=np.float64)
        F[:3, 3:] = np.eye(3) * dt

        # CV 모델용 Q (가속도 분산 = self.q_acc)
        q = getattr(self, "q_acc", 0.5)  # 기본 가속 잡음 세기
        dt2, dt3, dt4 = dt*dt, dt*dt*dt, dt*dt*dt*dt
        Q_pos = (dt4/4.0) * q * np.eye(3)
        Q_cross = (dt3/2.0) * q * np.eye(3)
        Q_vel = (dt2) * q * np.eye(3)
        Q = np.block([[Q_pos,  Q_cross],
                    [Q_cross, Q_vel]])
        # 상태/공분산 예측
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement, measurement_noise=None, R_scale: float = 1.0, R_mat: np.ndarray = None):
        z = np.array(measurement, dtype=np.float64)

        if R_mat is not None:
            R = R_mat
        else:
            base_R = self.R if measurement_noise is None else np.eye(3) * float(measurement_noise)
            R = base_R * float(R_scale)

        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ self.H.T @ np.linalg.pinv(S)

        self.state = self.state + K @ y
        I_KH = np.eye(6) - K @ self.H
        # Joseph form 유지 OK
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        self.update_count += 1
        # 위치분산 기준으로 초기화 판정 (예: < 1.0m)
        pos_std = np.sqrt(np.diag(self.P[:3, :3])).max()
        self.is_initialized = (self.update_count >= 2 and pos_std < 1.0)
    
    def process_measurement(self, position, timestamp=None):
        """
        Process new measurement (predict + update)
        Args:
            position: (x, y, z) measured position
            timestamp: measurement timestamp (uses current time if None)
        """
        current_time = time.time() if timestamp is None else timestamp
        dt = current_time - self.last_update_time
        
        # Predict step
        if dt > 0 and self.update_count > 0:
            self.predict(dt)
        
        # Update step
        self.update(position)
        
        self.last_update_time = current_time
    
    def get_position(self):
        """Get current estimated position"""
        return self.state[:3].copy()
    
    def get_velocity(self):
        """Get current estimated velocity"""
        if not self.is_initialized:
            return None
        return self.state[3:].copy()
    
    def get_speed(self):
        """Get current estimated speed (velocity magnitude)"""
        velocity = self.get_velocity()
        if velocity is None:
            return 0.0
        return float(np.linalg.norm(velocity))
    
    def get_position_uncertainty(self):
        """Get position uncertainty (diagonal of position covariance)"""
        return np.sqrt(np.diag(self.P[:3, :3]))
    
    def get_velocity_uncertainty(self):
        """Get velocity uncertainty (diagonal of velocity covariance)"""
        return np.sqrt(np.diag(self.P[3:, 3:]))
    
    def is_stable(self, max_position_uncertainty=1.0, max_velocity_uncertainty=2.0):
        """
        Check if tracking is stable based on uncertainty
        """
        if not self.is_initialized:
            return False
            
        pos_uncertainty = np.max(self.get_position_uncertainty())
        vel_uncertainty = np.max(self.get_velocity_uncertainty())
        
        return (pos_uncertainty < max_position_uncertainty and 
                vel_uncertainty < max_velocity_uncertainty)
    
    def should_reject_measurement(self, measurement, chi2_thresh: float = 7.815):
        if self.update_count < 2:
            return False
        z = np.array(measurement, dtype=np.float64)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        try:
            d2 = float(y.T @ np.linalg.inv(S) @ y)
            return d2 > chi2_thresh  # dof=3, p≈0.05
        except np.linalg.LinAlgError:
            return False


class KalmanVelocityTracker:
    """
    Manages Kalman filters for multiple objects
    """
    
    def __init__(self, process_noise=0.1, measurement_noise=0.5, cleanup_timeout=5.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.cleanup_timeout = cleanup_timeout
        
        # Dictionary to store Kalman trackers for each object
        self.trackers = {}  # {track_id: KalmanTracker}
    
    def update_position(self, track_id, position, timestamp):
        """
        Update position for tracked object
        """
        # Use system time for consistency
        timestamp_sec = time.time()
        
        # Create new tracker if needed
        if track_id not in self.trackers:
            self.trackers[track_id] = KalmanTracker(
                position, self.process_noise, self.measurement_noise
            )
            self.trackers[track_id].track_id = track_id
        
        tracker = self.trackers[track_id]
        
        # Check if measurement should be rejected
        if tracker.should_reject_measurement(position):
            print(f"DEBUG: Rejecting outlier measurement for track {track_id}")
            return
        
        # Process measurement
        tracker.process_measurement(position, timestamp_sec)
    
    def get_position(self, track_id):
        """Get filtered position for specific track ID"""
        if track_id not in self.trackers:
            return None
        return self.trackers[track_id].get_position()
    
    def get_velocity(self, track_id):
        """Get velocity for specific track ID"""
        if track_id not in self.trackers:
            return None
        return self.trackers[track_id].get_velocity()
    
    def get_speed(self, track_id):
        """Get speed for specific track ID"""
        if track_id not in self.trackers:
            return 0.0
        return self.trackers[track_id].get_speed()
    
    def is_stable(self, track_id):
        """Check if tracking is stable for specific track ID"""
        if track_id not in self.trackers:
            return False
        return self.trackers[track_id].is_stable()
    
    def cleanup_old_tracks(self, active_track_ids, current_time=None):
        """Remove tracking for inactive objects"""
        if current_time is None:
            current_time = time.time()
        
        tracks_to_remove = []
        for track_id, tracker in self.trackers.items():
            if (track_id not in active_track_ids and 
                current_time - tracker.last_update_time > self.cleanup_timeout):
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.trackers[track_id]
        
        return len(tracks_to_remove)
    
    def get_tracking_stats(self):
        """Get statistics about current tracking"""
        stats = {
            'total_tracks': len(self.trackers),
            'stable_tracks': 0,
            'moving_tracks': 0,
            'initialized_tracks': 0
        }
        
        for tracker in self.trackers.values():
            if tracker.is_initialized:
                stats['initialized_tracks'] += 1
            if tracker.is_stable():
                stats['stable_tracks'] += 1
            if tracker.get_speed() > 0.05:  # 5cm/s threshold
                stats['moving_tracks'] += 1
        
        return stats