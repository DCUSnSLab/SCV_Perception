#!/usr/bin/env python3

import numpy as np
from collections import deque, defaultdict
import time


class ObjectRecord:
    """Single object detection record"""
    def __init__(self, track_id, class_name, centroid, confidence, timestamp, velocity=None, speed=0.0):
        self.track_id = track_id
        self.class_name = class_name
        self.centroid = np.array(centroid)
        self.confidence = confidence
        self.timestamp = timestamp
        self.velocity = velocity
        self.speed = speed
        self.frame_time = time.time()


class ObjectHistory:
    """Stores and manages history of all detected objects"""
    
    def __init__(self, max_history_size=1000, max_age_seconds=300):
        self.max_history_size = max_history_size
        self.max_age_seconds = max_age_seconds
        
        # Store all object records
        self.all_records = deque(maxlen=max_history_size)
        
        # Track records by ID for easy access
        self.records_by_id = defaultdict(list)
        
        # Statistics
        self.total_objects_seen = 0
        self.unique_tracks = set()
    
    def add_objects(self, tracked_objects, timestamp):
        """Add a batch of tracked objects to history"""
        timestamp_sec = timestamp.sec + timestamp.nanosec * 1e-9
        
        for obj in tracked_objects:
            record = ObjectRecord(
                track_id=obj['track_id'],
                class_name=obj['class_name'], 
                centroid=obj['centroid'],
                confidence=obj['confidence'],
                timestamp=timestamp_sec,
                velocity=obj.get('velocity'),
                speed=obj.get('speed', 0.0)
            )
            
            # Add to main history
            self.all_records.append(record)
            
            # Add to ID-specific history
            self.records_by_id[obj['track_id']].append(record)
            
            # Update statistics
            self.total_objects_seen += 1
            self.unique_tracks.add(obj['track_id'])
        
        # Clean old records periodically
        if len(self.all_records) % 50 == 0:  # Every 50 records
            self._cleanup_old_records()
    
    def _cleanup_old_records(self):
        """Remove records older than max_age_seconds"""
        current_time = time.time()
        cutoff_time = current_time - self.max_age_seconds
        
        # Clean main records
        while (self.all_records and 
               self.all_records[0].frame_time < cutoff_time):
            old_record = self.all_records.popleft()
            
            # Remove from ID-specific records
            if old_record.track_id in self.records_by_id:
                id_records = self.records_by_id[old_record.track_id]
                # Remove records older than cutoff
                self.records_by_id[old_record.track_id] = [
                    r for r in id_records if r.frame_time >= cutoff_time
                ]
                
                # Remove empty ID entries
                if not self.records_by_id[old_record.track_id]:
                    del self.records_by_id[old_record.track_id]
    
    def get_all_positions(self):
        """Get all recorded positions as (x, y, z) points"""
        return [record.centroid for record in self.all_records]
    
    def get_positions_by_class(self, class_name):
        """Get positions for specific class"""
        return [record.centroid for record in self.all_records 
                if record.class_name.lower() == class_name.lower()]
    
    def get_trajectory_by_id(self, track_id, min_points=2):
        """Get trajectory for specific track ID"""
        if track_id not in self.records_by_id:
            return []
        
        records = self.records_by_id[track_id]
        if len(records) < min_points:
            return []
        
        # Sort by timestamp and return positions
        sorted_records = sorted(records, key=lambda r: r.timestamp)
        return [record.centroid for record in sorted_records]
    
    def get_active_trajectories(self, min_points=3, max_age_seconds=60):
        """Get all trajectories that are recent and have enough points"""
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        
        active_trajectories = {}
        
        for track_id, records in self.records_by_id.items():
            # Filter recent records
            recent_records = [r for r in records if r.frame_time >= cutoff_time]
            
            if len(recent_records) >= min_points:
                # Sort by timestamp
                sorted_records = sorted(recent_records, key=lambda r: r.timestamp)
                trajectory = [record.centroid for record in sorted_records]
                
                active_trajectories[track_id] = {
                    'trajectory': trajectory,
                    'class_name': sorted_records[-1].class_name,
                    'last_seen': sorted_records[-1].frame_time,
                    'duration': sorted_records[-1].timestamp - sorted_records[0].timestamp
                }
        
        return active_trajectories
    
    def get_class_statistics(self):
        """Get statistics by class"""
        class_counts = defaultdict(int)
        class_positions = defaultdict(list)
        
        for record in self.all_records:
            class_counts[record.class_name] += 1
            class_positions[record.class_name].append(record.centroid)
        
        return dict(class_counts), dict(class_positions)
    
    def get_spatial_bounds(self):
        """Get min/max bounds of all recorded positions"""
        if not self.all_records:
            return None
        
        positions = np.array([record.centroid for record in self.all_records])
        
        return {
            'min': np.min(positions, axis=0),
            'max': np.max(positions, axis=0),
            'center': np.mean(positions, axis=0),
            'range': np.max(positions, axis=0) - np.min(positions, axis=0)
        }
    
    def get_summary_stats(self):
        """Get summary statistics"""
        current_time = time.time()
        
        # Recent activity (last 60 seconds)
        recent_records = [r for r in self.all_records 
                         if current_time - r.frame_time <= 60]
        
        recent_ids = set(r.track_id for r in recent_records)
        
        return {
            'total_detections': len(self.all_records),
            'total_unique_tracks': len(self.unique_tracks),
            'recent_detections': len(recent_records),
            'recent_active_tracks': len(recent_ids),
            'oldest_record_age': (current_time - self.all_records[0].frame_time) if self.all_records else 0,
            'newest_record_age': (current_time - self.all_records[-1].frame_time) if self.all_records else 0
        }