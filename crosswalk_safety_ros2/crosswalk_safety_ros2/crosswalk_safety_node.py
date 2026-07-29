#!/usr/bin/env python3
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import rclpy
from perception_interface.msg import DetectionArray
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


@dataclass
class TrackState:
    track_id: int
    class_name: str
    center_x: float
    center_y: float
    distance_m: float
    stamp_sec: float
    speed_mps: float = 0.0
    ttc_sec: float = math.inf
    decrease_count: int = 0
    last_seen_sec: float = field(default=0.0)


class CrosswalkSafetyNode(Node):
    def __init__(self) -> None:
        super().__init__('crosswalk_safety_node')
        self.next_track_id = 0
        self.tracks_by_side: Dict[str, Dict[int, TrackState]] = {'left': {}, 'right': {}}

        self.declare_parameter('left_detections_topic', '/perception/left/detections',
                               ParameterDescriptor(description='Left camera detections topic.'))
        self.declare_parameter('right_detections_topic', '/perception/right/detections',
                               ParameterDescriptor(description='Right camera detections topic.'))
        self.declare_parameter('safety_state_topic', '/crosswalk/safety_state',
                               ParameterDescriptor(description='Safety state string topic.'))
        self.declare_parameter('unsafe_topic', '/crosswalk/unsafe',
                               ParameterDescriptor(description='Unsafe boolean topic.'))
        self.declare_parameter('min_ttc_topic', '/crosswalk/min_ttc',
                               ParameterDescriptor(description='Minimum TTC topic.'))
        self.declare_parameter('vehicle_classes', ['car', 'bus', 'truck', 'motorcycle'],
                               ParameterDescriptor(description='Classes treated as road threats.'))
        self.declare_parameter('max_match_distance_px', 120.0,
                               ParameterDescriptor(description='Max image-space association distance.'))
        self.declare_parameter('track_timeout_sec', 0.75,
                               ParameterDescriptor(description='How long tracks survive without updates.'))
        self.declare_parameter('hazard_distance_m', 12.0,
                               ParameterDescriptor(description='Distance gate for potential hazards.'))
        self.declare_parameter('unsafe_distance_m', 8.0,
                               ParameterDescriptor(description='Immediate unsafe distance threshold.'))
        self.declare_parameter('min_closing_speed_mps', 0.8,
                               ParameterDescriptor(description='Minimum closing speed to treat as approaching.'))
        self.declare_parameter('unsafe_ttc_sec', 4.0,
                               ParameterDescriptor(description='Unsafe TTC threshold.'))
        self.declare_parameter('caution_ttc_sec', 7.0,
                               ParameterDescriptor(description='Caution TTC threshold.'))
        self.declare_parameter('required_decrease_count', 2,
                               ParameterDescriptor(description='How many consecutive distance drops imply approach.'))

        self.vehicle_classes = set(self.get_parameter('vehicle_classes').value)
        self.max_match_distance_px = float(self.get_parameter('max_match_distance_px').value)
        self.track_timeout_sec = float(self.get_parameter('track_timeout_sec').value)
        self.hazard_distance_m = float(self.get_parameter('hazard_distance_m').value)
        self.unsafe_distance_m = float(self.get_parameter('unsafe_distance_m').value)
        self.min_closing_speed_mps = float(self.get_parameter('min_closing_speed_mps').value)
        self.unsafe_ttc_sec = float(self.get_parameter('unsafe_ttc_sec').value)
        self.caution_ttc_sec = float(self.get_parameter('caution_ttc_sec').value)
        self.required_decrease_count = int(self.get_parameter('required_decrease_count').value)

        left_topic = self.get_parameter('left_detections_topic').value
        right_topic = self.get_parameter('right_detections_topic').value
        self.create_subscription(DetectionArray, left_topic, self.left_callback, 10)
        self.create_subscription(DetectionArray, right_topic, self.right_callback, 10)

        self.state_pub = self.create_publisher(String, self.get_parameter('safety_state_topic').value, 10)
        self.unsafe_pub = self.create_publisher(Bool, self.get_parameter('unsafe_topic').value, 10)
        self.min_ttc_pub = self.create_publisher(Float32, self.get_parameter('min_ttc_topic').value, 10)
        self.create_timer(0.2, self.publish_state)

        self.get_logger().info('Crosswalk safety node started.')

    def left_callback(self, msg: DetectionArray) -> None:
        self._update_side('left', msg)

    def right_callback(self, msg: DetectionArray) -> None:
        self._update_side('right', msg)

    def _update_side(self, side: str, msg: DetectionArray) -> None:
        now_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        existing = self.tracks_by_side[side]
        assigned_ids = set()
        fresh_tracks: Dict[int, TrackState] = {}

        for detection in msg.detections:
            if detection.class_name not in self.vehicle_classes:
                continue
            if math.isnan(detection.centroid.z) or detection.centroid.z <= 0.0:
                continue

            center_x = float(detection.centroid.x)
            center_y = float(detection.centroid.y)
            distance_m = float(detection.centroid.z)

            matched_id = self._match_track(existing, assigned_ids, detection.class_name, center_x, center_y)
            if matched_id is None:
                track = TrackState(
                    track_id=self.next_track_id,
                    class_name=detection.class_name,
                    center_x=center_x,
                    center_y=center_y,
                    distance_m=distance_m,
                    stamp_sec=now_sec,
                    last_seen_sec=now_sec,
                )
                self.next_track_id += 1
            else:
                prev = existing[matched_id]
                track = self._updated_track(prev, center_x, center_y, distance_m, now_sec)

            fresh_tracks[track.track_id] = track
            assigned_ids.add(track.track_id)

        for track_id, track in existing.items():
            if track_id in fresh_tracks:
                continue
            if now_sec - track.last_seen_sec <= self.track_timeout_sec:
                fresh_tracks[track_id] = track

        self.tracks_by_side[side] = fresh_tracks

    def _match_track(
        self,
        existing: Dict[int, TrackState],
        assigned_ids: set,
        class_name: str,
        center_x: float,
        center_y: float,
    ) -> Optional[int]:
        best_track_id = None
        best_distance = self.max_match_distance_px
        for track_id, track in existing.items():
            if track_id in assigned_ids or track.class_name != class_name:
                continue
            image_distance = math.hypot(center_x - track.center_x, center_y - track.center_y)
            if image_distance < best_distance:
                best_distance = image_distance
                best_track_id = track_id
        return best_track_id

    def _updated_track(
        self,
        prev: TrackState,
        center_x: float,
        center_y: float,
        distance_m: float,
        now_sec: float,
    ) -> TrackState:
        dt = max(1e-3, now_sec - prev.stamp_sec)
        closing_speed = (prev.distance_m - distance_m) / dt
        decrease_count = prev.decrease_count + 1 if distance_m < prev.distance_m else 0
        ttc_sec = distance_m / closing_speed if closing_speed > 1e-3 else math.inf
        return TrackState(
            track_id=prev.track_id,
            class_name=prev.class_name,
            center_x=center_x,
            center_y=center_y,
            distance_m=distance_m,
            stamp_sec=now_sec,
            speed_mps=closing_speed,
            ttc_sec=ttc_sec,
            decrease_count=decrease_count,
            last_seen_sec=now_sec,
        )

    def publish_state(self) -> None:
        active_tracks = self._active_tracks()
        state = 'SAFE'
        min_ttc = math.inf

        for side, track in active_tracks:
            if not self._is_approaching(track):
                continue
            min_ttc = min(min_ttc, track.ttc_sec)
            if track.distance_m <= self.unsafe_distance_m or track.ttc_sec <= self.unsafe_ttc_sec:
                state = 'UNSAFE'
                break
            if track.distance_m <= self.hazard_distance_m or track.ttc_sec <= self.caution_ttc_sec:
                state = 'CAUTION'

        state_msg = String()
        state_msg.data = state
        self.state_pub.publish(state_msg)

        unsafe_msg = Bool()
        unsafe_msg.data = state == 'UNSAFE'
        self.unsafe_pub.publish(unsafe_msg)

        ttc_msg = Float32()
        ttc_msg.data = float(min_ttc if math.isfinite(min_ttc) else -1.0)
        self.min_ttc_pub.publish(ttc_msg)

    def _active_tracks(self) -> List[Tuple[str, TrackState]]:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        active: List[Tuple[str, TrackState]] = []
        for side, tracks in self.tracks_by_side.items():
            for track in tracks.values():
                if now_sec - track.last_seen_sec <= self.track_timeout_sec:
                    active.append((side, track))
        return active

    def _is_approaching(self, track: TrackState) -> bool:
        if track.distance_m > self.hazard_distance_m:
            return False
        if track.speed_mps >= self.min_closing_speed_mps:
            return True
        return track.decrease_count >= self.required_decrease_count


def main(args: Iterable[str] = None) -> None:
    rclpy.init(args=args)
    node = CrosswalkSafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
