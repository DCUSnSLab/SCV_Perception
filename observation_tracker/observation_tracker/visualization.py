"""Observation-state debug boxes and labels for RViz2."""

import math

from rclpy.duration import Duration
from visualization_msgs.msg import Marker, MarkerArray


DEFAULT_COLORS = {
    "MULTIMODAL_STRONG": (0.10, 0.95, 0.25),
    "LIDAR_DOMINANT": (0.10, 0.55, 1.00),
    "CAMERA_DOMINANT": (0.95, 0.75, 0.10),
    "WEAK_OBSERVATION": (0.95, 0.15, 0.15),
}


class ObservationVisualizer:
    def __init__(self, colors=None, alpha=0.55, lifetime=0.0):
        self.colors = colors or DEFAULT_COLORS
        self.alpha = float(alpha)
        self.lifetime = float(lifetime)
        self.previous_ids = set()
        self.previous_fused_detection_ids = set()
        self.previous_camera_depth_ids = set()

    def build(self, tracks, header):
        output = MarkerArray()
        active_ids = set()
        lifetime = Duration(seconds=self.lifetime).to_msg()
        for track in tracks:
            track_id = int(track["id"])
            active_ids.add(track_id)
            box = track["box"]
            state = track["observation_state"]
            color = self.colors.get(state, DEFAULT_COLORS["WEAK_OBSERVATION"])
            marker = Marker()
            marker.header = header
            marker.ns = "observation_tracks"
            marker.id = track_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(box[0])
            marker.pose.position.y = float(box[1])
            marker.pose.position.z = float(box[2])
            marker.pose.orientation.z = math.sin(float(box[6]) * 0.5)
            marker.pose.orientation.w = math.cos(float(box[6]) * 0.5)
            marker.scale.x = float(box[3])
            marker.scale.y = float(box[4])
            marker.scale.z = float(box[5])
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = self.alpha
            marker.lifetime = lifetime
            output.markers.append(marker)

            observation = track.get("observation")
            text = Marker()
            text.header = header
            text.ns = "observation_track_info"
            text.id = track_id + 100000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(box[0])
            text.pose.position.y = float(box[1])
            text.pose.position.z = float(box[2] + box[5] * 0.75)
            text.scale.z = 0.42
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.lifetime = lifetime
            if observation is None:
                text.text = (
                    f"ID {track_id} {state}\n{track['track_status']} "
                    f"conf={track['track_confidence']:.2f}")
            else:
                text.text = (
                    f"ID {track_id} {state}\n{track['track_status']} "
                    f"conf={track['track_confidence']:.2f}\n"
                    f"L={observation.lidar_state.value} "
                    f"pts={observation.lidar_point_count} "
                    f"C={'Y' if observation.camera_supported else 'N'} "
                    f"D={observation.depth_state.value} "
                    f"support={observation.depth_support_ratio:.2f}")
            output.markers.append(text)

        for track_id in self.previous_ids - active_ids:
            for namespace, marker_id in (
                    ("observation_tracks", track_id),
                    ("observation_track_info", track_id + 100000)):
                marker = Marker()
                marker.header = header
                marker.ns = namespace
                marker.id = marker_id
                marker.action = Marker.DELETE
                output.markers.append(marker)
        self.previous_ids = active_ids
        return output

    def build_fused_detections(
            self, detections, observations, class_names, header,
            camera_depth_candidates=(), tracks=()):
        """Show raw 3-D detections and their projected 2-D matches together."""
        output = MarkerArray()
        active_ids = set()
        lifetime = Duration(seconds=self.lifetime).to_msg()
        detection_track_ids = {
            int(track["matched_detection_id"]): int(track["id"])
            for track in tracks
            if int(track.get("matched_detection_id", -1)) >= 0
        }
        for detection_id, detection in enumerate(detections):
            active_ids.add(detection_id)
            observation = (
                observations[detection_id]
                if detection_id < len(observations) else None)
            if observation is not None and observation.camera_supported:
                color = (0.10, 0.95, 0.25)
            elif observation is not None and observation.camera_available:
                color = (1.00, 0.55, 0.05)
            else:
                color = (0.10, 0.55, 1.00)

            box = Marker()
            box.header = header
            box.ns = "fused_3d_detections"
            box.id = detection_id
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = float(detection[0])
            box.pose.position.y = float(detection[1])
            box.pose.position.z = float(detection[2])
            box.pose.orientation.z = math.sin(float(detection[6]) * 0.5)
            box.pose.orientation.w = math.cos(float(detection[6]) * 0.5)
            box.scale.x = float(detection[3])
            box.scale.y = float(detection[4])
            box.scale.z = float(detection[5])
            box.color.r, box.color.g, box.color.b = color
            box.color.a = self.alpha
            box.lifetime = lifetime
            output.markers.append(box)

            label = int(detection[7])
            class_name = (
                class_names[label - 1]
                if 0 < label <= len(class_names) else "Unknown")
            text = Marker()
            text.header = header
            text.ns = "fused_2d_3d_info"
            text.id = detection_id + 100000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(detection[0])
            text.pose.position.y = float(detection[1])
            text.pose.position.z = float(
                detection[2] + detection[5] * 0.75)
            text.scale.z = 0.42
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.lifetime = lifetime
            object_id = detection_track_ids.get(detection_id)
            prefix = f"ID {object_id} " if object_id is not None else ""
            text.text = (
                f"{prefix}3D {class_name} {float(detection[8]):.2f}")
            if observation is None or not observation.camera_available:
                text.text += "\n2D unavailable"
            elif observation.camera_supported:
                text.text += (
                    f"\n2D {observation.camera_detection_class} "
                    f"{observation.camera_detection_score:.2f} "
                    f"IoU={observation.camera_iou:.2f}")
            else:
                text.text += "\n2D no match"
            output.markers.append(text)

        camera_depth_ids = set()
        for candidate in camera_depth_candidates:
            candidate_id = int(candidate["id"])
            camera_depth_ids.add(candidate_id)
            position = candidate["position"]
            state = candidate["state"]
            color = self.colors.get(
                state, (0.95, 0.20, 0.95))

            size = candidate["size"]
            box = Marker()
            box.header = header
            box.ns = "camera_depth_candidates"
            box.id = candidate_id
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = float(position[0])
            box.pose.position.y = float(position[1])
            box.pose.position.z = float(position[2])
            box.pose.orientation.w = 1.0
            box.scale.x = float(size[0])
            box.scale.y = float(size[1])
            box.scale.z = float(size[2])
            box.color.r, box.color.g, box.color.b = color
            box.color.a = self.alpha
            box.lifetime = lifetime
            output.markers.append(box)

            text = Marker()
            text.header = header
            text.ns = "camera_depth_candidate_info"
            text.id = candidate_id + 200000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(position[0])
            text.pose.position.y = float(position[1])
            text.pose.position.z = float(position[2] + 0.6)
            text.scale.z = 0.36
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.lifetime = lifetime
            prefix = (
                f"CAMERA_DOMINANT ID {candidate['track_id']}"
                if state == "CAMERA_DOMINANT"
                else "CAMERA_DEPTH_CANDIDATE")
            text.text = (
                f"{prefix}\n{candidate['class_name']} "
                f"{candidate['score']:.2f} depth={candidate['depth']:.2f}m")
            output.markers.append(text)

        for detection_id in self.previous_fused_detection_ids - active_ids:
            for namespace, marker_id in (
                    ("fused_3d_detections", detection_id),
                    ("fused_2d_3d_info", detection_id + 100000)):
                marker = Marker()
                marker.header = header
                marker.ns = namespace
                marker.id = marker_id
                marker.action = Marker.DELETE
                output.markers.append(marker)
        self.previous_fused_detection_ids = active_ids
        for candidate_id in self.previous_camera_depth_ids - camera_depth_ids:
            for namespace, marker_id in (
                    ("camera_depth_candidates", candidate_id),
                    ("camera_depth_candidate_info", candidate_id + 200000)):
                marker = Marker()
                marker.header = header
                marker.ns = namespace
                marker.id = marker_id
                marker.action = Marker.DELETE
                output.markers.append(marker)
        self.previous_camera_depth_ids = camera_depth_ids

        return output
