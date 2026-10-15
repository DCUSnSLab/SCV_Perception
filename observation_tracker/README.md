# observation_tracker

`observation_tracker` is a standalone ROS 2 Humble 3-D tracker with object-wise
VLP-32C/D555 observation states. It contains its own Kalman filter, track ID,
ego-motion compensation, message parsing and output implementation. The
observation policy is intentionally deterministic and lightweight for ablation.

## Pipeline

```text
3-D detector output + predicted tracks
  -> object observation region
  -> raw LiDAR point count + centroid consistency
  -> D555 projection + 2-D detection match + aligned-depth consistency
  -> MULTIMODAL_STRONG / LIDAR_DOMINANT / CAMERA_DOMINANT / WEAK_OBSERVATION
  -> unified uncertainty-based Hungarian association
  -> observation-aware lifecycle and temporal memory
  -> existing DetectedObjectArray + state markers + CSV
```

Active, lost and occluded tracks share one association pass. Its only spatial
hard gate is the full Kalman innovation covariance (`H P H^T + R`); 3-D IoU,
camera overlap and depth consistency remain soft costs. Thus the admissible
region grows with prediction uncertainty instead of using a fixed distance.
Camera-depth evidence may retain an unmatched existing 3-D track, but never
initializes a camera-only track.

Depth does not move the calibrated projection. The analyzer samples the center
of each projected ROI, counts pixels within the configured tolerance of the
predicted camera-frame depth, and uses that support ratio for consistency and
association cost. A closer median depth is treated as occlusion.

## Architecture

- `pointpillars_coda_detector` and `observation_tracker` run as separate nodes.
- They communicate through `tracking_msgs/DetectedObjectArray`; the detector is
  not imported or launched by the tracker.
- local `TimeAwareTrack`: Kalman filter, ID and motion history
- local `StaticCvTrack`: evidence-weighted stationary/CV prediction mixture
- local odometry/IMU interpolation and body-frame ego-motion compensation
- current `tracking_msgs/DetectedObjectArray` parsing and publication
- TF lookup, PointCloud2 parsing, sensor caches and RViz baseline markers
- SciPy Hungarian assignment

`tracking_msgs` remains a shared interface package, not a tracker implementation
dependency. A different detector can be connected by publishing the same input
message type and topic.

## Topics

| Input/output | Default topic | Type |
|---|---|---|
| 3-D detection | `/detected_objects_3d` | `tracking_msgs/DetectedObjectArray` |
| raw LiDAR | `/velodyne_points` | `sensor_msgs/PointCloud2` |
| D555 RGB | `/camera/camera/color/image_raw` | `sensor_msgs/Image` |
| 2-D detection | `/yolo/detections` | `perception_interface/DetectionArray` |
| CameraInfo | `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` |
| aligned depth | `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` |
| wheel odometry | `/odom` | `nav_msgs/Odometry` |
| IMU | `/vectornav/imu` | `sensor_msgs/Imu` |
| tracks | `/tracked_objects_3d` | `tracking_msgs/DetectedObjectArray` |
| state markers | `/observation_tracker/debug_markers` | `visualization_msgs/MarkerArray` |
| unified 2-D/3-D markers | `/observation_tracker/fused_detection_markers` | `visualization_msgs/MarkerArray` |

For the recorded traversability bag use `odom_topic:=/odom use_sim_time:=true`.

## Build and run

```bash
cd /home/jay/SCV_Perception
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --packages-up-to observation_tracker
source install/setup.bash
ros2 launch observation_tracker observation_tracker.launch.py
```

RViz debug markers are disabled by default to avoid adding marker construction
to tracking latency. Enable them only when inspecting a run:

```bash
ros2 launch observation_tracker observation_tracker.launch.py \
  enable_debug_visualization:=true \
  enable_fused_detection_visualization:=true
```

Only the 3-D detection topic is required. Optional sensor inputs can be
disabled without creating their subscriptions. For example, LiDAR-only
tracking without camera, depth, odometry or IMU is:

```bash
ros2 launch observation_tracker observation_tracker.launch.py \
  enable_camera_observation:=false \
  enable_depth_consistency:=false \
  enable_odometry:=false \
  enable_imu:=false
```

Disable every observation input to run the motion/detection baseline:

```bash
ros2 launch observation_tracker observation_tracker.launch.py \
  enable_lidar_observation:=false \
  enable_camera_observation:=false \
  enable_depth_consistency:=false \
  enable_odometry:=false \
  enable_imu:=false
```

All input topics can be overridden from the same launch command, for example
`lidar_points_topic:=/no_ground_points` or
`camera_detection_topic:=/my/detections_2d`. Run
`ros2 launch observation_tracker observation_tracker.launch.py --show-args`
for the complete argument list. Depth is automatically disabled when camera
observation is disabled.

## Unified RViz detection view

Add one `MarkerArray` display in RViz and select
`/observation_tracker/fused_detection_markers`. It contains each raw 3-D
detection box plus its projected 2-D match information:

- green: projected 3-D ROI matched a 2-D detection
- orange: camera was available but no 2-D detection passed the match gates
- blue: camera evidence was unavailable

The text shows both detector classes/confidences and Camera IoU. A 3-D
detection associated with a track is prefixed with its stable `ID n`.
Unassociated raw detections do not display a temporary detection number. An
unmatched 2-D detection with valid aligned depth is also back-projected into
the 3-D grid as a magenta `CAMERA_DEPTH_CANDIDATE` box. Its visible width and
height come from the 2-D bounds and median aligned depth. Camera-only
candidates never initialize a new track ID. The fused topic removes a 3-D box
as soon as it no longer has a current detector match; the internal track and
its ID may still be retained for later recovery.

A weak 3-D detection participates in the same covariance-aware assignment but
does not count as a normal Kalman-filter hit or confirm a tentative track.
There is no pedestrian-only recovery gate: all classes reconnect through the
same uncertainty policy even when their predicted and detected boxes do not
overlap.

When the 3-D detector misses but LiDAR box support or class-compatible camera
support with non-conflicting depth still confirms the predicted object, the
track enters `SUPPORTED_PREDICTION`. This resets the detector-miss streak
without counting a detector hit, so `max_miss_count` only terminates tracks
that also lose independent sensor evidence.

For pedestrian crossings, a missed rear track becomes `OCCLUDED` when its
depth ROI is blocked and a nearer supported pedestrian covers at least 45% of
its projected ROI for two frames. The hidden ID is retained for up to eight
frames and participates in the same global assignment before an unmatched
detection can create a replacement ID. The occluding track ID and overlap
ratio are logged.

The 2-D detector defaults to the D555 compressed color image
(`/camera/camera/color/image_raw/compressed`) and publishes raw detector
confidence without its internal 2-D tracker. Its default COCO class filter is
`person(0), car(2)`:

```bash
ros2 run ultralytics_ros2 seg_run
```

## Parameters and ablation

All thresholds and enable flags are in `config/observation_tracker.yaml`.
Important flags are:

- `enable_lidar_observation`
- `enable_camera_observation`
- `enable_depth_consistency`
- `enable_observation_state`
- `enable_adaptive_lifecycle`
- `enable_temporal_smoothing`
- `enable_occlusion_reasoning`

LiDAR strong thresholds are class- and distance-specific. Their initial values
are prototype values, not validated VLP-32C operating thresholds. CSV output
contains raw point counts, density, centroid error, sensor states, association
stage/cost and lifecycle state so validation sequences can set them empirically.
Depth tuning uses `depth_roi_inner_scale`, `min_depth_support_ratio`,
`min_valid_depth_pixels`, `depth_consistency_threshold`, and
`occlusion_depth_threshold`. The CSV includes valid/support pixel counts and
the support ratio.

The analyzer defaults to `/velodyne_points`. This preserves returns below and
around objects, but ground points can inflate counts for low boxes. Set
`lidar_points_topic: /no_ground_points` for an ablation using the existing
ground filter and re-tune class/distance thresholds for that input.

## Output compatibility

`/tracked_objects_3d` intentionally retains the existing message fields:
track ID, class, score, center, dimensions and yaw. Observation state and
lifecycle status are published through RViz text/colors and the experiment CSV
instead of changing downstream message definitions.

The implementation has no learned fusion, appearance Re-ID or camera-only 3-D
initialization. Those are outside the research core.
