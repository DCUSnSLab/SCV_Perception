# observation_tracker

`observation_tracker` extends the existing ROS 2 Humble Jay 3-D tracker with
object-wise VLP-32C/D555 observation states. It keeps the existing Kalman
filter, track IDs, ego-motion compensation, message parsing and output type.
The added policy is intentionally deterministic and lightweight for ablation.

## Pipeline

```text
3-D detections + predicted Jay tracks
  -> object observation region
  -> raw LiDAR point count + centroid consistency
  -> D555 projection + 2-D detection match + aligned-depth consistency
  -> MULTIMODAL_STRONG / LIDAR_DOMINANT / CAMERA_DOMINANT / WEAK_OBSERVATION
  -> cascaded association stages 1/2/3/4
  -> observation-aware lifecycle and temporal memory
  -> existing DetectedObjectArray + state markers + CSV
```

Stage 1 associates multimodal candidates. Stage 2 uses only 3-D LiDAR/motion
terms. Stage 3 keeps an unmatched existing 3-D track when camera and depth
support its prediction. It never initializes a 3-D track from camera-only
evidence. Stage 4 predicts without forced association and delegates deletion
to the lifecycle history.

Depth does not move the calibrated projection. The analyzer samples the center
of each projected ROI, counts pixels within the configured tolerance of the
predicted camera-frame depth, and uses that support ratio for consistency and
association cost. A closer median depth is treated as occlusion.

## Reused implementation

- `pcdet_tracker.jay_tracker.TimeAwareTrack`: KF, ID and motion prediction
- Jay odometry/IMU interpolation and body-frame ego-motion compensation
- current `tracking_msgs/DetectedObjectArray` parsing and publication
- TF lookup, PointCloud2 parsing, sensor caches and RViz baseline markers
- SciPy Hungarian assignment already used by Jay Tracker

The existing `pcdet_tracker` package is not modified. Run it separately for an
A0 baseline or run this package for the adaptive path.

## Topics

| Input/output | Default topic | Type |
|---|---|---|
| 3-D detection | `/detected_objects_3d` | `tracking_msgs/DetectedObjectArray` |
| raw LiDAR | `/velodyne_points` | `sensor_msgs/PointCloud2` |
| D555 RGB | `/camera/camera/color/image_raw` | `sensor_msgs/Image` |
| 2-D detection | `/yolo/detections` | `perception_interface/DetectionArray` |
| CameraInfo | `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` |
| aligned depth | `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` |
| wheel odometry | `/odometry/wheel` | `nav_msgs/Odometry` |
| IMU | `/vectornav/imu` | `sensor_msgs/Imu` |
| tracks | `/tracked_objects_3d` | `tracking_msgs/DetectedObjectArray` |
| state markers | `/observation_tracker/debug_markers` | `visualization_msgs/MarkerArray` |
| unified 2-D/3-D markers | `/observation_tracker/fused_detection_markers` | `visualization_msgs/MarkerArray` |

For the recorded traversability bag use `odom_topic:=/odom use_sim_time:=true`.

## Build and run

```bash
cd /home/scv/SCV
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select observation_tracker
source install/setup.bash
ros2 launch observation_tracker observation_tracker.launch.py
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
the 3-D grid as a magenta `CAMERA_DEPTH_CANDIDATE` sphere. Camera-only
candidates never initialize a new track ID. The fused topic removes a 3-D box
as soon as it no longer has a current detector match; the internal track and
its ID may still be retained for later recovery.

A weak 3-D detection can retain an existing confirmed/recovering ID when its
class, predicted center, and 3-D IoU pass the stricter weak-continuity gates.
This association does not count as a Kalman-filter hit, increase confidence,
or create a new track, so the false-positive suppression policy remains active.

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
- `enable_cascaded_association`
- `enable_adaptive_lifecycle`
- `enable_temporal_smoothing`

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

The initial implementation has no learned fusion, Re-ID, camera-only 3-D
initialization or adaptive Kalman model. Those are outside the research core.
