# terrain_seg

ROS 2 Humble package that runs
`nvidia/segformer-b0-finetuned-cityscapes-1024-1024` on the RealSense D555
color image and lifts synchronized, color-aligned depth pixels into a semantic
point cloud.

The default `input_mode: cloud` matches the current SCV rosbag, which contains
compressed RGB and an unorganized D555 color point cloud but no aligned-depth
image or CameraInfo. `input_mode: rgbd` remains available for the live D555
topics documented by `bring_up`.

The default point cloud contains only Cityscapes train IDs `0=road` and
`1=sidewalk`. Cityscapes includes curb pixels in `sidewalk`; it does not expose
a separate curb train ID. Low-confidence, invalid-depth, and all other class
pixels are omitted from the default cloud.

## Outputs

| Topic | Type | Contents |
|---|---|---|
| `/terrain_seg/class_mask` | `sensor_msgs/Image` (`mono8`) | Cityscapes train ID per pixel |
| `/terrain_seg/color_mask` | `sensor_msgs/Image` (`bgr8`) | Cityscapes palette |
| `/terrain_seg/overlay` | `sensor_msgs/Image` (`bgr8`) | RGB/mask overlay |
| `/terrain_seg/semantic_points` | `sensor_msgs/PointCloud2` | `x,y,z,rgb,label,confidence` |
| `/terrain_seg/semantic_lidar_points` | `sensor_msgs/PointCloud2` | Camera labels projected onto synchronized Velodyne returns |
| `/terrain_seg/geometry_points` | `sensor_msgs/PointCloud2` | Depth-only ground/elevated/curb/obstacle labels |
| `/terrain_seg/fused_points` | `sensor_msgs/PointCloud2` | RGB semantics plus geometry overrides |
| `/terrain_seg/labels` | `std_msgs/String` | JSON ID-to-name map, transient local |
| `/terrain_seg/geometry_labels` | `std_msgs/String` | Geometry label map |
| `/terrain_seg/fused_labels` | `std_msgs/String` | Fused label map |

Both clouds are transformed to `velodyne` by default using the calibrated TF in
`hunter2_description/urdf/tf_mounts.xacro`. Set `target_frame: ''` to retain
the aligned-depth optical frame. `static_tf_only: true` is appropriate for the
fixed D555/Velodyne extrinsics and avoids old-TF warnings on looped bags; disable
it only when selecting a genuinely moving target frame.

## Install and build

The current SCV Python environment already provides PyTorch. Install the
Hugging Face runtime once (prefer a project virtual environment when used in
deployment):

```bash
cd /home/scv/SCV/src/perception/terrain_seg
python3 -m pip install -r requirements.txt

cd /home/scv/SCV
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select terrain_seg
source install/setup.bash
```

The first run downloads the model weights into the Hugging Face cache. A local
model directory can be assigned to `model_id` for offline deployment.

## Run

Start the D555 and the vehicle TF, then launch segmentation:

```bash
ros2 launch bring_up sensors_start.launch.py
ros2 launch terrain_seg terrain_seg.launch.py
```

View the results:

```bash
rqt_image_view /terrain_seg/overlay
ros2 topic echo /terrain_seg/labels --once
```

In RViz2, use `velodyne` as Fixed Frame and add PointCloud2 displays for
`/terrain_seg/semantic_points` (D555 RGB-D points) and
`/terrain_seg/semantic_lidar_points` (camera labels transferred to Velodyne).
Select the `label` or `rgb` channel as the color transformer when supported.

Velodyne points are accepted only when their timestamps are close to the RGB
frame, they project inside the camera image, pass the confidence/class filter,
and do not contradict the aligned depth surface. Camera-FOV blind spots remain
unlabeled rather than being guessed.

The depth-only baseline uses a 5 cm XY grid, robust ground-plane fitting, point
height, and neighboring height transitions. Geometry IDs are `0=unknown`,
`1=ground`, `2=elevated_ground`, `3=curb`, and `4=obstacle`. Fused IDs are
`0=unknown`, `1=road`, `2=sidewalk`, `3=curb`, and `4=obstacle`. Tune
`expected_ground_z` and the height thresholds from measured SCV data before
using the result as an annotation proposal.

To retain every Cityscapes class in the 3D cloud, set
`included_class_ids: []`. Adjust `min_confidence` and `point_stride` in
`config/terrain_seg.yaml` to trade density for label quality and speed.

One way to write the semantic topic to PCD, when `pcl_ros` is installed, is:

```bash
ros2 run pcl_ros pointcloud_to_pcd --ros-args \
  -r input:=/terrain_seg/semantic_points
```

PCD consumers must preserve the custom `label` (`uint8`) and `confidence`
(`float32`) fields.

## Offline D555-Velodyne calibration check

The recorded point clouds and `/tf_static` can be checked without replaying the
bag. The evaluator pairs cloud header timestamps, measures Velodyne-to-D555
nearest-point residuals in the common field of view, and compares the recorded
extrinsic against an intentionally uncalibrated control:

```bash
source /opt/ros/humble/setup.bash
python3 tools/evaluate_lidar_camera_calibration.py \
  /home/scv/rosbag2_2026_09_17-10_51_36 \
  --output-dir calibration_report
```

It writes a Markdown report, complete JSON metrics, and an alignment plot. This
is a general-scene consistency check rather than a metrology certificate; a
static surveyed plane or checkerboard capture is still required to establish
absolute translation/rotation uncertainty.
