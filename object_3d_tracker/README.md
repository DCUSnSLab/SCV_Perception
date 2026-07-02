# Object 3D Tracker

3D object tracking from segmentation masks and depth images for ROS2.

## Overview

This package provides 3D object tracking by combining:
- Object detection results with segmentation masks (from `ultralytics_ros2` or similar)
- Depth camera images  
- Camera intrinsic parameters

The system converts 2D segmentation masks to 3D object centroids and visualizes them in RViz.

## Features

- **Mask-based 3D conversion**: Uses segmentation masks for accurate 3D positioning
- **Velocity tracking**: Real-time 3D velocity calculation with smoothing
- **Outlier filtering**: Statistical outlier removal for robust centroid estimation
- **RViz visualization**: Real-time 3D markers with object labels, tracking IDs, and velocity arrows
- **Configurable parameters**: Adjustable processing and visualization settings

## Dependencies

- `perception_interface` (custom detection messages)
- Standard ROS2 packages: `sensor_msgs`, `visualization_msgs`, `geometry_msgs`
- Python packages: `numpy`, `opencv-python`

## Usage

### Basic Launch

```bash
ros2 launch object_3d_tracker object_3d_tracker.launch.py
```

### With Custom Topics

```bash  
ros2 launch object_3d_tracker object_3d_tracker.launch.py \
    detection_topic:=/your/detection/topic \
    depth_topic:=/your/depth/topic \
    camera_info_topic:=/your/camera_info/topic
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detection_topic` | `/detection/results` | Input detection results topic |
| `depth_topic` | `/camera/depth/image_raw` | Input depth image topic |
| `camera_info_topic` | `/camera/depth/camera_info` | Camera info topic |
| `output_topic` | `/object_3d_tracker/markers` | Output markers topic |
| `min_mask_pixels` | 20 | Minimum mask pixels for processing |
| `min_depth` | 0.1 | Minimum depth value (meters) |
| `max_depth` | 10.0 | Maximum depth value (meters) |
| `marker_scale` | 0.2 | Scale of visualization markers |
| `use_morphology` | true | Apply morphological operations |
| `velocity_history_size` | 10 | History size for velocity calculation |
| `velocity_smoothing_window` | 3 | Smoothing window for velocity |
| `min_velocity_threshold` | 0.05 | Minimum velocity threshold (m/s) |
| `publish_velocity_markers` | true | Show velocity arrows in RViz |
| `velocity_arrow_scale` | 1.0 | Scale factor for velocity arrows |

## Message Types

### Input
- `perception_interface/DetectionArray`: Object detections with segmentation masks
- `sensor_msgs/Image`: Depth image (32FC1 or 16UC1)
- `sensor_msgs/CameraInfo`: Camera intrinsic parameters

### Output  
- `visualization_msgs/MarkerArray`: 3D object markers for RViz

## Architecture

```
object_3d_tracker/
├── object_3d_tracker_node.py    # Main ROS2 node
└── utils/
    ├── mask_processor.py        # Segmentation mask processing
    ├── depth_processor.py       # Depth image processing & 3D conversion
    └── visualizer.py           # RViz marker generation
```

## RViz Configuration

Add these display types in RViz:
- **MarkerArray**: Subscribe to `/object_3d_tracker/markers`
- Set the fixed frame to your depth camera frame

## Troubleshooting

1. **No 3D objects detected**: Check that detection messages contain segmentation masks
2. **Markers not appearing**: Verify camera frame transformations in RViz  
3. **Poor centroid quality**: Adjust `min_mask_pixels` and morphology settings
4. **Performance issues**: Reduce `max_depth` or increase `min_mask_pixels`