// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from perception_interface:msg/DetectionResult.idl
// generated code does not contain a copyright notice

#ifndef PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_RESULT__STRUCT_H_
#define PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_RESULT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'class_name'
#include "rosidl_runtime_c/string.h"
// Member 'bounding_box'
#include "geometry_msgs/msg/detail/polygon__struct.h"
// Member 'centroid'
#include "geometry_msgs/msg/detail/point__struct.h"
// Member 'mask'
#include "sensor_msgs/msg/detail/image__struct.h"

/// Struct defined in msg/DetectionResult in the package perception_interface.
/**
  * Detection result message for object detection
 */
typedef struct perception_interface__msg__DetectionResult
{
  std_msgs__msg__Header header;
  /// Detection info
  rosidl_runtime_c__String class_name;
  double confidence;
  geometry_msgs__msg__Polygon bounding_box;
  geometry_msgs__msg__Point centroid;
  /// Tracking info (optional, -1 if not available)
  int32_t track_id;
  /// Segmentation mask (optional, for segmentation models)
  sensor_msgs__msg__Image mask;
} perception_interface__msg__DetectionResult;

// Struct for a sequence of perception_interface__msg__DetectionResult.
typedef struct perception_interface__msg__DetectionResult__Sequence
{
  perception_interface__msg__DetectionResult * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} perception_interface__msg__DetectionResult__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_RESULT__STRUCT_H_
