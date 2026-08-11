// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from perception_interface:msg/DetectionArray.idl
// generated code does not contain a copyright notice

#ifndef PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_ARRAY__STRUCT_H_
#define PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_ARRAY__STRUCT_H_

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
// Member 'source_image'
#include "sensor_msgs/msg/detail/image__struct.h"
// Member 'model_name'
// Member 'model_version'
#include "rosidl_runtime_c/string.h"
// Member 'detections'
#include "perception_interface/msg/detail/detection_result__struct.h"

/// Struct defined in msg/DetectionArray in the package perception_interface.
/**
  * Array of detection results
 */
typedef struct perception_interface__msg__DetectionArray
{
  std_msgs__msg__Header header;
  /// Source image info
  sensor_msgs__msg__Image source_image;
  rosidl_runtime_c__String model_name;
  rosidl_runtime_c__String model_version;
  /// Detection results
  perception_interface__msg__DetectionResult__Sequence detections;
} perception_interface__msg__DetectionArray;

// Struct for a sequence of perception_interface__msg__DetectionArray.
typedef struct perception_interface__msg__DetectionArray__Sequence
{
  perception_interface__msg__DetectionArray * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} perception_interface__msg__DetectionArray__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_ARRAY__STRUCT_H_
