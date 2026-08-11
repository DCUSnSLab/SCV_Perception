// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from perception_interface:msg/DetectionResult.idl
// generated code does not contain a copyright notice
#include "perception_interface/msg/detail/detection_result__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `class_name`
#include "rosidl_runtime_c/string_functions.h"
// Member `bounding_box`
#include "geometry_msgs/msg/detail/polygon__functions.h"
// Member `centroid`
#include "geometry_msgs/msg/detail/point__functions.h"
// Member `mask`
#include "sensor_msgs/msg/detail/image__functions.h"

bool
perception_interface__msg__DetectionResult__init(perception_interface__msg__DetectionResult * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    perception_interface__msg__DetectionResult__fini(msg);
    return false;
  }
  // class_name
  if (!rosidl_runtime_c__String__init(&msg->class_name)) {
    perception_interface__msg__DetectionResult__fini(msg);
    return false;
  }
  // confidence
  // bounding_box
  if (!geometry_msgs__msg__Polygon__init(&msg->bounding_box)) {
    perception_interface__msg__DetectionResult__fini(msg);
    return false;
  }
  // centroid
  if (!geometry_msgs__msg__Point__init(&msg->centroid)) {
    perception_interface__msg__DetectionResult__fini(msg);
    return false;
  }
  // track_id
  // mask
  if (!sensor_msgs__msg__Image__init(&msg->mask)) {
    perception_interface__msg__DetectionResult__fini(msg);
    return false;
  }
  return true;
}

void
perception_interface__msg__DetectionResult__fini(perception_interface__msg__DetectionResult * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // class_name
  rosidl_runtime_c__String__fini(&msg->class_name);
  // confidence
  // bounding_box
  geometry_msgs__msg__Polygon__fini(&msg->bounding_box);
  // centroid
  geometry_msgs__msg__Point__fini(&msg->centroid);
  // track_id
  // mask
  sensor_msgs__msg__Image__fini(&msg->mask);
}

bool
perception_interface__msg__DetectionResult__are_equal(const perception_interface__msg__DetectionResult * lhs, const perception_interface__msg__DetectionResult * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // class_name
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->class_name), &(rhs->class_name)))
  {
    return false;
  }
  // confidence
  if (lhs->confidence != rhs->confidence) {
    return false;
  }
  // bounding_box
  if (!geometry_msgs__msg__Polygon__are_equal(
      &(lhs->bounding_box), &(rhs->bounding_box)))
  {
    return false;
  }
  // centroid
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->centroid), &(rhs->centroid)))
  {
    return false;
  }
  // track_id
  if (lhs->track_id != rhs->track_id) {
    return false;
  }
  // mask
  if (!sensor_msgs__msg__Image__are_equal(
      &(lhs->mask), &(rhs->mask)))
  {
    return false;
  }
  return true;
}

bool
perception_interface__msg__DetectionResult__copy(
  const perception_interface__msg__DetectionResult * input,
  perception_interface__msg__DetectionResult * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // class_name
  if (!rosidl_runtime_c__String__copy(
      &(input->class_name), &(output->class_name)))
  {
    return false;
  }
  // confidence
  output->confidence = input->confidence;
  // bounding_box
  if (!geometry_msgs__msg__Polygon__copy(
      &(input->bounding_box), &(output->bounding_box)))
  {
    return false;
  }
  // centroid
  if (!geometry_msgs__msg__Point__copy(
      &(input->centroid), &(output->centroid)))
  {
    return false;
  }
  // track_id
  output->track_id = input->track_id;
  // mask
  if (!sensor_msgs__msg__Image__copy(
      &(input->mask), &(output->mask)))
  {
    return false;
  }
  return true;
}

perception_interface__msg__DetectionResult *
perception_interface__msg__DetectionResult__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  perception_interface__msg__DetectionResult * msg = (perception_interface__msg__DetectionResult *)allocator.allocate(sizeof(perception_interface__msg__DetectionResult), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(perception_interface__msg__DetectionResult));
  bool success = perception_interface__msg__DetectionResult__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
perception_interface__msg__DetectionResult__destroy(perception_interface__msg__DetectionResult * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    perception_interface__msg__DetectionResult__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
perception_interface__msg__DetectionResult__Sequence__init(perception_interface__msg__DetectionResult__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  perception_interface__msg__DetectionResult * data = NULL;

  if (size) {
    data = (perception_interface__msg__DetectionResult *)allocator.zero_allocate(size, sizeof(perception_interface__msg__DetectionResult), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = perception_interface__msg__DetectionResult__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        perception_interface__msg__DetectionResult__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
perception_interface__msg__DetectionResult__Sequence__fini(perception_interface__msg__DetectionResult__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      perception_interface__msg__DetectionResult__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

perception_interface__msg__DetectionResult__Sequence *
perception_interface__msg__DetectionResult__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  perception_interface__msg__DetectionResult__Sequence * array = (perception_interface__msg__DetectionResult__Sequence *)allocator.allocate(sizeof(perception_interface__msg__DetectionResult__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = perception_interface__msg__DetectionResult__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
perception_interface__msg__DetectionResult__Sequence__destroy(perception_interface__msg__DetectionResult__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    perception_interface__msg__DetectionResult__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
perception_interface__msg__DetectionResult__Sequence__are_equal(const perception_interface__msg__DetectionResult__Sequence * lhs, const perception_interface__msg__DetectionResult__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!perception_interface__msg__DetectionResult__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
perception_interface__msg__DetectionResult__Sequence__copy(
  const perception_interface__msg__DetectionResult__Sequence * input,
  perception_interface__msg__DetectionResult__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(perception_interface__msg__DetectionResult);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    perception_interface__msg__DetectionResult * data =
      (perception_interface__msg__DetectionResult *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!perception_interface__msg__DetectionResult__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          perception_interface__msg__DetectionResult__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!perception_interface__msg__DetectionResult__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
