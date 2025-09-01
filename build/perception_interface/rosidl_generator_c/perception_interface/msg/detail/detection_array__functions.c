// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from perception_interface:msg/DetectionArray.idl
// generated code does not contain a copyright notice
#include "perception_interface/msg/detail/detection_array__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `source_image`
#include "sensor_msgs/msg/detail/image__functions.h"
// Member `model_name`
// Member `model_version`
#include "rosidl_runtime_c/string_functions.h"
// Member `detections`
#include "perception_interface/msg/detail/detection_result__functions.h"

bool
perception_interface__msg__DetectionArray__init(perception_interface__msg__DetectionArray * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    perception_interface__msg__DetectionArray__fini(msg);
    return false;
  }
  // source_image
  if (!sensor_msgs__msg__Image__init(&msg->source_image)) {
    perception_interface__msg__DetectionArray__fini(msg);
    return false;
  }
  // model_name
  if (!rosidl_runtime_c__String__init(&msg->model_name)) {
    perception_interface__msg__DetectionArray__fini(msg);
    return false;
  }
  // model_version
  if (!rosidl_runtime_c__String__init(&msg->model_version)) {
    perception_interface__msg__DetectionArray__fini(msg);
    return false;
  }
  // detections
  if (!perception_interface__msg__DetectionResult__Sequence__init(&msg->detections, 0)) {
    perception_interface__msg__DetectionArray__fini(msg);
    return false;
  }
  return true;
}

void
perception_interface__msg__DetectionArray__fini(perception_interface__msg__DetectionArray * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // source_image
  sensor_msgs__msg__Image__fini(&msg->source_image);
  // model_name
  rosidl_runtime_c__String__fini(&msg->model_name);
  // model_version
  rosidl_runtime_c__String__fini(&msg->model_version);
  // detections
  perception_interface__msg__DetectionResult__Sequence__fini(&msg->detections);
}

bool
perception_interface__msg__DetectionArray__are_equal(const perception_interface__msg__DetectionArray * lhs, const perception_interface__msg__DetectionArray * rhs)
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
  // source_image
  if (!sensor_msgs__msg__Image__are_equal(
      &(lhs->source_image), &(rhs->source_image)))
  {
    return false;
  }
  // model_name
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->model_name), &(rhs->model_name)))
  {
    return false;
  }
  // model_version
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->model_version), &(rhs->model_version)))
  {
    return false;
  }
  // detections
  if (!perception_interface__msg__DetectionResult__Sequence__are_equal(
      &(lhs->detections), &(rhs->detections)))
  {
    return false;
  }
  return true;
}

bool
perception_interface__msg__DetectionArray__copy(
  const perception_interface__msg__DetectionArray * input,
  perception_interface__msg__DetectionArray * output)
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
  // source_image
  if (!sensor_msgs__msg__Image__copy(
      &(input->source_image), &(output->source_image)))
  {
    return false;
  }
  // model_name
  if (!rosidl_runtime_c__String__copy(
      &(input->model_name), &(output->model_name)))
  {
    return false;
  }
  // model_version
  if (!rosidl_runtime_c__String__copy(
      &(input->model_version), &(output->model_version)))
  {
    return false;
  }
  // detections
  if (!perception_interface__msg__DetectionResult__Sequence__copy(
      &(input->detections), &(output->detections)))
  {
    return false;
  }
  return true;
}

perception_interface__msg__DetectionArray *
perception_interface__msg__DetectionArray__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  perception_interface__msg__DetectionArray * msg = (perception_interface__msg__DetectionArray *)allocator.allocate(sizeof(perception_interface__msg__DetectionArray), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(perception_interface__msg__DetectionArray));
  bool success = perception_interface__msg__DetectionArray__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
perception_interface__msg__DetectionArray__destroy(perception_interface__msg__DetectionArray * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    perception_interface__msg__DetectionArray__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
perception_interface__msg__DetectionArray__Sequence__init(perception_interface__msg__DetectionArray__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  perception_interface__msg__DetectionArray * data = NULL;

  if (size) {
    data = (perception_interface__msg__DetectionArray *)allocator.zero_allocate(size, sizeof(perception_interface__msg__DetectionArray), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = perception_interface__msg__DetectionArray__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        perception_interface__msg__DetectionArray__fini(&data[i - 1]);
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
perception_interface__msg__DetectionArray__Sequence__fini(perception_interface__msg__DetectionArray__Sequence * array)
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
      perception_interface__msg__DetectionArray__fini(&array->data[i]);
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

perception_interface__msg__DetectionArray__Sequence *
perception_interface__msg__DetectionArray__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  perception_interface__msg__DetectionArray__Sequence * array = (perception_interface__msg__DetectionArray__Sequence *)allocator.allocate(sizeof(perception_interface__msg__DetectionArray__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = perception_interface__msg__DetectionArray__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
perception_interface__msg__DetectionArray__Sequence__destroy(perception_interface__msg__DetectionArray__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    perception_interface__msg__DetectionArray__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
perception_interface__msg__DetectionArray__Sequence__are_equal(const perception_interface__msg__DetectionArray__Sequence * lhs, const perception_interface__msg__DetectionArray__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!perception_interface__msg__DetectionArray__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
perception_interface__msg__DetectionArray__Sequence__copy(
  const perception_interface__msg__DetectionArray__Sequence * input,
  perception_interface__msg__DetectionArray__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(perception_interface__msg__DetectionArray);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    perception_interface__msg__DetectionArray * data =
      (perception_interface__msg__DetectionArray *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!perception_interface__msg__DetectionArray__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          perception_interface__msg__DetectionArray__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!perception_interface__msg__DetectionArray__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
