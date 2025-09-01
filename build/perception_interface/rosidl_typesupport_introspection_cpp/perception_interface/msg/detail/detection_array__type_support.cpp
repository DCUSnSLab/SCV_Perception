// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from perception_interface:msg/DetectionArray.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "perception_interface/msg/detail/detection_array__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace perception_interface
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void DetectionArray_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) perception_interface::msg::DetectionArray(_init);
}

void DetectionArray_fini_function(void * message_memory)
{
  auto typed_message = static_cast<perception_interface::msg::DetectionArray *>(message_memory);
  typed_message->~DetectionArray();
}

size_t size_function__DetectionArray__detections(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<perception_interface::msg::DetectionResult> *>(untyped_member);
  return member->size();
}

const void * get_const_function__DetectionArray__detections(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<perception_interface::msg::DetectionResult> *>(untyped_member);
  return &member[index];
}

void * get_function__DetectionArray__detections(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<perception_interface::msg::DetectionResult> *>(untyped_member);
  return &member[index];
}

void fetch_function__DetectionArray__detections(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const perception_interface::msg::DetectionResult *>(
    get_const_function__DetectionArray__detections(untyped_member, index));
  auto & value = *reinterpret_cast<perception_interface::msg::DetectionResult *>(untyped_value);
  value = item;
}

void assign_function__DetectionArray__detections(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<perception_interface::msg::DetectionResult *>(
    get_function__DetectionArray__detections(untyped_member, index));
  const auto & value = *reinterpret_cast<const perception_interface::msg::DetectionResult *>(untyped_value);
  item = value;
}

void resize_function__DetectionArray__detections(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<perception_interface::msg::DetectionResult> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember DetectionArray_message_member_array[5] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(perception_interface::msg::DetectionArray, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "source_image",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<sensor_msgs::msg::Image>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(perception_interface::msg::DetectionArray, source_image),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "model_name",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(perception_interface::msg::DetectionArray, model_name),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "model_version",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(perception_interface::msg::DetectionArray, model_version),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "detections",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<perception_interface::msg::DetectionResult>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(perception_interface::msg::DetectionArray, detections),  // bytes offset in struct
    nullptr,  // default value
    size_function__DetectionArray__detections,  // size() function pointer
    get_const_function__DetectionArray__detections,  // get_const(index) function pointer
    get_function__DetectionArray__detections,  // get(index) function pointer
    fetch_function__DetectionArray__detections,  // fetch(index, &value) function pointer
    assign_function__DetectionArray__detections,  // assign(index, value) function pointer
    resize_function__DetectionArray__detections  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers DetectionArray_message_members = {
  "perception_interface::msg",  // message namespace
  "DetectionArray",  // message name
  5,  // number of fields
  sizeof(perception_interface::msg::DetectionArray),
  DetectionArray_message_member_array,  // message members
  DetectionArray_init_function,  // function to initialize message memory (memory has to be allocated)
  DetectionArray_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t DetectionArray_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DetectionArray_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace perception_interface


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<perception_interface::msg::DetectionArray>()
{
  return &::perception_interface::msg::rosidl_typesupport_introspection_cpp::DetectionArray_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, perception_interface, msg, DetectionArray)() {
  return &::perception_interface::msg::rosidl_typesupport_introspection_cpp::DetectionArray_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
