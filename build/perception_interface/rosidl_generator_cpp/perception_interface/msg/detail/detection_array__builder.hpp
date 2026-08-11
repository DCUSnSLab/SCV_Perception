// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from perception_interface:msg/DetectionArray.idl
// generated code does not contain a copyright notice

#ifndef PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_ARRAY__BUILDER_HPP_
#define PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_ARRAY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "perception_interface/msg/detail/detection_array__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace perception_interface
{

namespace msg
{

namespace builder
{

class Init_DetectionArray_detections
{
public:
  explicit Init_DetectionArray_detections(::perception_interface::msg::DetectionArray & msg)
  : msg_(msg)
  {}
  ::perception_interface::msg::DetectionArray detections(::perception_interface::msg::DetectionArray::_detections_type arg)
  {
    msg_.detections = std::move(arg);
    return std::move(msg_);
  }

private:
  ::perception_interface::msg::DetectionArray msg_;
};

class Init_DetectionArray_model_version
{
public:
  explicit Init_DetectionArray_model_version(::perception_interface::msg::DetectionArray & msg)
  : msg_(msg)
  {}
  Init_DetectionArray_detections model_version(::perception_interface::msg::DetectionArray::_model_version_type arg)
  {
    msg_.model_version = std::move(arg);
    return Init_DetectionArray_detections(msg_);
  }

private:
  ::perception_interface::msg::DetectionArray msg_;
};

class Init_DetectionArray_model_name
{
public:
  explicit Init_DetectionArray_model_name(::perception_interface::msg::DetectionArray & msg)
  : msg_(msg)
  {}
  Init_DetectionArray_model_version model_name(::perception_interface::msg::DetectionArray::_model_name_type arg)
  {
    msg_.model_name = std::move(arg);
    return Init_DetectionArray_model_version(msg_);
  }

private:
  ::perception_interface::msg::DetectionArray msg_;
};

class Init_DetectionArray_source_image
{
public:
  explicit Init_DetectionArray_source_image(::perception_interface::msg::DetectionArray & msg)
  : msg_(msg)
  {}
  Init_DetectionArray_model_name source_image(::perception_interface::msg::DetectionArray::_source_image_type arg)
  {
    msg_.source_image = std::move(arg);
    return Init_DetectionArray_model_name(msg_);
  }

private:
  ::perception_interface::msg::DetectionArray msg_;
};

class Init_DetectionArray_header
{
public:
  Init_DetectionArray_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DetectionArray_source_image header(::perception_interface::msg::DetectionArray::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_DetectionArray_source_image(msg_);
  }

private:
  ::perception_interface::msg::DetectionArray msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::perception_interface::msg::DetectionArray>()
{
  return perception_interface::msg::builder::Init_DetectionArray_header();
}

}  // namespace perception_interface

#endif  // PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_ARRAY__BUILDER_HPP_
