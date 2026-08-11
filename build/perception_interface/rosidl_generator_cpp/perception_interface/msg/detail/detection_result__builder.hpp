// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from perception_interface:msg/DetectionResult.idl
// generated code does not contain a copyright notice

#ifndef PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_RESULT__BUILDER_HPP_
#define PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_RESULT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "perception_interface/msg/detail/detection_result__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace perception_interface
{

namespace msg
{

namespace builder
{

class Init_DetectionResult_mask
{
public:
  explicit Init_DetectionResult_mask(::perception_interface::msg::DetectionResult & msg)
  : msg_(msg)
  {}
  ::perception_interface::msg::DetectionResult mask(::perception_interface::msg::DetectionResult::_mask_type arg)
  {
    msg_.mask = std::move(arg);
    return std::move(msg_);
  }

private:
  ::perception_interface::msg::DetectionResult msg_;
};

class Init_DetectionResult_track_id
{
public:
  explicit Init_DetectionResult_track_id(::perception_interface::msg::DetectionResult & msg)
  : msg_(msg)
  {}
  Init_DetectionResult_mask track_id(::perception_interface::msg::DetectionResult::_track_id_type arg)
  {
    msg_.track_id = std::move(arg);
    return Init_DetectionResult_mask(msg_);
  }

private:
  ::perception_interface::msg::DetectionResult msg_;
};

class Init_DetectionResult_centroid
{
public:
  explicit Init_DetectionResult_centroid(::perception_interface::msg::DetectionResult & msg)
  : msg_(msg)
  {}
  Init_DetectionResult_track_id centroid(::perception_interface::msg::DetectionResult::_centroid_type arg)
  {
    msg_.centroid = std::move(arg);
    return Init_DetectionResult_track_id(msg_);
  }

private:
  ::perception_interface::msg::DetectionResult msg_;
};

class Init_DetectionResult_bounding_box
{
public:
  explicit Init_DetectionResult_bounding_box(::perception_interface::msg::DetectionResult & msg)
  : msg_(msg)
  {}
  Init_DetectionResult_centroid bounding_box(::perception_interface::msg::DetectionResult::_bounding_box_type arg)
  {
    msg_.bounding_box = std::move(arg);
    return Init_DetectionResult_centroid(msg_);
  }

private:
  ::perception_interface::msg::DetectionResult msg_;
};

class Init_DetectionResult_confidence
{
public:
  explicit Init_DetectionResult_confidence(::perception_interface::msg::DetectionResult & msg)
  : msg_(msg)
  {}
  Init_DetectionResult_bounding_box confidence(::perception_interface::msg::DetectionResult::_confidence_type arg)
  {
    msg_.confidence = std::move(arg);
    return Init_DetectionResult_bounding_box(msg_);
  }

private:
  ::perception_interface::msg::DetectionResult msg_;
};

class Init_DetectionResult_class_name
{
public:
  explicit Init_DetectionResult_class_name(::perception_interface::msg::DetectionResult & msg)
  : msg_(msg)
  {}
  Init_DetectionResult_confidence class_name(::perception_interface::msg::DetectionResult::_class_name_type arg)
  {
    msg_.class_name = std::move(arg);
    return Init_DetectionResult_confidence(msg_);
  }

private:
  ::perception_interface::msg::DetectionResult msg_;
};

class Init_DetectionResult_header
{
public:
  Init_DetectionResult_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DetectionResult_class_name header(::perception_interface::msg::DetectionResult::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_DetectionResult_class_name(msg_);
  }

private:
  ::perception_interface::msg::DetectionResult msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::perception_interface::msg::DetectionResult>()
{
  return perception_interface::msg::builder::Init_DetectionResult_header();
}

}  // namespace perception_interface

#endif  // PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_RESULT__BUILDER_HPP_
