// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from perception_interface:msg/DetectionResult.idl
// generated code does not contain a copyright notice

#ifndef PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_RESULT__TRAITS_HPP_
#define PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_RESULT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "perception_interface/msg/detail/detection_result__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'bounding_box'
#include "geometry_msgs/msg/detail/polygon__traits.hpp"
// Member 'centroid'
#include "geometry_msgs/msg/detail/point__traits.hpp"
// Member 'mask'
#include "sensor_msgs/msg/detail/image__traits.hpp"

namespace perception_interface
{

namespace msg
{

inline void to_flow_style_yaml(
  const DetectionResult & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: class_name
  {
    out << "class_name: ";
    rosidl_generator_traits::value_to_yaml(msg.class_name, out);
    out << ", ";
  }

  // member: confidence
  {
    out << "confidence: ";
    rosidl_generator_traits::value_to_yaml(msg.confidence, out);
    out << ", ";
  }

  // member: bounding_box
  {
    out << "bounding_box: ";
    to_flow_style_yaml(msg.bounding_box, out);
    out << ", ";
  }

  // member: centroid
  {
    out << "centroid: ";
    to_flow_style_yaml(msg.centroid, out);
    out << ", ";
  }

  // member: track_id
  {
    out << "track_id: ";
    rosidl_generator_traits::value_to_yaml(msg.track_id, out);
    out << ", ";
  }

  // member: mask
  {
    out << "mask: ";
    to_flow_style_yaml(msg.mask, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const DetectionResult & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: class_name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "class_name: ";
    rosidl_generator_traits::value_to_yaml(msg.class_name, out);
    out << "\n";
  }

  // member: confidence
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "confidence: ";
    rosidl_generator_traits::value_to_yaml(msg.confidence, out);
    out << "\n";
  }

  // member: bounding_box
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "bounding_box:\n";
    to_block_style_yaml(msg.bounding_box, out, indentation + 2);
  }

  // member: centroid
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "centroid:\n";
    to_block_style_yaml(msg.centroid, out, indentation + 2);
  }

  // member: track_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "track_id: ";
    rosidl_generator_traits::value_to_yaml(msg.track_id, out);
    out << "\n";
  }

  // member: mask
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mask:\n";
    to_block_style_yaml(msg.mask, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DetectionResult & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace perception_interface

namespace rosidl_generator_traits
{

[[deprecated("use perception_interface::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const perception_interface::msg::DetectionResult & msg,
  std::ostream & out, size_t indentation = 0)
{
  perception_interface::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use perception_interface::msg::to_yaml() instead")]]
inline std::string to_yaml(const perception_interface::msg::DetectionResult & msg)
{
  return perception_interface::msg::to_yaml(msg);
}

template<>
inline const char * data_type<perception_interface::msg::DetectionResult>()
{
  return "perception_interface::msg::DetectionResult";
}

template<>
inline const char * name<perception_interface::msg::DetectionResult>()
{
  return "perception_interface/msg/DetectionResult";
}

template<>
struct has_fixed_size<perception_interface::msg::DetectionResult>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<perception_interface::msg::DetectionResult>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<perception_interface::msg::DetectionResult>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_RESULT__TRAITS_HPP_
