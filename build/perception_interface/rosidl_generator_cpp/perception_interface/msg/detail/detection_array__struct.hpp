// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from perception_interface:msg/DetectionArray.idl
// generated code does not contain a copyright notice

#ifndef PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_ARRAY__STRUCT_HPP_
#define PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_ARRAY__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'source_image'
#include "sensor_msgs/msg/detail/image__struct.hpp"
// Member 'detections'
#include "perception_interface/msg/detail/detection_result__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__perception_interface__msg__DetectionArray __attribute__((deprecated))
#else
# define DEPRECATED__perception_interface__msg__DetectionArray __declspec(deprecated)
#endif

namespace perception_interface
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct DetectionArray_
{
  using Type = DetectionArray_<ContainerAllocator>;

  explicit DetectionArray_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    source_image(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->model_name = "";
      this->model_version = "";
    }
  }

  explicit DetectionArray_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    source_image(_alloc, _init),
    model_name(_alloc),
    model_version(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->model_name = "";
      this->model_version = "";
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _source_image_type =
    sensor_msgs::msg::Image_<ContainerAllocator>;
  _source_image_type source_image;
  using _model_name_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _model_name_type model_name;
  using _model_version_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _model_version_type model_version;
  using _detections_type =
    std::vector<perception_interface::msg::DetectionResult_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<perception_interface::msg::DetectionResult_<ContainerAllocator>>>;
  _detections_type detections;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__source_image(
    const sensor_msgs::msg::Image_<ContainerAllocator> & _arg)
  {
    this->source_image = _arg;
    return *this;
  }
  Type & set__model_name(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->model_name = _arg;
    return *this;
  }
  Type & set__model_version(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->model_version = _arg;
    return *this;
  }
  Type & set__detections(
    const std::vector<perception_interface::msg::DetectionResult_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<perception_interface::msg::DetectionResult_<ContainerAllocator>>> & _arg)
  {
    this->detections = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    perception_interface::msg::DetectionArray_<ContainerAllocator> *;
  using ConstRawPtr =
    const perception_interface::msg::DetectionArray_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<perception_interface::msg::DetectionArray_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<perception_interface::msg::DetectionArray_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      perception_interface::msg::DetectionArray_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<perception_interface::msg::DetectionArray_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      perception_interface::msg::DetectionArray_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<perception_interface::msg::DetectionArray_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<perception_interface::msg::DetectionArray_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<perception_interface::msg::DetectionArray_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__perception_interface__msg__DetectionArray
    std::shared_ptr<perception_interface::msg::DetectionArray_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__perception_interface__msg__DetectionArray
    std::shared_ptr<perception_interface::msg::DetectionArray_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DetectionArray_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->source_image != other.source_image) {
      return false;
    }
    if (this->model_name != other.model_name) {
      return false;
    }
    if (this->model_version != other.model_version) {
      return false;
    }
    if (this->detections != other.detections) {
      return false;
    }
    return true;
  }
  bool operator!=(const DetectionArray_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DetectionArray_

// alias to use template instance with default allocator
using DetectionArray =
  perception_interface::msg::DetectionArray_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace perception_interface

#endif  // PERCEPTION_INTERFACE__MSG__DETAIL__DETECTION_ARRAY__STRUCT_HPP_
