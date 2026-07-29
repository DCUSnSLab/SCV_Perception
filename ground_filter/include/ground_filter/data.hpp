// Derived from autoware_universe perception/autoware_ground_segmentation
// (scan_ground_filter). Autoware plumbing removed; classification logic intact.

#ifndef GROUND_FILTER__DATA_HPP_
#define GROUND_FILTER__DATA_HPP_

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace ground_filter
{
using PointCloud2ConstPtr = sensor_msgs::msg::PointCloud2::ConstSharedPtr;

using PointIndices = std::vector<size_t>;

struct PointXYZ
{
  float x;
  float y;
  float z;
};

class PclDataAccessor
{
public:
  PclDataAccessor() = default;
  ~PclDataAccessor() = default;

  bool isInitialized() const { return data_offset_initialized_; }

  bool setField(const PointCloud2ConstPtr & input)
  {
    const int x_offset = fieldOffset(input, "x");
    const int y_offset = fieldOffset(input, "y");
    const int z_offset = fieldOffset(input, "z");
    if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
      return false;
    }
    data_offset_x_ = x_offset;
    data_offset_y_ = y_offset;
    data_offset_z_ = z_offset;
    data_offset_intensity_ = fieldOffset(input, "intensity");
    data_offset_initialized_ = true;
    return true;
  }

  inline void getPoint(
    const PointCloud2ConstPtr & input, const size_t data_index, PointXYZ & point) const
  {
    point.x = *reinterpret_cast<const float *>(&input->data[data_index + data_offset_x_]);
    point.y = *reinterpret_cast<const float *>(&input->data[data_index + data_offset_y_]);
    point.z = *reinterpret_cast<const float *>(&input->data[data_index + data_offset_z_]);
  }

private:
  static int fieldOffset(const PointCloud2ConstPtr & input, const std::string & name)
  {
    for (const auto & field : input->fields) {
      if (field.name == name) {
        return static_cast<int>(field.offset);
      }
    }
    return -1;
  }

  int data_offset_x_ = 0;
  int data_offset_y_ = 0;
  int data_offset_z_ = 0;
  int data_offset_intensity_ = -1;
  bool data_offset_initialized_ = false;
};

}  // namespace ground_filter

#endif  // GROUND_FILTER__DATA_HPP_
