#include "ground_filter/grid_ground_filter.hpp"

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace ground_filter
{
namespace
{
constexpr double kDegToRad = M_PI / 180.0;
}  // namespace

class GroundFilterNode : public rclcpp::Node
{
public:
  GroundFilterNode()
  : Node("ground_filter_node")
  {
    const auto input_topic = declare_parameter<std::string>("input_topic", "/velodyne_points");
    const auto nonground_topic = declare_parameter<std::string>("nonground_topic", "~/nonground");
    const auto ground_topic = declare_parameter<std::string>("ground_topic", "~/ground");
    publish_ground_ = declare_parameter<bool>("publish_ground", true);
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    transform_timeout_sec_ = declare_parameter<double>("transform_timeout_sec", 0.1);
    const bool input_best_effort = declare_parameter<bool>("input_best_effort", false);

    GridGroundFilterParameter param;
    param.global_slope_max_angle_rad = static_cast<float>(
      declare_parameter<double>("global_slope_max_angle_deg", 15.0) * kDegToRad);
    param.local_slope_max_angle_rad = static_cast<float>(
      declare_parameter<double>("local_slope_max_angle_deg", 13.0) * kDegToRad);
    param.radial_divider_angle_rad = static_cast<float>(
      declare_parameter<double>("radial_divider_angle_deg", 1.0) * kDegToRad);

    param.use_recheck_ground_cluster = declare_parameter<bool>("use_recheck_ground_cluster", true);
    param.recheck_start_distance =
      static_cast<float>(declare_parameter<double>("recheck_start_distance", 20.0));
    param.use_lowest_point = declare_parameter<bool>("use_lowest_point", true);
    param.detection_range_z_max =
      static_cast<float>(declare_parameter<double>("detection_range_z_max", 2.5));
    param.non_ground_height_threshold =
      static_cast<float>(declare_parameter<double>("non_ground_height_threshold", 0.05));

    param.grid_size_m = static_cast<float>(declare_parameter<double>("grid_size_m", 0.05));
    param.grid_radial_limit_m =
      static_cast<float>(declare_parameter<double>("grid_radial_limit_m", 15.0));
    param.gnd_grid_buffer_size = declare_parameter<int>("gnd_grid_buffer_size", 4);
    param.gnd_grid_continual_thresh =
      static_cast<uint16_t>(declare_parameter<int>("gnd_grid_continual_thresh", 40));

    const auto wheel_base_m = declare_parameter<double>("wheel_base_m", 0.65);
    const auto center_pcl_shift = declare_parameter<double>("center_pcl_shift", 0.0);
    param.virtual_lidar_x = static_cast<float>(wheel_base_m / 2.0 + center_pcl_shift);
    param.virtual_lidar_y = 0.0f;

    validateParameters(param);
    filter_ = std::make_unique<GridGroundFilter>(param);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    auto input_qos = rclcpp::SensorDataQoS().keep_last(1);
    if (!input_best_effort) {
      input_qos.reliable();
    }
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic, input_qos,
      std::bind(&GroundFilterNode::cloudCallback, this, std::placeholders::_1));

    nonground_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      nonground_topic, rclcpp::SensorDataQoS().keep_last(1).reliable());
    if (publish_ground_) {
      ground_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        ground_topic, rclcpp::SensorDataQoS().keep_last(1).reliable());
    }

    RCLCPP_INFO(
      get_logger(), "ground filter started: %s -> %s (base_frame=%s, radial_limit=%.1f m)",
      input_topic.c_str(), nonground_topic.c_str(), base_frame_.c_str(),
      static_cast<double>(param.grid_radial_limit_m));
  }

private:
  void validateParameters(const GridGroundFilterParameter & p) const
  {
    if (p.grid_size_m <= 0.0f) {
      throw std::runtime_error("grid_size_m must be positive");
    }
    if (p.grid_radial_limit_m <= p.grid_size_m) {
      throw std::runtime_error("grid_radial_limit_m must be larger than grid_size_m");
    }
    if (p.radial_divider_angle_rad <= 0.0f) {
      throw std::runtime_error("radial_divider_angle_deg must be positive");
    }
    if (p.global_slope_max_angle_rad <= 0.0f || p.global_slope_max_angle_rad >= M_PI_2) {
      throw std::runtime_error("global_slope_max_angle_deg must be in (0, 90)");
    }
    if (p.local_slope_max_angle_rad <= 0.0f || p.local_slope_max_angle_rad >= M_PI_2) {
      throw std::runtime_error("local_slope_max_angle_deg must be in (0, 90)");
    }
    if (p.non_ground_height_threshold <= 0.0f) {
      throw std::runtime_error("non_ground_height_threshold must be positive");
    }
    if (p.gnd_grid_buffer_size < 1) {
      throw std::runtime_error("gnd_grid_buffer_size must be at least 1");
    }
    if (p.gnd_grid_continual_thresh < 1) {
      throw std::runtime_error("gnd_grid_continual_thresh must be at least 1");
    }
  }

  sensor_msgs::msg::PointCloud2::ConstSharedPtr toBaseFrame(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
  {
    if (msg->header.frame_id == base_frame_) {
      return msg;
    }

    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_->lookupTransform(
        base_frame_, msg->header.frame_id, msg->header.stamp,
        rclcpp::Duration::from_seconds(transform_timeout_sec_));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "no transform %s -> %s: %s",
        msg->header.frame_id.c_str(), base_frame_.c_str(), ex.what());
      return nullptr;
    }

    auto transformed = std::make_shared<sensor_msgs::msg::PointCloud2>();
    tf2::doTransform(*msg, *transformed, tf);
    transformed->header.frame_id = base_frame_;
    return transformed;
  }

  static sensor_msgs::msg::PointCloud2 extractPoints(
    const sensor_msgs::msg::PointCloud2 & in, const std::vector<size_t> & byte_offsets)
  {
    sensor_msgs::msg::PointCloud2 out;
    out.header = in.header;
    out.fields = in.fields;
    out.point_step = in.point_step;
    out.is_bigendian = in.is_bigendian;
    out.is_dense = true;
    out.height = 1;
    out.width = static_cast<uint32_t>(byte_offsets.size());
    out.row_step = static_cast<uint32_t>(byte_offsets.size() * in.point_step);
    out.data.resize(out.row_step);

    size_t write_pos = 0;
    for (const auto offset : byte_offsets) {
      std::memcpy(&out.data[write_pos], &in.data[offset], in.point_step);
      write_pos += in.point_step;
    }
    return out;
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    if (msg->data.empty() || msg->point_step == 0) {
      return;
    }

    const auto cloud = toBaseFrame(msg);
    if (cloud == nullptr) {
      return;
    }

    if (!filter_->setDataAccessor(cloud)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "input cloud has no x/y/z fields");
      return;
    }

    // Byte offsets, not point ids. The two outputs are not complementary:
    // unclassified, too-high and out-of-range points appear in neither.
    PointIndices nonground_offsets;
    PointIndices ground_offsets;
    filter_->process(cloud, nonground_offsets, ground_offsets);

    nonground_pub_->publish(extractPoints(*cloud, nonground_offsets));
    if (publish_ground_ && ground_pub_->get_subscription_count() > 0) {
      ground_pub_->publish(extractPoints(*cloud, ground_offsets));
    }
  }

  bool publish_ground_{true};
  std::string base_frame_{"base_link"};
  double transform_timeout_sec_{0.1};

  std::unique_ptr<GridGroundFilter> filter_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nonground_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_pub_;
};

}  // namespace ground_filter

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ground_filter::GroundFilterNode>());
  rclcpp::shutdown();
  return 0;
}
