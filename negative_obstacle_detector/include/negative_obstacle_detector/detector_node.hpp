#ifndef NEGATIVE_OBSTACLE_DETECTOR__DETECTOR_NODE_HPP_
#define NEGATIVE_OBSTACLE_DETECTOR__DETECTOR_NODE_HPP_

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "negative_obstacle_detector/edge_detector.hpp"

namespace negative_obstacle_detector
{

class DetectorNode : public rclcpp::Node
{
public:
  explicit DetectorNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg);

  bool parsePointCloud(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud,
    std::vector<Point3D>& points);

  void transformPoints(
    std::vector<Point3D>& points,
    const geometry_msgs::msg::TransformStamped& transform);

  void publishCloud(const std::vector<Point3D>& points);

  bool lookupTransform(
    const std::string& target_frame,
    const std::string& source_frame,
    geometry_msgs::msg::TransformStamped& transform);

  int findFieldOffset(
    const sensor_msgs::msg::PointCloud2& cloud,
    const std::string& field_name) const;

  // Parameters
  std::string input_topic_;
  std::string output_topic_;
  std::string base_frame_;
  std::string sensor_frame_;
  double range_x_;
  double range_y_;
  double ground_z_min_;
  double ground_z_max_;
  double negative_z_max_;
  int num_sectors_;
  double cluster_tolerance_;
  double interpolation_res_;
  double interpolation_max_dist_;

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;

  // TF
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Detector
  std::unique_ptr<EdgeDetector> detector_;
};

}  // namespace negative_obstacle_detector

#endif  // NEGATIVE_OBSTACLE_DETECTOR__DETECTOR_NODE_HPP_
