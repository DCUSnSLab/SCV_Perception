#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "negative_obstacle_detector/detector_node.hpp"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<negative_obstacle_detector::DetectorNode>();

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
