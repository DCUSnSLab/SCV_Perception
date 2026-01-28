#include "negative_obstacle_detector/detector_node.hpp"

#include <cmath>
#include <cstring>

using std::placeholders::_1;

namespace negative_obstacle_detector
{

DetectorNode::DetectorNode(const rclcpp::NodeOptions& options)
  : Node("negative_obstacle_detector", options)
{
  // Parameters
  this->declare_parameter("input_topic", "/velodyne_points");
  this->declare_parameter("output_topic", "/cliff_edge");
  this->declare_parameter("base_frame", "base_link");
  this->declare_parameter("sensor_frame", "velodyne");
  this->declare_parameter("range_x", 15.0);
  this->declare_parameter("range_y", 5.0);
  this->declare_parameter("ground_z_min", -0.1);
  this->declare_parameter("ground_z_max", 0.1);
  this->declare_parameter("negative_z_max", -0.15);
  this->declare_parameter("num_sectors", 360);
  this->declare_parameter("cluster_tolerance", 0.5);
  this->declare_parameter("interpolation_resolution", 0.1);

  input_topic_ = this->get_parameter("input_topic").as_string();
  output_topic_ = this->get_parameter("output_topic").as_string();
  base_frame_ = this->get_parameter("base_frame").as_string();
  sensor_frame_ = this->get_parameter("sensor_frame").as_string();
  range_x_ = this->get_parameter("range_x").as_double();
  range_y_ = this->get_parameter("range_y").as_double();
  ground_z_min_ = this->get_parameter("ground_z_min").as_double();
  ground_z_max_ = this->get_parameter("ground_z_max").as_double();
  negative_z_max_ = this->get_parameter("negative_z_max").as_double();
  num_sectors_ = this->get_parameter("num_sectors").as_int();
  cluster_tolerance_ = this->get_parameter("cluster_tolerance").as_double();
  interpolation_res_ = this->get_parameter("interpolation_resolution").as_double();

  // TF
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Detector
  detector_ = std::make_unique<EdgeDetector>();
  detector_->initialize(range_x_, range_y_, ground_z_min_, ground_z_max_, negative_z_max_, num_sectors_,
                        cluster_tolerance_, interpolation_res_);

  // ROS
  rclcpp::QoS qos(10);
  qos.reliability(rclcpp::ReliabilityPolicy::Reliable);

  pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    input_topic_, qos,
    std::bind(&DetectorNode::pointCloudCallback, this, _1));

  pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, qos);

  RCLCPP_INFO(this->get_logger(), "Cliff edge detector initialized");
  RCLCPP_INFO(this->get_logger(), "  Input: %s", input_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "  Output: %s", output_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "  Range X: %.1f m, Y: %.1f m", range_x_, range_y_);
  RCLCPP_INFO(this->get_logger(), "  Ground Z: %.2f ~ %.2f m", ground_z_min_, ground_z_max_);
  RCLCPP_INFO(this->get_logger(), "  Negative Z max: %.2f m", negative_z_max_);
  RCLCPP_INFO(this->get_logger(), "  Sectors: %d", num_sectors_);
  RCLCPP_INFO(this->get_logger(), "  Cluster tolerance: %.2f m", cluster_tolerance_);
  RCLCPP_INFO(this->get_logger(), "  Interpolation res: %.2f m", interpolation_res_);
}

bool DetectorNode::lookupTransform(
  const std::string& target_frame,
  const std::string& source_frame,
  geometry_msgs::msg::TransformStamped& transform)
{
  try {
    transform = tf_buffer_->lookupTransform(target_frame, source_frame, tf2::TimePointZero);
    return true;
  } catch (const tf2::TransformException& ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "TF lookup failed: %s", ex.what());
    return false;
  }
}

int DetectorNode::findFieldOffset(
  const sensor_msgs::msg::PointCloud2& cloud,
  const std::string& field_name) const
{
  for (const auto& field : cloud.fields) {
    if (field.name == field_name) {
      return static_cast<int>(field.offset);
    }
  }
  return -1;
}

bool DetectorNode::parsePointCloud(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud,
  std::vector<Point3D>& points)
{
  int x_off = findFieldOffset(*cloud, "x");
  int y_off = findFieldOffset(*cloud, "y");
  int z_off = findFieldOffset(*cloud, "z");

  if (x_off < 0 || y_off < 0 || z_off < 0) {
    return false;
  }

  const uint32_t step = cloud->point_step;
  const size_t n = cloud->width * cloud->height;
  const uint8_t* ptr = cloud->data.data();

  points.clear();
  points.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    const uint8_t* p = ptr + i * step;
    float x, y, z;
    std::memcpy(&x, p + x_off, sizeof(float));
    std::memcpy(&y, p + y_off, sizeof(float));
    std::memcpy(&z, p + z_off, sizeof(float));

    if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
      points.push_back({x, y, z});
    }
  }

  return true;
}

void DetectorNode::transformPoints(
  std::vector<Point3D>& points,
  const geometry_msgs::msg::TransformStamped& transform)
{
  const auto& t = transform.transform.translation;
  const auto& r = transform.transform.rotation;

  float qx = static_cast<float>(r.x);
  float qy = static_cast<float>(r.y);
  float qz = static_cast<float>(r.z);
  float qw = static_cast<float>(r.w);

  float xx = qx * qx, yy = qy * qy, zz = qz * qz;
  float xy = qx * qy, xz = qx * qz, yz = qy * qz;
  float wx = qw * qx, wy = qw * qy, wz = qw * qz;

  float r00 = 1.0f - 2.0f * (yy + zz);
  float r01 = 2.0f * (xy - wz);
  float r02 = 2.0f * (xz + wy);
  float r10 = 2.0f * (xy + wz);
  float r11 = 1.0f - 2.0f * (xx + zz);
  float r12 = 2.0f * (yz - wx);
  float r20 = 2.0f * (xz - wy);
  float r21 = 2.0f * (yz + wx);
  float r22 = 1.0f - 2.0f * (xx + yy);

  float tx = static_cast<float>(t.x);
  float ty = static_cast<float>(t.y);
  float tz = static_cast<float>(t.z);

  for (auto& p : points) {
    float x = p.x, y = p.y, z = p.z;
    p.x = r00 * x + r01 * y + r02 * z + tx;
    p.y = r10 * x + r11 * y + r12 * z + ty;
    p.z = r20 * x + r21 * y + r22 * z + tz;
  }
}

void DetectorNode::pointCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg)
{
  std::string source_frame = msg->header.frame_id;
  if (source_frame.empty()) {
    source_frame = sensor_frame_;
  }
  if (!source_frame.empty() && source_frame[0] == '/') {
    source_frame = source_frame.substr(1);
  }

  geometry_msgs::msg::TransformStamped tf;
  if (!lookupTransform(base_frame_, source_frame, tf)) {
    return;
  }

  std::vector<Point3D> points;
  if (!parsePointCloud(msg, points)) {
    return;
  }

  transformPoints(points, tf);

  auto edges = detector_->filter(points);

  if (!edges.empty()) {
    publishCloud(edges);
  }
}

void DetectorNode::publishCloud(const std::vector<Point3D>& points)
{
  sensor_msgs::msg::PointCloud2 msg;

  msg.header.stamp = this->now();
  msg.header.frame_id = base_frame_;
  msg.height = 1;
  msg.width = static_cast<uint32_t>(points.size());

  sensor_msgs::msg::PointField fx, fy, fz;
  fx.name = "x"; fx.offset = 0; fx.datatype = sensor_msgs::msg::PointField::FLOAT32; fx.count = 1;
  fy.name = "y"; fy.offset = 4; fy.datatype = sensor_msgs::msg::PointField::FLOAT32; fy.count = 1;
  fz.name = "z"; fz.offset = 8; fz.datatype = sensor_msgs::msg::PointField::FLOAT32; fz.count = 1;
  msg.fields = {fx, fy, fz};

  msg.is_bigendian = false;
  msg.point_step = 12;
  msg.row_step = msg.point_step * msg.width;
  msg.is_dense = true;

  msg.data.resize(msg.row_step);
  float* ptr = reinterpret_cast<float*>(msg.data.data());

  for (const auto& p : points) {
    *ptr++ = p.x;
    *ptr++ = p.y;
    *ptr++ = p.z;
  }

  pc_pub_->publish(msg);
}

}  // namespace negative_obstacle_detector
