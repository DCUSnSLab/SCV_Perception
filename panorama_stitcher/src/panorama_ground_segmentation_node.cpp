#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace panorama_stitcher
{

using PointCloud2 = sensor_msgs::msg::PointCloud2;
constexpr double kPi = 3.14159265358979323846;

struct ColoredPoint
{
  cv::Vec3d position;
  std::uint32_t rgb{0};
};

struct PlaneModel
{
  cv::Vec3d normal{0.0, -1.0, 0.0};
  double offset{0.0};
  std::size_t inliers{0};
  double squared_error{std::numeric_limits<double>::infinity()};
  bool valid{false};
};

struct VoxelKey
{
  std::int64_t x;
  std::int64_t y;
  std::int64_t z;

  bool operator==(const VoxelKey & other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey & key) const
  {
    std::size_t seed = std::hash<std::int64_t>{}(key.x);
    seed ^= std::hash<std::int64_t>{}(key.y) + 0x9e3779b9U +
      (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<std::int64_t>{}(key.z) + 0x9e3779b9U +
      (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

class PanoramaGroundSegmentationNode : public rclcpp::Node
{
public:
  PanoramaGroundSegmentationNode()
  : Node("panorama_ground_segmentation"),
    random_engine_(
      static_cast<std::mt19937::result_type>(
        declare_parameter<int>("random_seed", 42)))
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/panorama/points");
    obstacle_topic_ = declare_parameter<std::string>(
      "obstacle_topic", "/panorama/obstacle_points");
    ground_topic_ = declare_parameter<std::string>(
      "ground_topic", "/panorama/ground_points");
    publish_ground_cloud_ = declare_parameter<bool>(
      "publish_ground_cloud", true);

    const auto expected_up = declare_parameter<std::vector<double>>(
      "expected_up_vector", {0.0, -1.0, 0.0});
    if (expected_up.size() != 3) {
      throw std::runtime_error(
              "expected_up_vector must contain exactly 3 values");
    }
    expected_up_ = cv::Vec3d(
      expected_up[0], expected_up[1], expected_up[2]);
    const double up_norm = cv::norm(expected_up_);
    if (up_norm < 1e-9) {
      throw std::runtime_error("expected_up_vector must be non-zero");
    }
    expected_up_ /= up_norm;

    ransac_iterations_ = declare_parameter<int>(
      "ransac_iterations", 160);
    max_ransac_points_ = declare_parameter<int>(
      "max_ransac_points", 30000);
    ransac_distance_threshold_m_ = declare_parameter<double>(
      "ransac_distance_threshold_m", 0.05);
    max_ground_tilt_deg_ = declare_parameter<double>(
      "max_ground_tilt_deg", 25.0);
    min_ground_inliers_ = declare_parameter<int>(
      "min_ground_inliers", 300);
    min_ground_inlier_ratio_ = declare_parameter<double>(
      "min_ground_inlier_ratio", 0.03);
    ground_candidate_min_range_m_ = declare_parameter<double>(
      "ground_candidate_min_range_m", 0.4);
    ground_candidate_max_range_m_ = declare_parameter<double>(
      "ground_candidate_max_range_m", 8.0);
    ground_candidate_min_down_m_ = declare_parameter<double>(
      "ground_candidate_min_down_m", 0.15);
    ground_candidate_max_down_m_ = declare_parameter<double>(
      "ground_candidate_max_down_m", 2.5);
    min_plane_distance_from_origin_m_ = declare_parameter<double>(
      "min_plane_distance_from_origin_m", 0.15);
    max_plane_distance_from_origin_m_ = declare_parameter<double>(
      "max_plane_distance_from_origin_m", 2.5);
    obstacle_min_height_m_ = declare_parameter<double>(
      "obstacle_min_height_m", 0.10);
    obstacle_max_height_m_ = declare_parameter<double>(
      "obstacle_max_height_m", 2.0);
    obstacle_min_range_m_ = declare_parameter<double>(
      "obstacle_min_range_m", 0.25);
    obstacle_max_range_m_ = declare_parameter<double>(
      "obstacle_max_range_m", 8.0);
    obstacle_voxel_size_m_ = declare_parameter<double>(
      "obstacle_voxel_size_m", 0.0);
    skip_when_unsubscribed_ = declare_parameter<bool>(
      "skip_when_unsubscribed", true);
    diagnostics_period_sec_ = declare_parameter<double>(
      "diagnostics_period_sec", 2.0);
    validate_parameters();

    const auto output_qos =
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
    const auto input_qos = rclcpp::SensorDataQoS().keep_last(1);
    obstacle_publisher_ = create_publisher<PointCloud2>(
      obstacle_topic_, output_qos);
    if (publish_ground_cloud_) {
      ground_publisher_ = create_publisher<PointCloud2>(
        ground_topic_, output_qos);
    }
    pointcloud_subscriber_ = create_subscription<PointCloud2>(
      input_topic_, input_qos,
      std::bind(
        &PanoramaGroundSegmentationNode::pointcloud_callback, this,
        std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "Panorama ground segmentation: %s -> %s, expected_up=[%.2f %.2f %.2f], "
      "tilt<=%.1f deg, RANSAC threshold=%.3f m, obstacle height=%.2f..%.2f m",
      input_topic_.c_str(), obstacle_topic_.c_str(),
      expected_up_[0], expected_up_[1], expected_up_[2],
      max_ground_tilt_deg_, ransac_distance_threshold_m_,
      obstacle_min_height_m_, obstacle_max_height_m_);
  }

private:
  void validate_parameters()
  {
    ransac_iterations_ = std::max(ransac_iterations_, 1);
    max_ransac_points_ = std::max(max_ransac_points_, 3);
    ransac_distance_threshold_m_ = std::max(
      ransac_distance_threshold_m_, 0.001);
    max_ground_tilt_deg_ = std::clamp(max_ground_tilt_deg_, 0.0, 89.0);
    min_ground_inliers_ = std::max(min_ground_inliers_, 3);
    min_ground_inlier_ratio_ = std::clamp(
      min_ground_inlier_ratio_, 0.0, 1.0);
    ground_candidate_min_range_m_ = std::max(
      ground_candidate_min_range_m_, 0.0);
    ground_candidate_max_range_m_ = std::max(
      ground_candidate_max_range_m_, ground_candidate_min_range_m_);
    ground_candidate_min_down_m_ = std::max(
      ground_candidate_min_down_m_, 0.0);
    ground_candidate_max_down_m_ = std::max(
      ground_candidate_max_down_m_, ground_candidate_min_down_m_);
    min_plane_distance_from_origin_m_ = std::max(
      min_plane_distance_from_origin_m_, 0.0);
    max_plane_distance_from_origin_m_ = std::max(
      max_plane_distance_from_origin_m_,
      min_plane_distance_from_origin_m_);
    obstacle_min_height_m_ = std::max(obstacle_min_height_m_, 0.0);
    obstacle_max_height_m_ = std::max(
      obstacle_max_height_m_, obstacle_min_height_m_);
    obstacle_min_range_m_ = std::max(obstacle_min_range_m_, 0.0);
    obstacle_max_range_m_ = std::max(
      obstacle_max_range_m_, obstacle_min_range_m_);
    obstacle_voxel_size_m_ = std::max(obstacle_voxel_size_m_, 0.0);
    diagnostics_period_sec_ = std::max(diagnostics_period_sec_, 0.2);
    minimum_up_alignment_ = std::cos(max_ground_tilt_deg_ * kPi / 180.0);
  }

  // Byte offsets of the xyz+rgb fields, or nullopt when the layout is not the
  // dense float layout this node can walk with plain pointer arithmetic.
  struct CloudLayout
  {
    std::size_t x{0};
    std::size_t y{4};
    std::size_t z{8};
    std::size_t rgb{12};
    bool packed{false};
  };

  static CloudLayout inspect_layout(const PointCloud2 & message)
  {
    CloudLayout layout;
    int found = 0;
    for (const auto & field : message.fields) {
      if (field.datatype != sensor_msgs::msg::PointField::FLOAT32 ||
        field.count != 1)
      {
        continue;
      }
      if (field.name == "x") {
        layout.x = field.offset;
        ++found;
      } else if (field.name == "y") {
        layout.y = field.offset;
        ++found;
      } else if (field.name == "z") {
        layout.z = field.offset;
        ++found;
      } else if (field.name == "rgb") {
        layout.rgb = field.offset;
        ++found;
      }
    }
    layout.packed = found == 4 && message.point_step >= 16 &&
      message.data.size() >=
      static_cast<std::size_t>(message.point_step) *
      static_cast<std::size_t>(message.width) *
      static_cast<std::size_t>(message.height);
    return layout;
  }

  // PointCloud2ConstIterator recomputes a field offset on every dereference.
  // Walking the buffer directly is several times faster for the ~75k points
  // the panorama publishes at 20 Hz.
  std::vector<ColoredPoint> parse_cloud(const PointCloud2 & message) const
  {
    std::vector<ColoredPoint> points;
    const std::size_t point_total =
      static_cast<std::size_t>(message.width) *
      static_cast<std::size_t>(message.height);
    points.reserve(point_total);

    const CloudLayout layout = inspect_layout(message);
    if (!layout.packed) {
      sensor_msgs::PointCloud2ConstIterator<float> input_x(message, "x");
      sensor_msgs::PointCloud2ConstIterator<float> input_y(message, "y");
      sensor_msgs::PointCloud2ConstIterator<float> input_z(message, "z");
      sensor_msgs::PointCloud2ConstIterator<float> input_rgb(message, "rgb");
      for (; input_x != input_x.end();
        ++input_x, ++input_y, ++input_z, ++input_rgb)
      {
        const double x = static_cast<double>(*input_x);
        const double y = static_cast<double>(*input_y);
        const double z = static_cast<double>(*input_z);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
          continue;
        }
        const float packed_rgb = *input_rgb;
        std::uint32_t rgb;
        std::memcpy(&rgb, &packed_rgb, sizeof(rgb));
        points.push_back(ColoredPoint{cv::Vec3d(x, y, z), rgb});
      }
      return points;
    }

    const std::uint8_t * cursor = message.data.data();
    const std::size_t step = message.point_step;
    for (std::size_t index = 0; index < point_total; ++index, cursor += step) {
      float x;
      float y;
      float z;
      std::uint32_t rgb;
      std::memcpy(&x, cursor + layout.x, sizeof(x));
      std::memcpy(&y, cursor + layout.y, sizeof(y));
      std::memcpy(&z, cursor + layout.z, sizeof(z));
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        continue;
      }
      std::memcpy(&rgb, cursor + layout.rgb, sizeof(rgb));
      points.push_back(
        ColoredPoint{
          cv::Vec3d(
            static_cast<double>(x), static_cast<double>(y),
            static_cast<double>(z)), rgb});
    }
    return points;
  }

  std::vector<std::size_t> select_ground_candidates(
    const std::vector<ColoredPoint> & points) const
  {
    std::vector<std::size_t> indices;
    indices.reserve(points.size());
    // Compare squared ranges: std::hypot is several times slower than a
    // multiply-add and its overflow guarantees are irrelevant at these scales.
    const double minimum_squared_range =
      ground_candidate_min_range_m_ * ground_candidate_min_range_m_;
    const double maximum_squared_range =
      ground_candidate_max_range_m_ * ground_candidate_max_range_m_;
    for (std::size_t index = 0; index < points.size(); ++index) {
      const cv::Vec3d & point = points[index].position;
      const double squared_range =
        point[0] * point[0] + point[2] * point[2];
      const double down = -expected_up_.dot(point);
      if (
        squared_range >= minimum_squared_range &&
        squared_range <= maximum_squared_range &&
        down >= ground_candidate_min_down_m_ &&
        down <= ground_candidate_max_down_m_)
      {
        indices.push_back(index);
      }
    }
    return indices;
  }

  std::vector<std::size_t> subsample_candidates(
    const std::vector<std::size_t> & candidate_indices) const
  {
    if (
      candidate_indices.size() <=
      static_cast<std::size_t>(max_ransac_points_))
    {
      return candidate_indices;
    }

    std::vector<std::size_t> sample;
    sample.reserve(static_cast<std::size_t>(max_ransac_points_));
    for (int index = 0; index < max_ransac_points_; ++index) {
      const std::size_t source_index =
        static_cast<std::size_t>(index) * candidate_indices.size() /
        static_cast<std::size_t>(max_ransac_points_);
      sample.push_back(candidate_indices[source_index]);
    }
    return sample;
  }

  bool orient_and_validate_plane(cv::Vec3d & normal, double & offset) const
  {
    const double normal_norm = cv::norm(normal);
    if (normal_norm < 1e-9) {
      return false;
    }
    normal /= normal_norm;
    offset /= normal_norm;
    if (normal.dot(expected_up_) < 0.0) {
      normal = -normal;
      offset = -offset;
    }
    if (normal.dot(expected_up_) < minimum_up_alignment_) {
      return false;
    }
    return
      offset >= min_plane_distance_from_origin_m_ &&
      offset <= max_plane_distance_from_origin_m_;
  }

  PlaneModel fit_plane_ransac(
    const std::vector<ColoredPoint> & points,
    const std::vector<std::size_t> & candidate_indices)
  {
    PlaneModel best;
    const auto sample_indices = subsample_candidates(candidate_indices);
    if (sample_indices.size() < 3) {
      return best;
    }

    std::uniform_int_distribution<std::size_t> distribution(
      0, sample_indices.size() - 1);
    for (int iteration = 0; iteration < ransac_iterations_; ++iteration) {
      const std::size_t first = distribution(random_engine_);
      const std::size_t second = distribution(random_engine_);
      const std::size_t third = distribution(random_engine_);
      if (first == second || first == third || second == third) {
        continue;
      }

      const cv::Vec3d & point_a =
        points[sample_indices[first]].position;
      const cv::Vec3d & point_b =
        points[sample_indices[second]].position;
      const cv::Vec3d & point_c =
        points[sample_indices[third]].position;
      cv::Vec3d normal =
        (point_b - point_a).cross(point_c - point_a);
      double offset = -normal.dot(point_a);
      if (!orient_and_validate_plane(normal, offset)) {
        continue;
      }

      std::size_t inliers = 0;
      double squared_error = 0.0;
      for (const std::size_t index : sample_indices) {
        const double distance =
          std::abs(normal.dot(points[index].position) + offset);
        if (distance <= ransac_distance_threshold_m_) {
          ++inliers;
          squared_error += distance * distance;
        }
      }
      if (
        inliers > best.inliers ||
        (inliers == best.inliers && squared_error < best.squared_error))
      {
        best.normal = normal;
        best.offset = offset;
        best.inliers = inliers;
        best.squared_error = squared_error;
        best.valid = true;
      }
    }

    if (!best.valid) {
      return best;
    }
    const double sample_ratio =
      static_cast<double>(best.inliers) /
      static_cast<double>(sample_indices.size());
    const std::size_t scaled_minimum_inliers = std::min(
      static_cast<std::size_t>(min_ground_inliers_),
      sample_indices.size());
    if (
      best.inliers < scaled_minimum_inliers ||
      sample_ratio < min_ground_inlier_ratio_)
    {
      best.valid = false;
      return best;
    }

    std::vector<std::size_t> inlier_indices;
    inlier_indices.reserve(candidate_indices.size());
    cv::Vec3d centroid(0.0, 0.0, 0.0);
    for (const std::size_t index : candidate_indices) {
      const cv::Vec3d & point = points[index].position;
      if (
        std::abs(best.normal.dot(point) + best.offset) <=
        ransac_distance_threshold_m_)
      {
        inlier_indices.push_back(index);
        centroid += point;
      }
    }
    if (inlier_indices.size() < 3) {
      best.valid = false;
      return best;
    }
    centroid *= 1.0 / static_cast<double>(inlier_indices.size());

    cv::Matx33d covariance = cv::Matx33d::zeros();
    for (const std::size_t index : inlier_indices) {
      const cv::Vec3d delta = points[index].position - centroid;
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          covariance(row, column) += delta[row] * delta[column];
        }
      }
    }

    cv::Mat eigenvalues;
    cv::Mat eigenvectors;
    if (!cv::eigen(cv::Mat(covariance), eigenvalues, eigenvectors)) {
      best.valid = false;
      return best;
    }
    cv::Vec3d refined_normal(
      eigenvectors.at<double>(2, 0),
      eigenvectors.at<double>(2, 1),
      eigenvectors.at<double>(2, 2));
    double refined_offset = -refined_normal.dot(centroid);
    if (!orient_and_validate_plane(refined_normal, refined_offset)) {
      best.valid = false;
      return best;
    }

    best.normal = refined_normal;
    best.offset = refined_offset;
    best.inliers = 0;
    best.squared_error = 0.0;
    for (const std::size_t index : candidate_indices) {
      const double distance = std::abs(
        best.normal.dot(points[index].position) + best.offset);
      if (distance <= ransac_distance_threshold_m_) {
        ++best.inliers;
        best.squared_error += distance * distance;
      }
    }
    const double full_ratio =
      static_cast<double>(best.inliers) /
      static_cast<double>(candidate_indices.size());
    best.valid =
      best.inliers >= static_cast<std::size_t>(min_ground_inliers_) &&
      full_ratio >= min_ground_inlier_ratio_;
    return best;
  }

  PointCloud2 make_cloud(
    const std_msgs::msg::Header & header,
    const std::vector<ColoredPoint> & points) const
  {
    PointCloud2 cloud;
    cloud.header = header;
    cloud.height = 1;
    cloud.is_dense = true;
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(points.size());

    // Write through the field offsets rather than assuming four packed floats:
    // setPointCloud2FieldsByString(2, "xyz", "rgb") pads the point to 32 bytes
    // and places rgb at offset 16.
    const CloudLayout layout = inspect_layout(cloud);
    if (!layout.packed) {
      sensor_msgs::PointCloud2Iterator<float> output_x(cloud, "x");
      sensor_msgs::PointCloud2Iterator<float> output_y(cloud, "y");
      sensor_msgs::PointCloud2Iterator<float> output_z(cloud, "z");
      sensor_msgs::PointCloud2Iterator<float> output_rgb(cloud, "rgb");
      for (const auto & point : points) {
        *output_x = static_cast<float>(point.position[0]);
        *output_y = static_cast<float>(point.position[1]);
        *output_z = static_cast<float>(point.position[2]);
        float packed_rgb;
        std::memcpy(&packed_rgb, &point.rgb, sizeof(packed_rgb));
        *output_rgb = packed_rgb;
        ++output_x;
        ++output_y;
        ++output_z;
        ++output_rgb;
      }
      return cloud;
    }

    std::uint8_t * cursor = cloud.data.data();
    const std::size_t step = cloud.point_step;
    for (const auto & point : points) {
      const float values[3] = {
        static_cast<float>(point.position[0]),
        static_cast<float>(point.position[1]),
        static_cast<float>(point.position[2])
      };
      std::memcpy(cursor + layout.x, &values[0], sizeof(float));
      std::memcpy(cursor + layout.y, &values[1], sizeof(float));
      std::memcpy(cursor + layout.z, &values[2], sizeof(float));
      std::memcpy(cursor + layout.rgb, &point.rgb, sizeof(point.rgb));
      cursor += step;
    }
    return cloud;
  }

  void voxel_downsample(std::vector<ColoredPoint> & points) const
  {
    if (obstacle_voxel_size_m_ <= 0.0 || points.empty()) {
      return;
    }

    const double inverse_voxel_size = 1.0 / obstacle_voxel_size_m_;
    std::unordered_set<VoxelKey, VoxelKeyHash> occupied_voxels;
    occupied_voxels.reserve(points.size());
    std::vector<ColoredPoint> downsampled;
    downsampled.reserve(points.size());
    for (const auto & point : points) {
      const VoxelKey key{
        static_cast<std::int64_t>(
          std::floor(point.position[0] * inverse_voxel_size)),
        static_cast<std::int64_t>(
          std::floor(point.position[1] * inverse_voxel_size)),
        static_cast<std::int64_t>(
          std::floor(point.position[2] * inverse_voxel_size))};
      if (occupied_voxels.insert(key).second) {
        downsampled.push_back(point);
      }
    }
    points = std::move(downsampled);
  }

  void publish_empty_outputs(const std_msgs::msg::Header & header)
  {
    obstacle_publisher_->publish(make_cloud(header, {}));
    if (ground_publisher_) {
      ground_publisher_->publish(make_cloud(header, {}));
    }
  }

  void pointcloud_callback(const PointCloud2::ConstSharedPtr message)
  {
    const auto start = std::chrono::steady_clock::now();
    // RANSAC over tens of thousands of points is the most expensive thing this
    // node does. Skip it entirely while nothing consumes the result.
    if (
      skip_when_unsubscribed_ &&
      obstacle_publisher_->get_subscription_count() == 0 &&
      obstacle_publisher_->get_intra_process_subscription_count() == 0 &&
      (!ground_publisher_ ||
      ground_publisher_->get_subscription_count() == 0))
    {
      ++idle_frame_count_;
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(),
        static_cast<std::uint64_t>(diagnostics_period_sec_ * 1000.0),
        "idle: no subscriber on %s, %zu clouds skipped",
        obstacle_topic_.c_str(), idle_frame_count_);
      return;
    }
    try {
      const auto points = parse_cloud(*message);
      const auto candidate_indices = select_ground_candidates(points);
      const PlaneModel plane = fit_plane_ransac(points, candidate_indices);
      if (!plane.valid) {
        publish_empty_outputs(message->header);
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(),
          static_cast<std::uint64_t>(diagnostics_period_sec_ * 1000.0),
          "No valid ground plane: input=%zu candidates=%zu",
          points.size(), candidate_indices.size());
        return;
      }

      std::vector<ColoredPoint> obstacles;
      std::vector<ColoredPoint> ground;
      obstacles.reserve(points.size() / 4);
      const bool publish_ground =
        ground_publisher_ &&
        ground_publisher_->get_subscription_count() > 0;
      if (publish_ground) {
        ground.reserve(plane.inliers);
      }
      const double minimum_squared_range =
        obstacle_min_range_m_ * obstacle_min_range_m_;
      const double maximum_squared_range =
        obstacle_max_range_m_ * obstacle_max_range_m_;
      for (const auto & point : points) {
        const double squared_range =
          point.position[0] * point.position[0] +
          point.position[2] * point.position[2];
        const double signed_height =
          plane.normal.dot(point.position) + plane.offset;
        if (
          squared_range >= minimum_squared_range &&
          squared_range <= maximum_squared_range &&
          signed_height >= obstacle_min_height_m_ &&
          signed_height <= obstacle_max_height_m_)
        {
          obstacles.push_back(point);
        }
        if (
          publish_ground &&
          std::abs(signed_height) <= ransac_distance_threshold_m_)
        {
          ground.push_back(point);
        }
      }

      const std::size_t raw_obstacle_count = obstacles.size();
      voxel_downsample(obstacles);
      obstacle_publisher_->publish(make_cloud(message->header, obstacles));
      if (publish_ground) {
        ground_publisher_->publish(make_cloud(message->header, ground));
      }

      const auto end = std::chrono::steady_clock::now();
      const double processing_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
      const double inlier_ratio = candidate_indices.empty() ? 0.0 :
        static_cast<double>(plane.inliers) /
        static_cast<double>(candidate_indices.size());
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(),
        static_cast<std::uint64_t>(diagnostics_period_sec_ * 1000.0),
        "ground=[%.4f %.4f %.4f %.4f] input=%zu candidates=%zu "
        "inliers=%zu(%.1f%%) obstacles=%zu->%zu processing=%.1f ms",
        plane.normal[0], plane.normal[1], plane.normal[2], plane.offset,
        points.size(), candidate_indices.size(), plane.inliers,
        100.0 * inlier_ratio, raw_obstacle_count, obstacles.size(),
        processing_ms);
    } catch (const std::exception & error) {
      publish_empty_outputs(message->header);
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Panorama ground segmentation failed: %s", error.what());
    }
  }

  std::string input_topic_;
  std::string obstacle_topic_;
  std::string ground_topic_;
  bool publish_ground_cloud_{true};
  cv::Vec3d expected_up_{0.0, -1.0, 0.0};
  int ransac_iterations_{160};
  int max_ransac_points_{30000};
  double ransac_distance_threshold_m_{0.05};
  double max_ground_tilt_deg_{25.0};
  int min_ground_inliers_{300};
  double min_ground_inlier_ratio_{0.03};
  double ground_candidate_min_range_m_{0.4};
  double ground_candidate_max_range_m_{8.0};
  double ground_candidate_min_down_m_{0.15};
  double ground_candidate_max_down_m_{2.5};
  double min_plane_distance_from_origin_m_{0.15};
  double max_plane_distance_from_origin_m_{2.5};
  double obstacle_min_height_m_{0.10};
  double obstacle_max_height_m_{2.0};
  double obstacle_min_range_m_{0.25};
  double obstacle_max_range_m_{8.0};
  double obstacle_voxel_size_m_{0.0};
  bool skip_when_unsubscribed_{true};
  double diagnostics_period_sec_{2.0};
  double minimum_up_alignment_{0.0};
  std::size_t idle_frame_count_{0};
  std::mt19937 random_engine_;

  rclcpp::Subscription<PointCloud2>::SharedPtr pointcloud_subscriber_;
  rclcpp::Publisher<PointCloud2>::SharedPtr obstacle_publisher_;
  rclcpp::Publisher<PointCloud2>::SharedPtr ground_publisher_;
};

}  // namespace panorama_stitcher

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<
      panorama_stitcher::PanoramaGroundSegmentationNode>());
  rclcpp::shutdown();
  return 0;
}
