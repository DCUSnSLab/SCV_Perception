#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmw/qos_profiles.h>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace panorama_stitcher
{

using CameraInfo = sensor_msgs::msg::CameraInfo;
using Image = sensor_msgs::msg::Image;
using PointCloud2 = sensor_msgs::msg::PointCloud2;
using RgbdSyncPolicy =
  message_filters::sync_policies::ApproximateTime<Image, Image, Image, Image>;

constexpr double kPi = 3.14159265358979323846;

struct CameraModel
{
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
  int width{0};
  int height{0};
  cv::Matx33d rotation{cv::Matx33d::eye()};
  cv::Vec3d translation{0.0, 0.0, 0.0};
  double min_angle{0.0};
  double max_angle{0.0};
  double min_vertical{0.0};
  double max_vertical{0.0};

  bool valid() const
  {
    return fx > 0.0 && fy > 0.0 && width > 0 && height > 0;
  }
};

struct ColoredPoint
{
  float x;
  float y;
  float z;
  uint32_t rgb;
};

class FrontRgbdFusionNode : public rclcpp::Node
{
public:
  FrontRgbdFusionNode()
  : Node("front_rgbd_fusion"),
    last_diagnostics_time_(std::chrono::steady_clock::now())
  {
    left_color_topic_ = declare_parameter<std::string>(
      "left_color_topic", "/front_left/front_left/color/image_raw");
    left_depth_topic_ = declare_parameter<std::string>(
      "left_depth_topic", "/front_left/front_left/aligned_depth_to_color/image_raw");
    left_info_topic_ = declare_parameter<std::string>(
      "left_camera_info_topic", "/front_left/front_left/color/camera_info");
    right_color_topic_ = declare_parameter<std::string>(
      "right_color_topic", "/front_right/front_right/color/image_raw");
    right_depth_topic_ = declare_parameter<std::string>(
      "right_depth_topic", "/front_right/front_right/aligned_depth_to_color/image_raw");
    right_info_topic_ = declare_parameter<std::string>(
      "right_camera_info_topic", "/front_right/front_right/color/camera_info");

    color_output_topic_ = declare_parameter<std::string>(
      "color_output_topic", "/parking/front/color_mosaic");
    depth_output_topic_ = declare_parameter<std::string>(
      "depth_output_topic", "/parking/front/depth_mosaic");
    cloud_output_topic_ = declare_parameter<std::string>(
      "pointcloud_output_topic", "/parking/front/points");
    output_frame_id_ = declare_parameter<std::string>(
      "output_frame_id", "front_rgbd_rig_optical_frame");

    sync_queue_size_ = static_cast<int>(std::max<int64_t>(
      declare_parameter<int>("sync_queue_size", 50), 4));
    sync_slop_ms_ = std::max(
      declare_parameter<double>("sync_slop_ms", 45.0), 1.0);
    max_publish_rate_hz_ = std::max(
      declare_parameter<double>("max_publish_rate_hz", 15.0), 0.0);
    input_images_rotated_180_ = declare_parameter<bool>(
      "input_images_rotated_180", true);
    rotate_aligned_depth_180_ = declare_parameter<bool>(
      "rotate_aligned_depth_180", false);
    pixel_preserving_output_ = declare_parameter<bool>(
      "pixel_preserving_output", true);
    right_x_offset_px_ = static_cast<int>(std::max<int64_t>(
      declare_parameter<int>("right_x_offset_px", 1900), 0));
    right_y_offset_px_ = declare_parameter<int>("right_y_offset_px", 17);

    projection_scale_ = std::clamp(
      declare_parameter<double>("projection_scale", 0.5), 0.1, 1.0);
    seam_angle_deg_ = declare_parameter<double>("seam_angle_deg", 0.0);
    seam_feather_px_ = static_cast<int>(std::max<int64_t>(
      declare_parameter<int>("seam_feather_px", 4), 0));
    depth_projection_stride_ = static_cast<int>(std::max<int64_t>(
      declare_parameter<int>("depth_projection_stride", 2), 1));
    pointcloud_stride_ = static_cast<int>(std::max<int64_t>(
      declare_parameter<int>("pointcloud_stride", 4),
      depth_projection_stride_));
    projection_splat_radius_px_ = static_cast<int>(std::clamp<int64_t>(
      declare_parameter<int>("projection_splat_radius_px", 0), 0, 2));
    depth_aware_color_ = declare_parameter<bool>(
      "depth_aware_color", true);
    depth_scale_m_ = std::max(
      declare_parameter<double>("depth_scale_m", 0.001), 1e-6);
    min_depth_m_ = std::max(
      declare_parameter<double>("min_depth_m", 0.20), 0.01);
    max_depth_m_ = std::max(
      declare_parameter<double>("max_depth_m", 15.0), min_depth_m_);

    exposure_compensation_ = declare_parameter<bool>(
      "enable_exposure_compensation", true);
    exposure_smoothing_ = std::clamp(
      declare_parameter<double>("exposure_smoothing", 0.15), 0.0, 1.0);
    min_exposure_gain_ = std::max(
      declare_parameter<double>("min_exposure_gain", 0.75), 0.01);
    max_exposure_gain_ = std::max(
      declare_parameter<double>("max_exposure_gain", 1.33),
      min_exposure_gain_);
    diagnostics_period_sec_ = std::max(
      declare_parameter<double>("diagnostics_period_sec", 2.0), 0.2);

    configure_camera(
      left_model_, "left",
      {0.0, 0.0, -32.0}, {-0.055, 0.0, 0.0},
      1369.7860107421875, 1369.6165771484375,
      967.3739013671875, 566.1657104492188);
    configure_camera(
      right_model_, "right",
      {0.0, 0.0, 32.0}, {0.055, 0.0, 0.0},
      1375.93896484375, 1376.0078125,
      962.9755859375, 539.9728393554688);

    const auto output_qos = rclcpp::SensorDataQoS().keep_last(2);
    color_publisher_ = create_publisher<Image>(
      color_output_topic_, output_qos);
    depth_publisher_ = create_publisher<Image>(
      depth_output_topic_, output_qos);
    cloud_publisher_ = create_publisher<PointCloud2>(
      cloud_output_topic_, output_qos);

    const auto info_qos =
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
    left_info_subscriber_ = create_subscription<CameraInfo>(
      left_info_topic_, info_qos,
      [this](const CameraInfo::ConstSharedPtr message) {
        update_intrinsics(left_model_, *message, "left");
      });
    right_info_subscriber_ = create_subscription<CameraInfo>(
      right_info_topic_, info_qos,
      [this](const CameraInfo::ConstSharedPtr message) {
        update_intrinsics(right_model_, *message, "right");
      });

    left_color_subscriber_.subscribe(
      this, left_color_topic_, rmw_qos_profile_sensor_data);
    left_depth_subscriber_.subscribe(
      this, left_depth_topic_, rmw_qos_profile_sensor_data);
    right_color_subscriber_.subscribe(
      this, right_color_topic_, rmw_qos_profile_sensor_data);
    right_depth_subscriber_.subscribe(
      this, right_depth_topic_, rmw_qos_profile_sensor_data);

    synchronizer_ =
      std::make_shared<message_filters::Synchronizer<RgbdSyncPolicy>>(
      RgbdSyncPolicy(sync_queue_size_),
      left_color_subscriber_, left_depth_subscriber_,
      right_color_subscriber_, right_depth_subscriber_);
    synchronizer_->setMaxIntervalDuration(
      rclcpp::Duration::from_seconds(sync_slop_ms_ / 1000.0));
    synchronizer_->registerCallback(
      std::bind(
        &FrontRgbdFusionNode::image_callback, this,
        std::placeholders::_1, std::placeholders::_2,
        std::placeholders::_3, std::placeholders::_4));

    RCLCPP_INFO(
      get_logger(),
      "Front RGB-D fusion ready: color=%s depth=%s cloud=%s frame=%s mode=%s",
      color_output_topic_.c_str(), depth_output_topic_.c_str(),
      cloud_output_topic_.c_str(), output_frame_id_.c_str(),
      pixel_preserving_output_ ? "pixel-preserving" : "cylindrical");
    RCLCPP_WARN(
      get_logger(),
      "Rig extrinsics currently use installation estimates. "
      "Tune left/right rotation_rpy_deg and translation_xyz_m after the mount is fixed.");
  }

private:
  static cv::Matx33d rotation_from_rpy_deg(
    const std::vector<double> & rpy_deg)
  {
    const double roll = rpy_deg[0] * kPi / 180.0;
    const double pitch = rpy_deg[1] * kPi / 180.0;
    const double yaw = rpy_deg[2] * kPi / 180.0;

    const cv::Matx33d yaw_y(
      std::cos(yaw), 0.0, std::sin(yaw),
      0.0, 1.0, 0.0,
      -std::sin(yaw), 0.0, std::cos(yaw));
    const cv::Matx33d pitch_x(
      1.0, 0.0, 0.0,
      0.0, std::cos(pitch), -std::sin(pitch),
      0.0, std::sin(pitch), std::cos(pitch));
    const cv::Matx33d roll_z(
      std::cos(roll), -std::sin(roll), 0.0,
      std::sin(roll), std::cos(roll), 0.0,
      0.0, 0.0, 1.0);
    return roll_z * pitch_x * yaw_y;
  }

  void configure_camera(
    CameraModel & model, const std::string & prefix,
    const std::vector<double> & default_rpy_deg,
    const std::vector<double> & default_translation,
    double default_fx, double default_fy,
    double default_cx, double default_cy)
  {
    auto rpy = declare_parameter<std::vector<double>>(
      prefix + "_rotation_rpy_deg", default_rpy_deg);
    auto translation = declare_parameter<std::vector<double>>(
      prefix + "_translation_xyz_m", default_translation);
    if (rpy.size() != 3 || translation.size() != 3) {
      throw std::runtime_error(
              prefix + " rotation and translation parameters must have 3 values");
    }
    model.rotation = rotation_from_rpy_deg(rpy);
    model.translation = cv::Vec3d(
      translation[0], translation[1], translation[2]);
    model.fx = declare_parameter<double>(prefix + "_fx", default_fx);
    model.fy = declare_parameter<double>(prefix + "_fy", default_fy);
    model.cx = declare_parameter<double>(prefix + "_cx", default_cx);
    model.cy = declare_parameter<double>(prefix + "_cy", default_cy);
    model.width = declare_parameter<int>(prefix + "_width", 1920);
    model.height = declare_parameter<int>(prefix + "_height", 1080);
    adjust_intrinsics_for_rotation(model);

    RCLCPP_INFO(
      get_logger(),
      "%s extrinsic rpy(deg)=%.2f/%.2f/%.2f xyz(m)=%.3f/%.3f/%.3f",
      prefix.c_str(), rpy[0], rpy[1], rpy[2],
      translation[0], translation[1], translation[2]);
  }

  void adjust_intrinsics_for_rotation(CameraModel & model) const
  {
    if (!input_images_rotated_180_) {
      return;
    }
    model.cx = static_cast<double>(model.width - 1) - model.cx;
    model.cy = static_cast<double>(model.height - 1) - model.cy;
  }

  void update_intrinsics(
    CameraModel & model, const CameraInfo & message,
    const char * camera_name)
  {
    CameraModel updated = model;
    updated.fx = message.k[0];
    updated.fy = message.k[4];
    updated.cx = message.k[2];
    updated.cy = message.k[5];
    updated.width = static_cast<int>(message.width);
    updated.height = static_cast<int>(message.height);
    adjust_intrinsics_for_rotation(updated);

    const bool changed =
      !model.valid() ||
      model.width != updated.width ||
      model.height != updated.height ||
      std::abs(model.fx - updated.fx) > 1e-6 ||
      std::abs(model.fy - updated.fy) > 1e-6 ||
      std::abs(model.cx - updated.cx) > 1e-6 ||
      std::abs(model.cy - updated.cy) > 1e-6;
    if (!changed) {
      return;
    }
    model = updated;
    projection_dirty_ = true;
    RCLCPP_INFO(
      get_logger(),
      "%s CameraInfo: %dx%d fx/fy=%.3f/%.3f cx/cy=%.3f/%.3f",
      camera_name, model.width, model.height,
      model.fx, model.fy, model.cx, model.cy);
  }

  static int64_t stamp_nanoseconds(
    const builtin_interfaces::msg::Time & stamp)
  {
    return static_cast<int64_t>(stamp.sec) * 1000000000LL +
           static_cast<int64_t>(stamp.nanosec);
  }

  static cv::Mat to_bgr(const Image::ConstSharedPtr & message)
  {
    return cv_bridge::toCvShare(
      message, sensor_msgs::image_encodings::BGR8)->image;
  }

  cv::Mat to_depth(const Image::ConstSharedPtr & message) const
  {
    const cv::Mat shared = cv_bridge::toCvShare(message)->image;
    if (!rotate_aligned_depth_180_) {
      return shared;
    }
    cv::Mat rotated;
    cv::rotate(shared, rotated, cv::ROTATE_180);
    return rotated;
  }

  double depth_metres(const cv::Mat & depth, int v, int u) const
  {
    if (depth.type() == CV_16UC1) {
      return static_cast<double>(depth.at<uint16_t>(v, u)) * depth_scale_m_;
    }
    if (depth.type() == CV_32FC1) {
      return static_cast<double>(depth.at<float>(v, u));
    }
    return 0.0;
  }

  static uint32_t pack_rgb(const cv::Vec3b & bgr)
  {
    return
      (static_cast<uint32_t>(bgr[2]) << 16) |
      (static_cast<uint32_t>(bgr[1]) << 8) |
      static_cast<uint32_t>(bgr[0]);
  }

  void update_camera_bounds(CameraModel & model) const
  {
    model.min_angle = std::numeric_limits<double>::infinity();
    model.max_angle = -std::numeric_limits<double>::infinity();
    model.min_vertical = std::numeric_limits<double>::infinity();
    model.max_vertical = -std::numeric_limits<double>::infinity();

    for (const double v : {0.0, static_cast<double>(model.height - 1)}) {
      for (const double u : {0.0, static_cast<double>(model.width - 1)}) {
        const cv::Vec3d camera_ray(
          (u - model.cx) / model.fx,
          (v - model.cy) / model.fy,
          1.0);
        const cv::Vec3d rig_ray = model.rotation * camera_ray;
        if (rig_ray[2] <= 0.0) {
          continue;
        }
        const double horizontal =
          std::hypot(rig_ray[0], rig_ray[2]);
        const double angle = std::atan2(rig_ray[0], rig_ray[2]);
        const double vertical = rig_ray[1] / horizontal;
        model.min_angle = std::min(model.min_angle, angle);
        model.max_angle = std::max(model.max_angle, angle);
        model.min_vertical = std::min(model.min_vertical, vertical);
        model.max_vertical = std::max(model.max_vertical, vertical);
      }
    }
  }

  void build_projection_if_needed(int width, int height)
  {
    if (!projection_dirty_ &&
      source_width_ == width && source_height_ == height)
    {
      return;
    }
    if (!left_model_.valid() || !right_model_.valid()) {
      throw std::runtime_error("camera intrinsics are not valid");
    }
    if (left_model_.width != width || left_model_.height != height ||
      right_model_.width != width || right_model_.height != height)
    {
      throw std::runtime_error(
              "color image dimensions do not match CameraInfo");
    }

    source_width_ = width;
    source_height_ = height;

    if (pixel_preserving_output_) {
      const int left_canvas_y = std::max(0, -right_y_offset_px_);
      const int right_canvas_y = std::max(0, right_y_offset_px_);
      const int common_top = std::max(left_canvas_y, right_canvas_y);
      const int common_bottom = std::min(
        left_canvas_y + height, right_canvas_y + height);
      if (common_bottom <= common_top) {
        throw std::runtime_error(
                "fixed pixel layout has no common vertical image region");
      }
      fixed_left_source_y_ = common_top - left_canvas_y;
      fixed_right_source_y_ = common_top - right_canvas_y;
      panorama_width_ = std::max(width, right_x_offset_px_ + width);
      panorama_height_ = common_bottom - common_top;
      fixed_overlap_start_x_ = std::max(0, right_x_offset_px_);
      fixed_overlap_end_x_ = std::min(width, right_x_offset_px_ + width);
      seam_x_ = fixed_overlap_start_x_ +
        std::max(0, fixed_overlap_end_x_ - fixed_overlap_start_x_) / 2;
      projection_dirty_ = false;
      RCLCPP_INFO(
        get_logger(),
        "Pixel-preserving output=%dx%d offset=(%d,%d) overlap=%dpx seam=%d",
        panorama_width_, panorama_height_,
        right_x_offset_px_, right_y_offset_px_,
        std::max(0, fixed_overlap_end_x_ - fixed_overlap_start_x_), seam_x_);
      return;
    }

    update_camera_bounds(left_model_);
    update_camera_bounds(right_model_);

    panorama_focal_px_ =
      0.5 * (left_model_.fx + right_model_.fx) * projection_scale_;
    panorama_min_angle_ =
      std::min(left_model_.min_angle, right_model_.min_angle);
    panorama_max_angle_ =
      std::max(left_model_.max_angle, right_model_.max_angle);
    panorama_min_vertical_ =
      std::min(left_model_.min_vertical, right_model_.min_vertical);
    panorama_max_vertical_ =
      std::max(left_model_.max_vertical, right_model_.max_vertical);
    overlap_min_angle_ =
      std::max(left_model_.min_angle, right_model_.min_angle);
    overlap_max_angle_ =
      std::min(left_model_.max_angle, right_model_.max_angle);

    panorama_width_ = static_cast<int>(std::ceil(
      (panorama_max_angle_ - panorama_min_angle_) *
      panorama_focal_px_)) + 1;
    panorama_height_ = static_cast<int>(std::ceil(
      (panorama_max_vertical_ - panorama_min_vertical_) *
      panorama_focal_px_)) + 1;
    if (panorama_width_ <= 0 || panorama_height_ <= 0) {
      throw std::runtime_error("invalid cylindrical panorama dimensions");
    }

    build_inverse_map(
      left_model_, left_map_x_, left_map_y_, left_mask_);
    build_inverse_map(
      right_model_, right_map_x_, right_map_y_, right_mask_);
    overlap_mask_ = left_mask_ & right_mask_;

    seam_x_ = static_cast<int>(std::lround(
      (seam_angle_deg_ * kPi / 180.0 - panorama_min_angle_) *
      panorama_focal_px_));
    seam_x_ = std::clamp(seam_x_, 0, panorama_width_ - 1);
    projection_dirty_ = false;

    RCLCPP_INFO(
      get_logger(),
      "Cylindrical output=%dx%d view=%.1f..%.1f deg "
      "overlap=%.1f..%.1f deg seam=%d",
      panorama_width_, panorama_height_,
      panorama_min_angle_ * 180.0 / kPi,
      panorama_max_angle_ * 180.0 / kPi,
      overlap_min_angle_ * 180.0 / kPi,
      overlap_max_angle_ * 180.0 / kPi,
      seam_x_);
  }

  void build_inverse_map(
    const CameraModel & model,
    cv::Mat & map_x, cv::Mat & map_y, cv::Mat & mask) const
  {
    map_x.create(panorama_height_, panorama_width_, CV_32FC1);
    map_y.create(panorama_height_, panorama_width_, CV_32FC1);
    mask = cv::Mat::zeros(
      panorama_height_, panorama_width_, CV_8UC1);
    const cv::Matx33d inverse_rotation = model.rotation.t();

    for (int y = 0; y < panorama_height_; ++y) {
      float * map_x_row = map_x.ptr<float>(y);
      float * map_y_row = map_y.ptr<float>(y);
      uint8_t * mask_row = mask.ptr<uint8_t>(y);
      const double vertical =
        panorama_min_vertical_ +
        static_cast<double>(y) / panorama_focal_px_;
      for (int x = 0; x < panorama_width_; ++x) {
        const double angle =
          panorama_min_angle_ +
          static_cast<double>(x) / panorama_focal_px_;
        const cv::Vec3d rig_ray(
          std::sin(angle), vertical, std::cos(angle));
        const cv::Vec3d camera_ray = inverse_rotation * rig_ray;
        if (camera_ray[2] <= 0.0) {
          map_x_row[x] = -1.0F;
          map_y_row[x] = -1.0F;
          continue;
        }
        const double source_x =
          model.fx * camera_ray[0] / camera_ray[2] + model.cx;
        const double source_y =
          model.fy * camera_ray[1] / camera_ray[2] + model.cy;
        map_x_row[x] = static_cast<float>(source_x);
        map_y_row[x] = static_cast<float>(source_y);
        if (source_x >= 0.0 && source_x <= source_width_ - 1.0 &&
          source_y >= 0.0 && source_y <= source_height_ - 1.0)
        {
          mask_row[x] = 255;
        }
      }
    }
  }

  cv::Vec3d estimate_right_gain(
    const cv::Mat & left, const cv::Mat & right)
  {
    if (!exposure_compensation_ ||
      cv::countNonZero(overlap_mask_) < 500)
    {
      return smoothed_gain_;
    }
    cv::Mat left_gray;
    cv::Mat right_gray;
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
    const cv::Mat valid_mask =
      overlap_mask_ &
      (left_gray > 25) & (left_gray < 235) &
      (right_gray > 25) & (right_gray < 235);
    if (cv::countNonZero(valid_mask) < 500) {
      return smoothed_gain_;
    }
    const cv::Scalar left_mean = cv::mean(left, valid_mask);
    const cv::Scalar right_mean = cv::mean(right, valid_mask);
    for (int channel = 0; channel < 3; ++channel) {
      const double measured = std::clamp(
        left_mean[channel] / std::max(right_mean[channel], 1.0),
        min_exposure_gain_, max_exposure_gain_);
      smoothed_gain_[channel] =
        (1.0 - exposure_smoothing_) * smoothed_gain_[channel] +
        exposure_smoothing_ * measured;
    }
    return smoothed_gain_;
  }

  static cv::Mat apply_gain(
    const cv::Mat & image, const cv::Vec3d & gain)
  {
    const cv::Mat transform = (
      cv::Mat_<double>(3, 4) <<
      gain[0], 0.0, 0.0, 0.0,
      0.0, gain[1], 0.0, 0.0,
      0.0, 0.0, gain[2], 0.0);
    cv::Mat adjusted;
    cv::transform(image, adjusted, transform);
    return adjusted;
  }

  cv::Mat blend_base_color(
    const cv::Mat & left, const cv::Mat & right) const
  {
    cv::Mat panorama = cv::Mat::zeros(
      panorama_height_, panorama_width_, CV_8UC3);
    const int blend_left = seam_x_ - seam_feather_px_;
    const int blend_right = seam_x_ + seam_feather_px_;

    for (int y = 0; y < panorama_height_; ++y) {
      const cv::Vec3b * left_row = left.ptr<cv::Vec3b>(y);
      const cv::Vec3b * right_row = right.ptr<cv::Vec3b>(y);
      const uint8_t * left_mask_row = left_mask_.ptr<uint8_t>(y);
      const uint8_t * right_mask_row = right_mask_.ptr<uint8_t>(y);
      cv::Vec3b * output_row = panorama.ptr<cv::Vec3b>(y);
      for (int x = 0; x < panorama_width_; ++x) {
        const bool left_valid = left_mask_row[x] != 0;
        const bool right_valid = right_mask_row[x] != 0;
        if (!left_valid && !right_valid) {
          continue;
        }
        if (!right_valid || (left_valid && x < blend_left)) {
          output_row[x] = left_row[x];
          continue;
        }
        if (!left_valid || x > blend_right) {
          output_row[x] = right_row[x];
          continue;
        }
        const double denominator = std::max(2 * seam_feather_px_, 1);
        const double right_weight = std::clamp(
          static_cast<double>(x - blend_left) / denominator, 0.0, 1.0);
        for (int channel = 0; channel < 3; ++channel) {
          output_row[x][channel] = cv::saturate_cast<uint8_t>(
            (1.0 - right_weight) * left_row[x][channel] +
            right_weight * right_row[x][channel]);
        }
      }
    }
    return panorama;
  }

  cv::Mat fixed_pixel_color(
    const cv::Mat & left, const cv::Mat & right,
    cv::Mat & adjusted_right)
  {
    const cv::Mat left_view = left.rowRange(
      fixed_left_source_y_, fixed_left_source_y_ + panorama_height_);
    const cv::Mat right_view = right.rowRange(
      fixed_right_source_y_, fixed_right_source_y_ + panorama_height_);
    const int overlap_width =
      std::max(0, fixed_overlap_end_x_ - fixed_overlap_start_x_);

    cv::Vec3d gain = smoothed_gain_;
    if (overlap_width > 0) {
      const cv::Rect left_overlap(
        fixed_overlap_start_x_, 0, overlap_width, panorama_height_);
      const cv::Rect right_overlap(
        fixed_overlap_start_x_ - right_x_offset_px_,
        0, overlap_width, panorama_height_);
      overlap_mask_ = cv::Mat(
        panorama_height_, overlap_width, CV_8UC1, cv::Scalar(255));
      gain = estimate_right_gain(
        left_view(left_overlap), right_view(right_overlap));
    }
    adjusted_right = apply_gain(right, gain);
    const cv::Mat adjusted_right_view = adjusted_right.rowRange(
      fixed_right_source_y_, fixed_right_source_y_ + panorama_height_);

    cv::Mat mosaic = cv::Mat::zeros(
      panorama_height_, panorama_width_, CV_8UC3);
    left_view.copyTo(mosaic(cv::Rect(
      0, 0, left_view.cols, panorama_height_)));
    adjusted_right_view.copyTo(mosaic(cv::Rect(
      right_x_offset_px_, 0, adjusted_right_view.cols, panorama_height_)));

    if (overlap_width <= 0) {
      return mosaic;
    }
    const int blend_left = std::max(
      fixed_overlap_start_x_, seam_x_ - seam_feather_px_);
    const int blend_right = std::min(
      fixed_overlap_end_x_ - 1, seam_x_ + seam_feather_px_);
    const double denominator = std::max(blend_right - blend_left, 1);
    for (int y = 0; y < panorama_height_; ++y) {
      const cv::Vec3b * left_row = left_view.ptr<cv::Vec3b>(y);
      const cv::Vec3b * right_row = adjusted_right_view.ptr<cv::Vec3b>(y);
      cv::Vec3b * output_row = mosaic.ptr<cv::Vec3b>(y);
      for (int x = fixed_overlap_start_x_;
        x < fixed_overlap_end_x_; ++x)
      {
        const int right_x = x - right_x_offset_px_;
        if (x < blend_left) {
          output_row[x] = left_row[x];
          continue;
        }
        if (x > blend_right) {
          output_row[x] = right_row[right_x];
          continue;
        }
        const double right_weight = std::clamp(
          static_cast<double>(x - blend_left) / denominator, 0.0, 1.0);
        for (int channel = 0; channel < 3; ++channel) {
          output_row[x][channel] = cv::saturate_cast<uint8_t>(
            (1.0 - right_weight) * left_row[x][channel] +
            right_weight * right_row[right_x][channel]);
        }
      }
    }
    return mosaic;
  }

  cv::Mat depth_to_metres_image(const cv::Mat & depth) const
  {
    cv::Mat metres;
    if (depth.type() == CV_16UC1) {
      depth.convertTo(metres, CV_32FC1, depth_scale_m_);
    } else if (depth.type() == CV_32FC1) {
      metres = depth.clone();
    } else {
      throw std::runtime_error("depth encoding must be 16UC1 or 32FC1");
    }
    const float invalid = std::numeric_limits<float>::quiet_NaN();
    for (int y = 0; y < metres.rows; ++y) {
      float * row = metres.ptr<float>(y);
      for (int x = 0; x < metres.cols; ++x) {
        if (!std::isfinite(row[x]) ||
          row[x] < min_depth_m_ || row[x] > max_depth_m_)
        {
          row[x] = invalid;
        }
      }
    }
    return metres;
  }

  cv::Mat fixed_pixel_depth(
    const cv::Mat & left_depth, const cv::Mat & right_depth) const
  {
    const cv::Mat left_metres = depth_to_metres_image(left_depth);
    const cv::Mat right_metres = depth_to_metres_image(right_depth);
    const cv::Mat left_view = left_metres.rowRange(
      fixed_left_source_y_, fixed_left_source_y_ + panorama_height_);
    const cv::Mat right_view = right_metres.rowRange(
      fixed_right_source_y_, fixed_right_source_y_ + panorama_height_);
    cv::Mat mosaic(
      panorama_height_, panorama_width_, CV_32FC1,
      cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    left_view.copyTo(mosaic(cv::Rect(
      0, 0, left_view.cols, panorama_height_)));
    right_view.copyTo(mosaic(cv::Rect(
      right_x_offset_px_, 0, right_view.cols, panorama_height_)));

    if (fixed_overlap_end_x_ > fixed_overlap_start_x_) {
      const int left_width = std::max(
        0, seam_x_ - fixed_overlap_start_x_);
      if (left_width > 0) {
        left_view(cv::Rect(
          fixed_overlap_start_x_, 0, left_width, panorama_height_)).copyTo(
          mosaic(cv::Rect(
            fixed_overlap_start_x_, 0, left_width, panorama_height_)));
      }
    }
    return mosaic;
  }

  void append_cloud_camera(
    const cv::Mat & color, const cv::Mat & depth,
    const CameraModel & model,
    std::vector<ColoredPoint> * cloud_points,
    size_t & valid_depth_points) const
  {
    if (depth.size() != color.size()) {
      throw std::runtime_error(
              "aligned depth and color dimensions do not match");
    }
    valid_depth_points = 0;
    for (int v = 0; v < source_height_; v += pointcloud_stride_) {
      const cv::Vec3b * color_row = color.ptr<cv::Vec3b>(v);
      for (int u = 0; u < source_width_; u += pointcloud_stride_) {
        const double depth_m = depth_metres(depth, v, u);
        if (!std::isfinite(depth_m) ||
          depth_m < min_depth_m_ || depth_m > max_depth_m_)
        {
          continue;
        }
        const cv::Vec3d camera_point(
          (static_cast<double>(u) - model.cx) / model.fx * depth_m,
          (static_cast<double>(v) - model.cy) / model.fy * depth_m,
          depth_m);
        const cv::Vec3d rig_point =
          model.rotation * camera_point + model.translation;
        if (cloud_points != nullptr) {
          cloud_points->push_back(ColoredPoint{
              static_cast<float>(rig_point[0]),
              static_cast<float>(rig_point[1]),
              static_cast<float>(rig_point[2]),
              pack_rgb(color_row[u])
            });
        }
        ++valid_depth_points;
      }
    }
  }

  void project_camera(
    const cv::Mat & color, const cv::Mat & depth,
    const CameraModel & model,
    cv::Mat & z_buffer, cv::Mat & depth_mosaic,
    cv::Mat & projected_color, cv::Mat & projected_mask,
    std::vector<ColoredPoint> * cloud_points,
    size_t & valid_depth_points) const
  {
    if (depth.size() != color.size()) {
      throw std::runtime_error(
              "aligned depth and color dimensions do not match");
    }
    if (depth.type() != CV_16UC1 && depth.type() != CV_32FC1) {
      throw std::runtime_error("depth encoding must be 16UC1 or 32FC1");
    }

    for (int v = 0; v < source_height_; v += depth_projection_stride_) {
      const cv::Vec3b * color_row = color.ptr<cv::Vec3b>(v);
      for (int u = 0; u < source_width_; u += depth_projection_stride_) {
        const double depth_m = depth_metres(depth, v, u);
        if (!std::isfinite(depth_m) ||
          depth_m < min_depth_m_ || depth_m > max_depth_m_)
        {
          continue;
        }

        const cv::Vec3d camera_point(
          (static_cast<double>(u) - model.cx) / model.fx * depth_m,
          (static_cast<double>(v) - model.cy) / model.fy * depth_m,
          depth_m);
        const cv::Vec3d rig_point =
          model.rotation * camera_point + model.translation;
        const double horizontal =
          std::hypot(rig_point[0], rig_point[2]);
        if (horizontal <= 1e-6 || rig_point[2] <= 0.0) {
          continue;
        }
        const double angle = std::atan2(rig_point[0], rig_point[2]);
        const double vertical = rig_point[1] / horizontal;
        const int output_x = static_cast<int>(std::lround(
          (angle - panorama_min_angle_) * panorama_focal_px_));
        const int output_y = static_cast<int>(std::lround(
          (vertical - panorama_min_vertical_) * panorama_focal_px_));
        if (output_x < 0 || output_x >= panorama_width_ ||
          output_y < 0 || output_y >= panorama_height_)
        {
          continue;
        }

        const float range_m = static_cast<float>(cv::norm(rig_point));
        for (int dy = -projection_splat_radius_px_;
          dy <= projection_splat_radius_px_; ++dy)
        {
          const int y = output_y + dy;
          if (y < 0 || y >= panorama_height_) {
            continue;
          }
          for (int dx = -projection_splat_radius_px_;
            dx <= projection_splat_radius_px_; ++dx)
          {
            const int x = output_x + dx;
            if (x < 0 || x >= panorama_width_) {
              continue;
            }
            float & previous_range = z_buffer.at<float>(y, x);
            if (range_m < previous_range) {
              previous_range = range_m;
              depth_mosaic.at<float>(y, x) = range_m;
              projected_color.at<cv::Vec3b>(y, x) = color_row[u];
              projected_mask.at<uint8_t>(y, x) = 255;
            }
          }
        }

        if (cloud_points != nullptr &&
          (u % pointcloud_stride_) == 0 &&
          (v % pointcloud_stride_) == 0)
        {
          cloud_points->push_back(ColoredPoint{
              static_cast<float>(rig_point[0]),
              static_cast<float>(rig_point[1]),
              static_cast<float>(rig_point[2]),
              pack_rgb(color_row[u])
            });
        }
        ++valid_depth_points;
      }
    }
  }

  void publish_cloud(
    const std_msgs::msg::Header & header,
    const std::vector<ColoredPoint> & points)
  {
    PointCloud2 cloud;
    cloud.header = header;
    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(points.size());
    cloud.is_bigendian = false;
    cloud.is_dense = false;

    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(points.size());
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_rgb(cloud, "rgb");
    for (const auto & point : points) {
      *iter_x = point.x;
      *iter_y = point.y;
      *iter_z = point.z;
      float rgb_float;
      std::memcpy(&rgb_float, &point.rgb, sizeof(float));
      *iter_rgb = rgb_float;
      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_rgb;
    }
    cloud_publisher_->publish(cloud);
  }

  void image_callback(
    const Image::ConstSharedPtr & left_color_message,
    const Image::ConstSharedPtr & left_depth_message,
    const Image::ConstSharedPtr & right_color_message,
    const Image::ConstSharedPtr & right_depth_message)
  {
    const auto callback_start = std::chrono::steady_clock::now();
    if (max_publish_rate_hz_ > 0.0 && last_publish_time_.time_since_epoch().count() > 0) {
      const double elapsed =
        std::chrono::duration<double>(callback_start - last_publish_time_).count();
      if (elapsed < 1.0 / max_publish_rate_hz_) {
        return;
      }
    }
    last_publish_time_ = callback_start;

    try {
      const cv::Mat left_color = to_bgr(left_color_message);
      const cv::Mat right_color = to_bgr(right_color_message);
      const cv::Mat left_depth = to_depth(left_depth_message);
      const cv::Mat right_depth = to_depth(right_depth_message);
      build_projection_if_needed(left_color.cols, left_color.rows);

      const bool publish_cloud_now =
        cloud_publisher_->get_subscription_count() > 0;
      std::vector<ColoredPoint> cloud_points;
      if (publish_cloud_now) {
        const size_t estimate =
          2ULL * static_cast<size_t>(source_width_ / pointcloud_stride_ + 1) *
          static_cast<size_t>(source_height_ / pointcloud_stride_ + 1);
        cloud_points.reserve(estimate);
      }
      size_t left_valid_depth = 0;
      size_t right_valid_depth = 0;

      cv::Mat color_mosaic;
      cv::Mat depth_mosaic;
      cv::Mat adjusted_right;
      if (pixel_preserving_output_) {
        color_mosaic = fixed_pixel_color(
          left_color, right_color, adjusted_right);
        depth_mosaic = fixed_pixel_depth(left_depth, right_depth);
        append_cloud_camera(
          left_color, left_depth, left_model_,
          publish_cloud_now ? &cloud_points : nullptr, left_valid_depth);
        append_cloud_camera(
          adjusted_right, right_depth, right_model_,
          publish_cloud_now ? &cloud_points : nullptr, right_valid_depth);
      } else {
        cv::Mat left_base;
        cv::Mat right_base;
        cv::remap(
          left_color, left_base, left_map_x_, left_map_y_,
          cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        cv::remap(
          right_color, right_base, right_map_x_, right_map_y_,
          cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        const cv::Vec3d gain = estimate_right_gain(left_base, right_base);
        adjusted_right = apply_gain(right_color, gain);
        cv::remap(
          adjusted_right, right_base, right_map_x_, right_map_y_,
          cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        color_mosaic = blend_base_color(left_base, right_base);

        cv::Mat z_buffer(
          panorama_height_, panorama_width_, CV_32FC1,
          cv::Scalar(std::numeric_limits<float>::infinity()));
        depth_mosaic = cv::Mat(
          panorama_height_, panorama_width_, CV_32FC1,
          cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
        cv::Mat projected_color = cv::Mat::zeros(
          panorama_height_, panorama_width_, CV_8UC3);
        cv::Mat projected_mask = cv::Mat::zeros(
          panorama_height_, panorama_width_, CV_8UC1);
        project_camera(
          left_color, left_depth, left_model_,
          z_buffer, depth_mosaic, projected_color, projected_mask,
          publish_cloud_now ? &cloud_points : nullptr, left_valid_depth);
        project_camera(
          adjusted_right, right_depth, right_model_,
          z_buffer, depth_mosaic, projected_color, projected_mask,
          publish_cloud_now ? &cloud_points : nullptr, right_valid_depth);
        if (depth_aware_color_) {
          const cv::Mat overlap_depth_mask = projected_mask & overlap_mask_;
          projected_color.copyTo(color_mosaic, overlap_depth_mask);
        }
      }

      const std::array<Image::ConstSharedPtr, 4> messages = {
        left_color_message, left_depth_message,
        right_color_message, right_depth_message
      };
      auto newest_message = messages.front();
      int64_t minimum_stamp = stamp_nanoseconds(messages.front()->header.stamp);
      int64_t maximum_stamp = minimum_stamp;
      for (const auto & message : messages) {
        const int64_t stamp = stamp_nanoseconds(message->header.stamp);
        minimum_stamp = std::min(minimum_stamp, stamp);
        if (stamp > maximum_stamp) {
          maximum_stamp = stamp;
          newest_message = message;
        }
      }

      std_msgs::msg::Header output_header = newest_message->header;
      output_header.frame_id = output_frame_id_;
      color_publisher_->publish(
        *cv_bridge::CvImage(
          output_header, sensor_msgs::image_encodings::BGR8,
          color_mosaic).toImageMsg());
      depth_publisher_->publish(
        *cv_bridge::CvImage(
          output_header, sensor_msgs::image_encodings::TYPE_32FC1,
          depth_mosaic).toImageMsg());
      if (publish_cloud_now) {
        publish_cloud(output_header, cloud_points);
      }

      const auto callback_end = std::chrono::steady_clock::now();
      const double sync_span_ms =
        static_cast<double>(maximum_stamp - minimum_stamp) / 1e6;
      sync_span_sum_ms_ += sync_span_ms;
      sync_span_max_ms_ = std::max(sync_span_max_ms_, sync_span_ms);
      processing_time_sum_ms_ +=
        std::chrono::duration<double, std::milli>(
        callback_end - callback_start).count();
      last_left_valid_depth_ = left_valid_depth;
      last_right_valid_depth_ = right_valid_depth;
      last_cloud_points_ = cloud_points.size();
      ++frame_count_;
      ++diagnostic_frame_count_;
      maybe_log_diagnostics(callback_end);
    } catch (const cv_bridge::Exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "cv_bridge conversion failed: %s", error.what());
    } catch (const cv::Exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "OpenCV RGB-D fusion failed: %s", error.what());
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Front RGB-D fusion failed: %s", error.what());
    }
  }

  void maybe_log_diagnostics(
    const std::chrono::steady_clock::time_point & now)
  {
    const double elapsed =
      std::chrono::duration<double>(now - last_diagnostics_time_).count();
    if (elapsed < diagnostics_period_sec_ || diagnostic_frame_count_ == 0) {
      return;
    }
    const double count = static_cast<double>(diagnostic_frame_count_);
    RCLCPP_INFO(
      get_logger(),
      "output=%dx%d fps=%.1f processing=%.1f ms "
      "sync(avg/max)=%.1f/%.1f ms valid_depth(L/R)=%zu/%zu "
      "cloud=%zu gain(BGR)=%.2f/%.2f/%.2f total=%zu",
      panorama_width_, panorama_height_,
      count / elapsed, processing_time_sum_ms_ / count,
      sync_span_sum_ms_ / count, sync_span_max_ms_,
      last_left_valid_depth_, last_right_valid_depth_,
      last_cloud_points_,
      smoothed_gain_[0], smoothed_gain_[1], smoothed_gain_[2],
      frame_count_);

    last_diagnostics_time_ = now;
    diagnostic_frame_count_ = 0;
    sync_span_sum_ms_ = 0.0;
    sync_span_max_ms_ = 0.0;
    processing_time_sum_ms_ = 0.0;
  }

  std::string left_color_topic_;
  std::string left_depth_topic_;
  std::string left_info_topic_;
  std::string right_color_topic_;
  std::string right_depth_topic_;
  std::string right_info_topic_;
  std::string color_output_topic_;
  std::string depth_output_topic_;
  std::string cloud_output_topic_;
  std::string output_frame_id_;

  int sync_queue_size_{50};
  double sync_slop_ms_{45.0};
  double max_publish_rate_hz_{15.0};
  bool input_images_rotated_180_{true};
  bool rotate_aligned_depth_180_{false};
  bool pixel_preserving_output_{true};
  int right_x_offset_px_{1900};
  int right_y_offset_px_{17};
  double projection_scale_{0.5};
  double seam_angle_deg_{0.0};
  int seam_feather_px_{4};
  int depth_projection_stride_{2};
  int pointcloud_stride_{4};
  int projection_splat_radius_px_{0};
  bool depth_aware_color_{true};
  double depth_scale_m_{0.001};
  double min_depth_m_{0.20};
  double max_depth_m_{15.0};
  bool exposure_compensation_{true};
  double exposure_smoothing_{0.15};
  double min_exposure_gain_{0.75};
  double max_exposure_gain_{1.33};
  double diagnostics_period_sec_{2.0};

  CameraModel left_model_;
  CameraModel right_model_;
  bool projection_dirty_{true};
  int source_width_{0};
  int source_height_{0};
  int panorama_width_{0};
  int panorama_height_{0};
  int seam_x_{0};
  int fixed_left_source_y_{0};
  int fixed_right_source_y_{0};
  int fixed_overlap_start_x_{0};
  int fixed_overlap_end_x_{0};
  double panorama_focal_px_{0.0};
  double panorama_min_angle_{0.0};
  double panorama_max_angle_{0.0};
  double panorama_min_vertical_{0.0};
  double panorama_max_vertical_{0.0};
  double overlap_min_angle_{0.0};
  double overlap_max_angle_{0.0};
  cv::Mat left_map_x_;
  cv::Mat left_map_y_;
  cv::Mat right_map_x_;
  cv::Mat right_map_y_;
  cv::Mat left_mask_;
  cv::Mat right_mask_;
  cv::Mat overlap_mask_;
  cv::Vec3d smoothed_gain_{1.0, 1.0, 1.0};

  message_filters::Subscriber<Image> left_color_subscriber_;
  message_filters::Subscriber<Image> left_depth_subscriber_;
  message_filters::Subscriber<Image> right_color_subscriber_;
  message_filters::Subscriber<Image> right_depth_subscriber_;
  std::shared_ptr<message_filters::Synchronizer<RgbdSyncPolicy>> synchronizer_;
  rclcpp::Subscription<CameraInfo>::SharedPtr left_info_subscriber_;
  rclcpp::Subscription<CameraInfo>::SharedPtr right_info_subscriber_;
  rclcpp::Publisher<Image>::SharedPtr color_publisher_;
  rclcpp::Publisher<Image>::SharedPtr depth_publisher_;
  rclcpp::Publisher<PointCloud2>::SharedPtr cloud_publisher_;

  std::chrono::steady_clock::time_point last_publish_time_{};
  std::chrono::steady_clock::time_point last_diagnostics_time_;
  size_t frame_count_{0};
  size_t diagnostic_frame_count_{0};
  size_t last_left_valid_depth_{0};
  size_t last_right_valid_depth_{0};
  size_t last_cloud_points_{0};
  double sync_span_sum_ms_{0.0};
  double sync_span_max_ms_{0.0};
  double processing_time_sum_ms_{0.0};
};

}  // namespace panorama_stitcher

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<panorama_stitcher::FrontRgbdFusionNode>());
  rclcpp::shutdown();
  return 0;
}
