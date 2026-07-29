#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmw/qos_profiles.h>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#ifdef PANORAMA_WITH_CUDA
#include "cuda_panorama_backend.hpp"
#endif

namespace panorama_stitcher
{

using CameraInfo = sensor_msgs::msg::CameraInfo;
using Image = sensor_msgs::msg::Image;
constexpr double kPi = 3.14159265358979323846;

struct CameraModel
{
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
  int width{0};
  int height{0};
  cv::Matx33d rotation_camera_to_rig{cv::Matx33d::eye()};
  cv::Vec3d translation_camera_in_rig{0.0, 0.0, 0.0};

  bool valid() const
  {
    return fx > 0.0 && fy > 0.0 && width > 0 && height > 0;
  }
};

struct ProjectedDepth
{
  cv::Mat color;
  cv::Mat range;
  cv::Mat mask;
  size_t accepted_points{0};
};

class RgbdPanoramaStitcherNode : public rclcpp::Node
{
public:
  RgbdPanoramaStitcherNode()
  : Node("panorama_stitcher"),
    last_diagnostics_time_(std::chrono::steady_clock::now())
  {
    left_color_topic_ = declare_parameter<std::string>(
      "left_color_topic", "/front/front/color/image_raw");
    left_depth_topic_ = declare_parameter<std::string>(
      "left_depth_topic", "/front/front/aligned_depth_to_color/image_raw");
    left_camera_info_topic_ = declare_parameter<std::string>(
      "left_camera_info_topic", "/front/front/color/camera_info");
    right_color_topic_ = declare_parameter<std::string>(
      "right_color_topic", "/camera/camera/color/image_raw");
    right_depth_topic_ = declare_parameter<std::string>(
      "right_depth_topic", "/camera/camera/aligned_depth_to_color/image_raw");
    right_camera_info_topic_ = declare_parameter<std::string>(
      "right_camera_info_topic", "/camera/camera/color/camera_info");
    output_topic_ = declare_parameter<std::string>(
      "output_topic", "/panorama/image_raw");
    validity_topic_ = declare_parameter<std::string>(
      "validity_topic", "/panorama/validity");
    range_topic_ = declare_parameter<std::string>(
      "range_topic", "/panorama/range");
    output_frame_id_ = declare_parameter<std::string>(
      "output_frame_id", "panorama_optical_frame");
    publish_auxiliary_outputs_ = declare_parameter<bool>(
      "publish_auxiliary_outputs", false);
    publisher_best_effort_ = declare_parameter<bool>(
      "publisher_best_effort", false);

    sync_queue_size_ = declare_parameter<int>("sync_queue_size", 50);
    sync_slop_ms_ = declare_parameter<double>("sync_slop_ms", 45.0);
    input_images_rotated_180_ = declare_parameter<bool>(
      "input_images_rotated_180", true);
    rotate_color_180_ = declare_parameter<bool>(
      "rotate_color_180", false);
    rotate_aligned_depth_180_ = declare_parameter<bool>(
      "rotate_aligned_depth_180", false);

    const double half_yaw_deg = declare_parameter<double>(
      "camera_half_yaw_deg", 32.0);
    baseline_m_ = declare_parameter<double>("camera_baseline_m", 0.10);
    set_default_yaw_extrinsics(
      left_model_, -half_yaw_deg * kPi / 180.0, -baseline_m_ * 0.5);
    set_default_yaw_extrinsics(
      right_model_, half_yaw_deg * kPi / 180.0, baseline_m_ * 0.5);
    set_parameter_camera_extrinsics(left_model_, "left");
    set_parameter_camera_extrinsics(right_model_, "right");

    projection_scale_ = declare_parameter<double>("projection_scale", 0.5);
    projection_model_ = declare_parameter<std::string>(
      "projection_model", "cylindrical");
    color_reference_plane_z_m_ = declare_parameter<double>(
      "color_reference_plane_z_m", 0.0);
    rectilinear_width_ = declare_parameter<int>(
      "rectilinear_width", 3754);
    rectilinear_height_ = declare_parameter<int>(
      "rectilinear_height", 1071);
    rectilinear_fy_px_ = declare_parameter<double>(
      "rectilinear_fy_px", 1373.0);
    rectilinear_auto_height_ = declare_parameter<bool>(
      "rectilinear_auto_height", false);
    depth_scale_m_ = declare_parameter<double>("depth_scale_m", 0.001);
    min_depth_m_ = declare_parameter<double>("min_depth_m", 0.20);
    max_depth_m_ = declare_parameter<double>("max_depth_m", 15.0);
    depth_overlap_margin_deg_ = declare_parameter<double>(
      "depth_overlap_margin_deg", 2.0);
    full_depth_reprojection_ = declare_parameter<bool>(
      "full_depth_reprojection", false);
    depth_discontinuity_abs_m_ = declare_parameter<double>(
      "depth_discontinuity_abs_m", 0.08);
    depth_discontinuity_relative_ = declare_parameter<double>(
      "depth_discontinuity_relative", 0.04);
    depth_splat_radius_px_ = declare_parameter<int>(
      "depth_splat_radius_px", 1);
    depth_edge_splat_radius_px_ = declare_parameter<int>(
      "depth_edge_splat_radius_px", 0);
    projected_hole_radius_px_ = declare_parameter<int>(
      "projected_hole_radius_px", 0);
    allow_color_fallback_ = declare_parameter<bool>(
      "allow_color_fallback", true);
    seam_angle_deg_ = declare_parameter<double>("seam_angle_deg", 0.0);
    auto_seam_center_ = declare_parameter<bool>(
      "auto_seam_center", false);
    seam_feather_px_ = declare_parameter<int>("seam_feather_px", 2);
    depth_aware_color_ = declare_parameter<bool>(
      "depth_aware_color", true);
    render_depth_reprojected_color_ = declare_parameter<bool>(
      "render_depth_reprojected_color", true);
    use_rgbd_synchronization_ = declare_parameter<bool>(
      "use_rgbd_synchronization", true);
    depth_temporal_stabilization_ = declare_parameter<bool>(
      "depth_temporal_stabilization", true);
    depth_temporal_alpha_ = declare_parameter<double>(
      "depth_temporal_alpha", 0.35);
    depth_temporal_reset_m_ = declare_parameter<double>(
      "depth_temporal_reset_m", 0.05);
    depth_median_kernel_ = declare_parameter<int>(
      "depth_median_kernel", 3);
    occlusion_switch_margin_m_ = declare_parameter<double>(
      "occlusion_switch_margin_m", 0.05);
    prefer_seam_camera_when_both_depth_valid_ = declare_parameter<bool>(
      "prefer_seam_camera_when_both_depth_valid", false);
    exposure_compensation_ = declare_parameter<bool>(
      "enable_exposure_compensation", true);
    exposure_smoothing_ = declare_parameter<double>(
      "exposure_smoothing", 0.15);
    min_exposure_gain_ = declare_parameter<double>(
      "min_exposure_gain", 0.75);
    max_exposure_gain_ = declare_parameter<double>(
      "max_exposure_gain", 1.33);
    diagnostics_period_sec_ = declare_parameter<double>(
      "diagnostics_period_sec", 2.0);
    use_cuda_ = declare_parameter<bool>("use_cuda", true);

    set_parameter_camera_model(
      left_model_, "left", 1369.7860107421875, 1369.6165771484375,
      967.3739013671875, 566.1657104492188);
    set_parameter_camera_model(
      right_model_, "right", 1375.93896484375, 1376.0078125,
      962.9755859375, 539.9728393554688);
    validate_parameters();

#ifdef PANORAMA_WITH_CUDA
    if (use_cuda_) {
      std::string cuda_description;
      if (CudaPanoramaBackend::runtime_available(cuda_description)) {
        cuda_backend_ = std::make_unique<CudaPanoramaBackend>();
        RCLCPP_INFO(
          get_logger(), "CUDA panorama enabled: %s",
          cuda_description.c_str());
      } else {
        RCLCPP_WARN(
          get_logger(),
          "CUDA requested but unavailable (%s); using CPU fallback",
          cuda_description.c_str());
      }
    }
#else
    if (use_cuda_) {
      RCLCPP_WARN(
        get_logger(),
        "CUDA requested but this binary was built without CUDA; "
        "using CPU fallback");
    }
#endif

    auto output_qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile();
    if (publisher_best_effort_) {
      output_qos.best_effort();
    } else {
      output_qos.reliable();
    }
    output_publisher_ = create_publisher<Image>(output_topic_, output_qos);
    if (publish_auxiliary_outputs_) {
      validity_publisher_ = create_publisher<Image>(
        validity_topic_, output_qos);
      range_publisher_ = create_publisher<Image>(
        range_topic_, output_qos);
    }

    const auto camera_info_qos =
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
    left_camera_info_subscriber_ = create_subscription<CameraInfo>(
      left_camera_info_topic_, camera_info_qos,
      [this](const CameraInfo::ConstSharedPtr message) {
        const std::lock_guard<std::mutex> lock(projection_mutex_);
        update_camera_model(left_model_, *message, "left");
      });
    right_camera_info_subscriber_ = create_subscription<CameraInfo>(
      right_camera_info_topic_, camera_info_qos,
      [this](const CameraInfo::ConstSharedPtr message) {
        const std::lock_guard<std::mutex> lock(projection_mutex_);
        update_camera_model(right_model_, *message, "right");
      });

    if (depth_aware_color_ || use_rgbd_synchronization_) {
      image_callback_group_ = create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);
      rclcpp::SubscriptionOptions image_subscription_options;
      image_subscription_options.callback_group = image_callback_group_;
      const auto color_qos =
        rclcpp::QoS(rclcpp::KeepLast(2)).reliable().durability_volatile();
      const auto depth_qos =
        rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
      left_color_rgbd_subscriber_ = create_subscription<Image>(
        left_color_topic_, color_qos,
        [this](const Image::ConstSharedPtr message) {
          receive_rgbd_message(true, true, message);
        },
        image_subscription_options);
      left_depth_rgbd_subscriber_ = create_subscription<Image>(
        left_depth_topic_, depth_qos,
        [this](const Image::ConstSharedPtr message) {
          receive_rgbd_message(true, false, message);
        },
        image_subscription_options);
      right_color_rgbd_subscriber_ = create_subscription<Image>(
        right_color_topic_, color_qos,
        [this](const Image::ConstSharedPtr message) {
          receive_rgbd_message(false, true, message);
        },
        image_subscription_options);
      right_depth_rgbd_subscriber_ = create_subscription<Image>(
        right_depth_topic_, depth_qos,
        [this](const Image::ConstSharedPtr message) {
          receive_rgbd_message(false, false, message);
        },
        image_subscription_options);
      processing_thread_ = std::thread(
        &RgbdPanoramaStitcherNode::processing_loop, this);
    } else {
      const auto image_qos = rclcpp::SensorDataQoS().keep_last(1);
      left_color_direct_subscriber_ = create_subscription<Image>(
        left_color_topic_, image_qos,
        [this](const Image::ConstSharedPtr message) {
          latest_left_color_ = message;
          latest_left_arrival_ = std::chrono::steady_clock::now();
          try_process_latest_colors();
        });
      right_color_direct_subscriber_ = create_subscription<Image>(
        right_color_topic_, image_qos,
        [this](const Image::ConstSharedPtr message) {
          latest_right_color_ = message;
          latest_right_arrival_ = std::chrono::steady_clock::now();
          try_process_latest_colors();
        });
    }

    RCLCPP_INFO(
      get_logger(),
      "RGB-D panorama ready: optical-axis yaw(left/right)=%.2f/%.2f deg, "
      "camera distance=%.3f m, scale=%.2f, sync<=%.1f ms",
      camera_axis_angle(left_model_) * 180.0 / kPi,
      camera_axis_angle(right_model_) * 180.0 / kPi,
      cv::norm(
        left_model_.translation_camera_in_rig -
        right_model_.translation_camera_in_rig),
      projection_scale_, sync_slop_ms_);
  }

  ~RgbdPanoramaStitcherNode() override
  {
    {
      const std::lock_guard<std::mutex> lock(processing_mutex_);
      stop_processing_ = true;
    }
    processing_condition_.notify_one();
    if (processing_thread_.joinable()) {
      processing_thread_.join();
    }
  }

private:
  static cv::Matx33d yaw_rotation(double yaw_rad)
  {
    const double cosine = std::cos(yaw_rad);
    const double sine = std::sin(yaw_rad);
    return cv::Matx33d(
      cosine, 0.0, sine,
      0.0, 1.0, 0.0,
      -sine, 0.0, cosine);
  }

  static void set_default_yaw_extrinsics(
    CameraModel & model, double yaw_rad, double translation_x_m)
  {
    model.rotation_camera_to_rig = yaw_rotation(yaw_rad);
    model.translation_camera_in_rig =
      cv::Vec3d(translation_x_m, 0.0, 0.0);
  }

  void set_parameter_camera_extrinsics(
    CameraModel & model, const std::string & prefix)
  {
    std::vector<double> default_rotation(9);
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        default_rotation[3 * row + column] =
          model.rotation_camera_to_rig(row, column);
      }
    }
    const std::vector<double> default_translation = {
      model.translation_camera_in_rig[0],
      model.translation_camera_in_rig[1],
      model.translation_camera_in_rig[2]
    };
    const auto rotation = declare_parameter<std::vector<double>>(
      prefix + "_rotation_camera_to_rig", default_rotation);
    const auto translation = declare_parameter<std::vector<double>>(
      prefix + "_translation_camera_in_rig_m", default_translation);
    if (rotation.size() != 9 || translation.size() != 3) {
      throw std::runtime_error(
              prefix + " camera extrinsics must contain 9 rotation and "
              "3 translation values");
    }

    model.rotation_camera_to_rig = cv::Matx33d(
      rotation[0], rotation[1], rotation[2],
      rotation[3], rotation[4], rotation[5],
      rotation[6], rotation[7], rotation[8]);
    model.translation_camera_in_rig = cv::Vec3d(
      translation[0], translation[1], translation[2]);

    const cv::Matx33d orthogonality =
      model.rotation_camera_to_rig *
      model.rotation_camera_to_rig.t();
    const double orthogonality_error = cv::norm(
      cv::Mat(orthogonality - cv::Matx33d::eye()));
    const double determinant = cv::determinant(
      cv::Mat(model.rotation_camera_to_rig));
    if (orthogonality_error > 1e-3 || std::abs(determinant - 1.0) > 1e-3) {
      throw std::runtime_error(
              prefix + " camera rotation is not a valid rotation matrix");
    }
  }

  static double camera_axis_angle(const CameraModel & model)
  {
    const cv::Vec3d optical_axis =
      model.rotation_camera_to_rig * cv::Vec3d(0.0, 0.0, 1.0);
    return std::atan2(optical_axis[0], optical_axis[2]);
  }

  void set_parameter_camera_model(
    CameraModel & model, const std::string & prefix,
    double default_fx, double default_fy, double default_cx, double default_cy)
  {
    model.fx = declare_parameter<double>(prefix + "_fx", default_fx);
    model.fy = declare_parameter<double>(prefix + "_fy", default_fy);
    model.cx = declare_parameter<double>(prefix + "_cx", default_cx);
    model.cy = declare_parameter<double>(prefix + "_cy", default_cy);
    model.width = declare_parameter<int>(prefix + "_width", 1920);
    model.height = declare_parameter<int>(prefix + "_height", 1080);
    adjust_intrinsics_for_input_rotation(model);
  }

  void validate_parameters()
  {
    sync_queue_size_ = std::max(sync_queue_size_, 4);
    sync_slop_ms_ = std::max(sync_slop_ms_, 1.0);
    baseline_m_ = std::max(baseline_m_, 0.0);
    projection_scale_ = std::clamp(projection_scale_, 0.1, 1.0);
    color_reference_plane_z_m_ = std::max(
      color_reference_plane_z_m_, 0.0);
    if (projection_model_ != "cylindrical" &&
      projection_model_ != "rectilinear")
    {
      RCLCPP_WARN(
        get_logger(),
        "Unknown projection_model '%s'; using cylindrical",
        projection_model_.c_str());
      projection_model_ = "cylindrical";
    }
    rectilinear_width_ = std::max(rectilinear_width_, 320);
    rectilinear_height_ = std::max(rectilinear_height_, 180);
    rectilinear_fy_px_ = std::max(rectilinear_fy_px_, 1.0);
    depth_scale_m_ = std::max(depth_scale_m_, 1e-6);
    min_depth_m_ = std::max(min_depth_m_, 0.01);
    max_depth_m_ = std::max(max_depth_m_, min_depth_m_);
    depth_overlap_margin_deg_ = std::max(depth_overlap_margin_deg_, 0.0);
    depth_discontinuity_abs_m_ = std::max(
      depth_discontinuity_abs_m_, 0.0);
    depth_discontinuity_relative_ = std::max(
      depth_discontinuity_relative_, 0.0);
    depth_splat_radius_px_ = std::clamp(depth_splat_radius_px_, 0, 3);
    depth_edge_splat_radius_px_ = std::clamp(
      depth_edge_splat_radius_px_, 0, depth_splat_radius_px_);
    projected_hole_radius_px_ = std::clamp(
      projected_hole_radius_px_, 0, 3);
    seam_feather_px_ = std::max(seam_feather_px_, 0);
    depth_temporal_alpha_ = std::clamp(
      depth_temporal_alpha_, 0.01, 1.0);
    depth_temporal_reset_m_ = std::max(
      depth_temporal_reset_m_, 0.0);
    depth_median_kernel_ = std::max(depth_median_kernel_, 1);
    if (depth_median_kernel_ % 2 == 0) {
      ++depth_median_kernel_;
    }
    depth_median_kernel_ = std::min(depth_median_kernel_, 5);
    occlusion_switch_margin_m_ = std::max(
      occlusion_switch_margin_m_, 0.0);
    exposure_smoothing_ = std::clamp(exposure_smoothing_, 0.0, 1.0);
    min_exposure_gain_ = std::max(min_exposure_gain_, 0.01);
    max_exposure_gain_ = std::max(max_exposure_gain_, min_exposure_gain_);
    diagnostics_period_sec_ = std::max(diagnostics_period_sec_, 0.2);
  }

  void adjust_intrinsics_for_input_rotation(CameraModel & model) const
  {
    if (!input_images_rotated_180_) {
      return;
    }
    model.cx = static_cast<double>(model.width - 1) - model.cx;
    model.cy = static_cast<double>(model.height - 1) - model.cy;
  }

  void update_camera_model(
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
    adjust_intrinsics_for_input_rotation(updated);

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
      "%s CameraInfo: %dx%d fx/fy=%.3f/%.3f cx/cy=%.3f/%.3f%s",
      camera_name, model.width, model.height,
      model.fx, model.fy, model.cx, model.cy,
      input_images_rotated_180_ ? " (adjusted for 180 deg image rotation)" : "");
  }

  static int64_t stamp_nanoseconds(const builtin_interfaces::msg::Time & stamp)
  {
    return static_cast<int64_t>(stamp.sec) * 1000000000LL +
           static_cast<int64_t>(stamp.nanosec);
  }

  cv::Mat to_bgr(const Image::ConstSharedPtr & message) const
  {
    const cv::Mat color = cv_bridge::toCvShare(
      message, sensor_msgs::image_encodings::BGR8)->image;
    if (!rotate_color_180_) {
      return color;
    }
    cv::Mat rotated;
    cv::rotate(color, rotated, cv::ROTATE_180);
    return rotated;
  }

  cv::Mat to_depth(const Image::ConstSharedPtr & message) const
  {
    const cv::Mat depth = cv_bridge::toCvShare(
      message, sensor_msgs::image_encodings::TYPE_16UC1)->image;
    if (!rotate_aligned_depth_180_) {
      return depth;
    }
    cv::Mat rotated;
    cv::rotate(depth, rotated, cv::ROTATE_180);
    return rotated;
  }

  static cv::Vec3d camera_ray(
    const CameraModel & model, double image_x, double image_y)
  {
    return cv::Vec3d(
      (image_x - model.cx) / model.fx,
      (image_y - model.cy) / model.fy,
      1.0);
  }

  static cv::Vec3d rig_ray(
    const CameraModel & model, double image_x, double image_y)
  {
    return model.rotation_camera_to_rig *
           camera_ray(model, image_x, image_y);
  }

  static double horizontal_angle(
    const CameraModel & model, double image_x, double image_y)
  {
    const cv::Vec3d ray = rig_ray(model, image_x, image_y);
    return std::atan2(ray[0], ray[2]);
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
    panorama_focal_px_ =
      0.5 * (left_model_.fx + right_model_.fx) * projection_scale_;

    const auto camera_bounds =
      [this, width, height](const CameraModel & model) {
        std::array<double, 4> bounds = {
          std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity()
        };
        for (const double image_y :
          {0.0, static_cast<double>(height - 1)})
        {
          for (const double image_x :
            {0.0, static_cast<double>(width - 1)})
          {
            cv::Vec3d ray = rig_ray(model, image_x, image_y);
            if (ray[2] <= 1e-6) {
              continue;
            }
            if (color_reference_plane_z_m_ > 0.0) {
              const double scale =
                (color_reference_plane_z_m_ -
                model.translation_camera_in_rig[2]) / ray[2];
              if (scale <= 0.0) {
                continue;
              }
              ray = model.translation_camera_in_rig + scale * ray;
            }
            const double angle = std::atan2(ray[0], ray[2]);
            const double vertical_ratio =
              projection_model_ == "rectilinear" ?
              ray[1] / ray[2] :
              ray[1] / std::hypot(ray[0], ray[2]);
            bounds[0] = std::min(bounds[0], angle);
            bounds[1] = std::max(bounds[1], angle);
            bounds[2] = std::min(bounds[2], vertical_ratio);
            bounds[3] = std::max(bounds[3], vertical_ratio);
          }
        }
        return bounds;
      };

    const auto left_bounds = camera_bounds(left_model_);
    const auto right_bounds = camera_bounds(right_model_);
    left_min_angle_ = left_bounds[0];
    left_max_angle_ = left_bounds[1];
    right_min_angle_ = right_bounds[0];
    right_max_angle_ = right_bounds[1];
    panorama_min_angle_ = std::min(left_min_angle_, right_min_angle_);
    panorama_max_angle_ = std::max(left_max_angle_, right_max_angle_);
    overlap_min_angle_ = std::max(left_min_angle_, right_min_angle_);
    overlap_max_angle_ = std::min(left_max_angle_, right_max_angle_);
    const double panorama_min_vertical_ratio =
      std::min(left_bounds[2], right_bounds[2]);
    const double panorama_max_vertical_ratio =
      std::max(left_bounds[3], right_bounds[3]);

    if (projection_model_ == "rectilinear") {
      panorama_width_ = rectilinear_width_;
      const double minimum_tangent = std::tan(panorama_min_angle_);
      const double maximum_tangent = std::tan(panorama_max_angle_);
      virtual_fx_px_ =
        static_cast<double>(panorama_width_ - 1) /
        (maximum_tangent - minimum_tangent);
      virtual_cx_px_ = -virtual_fx_px_ * minimum_tangent;
      virtual_fy_px_ = rectilinear_fy_px_;
      if (rectilinear_auto_height_) {
        panorama_height_ = static_cast<int>(std::ceil(
          (panorama_max_vertical_ratio - panorama_min_vertical_ratio) *
          virtual_fy_px_)) + 1;
        virtual_cy_px_ =
          -virtual_fy_px_ * panorama_min_vertical_ratio;
      } else {
        panorama_height_ = rectilinear_height_;
        virtual_cy_px_ =
          0.5 * static_cast<double>(panorama_height_ - 1) -
          0.5 * virtual_fy_px_ *
          (panorama_min_vertical_ratio + panorama_max_vertical_ratio);
      }
      panorama_min_vertical_ = 0.0;
    } else {
      panorama_width_ = static_cast<int>(
        std::ceil(
          (panorama_max_angle_ - panorama_min_angle_) *
          panorama_focal_px_)) + 1;

      panorama_min_vertical_ =
        panorama_focal_px_ * panorama_min_vertical_ratio;
      const double panorama_max_vertical =
        panorama_focal_px_ * panorama_max_vertical_ratio;
      panorama_height_ = static_cast<int>(
        std::ceil(panorama_max_vertical - panorama_min_vertical_)) + 1;
    }

    if (panorama_width_ <= 0 || panorama_height_ <= 0 ||
      overlap_max_angle_ <= overlap_min_angle_)
    {
      throw std::runtime_error("invalid panorama projection geometry");
    }

    build_inverse_map(
      left_model_, left_map_x_, left_map_y_, left_base_mask_);
    build_inverse_map(
      right_model_, right_map_x_, right_map_y_, right_base_mask_);

    const double seam_angle = auto_seam_center_ ?
      0.5 * (overlap_min_angle_ + overlap_max_angle_) :
      seam_angle_deg_ * kPi / 180.0;
    if (projection_model_ == "rectilinear") {
      seam_x_ = static_cast<int>(std::lround(
        virtual_fx_px_ * std::tan(seam_angle) + virtual_cx_px_));
    } else {
      seam_x_ = static_cast<int>(std::lround(
        (seam_angle - panorama_min_angle_) * panorama_focal_px_));
    }
    seam_x_ = std::clamp(seam_x_, 0, panorama_width_ - 1);
    projection_dirty_ = false;
#ifdef PANORAMA_WITH_CUDA
    cuda_backend_configured_ = false;
#endif

    RCLCPP_INFO(
      get_logger(),
      "Projection (%s): %dx%d, angular view %.1f..%.1f deg, "
      "overlap %.1f..%.1f deg, seam x=%d",
      projection_model_.c_str(), panorama_width_, panorama_height_,
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

    for (int y = 0; y < panorama_height_; ++y) {
      float * map_x_row = map_x.ptr<float>(y);
      float * map_y_row = map_y.ptr<float>(y);
      uint8_t * mask_row = mask.ptr<uint8_t>(y);
      for (int x = 0; x < panorama_width_; ++x) {
        double source_x;
        double source_y;
        cv::Vec3d panorama_ray;
        if (projection_model_ == "rectilinear") {
          panorama_ray[0] =
            (static_cast<double>(x) - virtual_cx_px_) / virtual_fx_px_;
          panorama_ray[1] =
            (static_cast<double>(y) - virtual_cy_px_) / virtual_fy_px_;
          panorama_ray[2] = 1.0;
        } else {
          const double global_angle =
            panorama_min_angle_ +
            static_cast<double>(x) / panorama_focal_px_;
          const double cylinder_y =
            panorama_min_vertical_ + static_cast<double>(y);
          panorama_ray = cv::Vec3d(
            std::sin(global_angle),
            cylinder_y / panorama_focal_px_,
            std::cos(global_angle));
        }
        cv::Vec3d source_ray;
        if (color_reference_plane_z_m_ > 0.0) {
          const double rig_forward = panorama_ray[2];
          if (rig_forward <= 1e-6) {
            map_x_row[x] = -1.0F;
            map_y_row[x] = -1.0F;
            continue;
          }
          const cv::Vec3d rig_point =
            (color_reference_plane_z_m_ / rig_forward) * panorama_ray;
          source_ray = model.rotation_camera_to_rig.t() *
            (rig_point - model.translation_camera_in_rig);
        } else {
          source_ray =
            model.rotation_camera_to_rig.t() * panorama_ray;
        }
        const double camera_forward = source_ray[2];
        source_x =
          model.fx * source_ray[0] / camera_forward + model.cx;
        source_y =
          model.fy * source_ray[1] / camera_forward + model.cy;
        map_x_row[x] = static_cast<float>(source_x);
        map_y_row[x] = static_cast<float>(source_y);
        if (camera_forward > 0.0 &&
          source_x >= 0.0 && source_x <= source_width_ - 1.0 &&
          source_y >= 0.0 && source_y <= source_height_ - 1.0)
        {
          mask_row[x] = 255;
        }
      }
    }
  }

  std::array<int, 2> depth_source_columns(
    const CameraModel & model) const
  {
    if (full_depth_reprojection_) {
      return {0, source_width_ - 1};
    }
    const double margin = depth_overlap_margin_deg_ * kPi / 180.0;
    const double min_angle = overlap_min_angle_ - margin;
    const double max_angle = overlap_max_angle_ + margin;
    int minimum_column = source_width_;
    int maximum_column = -1;
    for (int column = 0; column < source_width_; ++column) {
      const double angle = horizontal_angle(
        model, static_cast<double>(column), model.cy);
      if (angle >= min_angle && angle <= max_angle) {
        minimum_column = std::min(minimum_column, column);
        maximum_column = std::max(maximum_column, column);
      }
    }
    return {minimum_column, maximum_column};
  }

  ProjectedDepth project_depth_overlap(
    const cv::Mat & color, const cv::Mat & depth,
    const CameraModel & model) const
  {
    ProjectedDepth projected;
    projected.color = cv::Mat::zeros(
      panorama_height_, panorama_width_, CV_8UC3);
    projected.range = cv::Mat(
      panorama_height_, panorama_width_, CV_32FC1,
      cv::Scalar(std::numeric_limits<float>::infinity()));
    projected.mask = cv::Mat::zeros(
      panorama_height_, panorama_width_, CV_8UC1);

    if (depth.size() != color.size()) {
      return projected;
    }

    const auto columns = depth_source_columns(model);
    if (columns[1] < columns[0]) {
      return projected;
    }

    const bool depth_is_float = depth.type() == CV_32FC1;
    const bool depth_is_uint16 = depth.type() == CV_16UC1;
    if (!depth_is_float && !depth_is_uint16) {
      return projected;
    }

    for (int v = 0; v < source_height_; ++v) {
      const uint16_t * depth_row_uint16 =
        depth_is_uint16 ? depth.ptr<uint16_t>(v) : nullptr;
      const float * depth_row_float =
        depth_is_float ? depth.ptr<float>(v) : nullptr;
      const cv::Vec3b * color_row = color.ptr<cv::Vec3b>(v);
      for (int u = columns[0]; u <= columns[1]; ++u) {
        const double depth_m = depth_is_float ?
          static_cast<double>(depth_row_float[u]) :
          static_cast<double>(depth_row_uint16[u]) * depth_scale_m_;
        if (depth_m < min_depth_m_ || depth_m > max_depth_m_) {
          continue;
        }

        const double local_x =
          (static_cast<double>(u) - model.cx) /
          model.fx * depth_m;
        const double local_y =
          (static_cast<double>(v) - model.cy) /
          model.fy * depth_m;
        const cv::Vec3d rig_point =
          model.rotation_camera_to_rig *
          cv::Vec3d(local_x, local_y, depth_m) +
          model.translation_camera_in_rig;
        const double rig_x = rig_point[0];
        const double rig_y = rig_point[1];
        const double rig_z = rig_point[2];
        if (rig_z <= 0.0) {
          continue;
        }

        const double horizontal_range = std::hypot(rig_x, rig_z);
        int output_x;
        int output_y;
        if (projection_model_ == "rectilinear") {
          output_x = static_cast<int>(std::lround(
            virtual_fx_px_ * rig_x / rig_z + virtual_cx_px_));
          output_y = static_cast<int>(std::lround(
            virtual_fy_px_ * rig_y / rig_z + virtual_cy_px_));
        } else {
          const double global_angle = std::atan2(rig_x, rig_z);
          output_x = static_cast<int>(std::lround(
            (global_angle - panorama_min_angle_) * panorama_focal_px_));
          output_y = static_cast<int>(std::lround(
            panorama_focal_px_ * rig_y / horizontal_range -
            panorama_min_vertical_));
        }
        if (output_x < 0 || output_x >= panorama_width_ ||
          output_y < 0 || output_y >= panorama_height_)
        {
          continue;
        }

        const double discontinuity_threshold = std::max(
          depth_discontinuity_abs_m_,
          depth_discontinuity_relative_ * depth_m);
        bool on_depth_edge = false;
        constexpr int neighbor_offsets[4][2] = {
          {-1, 0}, {1, 0}, {0, -1}, {0, 1}
        };
        for (const auto & offset : neighbor_offsets) {
          const int neighbor_u = u + offset[0];
          const int neighbor_v = v + offset[1];
          if (
            neighbor_u < 0 || neighbor_u >= source_width_ ||
            neighbor_v < 0 || neighbor_v >= source_height_)
          {
            continue;
          }
          const double neighbor_depth_m = depth_is_float ?
            static_cast<double>(
            depth.ptr<float>(neighbor_v)[neighbor_u]) :
            static_cast<double>(
            depth.ptr<uint16_t>(neighbor_v)[neighbor_u]) * depth_scale_m_;
          if (
            neighbor_depth_m < min_depth_m_ ||
            neighbor_depth_m > max_depth_m_ ||
            std::abs(neighbor_depth_m - depth_m) >
            discontinuity_threshold)
          {
            on_depth_edge = true;
            break;
          }
        }
        const int splat_radius = on_depth_edge ?
          depth_edge_splat_radius_px_ :
          depth_splat_radius_px_;
        for (int dy = -splat_radius; dy <= splat_radius; ++dy) {
          const int target_y = output_y + dy;
          if (target_y < 0 || target_y >= panorama_height_) {
            continue;
          }
          for (int dx = -splat_radius; dx <= splat_radius; ++dx) {
            const int target_x = output_x + dx;
            if (target_x < 0 || target_x >= panorama_width_) {
              continue;
            }
            float & previous_range =
              projected.range.at<float>(target_y, target_x);
            if (horizontal_range < previous_range) {
              previous_range = static_cast<float>(horizontal_range);
              projected.color.at<cv::Vec3b>(target_y, target_x) =
                color_row[u];
              projected.mask.at<uint8_t>(target_y, target_x) = 255;
            }
          }
        }
        ++projected.accepted_points;
      }
    }

    return projected;
  }

  cv::Mat stabilize_depth(
    const cv::Mat & depth, cv::Mat & state_depth_m,
    const CameraModel & model) const
  {
    const auto columns = depth_source_columns(model);
    if (columns[1] < columns[0]) {
      return depth;
    }
    const cv::Rect region(
      columns[0], 0, columns[1] - columns[0] + 1, depth.rows);
    const cv::Mat depth_region = depth(region);

    cv::Mat filtered_depth = depth_region;
    cv::Mat median_depth;
    if (depth_median_kernel_ > 1) {
      cv::medianBlur(
        depth_region, median_depth, depth_median_kernel_);
      filtered_depth = median_depth;
    }

    cv::Mat current_depth_m;
    if (filtered_depth.type() == CV_16UC1) {
      filtered_depth.convertTo(
        current_depth_m, CV_32FC1, depth_scale_m_);
    } else if (filtered_depth.type() == CV_32FC1) {
      current_depth_m = filtered_depth;
    } else {
      return depth;
    }

    const cv::Mat current_valid =
      (current_depth_m >= min_depth_m_) &
      (current_depth_m <= max_depth_m_);
    if (state_depth_m.empty() ||
      state_depth_m.size() != depth.size())
    {
      state_depth_m = cv::Mat::zeros(
        depth.size(), CV_32FC1);
      current_depth_m.copyTo(state_depth_m(region), current_valid);
      return state_depth_m;
    }

    cv::Mat state_region = state_depth_m(region);
    const cv::Mat previous_valid =
      (state_region >= min_depth_m_) &
      (state_region <= max_depth_m_);
    cv::Mat depth_delta;
    cv::absdiff(current_depth_m, state_region, depth_delta);
    const cv::Mat stable_measurement =
      current_valid & previous_valid &
      (depth_delta <= depth_temporal_reset_m_);
    const cv::Mat reset_measurement =
      current_valid & ~stable_measurement;

    cv::Mat blended_depth =
      (1.0 - depth_temporal_alpha_) * state_region +
      depth_temporal_alpha_ * current_depth_m;
    blended_depth.copyTo(state_region, stable_measurement);
    current_depth_m.copyTo(state_region, reset_measurement);
    state_region.setTo(0.0f, ~current_valid);
    return state_depth_m;
  }

  static void fill_projected_holes(
    ProjectedDepth & projected, int iterations)
  {
    std::vector<cv::Point> valid_points;
    cv::findNonZero(projected.mask, valid_points);
    if (valid_points.empty()) {
      return;
    }
    cv::Rect region = cv::boundingRect(valid_points);
    region.x = std::max(region.x - iterations, 0);
    region.y = std::max(region.y - iterations, 0);
    region.width = std::min(
      region.width + 2 * iterations,
      projected.mask.cols - region.x);
    region.height = std::min(
      region.height + 2 * iterations,
      projected.mask.rows - region.y);

    cv::Mat color_region = projected.color(region);
    cv::Mat mask_region = projected.mask(region);
    for (int iteration = 0; iteration < iterations; ++iteration) {
      cv::Mat mask_float;
      mask_region.convertTo(mask_float, CV_32FC1, 1.0 / 255.0);
      cv::Mat blurred_mask;
      cv::blur(mask_float, blurred_mask, cv::Size(3, 3));
      cv::Mat fill_mask =
        (mask_region == 0) & (blurred_mask > 1e-6);
      if (cv::countNonZero(fill_mask) == 0) {
        break;
      }

      cv::Mat color_float;
      color_region.convertTo(color_float, CV_32FC3);
      std::vector<cv::Mat> channels;
      cv::split(color_float, channels);
      for (cv::Mat & channel : channels) {
        channel = channel.mul(mask_float);
        cv::blur(channel, channel, cv::Size(3, 3));
        cv::divide(
          channel, blurred_mask + cv::Scalar::all(1e-6),
          channel);
      }

      cv::Mat filled_float;
      cv::merge(channels, filled_float);
      cv::Mat filled;
      filled_float.convertTo(filled, CV_8UC3);
      filled.copyTo(color_region, fill_mask);
      mask_region.setTo(255, fill_mask);
    }
  }

  cv::Vec3d estimate_right_gain(
    const cv::Mat & left, const cv::Mat & right,
    const cv::Mat & overlap_mask)
  {
    if (!exposure_compensation_ || cv::countNonZero(overlap_mask) < 500) {
      return smoothed_gain_;
    }

    cv::Mat left_gray;
    cv::Mat right_gray;
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
    cv::Mat valid_mask =
      overlap_mask &
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

  cv::Vec3d estimate_right_gain_from_sources(
    const cv::Mat & left, const cv::Mat & right)
  {
    if (!exposure_compensation_) {
      return smoothed_gain_;
    }
    const auto left_columns = depth_source_columns(left_model_);
    const auto right_columns = depth_source_columns(right_model_);
    if (
      left_columns[1] < left_columns[0] ||
      right_columns[1] < right_columns[0])
    {
      return smoothed_gain_;
    }

    const cv::Mat left_region = left(
      cv::Rect(
        left_columns[0], 0,
        left_columns[1] - left_columns[0] + 1,
        left.rows));
    const cv::Mat right_region = right(
      cv::Rect(
        right_columns[0], 0,
        right_columns[1] - right_columns[0] + 1,
        right.rows));
    cv::Mat left_gray;
    cv::Mat right_gray;
    cv::cvtColor(left_region, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_region, right_gray, cv::COLOR_BGR2GRAY);
    const cv::Mat left_valid = (left_gray > 25) & (left_gray < 235);
    const cv::Mat right_valid = (right_gray > 25) & (right_gray < 235);
    if (
      cv::countNonZero(left_valid) < 500 ||
      cv::countNonZero(right_valid) < 500)
    {
      return smoothed_gain_;
    }

    const cv::Scalar left_mean = cv::mean(left_region, left_valid);
    const cv::Scalar right_mean = cv::mean(right_region, right_valid);
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

#ifdef PANORAMA_WITH_CUDA
  CudaCameraModel make_cuda_camera_model(
    const CameraModel & model) const
  {
    CudaCameraModel cuda_model;
    cuda_model.fx = static_cast<float>(model.fx);
    cuda_model.fy = static_cast<float>(model.fy);
    cuda_model.cx = static_cast<float>(model.cx);
    cuda_model.cy = static_cast<float>(model.cy);
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        cuda_model.rotation_camera_to_rig[3 * row + column] =
          static_cast<float>(model.rotation_camera_to_rig(row, column));
      }
      cuda_model.translation_camera_in_rig[row] =
        static_cast<float>(model.translation_camera_in_rig[row]);
    }
    const auto columns = depth_source_columns(model);
    cuda_model.minimum_depth_column = columns[0];
    cuda_model.maximum_depth_column = columns[1];
    return cuda_model;
  }

  bool ensure_cuda_backend_configured()
  {
    if (
      !use_cuda_ || cuda_backend_failed_ ||
      cuda_backend_ == nullptr)
    {
      return false;
    }
    if (cuda_backend_configured_) {
      return true;
    }

    CudaPanoramaConfig config;
    config.source_width = source_width_;
    config.source_height = source_height_;
    config.panorama_width = panorama_width_;
    config.panorama_height = panorama_height_;
    config.projection_model =
      projection_model_ == "rectilinear" ? 1 : 0;
    config.panorama_focal_px =
      static_cast<float>(panorama_focal_px_);
    config.panorama_min_angle =
      static_cast<float>(panorama_min_angle_);
    config.panorama_min_vertical =
      static_cast<float>(panorama_min_vertical_);
    config.virtual_fx_px = static_cast<float>(virtual_fx_px_);
    config.virtual_fy_px = static_cast<float>(virtual_fy_px_);
    config.virtual_cx_px = static_cast<float>(virtual_cx_px_);
    config.virtual_cy_px = static_cast<float>(virtual_cy_px_);
    config.minimum_depth_m = static_cast<float>(min_depth_m_);
    config.maximum_depth_m = static_cast<float>(max_depth_m_);
    config.depth_discontinuity_abs_m =
      static_cast<float>(depth_discontinuity_abs_m_);
    config.depth_discontinuity_relative =
      static_cast<float>(depth_discontinuity_relative_);
    config.occlusion_switch_margin_m =
      static_cast<float>(occlusion_switch_margin_m_);
    config.depth_aware_color = depth_aware_color_;
    config.render_depth_reprojected_color =
      render_depth_reprojected_color_;
    config.allow_color_fallback = allow_color_fallback_;
    config.prefer_seam_camera_when_both_depth_valid =
      prefer_seam_camera_when_both_depth_valid_;
    config.seam_x = seam_x_;
    config.seam_feather_px = seam_feather_px_;
    config.depth_splat_radius_px = depth_splat_radius_px_;
    config.depth_edge_splat_radius_px = depth_edge_splat_radius_px_;
    config.projected_hole_radius = projected_hole_radius_px_;

    std::string error;
    if (!cuda_backend_->configure(
        config,
        make_cuda_camera_model(left_model_),
        make_cuda_camera_model(right_model_),
        left_base_mask_, right_base_mask_,
        left_map_x_, left_map_y_,
        right_map_x_, right_map_y_, error))
    {
      cuda_backend_failed_ = true;
      RCLCPP_ERROR(
        get_logger(),
        "CUDA panorama configuration failed (%s); using CPU fallback",
        error.c_str());
      return false;
    }
    cuda_backend_configured_ = true;
    RCLCPP_INFO(
      get_logger(),
      "CUDA buffers ready for %dx%d panorama",
      panorama_width_, panorama_height_);
    return true;
  }

  static cv::Mat depth_as_float_meters(
    const cv::Mat & depth, double depth_scale_m)
  {
    if (depth.type() == CV_32FC1) {
      return depth;
    }
    cv::Mat depth_m;
    depth.convertTo(depth_m, CV_32FC1, depth_scale_m);
    return depth_m;
  }
#endif

  cv::Mat stitch_rgbd(
    const cv::Mat & left_color, const cv::Mat & left_depth,
    const cv::Mat & right_color, const cv::Mat & right_depth)
  {
    build_projection_if_needed(left_color.cols, left_color.rows);

    cv::Mat left_projection_depth = left_depth;
    cv::Mat right_projection_depth = right_depth;
    if (depth_aware_color_ && depth_temporal_stabilization_) {
      left_projection_depth = stabilize_depth(
        left_depth, left_stabilized_depth_m_, left_model_);
      right_projection_depth = stabilize_depth(
        right_depth, right_stabilized_depth_m_, right_model_);
    }

#ifdef PANORAMA_WITH_CUDA
    if (ensure_cuda_backend_configured()) {
      const cv::Vec3d gain =
        estimate_right_gain_from_sources(left_color, right_color);
      const cv::Mat left_depth_m = depth_as_float_meters(
        left_projection_depth, depth_scale_m_);
      const cv::Mat right_depth_m = depth_as_float_meters(
        right_projection_depth, depth_scale_m_);
      cv::Mat panorama;
      CudaPanoramaStats stats;
      std::string error;
      if (cuda_backend_->process(
          left_color, left_depth_m,
          right_color, right_depth_m, gain,
          panorama, last_validity_mask_, last_range_m_,
          stats, error))
      {
        last_left_depth_points_ = stats.left_depth_points;
        last_right_depth_points_ = stats.right_depth_points;
        last_gpu_time_ms_ = stats.gpu_time_ms;
        used_cuda_last_frame_ = true;
        return panorama;
      }
      cuda_backend_failed_ = true;
      cuda_backend_configured_ = false;
      RCLCPP_ERROR(
        get_logger(),
        "CUDA panorama processing failed (%s); using CPU fallback",
        error.c_str());
    }
#endif

    cv::Mat left_base;
    cv::Mat right_base;
    cv::remap(
      left_color, left_base, left_map_x_, left_map_y_,
      cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(
      right_color, right_base, right_map_x_, right_map_y_,
      cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    const cv::Mat overlap_mask = left_base_mask_ & right_base_mask_;
    const cv::Vec3d gain =
      estimate_right_gain(left_base, right_base, overlap_mask);

    used_cuda_last_frame_ = false;
    last_gpu_time_ms_ = 0.0;
    ProjectedDepth left_projected;
    ProjectedDepth right_projected;
    if (depth_aware_color_) {
      left_projected =
        project_depth_overlap(
        left_color, left_projection_depth, left_model_);
      right_projected =
        project_depth_overlap(
        right_color, right_projection_depth, right_model_);
      fill_projected_holes(left_projected, projected_hole_radius_px_);
      fill_projected_holes(right_projected, projected_hole_radius_px_);

      if (render_depth_reprojected_color_) {
        left_projected.color.copyTo(left_base, left_projected.mask);
        right_projected.color.copyTo(right_base, right_projected.mask);
      }
    }

    right_base = apply_gain(right_base, gain);

    cv::Mat panorama = cv::Mat::zeros(
      panorama_height_, panorama_width_, CV_8UC3);

    for (int y = 0; y < panorama_height_; ++y) {
      const int blend_left = seam_x_ - seam_feather_px_;
      const int blend_right = seam_x_ + seam_feather_px_;
      const cv::Vec3b * left_row = left_base.ptr<cv::Vec3b>(y);
      const cv::Vec3b * right_row = right_base.ptr<cv::Vec3b>(y);
      const uint8_t * left_mask_row = left_base_mask_.ptr<uint8_t>(y);
      const uint8_t * right_mask_row = right_base_mask_.ptr<uint8_t>(y);
      cv::Vec3b * output_row = panorama.ptr<cv::Vec3b>(y);

      for (int x = 0; x < panorama_width_; ++x) {
        const bool left_valid = left_mask_row[x] != 0;
        const bool right_valid = right_mask_row[x] != 0;
        if (!left_valid && !right_valid) {
          continue;
        }

        // A fixed left/right seam assumes both cameras share one optical
        // center. With a real baseline, close objects move to opposite sides
        // of that seam and central content is dropped. In the calibrated
        // overlap, preserve the union of both depth-reprojected views instead.
        if (depth_aware_color_ && render_depth_reprojected_color_) {
          const bool left_depth_valid =
            left_projected.mask.at<uint8_t>(y, x) != 0;
          const bool right_depth_valid =
            right_projected.mask.at<uint8_t>(y, x) != 0;
          if (left_depth_valid || right_depth_valid) {
            if (prefer_seam_camera_when_both_depth_valid_) {
              const float left_range =
                left_depth_valid ?
                left_projected.range.at<float>(y, x) :
                std::numeric_limits<float>::quiet_NaN();
              const float right_range =
                right_depth_valid ?
                right_projected.range.at<float>(y, x) :
                std::numeric_limits<float>::quiet_NaN();
              const bool blend_same_surface =
                left_depth_valid && right_depth_valid &&
                seam_feather_px_ > 0 &&
                x >= blend_left && x <= blend_right &&
                std::isfinite(left_range) &&
                std::isfinite(right_range) &&
                std::abs(left_range - right_range) <=
                occlusion_switch_margin_m_;
              if (blend_same_surface) {
                const double right_weight = std::clamp(
                  static_cast<double>(x - blend_left) /
                  std::max(2 * seam_feather_px_, 1), 0.0, 1.0);
                for (int channel = 0; channel < 3; ++channel) {
                  output_row[x][channel] =
                    cv::saturate_cast<uint8_t>(
                    (1.0 - right_weight) * left_row[x][channel] +
                    right_weight * right_row[x][channel]);
                }
              } else if (x <= seam_x_ && left_valid) {
                output_row[x] = left_row[x];
              } else if (x > seam_x_ && right_valid) {
                output_row[x] = right_row[x];
              } else if (left_depth_valid) {
                output_row[x] = left_row[x];
              } else {
                output_row[x] = right_row[x];
              }
            } else {
              if (left_depth_valid && !right_depth_valid) {
                output_row[x] = left_row[x];
                continue;
              }
              if (!left_depth_valid && right_depth_valid) {
                output_row[x] = right_row[x];
                continue;
              }
              const float left_range =
                left_projected.range.at<float>(y, x);
              const float right_range =
                right_projected.range.at<float>(y, x);
              const bool both_finite =
                std::isfinite(left_range) &&
                std::isfinite(right_range);
              if (both_finite &&
                std::abs(left_range - right_range) <=
                occlusion_switch_margin_m_)
              {
                // Both measurements describe the same physical surface.
                // Keep a deterministic camera choice so millimetre-scale
                // depth noise cannot make the seam alternate every frame.
                output_row[x] =
                  x <= seam_x_ ? left_row[x] : right_row[x];
              } else if (std::isfinite(left_range) &&
                (!std::isfinite(right_range) ||
                left_range <= right_range))
              {
                output_row[x] = left_row[x];
              } else if (std::isfinite(right_range)) {
                output_row[x] = right_row[x];
              } else {
                for (int channel = 0; channel < 3; ++channel) {
                  output_row[x][channel] =
                    cv::saturate_cast<uint8_t>(
                    0.5 * static_cast<double>(left_row[x][channel]) +
                    0.5 * static_cast<double>(right_row[x][channel]));
                }
              }
            }
            continue;
          }
          if (!allow_color_fallback_) {
            continue;
          }
        }

        if (!right_valid || (left_valid && x < blend_left)) {
          output_row[x] = left_row[x];
          continue;
        }
        if (!left_valid || x > blend_right) {
          output_row[x] = right_row[x];
          continue;
        }

        const double denominator =
          std::max(2 * seam_feather_px_, 1);
        const double right_weight = std::clamp(
          static_cast<double>(x - blend_left) / denominator,
          0.0, 1.0);
        for (int channel = 0; channel < 3; ++channel) {
          output_row[x][channel] = cv::saturate_cast<uint8_t>(
            (1.0 - right_weight) * left_row[x][channel] +
            right_weight * right_row[x][channel]);
        }
      }
    }

    last_left_depth_points_ = left_projected.accepted_points;
    last_right_depth_points_ = right_projected.accepted_points;
    if (!depth_aware_color_) {
      last_validity_mask_ = cv::Mat::zeros(
        panorama_height_, panorama_width_, CV_8UC1);
      last_range_m_ = cv::Mat::zeros(
        panorama_height_, panorama_width_, CV_32FC1);
      return panorama;
    }

    last_validity_mask_ = left_projected.mask | right_projected.mask;
    last_range_m_ = cv::Mat::zeros(
      panorama_height_, panorama_width_, CV_32FC1);
    for (int y = 0; y < panorama_height_; ++y) {
      const uint8_t * left_mask_row =
        left_projected.mask.ptr<uint8_t>(y);
      const uint8_t * right_mask_row =
        right_projected.mask.ptr<uint8_t>(y);
      const float * left_range_row =
        left_projected.range.ptr<float>(y);
      const float * right_range_row =
        right_projected.range.ptr<float>(y);
      float * output_range_row = last_range_m_.ptr<float>(y);
      for (int x = 0; x < panorama_width_; ++x) {
        if (left_mask_row[x] && right_mask_row[x]) {
          if (prefer_seam_camera_when_both_depth_valid_) {
            output_range_row[x] =
              x <= seam_x_ ? left_range_row[x] : right_range_row[x];
          } else {
            output_range_row[x] = std::min(
              left_range_row[x], right_range_row[x]);
          }
        } else if (left_mask_row[x]) {
          output_range_row[x] = left_range_row[x];
        } else if (right_mask_row[x]) {
          output_range_row[x] = right_range_row[x];
        }
      }
    }
    return panorama;
  }

  void receive_rgbd_message(
    bool is_left_camera,
    bool is_color,
    const Image::ConstSharedPtr & message)
  {
    if (is_left_camera && is_color) {
      ++received_left_color_count_;
    } else if (is_left_camera) {
      ++received_left_depth_count_;
    } else if (is_color) {
      ++received_right_color_count_;
    } else {
      ++received_right_depth_count_;
    }
    bool synchronized_set_ready = false;
    {
      const std::lock_guard<std::mutex> lock(processing_mutex_);
      auto & queue =
        is_left_camera ?
        (is_color ? left_color_queue_ : left_depth_queue_) :
        (is_color ? right_color_queue_ : right_depth_queue_);
      queue.push_back(message);
      while (queue.size() > 4) {
        queue.pop_front();
      }
      synchronized_set_ready =
        try_form_synchronized_set_locked();
    }
    if (synchronized_set_ready) {
      processing_condition_.notify_one();
    }
  }

  bool try_form_synchronized_set_locked()
  {
    if (
      left_color_queue_.empty() || right_color_queue_.empty() ||
      left_depth_queue_.empty() || right_depth_queue_.empty())
    {
      return false;
    }

    std::size_t best_left_color = 0;
    std::size_t best_right_color = 0;
    int64_t best_color_delta = std::numeric_limits<int64_t>::max();
    int64_t best_color_time = std::numeric_limits<int64_t>::min();
    for (
      std::size_t left_index = 0;
      left_index < left_color_queue_.size();
      ++left_index)
    {
      const int64_t left_stamp = stamp_nanoseconds(
        left_color_queue_[left_index]->header.stamp);
      for (
        std::size_t right_index = 0;
        right_index < right_color_queue_.size();
        ++right_index)
      {
        const int64_t right_stamp = stamp_nanoseconds(
          right_color_queue_[right_index]->header.stamp);
        const int64_t delta = std::llabs(left_stamp - right_stamp);
        const int64_t pair_time = std::max(left_stamp, right_stamp);
        if (
          delta < best_color_delta ||
          (delta == best_color_delta && pair_time > best_color_time))
        {
          best_color_delta = delta;
          best_color_time = pair_time;
          best_left_color = left_index;
          best_right_color = right_index;
        }
      }
    }
    if (static_cast<double>(best_color_delta) / 1e6 > sync_slop_ms_) {
      return false;
    }

    const auto nearest_depth = [](const auto & depth_queue, int64_t color_stamp) {
        std::size_t best_index = 0;
        int64_t best_delta = std::numeric_limits<int64_t>::max();
        for (std::size_t index = 0; index < depth_queue.size(); ++index) {
          const int64_t depth_stamp = stamp_nanoseconds(
            depth_queue[index]->header.stamp);
          const int64_t delta = std::llabs(color_stamp - depth_stamp);
          if (delta < best_delta) {
            best_delta = delta;
            best_index = index;
          }
        }
        return std::make_pair(best_index, best_delta);
      };

    const int64_t left_color_stamp = stamp_nanoseconds(
      left_color_queue_[best_left_color]->header.stamp);
    const int64_t right_color_stamp = stamp_nanoseconds(
      right_color_queue_[best_right_color]->header.stamp);
    const auto left_depth_match =
      nearest_depth(left_depth_queue_, left_color_stamp);
    const auto right_depth_match =
      nearest_depth(right_depth_queue_, right_color_stamp);
    if (
      static_cast<double>(left_depth_match.second) / 1e6 >
      sync_slop_ms_ ||
      static_cast<double>(right_depth_match.second) / 1e6 >
      sync_slop_ms_)
    {
      return false;
    }

    pending_left_color_ = left_color_queue_[best_left_color];
    pending_right_color_ = right_color_queue_[best_right_color];
    pending_left_depth_ = left_depth_queue_[left_depth_match.first];
    pending_right_depth_ = right_depth_queue_[right_depth_match.first];
    last_left_depth_age_ms_ =
      static_cast<double>(left_depth_match.second) / 1e6;
    last_right_depth_age_ms_ =
      static_cast<double>(right_depth_match.second) / 1e6;
    ++pending_sequence_;

    const auto erase_through = [](auto & queue, std::size_t index) {
        queue.erase(
          queue.begin(),
          queue.begin() + static_cast<std::ptrdiff_t>(index + 1));
      };
    erase_through(left_color_queue_, best_left_color);
    erase_through(right_color_queue_, best_right_color);
    erase_through(left_depth_queue_, left_depth_match.first);
    erase_through(right_depth_queue_, right_depth_match.first);
    return true;
  }

  void processing_loop()
  {
    std::size_t processed_sequence = 0;
    while (true) {
      Image::ConstSharedPtr left_color;
      Image::ConstSharedPtr right_color;
      Image::ConstSharedPtr left_depth;
      Image::ConstSharedPtr right_depth;
      {
        std::unique_lock<std::mutex> lock(processing_mutex_);
        processing_condition_.wait(
          lock,
          [this, &processed_sequence]() {
            return stop_processing_ ||
                   pending_sequence_ != processed_sequence;
          });
        if (stop_processing_) {
          return;
        }
        left_color = pending_left_color_;
        right_color = pending_right_color_;
        left_depth = pending_left_depth_;
        right_depth = pending_right_depth_;
        processed_sequence = pending_sequence_;
      }
      process_messages(
        left_color, right_color, left_depth, right_depth,
        {left_color, right_color});
    }
  }

  void color_image_callback(
    const Image::ConstSharedPtr & left_color_message,
    const Image::ConstSharedPtr & right_color_message)
  {
    process_messages(
      left_color_message, right_color_message,
      nullptr, nullptr,
      {left_color_message, right_color_message});
  }

  void try_process_latest_colors()
  {
    if (!latest_left_color_ || !latest_right_color_) {
      return;
    }
    const int64_t left_stamp =
      stamp_nanoseconds(latest_left_color_->header.stamp);
    const int64_t right_stamp =
      stamp_nanoseconds(latest_right_color_->header.stamp);
    const double arrival_span_ms =
      std::chrono::duration<double, std::milli>(
      latest_left_arrival_ - latest_right_arrival_).count();
    if (std::abs(arrival_span_ms) > sync_slop_ms_) {
      return;
    }
    if (left_stamp == last_left_color_stamp_ ||
      right_stamp == last_right_color_stamp_)
    {
      return;
    }

    const auto left = latest_left_color_;
    const auto right = latest_right_color_;
    last_left_color_stamp_ = left_stamp;
    last_right_color_stamp_ = right_stamp;
    color_image_callback(left, right);
  }

  void process_messages(
    const Image::ConstSharedPtr & left_color_message,
    const Image::ConstSharedPtr & right_color_message,
    const Image::ConstSharedPtr & left_depth_message,
    const Image::ConstSharedPtr & right_depth_message,
    const std::vector<Image::ConstSharedPtr> & messages)
  {
    const auto callback_start = std::chrono::steady_clock::now();
    try {
      const cv::Mat left_color = to_bgr(left_color_message);
      const cv::Mat right_color = to_bgr(right_color_message);
      const cv::Mat left_depth = left_depth_message ?
        to_depth(left_depth_message) : cv::Mat();
      const cv::Mat right_depth = right_depth_message ?
        to_depth(right_depth_message) : cv::Mat();
      cv::Mat panorama;
      {
        const std::lock_guard<std::mutex> lock(projection_mutex_);
        panorama = stitch_rgbd(
          left_color, left_depth, right_color, right_depth);
      }

      auto newest_message = messages.front();
      int64_t minimum_stamp = stamp_nanoseconds(
        messages.front()->header.stamp);
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
      output_publisher_->publish(
        *cv_bridge::CvImage(
          output_header, sensor_msgs::image_encodings::BGR8,
          panorama).toImageMsg());
      if (
        publish_auxiliary_outputs_ &&
        !last_validity_mask_.empty() &&
        !last_range_m_.empty())
      {
        validity_publisher_->publish(
          *cv_bridge::CvImage(
            output_header, sensor_msgs::image_encodings::MONO8,
            last_validity_mask_).toImageMsg());
        range_publisher_->publish(
          *cv_bridge::CvImage(
            output_header, sensor_msgs::image_encodings::TYPE_32FC1,
            last_range_m_).toImageMsg());
        last_validity_ratio_ =
          static_cast<double>(cv::countNonZero(last_validity_mask_)) /
          static_cast<double>(last_validity_mask_.total());
      }

      const double sync_span_ms =
        static_cast<double>(maximum_stamp - minimum_stamp) / 1e6;
      sync_span_sum_ms_ += sync_span_ms;
      sync_span_max_ms_ = std::max(sync_span_max_ms_, sync_span_ms);
      ++frame_count_;
      ++diagnostic_frame_count_;
      const auto callback_end = std::chrono::steady_clock::now();
      processing_time_sum_ms_ +=
        std::chrono::duration<double, std::milli>(
        callback_end - callback_start).count();
      maybe_log_diagnostics(callback_end);
    } catch (const cv_bridge::Exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "cv_bridge conversion failed: %s", error.what());
    } catch (const cv::Exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "OpenCV RGB-D panorama failed: %s", error.what());
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "RGB-D panorama failed: %s", error.what());
    }
  }

  void maybe_log_diagnostics(
    const std::chrono::steady_clock::time_point & now)
  {
    const double elapsed_sec =
      std::chrono::duration<double>(now - last_diagnostics_time_).count();
    if (elapsed_sec < diagnostics_period_sec_ || diagnostic_frame_count_ == 0) {
      return;
    }

    const double count = static_cast<double>(diagnostic_frame_count_);
    const double left_color_input_hz =
      static_cast<double>(received_left_color_count_.exchange(0)) /
      elapsed_sec;
    const double left_depth_input_hz =
      static_cast<double>(received_left_depth_count_.exchange(0)) /
      elapsed_sec;
    const double right_color_input_hz =
      static_cast<double>(received_right_color_count_.exchange(0)) /
      elapsed_sec;
    const double right_depth_input_hz =
      static_cast<double>(received_right_depth_count_.exchange(0)) /
      elapsed_sec;
    RCLCPP_INFO(
      get_logger(),
      "output=%dx%d fps=%.1f processing=%.1f ms backend=%s gpu=%.1f ms "
      "input_hz(Lc/Ld/Rc/Rd)=%.1f/%.1f/%.1f/%.1f "
      "depth_age(L/R)=%.1f/%.1f ms "
      "sync_span(avg/max)=%.1f/%.1f ms depth_points(left/right)=%zu/%zu "
      "validity=%.1f%% "
      "gain(BGR)=%.2f/%.2f/%.2f total=%zu",
      panorama_width_, panorama_height_, count / elapsed_sec,
      processing_time_sum_ms_ / count,
      used_cuda_last_frame_ ? "CUDA" : "CPU", last_gpu_time_ms_,
      left_color_input_hz, left_depth_input_hz,
      right_color_input_hz, right_depth_input_hz,
      last_left_depth_age_ms_, last_right_depth_age_ms_,
      sync_span_sum_ms_ / count, sync_span_max_ms_,
      last_left_depth_points_, last_right_depth_points_,
      100.0 * last_validity_ratio_,
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
  std::string left_camera_info_topic_;
  std::string right_color_topic_;
  std::string right_depth_topic_;
  std::string right_camera_info_topic_;
  std::string output_topic_;
  std::string validity_topic_;
  std::string range_topic_;
  std::string output_frame_id_;
  bool publish_auxiliary_outputs_{false};
  bool publisher_best_effort_{false};

  int sync_queue_size_{50};
  double sync_slop_ms_{45.0};
  bool input_images_rotated_180_{true};
  bool rotate_color_180_{false};
  bool rotate_aligned_depth_180_{false};
  double baseline_m_{0.10};
  double projection_scale_{0.5};
  std::string projection_model_{"cylindrical"};
  double color_reference_plane_z_m_{0.0};
  int rectilinear_width_{3754};
  int rectilinear_height_{1071};
  double rectilinear_fy_px_{1373.0};
  bool rectilinear_auto_height_{false};
  double depth_scale_m_{0.001};
  double min_depth_m_{0.20};
  double max_depth_m_{15.0};
  double depth_overlap_margin_deg_{2.0};
  bool full_depth_reprojection_{false};
  double depth_discontinuity_abs_m_{0.08};
  double depth_discontinuity_relative_{0.04};
  int depth_splat_radius_px_{1};
  int depth_edge_splat_radius_px_{0};
  int projected_hole_radius_px_{0};
  bool allow_color_fallback_{true};
  double seam_angle_deg_{0.0};
  bool auto_seam_center_{false};
  int seam_feather_px_{2};
  bool depth_aware_color_{true};
  bool render_depth_reprojected_color_{true};
  bool use_rgbd_synchronization_{true};
  bool depth_temporal_stabilization_{true};
  double depth_temporal_alpha_{0.35};
  double depth_temporal_reset_m_{0.05};
  int depth_median_kernel_{3};
  double occlusion_switch_margin_m_{0.05};
  bool prefer_seam_camera_when_both_depth_valid_{false};
  bool exposure_compensation_{true};
  double exposure_smoothing_{0.15};
  double min_exposure_gain_{0.75};
  double max_exposure_gain_{1.33};
  double diagnostics_period_sec_{2.0};
  bool use_cuda_{true};

  CameraModel left_model_;
  CameraModel right_model_;
  bool projection_dirty_{true};
  int source_width_{0};
  int source_height_{0};
  int panorama_width_{0};
  int panorama_height_{0};
  int seam_x_{0};
  double panorama_focal_px_{0.0};
  double panorama_min_angle_{0.0};
  double panorama_max_angle_{0.0};
  double panorama_min_vertical_{0.0};
  double virtual_fx_px_{0.0};
  double virtual_fy_px_{0.0};
  double virtual_cx_px_{0.0};
  double virtual_cy_px_{0.0};
  double left_min_angle_{0.0};
  double left_max_angle_{0.0};
  double right_min_angle_{0.0};
  double right_max_angle_{0.0};
  double overlap_min_angle_{0.0};
  double overlap_max_angle_{0.0};
  cv::Mat left_map_x_;
  cv::Mat left_map_y_;
  cv::Mat right_map_x_;
  cv::Mat right_map_y_;
  cv::Mat left_base_mask_;
  cv::Mat right_base_mask_;
  cv::Mat left_stabilized_depth_m_;
  cv::Mat right_stabilized_depth_m_;
  cv::Vec3d smoothed_gain_{1.0, 1.0, 1.0};
  cv::Mat last_validity_mask_;
  cv::Mat last_range_m_;
#ifdef PANORAMA_WITH_CUDA
  std::unique_ptr<CudaPanoramaBackend> cuda_backend_;
  bool cuda_backend_configured_{false};
  bool cuda_backend_failed_{false};
#endif
  bool used_cuda_last_frame_{false};
  double last_gpu_time_ms_{0.0};

  rclcpp::Subscription<Image>::SharedPtr left_color_rgbd_subscriber_;
  rclcpp::Subscription<Image>::SharedPtr left_depth_rgbd_subscriber_;
  rclcpp::Subscription<Image>::SharedPtr right_color_rgbd_subscriber_;
  rclcpp::Subscription<Image>::SharedPtr right_depth_rgbd_subscriber_;
  rclcpp::Subscription<Image>::SharedPtr left_color_direct_subscriber_;
  rclcpp::Subscription<Image>::SharedPtr right_color_direct_subscriber_;
  Image::ConstSharedPtr latest_left_color_;
  Image::ConstSharedPtr latest_right_color_;
  std::chrono::steady_clock::time_point latest_left_arrival_{};
  std::chrono::steady_clock::time_point latest_right_arrival_{};
  int64_t last_left_color_stamp_{std::numeric_limits<int64_t>::min()};
  int64_t last_right_color_stamp_{std::numeric_limits<int64_t>::min()};
  rclcpp::Subscription<CameraInfo>::SharedPtr left_camera_info_subscriber_;
  rclcpp::Subscription<CameraInfo>::SharedPtr right_camera_info_subscriber_;
  rclcpp::Publisher<Image>::SharedPtr output_publisher_;
  rclcpp::Publisher<Image>::SharedPtr validity_publisher_;
  rclcpp::Publisher<Image>::SharedPtr range_publisher_;
  rclcpp::CallbackGroup::SharedPtr image_callback_group_;

  std::mutex projection_mutex_;
  std::mutex processing_mutex_;
  std::condition_variable processing_condition_;
  std::thread processing_thread_;
  bool stop_processing_{false};
  std::size_t pending_sequence_{0};
  Image::ConstSharedPtr pending_left_color_;
  Image::ConstSharedPtr pending_right_color_;
  Image::ConstSharedPtr pending_left_depth_;
  Image::ConstSharedPtr pending_right_depth_;
  std::deque<Image::ConstSharedPtr> left_color_queue_;
  std::deque<Image::ConstSharedPtr> right_color_queue_;
  std::deque<Image::ConstSharedPtr> left_depth_queue_;
  std::deque<Image::ConstSharedPtr> right_depth_queue_;
  std::atomic<std::size_t> received_left_color_count_{0};
  std::atomic<std::size_t> received_left_depth_count_{0};
  std::atomic<std::size_t> received_right_color_count_{0};
  std::atomic<std::size_t> received_right_depth_count_{0};
  double last_left_depth_age_ms_{0.0};
  double last_right_depth_age_ms_{0.0};

  std::chrono::steady_clock::time_point last_diagnostics_time_;
  size_t frame_count_{0};
  size_t diagnostic_frame_count_{0};
  size_t last_left_depth_points_{0};
  size_t last_right_depth_points_{0};
  double sync_span_sum_ms_{0.0};
  double sync_span_max_ms_{0.0};
  double processing_time_sum_ms_{0.0};
  double last_validity_ratio_{0.0};
};

}  // namespace panorama_stitcher

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  const auto node =
    std::make_shared<panorama_stitcher::RgbdPanoramaStitcherNode>();
  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
