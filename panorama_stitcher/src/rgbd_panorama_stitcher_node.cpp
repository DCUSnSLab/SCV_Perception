#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef __GLIBC__
#include <malloc.h>
#endif

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmw/qos_profiles.h>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#ifdef PANORAMA_WITH_CUDA
#include "cuda_panorama_backend.hpp"
#endif

namespace panorama_stitcher
{

using CameraInfo = sensor_msgs::msg::CameraInfo;
using Image = sensor_msgs::msg::Image;
using PointCloud2 = sensor_msgs::msg::PointCloud2;
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

// What a single frame has to produce. Everything here is derived from live
// subscription counts, so an unobserved output costs neither GPU time,
// PCIe bandwidth nor DDS traffic.
struct FrameDemand
{
  bool image{false};
  bool validity{false};
  bool range{false};
  bool pointcloud{false};
  // The range/validity images are only downloaded when a consumer needs them
  // on the host: the debug publishers, or the CPU point-cloud fallback.
  bool host_validity{false};
  bool host_range{false};

  bool any() const
  {
    return image || validity || range || pointcloud;
  }
};

class RgbdPanoramaStitcherNode : public rclcpp::Node
{
public:
  RgbdPanoramaStitcherNode()
  : Node("panorama_stitcher"),
    last_diagnostics_time_(std::chrono::steady_clock::now())
  {
    left_color_topic_ = declare_parameter<std::string>(
      "left_color_topic", "/front_left/front_left/color/image_raw");
    left_depth_topic_ = declare_parameter<std::string>(
      "left_depth_topic", "/front_left/front_left/aligned_depth_to_color/image_raw");
    left_camera_info_topic_ = declare_parameter<std::string>(
      "left_camera_info_topic", "/front_left/front_left/color/camera_info");
    right_color_topic_ = declare_parameter<std::string>(
      "right_color_topic", "/front_right/front_right/color/image_raw");
    right_depth_topic_ = declare_parameter<std::string>(
      "right_depth_topic", "/front_right/front_right/aligned_depth_to_color/image_raw");
    right_camera_info_topic_ = declare_parameter<std::string>(
      "right_camera_info_topic", "/front_right/front_right/color/camera_info");
    output_topic_ = declare_parameter<std::string>(
      "output_topic", "/panorama/image_raw");
    validity_topic_ = declare_parameter<std::string>(
      "validity_topic", "/panorama/validity");
    range_topic_ = declare_parameter<std::string>(
      "range_topic", "/panorama/range");
    pointcloud_topic_ = declare_parameter<std::string>(
      "pointcloud_topic", "/panorama/points");
    output_frame_id_ = declare_parameter<std::string>(
      "output_frame_id", "panorama_optical_frame");
    publish_auxiliary_outputs_ = declare_parameter<bool>(
      "publish_auxiliary_outputs", false);
    publish_validity_output_ = declare_parameter<bool>(
      "publish_validity_output", publish_auxiliary_outputs_);
    publish_range_output_ = declare_parameter<bool>(
      "publish_range_output", publish_auxiliary_outputs_);
    auxiliary_output_rate_hz_ = declare_parameter<double>(
      "auxiliary_output_rate_hz", 0.0);
    publish_pointcloud_ = declare_parameter<bool>(
      "publish_pointcloud", false);
    pointcloud_stride_ = declare_parameter<int>(
      "pointcloud_stride", 4);
    publisher_best_effort_ = declare_parameter<bool>(
      "publisher_best_effort", false);
    auxiliary_publisher_best_effort_ = declare_parameter<bool>(
      "auxiliary_publisher_best_effort", true);
    pointcloud_publisher_best_effort_ = declare_parameter<bool>(
      "pointcloud_publisher_best_effort", true);
    max_output_rate_hz_ = declare_parameter<double>(
      "max_output_rate_hz", 0.0);
    // A 3220x919 panorama is roughly 9 MB per frame and the range image another
    // 12 MB. Producing them while nothing is subscribed burns CPU, GPU and DDS
    // bandwidth for nobody, so every output is gated on demand by default.
    publish_only_when_subscribed_ = declare_parameter<bool>(
      "publish_only_when_subscribed", true);
    // Exposure gain is a slowly varying mean ratio. Sampling every Nth pixel
    // of the 1080p sources gives the same gain for a small fraction of the CPU.
    exposure_sample_stride_ = declare_parameter<int>(
      "exposure_sample_stride", 4);
    // Keep the reliable reader. Best-effort looks attractive because this node
    // only ever uses the newest synchronized set, but measured against these
    // cameras it makes the middleware discard most 6 MB colour samples before
    // the callback sees them (colour fell to 1-7 Hz while depth stayed near
    // 30 Hz), and the four-stream timestamp match then almost never succeeds.
    input_best_effort_ = declare_parameter<bool>("input_best_effort", false);

    sync_queue_size_ = declare_parameter<int>("sync_queue_size", 50);
    sync_slop_ms_ = declare_parameter<double>("sync_slop_ms", 45.0);
    input_images_rotated_180_ = declare_parameter<bool>(
      "input_images_rotated_180", true);
    left_input_image_rotated_180_ = declare_parameter<bool>(
      "left_input_image_rotated_180", input_images_rotated_180_);
    right_input_image_rotated_180_ = declare_parameter<bool>(
      "right_input_image_rotated_180", input_images_rotated_180_);
    rotate_color_180_ = declare_parameter<bool>(
      "rotate_color_180", false);
    rotate_aligned_depth_180_ = declare_parameter<bool>(
      "rotate_aligned_depth_180", false);

    const double half_yaw_deg = declare_parameter<double>(
      "camera_half_yaw_deg", 30.397);
    baseline_m_ = declare_parameter<double>(
      "camera_baseline_m", 0.06339742196001438);
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
    cuda_depth_spatial_filter_ = declare_parameter<bool>(
      "cuda_depth_spatial_filter", false);
    cuda_depth_spatial_delta_m_ = declare_parameter<double>(
      "cuda_depth_spatial_delta_m", 0.03);
    cuda_depth_spatial_delta_relative_ = declare_parameter<double>(
      "cuda_depth_spatial_delta_relative", 0.01);
    cuda_depth_temporal_filter_ = declare_parameter<bool>(
      "cuda_depth_temporal_filter", false);
    cuda_depth_temporal_alpha_ = declare_parameter<double>(
      "cuda_depth_temporal_alpha", 0.65);
    cuda_depth_temporal_reset_m_ = declare_parameter<double>(
      "cuda_depth_temporal_reset_m", 0.08);
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
    depth_color_overlap_only_ = declare_parameter<bool>(
      "depth_color_overlap_only", false);
    depth_color_band_margin_deg_ = declare_parameter<double>(
      "depth_color_band_margin_deg", 0.0);
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
    content_aware_seam_ = declare_parameter<bool>(
      "content_aware_seam", false);
    seam_color_weight_ = declare_parameter<double>(
      "seam_color_weight", 1.0);
    seam_depth_weight_ = declare_parameter<double>(
      "seam_depth_weight", 2.0);
    seam_foreground_weight_ = declare_parameter<double>(
      "seam_foreground_weight", 0.35);
    seam_center_weight_ = declare_parameter<double>(
      "seam_center_weight", 0.03);
    seam_temporal_weight_ = declare_parameter<double>(
      "seam_temporal_weight", 0.08);
    seam_max_step_px_ = declare_parameter<int>(
      "seam_max_step_px", 3);
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
    cuda_timeout_ms_ = declare_parameter<int>("cuda_timeout_ms", 500);
    cuda_slow_frame_ms_ = declare_parameter<double>(
      "cuda_slow_frame_ms", 150.0);
    cuda_slow_frame_limit_ = declare_parameter<int>(
      "cuda_slow_frame_limit", 3);

    set_parameter_camera_model(
      left_model_, "left", 1375.93896484375, 1376.0078125,
      962.9755859375, 539.9728393554688);
    set_parameter_camera_model(
      right_model_, "right", 1369.7860107421875, 1369.6165771484375,
      967.3739013671875, 566.1657104492188);
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

    const auto output_qos = make_output_qos(publisher_best_effort_);
    output_publisher_ = create_publisher<Image>(output_topic_, output_qos);
    if (publish_validity_output_ || publish_range_output_) {
      const auto auxiliary_qos = make_output_qos(
        auxiliary_publisher_best_effort_);
      if (publish_validity_output_) {
        validity_publisher_ = create_publisher<Image>(
          validity_topic_, auxiliary_qos);
      }
      if (publish_range_output_) {
        range_publisher_ = create_publisher<Image>(
          range_topic_, auxiliary_qos);
      }
    }
    if (publish_pointcloud_) {
      const auto pointcloud_qos = make_output_qos(
        pointcloud_publisher_best_effort_);
      pointcloud_publisher_ = create_publisher<PointCloud2>(
        pointcloud_topic_, pointcloud_qos);
    }

    // CameraInfo arrives at the full camera rate. Publishing it into a small
    // staging slot keeps these callbacks off projection_mutex_, which the
    // processing thread holds for the whole stitch.
    pending_left_model_ = left_model_;
    pending_right_model_ = right_model_;
    const auto camera_info_qos =
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
    left_camera_info_subscriber_ = create_subscription<CameraInfo>(
      left_camera_info_topic_, camera_info_qos,
      [this](const CameraInfo::ConstSharedPtr message) {
        stage_camera_model(
          pending_left_model_, *message, "left",
          left_input_image_rotated_180_);
      });
    right_camera_info_subscriber_ = create_subscription<CameraInfo>(
      right_camera_info_topic_, camera_info_qos,
      [this](const CameraInfo::ConstSharedPtr message) {
        stage_camera_model(
          pending_right_model_, *message, "right",
          right_input_image_rotated_180_);
      });

    if (depth_aware_color_ || use_rgbd_synchronization_) {
      image_callback_group_ = create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);
      rclcpp::SubscriptionOptions image_subscription_options;
      image_subscription_options.callback_group = image_callback_group_;
      const auto color_qos = make_input_qos();
      const auto depth_qos = make_input_qos();
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
  rclcpp::QoS make_input_qos() const
  {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile();
    if (input_best_effort_) {
      qos.best_effort();
    } else {
      qos.reliable();
    }
    return qos;
  }

  static rclcpp::QoS make_output_qos(bool best_effort)
  {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile();
    if (best_effort) {
      qos.best_effort();
    } else {
      qos.reliable();
    }
    return qos;
  }

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
    const bool image_rotated_180 =
      prefix == "left" ? left_input_image_rotated_180_ :
      right_input_image_rotated_180_;
    adjust_intrinsics_for_input_rotation(model, image_rotated_180);
  }

  void validate_parameters()
  {
    // Four synchronized 1080p streams are large. Bound the history even when
    // a stale configuration requests an excessive queue.
    sync_queue_size_ = std::clamp(sync_queue_size_, 4, 16);
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
    depth_color_band_margin_deg_ = std::max(
      depth_color_band_margin_deg_, 0.0);
    depth_discontinuity_abs_m_ = std::max(
      depth_discontinuity_abs_m_, 0.0);
    depth_discontinuity_relative_ = std::max(
      depth_discontinuity_relative_, 0.0);
    cuda_depth_spatial_delta_m_ = std::max(
      cuda_depth_spatial_delta_m_, 0.0);
    cuda_depth_spatial_delta_relative_ = std::max(
      cuda_depth_spatial_delta_relative_, 0.0);
    cuda_depth_temporal_alpha_ = std::clamp(
      cuda_depth_temporal_alpha_, 0.01, 1.0);
    cuda_depth_temporal_reset_m_ = std::max(
      cuda_depth_temporal_reset_m_, 0.0);
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
    seam_color_weight_ = std::max(seam_color_weight_, 0.0);
    seam_depth_weight_ = std::max(seam_depth_weight_, 0.0);
    seam_foreground_weight_ = std::max(
      seam_foreground_weight_, 0.0);
    seam_center_weight_ = std::max(seam_center_weight_, 0.0);
    seam_temporal_weight_ = std::max(
      seam_temporal_weight_, 0.0);
    seam_max_step_px_ = std::clamp(seam_max_step_px_, 1, 16);
    exposure_smoothing_ = std::clamp(exposure_smoothing_, 0.0, 1.0);
    min_exposure_gain_ = std::max(min_exposure_gain_, 0.01);
    max_exposure_gain_ = std::max(max_exposure_gain_, min_exposure_gain_);
    diagnostics_period_sec_ = std::max(diagnostics_period_sec_, 0.2);
    cuda_timeout_ms_ = std::clamp(cuda_timeout_ms_, 50, 5000);
    cuda_slow_frame_ms_ = std::max(cuda_slow_frame_ms_, 0.0);
    cuda_slow_frame_limit_ = std::clamp(cuda_slow_frame_limit_, 1, 100);
    pointcloud_stride_ = std::clamp(pointcloud_stride_, 1, 16);
    max_output_rate_hz_ = std::max(max_output_rate_hz_, 0.0);
    exposure_sample_stride_ = std::clamp(exposure_sample_stride_, 1, 16);
    projection_is_rectilinear_ = projection_model_ == "rectilinear";
  }

  static void adjust_intrinsics_for_input_rotation(
    CameraModel & model, bool image_rotated_180)
  {
    if (!image_rotated_180) {
      return;
    }
    model.cx = static_cast<double>(model.width - 1) - model.cx;
    model.cy = static_cast<double>(model.height - 1) - model.cy;
  }

  static bool intrinsics_differ(
    const CameraModel & model, const CameraModel & updated)
  {
    return
      !model.valid() ||
      model.width != updated.width ||
      model.height != updated.height ||
      std::abs(model.fx - updated.fx) > 1e-6 ||
      std::abs(model.fy - updated.fy) > 1e-6 ||
      std::abs(model.cx - updated.cx) > 1e-6 ||
      std::abs(model.cy - updated.cy) > 1e-6;
  }

  // Runs in the CameraInfo callback. It only touches the small staging model
  // under a dedicated mutex, so it never waits for a frame to finish stitching.
  void stage_camera_model(
    CameraModel & staged, const CameraInfo & message,
    const char * camera_name, bool image_rotated_180)
  {
    const std::lock_guard<std::mutex> lock(camera_info_mutex_);
    CameraModel updated = staged;
    updated.fx = message.k[0];
    updated.fy = message.k[4];
    updated.cx = message.k[2];
    updated.cy = message.k[5];
    updated.width = static_cast<int>(message.width);
    updated.height = static_cast<int>(message.height);
    adjust_intrinsics_for_input_rotation(updated, image_rotated_180);
    if (!intrinsics_differ(staged, updated)) {
      return;
    }

    staged = updated;
    camera_info_staged_ = true;
    RCLCPP_INFO(
      get_logger(),
      "%s CameraInfo: %dx%d fx/fy=%.3f/%.3f cx/cy=%.3f/%.3f%s",
      camera_name, updated.width, updated.height,
      updated.fx, updated.fy, updated.cx, updated.cy,
      image_rotated_180 ? " (adjusted for 180 deg image rotation)" : "");
  }

  // Called by the processing thread while it already owns projection_mutex_.
  void apply_staged_camera_models()
  {
    const std::lock_guard<std::mutex> lock(camera_info_mutex_);
    if (!camera_info_staged_) {
      return;
    }
    camera_info_staged_ = false;
    if (intrinsics_differ(left_model_, pending_left_model_)) {
      left_model_ = pending_left_model_;
      projection_dirty_ = true;
    }
    if (intrinsics_differ(right_model_, pending_right_model_)) {
      right_model_ = pending_right_model_;
      projection_dirty_ = true;
    }
  }

  static int64_t stamp_nanoseconds(const builtin_interfaces::msg::Time & stamp)
  {
    return static_cast<int64_t>(stamp.sec) * 1000000000LL +
           static_cast<int64_t>(stamp.nanosec);
  }

  struct SourceColor
  {
    cv::Mat image;
    bool is_rgb{false};
  };

  // The RealSense wrapper publishes rgb8. Asking cv_bridge for bgr8 forced a
  // full 1920x1080 conversion plus allocation for both cameras on every frame;
  // the channel order is now handled where the pixels are already being read.
  SourceColor to_source_color(const Image::ConstSharedPtr & message) const
  {
    SourceColor source;
    if (message->encoding == sensor_msgs::image_encodings::RGB8) {
      source.image = cv_bridge::toCvShare(message, message->encoding)->image;
      source.is_rgb = true;
    } else if (message->encoding == sensor_msgs::image_encodings::BGR8) {
      source.image = cv_bridge::toCvShare(message, message->encoding)->image;
    } else {
      source.image = cv_bridge::toCvCopy(
        message, sensor_msgs::image_encodings::BGR8)->image;
    }
    if (rotate_color_180_) {
      cv::Mat rotated;
      cv::rotate(source.image, rotated, cv::ROTATE_180);
      source.image = rotated;
    }
    return source;
  }

  cv::Mat to_depth(const Image::ConstSharedPtr & message) const
  {
    const cv::Mat depth = cv_bridge::toCvShare(
      message, message->encoding)->image;
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
    apply_staged_camera_models();
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
              projection_is_rectilinear_ ?
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

    if (projection_is_rectilinear_) {
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

    const auto panorama_x_for_angle = [this](double angle) {
        return projection_is_rectilinear_ ?
          virtual_fx_px_ * std::tan(angle) + virtual_cx_px_ :
          (angle - panorama_min_angle_) * panorama_focal_px_;
      };
    const double depth_color_margin =
      depth_color_band_margin_deg_ * kPi / 180.0;
    depth_color_min_x_ = std::clamp(
      static_cast<int>(std::ceil(
        panorama_x_for_angle(
          overlap_min_angle_ - depth_color_margin))),
      0, panorama_width_ - 1);
    depth_color_max_x_ = std::clamp(
      static_cast<int>(std::floor(
        panorama_x_for_angle(
          overlap_max_angle_ + depth_color_margin))),
      0, panorama_width_ - 1);

    build_inverse_map(
      left_model_, left_map_x_, left_map_y_, left_base_mask_);
    build_inverse_map(
      right_model_, right_map_x_, right_map_y_, right_base_mask_);

    const double seam_angle = auto_seam_center_ ?
      0.5 * (overlap_min_angle_ + overlap_max_angle_) :
      seam_angle_deg_ * kPi / 180.0;
    if (projection_is_rectilinear_) {
      seam_x_ = static_cast<int>(std::lround(
        virtual_fx_px_ * std::tan(seam_angle) + virtual_cx_px_));
    } else {
      seam_x_ = static_cast<int>(std::lround(
        (seam_angle - panorama_min_angle_) * panorama_focal_px_));
    }
    seam_x_ = std::clamp(seam_x_, 0, panorama_width_ - 1);

    // depth_source_columns() scans the full source width with trigonometry.
    // The result only depends on the projection, so cache it here instead of
    // recomputing it several times per frame.
    left_depth_columns_ = depth_source_columns(left_model_);
    right_depth_columns_ = depth_source_columns(right_model_);
    depth_columns_valid_ = true;

    if (!projection_is_rectilinear_) {
      column_sin_.resize(static_cast<std::size_t>(panorama_width_));
      column_cos_.resize(static_cast<std::size_t>(panorama_width_));
      for (int x = 0; x < panorama_width_; ++x) {
        const double angle = panorama_min_angle_ +
          static_cast<double>(x) / panorama_focal_px_;
        column_sin_[static_cast<std::size_t>(x)] = std::sin(angle);
        column_cos_[static_cast<std::size_t>(x)] = std::cos(angle);
      }
    } else {
      column_sin_.clear();
      column_cos_.clear();
    }
    projection_dirty_ = false;
#ifdef PANORAMA_WITH_CUDA
    cuda_backend_configured_ = false;
#endif

    RCLCPP_INFO(
      get_logger(),
      "Projection (%s): %dx%d, angular view %.1f..%.1f deg, "
      "overlap %.1f..%.1f deg (x=%d..%d), seam x=%d",
      projection_model_.c_str(), panorama_width_, panorama_height_,
      panorama_min_angle_ * 180.0 / kPi,
      panorama_max_angle_ * 180.0 / kPi,
      overlap_min_angle_ * 180.0 / kPi,
      overlap_max_angle_ * 180.0 / kPi,
      depth_color_min_x_, depth_color_max_x_,
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
        if (projection_is_rectilinear_) {
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

  // Cached counterpart of depth_source_columns() for the per-frame paths.
  std::array<int, 2> depth_columns_for(const CameraModel & model) const
  {
    if (!depth_columns_valid_) {
      return depth_source_columns(model);
    }
    return &model == &left_model_ ?
           left_depth_columns_ : right_depth_columns_;
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

    const auto columns = depth_columns_for(model);
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
        if (projection_is_rectilinear_) {
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
    const auto columns = depth_columns_for(model);
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

  // Mean of the well-exposed pixels on a coarse grid. The previous version
  // built two grayscale images, four full-resolution masks and two masked means
  // over both 1920x1080 sources on every frame; sampling every Nth pixel gives
  // the same slowly varying gain for a fraction of that cost and allocates
  // nothing. Channel sums stay in source order.
  bool sample_channel_means(
    const cv::Mat & image, const std::array<int, 2> & columns,
    std::array<double, 3> & means) const
  {
    const int first_column = std::clamp(columns[0], 0, image.cols - 1);
    const int last_column = std::clamp(columns[1], first_column, image.cols - 1);
    const int stride = exposure_sample_stride_;
    const bool source_is_rgb = source_is_rgb_;
    std::array<uint64_t, 3> sums{0, 0, 0};
    std::size_t count = 0;
    for (int y = 0; y < image.rows; y += stride) {
      const uint8_t * row = image.ptr<uint8_t>(y);
      for (int x = first_column; x <= last_column; x += stride) {
        const uint8_t * pixel = row + static_cast<std::size_t>(x) * 3U;
        const double luma = source_is_rgb ?
          0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2] :
          0.114 * pixel[0] + 0.587 * pixel[1] + 0.299 * pixel[2];
        if (luma <= 25.0 || luma >= 235.0) {
          continue;
        }
        sums[0] += pixel[0];
        sums[1] += pixel[1];
        sums[2] += pixel[2];
        ++count;
      }
    }
    if (count < 500) {
      return false;
    }
    for (int channel = 0; channel < 3; ++channel) {
      means[channel] = static_cast<double>(sums[channel]) /
        static_cast<double>(count);
    }
    return true;
  }

  cv::Vec3d estimate_right_gain_from_sources(
    const cv::Mat & left, const cv::Mat & right)
  {
    if (!exposure_compensation_) {
      return smoothed_gain_;
    }
    const auto left_columns = depth_columns_for(left_model_);
    const auto right_columns = depth_columns_for(right_model_);
    if (
      left_columns[1] < left_columns[0] ||
      right_columns[1] < right_columns[0])
    {
      return smoothed_gain_;
    }

    std::array<double, 3> left_mean{0.0, 0.0, 0.0};
    std::array<double, 3> right_mean{0.0, 0.0, 0.0};
    if (
      !sample_channel_means(left, left_columns, left_mean) ||
      !sample_channel_means(right, right_columns, right_mean))
    {
      return smoothed_gain_;
    }

    for (int channel = 0; channel < 3; ++channel) {
      const double measured = std::clamp(
        left_mean[channel] / std::max(right_mean[channel], 1.0),
        min_exposure_gain_, max_exposure_gain_);
      // smoothed_gain_ is always indexed as BGR because that is what both the
      // CUDA kernels and the CPU compositor expect.
      const int output_channel = source_is_rgb_ ? 2 - channel : channel;
      smoothed_gain_[output_channel] =
        (1.0 - exposure_smoothing_) * smoothed_gain_[output_channel] +
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
    if (cuda_backend_configured_ &&
      configured_source_is_rgb_ == source_is_rgb_)
    {
      return true;
    }
    configured_source_is_rgb_ = source_is_rgb_;

    CudaPanoramaConfig config;
    config.source_width = source_width_;
    config.source_height = source_height_;
    config.panorama_width = panorama_width_;
    config.panorama_height = panorama_height_;
    config.projection_model =
      projection_is_rectilinear_ ? 1 : 0;
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
    config.depth_spatial_filter = cuda_depth_spatial_filter_;
    config.depth_spatial_delta_m =
      static_cast<float>(cuda_depth_spatial_delta_m_);
    config.depth_spatial_delta_relative =
      static_cast<float>(cuda_depth_spatial_delta_relative_);
    config.depth_temporal_filter = cuda_depth_temporal_filter_;
    config.depth_temporal_alpha =
      static_cast<float>(cuda_depth_temporal_alpha_);
    config.depth_temporal_reset_m =
      static_cast<float>(cuda_depth_temporal_reset_m_);
    config.occlusion_switch_margin_m =
      static_cast<float>(occlusion_switch_margin_m_);
    config.depth_aware_color = depth_aware_color_;
    config.render_depth_reprojected_color =
      render_depth_reprojected_color_;
    config.depth_color_overlap_only = depth_color_overlap_only_;
    config.depth_color_min_x = depth_color_min_x_;
    config.depth_color_max_x = depth_color_max_x_;
    config.allow_color_fallback = allow_color_fallback_;
    config.prefer_seam_camera_when_both_depth_valid =
      prefer_seam_camera_when_both_depth_valid_;
    config.content_aware_seam = content_aware_seam_;
    config.seam_color_weight =
      static_cast<float>(seam_color_weight_);
    config.seam_depth_weight =
      static_cast<float>(seam_depth_weight_);
    config.seam_foreground_weight =
      static_cast<float>(seam_foreground_weight_);
    config.seam_center_weight =
      static_cast<float>(seam_center_weight_);
    config.seam_temporal_weight =
      static_cast<float>(seam_temporal_weight_);
    config.seam_max_step_px = seam_max_step_px_;
    config.seam_x = seam_x_;
    config.seam_feather_px = seam_feather_px_;
    config.depth_splat_radius_px = depth_splat_radius_px_;
    config.depth_edge_splat_radius_px = depth_edge_splat_radius_px_;
    config.projected_hole_radius = projected_hole_radius_px_;
    config.depth_scale_m = static_cast<float>(depth_scale_m_);
    config.source_channel_swap = source_is_rgb_;
    config.pointcloud_stride = publish_pointcloud_ ? pointcloud_stride_ : 0;
    config.operation_timeout_ms = cuda_timeout_ms_;

    std::string error;
    if (!cuda_backend_->configure(
        config,
        make_cuda_camera_model(left_model_),
        make_cuda_camera_model(right_model_),
        left_base_mask_, right_base_mask_,
        left_map_x_, left_map_y_,
        right_map_x_, right_map_y_, error))
    {
      cuda_backend_->quarantine(error);
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

#endif

  int pointcloud_capacity() const
  {
    if (pointcloud_stride_ <= 0) {
      return 0;
    }
    return
      ((panorama_width_ + pointcloud_stride_ - 1) / pointcloud_stride_) *
      ((panorama_height_ + pointcloud_stride_ - 1) / pointcloud_stride_);
  }

  void stitch_rgbd(
    const cv::Mat & left_source_color, const cv::Mat & left_depth,
    const cv::Mat & right_source_color, const cv::Mat & right_depth,
    const FrameDemand & demand, cv::Mat & output, PointCloud2 * cloud)
  {
    cv::Mat left_projection_depth = left_depth;
    cv::Mat right_projection_depth = right_depth;
    if (depth_aware_color_ && depth_temporal_stabilization_) {
      left_projection_depth = stabilize_depth(
        left_depth, left_stabilized_depth_m_, left_model_);
      right_projection_depth = stabilize_depth(
        right_depth, right_stabilized_depth_m_, right_model_);
    }

    gpu_cloud_points_ = 0;
    gpu_cloud_filled_ = false;
#ifdef PANORAMA_WITH_CUDA
    if (ensure_cuda_backend_configured()) {
      const cv::Vec3d gain = estimate_right_gain_from_sources(
        left_source_color, right_source_color);
      CudaProcessOptions options;
      options.download_panorama = demand.image;
      options.download_validity = demand.host_validity;
      options.download_range = demand.host_range;
      CudaPointCloudRequest cloud_request;
      const int capacity = pointcloud_capacity();
      if (demand.pointcloud && cloud != nullptr && capacity > 0) {
        prepare_pointcloud_message(*cloud, capacity);
        // Read the layout back out of the message instead of assuming it.
        // PointCloud2Modifier pads an xyz+rgb cloud to 32 bytes per point with
        // rgb at offset 16.
        cloud_request.destination = cloud->data.data();
        cloud_request.capacity_points = capacity;
        cloud_request.point_step_bytes = static_cast<int>(cloud->point_step);
        cloud_request.x_offset_bytes = field_offset(*cloud, "x");
        cloud_request.y_offset_bytes = field_offset(*cloud, "y");
        cloud_request.z_offset_bytes = field_offset(*cloud, "z");
        cloud_request.rgb_offset_bytes = field_offset(*cloud, "rgb");
        options.build_pointcloud =
          cloud_request.x_offset_bytes >= 0 &&
          cloud_request.y_offset_bytes >= 0 &&
          cloud_request.z_offset_bytes >= 0 &&
          cloud_request.rgb_offset_bytes >= 0;
      }
      CudaPanoramaStats stats;
      std::string error;
      if (cuda_backend_->process(
          left_source_color, left_projection_depth,
          right_source_color, right_projection_depth, gain,
          output, last_validity_mask_, last_range_m_,
          stats, error, options,
          options.build_pointcloud ? &cloud_request : nullptr))
      {
        last_left_depth_points_ = stats.left_depth_points;
        last_right_depth_points_ = stats.right_depth_points;
        last_gpu_time_ms_ = stats.gpu_time_ms;
        last_content_aware_seam_used_ =
          stats.content_aware_seam_used;
        last_seam_min_x_ = stats.seam_min_x;
        last_seam_max_x_ = stats.seam_max_x;
        last_seam_mean_x_ = stats.seam_mean_x;
        used_cuda_last_frame_ = true;
        if (
          cuda_slow_frame_ms_ > 0.0 &&
          static_cast<double>(stats.gpu_time_ms) > cuda_slow_frame_ms_)
        {
          ++cuda_slow_frame_count_;
        } else {
          cuda_slow_frame_count_ = 0;
        }
        if (cuda_slow_frame_count_ >= cuda_slow_frame_limit_) {
          std::ostringstream reason;
          reason << "CUDA circuit breaker opened after "
                 << cuda_slow_frame_count_ << " consecutive frames over "
                 << cuda_slow_frame_ms_ << " ms (latest "
                 << stats.gpu_time_ms << " ms)";
          cuda_backend_->quarantine(reason.str());
          cuda_backend_failed_ = true;
          cuda_backend_configured_ = false;
          RCLCPP_ERROR(
            get_logger(), "%s; subsequent frames use CPU fallback",
            reason.str().c_str());
        }
        if (options.build_pointcloud) {
          gpu_cloud_points_ = cloud_request.point_count;
          gpu_cloud_filled_ = true;
          finalize_pointcloud_message(*cloud, gpu_cloud_points_);
        }
        return;
      }
      cuda_backend_->quarantine(error);
      cuda_backend_failed_ = true;
      cuda_backend_configured_ = false;
      RCLCPP_ERROR(
        get_logger(),
        "CUDA panorama processing failed (%s); using CPU fallback",
        error.c_str());
    }
#endif

    // CPU fallback. It composites in BGR, so an rgb8 source is converted once
    // here instead of on the fast path.
    cv::Mat left_color = left_source_color;
    cv::Mat right_color = right_source_color;
    if (source_is_rgb_) {
      cv::cvtColor(left_source_color, left_color, cv::COLOR_RGB2BGR);
      cv::cvtColor(right_source_color, right_color, cv::COLOR_RGB2BGR);
    }
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
        const auto render_projected_color =
          [this](const ProjectedDepth & projected, cv::Mat & base) {
            if (!depth_color_overlap_only_) {
              projected.color.copyTo(base, projected.mask);
              return;
            }
            if (depth_color_max_x_ < depth_color_min_x_) {
              return;
            }
            const cv::Rect band(
              depth_color_min_x_, 0,
              depth_color_max_x_ - depth_color_min_x_ + 1,
              panorama_height_);
            projected.color(band).copyTo(base(band), projected.mask(band));
          };
        render_projected_color(left_projected, left_base);
        render_projected_color(right_projected, right_base);
      }
    }

    right_base = apply_gain(right_base, gain);

    output.create(panorama_height_, panorama_width_, CV_8UC3);
    output.setTo(cv::Scalar::all(0));
    cv::Mat & panorama = output;

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
        const bool render_depth_color_here =
          render_depth_reprojected_color_ &&
          (!depth_color_overlap_only_ ||
          (x >= depth_color_min_x_ && x <= depth_color_max_x_));
        if (depth_aware_color_ && render_depth_color_here) {
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
      return;
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
      while (queue.size() > static_cast<std::size_t>(sync_queue_size_)) {
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

        // Wait for the next output slot while callbacks keep replacing the
        // pending set. At the deadline we process the newest synchronized
        // frames, avoiding a backlog and preserving low latency.
        if (
          max_output_rate_hz_ > 0.0 &&
          last_processing_start_.time_since_epoch().count() != 0)
        {
          const auto minimum_period =
            std::chrono::duration<double>(1.0 / max_output_rate_hz_);
          const auto next_start = last_processing_start_ +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            minimum_period);
          processing_condition_.wait_until(
            lock, next_start, [this]() {return stop_processing_;});
          if (stop_processing_) {
            return;
          }
        }
        left_color = pending_left_color_;
        right_color = pending_right_color_;
        left_depth = pending_left_depth_;
        right_depth = pending_right_depth_;
        processed_sequence = pending_sequence_;
        last_processing_start_ = std::chrono::steady_clock::now();
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

  bool has_subscribers(const rclcpp::PublisherBase::SharedPtr & publisher) const
  {
    if (!publisher) {
      return false;
    }
    return !publish_only_when_subscribed_ ||
           publisher->get_subscription_count() > 0 ||
           publisher->get_intra_process_subscription_count() > 0;
  }

  FrameDemand compute_frame_demand(bool & auxiliary_due)
  {
    FrameDemand demand;
    demand.image = has_subscribers(output_publisher_);
    demand.pointcloud = publish_pointcloud_ &&
      has_subscribers(pointcloud_publisher_);

    auxiliary_due = false;
    const bool validity_wanted = publish_validity_output_ &&
      has_subscribers(validity_publisher_);
    const bool range_wanted = publish_range_output_ &&
      has_subscribers(range_publisher_);
    if (validity_wanted || range_wanted) {
      auxiliary_due = true;
      if (auxiliary_output_rate_hz_ > 0.0 &&
        last_auxiliary_publish_time_.time_since_epoch().count() != 0)
      {
        const double elapsed_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() -
          last_auxiliary_publish_time_).count();
        auxiliary_due = elapsed_seconds >= 1.0 / auxiliary_output_rate_hz_;
      }
    }
    demand.validity = validity_wanted && auxiliary_due;
    demand.range = range_wanted && auxiliary_due;

    // Without a CUDA backend the cloud is built on the host, which needs both
    // images downloaded.
    const bool cloud_needs_host = demand.pointcloud && !used_cuda_last_frame_;
    demand.host_validity = demand.validity || cloud_needs_host;
    demand.host_range = demand.range || cloud_needs_host;
    return demand;
  }

  static int field_offset(const PointCloud2 & cloud, const char * name)
  {
    for (const auto & field : cloud.fields) {
      if (field.name == name &&
        field.datatype == sensor_msgs::msg::PointField::FLOAT32 &&
        field.count == 1)
      {
        return static_cast<int>(field.offset);
      }
    }
    return -1;
  }

  void prepare_pointcloud_message(PointCloud2 & cloud, int capacity_points)
  {
    cloud.height = 1;
    cloud.is_dense = true;
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(static_cast<std::size_t>(capacity_points));
  }

  static void finalize_pointcloud_message(
    PointCloud2 & cloud, std::size_t point_count)
  {
    cloud.width = static_cast<std::uint32_t>(point_count);
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.resize(static_cast<std::size_t>(cloud.row_step));
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
      bool auxiliary_due = false;
      validity_ratio_from_mask_this_frame_ = false;
      const FrameDemand demand = compute_frame_demand(auxiliary_due);
      if (!demand.any()) {
        ++idle_frame_count_;
        maybe_log_diagnostics(std::chrono::steady_clock::now());
        return;
      }

      const SourceColor left_source = to_source_color(left_color_message);
      const SourceColor right_source = to_source_color(right_color_message);
      cv::Mat right_color = right_source.image;
      if (left_source.is_rgb != right_source.is_rgb) {
        // Mixed encodings would corrupt the channel order of one half.
        cv::cvtColor(
          right_source.image, right_color, cv::COLOR_RGB2BGR);
      }
      source_is_rgb_ = left_source.is_rgb;
      const cv::Mat left_depth = left_depth_message ?
        to_depth(left_depth_message) : cv::Mat();
      const cv::Mat right_depth = right_depth_message ?
        to_depth(right_depth_message) : cv::Mat();

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

      // The panorama is rendered straight into the outgoing message so the
      // roughly 9 MB frame is never copied on the host.
      auto image_message = std::make_unique<Image>();
      auto cloud_message = std::make_unique<PointCloud2>();
      cv::Mat panorama;
      {
        const std::lock_guard<std::mutex> lock(projection_mutex_);
        build_projection_if_needed(
          left_source.image.cols, left_source.image.rows);
        if (demand.image) {
          image_message->header = output_header;
          image_message->height = static_cast<std::uint32_t>(panorama_height_);
          image_message->width = static_cast<std::uint32_t>(panorama_width_);
          image_message->encoding = sensor_msgs::image_encodings::BGR8;
          image_message->is_bigendian = 0U;
          image_message->step =
            static_cast<std::uint32_t>(panorama_width_) * 3U;
          image_message->data.resize(
            static_cast<std::size_t>(image_message->step) *
            static_cast<std::size_t>(panorama_height_));
          panorama = cv::Mat(
            panorama_height_, panorama_width_, CV_8UC3,
            image_message->data.data());
        }
        cloud_message->header = output_header;
        stitch_rgbd(
          left_source.image, left_depth, right_color, right_depth,
          demand, panorama, cloud_message.get());
      }

      if (
        auxiliary_due && !last_validity_mask_.empty() &&
        !last_range_m_.empty())
      {
        if (demand.validity) {
          validity_publisher_->publish(
            *cv_bridge::CvImage(
              output_header, sensor_msgs::image_encodings::MONO8,
              last_validity_mask_).toImageMsg());
        }
        if (demand.range) {
          range_publisher_->publish(
            *cv_bridge::CvImage(
              output_header, sensor_msgs::image_encodings::TYPE_32FC1,
              last_range_m_).toImageMsg());
        }
        last_auxiliary_publish_time_ = std::chrono::steady_clock::now();
        if (demand.host_validity) {
          last_validity_ratio_ =
            static_cast<double>(cv::countNonZero(last_validity_mask_)) /
            static_cast<double>(last_validity_mask_.total());
          validity_ratio_from_mask_this_frame_ = true;
        }
      }
      if (demand.pointcloud) {
        publish_panorama_pointcloud(panorama, std::move(cloud_message));
      } else {
        last_pointcloud_points_ = 0;
      }
      // Published last on purpose: `panorama` is a view into image_message's
      // buffer, and the CPU cloud fallback reads it. Handing the message to the
      // middleware before that would leave the view dangling.
      if (demand.image) {
        output_publisher_->publish(std::move(image_message));
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

  void publish_panorama_pointcloud(
    const cv::Mat & panorama, std::unique_ptr<PointCloud2> cloud)
  {
    if (!pointcloud_publisher_) {
      last_pointcloud_points_ = 0;
      return;
    }

    // The CUDA backend already compacted the cloud straight into this message.
    if (gpu_cloud_filled_) {
      last_pointcloud_points_ = gpu_cloud_points_;
      update_validity_ratio_from_cloud(gpu_cloud_points_);
      pointcloud_publisher_->publish(std::move(cloud));
      return;
    }

    const std::lock_guard<std::mutex> lock(projection_mutex_);
    if (
      panorama.empty() || last_range_m_.empty() ||
      last_validity_mask_.empty() ||
      panorama.size() != last_range_m_.size() ||
      panorama.size() != last_validity_mask_.size())
    {
      prepare_pointcloud_message(*cloud, 0);
      pointcloud_publisher_->publish(std::move(cloud));
      last_pointcloud_points_ = 0;
      return;
    }

    // Fill the message up to the sampled-grid capacity in a single pass and
    // shrink afterwards. Counting first meant walking the panorama twice.
    const int capacity = pointcloud_capacity();
    prepare_pointcloud_message(*cloud, capacity);
    const int point_step = static_cast<int>(cloud->point_step);
    const int x_offset = field_offset(*cloud, "x");
    const int y_offset = field_offset(*cloud, "y");
    const int z_offset = field_offset(*cloud, "z");
    const int rgb_offset = field_offset(*cloud, "rgb");
    if (x_offset < 0 || y_offset < 0 || z_offset < 0 || rgb_offset < 0) {
      prepare_pointcloud_message(*cloud, 0);
      pointcloud_publisher_->publish(std::move(cloud));
      last_pointcloud_points_ = 0;
      return;
    }
    const bool use_lut =
      !projection_is_rectilinear_ &&
      column_sin_.size() == static_cast<std::size_t>(panorama.cols);
    std::size_t point_count = 0;
    std::uint8_t * points = cloud->data.data();
    for (int row = 0; row < panorama.rows; row += pointcloud_stride_) {
      const auto * color_row = panorama.ptr<cv::Vec3b>(row);
      const auto * range_row = last_range_m_.ptr<float>(row);
      const auto * validity_row = last_validity_mask_.ptr<std::uint8_t>(row);
      const double vertical_ratio =
        (panorama_min_vertical_ + static_cast<double>(row)) /
        panorama_focal_px_;
      for (int column = 0; column < panorama.cols;
        column += pointcloud_stride_)
      {
        const float horizontal_range = range_row[column];
        if (
          validity_row[column] == 0 ||
          !std::isfinite(horizontal_range) || horizontal_range <= 0.0F)
        {
          continue;
        }
        if (static_cast<int>(point_count) >= capacity) {
          break;
        }

        double x;
        double y;
        double z;
        if (projection_is_rectilinear_) {
          const double ray_x =
            (static_cast<double>(column) - virtual_cx_px_) / virtual_fx_px_;
          const double ray_y =
            (static_cast<double>(row) - virtual_cy_px_) / virtual_fy_px_;
          z = static_cast<double>(horizontal_range) /
            std::hypot(ray_x, 1.0);
          x = ray_x * z;
          y = ray_y * z;
        } else {
          const double sine = use_lut ?
            column_sin_[static_cast<std::size_t>(column)] :
            std::sin(
            panorama_min_angle_ +
            static_cast<double>(column) / panorama_focal_px_);
          const double cosine = use_lut ?
            column_cos_[static_cast<std::size_t>(column)] :
            std::cos(
            panorama_min_angle_ +
            static_cast<double>(column) / panorama_focal_px_);
          x = static_cast<double>(horizontal_range) * sine;
          y = static_cast<double>(horizontal_range) * vertical_ratio;
          z = static_cast<double>(horizontal_range) * cosine;
        }

        const cv::Vec3b bgr = color_row[column];
        const std::uint32_t rgb =
          (static_cast<std::uint32_t>(bgr[2]) << 16U) |
          (static_cast<std::uint32_t>(bgr[1]) << 8U) |
          static_cast<std::uint32_t>(bgr[0]);
        std::uint8_t * destination =
          points + point_count * static_cast<std::size_t>(point_step);
        const float values[3] = {
          static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)
        };
        std::memcpy(destination + x_offset, &values[0], sizeof(float));
        std::memcpy(destination + y_offset, &values[1], sizeof(float));
        std::memcpy(destination + z_offset, &values[2], sizeof(float));
        std::memcpy(destination + rgb_offset, &rgb, sizeof(rgb));
        ++point_count;
      }
    }

    finalize_pointcloud_message(*cloud, point_count);
    last_pointcloud_points_ = point_count;
    update_validity_ratio_from_cloud(point_count);
    pointcloud_publisher_->publish(std::move(cloud));
  }

  // The exact ratio comes from counting the downloaded validity mask, but that
  // download only happens when a debug consumer asked for it. Otherwise derive
  // the ratio from the sampled cloud so the diagnostic never reports a stale 0.
  void update_validity_ratio_from_cloud(std::size_t point_count)
  {
    if (validity_ratio_from_mask_this_frame_) {
      return;
    }
    const int capacity = pointcloud_capacity();
    last_validity_ratio_ = capacity <= 0 ? 0.0 :
      static_cast<double>(point_count) / static_cast<double>(capacity);
  }

  void maybe_log_diagnostics(
    const std::chrono::steady_clock::time_point & now)
  {
    const double elapsed_sec =
      std::chrono::duration<double>(now - last_diagnostics_time_).count();
    if (elapsed_sec < diagnostics_period_sec_) {
      return;
    }
    if (diagnostic_frame_count_ == 0) {
      if (idle_frame_count_ > 0) {
        RCLCPP_INFO(
          get_logger(),
          "idle: no subscriber on the panorama outputs, %zu synchronized "
          "frames skipped in the last %.1f s",
          idle_frame_count_, elapsed_sec);
        idle_frame_count_ = 0;
        last_diagnostics_time_ = now;
      }
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
      "panorama_points=%zu validity=%.1f%% seam=%s:%d..%d(mean=%.1f) "
      "gain(BGR)=%.2f/%.2f/%.2f total=%zu",
      panorama_width_, panorama_height_, count / elapsed_sec,
      processing_time_sum_ms_ / count,
      used_cuda_last_frame_ ? "CUDA" : "CPU", last_gpu_time_ms_,
      left_color_input_hz, left_depth_input_hz,
      right_color_input_hz, right_depth_input_hz,
      last_left_depth_age_ms_, last_right_depth_age_ms_,
      sync_span_sum_ms_ / count, sync_span_max_ms_,
      last_left_depth_points_, last_right_depth_points_,
      last_pointcloud_points_,
      100.0 * last_validity_ratio_,
      last_content_aware_seam_used_ ? "content" : "fixed",
      last_seam_min_x_, last_seam_max_x_, last_seam_mean_x_,
      smoothed_gain_[0], smoothed_gain_[1], smoothed_gain_[2],
      frame_count_);

    last_diagnostics_time_ = now;
    diagnostic_frame_count_ = 0;
    idle_frame_count_ = 0;
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
  std::string pointcloud_topic_;
  std::string output_frame_id_;
  bool publish_auxiliary_outputs_{false};
  bool publish_validity_output_{false};
  bool publish_range_output_{false};
  bool publish_pointcloud_{false};
  int pointcloud_stride_{4};
  bool publisher_best_effort_{false};
  bool auxiliary_publisher_best_effort_{true};
  bool pointcloud_publisher_best_effort_{true};
  double max_output_rate_hz_{0.0};
  double auxiliary_output_rate_hz_{0.0};

  int sync_queue_size_{50};
  double sync_slop_ms_{45.0};
  bool input_images_rotated_180_{true};
  bool left_input_image_rotated_180_{true};
  bool right_input_image_rotated_180_{true};
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
  bool cuda_depth_spatial_filter_{false};
  double cuda_depth_spatial_delta_m_{0.03};
  double cuda_depth_spatial_delta_relative_{0.01};
  bool cuda_depth_temporal_filter_{false};
  double cuda_depth_temporal_alpha_{0.65};
  double cuda_depth_temporal_reset_m_{0.08};
  int depth_splat_radius_px_{1};
  int depth_edge_splat_radius_px_{0};
  int projected_hole_radius_px_{0};
  bool allow_color_fallback_{true};
  double seam_angle_deg_{0.0};
  bool auto_seam_center_{false};
  int seam_feather_px_{2};
  bool depth_aware_color_{true};
  bool render_depth_reprojected_color_{true};
  bool depth_color_overlap_only_{false};
  double depth_color_band_margin_deg_{0.0};
  bool use_rgbd_synchronization_{true};
  bool depth_temporal_stabilization_{true};
  double depth_temporal_alpha_{0.35};
  double depth_temporal_reset_m_{0.05};
  int depth_median_kernel_{3};
  double occlusion_switch_margin_m_{0.05};
  bool prefer_seam_camera_when_both_depth_valid_{false};
  bool content_aware_seam_{false};
  double seam_color_weight_{1.0};
  double seam_depth_weight_{2.0};
  double seam_foreground_weight_{0.35};
  double seam_center_weight_{0.03};
  double seam_temporal_weight_{0.08};
  int seam_max_step_px_{3};
  bool exposure_compensation_{true};
  double exposure_smoothing_{0.15};
  double min_exposure_gain_{0.75};
  double max_exposure_gain_{1.33};
  double diagnostics_period_sec_{2.0};
  bool use_cuda_{true};
  int cuda_timeout_ms_{500};
  double cuda_slow_frame_ms_{150.0};
  int cuda_slow_frame_limit_{3};
  bool publish_only_when_subscribed_{true};
  int exposure_sample_stride_{4};
  bool input_best_effort_{true};
  bool projection_is_rectilinear_{false};
  bool source_is_rgb_{false};

  CameraModel left_model_;
  CameraModel right_model_;
  CameraModel pending_left_model_;
  CameraModel pending_right_model_;
  std::mutex camera_info_mutex_;
  bool camera_info_staged_{false};
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
  int depth_color_min_x_{0};
  int depth_color_max_x_{-1};
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
  std::array<int, 2> left_depth_columns_{0, -1};
  std::array<int, 2> right_depth_columns_{0, -1};
  bool depth_columns_valid_{false};
  std::vector<double> column_sin_;
  std::vector<double> column_cos_;
  bool gpu_cloud_filled_{false};
  std::size_t gpu_cloud_points_{0};
  bool validity_ratio_from_mask_this_frame_{false};
#ifdef PANORAMA_WITH_CUDA
  std::unique_ptr<CudaPanoramaBackend> cuda_backend_;
  bool cuda_backend_configured_{false};
  bool cuda_backend_failed_{false};
  bool configured_source_is_rgb_{false};
  int cuda_slow_frame_count_{0};
#endif
  bool used_cuda_last_frame_{false};
  double last_gpu_time_ms_{0.0};
  bool last_content_aware_seam_used_{false};
  int last_seam_min_x_{0};
  int last_seam_max_x_{0};
  double last_seam_mean_x_{0.0};

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
  rclcpp::Publisher<PointCloud2>::SharedPtr pointcloud_publisher_;
  rclcpp::CallbackGroup::SharedPtr image_callback_group_;

  std::mutex projection_mutex_;
  std::mutex processing_mutex_;
  std::condition_variable processing_condition_;
  std::thread processing_thread_;
  bool stop_processing_{false};
  std::chrono::steady_clock::time_point last_processing_start_{};
  std::chrono::steady_clock::time_point last_auxiliary_publish_time_{};
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
  size_t idle_frame_count_{0};
  size_t last_left_depth_points_{0};
  size_t last_right_depth_points_{0};
  size_t last_pointcloud_points_{0};
  double sync_span_sum_ms_{0.0};
  double sync_span_max_ms_{0.0};
  double processing_time_sum_ms_{0.0};
  double last_validity_ratio_{0.0};
};

}  // namespace panorama_stitcher

int main(int argc, char ** argv)
{
#ifdef __GLIBC__
  // Every frame allocates a multi-megabyte output message. With the default
  // 128 kB threshold glibc serves those from fresh mmap regions and returns
  // them immediately, so the process re-faults thousands of pages per second.
  // Keeping the arena warm removes that per-frame page-fault storm.
  mallopt(M_MMAP_THRESHOLD, 256 * 1024 * 1024);
  mallopt(M_TRIM_THRESHOLD, 256 * 1024 * 1024);
#endif
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
