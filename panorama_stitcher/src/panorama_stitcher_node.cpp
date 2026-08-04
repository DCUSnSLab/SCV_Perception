#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmw/qos_profiles.h>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace panorama_stitcher
{

using Image = sensor_msgs::msg::Image;
using ApproximatePolicy =
  message_filters::sync_policies::ApproximateTime<Image, Image>;

class PanoramaStitcherNode : public rclcpp::Node
{
public:
  PanoramaStitcherNode()
  : Node("panorama_stitcher"),
    last_diagnostics_time_(std::chrono::steady_clock::now())
  {
    left_topic_ = declare_parameter<std::string>(
      "left_topic", "/front_left/front_left/color/image_raw");
    right_topic_ = declare_parameter<std::string>(
      "right_topic", "/front_right/front_right/color/image_raw");
    output_topic_ = declare_parameter<std::string>(
      "output_topic", "/panorama/image_raw");
    output_frame_id_ = declare_parameter<std::string>(
      "output_frame_id", "panorama_optical_frame");

    sync_queue_size_ = declare_parameter<int>("sync_queue_size", 30);
    sync_slop_ms_ = declare_parameter<double>("sync_slop_ms", 25.0);
    right_x_offset_px_ = declare_parameter<int>("right_x_offset_px", 1834);
    right_y_offset_px_ = declare_parameter<int>("right_y_offset_px", 9);
    blend_width_px_ = declare_parameter<int>("blend_width_px", 86);
    crop_to_common_height_ = declare_parameter<bool>(
      "crop_to_common_height", true);
    rotate_left_180_ = declare_parameter<bool>("rotate_left_180", false);
    rotate_right_180_ = declare_parameter<bool>("rotate_right_180", false);

    exposure_compensation_ = declare_parameter<bool>(
      "enable_exposure_compensation", true);
    exposure_smoothing_ = declare_parameter<double>(
      "exposure_smoothing", 0.15);
    min_exposure_gain_ = declare_parameter<double>(
      "min_exposure_gain", 0.75);
    max_exposure_gain_ = declare_parameter<double>(
      "max_exposure_gain", 1.33);
    output_scale_ = declare_parameter<double>("output_scale", 1.0);
    diagnostics_period_sec_ = declare_parameter<double>(
      "diagnostics_period_sec", 2.0);

    validate_parameters();

    output_publisher_ = create_publisher<Image>(
      output_topic_,
      rclcpp::QoS(rclcpp::KeepLast(2)).reliable().durability_volatile());

    left_subscriber_.subscribe(this, left_topic_, rmw_qos_profile_sensor_data);
    right_subscriber_.subscribe(this, right_topic_, rmw_qos_profile_sensor_data);

    synchronizer_ = std::make_shared<
      message_filters::Synchronizer<ApproximatePolicy>>(
      ApproximatePolicy(sync_queue_size_),
      left_subscriber_,
      right_subscriber_);
    synchronizer_->setMaxIntervalDuration(
      rclcpp::Duration::from_seconds(sync_slop_ms_ / 1000.0));
    synchronizer_->registerCallback(
      std::bind(
        &PanoramaStitcherNode::image_callback,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    RCLCPP_INFO(
      get_logger(),
      "Panorama ready: left=%s right=%s output=%s, offset=(%d,%d), "
      "blend=%d px, sync<=%.1f ms, scale=%.2f",
      left_topic_.c_str(),
      right_topic_.c_str(),
      output_topic_.c_str(),
      right_x_offset_px_,
      right_y_offset_px_,
      blend_width_px_,
      sync_slop_ms_,
      output_scale_);
  }

private:
  void validate_parameters()
  {
    sync_queue_size_ = std::max(sync_queue_size_, 2);
    sync_slop_ms_ = std::max(sync_slop_ms_, 0.1);
    right_x_offset_px_ = std::max(right_x_offset_px_, 0);
    blend_width_px_ = std::max(blend_width_px_, 0);
    exposure_smoothing_ = std::clamp(exposure_smoothing_, 0.0, 1.0);
    min_exposure_gain_ = std::max(min_exposure_gain_, 0.01);
    max_exposure_gain_ = std::max(max_exposure_gain_, min_exposure_gain_);
    output_scale_ = std::clamp(output_scale_, 0.05, 1.0);
    diagnostics_period_sec_ = std::max(diagnostics_period_sec_, 0.2);
  }

  static int64_t stamp_nanoseconds(const builtin_interfaces::msg::Time & stamp)
  {
    return static_cast<int64_t>(stamp.sec) * 1000000000LL +
           static_cast<int64_t>(stamp.nanosec);
  }

  static cv::Mat to_bgr(
    const Image::ConstSharedPtr & message,
    bool rotate_180)
  {
    const auto shared = cv_bridge::toCvShare(
      message, sensor_msgs::image_encodings::BGR8);
    if (!rotate_180) {
      return shared->image;
    }

    cv::Mat rotated;
    cv::rotate(shared->image, rotated, cv::ROTATE_180);
    return rotated;
  }

  cv::Vec3d estimate_right_gain(
    const cv::Mat & left_overlap,
    const cv::Mat & right_overlap)
  {
    if (!exposure_compensation_ ||
      left_overlap.empty() ||
      right_overlap.empty())
    {
      return cv::Vec3d(1.0, 1.0, 1.0);
    }

    cv::Mat left_gray;
    cv::Mat right_gray;
    cv::cvtColor(left_overlap, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_overlap, right_gray, cv::COLOR_BGR2GRAY);

    cv::Mat valid_mask =
      (left_gray > 25) & (left_gray < 235) &
      (right_gray > 25) & (right_gray < 235);

    if (cv::countNonZero(valid_mask) < 500) {
      return smoothed_gain_;
    }

    const cv::Scalar left_mean = cv::mean(left_overlap, valid_mask);
    const cv::Scalar right_mean = cv::mean(right_overlap, valid_mask);

    cv::Vec3d measured_gain;
    for (int channel = 0; channel < 3; ++channel) {
      const double denominator = std::max(right_mean[channel], 1.0);
      measured_gain[channel] = std::clamp(
        left_mean[channel] / denominator,
        min_exposure_gain_,
        max_exposure_gain_);
      smoothed_gain_[channel] =
        (1.0 - exposure_smoothing_) * smoothed_gain_[channel] +
        exposure_smoothing_ * measured_gain[channel];
    }

    return smoothed_gain_;
  }

  static cv::Mat apply_gain(const cv::Mat & image, const cv::Vec3d & gain)
  {
    cv::Mat transform = (
      cv::Mat_<double>(3, 4) <<
      gain[0], 0.0, 0.0, 0.0,
      0.0, gain[1], 0.0, 0.0,
      0.0, 0.0, gain[2], 0.0);

    cv::Mat adjusted;
    cv::transform(image, adjusted, transform);
    return adjusted;
  }

  void prepare_blend_weights(int width, int height)
  {
    if (width <= 0 || height <= 0) {
      left_blend_weight_.release();
      right_blend_weight_.release();
      cached_blend_width_ = 0;
      cached_blend_height_ = 0;
      return;
    }

    if (cached_blend_width_ == width &&
      cached_blend_height_ == height)
    {
      return;
    }

    cv::Mat right_row(1, width, CV_32FC1);
    if (width == 1) {
      right_row.at<float>(0, 0) = 0.5F;
    } else {
      for (int x = 0; x < width; ++x) {
        right_row.at<float>(0, x) =
          static_cast<float>(x) / static_cast<float>(width - 1);
      }
    }

    cv::repeat(right_row, height, 1, right_blend_weight_);
    left_blend_weight_ = cv::Scalar::all(1.0) - right_blend_weight_;
    cached_blend_width_ = width;
    cached_blend_height_ = height;
  }

  cv::Mat stitch(const cv::Mat & left, const cv::Mat & right)
  {
    if (left.empty() || right.empty()) {
      throw std::runtime_error("received an empty image");
    }

    if (left.type() != CV_8UC3 || right.type() != CV_8UC3) {
      throw std::runtime_error("input conversion did not produce BGR8");
    }

    const int left_y = std::max(0, -right_y_offset_px_);
    const int right_y = std::max(0, right_y_offset_px_);

    const cv::Rect left_rect(0, left_y, left.cols, left.rows);
    const cv::Rect right_rect(
      right_x_offset_px_, right_y, right.cols, right.rows);

    const int output_width = std::max(
      left_rect.x + left_rect.width,
      right_rect.x + right_rect.width);
    const int output_height = std::max(
      left_rect.y + left_rect.height,
      right_rect.y + right_rect.height);

    cv::Rect overlap = left_rect & right_rect;
    cv::Mat adjusted_right = right;

    if (overlap.area() > 0) {
      const cv::Rect left_local(
        overlap.x - left_rect.x,
        overlap.y - left_rect.y,
        overlap.width,
        overlap.height);
      const cv::Rect right_local(
        overlap.x - right_rect.x,
        overlap.y - right_rect.y,
        overlap.width,
        overlap.height);

      const cv::Vec3d gain = estimate_right_gain(
        left(left_local), right(right_local));
      adjusted_right = apply_gain(right, gain);
    }

    cv::Mat panorama = cv::Mat::zeros(
      output_height, output_width, CV_8UC3);
    left.copyTo(panorama(left_rect));
    adjusted_right.copyTo(panorama(right_rect));

    if (overlap.area() > 0) {
      const cv::Rect left_local(
        overlap.x - left_rect.x,
        overlap.y - left_rect.y,
        overlap.width,
        overlap.height);
      const cv::Rect right_local(
        overlap.x - right_rect.x,
        overlap.y - right_rect.y,
        overlap.width,
        overlap.height);

      const int blend_width =
        blend_width_px_ == 0 ?
        overlap.width :
        std::min(blend_width_px_, overlap.width);
      const int blend_start = overlap.width - blend_width;

      if (blend_start > 0) {
      left(left_local).colRange(0, blend_start).copyTo(
          panorama(overlap).colRange(0, blend_start));
      }

      const cv::Rect left_blend_local(
        left_local.x + blend_start,
        left_local.y,
        blend_width,
        overlap.height);
      const cv::Rect right_blend_local(
        right_local.x + blend_start,
        right_local.y,
        blend_width,
        overlap.height);
      const cv::Rect output_blend(
        overlap.x + blend_start,
        overlap.y,
        blend_width,
        overlap.height);

      prepare_blend_weights(blend_width, overlap.height);
      cv::blendLinear(
        left(left_blend_local),
        adjusted_right(right_blend_local),
        left_blend_weight_,
        right_blend_weight_,
        panorama(output_blend));
    }

    if (crop_to_common_height_) {
      const int common_top = std::max(left_rect.y, right_rect.y);
      const int common_bottom = std::min(
        left_rect.y + left_rect.height,
        right_rect.y + right_rect.height);
      if (common_bottom > common_top) {
        panorama = panorama.rowRange(common_top, common_bottom).clone();
      }
    }

    if (std::abs(output_scale_ - 1.0) < 1e-6) {
      return panorama;
    }

    cv::Mat resized;
    cv::resize(
      panorama,
      resized,
      cv::Size(),
      output_scale_,
      output_scale_,
      output_scale_ < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
    return resized;
  }

  void image_callback(
    const Image::ConstSharedPtr & left_message,
    const Image::ConstSharedPtr & right_message)
  {
    const auto callback_start = std::chrono::steady_clock::now();

    try {
      const cv::Mat left = to_bgr(left_message, rotate_left_180_);
      const cv::Mat right = to_bgr(right_message, rotate_right_180_);
      const cv::Mat panorama = stitch(left, right);

      std_msgs::msg::Header output_header = left_message->header;
      if (stamp_nanoseconds(right_message->header.stamp) >
        stamp_nanoseconds(left_message->header.stamp))
      {
        output_header.stamp = right_message->header.stamp;
      }
      output_header.frame_id = output_frame_id_;

      output_publisher_->publish(
        *cv_bridge::CvImage(
          output_header,
          sensor_msgs::image_encodings::BGR8,
          panorama).toImageMsg());

      const double sync_delta_ms = std::abs(
        static_cast<double>(
          stamp_nanoseconds(left_message->header.stamp) -
          stamp_nanoseconds(right_message->header.stamp))) / 1e6;
      sync_delta_sum_ms_ += sync_delta_ms;
      sync_delta_max_ms_ = std::max(sync_delta_max_ms_, sync_delta_ms);
      ++frame_count_;
      ++diagnostic_frame_count_;

      const auto callback_end = std::chrono::steady_clock::now();
      processing_time_sum_ms_ +=
        std::chrono::duration<double, std::milli>(
        callback_end - callback_start).count();

      maybe_log_diagnostics(panorama.cols, panorama.rows, callback_end);
    } catch (const cv_bridge::Exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "cv_bridge conversion failed: %s", error.what());
    } catch (const cv::Exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "OpenCV stitching failed: %s", error.what());
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Panorama stitching failed: %s", error.what());
    }
  }

  void maybe_log_diagnostics(
    int output_width,
    int output_height,
    const std::chrono::steady_clock::time_point & now)
  {
    const double elapsed_sec =
      std::chrono::duration<double>(now - last_diagnostics_time_).count();
    if (elapsed_sec < diagnostics_period_sec_ || diagnostic_frame_count_ == 0) {
      return;
    }

    const double fps =
      static_cast<double>(diagnostic_frame_count_) / elapsed_sec;
    const double average_sync_ms =
      sync_delta_sum_ms_ / static_cast<double>(diagnostic_frame_count_);
    const double average_processing_ms =
      processing_time_sum_ms_ / static_cast<double>(diagnostic_frame_count_);

    RCLCPP_INFO(
      get_logger(),
      "output=%dx%d fps=%.1f processing=%.1f ms sync(avg/max)=%.1f/%.1f ms "
      "right_gain(BGR)=%.2f/%.2f/%.2f total=%zu",
      output_width,
      output_height,
      fps,
      average_processing_ms,
      average_sync_ms,
      sync_delta_max_ms_,
      smoothed_gain_[0],
      smoothed_gain_[1],
      smoothed_gain_[2],
      frame_count_);

    last_diagnostics_time_ = now;
    diagnostic_frame_count_ = 0;
    sync_delta_sum_ms_ = 0.0;
    sync_delta_max_ms_ = 0.0;
    processing_time_sum_ms_ = 0.0;
  }

  std::string left_topic_;
  std::string right_topic_;
  std::string output_topic_;
  std::string output_frame_id_;

  int sync_queue_size_{30};
  double sync_slop_ms_{25.0};
  int right_x_offset_px_{1834};
  int right_y_offset_px_{9};
  int blend_width_px_{86};
  bool crop_to_common_height_{true};
  bool rotate_left_180_{false};
  bool rotate_right_180_{false};
  bool exposure_compensation_{true};
  double exposure_smoothing_{0.15};
  double min_exposure_gain_{0.75};
  double max_exposure_gain_{1.33};
  double output_scale_{1.0};
  double diagnostics_period_sec_{2.0};

  cv::Vec3d smoothed_gain_{1.0, 1.0, 1.0};
  cv::Mat left_blend_weight_;
  cv::Mat right_blend_weight_;
  int cached_blend_width_{0};
  int cached_blend_height_{0};

  message_filters::Subscriber<Image> left_subscriber_;
  message_filters::Subscriber<Image> right_subscriber_;
  std::shared_ptr<message_filters::Synchronizer<ApproximatePolicy>>
    synchronizer_;
  rclcpp::Publisher<Image>::SharedPtr output_publisher_;

  std::chrono::steady_clock::time_point last_diagnostics_time_;
  size_t frame_count_{0};
  size_t diagnostic_frame_count_{0};
  double sync_delta_sum_ms_{0.0};
  double sync_delta_max_ms_{0.0};
  double processing_time_sum_ms_{0.0};
};

}  // namespace panorama_stitcher

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<panorama_stitcher::PanoramaStitcherNode>());
  rclcpp::shutdown();
  return 0;
}
