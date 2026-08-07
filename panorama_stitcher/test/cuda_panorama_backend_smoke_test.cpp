#include "cuda_panorama_backend.hpp"

#include <cmath>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>

namespace
{

using panorama_stitcher::CudaCameraModel;
using panorama_stitcher::CudaPanoramaBackend;
using panorama_stitcher::CudaPanoramaConfig;
using panorama_stitcher::CudaPanoramaStats;

bool expect(bool condition, const std::string & message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

}  // namespace

int main()
{
  std::string description;
  if (!CudaPanoramaBackend::runtime_available(description)) {
    std::cout << "SKIP: " << description << '\n';
    return 77;
  }

  constexpr int source_width = 32;
  constexpr int source_height = 24;
  constexpr int panorama_width = source_width * 2;
  constexpr int panorama_height = source_height;

  CudaPanoramaConfig config;
  config.source_width = source_width;
  config.source_height = source_height;
  config.panorama_width = panorama_width;
  config.panorama_height = panorama_height;
  config.minimum_depth_m = 0.1F;
  config.maximum_depth_m = 20.0F;
  config.depth_aware_color = false;
  config.allow_color_fallback = true;
  config.depth_color_min_x = 0;
  config.depth_color_max_x = panorama_width - 1;
  config.seam_x = source_width;
  config.operation_timeout_ms = 1000;

  CudaCameraModel camera;
  camera.fx = 30.0F;
  camera.fy = 30.0F;
  camera.cx = (source_width - 1) * 0.5F;
  camera.cy = (source_height - 1) * 0.5F;
  camera.minimum_depth_column = 0;
  camera.maximum_depth_column = source_width - 1;
  camera.rotation_camera_to_rig[0] = 1.0F;
  camera.rotation_camera_to_rig[4] = 1.0F;
  camera.rotation_camera_to_rig[8] = 1.0F;

  cv::Mat left_mask(
    panorama_height, panorama_width, CV_8UC1, cv::Scalar(0));
  cv::Mat right_mask = left_mask.clone();
  left_mask.colRange(0, source_width).setTo(255);
  right_mask.colRange(source_width, panorama_width).setTo(255);

  cv::Mat left_map_x(
    panorama_height, panorama_width, CV_32FC1, cv::Scalar(-1.0F));
  cv::Mat left_map_y = left_map_x.clone();
  cv::Mat right_map_x = left_map_x.clone();
  cv::Mat right_map_y = left_map_x.clone();
  for (int y = 0; y < panorama_height; ++y) {
    for (int x = 0; x < source_width; ++x) {
      left_map_x.at<float>(y, x) = static_cast<float>(x);
      left_map_y.at<float>(y, x) = static_cast<float>(y);
      right_map_x.at<float>(y, x + source_width) = static_cast<float>(x);
      right_map_y.at<float>(y, x + source_width) = static_cast<float>(y);
    }
  }

  cv::Mat left_color(
    source_height, source_width, CV_8UC3, cv::Scalar(10, 20, 30));
  cv::Mat right_color(
    source_height, source_width, CV_8UC3, cv::Scalar(40, 50, 60));
  const cv::Mat no_depth;
  cv::Mat panorama;
  cv::Mat validity;
  cv::Mat range;
  CudaPanoramaStats stats;
  std::string error;

  CudaPanoramaBackend backend;
  if (!backend.configure(
      config, camera, camera, left_mask, right_mask,
      left_map_x, left_map_y, right_map_x, right_map_y, error))
  {
    std::cerr << "FAIL: configure: " << error << '\n';
    return 1;
  }

  for (int iteration = 0; iteration < 25; ++iteration) {
    if (!backend.process(
        left_color, no_depth, right_color, no_depth,
        cv::Vec3d(1.0, 1.0, 1.0), panorama, validity, range,
        stats, error))
    {
      std::cerr << "FAIL: process iteration " << iteration << ": "
                << error << '\n';
      return 1;
    }
  }

  bool ok = true;
  ok &= expect(!backend.is_quarantined(), "backend unexpectedly quarantined");
  ok &= expect(
    panorama.rows == panorama_height && panorama.cols == panorama_width,
    "unexpected panorama dimensions");
  ok &= expect(
    panorama.at<cv::Vec3b>(panorama_height / 2, source_width / 2) ==
    cv::Vec3b(10, 20, 30),
    "left source color did not survive the compositor");
  ok &= expect(
    panorama.at<cv::Vec3b>(panorama_height / 2, source_width + source_width / 2) ==
    cv::Vec3b(40, 50, 60),
    "right source color did not survive the compositor");
  ok &= expect(std::isfinite(stats.gpu_time_ms), "invalid GPU timing");

  std::cout << "CUDA smoke test on " << description
            << ": gpu=" << stats.gpu_time_ms << " ms\n";
  return ok ? 0 : 1;
}
