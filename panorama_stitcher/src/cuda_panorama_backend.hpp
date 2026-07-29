#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace panorama_stitcher
{

struct CudaCameraModel
{
  float fx{0.0F};
  float fy{0.0F};
  float cx{0.0F};
  float cy{0.0F};
  float rotation_camera_to_rig[9]{};
  float translation_camera_in_rig[3]{};
  int minimum_depth_column{0};
  int maximum_depth_column{-1};
};

struct CudaPanoramaConfig
{
  int source_width{0};
  int source_height{0};
  int panorama_width{0};
  int panorama_height{0};
  int projection_model{0};  // 0: cylindrical, 1: rectilinear
  float panorama_focal_px{0.0F};
  float panorama_min_angle{0.0F};
  float panorama_min_vertical{0.0F};
  float virtual_fx_px{0.0F};
  float virtual_fy_px{0.0F};
  float virtual_cx_px{0.0F};
  float virtual_cy_px{0.0F};
  float minimum_depth_m{0.0F};
  float maximum_depth_m{0.0F};
  float occlusion_switch_margin_m{0.0F};
  bool depth_aware_color{true};
  bool prefer_seam_camera_when_both_depth_valid{false};
  int seam_x{0};
  int seam_feather_px{0};
  int projected_hole_radius{2};
};

struct CudaPanoramaStats
{
  std::size_t left_depth_points{0};
  std::size_t right_depth_points{0};
  float gpu_time_ms{0.0F};
};

class CudaPanoramaBackend
{
public:
  CudaPanoramaBackend();
  ~CudaPanoramaBackend();

  CudaPanoramaBackend(const CudaPanoramaBackend &) = delete;
  CudaPanoramaBackend & operator=(const CudaPanoramaBackend &) = delete;

  static bool runtime_available(std::string & description);

  bool configure(
    const CudaPanoramaConfig & config,
    const CudaCameraModel & left_camera,
    const CudaCameraModel & right_camera,
    const cv::Mat & left_base_mask,
    const cv::Mat & right_base_mask,
    const cv::Mat & left_map_x,
    const cv::Mat & left_map_y,
    const cv::Mat & right_map_x,
    const cv::Mat & right_map_y,
    std::string & error);

  bool process(
    const cv::Mat & left_source_color,
    const cv::Mat & left_depth_m,
    const cv::Mat & right_source_color,
    const cv::Mat & right_depth_m,
    const cv::Vec3d & right_gain_bgr,
    cv::Mat & panorama,
    CudaPanoramaStats & stats,
    std::string & error);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace panorama_stitcher
