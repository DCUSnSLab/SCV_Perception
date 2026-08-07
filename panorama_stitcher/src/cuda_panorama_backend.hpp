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
  float depth_discontinuity_abs_m{0.08F};
  float depth_discontinuity_relative{0.04F};
  bool depth_spatial_filter{false};
  float depth_spatial_delta_m{0.03F};
  float depth_spatial_delta_relative{0.01F};
  bool depth_temporal_filter{false};
  float depth_temporal_alpha{0.65F};
  float depth_temporal_reset_m{0.08F};
  float occlusion_switch_margin_m{0.0F};
  bool depth_aware_color{true};
  bool render_depth_reprojected_color{true};
  bool depth_color_overlap_only{false};
  bool allow_color_fallback{true};
  bool prefer_seam_camera_when_both_depth_valid{false};
  bool content_aware_seam{false};
  float seam_color_weight{1.0F};
  float seam_depth_weight{2.0F};
  float seam_foreground_weight{0.35F};
  float seam_center_weight{0.03F};
  float seam_temporal_weight{0.08F};
  int seam_max_step_px{3};
  int depth_color_min_x{0};
  int depth_color_max_x{-1};
  int seam_x{0};
  int seam_feather_px{0};
  int depth_splat_radius_px{1};
  int depth_edge_splat_radius_px{0};
  int projected_hole_radius{0};
  // Scale applied when depth arrives as raw 16UC1 device millimetres. Keeping
  // the conversion on the GPU halves the PCIe upload and removes a full-frame
  // CPU convertTo from every callback.
  float depth_scale_m{0.001F};
  // Source rows are RGB (the RealSense wrapper publishes rgb8). Swapping in
  // the kernels avoids a host-side colour conversion of every input frame
  // while the panorama output stays BGR8.
  bool source_channel_swap{false};
  // Sampling stride of the GPU point-cloud builder. 0 disables it.
  int pointcloud_stride{0};
  // Maximum time a submitted frame may remain incomplete. A timeout opens
  // the circuit breaker: the backend is quarantined and must not issue any
  // more CUDA runtime calls in this process.
  int operation_timeout_ms{500};
};

// Per-frame selection of the outputs that are actually needed. Downloading the
// large range/validity images costs about 15 MB per frame, so a frame that only
// feeds the point cloud must not pay for them.
struct CudaProcessOptions
{
  bool download_panorama{true};
  bool download_validity{true};
  bool download_range{true};
  bool build_pointcloud{false};
};

// Destination for the compacted GPU point cloud. The caller passes the exact
// PointCloud2 layout because it is not simply four packed floats:
// setPointCloud2FieldsByString(2, "xyz", "rgb") yields point_step 32 with rgb
// at offset 16, and writing a 16-byte stride there silently produces half the
// points with colours read out of the neighbouring coordinates.
struct CudaPointCloudRequest
{
  unsigned char * destination{nullptr};
  int capacity_points{0};
  int point_step_bytes{0};
  int x_offset_bytes{0};
  int y_offset_bytes{4};
  int z_offset_bytes{8};
  int rgb_offset_bytes{12};
  std::size_t point_count{0};
};

struct CudaPanoramaStats
{
  std::size_t left_depth_points{0};
  std::size_t right_depth_points{0};
  float gpu_time_ms{0.0F};
  bool content_aware_seam_used{false};
  int seam_min_x{0};
  int seam_max_x{0};
  float seam_mean_x{0.0F};
};

class CudaPanoramaBackend
{
public:
  CudaPanoramaBackend();
  ~CudaPanoramaBackend();

  CudaPanoramaBackend(const CudaPanoramaBackend &) = delete;
  CudaPanoramaBackend & operator=(const CudaPanoramaBackend &) = delete;

  static bool runtime_available(std::string & description);

  // Permanently disables this instance without touching CUDA resources. This
  // is deliberate: after a timeout or fatal runtime error, cleanup calls such
  // as cudaFree may themselves synchronize with a wedged context. The node
  // falls back to CPU and process teardown lets the OS/driver reclaim the
  // quarantined context.
  void quarantine(const std::string & reason) noexcept;
  bool is_quarantined() const noexcept;
  std::string quarantine_reason() const;

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

  // left_depth / right_depth accept CV_16UC1 raw depth (scaled on the GPU by
  // config.depth_scale_m) or CV_32FC1 metres.
  bool process(
    const cv::Mat & left_source_color,
    const cv::Mat & left_depth,
    const cv::Mat & right_source_color,
    const cv::Mat & right_depth,
    const cv::Vec3d & right_gain_bgr,
    cv::Mat & panorama,
    cv::Mat & validity,
    cv::Mat & range_m,
    CudaPanoramaStats & stats,
    std::string & error,
    const CudaProcessOptions & options = CudaProcessOptions(),
    CudaPointCloudRequest * pointcloud = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace panorama_stitcher
