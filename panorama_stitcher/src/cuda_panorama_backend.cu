#include "cuda_panorama_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace panorama_stitcher
{
namespace
{

constexpr unsigned long long kInvalidProjectionKey =
  0xffffffffffffffffULL;
constexpr unsigned int kInvalidProjectionRange = 0xffffffffU;
// PointCloud2 stride the device cloud buffer is sized for. An xyz+rgb cloud
// built by PointCloud2Modifier uses 32 bytes per point.
constexpr int kMaxPointCloudStride = 32;
constexpr int kDefaultGpuTimeoutMs = 500;

std::string cuda_error_message(
  const char * operation, cudaError_t result)
{
  std::ostringstream stream;
  stream << operation << ": " << cudaGetErrorString(result);
  return stream.str();
}

bool check_cuda(
  cudaError_t result, const char * operation, std::string & error)
{
  if (result == cudaSuccess) {
    return true;
  }
  error = cuda_error_message(operation, result);
  return false;
}

bool wait_for_cuda_event(
  cudaEvent_t event, int timeout_ms,
  const char * operation, std::string & error)
{
  const auto timeout = std::chrono::milliseconds(
    std::max(timeout_ms, 1));
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    const cudaError_t result = cudaEventQuery(event);
    if (result == cudaSuccess) {
      return true;
    }
    if (result != cudaErrorNotReady) {
      error = cuda_error_message(operation, result);
      return false;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      std::ostringstream stream;
      stream << operation << " timed out after " << timeout.count() << " ms";
      error = stream.str();
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void copy_rows_to_contiguous(
  unsigned char * destination,
  const cv::Mat & source,
  std::size_t row_bytes,
  int rows)
{
  for (int row = 0; row < rows; ++row) {
    std::memcpy(
      destination + static_cast<std::size_t>(row) * row_bytes,
      source.ptr(row), row_bytes);
  }
}

void copy_contiguous_to_rows(
  cv::Mat & destination,
  const unsigned char * source,
  std::size_t row_bytes,
  int rows)
{
  for (int row = 0; row < rows; ++row) {
    std::memcpy(
      destination.ptr(row),
      source + static_cast<std::size_t>(row) * row_bytes,
      row_bytes);
  }
}

__device__ __forceinline__ float clamp_channel(float value)
{
  return fminf(fmaxf(value, 0.0F), 255.0F);
}

__device__ __forceinline__ unsigned char gained_channel(
  unsigned char value, float gain)
{
  return static_cast<unsigned char>(
    __float2int_rn(clamp_channel(static_cast<float>(value) * gain)));
}

__device__ __forceinline__ unsigned char float_channel(float value)
{
  return static_cast<unsigned char>(
    __float2int_rn(clamp_channel(value)));
}

__device__ __forceinline__ bool valid_depth(
  float depth, const CudaPanoramaConfig & config)
{
  return isfinite(depth) &&
         depth >= config.minimum_depth_m &&
         depth <= config.maximum_depth_m;
}

__device__ __forceinline__ bool depth_discontinuity(
  const float * depth_m,
  int u,
  int v,
  float center_depth,
  const CudaPanoramaConfig & config)
{
  const float threshold = fmaxf(
    config.depth_discontinuity_abs_m,
    config.depth_discontinuity_relative * center_depth);
  constexpr int offsets[4][2] = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1}
  };
  for (int index = 0; index < 4; ++index) {
    const int neighbor_u = u + offsets[index][0];
    const int neighbor_v = v + offsets[index][1];
    if (
      neighbor_u < 0 || neighbor_u >= config.source_width ||
      neighbor_v < 0 || neighbor_v >= config.source_height)
    {
      continue;
    }
    const float neighbor_depth =
      depth_m[neighbor_v * config.source_width + neighbor_u];
    if (!valid_depth(neighbor_depth, config)) {
      return true;
    }
    if (fabsf(neighbor_depth - center_depth) > threshold) {
      return true;
    }
  }
  return false;
}

__global__ void edge_aware_depth_filter_kernel(
  const float * source_depth,
  float * filtered_depth,
  CudaPanoramaConfig config)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixel_count = config.source_width * config.source_height;
  if (index >= pixel_count) {
    return;
  }

  const float center = source_depth[index];
  if (!valid_depth(center, config)) {
    filtered_depth[index] = 0.0F;
    return;
  }

  const int center_x = index % config.source_width;
  const int center_y = index / config.source_width;
  const float threshold = fmaxf(
    config.depth_spatial_delta_m,
    config.depth_spatial_delta_relative * center);
  float sum = 0.0F;
  int count = 0;
  for (int offset_y = -1; offset_y <= 1; ++offset_y) {
    const int y = center_y + offset_y;
    if (y < 0 || y >= config.source_height) {
      continue;
    }
    for (int offset_x = -1; offset_x <= 1; ++offset_x) {
      const int x = center_x + offset_x;
      if (x < 0 || x >= config.source_width) {
        continue;
      }
      const float candidate =
        source_depth[y * config.source_width + x];
      if (
        valid_depth(candidate, config) &&
        fabsf(candidate - center) <= threshold)
      {
        sum += candidate;
        ++count;
      }
    }
  }
  filtered_depth[index] =
    count > 0 ? sum / static_cast<float>(count) : center;
}

__global__ void temporal_depth_filter_kernel(
  const float * current_depth,
  float * previous_depth,
  float * filtered_depth,
  CudaPanoramaConfig config)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixel_count = config.source_width * config.source_height;
  if (index >= pixel_count) {
    return;
  }

  const float current = current_depth[index];
  if (!valid_depth(current, config)) {
    // Do not persist old silhouettes into a newly invalid pixel.
    previous_depth[index] = 0.0F;
    filtered_depth[index] = 0.0F;
    return;
  }

  const float previous = previous_depth[index];
  float output = current;
  if (
    valid_depth(previous, config) &&
    fabsf(current - previous) <= config.depth_temporal_reset_m)
  {
    output =
      (1.0F - config.depth_temporal_alpha) * previous +
      config.depth_temporal_alpha * current;
  }
  previous_depth[index] = output;
  filtered_depth[index] = output;
}

__global__ void convert_depth_kernel(
  const unsigned short * source,
  float * destination,
  float scale,
  int pixel_count)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) {
    return;
  }
  destination[index] = static_cast<float>(source[index]) * scale;
}

__device__ __forceinline__ bool project_depth_pixel(
  int u,
  int v,
  float depth,
  const CudaCameraModel & camera,
  const CudaPanoramaConfig & config,
  float & output_x,
  float & output_y,
  float & horizontal_range)
{
  const float local_x =
    (static_cast<float>(u) - camera.cx) / camera.fx * depth;
  const float local_y =
    (static_cast<float>(v) - camera.cy) / camera.fy * depth;
  const float rig_x =
    camera.rotation_camera_to_rig[0] * local_x +
    camera.rotation_camera_to_rig[1] * local_y +
    camera.rotation_camera_to_rig[2] * depth +
    camera.translation_camera_in_rig[0];
  const float rig_y =
    camera.rotation_camera_to_rig[3] * local_x +
    camera.rotation_camera_to_rig[4] * local_y +
    camera.rotation_camera_to_rig[5] * depth +
    camera.translation_camera_in_rig[1];
  const float rig_z =
    camera.rotation_camera_to_rig[6] * local_x +
    camera.rotation_camera_to_rig[7] * local_y +
    camera.rotation_camera_to_rig[8] * depth +
    camera.translation_camera_in_rig[2];
  if (rig_z <= 0.0F) {
    return false;
  }

  horizontal_range = hypotf(rig_x, rig_z);
  if (config.projection_model == 1) {
    output_x =
      config.virtual_fx_px * rig_x / rig_z +
      config.virtual_cx_px;
    output_y =
      config.virtual_fy_px * rig_y / rig_z +
      config.virtual_cy_px;
  } else {
    const float global_angle = atan2f(rig_x, rig_z);
    output_x =
      (global_angle - config.panorama_min_angle) *
      config.panorama_focal_px;
    output_y =
      config.panorama_focal_px * rig_y / horizontal_range -
      config.panorama_min_vertical;
  }
  return
    output_x >= -static_cast<float>(config.depth_splat_radius_px) &&
    output_x < static_cast<float>(
      config.panorama_width + config.depth_splat_radius_px) &&
    output_y >= -static_cast<float>(config.depth_splat_radius_px) &&
    output_y < static_cast<float>(
      config.panorama_height + config.depth_splat_radius_px);
}

__global__ void project_min_range_kernel(
  const float * depth_m,
  unsigned int * minimum_ranges,
  CudaCameraModel camera,
  CudaPanoramaConfig config)
{
  const int column_count =
    camera.maximum_depth_column - camera.minimum_depth_column + 1;
  const int work_items = column_count * config.source_height;
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= work_items || column_count <= 0) {
    return;
  }

  const int v = item / column_count;
  const int u = camera.minimum_depth_column + item % column_count;
  const int source_index = v * config.source_width + u;
  const float depth = depth_m[source_index];
  if (!valid_depth(depth, config)) {
    return;
  }

  float projected_x;
  float projected_y;
  float horizontal_range;
  if (!project_depth_pixel(
      u, v, depth, camera, config,
      projected_x, projected_y, horizontal_range))
  {
    return;
  }

  const int output_x = __float2int_rn(projected_x);
  const int output_y = __float2int_rn(projected_y);
  const bool on_depth_edge = depth_discontinuity(
    depth_m, u, v, depth, config);
  const int splat_radius = on_depth_edge ?
    config.depth_edge_splat_radius_px :
    config.depth_splat_radius_px;
  for (int offset_y = -splat_radius; offset_y <= splat_radius; ++offset_y) {
    const int target_y = output_y + offset_y;
    if (target_y < 0 || target_y >= config.panorama_height) {
      continue;
    }
    for (int offset_x = -splat_radius; offset_x <= splat_radius; ++offset_x) {
      const int target_x = output_x + offset_x;
      if (target_x < 0 || target_x >= config.panorama_width) {
        continue;
      }
      atomicMin(
        &minimum_ranges[target_y * config.panorama_width + target_x],
        __float_as_uint(horizontal_range));
    }
  }
}

__global__ void project_depth_kernel(
  const float * depth_m,
  const unsigned int * minimum_ranges,
  unsigned long long * projection_keys,
  unsigned int * accepted_points,
  CudaCameraModel camera,
  CudaPanoramaConfig config)
{
  const int column_count =
    camera.maximum_depth_column - camera.minimum_depth_column + 1;
  const int work_items = column_count * config.source_height;
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= work_items || column_count <= 0) {
    return;
  }

  const int v = item / column_count;
  const int u = camera.minimum_depth_column + item % column_count;
  const int source_index = v * config.source_width + u;
  const float depth = depth_m[source_index];
  if (!valid_depth(depth, config)) {
    return;
  }

  float projected_x;
  float projected_y;
  float horizontal_range;
  if (!project_depth_pixel(
      u, v, depth, camera, config,
      projected_x, projected_y, horizontal_range))
  {
    return;
  }

  const int output_x = __float2int_rn(projected_x);
  const int output_y = __float2int_rn(projected_y);
  const bool on_depth_edge = depth_discontinuity(
    depth_m, u, v, depth, config);
  const int splat_radius = on_depth_edge ?
    config.depth_edge_splat_radius_px :
    config.depth_splat_radius_px;
  const float same_surface_margin =
    fmaxf(config.occlusion_switch_margin_m, 0.02F);
  const unsigned int range_mm = min(
    static_cast<unsigned int>(
      __float2uint_rn(horizontal_range * 1000.0F)),
    65534U);
  for (int offset_y = -splat_radius; offset_y <= splat_radius; ++offset_y) {
    const int target_y = output_y + offset_y;
    if (target_y < 0 || target_y >= config.panorama_height) {
      continue;
    }
    for (int offset_x = -splat_radius; offset_x <= splat_radius; ++offset_x) {
      const int target_x = output_x + offset_x;
      if (target_x < 0 || target_x >= config.panorama_width) {
        continue;
      }
      const int target_index =
        target_y * config.panorama_width + target_x;
      const unsigned int minimum_range_bits =
        minimum_ranges[target_index];
      if (minimum_range_bits == kInvalidProjectionRange) {
        continue;
      }
      const float minimum_range =
        __uint_as_float(minimum_range_bits);
      if (horizontal_range > minimum_range + same_surface_margin) {
        continue;
      }

      // Once z-buffering has rejected the hidden surface, select the sample
      // whose projected center is closest to this output pixel. Ordering
      // splats by raw depth here made millimetre depth noise repeatedly copy
      // a neighbouring RGB pixel across otherwise flat surfaces.
      const float delta_x =
        static_cast<float>(target_x) - projected_x;
      const float delta_y =
        static_cast<float>(target_y) - projected_y;
      const float distance_squared =
        delta_x * delta_x + delta_y * delta_y;
      const unsigned int distance_key = min(
        static_cast<unsigned int>(
          __float2uint_rn(distance_squared * 4096.0F)),
        65534U);
      const unsigned long long key =
        (static_cast<unsigned long long>(distance_key) << 48) |
        (static_cast<unsigned long long>(range_mm) << 32) |
        static_cast<unsigned int>(source_index);
      atomicMin(&projection_keys[target_index], key);
    }
  }
  atomicAdd(accepted_points, 1U);
}

__global__ void remap_color_kernel(
  const unsigned char * source,
  const float * map_x,
  const float * map_y,
  const unsigned char * valid_mask,
  unsigned char * remapped,
  CudaPanoramaConfig config)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= config.panorama_width || y >= config.panorama_height) {
    return;
  }

  const int output_index = y * config.panorama_width + x;
  const int output_color_index = output_index * 3;
  if (valid_mask[output_index] == 0U) {
    remapped[output_color_index] = 0U;
    remapped[output_color_index + 1] = 0U;
    remapped[output_color_index + 2] = 0U;
    return;
  }

  const float source_x = map_x[output_index];
  const float source_y = map_y[output_index];
  const int x0 = __float2int_rd(source_x);
  const int y0 = __float2int_rd(source_y);
  const int x1 = x0 + 1 < config.source_width ? x0 + 1 : x0;
  const int y1 = y0 + 1 < config.source_height ? y0 + 1 : y0;
  const float weight_x = source_x - static_cast<float>(x0);
  const float weight_y = source_y - static_cast<float>(y0);
  const float weight_00 = (1.0F - weight_x) * (1.0F - weight_y);
  const float weight_10 = weight_x * (1.0F - weight_y);
  const float weight_01 = (1.0F - weight_x) * weight_y;
  const float weight_11 = weight_x * weight_y;
  const int index_00 = (y0 * config.source_width + x0) * 3;
  const int index_10 = (y0 * config.source_width + x1) * 3;
  const int index_01 = (y1 * config.source_width + x0) * 3;
  const int index_11 = (y1 * config.source_width + x1) * 3;
  for (int channel = 0; channel < 3; ++channel) {
    // The remapped base image is always BGR. When the publisher sends rgb8 the
    // channel order is reversed here instead of on the CPU.
    const int destination_channel =
      config.source_channel_swap ? 2 - channel : channel;
    remapped[output_color_index + destination_channel] = float_channel(
      weight_00 * static_cast<float>(source[index_00 + channel]) +
      weight_10 * static_cast<float>(source[index_10 + channel]) +
      weight_01 * static_cast<float>(source[index_01 + channel]) +
      weight_11 * static_cast<float>(source[index_11 + channel]));
  }
}

__device__ __forceinline__ unsigned long long nearest_projection_key(
  const unsigned long long * keys,
  int x,
  int y,
  const CudaPanoramaConfig & config)
{
  const int index = y * config.panorama_width + x;
  unsigned long long key = keys[index];
  if (
    key != kInvalidProjectionKey ||
    config.projected_hole_radius <= 0)
  {
    return key;
  }

  unsigned long long closest = kInvalidProjectionKey;
  int closest_distance = 2147483647;
  for (
    int offset_y = -config.projected_hole_radius;
    offset_y <= config.projected_hole_radius;
    ++offset_y)
  {
    const int neighbor_y = y + offset_y;
    if (neighbor_y < 0 || neighbor_y >= config.panorama_height) {
      continue;
    }
    for (
      int offset_x = -config.projected_hole_radius;
      offset_x <= config.projected_hole_radius;
      ++offset_x)
    {
      const int neighbor_x = x + offset_x;
      if (neighbor_x < 0 || neighbor_x >= config.panorama_width) {
        continue;
      }
      const unsigned long long candidate =
        keys[neighbor_y * config.panorama_width + neighbor_x];
      if (candidate == kInvalidProjectionKey) {
        continue;
      }
      const int distance =
        offset_x * offset_x + offset_y * offset_y;
      if (distance < closest_distance ||
        (distance == closest_distance && candidate < closest))
      {
        closest_distance = distance;
        closest = candidate;
      }
    }
  }
  return closest;
}

__device__ __forceinline__ void source_color(
  const unsigned char * source,
  unsigned long long key,
  float gain_b,
  float gain_g,
  float gain_r,
  bool swap_channels,
  unsigned char & blue,
  unsigned char & green,
  unsigned char & red)
{
  const unsigned int source_index =
    static_cast<unsigned int>(key & 0xffffffffULL);
  const unsigned int color_index = source_index * 3U;
  const unsigned int blue_offset = swap_channels ? 2U : 0U;
  const unsigned int red_offset = swap_channels ? 0U : 2U;
  blue = gained_channel(source[color_index + blue_offset], gain_b);
  green = gained_channel(source[color_index + 1U], gain_g);
  red = gained_channel(source[color_index + red_offset], gain_r);
}

__device__ __forceinline__ float projection_range(
  unsigned long long key)
{
  const unsigned int range_mm =
    static_cast<unsigned int>((key >> 32) & 0xffffULL);
  return static_cast<float>(range_mm) * 0.001F;
}

__device__ __forceinline__ float pixel_luma(
  unsigned char blue, unsigned char green, unsigned char red)
{
  return
    0.114F * static_cast<float>(blue) +
    0.587F * static_cast<float>(green) +
    0.299F * static_cast<float>(red);
}

__global__ void seam_cost_kernel(
  const unsigned char * left_source,
  const unsigned char * right_source,
  const unsigned char * left_base,
  const unsigned char * right_base,
  const unsigned char * left_base_mask,
  const unsigned char * right_base_mask,
  const unsigned long long * left_projection_keys,
  const unsigned long long * right_projection_keys,
  float * seam_cost,
  float right_gain_b,
  float right_gain_g,
  float right_gain_r,
  CudaPanoramaConfig config)
{
  const int overlap_width =
    config.depth_color_max_x - config.depth_color_min_x + 1;
  const int local_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (local_x >= overlap_width || y >= config.panorama_height) {
    return;
  }

  const int x = config.depth_color_min_x + local_x;
  const int pixel_index = y * config.panorama_width + x;
  const int color_index = pixel_index * 3;
  const int cost_index = y * overlap_width + local_x;
  const bool left_base_valid = left_base_mask[pixel_index] != 0U;
  const bool right_base_valid = right_base_mask[pixel_index] != 0U;
  if (!left_base_valid || !right_base_valid) {
    seam_cost[cost_index] = 50.0F;
    return;
  }

  const unsigned long long left_key = config.depth_aware_color ?
    nearest_projection_key(
    left_projection_keys, x, y, config) : kInvalidProjectionKey;
  const unsigned long long right_key = config.depth_aware_color ?
    nearest_projection_key(
    right_projection_keys, x, y, config) : kInvalidProjectionKey;
  const bool left_depth_valid = left_key != kInvalidProjectionKey;
  const bool right_depth_valid = right_key != kInvalidProjectionKey;

  unsigned char left_blue = left_base[color_index];
  unsigned char left_green = left_base[color_index + 1];
  unsigned char left_red = left_base[color_index + 2];
  unsigned char right_blue = gained_channel(
    right_base[color_index], right_gain_b);
  unsigned char right_green = gained_channel(
    right_base[color_index + 1], right_gain_g);
  unsigned char right_red = gained_channel(
    right_base[color_index + 2], right_gain_r);
  if (left_depth_valid) {
    source_color(
      left_source, left_key, 1.0F, 1.0F, 1.0F,
      config.source_channel_swap,
      left_blue, left_green, left_red);
  }
  if (right_depth_valid) {
    source_color(
      right_source, right_key,
      right_gain_b, right_gain_g, right_gain_r,
      config.source_channel_swap,
      right_blue, right_green, right_red);
  }

  const float color_difference =
    (
    fabsf(static_cast<float>(left_blue) - right_blue) +
    fabsf(static_cast<float>(left_green) - right_green) +
    fabsf(static_cast<float>(left_red) - right_red)) /
    (3.0F * 255.0F);

  float depth_mismatch = 0.0F;
  float foreground_cost = 0.0F;
  if (left_depth_valid && right_depth_valid) {
    const float left_range = projection_range(left_key);
    const float right_range = projection_range(right_key);
    const float depth_scale = fmaxf(
      config.occlusion_switch_margin_m, 0.05F);
    depth_mismatch = fminf(
      fabsf(left_range - right_range) / depth_scale, 4.0F);
    foreground_cost =
      1.0F / fmaxf(fminf(left_range, right_range), 0.35F);
  } else if (left_depth_valid || right_depth_valid) {
    depth_mismatch = 2.5F;
    const float range = left_depth_valid ?
      projection_range(left_key) : projection_range(right_key);
    foreground_cost = 1.0F / fmaxf(range, 0.35F);
  } else {
    // A depth hole is allowed, but it is a less reliable place to cut than a
    // background surface observed by both cameras.
    depth_mismatch = 0.75F;
  }

  // A weak image-gradient penalty stops the path from running along a sharp
  // silhouette when both projected colors happen to be locally similar.
  float edge_cost = 0.0F;
  if (local_x > 0 && local_x + 1 < overlap_width) {
    const int left_color_index = color_index - 3;
    const int right_color_index = color_index + 3;
    const float left_luma_gradient = fabsf(
      pixel_luma(
        left_base[right_color_index],
        left_base[right_color_index + 1],
        left_base[right_color_index + 2]) -
      pixel_luma(
        left_base[left_color_index],
        left_base[left_color_index + 1],
        left_base[left_color_index + 2]));
    const float right_luma_gradient = fabsf(
      pixel_luma(
        right_base[right_color_index],
        right_base[right_color_index + 1],
        right_base[right_color_index + 2]) -
      pixel_luma(
        right_base[left_color_index],
        right_base[left_color_index + 1],
        right_base[left_color_index + 2]));
    edge_cost =
      0.5F * (left_luma_gradient + right_luma_gradient) / 255.0F;
  }

  seam_cost[cost_index] =
    config.seam_color_weight * (color_difference + 0.35F * edge_cost) +
    config.seam_depth_weight * depth_mismatch +
    config.seam_foreground_weight * foreground_cost;
}

__global__ void compose_panorama_kernel(
  const unsigned char * left_source,
  const unsigned char * right_source,
  const unsigned char * left_base,
  const unsigned char * right_base,
  const unsigned char * left_base_mask,
  const unsigned char * right_base_mask,
  const unsigned long long * left_projection_keys,
  const unsigned long long * right_projection_keys,
  const int * seam_by_row,
  unsigned char * output,
  unsigned char * validity,
  float * output_range_m,
  float right_gain_b,
  float right_gain_g,
  float right_gain_r,
  CudaPanoramaConfig config)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= config.panorama_width || y >= config.panorama_height) {
    return;
  }

  const int pixel_index = y * config.panorama_width + x;
  const int color_index = pixel_index * 3;
  const int seam_x = config.content_aware_seam ?
    seam_by_row[y] : config.seam_x;
  const bool left_base_valid = left_base_mask[pixel_index] != 0U;
  const bool right_base_valid = right_base_mask[pixel_index] != 0U;
  if (!left_base_valid && !right_base_valid) {
    output[color_index] = 0U;
    output[color_index + 1] = 0U;
    output[color_index + 2] = 0U;
    validity[pixel_index] = 0U;
    output_range_m[pixel_index] = 0.0F;
    return;
  }

  const unsigned long long left_key = config.depth_aware_color ?
    nearest_projection_key(
    left_projection_keys, x, y, config) : kInvalidProjectionKey;
  const unsigned long long right_key = config.depth_aware_color ?
    nearest_projection_key(
    right_projection_keys, x, y, config) : kInvalidProjectionKey;
  const bool left_depth_valid = left_key != kInvalidProjectionKey;
  const bool right_depth_valid = right_key != kInvalidProjectionKey;
  const bool render_depth_color_here =
    config.render_depth_reprojected_color &&
    (!config.depth_color_overlap_only ||
    (x >= config.depth_color_min_x && x <= config.depth_color_max_x));

  // RGB and range serve different downstream purposes. The range image keeps
  // the full metric reprojection, while RGB may remain in one smooth
  // rotation-based projection so depth holes and silhouette noise do not
  // make the visible image shimmer across the entire field of view.
  if (!config.depth_aware_color ||
    !render_depth_color_here)
  {
    if (left_depth_valid || right_depth_valid) {
      bool range_uses_left = left_depth_valid;
      if (left_depth_valid && right_depth_valid) {
        range_uses_left =
          config.prefer_seam_camera_when_both_depth_valid ?
          x <= seam_x :
          projection_range(left_key) <= projection_range(right_key);
      }
      validity[pixel_index] = 255U;
      output_range_m[pixel_index] = range_uses_left ?
        projection_range(left_key) : projection_range(right_key);
    } else {
      validity[pixel_index] = 0U;
      output_range_m[pixel_index] = 0.0F;
    }
    const int blend_left = seam_x - config.seam_feather_px;
    const int blend_right = seam_x + config.seam_feather_px;
    if (!right_base_valid || (left_base_valid && x < blend_left)) {
      output[color_index] = left_base[color_index];
      output[color_index + 1] = left_base[color_index + 1];
      output[color_index + 2] = left_base[color_index + 2];
      return;
    }
    if (!left_base_valid || x > blend_right) {
      output[color_index] = gained_channel(
        right_base[color_index], right_gain_b);
      output[color_index + 1] = gained_channel(
        right_base[color_index + 1], right_gain_g);
      output[color_index + 2] = gained_channel(
        right_base[color_index + 2], right_gain_r);
      return;
    }

    const int feather_width = 2 * config.seam_feather_px;
    const float denominator =
      static_cast<float>(feather_width > 1 ? feather_width : 1);
    const float right_weight = fminf(fmaxf(
        static_cast<float>(x - blend_left) / denominator,
        0.0F), 1.0F);
    const float left_weight = 1.0F - right_weight;
    output[color_index] = float_channel(
      left_weight * static_cast<float>(left_base[color_index]) +
      right_weight * static_cast<float>(right_base[color_index]) *
      right_gain_b);
    output[color_index + 1] = float_channel(
      left_weight * static_cast<float>(left_base[color_index + 1]) +
      right_weight * static_cast<float>(right_base[color_index + 1]) *
      right_gain_g);
    output[color_index + 2] = float_channel(
      left_weight * static_cast<float>(left_base[color_index + 2]) +
      right_weight * static_cast<float>(right_base[color_index + 2]) *
      right_gain_r);
    return;
  }

  if (left_depth_valid || right_depth_valid) {
    if (config.prefer_seam_camera_when_both_depth_valid) {
      const bool owner_is_left = x <= seam_x;
      if (left_depth_valid && right_depth_valid) {
        const float left_range = projection_range(left_key);
        const float right_range = projection_range(right_key);
        const int blend_left = seam_x - config.seam_feather_px;
        const int blend_right = seam_x + config.seam_feather_px;
        if (
          config.seam_feather_px > 0 &&
          x >= blend_left && x <= blend_right &&
          fabsf(left_range - right_range) <=
          config.occlusion_switch_margin_m)
        {
          unsigned char left_blue;
          unsigned char left_green;
          unsigned char left_red;
          unsigned char right_blue;
          unsigned char right_green;
          unsigned char right_red;
          source_color(
            left_source, left_key, 1.0F, 1.0F, 1.0F,
            config.source_channel_swap,
            left_blue, left_green, left_red);
          source_color(
            right_source, right_key,
            right_gain_b, right_gain_g, right_gain_r,
            config.source_channel_swap,
            right_blue, right_green, right_red);
          const float right_weight = static_cast<float>(
            x - blend_left) /
            static_cast<float>(2 * config.seam_feather_px);
          output[color_index] = static_cast<unsigned char>(
            (1.0F - right_weight) * left_blue +
            right_weight * right_blue);
          output[color_index + 1] = static_cast<unsigned char>(
            (1.0F - right_weight) * left_green +
            right_weight * right_green);
          output[color_index + 2] = static_cast<unsigned char>(
            (1.0F - right_weight) * left_red +
            right_weight * right_red);
          validity[pixel_index] = 255U;
          output_range_m[pixel_index] = fminf(left_range, right_range);
          return;
        }
      }

      // Keep one camera owner on each side of the seam. If that camera has a
      // depth hole, its rotation-warped color is safer than injecting a
      // fragment from the other viewpoint into the foreground object.
      if (owner_is_left && left_base_valid) {
        if (left_depth_valid) {
          source_color(
            left_source, left_key, 1.0F, 1.0F, 1.0F,
            config.source_channel_swap,
            output[color_index], output[color_index + 1],
            output[color_index + 2]);
          validity[pixel_index] = 255U;
          output_range_m[pixel_index] = projection_range(left_key);
        } else {
          output[color_index] = config.allow_color_fallback ?
            left_base[color_index] : 0U;
          output[color_index + 1] = config.allow_color_fallback ?
            left_base[color_index + 1] : 0U;
          output[color_index + 2] = config.allow_color_fallback ?
            left_base[color_index + 2] : 0U;
          validity[pixel_index] = 0U;
          output_range_m[pixel_index] = 0.0F;
        }
        return;
      }
      if (!owner_is_left && right_base_valid) {
        if (right_depth_valid) {
          source_color(
            right_source, right_key,
            right_gain_b, right_gain_g, right_gain_r,
            config.source_channel_swap,
            output[color_index], output[color_index + 1],
            output[color_index + 2]);
          validity[pixel_index] = 255U;
          output_range_m[pixel_index] = projection_range(right_key);
        } else {
          output[color_index] = config.allow_color_fallback ?
            gained_channel(right_base[color_index], right_gain_b) : 0U;
          output[color_index + 1] = config.allow_color_fallback ?
            gained_channel(right_base[color_index + 1], right_gain_g) : 0U;
          output[color_index + 2] = config.allow_color_fallback ?
            gained_channel(right_base[color_index + 2], right_gain_r) : 0U;
          validity[pixel_index] = 0U;
          output_range_m[pixel_index] = 0.0F;
        }
        return;
      }
    }

    bool use_left = left_depth_valid;
    if (left_depth_valid && right_depth_valid) {
      const float left_range = projection_range(left_key);
      const float right_range = projection_range(right_key);
      if (
        fabsf(left_range - right_range) <=
        config.occlusion_switch_margin_m)
      {
        use_left = x <= seam_x;
      } else {
        use_left = left_range <= right_range;
      }
    }

    unsigned char blue;
    unsigned char green;
    unsigned char red;
    if (use_left) {
      source_color(
        left_source, left_key, 1.0F, 1.0F, 1.0F,
        config.source_channel_swap,
        blue, green, red);
    } else {
      source_color(
        right_source, right_key,
        right_gain_b, right_gain_g, right_gain_r,
        config.source_channel_swap,
        blue, green, red);
    }
    output[color_index] = blue;
    output[color_index + 1] = green;
    output[color_index + 2] = red;
    validity[pixel_index] = 255U;
    output_range_m[pixel_index] = use_left ?
      projection_range(left_key) :
      projection_range(right_key);
    return;
  }

  validity[pixel_index] = 0U;
  output_range_m[pixel_index] = 0.0F;
  if (!config.allow_color_fallback) {
    output[color_index] = 0U;
    output[color_index + 1] = 0U;
    output[color_index + 2] = 0U;
    return;
  }

  const int blend_left =
    seam_x - config.seam_feather_px;
  const int blend_right =
    seam_x + config.seam_feather_px;
  if (!right_base_valid || (left_base_valid && x < blend_left)) {
    output[color_index] = left_base[color_index];
    output[color_index + 1] = left_base[color_index + 1];
    output[color_index + 2] = left_base[color_index + 2];
    return;
  }
  if (!left_base_valid || x > blend_right) {
    output[color_index] = gained_channel(
      right_base[color_index], right_gain_b);
    output[color_index + 1] = gained_channel(
      right_base[color_index + 1], right_gain_g);
    output[color_index + 2] = gained_channel(
      right_base[color_index + 2], right_gain_r);
    return;
  }

  const int feather_width = 2 * config.seam_feather_px;
  const float denominator =
    static_cast<float>(feather_width > 1 ? feather_width : 1);
  const float right_weight = fminf(fmaxf(
      static_cast<float>(x - blend_left) / denominator,
      0.0F), 1.0F);
  const float left_weight = 1.0F - right_weight;
  output[color_index] = float_channel(
    left_weight * static_cast<float>(left_base[color_index]) +
    right_weight * static_cast<float>(right_base[color_index]) *
    right_gain_b);
  output[color_index + 1] = float_channel(
    left_weight * static_cast<float>(left_base[color_index + 1]) +
    right_weight * static_cast<float>(right_base[color_index + 1]) *
    right_gain_g);
  output[color_index + 2] = float_channel(
    left_weight * static_cast<float>(left_base[color_index + 2]) +
    right_weight * static_cast<float>(right_base[color_index + 2]) *
    right_gain_r);
}

// Building the cloud on the GPU keeps the 12 MB range image and the 3 MB
// validity mask on the device. Only the compacted points cross PCIe, and the
// per-pixel trigonometry no longer runs on the CPU.
struct CudaPointCloudLayout
{
  int point_step;
  int x_offset;
  int y_offset;
  int z_offset;
  int rgb_offset;
};

__global__ void build_pointcloud_kernel(
  const unsigned char * panorama,
  const unsigned char * validity,
  const float * range_m,
  unsigned char * points,
  unsigned int * point_count,
  int capacity_points,
  CudaPointCloudLayout layout,
  CudaPanoramaConfig config)
{
  const int stride = config.pointcloud_stride;
  if (stride <= 0) {
    return;
  }
  const int sampled_columns =
    (config.panorama_width + stride - 1) / stride;
  const int sampled_rows =
    (config.panorama_height + stride - 1) / stride;
  const int sample_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int sample_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (sample_x >= sampled_columns || sample_y >= sampled_rows) {
    return;
  }

  const int x = sample_x * stride;
  const int y = sample_y * stride;
  const int pixel_index = y * config.panorama_width + x;
  const float horizontal_range = range_m[pixel_index];
  if (
    validity[pixel_index] == 0U ||
    !isfinite(horizontal_range) || horizontal_range <= 0.0F)
  {
    return;
  }

  float point_x;
  float point_y;
  float point_z;
  if (config.projection_model == 1) {
    const float ray_x =
      (static_cast<float>(x) - config.virtual_cx_px) / config.virtual_fx_px;
    const float ray_y =
      (static_cast<float>(y) - config.virtual_cy_px) / config.virtual_fy_px;
    point_z = horizontal_range / hypotf(ray_x, 1.0F);
    point_x = ray_x * point_z;
    point_y = ray_y * point_z;
  } else {
    const float angle =
      config.panorama_min_angle +
      static_cast<float>(x) / config.panorama_focal_px;
    const float vertical_ratio =
      (config.panorama_min_vertical + static_cast<float>(y)) /
      config.panorama_focal_px;
    point_x = horizontal_range * sinf(angle);
    point_y = horizontal_range * vertical_ratio;
    point_z = horizontal_range * cosf(angle);
  }

  const unsigned int slot = atomicAdd(point_count, 1U);
  if (static_cast<int>(slot) >= capacity_points) {
    return;
  }
  const int color_index = pixel_index * 3;
  const unsigned int rgb =
    (static_cast<unsigned int>(panorama[color_index + 2]) << 16U) |
    (static_cast<unsigned int>(panorama[color_index + 1]) << 8U) |
    static_cast<unsigned int>(panorama[color_index]);
  unsigned char * destination =
    points + static_cast<std::size_t>(slot) *
    static_cast<std::size_t>(layout.point_step);
  *reinterpret_cast<float *>(destination + layout.x_offset) = point_x;
  *reinterpret_cast<float *>(destination + layout.y_offset) = point_y;
  *reinterpret_cast<float *>(destination + layout.z_offset) = point_z;
  *reinterpret_cast<unsigned int *>(destination + layout.rgb_offset) = rgb;
}

std::vector<int> find_content_aware_seam(
  const std::vector<float> & cost,
  int width,
  int height,
  int minimum_x,
  const CudaPanoramaConfig & config,
  const std::vector<int> & previous_seam)
{
  std::vector<int> seam(
    static_cast<std::size_t>(height), config.seam_x);
  if (
    width <= 0 || height <= 0 ||
    cost.size() != static_cast<std::size_t>(width * height))
  {
    return seam;
  }

  // A per-row seam can route around an object, but it creates the visible
  // "wriggling" boundary that is unacceptable for detector input. Use one
  // coherent vertical ownership boundary and move that boundary only when a
  // different overlap column is consistently safer over the full frame.
  const float half_width = std::max(0.5F * width, 1.0F);
  int previous_x = config.seam_x;
  if (previous_seam.size() == static_cast<std::size_t>(height)) {
    long long previous_sum = 0;
    for (const int value : previous_seam) {
      previous_sum += value;
    }
    previous_x = static_cast<int>(
      previous_sum / std::max(height, 1));
  }

  float best_cost = std::numeric_limits<float>::infinity();
  int best_absolute_x = config.seam_x;
  for (int x = 0; x < width; ++x) {
    float accumulated = 0.0F;
    for (int y = 0; y < height; ++y) {
      const float measured =
        cost[static_cast<std::size_t>(y * width + x)];
      // Cap isolated depth outliers so one bad pixel cannot move the full
      // height ownership boundary.
      accumulated += std::min(
        std::isfinite(measured) ? measured : 50.0F, 8.0F);
    }
    const int absolute_x = minimum_x + x;
    accumulated /= static_cast<float>(height);
    accumulated +=
      config.seam_center_weight *
      std::abs(static_cast<float>(absolute_x - config.seam_x)) /
      half_width;
    accumulated +=
      config.seam_temporal_weight *
      std::abs(static_cast<float>(absolute_x - previous_x)) /
      half_width;
    if (accumulated < best_cost) {
      best_cost = accumulated;
      best_absolute_x = absolute_x;
    }
  }

  const int maximum_step = std::max(config.seam_max_step_px, 1);
  const int selected_x = std::clamp(
    best_absolute_x,
    previous_x - maximum_step,
    previous_x + maximum_step);
  std::fill(seam.begin(), seam.end(), selected_x);
  return seam;
}

}  // namespace

struct CudaPanoramaBackend::Impl
{
  CudaPanoramaConfig config;
  CudaCameraModel left_camera;
  CudaCameraModel right_camera;
  bool configured{false};
  cudaStream_t stream{nullptr};
  cudaEvent_t start_event{nullptr};
  cudaEvent_t stop_event{nullptr};
  std::string initialization_error;
  bool quarantined{false};
  std::string quarantine_reason;

  unsigned char * left_source{nullptr};
  unsigned char * right_source{nullptr};
  unsigned short * left_raw_depth{nullptr};
  unsigned short * right_raw_depth{nullptr};
  float * left_depth{nullptr};
  float * right_depth{nullptr};
  float * left_spatial_depth{nullptr};
  float * right_spatial_depth{nullptr};
  float * left_filtered_depth{nullptr};
  float * right_filtered_depth{nullptr};
  float * left_previous_depth{nullptr};
  float * right_previous_depth{nullptr};
  unsigned char * left_base{nullptr};
  unsigned char * right_base{nullptr};
  unsigned char * left_base_mask{nullptr};
  unsigned char * right_base_mask{nullptr};
  float * left_map_x{nullptr};
  float * left_map_y{nullptr};
  float * right_map_x{nullptr};
  float * right_map_y{nullptr};
  unsigned int * left_minimum_ranges{nullptr};
  unsigned int * right_minimum_ranges{nullptr};
  unsigned long long * left_projection_keys{nullptr};
  unsigned long long * right_projection_keys{nullptr};
  unsigned int * left_accepted_points{nullptr};
  unsigned int * right_accepted_points{nullptr};
  float * seam_cost{nullptr};
  int * seam_by_row{nullptr};
  unsigned char * output{nullptr};
  unsigned char * validity{nullptr};
  float * output_range_m{nullptr};
  unsigned char * pointcloud{nullptr};
  unsigned int * pointcloud_count{nullptr};
  int pointcloud_capacity{0};

  // CUDA never reads or writes memory owned by process() callers. These
  // page-locked staging buffers remain valid if a timed-out stream is still
  // draining while the node switches permanently to the CPU backend.
  unsigned char * left_source_host{nullptr};
  unsigned char * right_source_host{nullptr};
  unsigned char * left_depth_host{nullptr};
  unsigned char * right_depth_host{nullptr};
  unsigned char * output_host{nullptr};
  unsigned char * validity_host{nullptr};
  float * output_range_host{nullptr};
  float * seam_cost_staging{nullptr};
  unsigned char * pointcloud_host{nullptr};
  unsigned int * left_count_host{nullptr};
  unsigned int * right_count_host{nullptr};
  unsigned int * pointcloud_count_host{nullptr};
  std::vector<float> seam_cost_host;
  std::vector<int> seam_host;
  std::vector<int> previous_seam_host;

  void release()
  {
    // Once CUDA has timed out or reported a fatal error, even deallocation can
    // synchronize with the unhealthy context. Keep all handles alive until
    // process exit instead of turning a recoverable GPU failure into a node or
    // machine shutdown hang.
    if (quarantined) {
      configured = false;
      return;
    }
    cudaFree(left_source);
    cudaFree(right_source);
    cudaFree(left_raw_depth);
    cudaFree(right_raw_depth);
    cudaFree(left_depth);
    cudaFree(right_depth);
    cudaFree(left_spatial_depth);
    cudaFree(right_spatial_depth);
    cudaFree(left_filtered_depth);
    cudaFree(right_filtered_depth);
    cudaFree(left_previous_depth);
    cudaFree(right_previous_depth);
    cudaFree(left_base);
    cudaFree(right_base);
    cudaFree(left_base_mask);
    cudaFree(right_base_mask);
    cudaFree(left_map_x);
    cudaFree(left_map_y);
    cudaFree(right_map_x);
    cudaFree(right_map_y);
    cudaFree(left_minimum_ranges);
    cudaFree(right_minimum_ranges);
    cudaFree(left_projection_keys);
    cudaFree(right_projection_keys);
    cudaFree(left_accepted_points);
    cudaFree(right_accepted_points);
    cudaFree(seam_cost);
    cudaFree(seam_by_row);
    cudaFree(output);
    cudaFree(validity);
    cudaFree(output_range_m);
    cudaFree(pointcloud);
    cudaFree(pointcloud_count);
    if (left_source_host != nullptr) {
      cudaFreeHost(left_source_host);
    }
    if (right_source_host != nullptr) {
      cudaFreeHost(right_source_host);
    }
    if (left_depth_host != nullptr) {
      cudaFreeHost(left_depth_host);
    }
    if (right_depth_host != nullptr) {
      cudaFreeHost(right_depth_host);
    }
    if (output_host != nullptr) {
      cudaFreeHost(output_host);
    }
    if (validity_host != nullptr) {
      cudaFreeHost(validity_host);
    }
    if (output_range_host != nullptr) {
      cudaFreeHost(output_range_host);
    }
    if (seam_cost_staging != nullptr) {
      cudaFreeHost(seam_cost_staging);
    }
    if (pointcloud_host != nullptr) {
      cudaFreeHost(pointcloud_host);
    }
    if (left_count_host != nullptr) {
      cudaFreeHost(left_count_host);
    }
    if (right_count_host != nullptr) {
      cudaFreeHost(right_count_host);
    }
    if (pointcloud_count_host != nullptr) {
      cudaFreeHost(pointcloud_count_host);
    }
    left_source = nullptr;
    right_source = nullptr;
    left_raw_depth = nullptr;
    right_raw_depth = nullptr;
    left_depth = nullptr;
    right_depth = nullptr;
    left_spatial_depth = nullptr;
    right_spatial_depth = nullptr;
    left_filtered_depth = nullptr;
    right_filtered_depth = nullptr;
    left_previous_depth = nullptr;
    right_previous_depth = nullptr;
    left_base = nullptr;
    right_base = nullptr;
    left_base_mask = nullptr;
    right_base_mask = nullptr;
    left_map_x = nullptr;
    left_map_y = nullptr;
    right_map_x = nullptr;
    right_map_y = nullptr;
    left_minimum_ranges = nullptr;
    right_minimum_ranges = nullptr;
    left_projection_keys = nullptr;
    right_projection_keys = nullptr;
    left_accepted_points = nullptr;
    right_accepted_points = nullptr;
    seam_cost = nullptr;
    seam_by_row = nullptr;
    output = nullptr;
    validity = nullptr;
    output_range_m = nullptr;
    pointcloud = nullptr;
    pointcloud_count = nullptr;
    pointcloud_capacity = 0;
    left_source_host = nullptr;
    right_source_host = nullptr;
    left_depth_host = nullptr;
    right_depth_host = nullptr;
    output_host = nullptr;
    validity_host = nullptr;
    output_range_host = nullptr;
    seam_cost_staging = nullptr;
    pointcloud_host = nullptr;
    left_count_host = nullptr;
    right_count_host = nullptr;
    pointcloud_count_host = nullptr;
    seam_cost_host.clear();
    seam_host.clear();
    previous_seam_host.clear();
    configured = false;
  }
};

CudaPanoramaBackend::CudaPanoramaBackend()
: impl_(std::make_unique<Impl>())
{
  cudaError_t result = cudaStreamCreateWithFlags(
    &impl_->stream, cudaStreamNonBlocking);
  if (result != cudaSuccess) {
    impl_->initialization_error = cuda_error_message(
      "cudaStreamCreateWithFlags", result);
    return;
  }
  result = cudaEventCreate(&impl_->start_event);
  if (result != cudaSuccess) {
    impl_->initialization_error = cuda_error_message(
      "cudaEventCreate start", result);
    return;
  }
  result = cudaEventCreate(&impl_->stop_event);
  if (result != cudaSuccess) {
    impl_->initialization_error = cuda_error_message(
      "cudaEventCreate stop", result);
  }
}

CudaPanoramaBackend::~CudaPanoramaBackend()
{
  if (impl_->quarantined) {
    return;
  }
  if (impl_->stream != nullptr) {
    const cudaError_t stream_state = cudaStreamQuery(impl_->stream);
    if (stream_state != cudaSuccess) {
      impl_->quarantined = true;
      impl_->quarantine_reason = stream_state == cudaErrorNotReady ?
        "CUDA stream still active during destruction" :
        cuda_error_message("query CUDA stream during destruction", stream_state);
      return;
    }
  }
  impl_->release();
  if (impl_->start_event != nullptr) {
    cudaEventDestroy(impl_->start_event);
  }
  if (impl_->stop_event != nullptr) {
    cudaEventDestroy(impl_->stop_event);
  }
  if (impl_->stream != nullptr) {
    cudaStreamDestroy(impl_->stream);
  }
}

void CudaPanoramaBackend::quarantine(const std::string & reason) noexcept
{
  if (!impl_->quarantined) {
    impl_->quarantine_reason = reason.empty() ?
      "unspecified CUDA backend failure" : reason;
  }
  impl_->quarantined = true;
  impl_->configured = false;
}

bool CudaPanoramaBackend::is_quarantined() const noexcept
{
  return impl_->quarantined;
}

std::string CudaPanoramaBackend::quarantine_reason() const
{
  return impl_->quarantine_reason;
}

bool CudaPanoramaBackend::runtime_available(std::string & description)
{
  int device_count = 0;
  const cudaError_t count_result = cudaGetDeviceCount(&device_count);
  if (count_result != cudaSuccess || device_count <= 0) {
    description = count_result == cudaSuccess ?
      "no CUDA device found" :
      cuda_error_message("cudaGetDeviceCount", count_result);
    return false;
  }
  cudaDeviceProp properties{};
  const cudaError_t property_result =
    cudaGetDeviceProperties(&properties, 0);
  if (property_result != cudaSuccess) {
    description = cuda_error_message(
      "cudaGetDeviceProperties", property_result);
    return false;
  }
  std::ostringstream stream;
  stream << properties.name << " sm_" << properties.major <<
    properties.minor;
  description = stream.str();
  return true;
}

bool CudaPanoramaBackend::configure(
  const CudaPanoramaConfig & config,
  const CudaCameraModel & left_camera,
  const CudaCameraModel & right_camera,
  const cv::Mat & left_base_mask,
  const cv::Mat & right_base_mask,
  const cv::Mat & left_map_x,
  const cv::Mat & left_map_y,
  const cv::Mat & right_map_x,
  const cv::Mat & right_map_y,
  std::string & error)
{
  if (impl_->quarantined) {
    error = "CUDA backend is quarantined: " + impl_->quarantine_reason;
    return false;
  }
  impl_->release();
  if (!impl_->initialization_error.empty()) {
    error = impl_->initialization_error;
    return false;
  }
  impl_->config = config;
  impl_->config.operation_timeout_ms =
    config.operation_timeout_ms > 0 ?
    config.operation_timeout_ms : kDefaultGpuTimeoutMs;
  impl_->left_camera = left_camera;
  impl_->right_camera = right_camera;

  const std::size_t source_pixels =
    static_cast<std::size_t>(config.source_width) *
    static_cast<std::size_t>(config.source_height);
  const std::size_t panorama_pixels =
    static_cast<std::size_t>(config.panorama_width) *
    static_cast<std::size_t>(config.panorama_height);
  const std::size_t source_color_bytes = source_pixels * 3U;
  const std::size_t source_depth_bytes = source_pixels * sizeof(float);
  const std::size_t panorama_color_bytes = panorama_pixels * 3U;
  const std::size_t panorama_mask_bytes = panorama_pixels;
  const std::size_t panorama_map_bytes =
    panorama_pixels * sizeof(float);
  const std::size_t projection_key_bytes =
    panorama_pixels * sizeof(unsigned long long);
  const std::size_t projection_range_bytes =
    panorama_pixels * sizeof(unsigned int);
  const int seam_width = std::max(
    config.depth_color_max_x - config.depth_color_min_x + 1, 1);
  const std::size_t seam_cost_bytes =
    static_cast<std::size_t>(seam_width) *
    static_cast<std::size_t>(config.panorama_height) *
    sizeof(float);
  const std::size_t seam_row_bytes =
    static_cast<std::size_t>(config.panorama_height) * sizeof(int);
  const std::size_t source_raw_depth_bytes =
    source_pixels * sizeof(unsigned short);
  const int pointcloud_stride = std::max(config.pointcloud_stride, 0);
  impl_->pointcloud_capacity = pointcloud_stride > 0 ?
    ((config.panorama_width + pointcloud_stride - 1) / pointcloud_stride) *
    ((config.panorama_height + pointcloud_stride - 1) / pointcloud_stride) :
    0;
  const std::size_t pointcloud_bytes =
    static_cast<std::size_t>(std::max(impl_->pointcloud_capacity, 1)) *
    static_cast<std::size_t>(kMaxPointCloudStride);

  auto allocate = [&error](void ** pointer, std::size_t bytes, const char * name) {
      const cudaError_t result = cudaMalloc(pointer, bytes);
      if (result == cudaSuccess) {
        return true;
      }
      error = cuda_error_message(name, result);
      return false;
    };
  auto allocate_host = [&error](void ** pointer, std::size_t bytes, const char * name) {
      const cudaError_t result = cudaHostAlloc(pointer, bytes, cudaHostAllocDefault);
      if (result == cudaSuccess) {
        return true;
      }
      error = cuda_error_message(name, result);
      return false;
    };
  if (
    !allocate(
      reinterpret_cast<void **>(&impl_->left_source),
      source_color_bytes, "cudaMalloc left source") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->right_source),
      source_color_bytes, "cudaMalloc right source") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->left_raw_depth),
      source_raw_depth_bytes, "cudaMalloc left raw depth") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->right_raw_depth),
      source_raw_depth_bytes, "cudaMalloc right raw depth") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->left_depth),
      source_depth_bytes, "cudaMalloc left depth") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->right_depth),
      source_depth_bytes, "cudaMalloc right depth") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->left_spatial_depth),
      source_depth_bytes, "cudaMalloc left spatial depth") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->right_spatial_depth),
      source_depth_bytes, "cudaMalloc right spatial depth") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->left_filtered_depth),
      source_depth_bytes, "cudaMalloc left filtered depth") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->right_filtered_depth),
      source_depth_bytes, "cudaMalloc right filtered depth") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->left_previous_depth),
      source_depth_bytes, "cudaMalloc left previous depth") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->right_previous_depth),
      source_depth_bytes, "cudaMalloc right previous depth") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->left_base),
      panorama_color_bytes, "cudaMalloc left base") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->right_base),
      panorama_color_bytes, "cudaMalloc right base") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->left_base_mask),
      panorama_mask_bytes, "cudaMalloc left mask") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->right_base_mask),
      panorama_mask_bytes, "cudaMalloc right mask") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->left_map_x),
      panorama_map_bytes, "cudaMalloc left map x") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->left_map_y),
      panorama_map_bytes, "cudaMalloc left map y") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->right_map_x),
      panorama_map_bytes, "cudaMalloc right map x") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->right_map_y),
      panorama_map_bytes, "cudaMalloc right map y") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->left_minimum_ranges),
      projection_range_bytes, "cudaMalloc left minimum ranges") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->right_minimum_ranges),
      projection_range_bytes, "cudaMalloc right minimum ranges") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->left_projection_keys),
      projection_key_bytes, "cudaMalloc left keys") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->right_projection_keys),
      projection_key_bytes, "cudaMalloc right keys") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->left_accepted_points),
      sizeof(unsigned int), "cudaMalloc left count") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->right_accepted_points),
      sizeof(unsigned int), "cudaMalloc right count") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->seam_cost),
      seam_cost_bytes, "cudaMalloc seam cost") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->seam_by_row),
      seam_row_bytes, "cudaMalloc seam rows") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->output),
      panorama_color_bytes, "cudaMalloc output") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->validity),
      panorama_mask_bytes, "cudaMalloc validity") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->output_range_m),
      panorama_pixels * sizeof(float), "cudaMalloc output range") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->pointcloud),
      pointcloud_bytes, "cudaMalloc point cloud") ||
    !allocate(
      reinterpret_cast<void **>(&impl_->pointcloud_count),
      sizeof(unsigned int), "cudaMalloc point cloud count"))
  {
    impl_->release();
    return false;
  }
  if (
    !allocate_host(
      reinterpret_cast<void **>(&impl_->left_source_host),
      source_color_bytes, "cudaHostAlloc left source") ||
    !allocate_host(
      reinterpret_cast<void **>(&impl_->right_source_host),
      source_color_bytes, "cudaHostAlloc right source") ||
    !allocate_host(
      reinterpret_cast<void **>(&impl_->left_depth_host),
      source_depth_bytes, "cudaHostAlloc left depth") ||
    !allocate_host(
      reinterpret_cast<void **>(&impl_->right_depth_host),
      source_depth_bytes, "cudaHostAlloc right depth") ||
    !allocate_host(
      reinterpret_cast<void **>(&impl_->output_host),
      panorama_color_bytes, "cudaHostAlloc output") ||
    !allocate_host(
      reinterpret_cast<void **>(&impl_->validity_host),
      panorama_mask_bytes, "cudaHostAlloc validity") ||
    !allocate_host(
      reinterpret_cast<void **>(&impl_->output_range_host),
      panorama_pixels * sizeof(float), "cudaHostAlloc output range") ||
    !allocate_host(
      reinterpret_cast<void **>(&impl_->seam_cost_staging),
      seam_cost_bytes, "cudaHostAlloc seam cost") ||
    !allocate_host(
      reinterpret_cast<void **>(&impl_->pointcloud_host),
      pointcloud_bytes, "cudaHostAlloc point cloud") ||
    !allocate_host(
      reinterpret_cast<void **>(&impl_->left_count_host),
      sizeof(unsigned int), "cudaHostAlloc left count") ||
    !allocate_host(
      reinterpret_cast<void **>(&impl_->right_count_host),
      sizeof(unsigned int), "cudaHostAlloc right count") ||
    !allocate_host(
      reinterpret_cast<void **>(&impl_->pointcloud_count_host),
      sizeof(unsigned int), "cudaHostAlloc point cloud count"))
  {
    impl_->release();
    return false;
  }
  if (
    !check_cuda(
      cudaMemset(
        impl_->left_previous_depth, 0, source_depth_bytes),
      "initialize left depth history", error) ||
    !check_cuda(
      cudaMemset(
        impl_->right_previous_depth, 0, source_depth_bytes),
      "initialize right depth history", error))
  {
    impl_->release();
    return false;
  }

  if (
    !check_cuda(
      cudaMemcpy2D(
        impl_->left_base_mask,
        config.panorama_width,
        left_base_mask.data,
        left_base_mask.step,
        config.panorama_width,
        config.panorama_height,
        cudaMemcpyHostToDevice),
      "copy left base mask", error) ||
    !check_cuda(
      cudaMemcpy2D(
        impl_->right_base_mask,
        config.panorama_width,
        right_base_mask.data,
        right_base_mask.step,
        config.panorama_width,
        config.panorama_height,
        cudaMemcpyHostToDevice),
      "copy right base mask", error))
  {
    impl_->release();
    return false;
  }
  auto copy_map = [
    &config, &error](
    float * destination, const cv::Mat & source,
    const char * operation)
    {
      return check_cuda(
        cudaMemcpy2D(
          destination,
          static_cast<std::size_t>(config.panorama_width) *
          sizeof(float),
          source.data,
          source.step,
          static_cast<std::size_t>(config.panorama_width) *
          sizeof(float),
          config.panorama_height,
          cudaMemcpyHostToDevice),
        operation, error);
    };
  if (
    !copy_map(impl_->left_map_x, left_map_x, "copy left map x") ||
    !copy_map(impl_->left_map_y, left_map_y, "copy left map y") ||
    !copy_map(impl_->right_map_x, right_map_x, "copy right map x") ||
    !copy_map(impl_->right_map_y, right_map_y, "copy right map y"))
  {
    impl_->release();
    return false;
  }
  impl_->seam_cost_host.assign(
    static_cast<std::size_t>(seam_width * config.panorama_height),
    0.0F);
  impl_->seam_host.assign(
    static_cast<std::size_t>(config.panorama_height), config.seam_x);
  impl_->previous_seam_host = impl_->seam_host;
  if (
    !check_cuda(
      cudaMemcpy(
        impl_->seam_by_row,
        impl_->seam_host.data(),
        seam_row_bytes,
        cudaMemcpyHostToDevice),
      "initialize seam rows", error))
  {
    impl_->release();
    return false;
  }
  impl_->configured = true;
  return true;
}

bool CudaPanoramaBackend::process(
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
  const CudaProcessOptions & options,
  CudaPointCloudRequest * pointcloud)
{
  if (impl_->quarantined) {
    error = "CUDA backend is quarantined: " + impl_->quarantine_reason;
    return false;
  }
  if (!impl_->configured) {
    error = "CUDA backend is not configured";
    return false;
  }
  const auto & config = impl_->config;
  if (
    left_source_color.type() != CV_8UC3 ||
    right_source_color.type() != CV_8UC3)
  {
    error = "CUDA backend received an unsupported color cv::Mat type";
    return false;
  }
  const bool depth_is_raw_uint16 =
    left_depth.type() == CV_16UC1 && right_depth.type() == CV_16UC1;
  const bool depth_is_metres =
    left_depth.type() == CV_32FC1 && right_depth.type() == CV_32FC1;
  if (config.depth_aware_color && !depth_is_raw_uint16 && !depth_is_metres) {
    error = "CUDA backend received an unsupported depth cv::Mat type";
    return false;
  }
  const bool build_pointcloud =
    options.build_pointcloud && pointcloud != nullptr &&
    pointcloud->destination != nullptr &&
    config.pointcloud_stride > 0 &&
    impl_->pointcloud_capacity > 0;
  if (build_pointcloud) {
    if (pointcloud->capacity_points < impl_->pointcloud_capacity) {
      error = "point cloud destination is smaller than the sampled grid";
      return false;
    }
    if (
      pointcloud->point_step_bytes <= 0 ||
      pointcloud->point_step_bytes > kMaxPointCloudStride)
    {
      error = "unsupported point cloud point_step";
      return false;
    }
  }
  if (pointcloud != nullptr) {
    pointcloud->point_count = 0;
  }
  if (options.download_panorama) {
    panorama.create(
      config.panorama_height, config.panorama_width, CV_8UC3);
  }
  if (options.download_validity) {
    validity.create(
      config.panorama_height, config.panorama_width, CV_8UC1);
  }
  if (options.download_range) {
    range_m.create(
      config.panorama_height, config.panorama_width, CV_32FC1);
  }
  const std::size_t source_color_row_bytes =
    static_cast<std::size_t>(config.source_width) * 3U;
  const std::size_t source_depth_row_bytes =
    static_cast<std::size_t>(config.source_width) *
    (depth_is_raw_uint16 ? sizeof(unsigned short) : sizeof(float));
  const std::size_t panorama_color_row_bytes =
    static_cast<std::size_t>(config.panorama_width) * 3U;
  const std::size_t panorama_pixels =
    static_cast<std::size_t>(config.panorama_width) *
    static_cast<std::size_t>(config.panorama_height);
  const std::size_t projection_key_bytes =
    panorama_pixels * sizeof(unsigned long long);
  const std::size_t projection_range_bytes =
    panorama_pixels * sizeof(unsigned int);

  // Copy caller-owned inputs before submitting CUDA work. If the GPU later
  // times out, no in-flight DMA retains pointers into ROS/OpenCV messages.
  copy_rows_to_contiguous(
    impl_->left_source_host, left_source_color,
    source_color_row_bytes, config.source_height);
  copy_rows_to_contiguous(
    impl_->right_source_host, right_source_color,
    source_color_row_bytes, config.source_height);
  if (config.depth_aware_color) {
    copy_rows_to_contiguous(
      impl_->left_depth_host, left_depth,
      source_depth_row_bytes, config.source_height);
    copy_rows_to_contiguous(
      impl_->right_depth_host, right_depth,
      source_depth_row_bytes, config.source_height);
  }

  if (!check_cuda(
      cudaEventRecord(impl_->start_event, impl_->stream),
      "record CUDA panorama start", error))
  {
    return false;
  }
  auto copy_to_device = [
    this, &error](
    void * destination,
    std::size_t destination_pitch,
    const void * source,
    std::size_t source_pitch,
    std::size_t row_bytes,
    int rows,
    const char * operation)
    {
      return check_cuda(
        cudaMemcpy2DAsync(
          destination,
          destination_pitch,
          source,
          source_pitch,
          row_bytes,
          rows,
          cudaMemcpyHostToDevice,
          impl_->stream),
        operation,
        error);
    };
  if (
    !copy_to_device(
      impl_->left_source, source_color_row_bytes,
      impl_->left_source_host, source_color_row_bytes,
      source_color_row_bytes,
      config.source_height, "upload left source") ||
    !copy_to_device(
      impl_->right_source, source_color_row_bytes,
      impl_->right_source_host, source_color_row_bytes,
      source_color_row_bytes,
      config.source_height, "upload right source"))
  {
    return false;
  }
  if (config.depth_aware_color) {
    // Raw 16UC1 depth is uploaded as-is and scaled by a trivial kernel: half
    // the PCIe traffic of float metres and no CPU conversion at all.
    void * const left_destination = depth_is_raw_uint16 ?
      static_cast<void *>(impl_->left_raw_depth) :
      static_cast<void *>(impl_->left_depth);
    void * const right_destination = depth_is_raw_uint16 ?
      static_cast<void *>(impl_->right_raw_depth) :
      static_cast<void *>(impl_->right_depth);
    if (
      !copy_to_device(
        left_destination, source_depth_row_bytes,
        impl_->left_depth_host, source_depth_row_bytes,
        source_depth_row_bytes,
        config.source_height, "upload left depth") ||
      !copy_to_device(
        right_destination, source_depth_row_bytes,
        impl_->right_depth_host, source_depth_row_bytes,
        source_depth_row_bytes,
        config.source_height, "upload right depth"))
    {
      return false;
    }
    if (depth_is_raw_uint16) {
      constexpr int convert_threads = 256;
      const int source_pixel_count =
        config.source_width * config.source_height;
      const int convert_blocks =
        (source_pixel_count + convert_threads - 1) / convert_threads;
      convert_depth_kernel<<<
        convert_blocks, convert_threads, 0, impl_->stream>>>(
        impl_->left_raw_depth, impl_->left_depth,
        config.depth_scale_m, source_pixel_count);
      convert_depth_kernel<<<
        convert_blocks, convert_threads, 0, impl_->stream>>>(
        impl_->right_raw_depth, impl_->right_depth,
        config.depth_scale_m, source_pixel_count);
      if (!check_cuda(
          cudaPeekAtLastError(), "launch CUDA depth conversion", error))
      {
        return false;
      }
    }
  }

  if (
    !check_cuda(
      cudaMemsetAsync(
        impl_->left_projection_keys, 0xff,
        projection_key_bytes, impl_->stream),
      "clear left projection keys", error) ||
    !check_cuda(
      cudaMemsetAsync(
        impl_->right_projection_keys, 0xff,
        projection_key_bytes, impl_->stream),
      "clear right projection keys", error) ||
    !check_cuda(
      cudaMemsetAsync(
        impl_->left_minimum_ranges, 0xff,
        projection_range_bytes, impl_->stream),
      "clear left minimum ranges", error) ||
    !check_cuda(
      cudaMemsetAsync(
        impl_->right_minimum_ranges, 0xff,
        projection_range_bytes, impl_->stream),
      "clear right minimum ranges", error) ||
    !check_cuda(
      cudaMemsetAsync(
        impl_->left_accepted_points, 0,
        sizeof(unsigned int), impl_->stream),
      "clear left count", error) ||
    !check_cuda(
      cudaMemsetAsync(
        impl_->right_accepted_points, 0,
        sizeof(unsigned int), impl_->stream),
      "clear right count", error))
  {
    return false;
  }

  const float * left_projection_depth = impl_->left_depth;
  const float * right_projection_depth = impl_->right_depth;
  if (config.depth_aware_color && config.depth_spatial_filter) {
    constexpr int filter_threads = 256;
    const int source_pixel_count =
      config.source_width * config.source_height;
    const int filter_blocks =
      (source_pixel_count + filter_threads - 1) / filter_threads;
    edge_aware_depth_filter_kernel<<<
      filter_blocks, filter_threads, 0, impl_->stream>>>(
      impl_->left_depth, impl_->left_spatial_depth, config);
    edge_aware_depth_filter_kernel<<<
      filter_blocks, filter_threads, 0, impl_->stream>>>(
      impl_->right_depth, impl_->right_spatial_depth, config);
    left_projection_depth = impl_->left_spatial_depth;
    right_projection_depth = impl_->right_spatial_depth;
  }
  if (config.depth_aware_color && config.depth_temporal_filter) {
    constexpr int filter_threads = 256;
    const int source_pixel_count =
      config.source_width * config.source_height;
    const int filter_blocks =
      (source_pixel_count + filter_threads - 1) / filter_threads;
    temporal_depth_filter_kernel<<<
      filter_blocks, filter_threads, 0, impl_->stream>>>(
      left_projection_depth,
      impl_->left_previous_depth,
      impl_->left_filtered_depth,
      config);
    temporal_depth_filter_kernel<<<
      filter_blocks, filter_threads, 0, impl_->stream>>>(
      right_projection_depth,
      impl_->right_previous_depth,
      impl_->right_filtered_depth,
      config);
    left_projection_depth = impl_->left_filtered_depth;
    right_projection_depth = impl_->right_filtered_depth;
  }
  if (!check_cuda(
      cudaPeekAtLastError(), "launch CUDA depth filters", error))
  {
    return false;
  }

  if (config.depth_aware_color) {
    constexpr int projection_threads = 256;
    const int left_work_items =
      (impl_->left_camera.maximum_depth_column -
      impl_->left_camera.minimum_depth_column + 1) *
      config.source_height;
    const int right_work_items =
      (impl_->right_camera.maximum_depth_column -
      impl_->right_camera.minimum_depth_column + 1) *
      config.source_height;
    project_min_range_kernel<<<
      (left_work_items + projection_threads - 1) / projection_threads,
      projection_threads, 0, impl_->stream>>>(
      left_projection_depth,
      impl_->left_minimum_ranges,
      impl_->left_camera,
      config);
    project_min_range_kernel<<<
      (right_work_items + projection_threads - 1) / projection_threads,
      projection_threads, 0, impl_->stream>>>(
      right_projection_depth,
      impl_->right_minimum_ranges,
      impl_->right_camera,
      config);
    project_depth_kernel<<<
      (left_work_items + projection_threads - 1) / projection_threads,
      projection_threads, 0, impl_->stream>>>(
      left_projection_depth,
      impl_->left_minimum_ranges,
      impl_->left_projection_keys,
      impl_->left_accepted_points,
      impl_->left_camera,
      config);
    project_depth_kernel<<<
      (right_work_items + projection_threads - 1) / projection_threads,
      projection_threads, 0, impl_->stream>>>(
      right_projection_depth,
      impl_->right_minimum_ranges,
      impl_->right_projection_keys,
      impl_->right_accepted_points,
      impl_->right_camera,
      config);
  }
  if (!check_cuda(
      cudaPeekAtLastError(), "launch CUDA depth projection", error))
  {
    return false;
  }

  const dim3 compose_threads(16, 16);
  const dim3 compose_blocks(
    (config.panorama_width + compose_threads.x - 1) /
    compose_threads.x,
    (config.panorama_height + compose_threads.y - 1) /
    compose_threads.y);
  remap_color_kernel<<<
    compose_blocks, compose_threads, 0, impl_->stream>>>(
    impl_->left_source,
    impl_->left_map_x,
    impl_->left_map_y,
    impl_->left_base_mask,
    impl_->left_base,
    config);
  remap_color_kernel<<<
    compose_blocks, compose_threads, 0, impl_->stream>>>(
    impl_->right_source,
    impl_->right_map_x,
    impl_->right_map_y,
    impl_->right_base_mask,
    impl_->right_base,
    config);
  if (!check_cuda(
      cudaPeekAtLastError(), "launch CUDA color remap", error))
  {
    return false;
  }

  const int seam_width =
    config.depth_color_max_x - config.depth_color_min_x + 1;
  const bool use_content_aware_seam =
    config.content_aware_seam &&
    config.prefer_seam_camera_when_both_depth_valid &&
    seam_width > 0;
  if (use_content_aware_seam) {
    const dim3 seam_blocks(
      (seam_width + compose_threads.x - 1) / compose_threads.x,
      (config.panorama_height + compose_threads.y - 1) /
      compose_threads.y);
    seam_cost_kernel<<<
      seam_blocks, compose_threads, 0, impl_->stream>>>(
      impl_->left_source,
      impl_->right_source,
      impl_->left_base,
      impl_->right_base,
      impl_->left_base_mask,
      impl_->right_base_mask,
      impl_->left_projection_keys,
      impl_->right_projection_keys,
      impl_->seam_cost,
      static_cast<float>(right_gain_bgr[0]),
      static_cast<float>(right_gain_bgr[1]),
      static_cast<float>(right_gain_bgr[2]),
      config);
    if (
      !check_cuda(
        cudaGetLastError(), "launch content-aware seam cost", error) ||
      !check_cuda(
        cudaMemcpy2DAsync(
          impl_->seam_cost_staging,
          static_cast<std::size_t>(seam_width) * sizeof(float),
          impl_->seam_cost,
          static_cast<std::size_t>(seam_width) * sizeof(float),
          static_cast<std::size_t>(seam_width) * sizeof(float),
          config.panorama_height,
          cudaMemcpyDeviceToHost,
          impl_->stream),
        "download seam cost", error) ||
      !check_cuda(
        cudaEventRecord(impl_->stop_event, impl_->stream),
        "record seam cost stop", error) ||
      !wait_for_cuda_event(
        impl_->stop_event, config.operation_timeout_ms,
        "CUDA seam cost", error))
    {
      return false;
    }
    std::memcpy(
      impl_->seam_cost_host.data(), impl_->seam_cost_staging,
      static_cast<std::size_t>(seam_width) *
      static_cast<std::size_t>(config.panorama_height) * sizeof(float));
    impl_->seam_host = find_content_aware_seam(
      impl_->seam_cost_host,
      seam_width,
      config.panorama_height,
      config.depth_color_min_x,
      config,
      impl_->previous_seam_host);
    impl_->previous_seam_host = impl_->seam_host;
    if (
      !check_cuda(
        cudaMemcpyAsync(
          impl_->seam_by_row,
          impl_->seam_host.data(),
          static_cast<std::size_t>(config.panorama_height) * sizeof(int),
          cudaMemcpyHostToDevice,
          impl_->stream),
        "upload content-aware seam", error))
    {
      return false;
    }
  } else {
    std::fill(
      impl_->seam_host.begin(), impl_->seam_host.end(), config.seam_x);
  }
  compose_panorama_kernel<<<
    compose_blocks, compose_threads, 0, impl_->stream>>>(
    impl_->left_source,
    impl_->right_source,
    impl_->left_base,
    impl_->right_base,
    impl_->left_base_mask,
    impl_->right_base_mask,
    impl_->left_projection_keys,
    impl_->right_projection_keys,
    impl_->seam_by_row,
    impl_->output,
    impl_->validity,
    impl_->output_range_m,
    static_cast<float>(right_gain_bgr[0]),
    static_cast<float>(right_gain_bgr[1]),
    static_cast<float>(right_gain_bgr[2]),
    config);
  if (!check_cuda(
      cudaGetLastError(), "launch CUDA panorama kernels", error))
  {
    return false;
  }

  if (build_pointcloud) {
    const int stride = config.pointcloud_stride;
    const int sampled_columns =
      (config.panorama_width + stride - 1) / stride;
    const int sampled_rows =
      (config.panorama_height + stride - 1) / stride;
    const dim3 cloud_blocks(
      (sampled_columns + compose_threads.x - 1) / compose_threads.x,
      (sampled_rows + compose_threads.y - 1) / compose_threads.y);
    CudaPointCloudLayout layout;
    layout.point_step = pointcloud->point_step_bytes;
    layout.x_offset = pointcloud->x_offset_bytes;
    layout.y_offset = pointcloud->y_offset_bytes;
    layout.z_offset = pointcloud->z_offset_bytes;
    layout.rgb_offset = pointcloud->rgb_offset_bytes;
    // Zero the whole buffer so the padding bytes between fields, and the tail
    // slots past point_count, never carry the previous frame's contents.
    if (
      !check_cuda(
        cudaMemsetAsync(
          impl_->pointcloud_count, 0, sizeof(unsigned int), impl_->stream),
        "clear point cloud count", error) ||
      !check_cuda(
        cudaMemsetAsync(
          impl_->pointcloud, 0,
          static_cast<std::size_t>(impl_->pointcloud_capacity) *
          static_cast<std::size_t>(layout.point_step),
          impl_->stream),
        "clear point cloud buffer", error))
    {
      return false;
    }
    build_pointcloud_kernel<<<
      cloud_blocks, compose_threads, 0, impl_->stream>>>(
      impl_->output,
      impl_->validity,
      impl_->output_range_m,
      impl_->pointcloud,
      impl_->pointcloud_count,
      impl_->pointcloud_capacity,
      layout,
      config);
    if (!check_cuda(
        cudaGetLastError(), "launch CUDA point cloud builder", error))
    {
      return false;
    }
  }

  if (
    options.download_panorama &&
    !check_cuda(
      cudaMemcpy2DAsync(
        impl_->output_host,
        panorama_color_row_bytes,
        impl_->output,
        panorama_color_row_bytes,
        panorama_color_row_bytes,
        config.panorama_height,
        cudaMemcpyDeviceToHost,
        impl_->stream),
      "download panorama", error))
  {
    return false;
  }
  if (
    options.download_validity &&
    !check_cuda(
      cudaMemcpy2DAsync(
        impl_->validity_host,
        static_cast<std::size_t>(config.panorama_width),
        impl_->validity,
        static_cast<std::size_t>(config.panorama_width),
        static_cast<std::size_t>(config.panorama_width),
        config.panorama_height,
        cudaMemcpyDeviceToHost,
        impl_->stream),
      "download validity", error))
  {
    return false;
  }
  if (
    options.download_range &&
    !check_cuda(
      cudaMemcpy2DAsync(
        impl_->output_range_host,
        static_cast<std::size_t>(config.panorama_width) * sizeof(float),
        impl_->output_range_m,
        static_cast<std::size_t>(config.panorama_width) * sizeof(float),
        static_cast<std::size_t>(config.panorama_width) * sizeof(float),
        config.panorama_height,
        cudaMemcpyDeviceToHost,
        impl_->stream),
      "download range", error))
  {
    return false;
  }
  if (build_pointcloud) {
    // The compacted points sit at the front of the buffer, so downloading the
    // whole sampled grid once avoids a second synchronization just to learn
    // the exact count.
    if (
      !check_cuda(
        cudaMemcpyAsync(
          impl_->pointcloud_host,
          impl_->pointcloud,
          static_cast<std::size_t>(impl_->pointcloud_capacity) *
          static_cast<std::size_t>(pointcloud->point_step_bytes),
          cudaMemcpyDeviceToHost,
          impl_->stream),
        "download point cloud", error) ||
      !check_cuda(
        cudaMemcpyAsync(
          impl_->pointcloud_count_host,
          impl_->pointcloud_count,
          sizeof(unsigned int),
          cudaMemcpyDeviceToHost,
          impl_->stream),
        "download point cloud count", error))
    {
      return false;
    }
  }
  if (
    !check_cuda(
      cudaMemcpyAsync(
        impl_->left_count_host,
        impl_->left_accepted_points,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost,
        impl_->stream),
      "download left count", error) ||
    !check_cuda(
      cudaMemcpyAsync(
        impl_->right_count_host,
        impl_->right_accepted_points,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost,
        impl_->stream),
      "download right count", error))
  {
    return false;
  }
  if (
    !check_cuda(
      cudaEventRecord(impl_->stop_event, impl_->stream),
      "record CUDA panorama stop", error) ||
    !wait_for_cuda_event(
      impl_->stop_event, config.operation_timeout_ms,
      "CUDA panorama", error))
  {
    return false;
  }
  if (options.download_panorama) {
    copy_contiguous_to_rows(
      panorama, impl_->output_host,
      panorama_color_row_bytes, config.panorama_height);
  }
  if (options.download_validity) {
    copy_contiguous_to_rows(
      validity, impl_->validity_host,
      static_cast<std::size_t>(config.panorama_width),
      config.panorama_height);
  }
  if (options.download_range) {
    copy_contiguous_to_rows(
      range_m,
      reinterpret_cast<const unsigned char *>(impl_->output_range_host),
      static_cast<std::size_t>(config.panorama_width) * sizeof(float),
      config.panorama_height);
  }
  const unsigned int left_count = *impl_->left_count_host;
  const unsigned int right_count = *impl_->right_count_host;
  if (build_pointcloud) {
    const std::size_t cloud_count = std::min(
      static_cast<std::size_t>(*impl_->pointcloud_count_host),
      static_cast<std::size_t>(impl_->pointcloud_capacity));
    std::memcpy(
      pointcloud->destination, impl_->pointcloud_host,
      cloud_count * static_cast<std::size_t>(pointcloud->point_step_bytes));
    pointcloud->point_count = cloud_count;
  }
  float elapsed_ms = 0.0F;
  if (!check_cuda(
      cudaEventElapsedTime(
        &elapsed_ms, impl_->start_event, impl_->stop_event),
      "measure CUDA panorama time", error))
  {
    return false;
  }
  stats.left_depth_points = left_count;
  stats.right_depth_points = right_count;
  stats.gpu_time_ms = elapsed_ms;
  stats.content_aware_seam_used = use_content_aware_seam;
  const auto seam_bounds = std::minmax_element(
    impl_->seam_host.begin(), impl_->seam_host.end());
  stats.seam_min_x = *seam_bounds.first;
  stats.seam_max_x = *seam_bounds.second;
  long long seam_sum = 0;
  for (const int seam_x : impl_->seam_host) {
    seam_sum += seam_x;
  }
  stats.seam_mean_x = impl_->seam_host.empty() ?
    static_cast<float>(config.seam_x) :
    static_cast<float>(seam_sum) /
    static_cast<float>(impl_->seam_host.size());
  return true;
}

}  // namespace panorama_stitcher
