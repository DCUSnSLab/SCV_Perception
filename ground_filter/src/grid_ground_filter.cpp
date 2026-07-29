// Derived from autoware_universe perception/autoware_ground_segmentation
// (scan_ground_filter). Autoware plumbing removed; classification logic intact.

#include "ground_filter/grid_ground_filter.hpp"

#include "ground_filter/data.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace ground_filter
{

void GridGroundFilter::convert()
{
  const size_t in_cloud_data_size = in_cloud_->data.size();
  const size_t in_cloud_point_step = in_cloud_->point_step;

  for (size_t data_index = 0; data_index + in_cloud_point_step <= in_cloud_data_size;
       data_index += in_cloud_point_step) {
    PointXYZ input_point;
    data_accessor_.getPoint(in_cloud_, data_index, input_point);
    grid_ptr_->addPoint(input_point.x, input_point.y, input_point.z, data_index);
  }
}

void GridGroundFilter::preprocess()
{
  grid_ptr_->setGridConnections();
}

bool GridGroundFilter::recursiveSearch(
  const int check_idx, const int search_cnt, std::vector<int> & idx) const
{
  constexpr size_t count_limit = 1023;
  return recursiveSearch(check_idx, search_cnt, idx, count_limit);
}

bool GridGroundFilter::recursiveSearch(
  const int check_idx, const int search_cnt, std::vector<int> & idx, size_t count) const
{
  if (count == 0) {
    return false;
  }
  count -= 1;
  if (check_idx < 0) {
    return false;
  }
  if (search_cnt == 0) {
    return true;
  }
  const auto & check_cell = grid_ptr_->getCell(check_idx);
  if (check_cell.has_ground_) {
    idx.push_back(check_idx);
    return recursiveSearch(check_cell.scan_grid_root_idx_, search_cnt - 1, idx, count);
  }
  return recursiveSearch(check_cell.scan_grid_root_idx_, search_cnt, idx, count);
}

void GridGroundFilter::fitLineFromGndGrid(const std::vector<int> & idx, float & a, float & b) const
{
  if (idx.empty()) {
    a = 0.0f;
    b = 0.0f;
    return;
  }
  if (idx.size() == 1) {
    const auto & cell = grid_ptr_->getCell(idx.front());
    a = cell.avg_height_ / cell.avg_radius_;
    b = 0.0f;
    return;
  }
  float sum_x = 0.0f;
  float sum_y = 0.0f;
  float sum_xy = 0.0f;
  float sum_x2 = 0.0f;
  for (const auto & i : idx) {
    const auto & cell = grid_ptr_->getCell(i);
    sum_x += cell.avg_radius_;
    sum_y += cell.avg_height_;
    sum_xy += cell.avg_radius_ * cell.avg_height_;
    sum_x2 += cell.avg_radius_ * cell.avg_radius_;
  }
  const float n = static_cast<float>(idx.size());
  const float denominator = n * sum_x2 - sum_x * sum_x;
  if (denominator != 0.0f) {
    a = (n * sum_xy - sum_x * sum_y) / denominator;
    a = std::clamp(a, -param_.global_slope_max_ratio, param_.global_slope_max_ratio);
    b = (sum_y - a * sum_x) / n;
  } else {
    const auto & cell = grid_ptr_->getCell(idx.front());
    a = cell.avg_height_ / cell.avg_radius_;
    b = 0.0f;
  }
}

namespace
{
void collectGroundPoints(const PointsCentroid & ground_bin, PointIndices & out_ground_indices)
{
  const auto & indices = ground_bin.getIndicesRef();
  for (size_t i = 0; i < indices.size(); ++i) {
    if (ground_bin.is_ground_list[i]) {
      out_ground_indices.push_back(indices[i]);
    }
  }
}
}  // namespace

void GridGroundFilter::initializeGround(
  PointIndices & out_no_ground_indices, PointIndices & out_ground_indices)
{
  const auto grid_size = grid_ptr_->getGridSize();
  for (size_t idx = 0; idx < grid_size; idx++) {
    auto & cell = grid_ptr_->getCell(idx);
    if (cell.is_ground_initialized_) continue;
    if (cell.isEmpty()) continue;

    if (cell.scan_grid_root_idx_ >= 0) {
      const Cell & prev_cell = grid_ptr_->getCell(cell.scan_grid_root_idx_);
      if (prev_cell.is_ground_initialized_) {
        cell.is_ground_initialized_ = true;
        continue;
      }
    }

    bool is_ground_found = false;
    PointsCentroid ground_bin;

    for (const auto & pt : cell.point_list_) {
      const size_t & pt_idx = pt.index;
      const float & radius = pt.distance;
      const float & height = pt.height;

      const float global_slope_threshold = param_.global_slope_max_ratio * radius;
      if (height >= global_slope_threshold && height > param_.non_ground_height_threshold) {
        out_no_ground_indices.push_back(pt_idx);
      } else if (
        std::abs(height) < global_slope_threshold &&
        std::abs(height) < param_.non_ground_height_threshold) {
        ground_bin.addPoint(radius, height, pt_idx);
        is_ground_found = true;
      }
    }
    cell.is_processed_ = true;
    cell.has_ground_ = is_ground_found;
    if (is_ground_found) {
      cell.is_ground_initialized_ = true;
      collectGroundPoints(ground_bin, out_ground_indices);
      ground_bin.processAverage();
      cell.avg_height_ = ground_bin.getAverageHeight();
      cell.avg_radius_ = ground_bin.getAverageRadius();
      cell.max_height_ = ground_bin.getMaxHeight();
      cell.min_height_ = ground_bin.getMinHeight();
      cell.gradient_ = std::clamp(
        cell.avg_height_ / cell.avg_radius_, -param_.global_slope_max_ratio,
        param_.global_slope_max_ratio);
      cell.intercept_ = 0.0f;
    } else {
      cell.is_ground_initialized_ = false;
    }
  }
}

void GridGroundFilter::SegmentContinuousCell(
  const Cell & cell, PointsCentroid & ground_bin, PointIndices & out_no_ground_indices)
{
  const Cell & prev_cell = grid_ptr_->getCell(cell.scan_grid_root_idx_);
  static const float local_thresh_angle_ratio = std::tan(5.0f * static_cast<float>(M_PI) / 180.0f);

  for (const auto & pt : cell.point_list_) {
    const size_t & pt_idx = pt.index;
    const float & radius = pt.distance;
    const float & height = pt.height;

    const float delta_z = height - prev_cell.avg_height_;
    if (delta_z > param_.detection_range_z_max) {
      continue;
    }

    if (height > param_.global_slope_max_ratio * radius) {
      out_no_ground_indices.push_back(pt_idx);
      continue;
    }

    const float delta_radius = radius - prev_cell.avg_radius_;
    if (std::abs(delta_z) < param_.local_slope_max_ratio * delta_radius) {
      ground_bin.addPoint(radius, height, pt_idx);
      continue;
    }

    const float next_gnd_z = cell.gradient_ * radius + cell.intercept_;
    const float gnd_z_local_thresh = local_thresh_angle_ratio * delta_radius;
    const float delta_gnd_z = height - next_gnd_z;
    const float gnd_z_threshold = param_.non_ground_height_threshold + gnd_z_local_thresh;
    if (delta_gnd_z > gnd_z_threshold) {
      out_no_ground_indices.push_back(pt_idx);
      continue;
    }
    if (std::abs(delta_gnd_z) <= gnd_z_threshold) {
      ground_bin.addPoint(radius, height, pt_idx);
      continue;
    }
  }
}

void GridGroundFilter::SegmentDiscontinuousCell(
  const Cell & cell, PointsCentroid & ground_bin, PointIndices & out_no_ground_indices)
{
  const Cell & prev_cell = grid_ptr_->getCell(cell.scan_grid_root_idx_);

  for (const auto & pt : cell.point_list_) {
    const size_t & pt_idx = pt.index;
    const float & radius = pt.distance;
    const float & height = pt.height;

    const float delta_avg_z = height - prev_cell.avg_height_;
    if (delta_avg_z > param_.detection_range_z_max) {
      continue;
    }

    if (height > param_.global_slope_max_ratio * radius) {
      out_no_ground_indices.push_back(pt_idx);
      continue;
    }
    const float delta_radius = radius - prev_cell.avg_radius_;
    const float local_slope_threshold = param_.local_slope_max_ratio * delta_radius;
    if (std::abs(delta_avg_z) < local_slope_threshold) {
      ground_bin.addPoint(radius, height, pt_idx);
      continue;
    }
    if (std::abs(delta_avg_z) < param_.non_ground_height_threshold) {
      ground_bin.addPoint(radius, height, pt_idx);
      continue;
    }
    const float delta_max_z = height - prev_cell.max_height_;
    if (std::abs(delta_max_z) < param_.non_ground_height_threshold) {
      ground_bin.addPoint(radius, height, pt_idx);
      continue;
    }
    if (delta_avg_z >= local_slope_threshold) {
      out_no_ground_indices.push_back(pt_idx);
      continue;
    }
  }
}

void GridGroundFilter::SegmentBreakCell(
  const Cell & cell, PointsCentroid & ground_bin, PointIndices & out_no_ground_indices)
{
  const Cell & prev_cell = grid_ptr_->getCell(cell.scan_grid_root_idx_);

  for (const auto & pt : cell.point_list_) {
    const size_t & pt_idx = pt.index;
    const float & radius = pt.distance;
    const float & height = pt.height;

    const float delta_z = height - prev_cell.avg_height_;
    if (delta_z > param_.detection_range_z_max) {
      continue;
    }

    if (height > param_.global_slope_max_ratio * radius) {
      out_no_ground_indices.push_back(pt_idx);
      continue;
    }

    const float delta_radius = radius - prev_cell.avg_radius_;
    const float global_slope_threshold = param_.global_slope_max_ratio * delta_radius;
    if (std::abs(delta_z) < global_slope_threshold) {
      ground_bin.addPoint(radius, height, pt_idx);
      continue;
    }
    if (delta_z >= global_slope_threshold) {
      out_no_ground_indices.push_back(pt_idx);
      continue;
    }
  }
}

void GridGroundFilter::classify(
  PointIndices & out_no_ground_indices, PointIndices & out_ground_indices)
{
  const auto grid_size = grid_ptr_->getGridSize();
  for (size_t idx = 0; idx < grid_size; idx++) {
    auto & cell = grid_ptr_->getCell(idx);
    if (cell.isEmpty()) continue;
    if (cell.is_processed_) continue;

    if (cell.scan_grid_root_idx_ < 0) continue;
    const Cell & prev_cell = grid_ptr_->getCell(cell.scan_grid_root_idx_);
    if (!(prev_cell.is_ground_initialized_)) continue;

    std::vector<int> grid_idcs;
    {
      const int search_count = param_.gnd_grid_buffer_size;
      const int check_cell_idx = cell.scan_grid_root_idx_;
      recursiveSearch(check_cell_idx, search_count, grid_idcs);
    }
    if (grid_idcs.empty()) continue;  // added: upstream dereferences .back() unchecked

    enum SegmentationMode { NONE, CONTINUOUS, DISCONTINUOUS, BREAK };
    SegmentationMode mode = SegmentationMode::NONE;
    {
      const int front_radial_id =
        grid_ptr_->getCell(grid_idcs.back()).radial_idx_ + grid_idcs.size();
      const float radial_diff_between_cells = cell.center_radius_ - prev_cell.center_radius_;

      if (radial_diff_between_cells < param_.gnd_grid_continual_thresh * cell.radial_size_) {
        if (cell.radial_idx_ - front_radial_id < param_.gnd_grid_continual_thresh) {
          mode = SegmentationMode::CONTINUOUS;
        } else {
          mode = SegmentationMode::DISCONTINUOUS;
        }
      } else {
        mode = SegmentationMode::BREAK;
      }
    }

    {
      PointsCentroid ground_bin;
      if (mode == SegmentationMode::CONTINUOUS) {
        float a, b;
        fitLineFromGndGrid(grid_idcs, a, b);
        cell.gradient_ = a;
        cell.intercept_ = b;

        SegmentContinuousCell(cell, ground_bin, out_no_ground_indices);
      } else if (mode == SegmentationMode::DISCONTINUOUS) {
        SegmentDiscontinuousCell(cell, ground_bin, out_no_ground_indices);
      } else if (mode == SegmentationMode::BREAK) {
        SegmentBreakCell(cell, ground_bin, out_no_ground_indices);
      }

      if (
        param_.use_recheck_ground_cluster && cell.avg_radius_ > param_.recheck_start_distance &&
        ground_bin.getGroundPointNum() > 0) {
        float reference_height = 0;
        if (param_.use_lowest_point) {
          reference_height = ground_bin.getMinHeightOnly();
        } else {
          ground_bin.processAverage();
          reference_height = ground_bin.getAverageHeight();
        }
        const float threshold = reference_height + param_.non_ground_height_threshold;
        const std::vector<size_t> & gnd_indices = ground_bin.getIndicesRef();
        const std::vector<float> & height_list = ground_bin.getHeightListRef();
        for (size_t j = 0; j < height_list.size(); ++j) {
          if (height_list.at(j) >= threshold) {
            out_no_ground_indices.push_back(gnd_indices.at(j));
            ground_bin.is_ground_list.at(j) = false;
          }
        }
      }

      collectGroundPoints(ground_bin, out_ground_indices);

      if (ground_bin.getGroundPointNum() > 0) {
        ground_bin.processAverage();
        cell.avg_height_ = ground_bin.getAverageHeight();
        cell.avg_radius_ = ground_bin.getAverageRadius();
        cell.max_height_ = ground_bin.getMaxHeight();
        cell.min_height_ = ground_bin.getMinHeight();
        cell.has_ground_ = true;
      } else {
        cell.avg_radius_ = prev_cell.avg_radius_;
        cell.avg_height_ = prev_cell.avg_height_;
        cell.max_height_ = prev_cell.max_height_;
        cell.min_height_ = prev_cell.min_height_;
        cell.has_ground_ = false;
      }

      cell.is_processed_ = true;
    }
  }
}

void GridGroundFilter::process(
  const PointCloud2ConstPtr & in_cloud, PointIndices & out_no_ground_indices,
  PointIndices & out_ground_indices)
{
  in_cloud_ = in_cloud;

  out_no_ground_indices.clear();
  out_ground_indices.clear();

  grid_ptr_->resetCells();

  convert();

  preprocess();

  initializeGround(out_no_ground_indices, out_ground_indices);

  classify(out_no_ground_indices, out_ground_indices);
}

}  // namespace ground_filter
