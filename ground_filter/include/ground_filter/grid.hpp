// Derived from autoware_universe perception/autoware_ground_segmentation
// (scan_ground_filter). Autoware plumbing removed; classification logic intact.

#ifndef GROUND_FILTER__GRID_HPP_
#define GROUND_FILTER__GRID_HPP_

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f 1.57079632679489661923f
#endif
#ifndef M_PI_4f
#define M_PI_4f 0.78539816339744830962f
#endif

namespace
{

float pseudoArcTan2(const float y, const float x)
{

  if (y == 0.0f) {
    if (x >= 0.0f) return 0.0f;
    return M_PIf;
  }
  if (x == 0.0f) {
    if (y >= 0.0f) return M_PI_2f;
    return -M_PI_2f;
  }

  const float x_abs = std::abs(x);
  const float y_abs = std::abs(y);

  constexpr float M_2PIf = 2.0f * M_PIf;
  constexpr float M_3PI_2f = 3.0f * M_PI_2f;
  if (x_abs > y_abs) {
    const float ratio = y_abs / x_abs;
    const float angle = ratio * M_PI_4f;
    if (y >= 0.0f) {
      if (x >= 0.0f) return angle;  // 1st zone
      return M_PIf - angle;         // 2nd zone
    } else {
      if (x >= 0.0f) return M_2PIf - angle;  // 4th zone
      return M_PIf + angle;                  // 3rd zone
    }
  } else {
    const float ratio = x_abs / y_abs;
    const float angle = ratio * M_PI_4f;
    if (y >= 0.0f) {
      if (x >= 0.0f) return M_PI_2f - angle;  // 1st zone
      return M_PI_2f + angle;                 // 2nd zone
    } else {
      if (x >= 0.0f) return M_3PI_2f + angle;  // 4th zone
      return M_3PI_2f - angle;                 // 3rd zone
    }
  }
}

}  // namespace

namespace ground_filter
{

struct Point
{
  size_t index;
  float distance;
  float height;
};

class Cell
{
public:
  std::vector<Point> point_list_;  // point index and distance

  inline bool isEmpty() const { return point_list_.empty(); }

  int grid_idx_;
  int radial_idx_;
  int azimuth_idx_;
  int next_grid_idx_;
  int prev_grid_idx_;

  int scan_grid_root_idx_;

  float center_radius_;
  float center_azimuth_;
  float radial_size_;
  float azimuth_size_;

  float avg_height_;
  float max_height_;
  float min_height_;
  float avg_radius_;
  float gradient_;
  float intercept_;

  bool is_processed_ = false;
  bool is_ground_initialized_ = false;
  bool has_ground_ = false;
};

class Grid
{
public:
  Grid(const float origin_x, const float origin_y) : origin_x_(origin_x), origin_y_(origin_y) {}
  ~Grid() = default;

  void initialize(
    const float grid_dist_size, const float grid_azimuth_size, const float grid_radial_limit)
  {
    grid_dist_size_ = grid_dist_size;
    grid_azimuth_size_ = grid_azimuth_size;

    grid_radial_limit_ = grid_radial_limit;
    grid_radial_max_num_ = std::ceil(grid_radial_limit / grid_dist_size_);
    grid_dist_size_inv_ = 1.0f / grid_dist_size_;

    setGridBoundaries();

    cells_.clear();
    cells_.resize(radial_idx_offsets_.back() + azimuth_grids_per_radial_.back());

    setCellGeometry();

    is_initialized_ = true;
  }

  void addPoint(const float x, const float y, const float z, const size_t point_idx)
  {
    const float x_fixed = x - origin_x_;
    const float y_fixed = y - origin_y_;
    const float radius = std::sqrt(x_fixed * x_fixed + y_fixed * y_fixed);
    const float azimuth = pseudoArcTan2(y_fixed, x_fixed);

    const int grid_idx = getGridIdx(radius, azimuth);

    if (grid_idx < 0) {
      return;
    }
    const size_t grid_idx_idx = static_cast<size_t>(grid_idx);

    cells_[grid_idx_idx].point_list_.emplace_back(Point{point_idx, radius, z});
  }

  size_t getGridSize() const { return cells_.size(); }

  inline Cell & getCell(const int grid_idx)
  {
    const size_t idx = static_cast<size_t>(grid_idx);
    if (idx >= cells_.size()) {
      throw std::out_of_range("Invalid grid index");
    }
    return cells_[idx];
  }

  void resetCells()
  {
    for (auto & cell : cells_) {
      cell.point_list_.clear();
      cell.is_processed_ = false;
      cell.is_ground_initialized_ = false;
      cell.has_ground_ = false;
    }
  }

  void setGridConnections()
  {
    for (Cell & cell : cells_) {
      cell.scan_grid_root_idx_ = cell.prev_grid_idx_;
      while (cell.scan_grid_root_idx_ >= 0) {
        const auto & prev_cell = cells_[cell.scan_grid_root_idx_];
        if (!prev_cell.isEmpty()) break;
        cell.scan_grid_root_idx_ = prev_cell.scan_grid_root_idx_;
      }
    }
  }

private:
  float origin_x_;
  float origin_y_;
  float grid_dist_size_ = 1.0f;      // meters
  float grid_azimuth_size_ = 0.01f;  // radians

  float grid_dist_size_inv_ = 0.0f;  // inverse of the grid size in meters
  bool is_initialized_ = false;

  float grid_radial_limit_ = 200.0f;  // meters
  int grid_radial_max_num_ = 0;

  std::vector<float> grid_radial_boundaries_;
  std::vector<int> azimuth_grids_per_radial_;
  std::vector<float> azimuth_interval_per_radial_;
  std::vector<int> radial_idx_offsets_;

  std::vector<Cell> cells_;

  void setGridBoundaries()
  {
    {
      for (int i = 0; i < grid_radial_max_num_; i++) {
        grid_radial_boundaries_.push_back(i * grid_dist_size_);
      }
    }

    const size_t radial_grid_num = grid_radial_boundaries_.size();

    {
      if (grid_azimuth_size_ <= 0) {
        throw std::runtime_error("Grid azimuth size is not positive.");
      }

      azimuth_grids_per_radial_.resize(radial_grid_num);
      azimuth_interval_per_radial_.resize(radial_grid_num);
      azimuth_grids_per_radial_[0] = 1;
      azimuth_interval_per_radial_[0] = 2.0f * M_PIf;

      const int max_azimuth_grid_num = static_cast<int>(2.0 * M_PIf / grid_azimuth_size_);

      int divider = 1;
      for (size_t i = radial_grid_num - 1; i > 0; --i) {
        const int grid_num = static_cast<int>(max_azimuth_grid_num / divider);
        const int azimuth_grid_num = std::max(std::min(grid_num, max_azimuth_grid_num), 1);
        const float azimuth_interval_evened = 2.0f * M_PIf / azimuth_grid_num;

        azimuth_grids_per_radial_[i] = azimuth_grid_num;
        azimuth_interval_per_radial_[i] = azimuth_interval_evened;
      }
    }

    radial_idx_offsets_.resize(radial_grid_num);
    radial_idx_offsets_[0] = 0;
    for (size_t i = 1; i < radial_grid_num; ++i) {
      radial_idx_offsets_[i] = radial_idx_offsets_[i - 1] + azimuth_grids_per_radial_[i - 1];
    }
  }

  int getAzimuthGridIdx(const int & radial_idx, const float & azimuth) const
  {
    const int azimuth_grid_num = azimuth_grids_per_radial_[radial_idx];

    int azimuth_grid_idx =
      static_cast<int>(std::floor(azimuth / azimuth_interval_per_radial_[radial_idx]));
    if (azimuth_grid_idx == azimuth_grid_num) {
      azimuth_grid_idx = 0;
    }
    return azimuth_grid_idx;
  }

  int getRadialIdx(const float & radius) const
  {
    if (radius > grid_radial_limit_) {
      return -1;
    }
    if (radius < 0) {
      return -1;
    }
    return static_cast<int>(radius * grid_dist_size_inv_);
  }

  int getGridIdx(const int & radial_idx, const int & azimuth_idx) const
  {
    return radial_idx_offsets_[radial_idx] + azimuth_idx;
  }

  int getGridIdx(const float & radius, const float & azimuth) const
  {
    const int grid_rad_idx = getRadialIdx(radius);
    if (grid_rad_idx < 0) {
      return -1;
    }

    const int grid_az_idx = getAzimuthGridIdx(grid_rad_idx, azimuth);
    if (grid_az_idx < 0) {
      return -1;
    }

    return getGridIdx(grid_rad_idx, grid_az_idx);
  }

  void getRadialAzimuthIdxFromCellIdx(const int cell_id, int & radial_idx, int & azimuth_idx) const
  {
    radial_idx = -1;
    azimuth_idx = -1;
    for (size_t i = 0; i < radial_idx_offsets_.size(); ++i) {
      if (cell_id < radial_idx_offsets_[i]) {
        radial_idx = i - 1;
        azimuth_idx = cell_id - radial_idx_offsets_[i - 1];
        break;
      }
    }
    if (cell_id >= radial_idx_offsets_.back()) {
      radial_idx = radial_idx_offsets_.size() - 1;
      azimuth_idx = cell_id - radial_idx_offsets_.back();
    }
  }

  void setCellGeometry()
  {
    for (size_t idx = 0; idx < cells_.size(); ++idx) {
      Cell & cell = cells_[idx];

      int radial_idx = 0;
      int azimuth_idx = 0;
      getRadialAzimuthIdxFromCellIdx(idx, radial_idx, azimuth_idx);

      cell.grid_idx_ = idx;
      cell.radial_idx_ = radial_idx;
      cell.azimuth_idx_ = azimuth_idx;

      const auto radial_grid_num = static_cast<int>(grid_radial_boundaries_.size() - 1);
      if (radial_idx < radial_grid_num) {
        cell.radial_size_ =
          grid_radial_boundaries_[radial_idx + 1] - grid_radial_boundaries_[radial_idx];
      } else {
        cell.radial_size_ = grid_radial_limit_ - grid_radial_boundaries_[radial_idx];
      }
      cell.azimuth_size_ = azimuth_interval_per_radial_[radial_idx];

      cell.center_radius_ = grid_radial_boundaries_[radial_idx] + cell.radial_size_ * 0.5f;
      cell.center_azimuth_ = (static_cast<float>(azimuth_idx) + 0.5f) * cell.azimuth_size_;

      int next_grid_idx = -1;
      if (radial_idx < radial_grid_num) {
        const float azimuth = cell.center_azimuth_;
        const size_t azimuth_idx_next_radial_grid = getAzimuthGridIdx(radial_idx + 1, azimuth);
        next_grid_idx = getGridIdx(radial_idx + 1, azimuth_idx_next_radial_grid);
      }
      cell.next_grid_idx_ = next_grid_idx;

      int prev_grid_idx = -1;
      if (radial_idx > 0) {
        const float azimuth = cell.center_azimuth_;
        const size_t azimuth_idx_prev_radial_grid = getAzimuthGridIdx(radial_idx - 1, azimuth);
        prev_grid_idx = getGridIdx(radial_idx - 1, azimuth_idx_prev_radial_grid);
      }
      cell.prev_grid_idx_ = prev_grid_idx;
      cell.scan_grid_root_idx_ = -1;
    }
  }
};

}  // namespace ground_filter

#endif  // GROUND_FILTER__GRID_HPP_
