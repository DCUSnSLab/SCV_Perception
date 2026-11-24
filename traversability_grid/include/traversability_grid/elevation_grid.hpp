// Robot-centric LiDAR elevation grid with step / obstacle / drop-off detection.
// ROS-free core; port of the offline Python prototype (v9).
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace traversability_grid
{

struct Params
{
  double res = 0.1;          // cell size (m)
  int ring_cells = 512;      // storage is ring_cells^2, must span window + sensor range
  double t_exp = 10.0;       // ground cell expiry (s)
  double t_obs = 0.3;        // obstacle-top memory (s)
  double r_slack = 0.5;      // skip ground updates from farther than best range + r_slack (m)
  double h_ovh = 1.3;        // ignore points this far above the ground (overhangs, canopy) (m)
  double ovh_cap = 0.4;      // grade allowance cap for cells without ground (m)
  double gnd_max = 0.5;      // first observation this far above robot ground is not ground (m)
  double obj_ext = 0.3;      // vertical extent per scan that marks a cell as an object (m)
  double fit_rmin = 1.5;     // scan-to-map height/slope fit ring (m)
  double fit_rmax = 10.0;
  double slope_pull = 0.05;  // per-scan blend of the fitted slope toward level
  bool freeze_standing = true;  // skip the blend while the robot does not move
  double half = 11.0;        // output window half size (m)
  double r_conf = 5.0;       // a step is confirmed once its cell was seen from within this range (m)
  int k_med = 5;             // median filter size (cells)
  double fill = 0.3;         // bridge unobserved gaps up to this distance (m)
  double r_step = 0.15;      // step neighbourhood radius (m)
  double r_gnd = 0.5;        // local ground neighbourhood radius (m)
  double obs_height = 0.3;   // obstacle top above local ground (m)
  double drop_min = 0.15;    // minimum drop-off height (m)
  double drop_min_gap = 0.2; // minimum occlusion gap (m)
  double lidar_x = 0.21;     // sensor position in base frame (m)
  double lidar_z = 0.905;
};

struct Point
{
  double x, y, z;
};

// Output window, row-major [i * n + j] with i along x, j along y. NaN = no value.
struct Window
{
  int n = 0;
  double wx = 0.0, wy = 0.0;  // world position of cell (0, 0) corner
  std::vector<float> zall, zmax;
  std::vector<float> step_all;  // step height from every observation
  std::vector<float> step_c;    // step height from cells seen within r_conf
  std::vector<float> hobs;      // obstacle top above local ground
  std::vector<float> drop;      // drop-off height at the edge cell
  std::vector<uint8_t> shadow;  // occlusion gap behind a drop-off
};

class ElevationGrid
{
public:
  // (ox, oy): world position of the global cell index origin.
  explicit ElevationGrid(const Params & p, double ox = 0.0, double oy = 0.0);

  // Add one scan. pts in base frame, pose (x, y, yaw) of base in the world frame at time t.
  void update(double t, const std::vector<Point> & pts, double x, double y, double yaw);

  Window window(double t, double x, double y, double yaw) const;

  double groundZ() const { return zs_; }
  const double * fit() const { return c_; }

private:
  struct Cell
  {
    float zavg, rng, zmax;
    double tmin, tmax;
    int32_t ki, kj;
  };

  long gi(double x) const;
  long gj(double y) const;
  size_t slot(long i, long j) const;
  Cell & touch(long i, long j);                 // reset if the slot holds another cell
  const Cell * find(long i, long j) const;      // nullptr if never written

  void align(double t, const std::vector<Point> & W, double cx, double cy);
  void add(double t, const std::vector<Point> & W, double cx, double cy);
  void features(const std::vector<float> & z, const std::vector<float> & zmax, int n,
                std::vector<float> & step, std::vector<float> * hobs, bool split) const;
  void drops(Window & w, double sx, double sy, double sz) const;

  Params p_;
  double ox_, oy_;
  int N_;
  std::vector<Cell> cells_;
  std::vector<std::pair<int, int>> net_;  // median sorting network
  double zs_ = 0.0;          // robot ground height in the world frame
  double bc_[2] = {0.0, 0.0};  // slope carried to the next fit
  double c_[3] = {0.0, 0.0, 0.0};
  bool has_last_ = false;
  double last_x_ = 0.0, last_y_ = 0.0;

  // per-scan scratch, indexed by slot
  std::vector<uint32_t> stamp_, obj_stamp_;
  std::vector<int32_t> best_;
  std::vector<double> zlo_, zhi_, rlo_;
  std::vector<long> ti_, tj_;  // cells touched by the current scan
  uint32_t scan_ = 0;
};

}  // namespace traversability_grid
