#ifndef NEGATIVE_OBSTACLE_DETECTOR__EDGE_DETECTOR_HPP_
#define NEGATIVE_OBSTACLE_DETECTOR__EDGE_DETECTOR_HPP_

#include <vector>
#include <cmath>

namespace negative_obstacle_detector
{

struct Point3D
{
  float x, y, z;

  float distance() const { return std::sqrt(x * x + y * y); }
  float azimuth() const { return std::atan2(y, x); }
};

class EdgeDetector
{
public:
  void initialize(double range_x, double range_y, double ground_z_min, double ground_z_max,
                  double negative_z_max, int num_sectors,
                  double cluster_tolerance, double interpolation_resolution);

  std::vector<Point3D> filter(const std::vector<Point3D>& points);

private:
  double range_x_ = 10.0;
  double range_y_ = 5.0;
  double ground_z_min_ = -0.1;
  double ground_z_max_ = 0.1;
  double negative_z_max_ = -0.15;
  int num_sectors_ = 360;
  double sector_width_;
  double cluster_tolerance_ = 0.5;       // 클러스터 거리 허용치 (m)
  double interpolation_res_ = 0.1;       // 보간 해상도 (m)

  int getSectorIndex(float azimuth) const;

  // 클러스터링 및 보간
  std::vector<Point3D> clusterAndInterpolate(const std::vector<Point3D>& edge_points);
};

}  // namespace negative_obstacle_detector

#endif  // NEGATIVE_OBSTACLE_DETECTOR__EDGE_DETECTOR_HPP_
