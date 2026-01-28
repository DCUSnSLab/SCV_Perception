#include "negative_obstacle_detector/edge_detector.hpp"

#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace negative_obstacle_detector
{

void EdgeDetector::initialize(double range_x, double range_y, double ground_z_min, double ground_z_max,
                              double negative_z_max, int num_sectors,
                              double cluster_tolerance, double interpolation_resolution)
{
  range_x_ = range_x / 2.0;
  range_y_ = range_y / 2.0;
  ground_z_min_ = ground_z_min;
  ground_z_max_ = ground_z_max;
  negative_z_max_ = negative_z_max;
  num_sectors_ = num_sectors;
  sector_width_ = 2.0 * M_PI / num_sectors_;
  cluster_tolerance_ = cluster_tolerance;
  interpolation_res_ = interpolation_resolution;
}

int EdgeDetector::getSectorIndex(float azimuth) const
{
  float angle = azimuth;
  if (angle < 0) {
    angle += 2.0f * static_cast<float>(M_PI);
  }
  int idx = static_cast<int>(angle / sector_width_);
  if (idx < 0) idx = 0;
  if (idx >= num_sectors_) idx = num_sectors_ - 1;
  return idx;
}

std::vector<Point3D> EdgeDetector::clusterAndInterpolate(const std::vector<Point3D>& edge_points)
{
  if (edge_points.empty()) {
    return {};
  }

  if (edge_points.size() == 1) {
    return edge_points;
  }

  // 각도순 정렬
  std::vector<Point3D> sorted = edge_points;
  std::sort(sorted.begin(), sorted.end(), [](const Point3D& a, const Point3D& b) {
    return a.azimuth() < b.azimuth();
  });

  // 클러스터링
  std::vector<std::vector<Point3D>> clusters;
  std::vector<Point3D> current_cluster;
  current_cluster.push_back(sorted[0]);

  for (size_t i = 1; i < sorted.size(); ++i) {
    const auto& prev = sorted[i - 1];
    const auto& curr = sorted[i];

    float dx = curr.x - prev.x;
    float dy = curr.y - prev.y;
    float dist = std::sqrt(dx * dx + dy * dy);

    if (dist <= cluster_tolerance_) {
      current_cluster.push_back(curr);
    } else {
      clusters.push_back(current_cluster);
      current_cluster.clear();
      current_cluster.push_back(curr);
    }
  }
  clusters.push_back(current_cluster);

  // 각 클러스터의 대표점 (중심점) 계산
  std::vector<Point3D> cluster_centers;
  for (const auto& cluster : clusters) {
    Point3D center = {0, 0, 0};
    for (const auto& p : cluster) {
      center.x += p.x;
      center.y += p.y;
      center.z += p.z;
    }
    float n = static_cast<float>(cluster.size());
    center.x /= n;
    center.y /= n;
    center.z /= n;
    cluster_centers.push_back(center);
  }

  // 클러스터 중심점들 사이 보간
  std::vector<Point3D> result;

  for (size_t i = 0; i < cluster_centers.size(); ++i) {
    const auto& p1 = cluster_centers[i];
    result.push_back(p1);

    // 다음 클러스터와 보간
    if (i + 1 < cluster_centers.size()) {
      const auto& p2 = cluster_centers[i + 1];

      float dx = p2.x - p1.x;
      float dy = p2.y - p1.y;
      float dz = p2.z - p1.z;
      float dist = std::sqrt(dx * dx + dy * dy);

      if (dist > interpolation_res_) {
        int num_interp = static_cast<int>(dist / interpolation_res_);
        for (int j = 1; j < num_interp; ++j) {
          float t = static_cast<float>(j) / num_interp;
          Point3D interp;
          interp.x = p1.x + t * dx;
          interp.y = p1.y + t * dy;
          interp.z = p1.z + t * dz;
          result.push_back(interp);
        }
      }
    }
  }

  return result;
}

std::vector<Point3D> EdgeDetector::filter(const std::vector<Point3D>& points)
{
  if (points.empty()) {
    return {};
  }

  // 각 섹터별로 가장 먼 지면 포인트와 negative 존재 여부 추적
  std::vector<Point3D> farthest_ground(num_sectors_);
  std::vector<float> max_ground_dist(num_sectors_, -1.0f);
  std::vector<bool> has_negative(num_sectors_, false);

  for (const auto& p : points) {
    // XY 범위 체크
    if (p.x < -range_x_ || p.x > range_x_ ||
        p.y < -range_y_ || p.y > range_y_) {
      continue;
    }

    int idx = getSectorIndex(p.azimuth());
    float dist = p.distance();

    // 지면 포인트 체크
    if (p.z >= ground_z_min_ && p.z <= ground_z_max_) {
      if (dist > max_ground_dist[idx]) {
        max_ground_dist[idx] = dist;
        farthest_ground[idx] = p;
      }
    }

    // Negative 포인트 체크
    if (p.z < negative_z_max_) {
      has_negative[idx] = true;
    }
  }

  // Negative가 있는 섹터의 지면 끝점 수집
  std::vector<Point3D> edges;
  for (int i = 0; i < num_sectors_; ++i) {
    if (has_negative[i] && max_ground_dist[i] > 0) {
      edges.push_back(farthest_ground[i]);
    }
  }

  // 클러스터링 및 보간
  return clusterAndInterpolate(edges);
}

}  // namespace negative_obstacle_detector
