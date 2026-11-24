// LiDAR traversability grid: step / low obstacle / drop-off costs around the robot,
// published as an OccupancyGrid in the odom frame.
#include "traversability_grid/elevation_grid.hpp"

#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <malloc.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace traversability_grid
{

class TraversabilityNode : public rclcpp::Node
{
public:
  TraversabilityNode()
  : Node("traversability_node")
  {
    const auto input_topic = declare_parameter<std::string>("input_topic", "/velodyne_points");
    const auto output_topic = declare_parameter<std::string>("output_topic", "~/grid");
    const bool input_best_effort = declare_parameter<bool>("input_best_effort", false);
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    transform_timeout_sec_ = declare_parameter<double>("transform_timeout_sec", 0.1);

    range_min_ = declare_parameter<double>("range_min", 0.3);
    range_max_ = declare_parameter<double>("range_max", 20.0);
    z_max_ = declare_parameter<double>("z_max", 2.5);
    body_ = declare_parameter<std::vector<double>>(
      "body_box", {-0.9, 0.65, -0.45, 0.45, -0.3, 1.3});

    step_high_ = declare_parameter<double>("step_high", 0.08);
    step_lethal_ = declare_parameter<double>("step_lethal", 0.12);
    cost_high_ = static_cast<int8_t>(declare_parameter<int>("cost_high", 70));

    p_.res = declare_parameter<double>("resolution", p_.res);
    p_.t_exp = declare_parameter<double>("ground_expiry_sec", p_.t_exp);
    p_.t_obs = declare_parameter<double>("obstacle_memory_sec", p_.t_obs);
    p_.r_slack = declare_parameter<double>("range_slack", p_.r_slack);
    p_.h_ovh = declare_parameter<double>("overhang_height", p_.h_ovh);
    p_.ovh_cap = declare_parameter<double>("grade_allowance_max", p_.ovh_cap);
    p_.gnd_max = declare_parameter<double>("ground_max_above_robot", p_.gnd_max);
    p_.obj_ext = declare_parameter<double>("object_extent", p_.obj_ext);
    p_.fit_rmin = declare_parameter<double>("fit_range_min", p_.fit_rmin);
    p_.fit_rmax = declare_parameter<double>("fit_range_max", p_.fit_rmax);
    p_.slope_pull = declare_parameter<double>("slope_pull", p_.slope_pull);
    p_.freeze_standing = declare_parameter<bool>("freeze_slope_when_standing", p_.freeze_standing);
    p_.half = declare_parameter<double>("window_half_size", p_.half);
    p_.r_conf = declare_parameter<double>("confirm_range", p_.r_conf);
    p_.k_med = static_cast<int>(declare_parameter<int>("median_size", p_.k_med));
    p_.fill = declare_parameter<double>("fill_gap", p_.fill);
    p_.r_step = declare_parameter<double>("step_radius", p_.r_step);
    p_.r_gnd = declare_parameter<double>("ground_radius", p_.r_gnd);
    p_.obs_height = declare_parameter<double>("obstacle_height", p_.obs_height);
    p_.drop_min = declare_parameter<double>("drop_min", p_.drop_min);
    p_.drop_min_gap = declare_parameter<double>("drop_min_gap", p_.drop_min_gap);
    // every cell touched by one scan and the output window must fit in the ring
    p_.ring_cells = static_cast<int>(std::ceil(2.0 * (std::max(range_max_, p_.half) + 1.0) / p_.res));

    if (body_.size() != 6) {
      throw std::runtime_error("body_box must be [x_min, x_max, y_min, y_max, z_min, z_max]");
    }

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    auto qos = rclcpp::SensorDataQoS();
    if (!input_best_effort) {qos.reliable();}
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic, qos, [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) {onCloud(m);});
    grid_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(output_topic, rclcpp::QoS(1));
    step_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("~/step_cm", rclcpp::QoS(1));

    RCLCPP_INFO(get_logger(), "traversability_grid: %s -> %s (%s), %.0f m window at %.2f m",
      input_topic.c_str(), grid_pub_->get_topic_name(), odom_frame_.c_str(), 2 * p_.half, p_.res);
  }

private:
  std::optional<tf2::Transform> lookup(const std::string & target, const std::string & source,
                                       const rclcpp::Time & stamp)
  {
    try {
      const auto tf = tf_buffer_->lookupTransform(
        target, source, stamp, rclcpp::Duration::from_seconds(transform_timeout_sec_));
      tf2::Transform out;
      tf2::fromMsg(tf.transform, out);
      return out;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "no transform %s -> %s: %s",
        source.c_str(), target.c_str(), ex.what());
      return std::nullopt;
    }
  }

  void onCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
  {
    const auto t0 = std::chrono::steady_clock::now();
    const rclcpp::Time stamp(msg->header.stamp);
    const auto base_sensor = lookup(base_frame_, msg->header.frame_id, stamp);
    const auto odom_base = lookup(odom_frame_, base_frame_, stamp);
    if (!base_sensor || !odom_base) {return;}
    const auto t1 = std::chrono::steady_clock::now();   // after waiting for TF

    if (!grid_) {
      p_.lidar_x = base_sensor->getOrigin().x();
      p_.lidar_z = base_sensor->getOrigin().z();
      grid_ = std::make_unique<ElevationGrid>(p_);
    }

    int off[3] = {-1, -1, -1};
    for (const auto & f : msg->fields) {
      const int k = f.name == "x" ? 0 : f.name == "y" ? 1 : f.name == "z" ? 2 : -1;
      if (k >= 0 && f.datatype == sensor_msgs::msg::PointField::FLOAT32) {off[k] = static_cast<int>(f.offset);}
    }
    if (off[0] < 0 || off[1] < 0 || off[2] < 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "input cloud has no float32 x/y/z");
      return;
    }
    const tf2::Matrix3x3 & Rm = base_sensor->getBasis();
    const tf2::Vector3 & To = base_sensor->getOrigin();
    pts_.clear();
    const size_t np = static_cast<size_t>(msg->width) * msg->height;
    pts_.reserve(np);
    for (size_t k = 0; k < np; ++k) {
      const uint8_t * d = &msg->data[k * msg->point_step];
      float v[3];
      for (int c = 0; c < 3; ++c) {std::memcpy(&v[c], d + off[c], sizeof(float));}
      if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {continue;}
      double q[3];
      for (int c = 0; c < 3; ++c) {q[c] = Rm[c][0] * v[0] + Rm[c][1] * v[1] + Rm[c][2] * v[2] + To[c];}
      const double r2 = q[0] * q[0] + q[1] * q[1];
      if (r2 <= range_min_ * range_min_ || r2 >= range_max_ * range_max_ || q[2] >= z_max_) {continue;}
      const bool body = q[0] > body_[0] && q[0] < body_[1] && q[1] > body_[2] && q[1] < body_[3] &&
        q[2] > body_[4] && q[2] < body_[5];
      if (!body) {pts_.push_back({q[0], q[1], q[2]});}
    }

    const double t = stamp.seconds();
    const double x = odom_base->getOrigin().x(), y = odom_base->getOrigin().y();
    const double yaw = tf2::getYaw(odom_base->getRotation());
    grid_->update(t, pts_, x, y, yaw);
    const Window w = grid_->window(t, x, y, yaw);
    publish(w, msg->header.stamp, x, y);

    const auto t2 = std::chrono::steady_clock::now();
    wait_ms_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
    sum_ms_ += std::chrono::duration<double, std::milli>(t2 - t1).count();
    if (++n_scans_ == 300) {
      RCLCPP_INFO(get_logger(), "compute %.1f ms/scan (TF wait %.1f ms/scan)",
        sum_ms_ / n_scans_, wait_ms_ / n_scans_);
      sum_ms_ = wait_ms_ = 0.0;
      n_scans_ = 0;
    }
  }

  void publish(const Window & w, const builtin_interfaces::msg::Time & stamp, double x, double y)
  {
    const double zs = grid_->groundZ();
    nav_msgs::msg::OccupancyGrid g;
    g.header.stamp = stamp;
    g.header.frame_id = odom_frame_;
    g.info.resolution = static_cast<float>(p_.res);
    g.info.width = g.info.height = static_cast<uint32_t>(w.n);
    g.info.origin.position.x = w.wx;
    g.info.origin.position.y = w.wy;
    g.info.origin.orientation.w = 1.0;
    const size_t nn = static_cast<size_t>(w.n) * w.n;
    g.data.assign(nn, -1);
    const bool want_step = step_pub_->get_subscription_count() > 0;
    nav_msgs::msg::OccupancyGrid s;
    if (want_step) {s = g; s.data.assign(nn, -1);}
    for (int i = 0; i < w.n; ++i) {
      for (int j = 0; j < w.n; ++j) {
        const size_t k = static_cast<size_t>(i) * w.n + j;   // window: i along x
        const size_t o = static_cast<size_t>(j) * w.n + i;   // grid msg: row = y
        // Confirmed step where available, provisional (seen only from far) elsewhere.
        // Far heights are noisy, so a provisional step stays at cost_high until it is
        // confirmed within confirm_range.
        const bool conf = !std::isnan(w.step_c[k]);
        const float st = conf ? w.step_c[k] : w.step_all[k];
        int8_t c = -1;
        if (!std::isnan(st)) {
          c = st >= step_lethal_ ? (conf ? 100 : cost_high_) : st >= step_high_ ? cost_high_ : 0;
        }
        // An object with no ground seen within ground_radius (the near blind zone right
        // after start-up) has no hobs. Judge it against the robot's ground plus the grade
        // allowance instead, so it is not published as unobserved.
        float ho = w.hobs[k];
        if (std::isnan(ho) && !std::isnan(w.zmax[k])) {
          const double dx = w.wx + (i + 0.5) * p_.res - x, dy = w.wy + (j + 0.5) * p_.res - y;
          ho = static_cast<float>(w.zmax[k] - (zs + std::min(0.15 * std::hypot(dx, dy), p_.ovh_cap)));
        }
        if (!std::isnan(ho) && ho > p_.obs_height) {c = 100;}
        if (!std::isnan(w.drop[k])) {c = 100;}
        g.data[o] = c;
        if (want_step && !std::isnan(st)) {s.data[o] = static_cast<int8_t>(std::min(100.0f, st * 100.0f));}
      }
    }
    grid_pub_->publish(g);
    if (want_step) {step_pub_->publish(s);}
  }

  Params p_;
  std::string odom_frame_, base_frame_;
  double transform_timeout_sec_;
  double range_min_, range_max_, z_max_;
  std::vector<double> body_;
  double step_high_, step_lethal_;
  int8_t cost_high_;

  std::unique_ptr<ElevationGrid> grid_;
  std::vector<Point> pts_;
  double sum_ms_ = 0.0, wait_ms_ = 0.0;
  int n_scans_ = 0;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_, step_pub_;
};

}  // namespace traversability_grid

int main(int argc, char ** argv)
{
  // The per-scan grids are a few hundred kB each. Keep such allocations on the heap
  // instead of fresh mmap pages every scan (~150 page faults per scan otherwise).
  mallopt(M_MMAP_THRESHOLD, 64 << 20);
  mallopt(M_TRIM_THRESHOLD, 256 << 20);
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<traversability_grid::TraversabilityNode>());
  rclcpp::shutdown();
  return 0;
}
