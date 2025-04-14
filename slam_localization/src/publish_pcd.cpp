#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "utm_transform_node");
  ros::NodeHandle nh;

  // (1) PCD 파일 경로 설정
  //     실제 사용 시 파라미터, argv, launch 등으로 받을 수 있습니다.
  std::string pcd_file = "/home/ssc/SCV_Perception_ws/src/SCV_Perception/slam_localization/data/map.pcd";

  // (2) PCD 읽기
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>());
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *cloud) == -1) {
    ROS_ERROR("Failed to load PCD file: %s", pcd_file.c_str());
    return -1;
  }
  ROS_INFO("Loaded %d points from %s", (int)cloud->points.size(), pcd_file.c_str());

  // (3) UTM 파일에서 easting, northing, altitude 읽기
  std::ifstream utm_file(pcd_file + ".utm");
  if(!utm_file.is_open()) {
    ROS_WARN("Failed to open UTM file: %s.utm -- Skip offset", pcd_file.c_str());
  } else {
    double utm_easting, utm_northing, altitude;
    utm_file >> utm_easting >> utm_northing >> altitude;
    utm_file.close();

    // (4) 전체 포인트에 UTM 오프셋 적용
    for(auto &pt : cloud->points) {
      pt.x -= utm_easting;
      pt.y -= utm_northing;
      pt.z -= altitude;
    }
    ROS_INFO_STREAM("Applied UTM offset (x=" << utm_easting
                    << ", y=" << utm_northing
                    << ", z=" << altitude << ")");
  }

  // (5) 퍼블리시를 위한 ROS 메시지 변환
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(*cloud, output);
  output.header.frame_id = "map";

  // (6) 퍼블리셔 생성
  ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/utm_points", 1);

  // (7) 주기적으로 퍼블리시
  ros::Rate loop_rate(1); // 1Hz
  while(ros::ok()) {
    output.header.stamp = ros::Time::now();
    cloud_pub.publish(output);

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
