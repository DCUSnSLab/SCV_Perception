#include "kf_tracker/CKalmanFilter.h"
#include "kf_tracker/featureDetection.h"
#include "opencv2/video/tracking.hpp"
#include "pcl_ros/point_cloud.h"
#include <algorithm>
#include <fstream>
#include <geometry_msgs/Point.h>
#include <iostream>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32MultiArray.h>
#include <string.h>

#include <pcl/common/centroid.h>
#include <pcl/common/geometry.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include <limits>
#include <utility>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <nav_msgs/Odometry.h> 
#include <tf2_geometry_msgs/tf2_geometry_msgs.h> 
#include <tf2/LinearMath/Matrix3x3.h>

using namespace std;
using namespace cv;

// 전방 선언: publish_cloud 함수
void publish_cloud(ros::Publisher &pub, pcl::PointCloud<pcl::PointXYZ>::Ptr cluster);

// -------------------- 글로벌 변수 --------------------
ros::Publisher objID_pub;
double integrated_vehicle_vx = 0.0;
double integrated_vehicle_vy = 0.0;

// KF init
int stateDim = 4; // [x, y, v_x, v_y]
int measDim  = 2;  // [z_x, z_y]
int ctrlDim  = 0;
cv::KalmanFilter KF0(stateDim, measDim, ctrlDim, CV_32F);
cv::KalmanFilter KF1(stateDim, measDim, ctrlDim, CV_32F);
cv::KalmanFilter KF2(stateDim, measDim, ctrlDim, CV_32F);
cv::KalmanFilter KF3(stateDim, measDim, ctrlDim, CV_32F);
cv::KalmanFilter KF4(stateDim, measDim, ctrlDim, CV_32F);
cv::KalmanFilter KF5(stateDim, measDim, ctrlDim, CV_32F);

ros::Publisher pub_cluster0;
ros::Publisher pub_cluster1;
ros::Publisher pub_cluster2;
ros::Publisher pub_cluster3;
ros::Publisher pub_cluster4;
ros::Publisher pub_cluster5;

ros::Publisher markerPub;

// Odometry publishers (객체별)
ros::Publisher odom_pub0;
ros::Publisher odom_pub1;
ros::Publisher odom_pub2;
ros::Publisher odom_pub3;
ros::Publisher odom_pub4;
ros::Publisher odom_pub5;

// Velocity arrow marker publisher
ros::Publisher vel_arrow_pub;

std::vector<geometry_msgs::Point> prevClusterCenters;  // 이전 프레임 클러스터 중심 (6개)
bool havePrevCenters = false;  // prevClusterCenters가 유효한지

cv::Mat state(stateDim, 1, CV_32F);
cv::Mat_<float> measurement(2, 1);

std::vector<int> objID; // KF와 클러스터 매칭 결과 (크기 6)
bool firstFrame = true;

// -------------------- 유틸 함수 --------------------
double euclidean_distance(const geometry_msgs::Point &p1, const geometry_msgs::Point &p2) {
  return sqrt((p1.x - p2.x)*(p1.x - p2.x) +
              (p1.y - p2.y)*(p1.y - p2.y) +
              (p1.z - p2.z)*(p1.z - p2.z));
}

std::pair<int,int> findIndexOfMin(std::vector<std::vector<float>> distMat) {
  float minEl = std::numeric_limits<float>::max();
  std::pair<int,int> minIndex(0,0);
  for (int i = 0; i < (int)distMat.size(); i++) {
    for (int j = 0; j < (int)distMat[0].size(); j++) {
      if (distMat[i][j] < minEl) {
        minEl = distMat[i][j];
        minIndex = std::make_pair(i, j);
      }
    }
  }
  return minIndex;
}

// -------------------- (1) 차량 Odometry 콜백 --------------------
// odom의 pose.orientation에서 yaw를 직접 추출하여 forward_speed와 함께 vx, vy 계산
void vehicleOdomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
  static ros::Time prev_time;
  static bool firstOdom = true;
  if (firstOdom) {
    prev_time = msg->header.stamp;
    firstOdom = false;
  }
  ros::Time current_time = msg->header.stamp;
  double dt = (current_time - prev_time).toSec();
  prev_time = current_time;

  tf2::Quaternion q;
  tf2::fromMsg(msg->pose.pose.orientation, q);
  double roll, pitch, yaw;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

  double forward_speed = msg->twist.twist.linear.x;
  integrated_vehicle_vx = forward_speed * std::cos(yaw);
  integrated_vehicle_vy = forward_speed * std::sin(yaw);

  double speed = sqrt(integrated_vehicle_vx * integrated_vehicle_vx +
                      integrated_vehicle_vy * integrated_vehicle_vy);
  ROS_INFO("[vehicleOdomCB] yaw=%.3f, forward=%.3f -> vx=%.3f, vy=%.3f, speed=%.3f",
           yaw, forward_speed, integrated_vehicle_vx, integrated_vehicle_vy, speed);
}

// -------------------- (2) KFT 함수 --------------------
// KF predict, 헝가리안 매칭, KF correct 및 클러스터 중심 추출 후
// 이전 프레임과의 좌표 차이를 이용하여 클러스터 속도를 계산하고,
// Odometry와 화살표 Marker로 퍼블리시한다.
void KFT(const std_msgs::Float32MultiArray &ccs, double dt) {
  // 1) KF predict
  std::vector<cv::Mat> pred {
    KF0.predict(), KF1.predict(), KF2.predict(),
    KF3.predict(), KF4.predict(), KF5.predict()
  };

  // 2) ccs → clusterCenters (6개)
  std::vector<geometry_msgs::Point> clusterCenters;
  clusterCenters.reserve(6);
  for (size_t i = 0; i < ccs.data.size(); i += 3) {
    geometry_msgs::Point pt;
    pt.x = ccs.data[i + 0];
    pt.y = ccs.data[i + 1];
    pt.z = ccs.data[i + 2];
    clusterCenters.push_back(pt);
  }

  // 3) KF 예측값 저장
  std::vector<geometry_msgs::Point> KFpredictions(6);
  for (int i = 0; i < 6; i++) {
    KFpredictions[i].x = pred[i].at<float>(0);
    KFpredictions[i].y = pred[i].at<float>(1);
    KFpredictions[i].z = pred[i].at<float>(2);
  }

  // 4) distMat (6x6) 계산: KFpredictions와 clusterCenters 간 거리
  std::vector<std::vector<float>> distMat(6, std::vector<float>(6, 0.f));
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      distMat[i][j] = euclidean_distance(KFpredictions[i], clusterCenters[j]);
    }
  }

  // 5) 헝가리안 방식(그리디) 매칭
  objID.clear();
  objID.resize(6);
  for (int count = 0; count < 6; count++) {
    auto minIdx = findIndexOfMin(distMat);
    int i_kf = minIdx.first;
    int j_cl = minIdx.second;
    objID[i_kf] = j_cl;
    for (int c = 0; c < 6; c++) {
      distMat[i_kf][c] = 999999.f;
    }
    for (int r = 0; r < 6; r++) {
      distMat[r][j_cl] = 999999.f;
    }
  }

  // 6) Marker (KF 예측 위치) 퍼블리시
  visualization_msgs::MarkerArray clusterMarkers;
  clusterMarkers.markers.reserve(6);
  for (int i = 0; i < 6; i++) {
    visualization_msgs::Marker mk;
    mk.header.frame_id = "base_link";
    mk.header.stamp = ros::Time::now();
    mk.ns = "clusters";
    mk.id = i;
    mk.type = visualization_msgs::Marker::CUBE;
    mk.action = visualization_msgs::Marker::ADD;
    mk.scale.x = 0.3;
    mk.scale.y = 0.3;
    mk.scale.z = 0.3;
    mk.color.a = 1.0;
    mk.color.r = (i % 2) ? 1.0 : 0.0;
    mk.color.g = (i % 3) ? 1.0 : 0.0;
    mk.color.b = (i % 4) ? 1.0 : 0.0;
    mk.pose.position.x = KFpredictions[i].x;
    mk.pose.position.y = KFpredictions[i].y;
    mk.pose.position.z = KFpredictions[i].z;
    clusterMarkers.markers.push_back(mk);
  }
  markerPub.publish(clusterMarkers);

  // 7) objID 퍼블리시
  std_msgs::Int32MultiArray obj_id;
  for (int i = 0; i < 6; i++) {
    obj_id.data.push_back(objID[i]);
  }
  objID_pub.publish(obj_id);

  // 8) KF correct
  float meas0[2] = { static_cast<float>(clusterCenters[objID[0]].x),
                     static_cast<float>(clusterCenters[objID[0]].y) };
  float meas1[2] = { static_cast<float>(clusterCenters[objID[1]].x),
                     static_cast<float>(clusterCenters[objID[1]].y) };
  float meas2[2] = { static_cast<float>(clusterCenters[objID[2]].x),
                     static_cast<float>(clusterCenters[objID[2]].y) };
  float meas3[2] = { static_cast<float>(clusterCenters[objID[3]].x),
                     static_cast<float>(clusterCenters[objID[3]].y) };
  float meas4[2] = { static_cast<float>(clusterCenters[objID[4]].x),
                     static_cast<float>(clusterCenters[objID[4]].y) };
  float meas5[2] = { static_cast<float>(clusterCenters[objID[5]].x),
                     static_cast<float>(clusterCenters[objID[5]].y) };

  cv::Mat meas0Mat(2, 1, CV_32F, meas0);
  cv::Mat meas1Mat(2, 1, CV_32F, meas1);
  cv::Mat meas2Mat(2, 1, CV_32F, meas2);
  cv::Mat meas3Mat(2, 1, CV_32F, meas3);
  cv::Mat meas4Mat(2, 1, CV_32F, meas4);
  cv::Mat meas5Mat(2, 1, CV_32F, meas5);

  if (!(meas0[0] == 0.0f && meas0[1] == 0.0f)) KF0.correct(meas0Mat);
  if (!(meas1[0] == 0.0f && meas1[1] == 0.0f)) KF1.correct(meas1Mat);
  if (!(meas2[0] == 0.0f && meas2[1] == 0.0f)) KF2.correct(meas2Mat);
  if (!(meas3[0] == 0.0f && meas3[1] == 0.0f)) KF3.correct(meas3Mat);
  if (!(meas4[0] == 0.0f && meas4[1] == 0.0f)) KF4.correct(meas4Mat);
  if (!(meas5[0] == 0.0f && meas5[1] == 0.0f)) KF5.correct(meas5Mat);

  // 9) 클러스터 속도 계산 (KF 보정된 위치 대신, 이전 프레임과 비교)
  std::vector<geometry_msgs::Point> currentCenters(6);
  for (int i = 0; i < 6; i++) {
    int idx = objID[i];
    currentCenters[i] = clusterCenters[idx];
  }

  // 첫 프레임은 속도 계산 없이 저장
  if (!havePrevCenters) {
    prevClusterCenters = currentCenters;
    havePrevCenters = true;
    ROS_WARN("First velocity calculation unavailable. Storing current centers...");
    return;
  }

  std::vector<double> vx(6, 0.0), vy(6, 0.0);
  for (int i = 0; i < 6; i++) {
    double dx = currentCenters[i].x - prevClusterCenters[i].x;
    double dy = currentCenters[i].y - prevClusterCenters[i].y;
    if (dt > 1e-5) {
      vx[i] = dx / dt;
      vy[i] = dy / dt;
    }
  }

  // Odometry 퍼블리시 (각 클러스터의 속도)
  auto pubOdom = [&](ros::Publisher &pub, int iKF) {
    nav_msgs::Odometry odom;
    odom.header.stamp = ros::Time::now();
    odom.header.frame_id = "base_link";
    odom.pose.pose.position.x = currentCenters[iKF].x;
    odom.pose.pose.position.y = currentCenters[iKF].y;
    odom.pose.pose.position.z = 0.0;
    odom.twist.twist.linear.x = vx[iKF];
    odom.twist.twist.linear.y = vy[iKF];
    odom.twist.twist.linear.z = 0.0;
    double marker_yaw = std::atan2(vy[iKF], vx[iKF]);
    tf2::Quaternion qq;
    qq.setRPY(0, 0, marker_yaw);
    odom.pose.pose.orientation = tf2::toMsg(qq);
    pub.publish(odom);
  };

  pubOdom(odom_pub0, 0);
  pubOdom(odom_pub1, 1);
  pubOdom(odom_pub2, 2);
  pubOdom(odom_pub3, 3);
  pubOdom(odom_pub4, 4);
  pubOdom(odom_pub5, 5);

  // 속도 화살표 MarkerArray 퍼블리시
  visualization_msgs::MarkerArray arrowArray;
  arrowArray.markers.reserve(6);
  const double REL_SPEED_THRESHOLD = 0.5;
  for (int i = 0; i < 6; i++) {
    visualization_msgs::Marker arrow;
    arrow.header.stamp = ros::Time::now();
    arrow.header.frame_id = "base_link";
    arrow.ns = "velocity_arrows";
    arrow.id = i;
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.action = visualization_msgs::Marker::ADD;
    arrow.pose.position.x = currentCenters[i].x;
    arrow.pose.position.y = currentCenters[i].y;
    arrow.pose.position.z = 0.0;
    double arrow_yaw = std::atan2(vy[i], vx[i]);
    tf2::Quaternion q;
    q.setRPY(0, 0, arrow_yaw);
    arrow.pose.orientation = tf2::toMsg(q);
    double cluster_speed = sqrt(vx[i]*vx[i] + vy[i]*vy[i]);
    // 차량 속도 (오도메트리)
    double vehicle_vx = integrated_vehicle_vx;
    double vehicle_vy = integrated_vehicle_vy;
    double rel_vx = vx[i] - vehicle_vx;
    double rel_vy = vy[i] - vehicle_vy;
    double rel_speed = sqrt(rel_vx*rel_vx + rel_vy*rel_vy);
    ROS_INFO("[Cluster %d] speed=%.2f, relative=%.2f", i, cluster_speed, rel_speed);
    arrow.scale.x = 1.0;
    arrow.scale.y = 0.1;
    arrow.scale.z = 0.2;
    if (rel_speed >= REL_SPEED_THRESHOLD) {
      arrow.color.r = 0.0;
      arrow.color.g = 1.0;
      arrow.color.b = 0.0;
    } else {
      arrow.color.r = 1.0;
      arrow.color.g = 0.0;
      arrow.color.b = 0.0;
    }
    arrow.color.a = 1.0;
    arrow.lifetime = ros::Duration(0.3);
    arrowArray.markers.push_back(arrow);
  }
  vel_arrow_pub.publish(arrowArray);

  // 업데이트: 현재 클러스터 중심을 prevClusterCenters로 저장
  prevClusterCenters = currentCenters;
  havePrevCenters = true;
}

// -------------------- (3) publish_cloud 함수 --------------------
void publish_cloud(ros::Publisher &pub, pcl::PointCloud<pcl::PointXYZ>::Ptr cluster) {
  sensor_msgs::PointCloud2::Ptr clustermsg(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*cluster, *clustermsg);
  clustermsg->header.frame_id = "base_link";
  clustermsg->header.stamp = ros::Time::now();
  pub.publish(*clustermsg);
}

// -------------------- (4) PointCloud 콜백 --------------------
void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input) {
  static ros::Time last_time;
  ros::Time current_time = ros::Time::now();
  double dt = 0.0;
  if(last_time.isZero()){
    last_time = current_time;
  }
  dt = (current_time - last_time).toSec();
  last_time = current_time;

  // 전이행렬 업데이트
  cv::Mat transMat = (cv::Mat_<float>(4,4) <<
                      1, 0, dt, 0,
                      0, 1, 0,  dt,
                      0, 0, 1,  0,
                      0, 0, 0,  1);
  KF0.transitionMatrix = transMat.clone();
  KF1.transitionMatrix = transMat.clone();
  KF2.transitionMatrix = transMat.clone();
  KF3.transitionMatrix = transMat.clone();
  KF4.transitionMatrix = transMat.clone();
  KF5.transitionMatrix = transMat.clone();

  if(firstFrame) {
    // KF 초기화 및 첫 클러스터링
    float dvx = 0.01f;
    float dvy = 0.01f;
    float dx  = 1.0f;
    float dy  = 1.0f;
    KF0.transitionMatrix = (Mat_<float>(4,4) <<
      dx, 0, 1, 0,
      0, dy, 0, 1,
      0, 0, dvx, 0,
      0, 0, 0, dvy);
    KF1.transitionMatrix = KF0.transitionMatrix.clone();
    KF2.transitionMatrix = KF0.transitionMatrix.clone();
    KF3.transitionMatrix = KF0.transitionMatrix.clone();
    KF4.transitionMatrix = KF0.transitionMatrix.clone();
    KF5.transitionMatrix = KF0.transitionMatrix.clone();

    cv::setIdentity(KF0.measurementMatrix);
    cv::setIdentity(KF1.measurementMatrix);
    cv::setIdentity(KF2.measurementMatrix);
    cv::setIdentity(KF3.measurementMatrix);
    cv::setIdentity(KF4.measurementMatrix);
    cv::setIdentity(KF5.measurementMatrix);

    float sigmaP = 0.01;
    float sigmaQ = 0.1;
    setIdentity(KF0.processNoiseCov, Scalar::all(sigmaP));
    setIdentity(KF1.processNoiseCov, Scalar::all(sigmaP));
    setIdentity(KF2.processNoiseCov, Scalar::all(sigmaP));
    setIdentity(KF3.processNoiseCov, Scalar::all(sigmaP));
    setIdentity(KF4.processNoiseCov, Scalar::all(sigmaP));
    setIdentity(KF5.processNoiseCov, Scalar::all(sigmaP));

    cv::setIdentity(KF0.measurementNoiseCov, cv::Scalar(sigmaQ));
    cv::setIdentity(KF1.measurementNoiseCov, cv::Scalar(sigmaQ));
    cv::setIdentity(KF2.measurementNoiseCov, cv::Scalar(sigmaQ));
    cv::setIdentity(KF3.measurementNoiseCov, cv::Scalar(sigmaQ));
    cv::setIdentity(KF4.measurementNoiseCov, cv::Scalar(sigmaQ));
    cv::setIdentity(KF5.measurementNoiseCov, cv::Scalar(sigmaQ));

    // 첫 클러스터링 수행
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input, *input_cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(input_cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.08);
    ec.setMinClusterSize(10);
    ec.setMaxClusterSize(600);
    ec.setSearchMethod(tree);
    ec.setInputCloud(input_cloud);
    ec.extract(cluster_indices);

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_vec;
    std::vector<pcl::PointXYZ> clusterCentroids;
    for (auto &inds : cluster_indices) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cCluster(new pcl::PointCloud<pcl::PointXYZ>);
      float sumx = 0, sumy = 0;
      int count = 0;
      for (auto idx : inds.indices) {
        cCluster->points.push_back(input_cloud->points[idx]);
        sumx += input_cloud->points[idx].x;
        sumy += input_cloud->points[idx].y;
        count++;
      }
      pcl::PointXYZ c;
      if(count > 0) {
        c.x = sumx / count;
        c.y = sumy / count;
      }
      c.z = 0.0;
      cluster_vec.push_back(cCluster);
      clusterCentroids.push_back(c);
    }
    while (cluster_vec.size() < 6) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr empty_cluster(new pcl::PointCloud<pcl::PointXYZ>);
      empty_cluster->points.push_back(pcl::PointXYZ(0, 0, 0));
      cluster_vec.push_back(empty_cluster);
    }
    while (clusterCentroids.size() < 6) {
      pcl::PointXYZ dummy; dummy.x = 0; dummy.y = 0; dummy.z = 0;
      clusterCentroids.push_back(dummy);
    }
    // KF 초기 statePre
    KF0.statePre.at<float>(0) = clusterCentroids.at(0).x;
    KF0.statePre.at<float>(1) = clusterCentroids.at(0).y;
    KF0.statePre.at<float>(2) = 0;
    KF0.statePre.at<float>(3) = 0;
    KF1.statePre.at<float>(0) = clusterCentroids.at(1).x;
    KF1.statePre.at<float>(1) = clusterCentroids.at(1).y;
    KF1.statePre.at<float>(2) = 0;
    KF1.statePre.at<float>(3) = 0;
    KF2.statePre.at<float>(0) = clusterCentroids.at(2).x;
    KF2.statePre.at<float>(1) = clusterCentroids.at(2).y;
    KF2.statePre.at<float>(2) = 0;
    KF2.statePre.at<float>(3) = 0;
    KF3.statePre.at<float>(0) = clusterCentroids.at(3).x;
    KF3.statePre.at<float>(1) = clusterCentroids.at(3).y;
    KF3.statePre.at<float>(2) = 0;
    KF3.statePre.at<float>(3) = 0;
    KF4.statePre.at<float>(0) = clusterCentroids.at(4).x;
    KF4.statePre.at<float>(1) = clusterCentroids.at(4).y;
    KF4.statePre.at<float>(2) = 0;
    KF4.statePre.at<float>(3) = 0;
    KF5.statePre.at<float>(0) = clusterCentroids.at(5).x;
    KF5.statePre.at<float>(1) = clusterCentroids.at(5).y;
    KF5.statePre.at<float>(2) = 0;
    KF5.statePre.at<float>(3) = 0;
    firstFrame = false;
    prevClusterCenters.resize(6);
    for (int i = 0; i < 6; i++) {
      geometry_msgs::Point pt;
      pt.x = clusterCentroids.at(i).x;
      pt.y = clusterCentroids.at(i).y;
      pt.z = 0.0;
      prevClusterCenters[i] = pt;
    }
    havePrevCenters = true;
  }
  else {
    // 이후 프레임 처리
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input, *input_cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(input_cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.3);
    ec.setMinClusterSize(10);
    ec.setMaxClusterSize(600);
    ec.setSearchMethod(tree);
    ec.setInputCloud(input_cloud);
    ec.extract(cluster_indices);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_vec;
    std::vector<pcl::PointXYZ> clusterCentroids;
    for(auto &inds : cluster_indices){
      pcl::PointCloud<pcl::PointXYZ>::Ptr cCluster(new pcl::PointCloud<pcl::PointXYZ>);
      float sumx = 0, sumy = 0;
      int count = 0;
      for(auto idx : inds.indices){
        cCluster->points.push_back(input_cloud->points[idx]);
        sumx += input_cloud->points[idx].x;
        sumy += input_cloud->points[idx].y;
        count++;
      }
      pcl::PointXYZ c;
      if(count > 0){
        c.x = sumx / count;
        c.y = sumy / count;
      }
      c.z = 0.0;
      cluster_vec.push_back(cCluster);
      clusterCentroids.push_back(c);
    }
    while (cluster_vec.size() < 6) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr emptyC(new pcl::PointCloud<pcl::PointXYZ>);
      emptyC->points.push_back(pcl::PointXYZ(0, 0, 0));
      cluster_vec.push_back(emptyC);
    }
    while (clusterCentroids.size() < 6) {
      pcl::PointXYZ dummy; dummy.x = 0; dummy.y = 0; dummy.z = 0;
      clusterCentroids.push_back(dummy);
    }
    std_msgs::Float32MultiArray cc;
    for (int i = 0; i < 6; i++) {
      cc.data.push_back(clusterCentroids.at(i).x);
      cc.data.push_back(clusterCentroids.at(i).y);
      cc.data.push_back(clusterCentroids.at(i).z);
    }
    KFT(cc, dt);
    // 클러스터 pointcloud 퍼블리시
    for (int i = 0; i < 6; i++) {
      switch (i) {
        case 0: publish_cloud(pub_cluster0, cluster_vec[objID[i]]); break;
        case 1: publish_cloud(pub_cluster1, cluster_vec[objID[i]]); break;
        case 2: publish_cloud(pub_cluster2, cluster_vec[objID[i]]); break;
        case 3: publish_cloud(pub_cluster3, cluster_vec[objID[i]]); break;
        case 4: publish_cloud(pub_cluster4, cluster_vec[objID[i]]); break;
        case 5: publish_cloud(pub_cluster5, cluster_vec[objID[i]]); break;
        default: break;
      }
    }
  }
}

// -------------------- 메인 함수 --------------------
int main(int argc, char **argv) {
  ros::init(argc, argv, "kf_tracker");
  ros::NodeHandle nh;

  // PointCloud 구독
  ros::Subscriber sub = nh.subscribe("filtered_cloud", 1, cloud_cb);

  // 클러스터 pointcloud 퍼블리셔
  pub_cluster0 = nh.advertise<sensor_msgs::PointCloud2>("cluster_0", 1);
  pub_cluster1 = nh.advertise<sensor_msgs::PointCloud2>("cluster_1", 1);
  pub_cluster2 = nh.advertise<sensor_msgs::PointCloud2>("cluster_2", 1);
  pub_cluster3 = nh.advertise<sensor_msgs::PointCloud2>("cluster_3", 1);
  pub_cluster4 = nh.advertise<sensor_msgs::PointCloud2>("cluster_4", 1);
  pub_cluster5 = nh.advertise<sensor_msgs::PointCloud2>("cluster_5", 1);

  objID_pub = nh.advertise<std_msgs::Int32MultiArray>("obj_id", 1);
  markerPub = nh.advertise<visualization_msgs::MarkerArray>("viz", 1);

  odom_pub0 = nh.advertise<nav_msgs::Odometry>("cluster0_odom", 1);
  odom_pub1 = nh.advertise<nav_msgs::Odometry>("cluster1_odom", 1);
  odom_pub2 = nh.advertise<nav_msgs::Odometry>("cluster2_odom", 1);
  odom_pub3 = nh.advertise<nav_msgs::Odometry>("cluster3_odom", 1);
  odom_pub4 = nh.advertise<nav_msgs::Odometry>("cluster4_odom", 1);
  odom_pub5 = nh.advertise<nav_msgs::Odometry>("cluster5_odom", 1);

  vel_arrow_pub = nh.advertise<visualization_msgs::MarkerArray>("velocity_arrows", 1);

  // 차량 Odometry 구독 (heading은 orientation에서 직접 추출)
  ros::Subscriber vehicleOdomSub = nh.subscribe("/odom/coordinate/gps", 1, vehicleOdomCallback);

  ros::spin();
  return 0;
}
