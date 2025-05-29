/**********************************************
 * median_based_kf_tracker_with_bbox.cpp
 * 중앙값(median)으로 군집 중심 계산 + 저역통과 필터(ego-motion 보상 및 시간적 평활화)
 * + 칼만 필터 + Bounding Box + 충돌 예측 (향후 5초)
 *********************************************/

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
 
 // odom 메시지 구독을 위한 헤더
 #include <nav_msgs/Odometry.h>
 
 // TF 변환 (base_link → gps_utm)
 #include <tf/transform_listener.h>
 #include <geometry_msgs/PointStamped.h>
 
 // Heading 계산 시 orientation 변환
 #include <tf/LinearMath/Quaternion.h>
 #include <tf/LinearMath/Matrix3x3.h>
 #include <tf/transform_datatypes.h>
 
 #include <Eigen/Dense>
 #include <pcl/common/transforms.h>
 #include <pcl/common/centroid.h>
 #include <pcl/common/pca.h>
 
 using namespace std;
 using namespace cv;
 
 // ------------------- 전역 변수/객체 ------------------- //
 
 // 최초 프레임 플래그
 static bool firstFrame = true;
 
 // reset 임계치: 이전 필터 상태와 측정값 차이가 1.0m 이상이면 필터 재설정
 static const double reset_threshold = 0.5;
 
 // 충돌 예측 임계치 (예: 2.0m 이내면 충돌 가능성 있음)
 static const double collision_threshold = 1.0;
 static const double static_threshold = 0.6;
 // 퍼블리셔
 static ros::Publisher objID_pub;
 static ros::Publisher pub_cluster0;
 static ros::Publisher pub_cluster1;
 static ros::Publisher pub_cluster2;
 static ros::Publisher pub_cluster3;
 static ros::Publisher pub_cluster4;
 static ros::Publisher pub_cluster5;
 static ros::Publisher markerPub;    // 칼만 필터 예측 위치 표시용
 static ros::Publisher arrowPub;     // Heading ARROW 표시용
 static ros::Publisher bboxPub;      // Bounding Box 표시용
 static ros::Publisher predPathPub;  // 충돌 위험 객체의 예상 경로 퍼블리시용
 
 // 칼만 필터 (최대 6개 객체)
 static cv::KalmanFilter KF0(4, 2, 0), KF1(4, 2, 0), KF2(4, 2, 0),
                         KF3(4, 2, 0), KF4(4, 2, 0), KF5(4, 2, 0);
 
 // 매칭된 객체 ID
 static std::vector<int> objID(6);
 
 // 이전 프레임 시간 및 초기화용
 static ros::Time prevTime;
 static bool firstTimeStamp = true;
 
 // base_link에서 이전 위치 (트래킹된 객체)
 static std::vector<geometry_msgs::Point> prevObjPositions(6);
 
 // gps_utm에서 이전 위치 (트래킹된 객체)
 static std::vector<geometry_msgs::Point> prevObjPositionsUTM(6);
 
 // 저역 통과 필터 결과 (월드 좌표계)
 static std::vector<double> world_x_filtered(6, 0.0);
 static std::vector<double> world_y_filtered(6, 0.0);
 static bool filter_initialized = false;
 static const double alpha = 0.2; // Low pass filter 계수
 
 // 차량의 odom (gps_utm) 좌표 및 속도 (global)
 static geometry_msgs::Point vehicle_pose; // gps_utm 기준
 static double vehicle_vel_x = 0.0;
 static double vehicle_vel_y = 0.0;
 
 // 트래킹 객체의 속도 (계산된 값, gps_utm 기준)
 static std::vector<double> object_vel_x(6, 0.0);
 static std::vector<double> object_vel_y(6, 0.0);
 
 // TF
 static tf::TransformListener* tfListener = nullptr;
 
 // 더미 데이터 여부 플래그 (각 객체별 KF측정이 더미인지)
 static std::vector<bool> isDummyMeasurement(6, false);
 
 // ★ 추가: 각 트랙의 이전 객체 ID를 저장하는 벡터 (초기값 -1: 아직 할당되지 않음)
 static std::vector<int> prevObjID(6, -1);
 
 // ------------------- distance 함수 ------------------- //
 double euclidean_distance(const geometry_msgs::Point &p1,
                           const geometry_msgs::Point &p2)
 {
   double dx = p1.x - p2.x;
   double dy = p1.y - p2.y;
   double dz = p1.z - p2.z;
   return std::sqrt(dx*dx + dy*dy + dz*dz);
 }
  
 // ------------------- distMat 최소값 위치 찾기 ------------------- //
 std::pair<int, int> findIndexOfMin(const std::vector<std::vector<float>> &distMat)
 {
   std::pair<int, int> minIndex;
   float minVal = std::numeric_limits<float>::max();
   for (int i = 0; i < (int)distMat.size(); i++) {
     for (int j = 0; j < (int)distMat[i].size(); j++) {
       if (distMat[i][j] < minVal) {
         minVal = distMat[i][j];
         minIndex = std::make_pair(i, j);
       }
     }
   }
   return minIndex;
 }
  
 // ------------------- median 계산 함수 ------------------- //
 pcl::PointXYZ getMedianPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster)
 {
   std::vector<float> xs, ys;
   xs.reserve(cluster->points.size());
   ys.reserve(cluster->points.size());
   for (auto &pt : cluster->points) {
     xs.push_back(pt.x);
     ys.push_back(pt.y);
   }
   pcl::PointXYZ medianPt;
   if (xs.empty()) {
     medianPt.x = 0;
     medianPt.y = 0;
     medianPt.z = 0;
     return medianPt;
   }
   size_t mid = xs.size() / 2;
   std::nth_element(xs.begin(), xs.begin() + mid, xs.end());
   std::nth_element(ys.begin(), ys.begin() + mid, ys.end());
   medianPt.x = xs[mid];
   medianPt.y = ys[mid];
   medianPt.z = 0.0f;
   return medianPt;
 }
  
 // ------------------- 오리엔티드 바운딩박스 계산 함수 ------------------- //
 bool computeOrientedBoundingBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster,
                                 geometry_msgs::Pose &obbPose,
                                 geometry_msgs::Vector3 &obbScale)
 {
   if (cluster->points.empty()) {
     ROS_WARN("Empty cluster for OBB computation!");
     return false;
   }
   if (!cluster || cluster->points.size() < 3) {
     ROS_WARN("Cannot compute OBB for cluster with <3 points.");
     return false;
   }
   Eigen::Vector4f pcaCentroid;
   pcl::compute3DCentroid(*cluster, pcaCentroid);
   
   pcl::PCA<pcl::PointXYZ> pca;
   pca.setInputCloud(cluster);
   Eigen::Matrix3f eigenVectors = pca.getEigenVectors();
   
   Eigen::Matrix4f projTransform(Eigen::Matrix4f::Identity());
   projTransform.block<3,3>(0,0) = eigenVectors.transpose();
   projTransform.block<3,1>(0,3) = -1.0f * (eigenVectors.transpose() * pcaCentroid.head<3>());
   
   pcl::PointCloud<pcl::PointXYZ>::Ptr cloudProjected(new pcl::PointCloud<pcl::PointXYZ>);
   pcl::transformPointCloud(*cluster, *cloudProjected, projTransform);
   
   pcl::PointXYZ minPt, maxPt;
   minPt.x = minPt.y = minPt.z = std::numeric_limits<float>::max();
   maxPt.x = maxPt.y = maxPt.z = -std::numeric_limits<float>::max();
   
   for (const auto &pt : cloudProjected->points) {
     if (pt.x < minPt.x) minPt.x = pt.x;
     if (pt.y < minPt.y) minPt.y = pt.y;
     if (pt.z < minPt.z) minPt.z = pt.z;
     if (pt.x > maxPt.x) maxPt.x = pt.x;
     if (pt.y > maxPt.y) maxPt.y = pt.y;
     if (pt.z > maxPt.z) maxPt.z = pt.z;
   }
   
   Eigen::Vector3f meanDiagonal = 0.5f * (Eigen::Vector3f(maxPt.x, maxPt.y, maxPt.z) +
                                          Eigen::Vector3f(minPt.x, minPt.y, minPt.z));
   Eigen::Vector3f obb_center_world = eigenVectors * meanDiagonal + pcaCentroid.head<3>();
   
   float length = maxPt.x - minPt.x;
   float width  = maxPt.y - minPt.y;
   float height = maxPt.z - minPt.z;
   
   Eigen::Matrix3f rotation = eigenVectors;
   Eigen::Quaternionf q(rotation);
   
   obbPose.position.x = obb_center_world.x();
   obbPose.position.y = obb_center_world.y();
   obbPose.position.z = obb_center_world.z();
   
   obbPose.orientation.x = q.x();
   obbPose.orientation.y = q.y();
   obbPose.orientation.z = q.z();
   obbPose.orientation.w = q.w();
   
   obbScale.x = fabs(length);
   obbScale.y = fabs(width);
   obbScale.z = fabs(height);
   
   return true;
 }
  
 // ------------------- 칼만필터 업데이트 함수 ------------------- //
 void KFT(const std_msgs::Float32MultiArray &ccs)
 {
   std::vector<cv::Mat> pred{
     KF0.predict(), KF1.predict(), KF2.predict(),
     KF3.predict(), KF4.predict(), KF5.predict()
   };
   
   std::vector<geometry_msgs::Point> clusterCenters;
   clusterCenters.reserve(6);
   for (size_t i = 0; i < ccs.data.size(); i += 3) {
     geometry_msgs::Point p;
     p.x = ccs.data[i];
     p.y = ccs.data[i+1];
     p.z = ccs.data[i+2];
     clusterCenters.push_back(p);
   }
   
   std::vector<geometry_msgs::Point> KFpredictions;
   KFpredictions.reserve(6);
   for (int i = 0; i < 6; i++) {
     geometry_msgs::Point pt;
     pt.x = pred[i].at<float>(0);
     pt.y = pred[i].at<float>(1);
     pt.z = 0.0;
     KFpredictions.push_back(pt);
   }
   
   std::vector<std::vector<float>> distMat(6, std::vector<float>(6, 0.0f));
   for (int i = 0; i < 6; i++) {
     for (int j = 0; j < 6; j++) {
       distMat[i][j] = euclidean_distance(KFpredictions[i], clusterCenters[j]);
     }
   }
   
   for (int c = 0; c < 6; c++) {
     auto minIndex = findIndexOfMin(distMat);
     objID[minIndex.first] = minIndex.second;
     for (int col = 0; col < 6; col++) {
       distMat[minIndex.first][col] = 1e5;
     }
     for (int row = 0; row < 6; row++) {
       distMat[row][minIndex.second] = 1e5;
     }
   }
   
   visualization_msgs::MarkerArray mkArr;
   mkArr.markers.reserve(6);
   for (int i = 0; i < 6; i++) {
     visualization_msgs::Marker mk;
     mk.header.frame_id = "base_link";
     mk.header.stamp = ros::Time::now();
     mk.id = i;
     mk.type = visualization_msgs::Marker::CUBE;
     mk.action = visualization_msgs::Marker::ADD;
     mk.pose.position.x = KFpredictions[i].x;
     mk.pose.position.y = KFpredictions[i].y;
     mk.pose.position.z = 0.0;
     mk.scale.x = 0.3;
     mk.scale.y = 0.3;
     mk.scale.z = 0.3;
     mk.color.a = 1.0;
     mk.color.r = (i % 2) ? 1 : 0;
     mk.color.g = (i % 3) ? 1 : 0;
     mk.color.b = (i % 4) ? 1 : 0;
     mkArr.markers.push_back(mk);
   }
   markerPub.publish(mkArr);
   
   std_msgs::Int32MultiArray ids;
   for (int i = 0; i < 6; i++) {
     ids.data.push_back(objID[i]);
   }
   objID_pub.publish(ids);
   
   for (int i = 0; i < 6; i++) {
     int cidx = objID[i];
     float meas[2];
     meas[0] = (float)clusterCenters[cidx].x;
     meas[1] = (float)clusterCenters[cidx].y;
     if (fabs(meas[0]) < 1e-6 && fabs(meas[1]) < 1e-6)
       isDummyMeasurement[i] = true;
     else {
       isDummyMeasurement[i] = false;
       cv::Mat measurement(2, 1, CV_32F, meas);
       switch(i) {
         case 0: KF0.correct(measurement); break;
         case 1: KF1.correct(measurement); break;
         case 2: KF2.correct(measurement); break;
         case 3: KF3.correct(measurement); break;
         case 4: KF4.correct(measurement); break;
         case 5: KF5.correct(measurement); break;
         default: break;
       }
     }
   }
 }
  
 // ------------------- 클라우드 퍼블리시 함수 ------------------- //
 void publish_cloud(ros::Publisher &pub, pcl::PointCloud<pcl::PointXYZ>::Ptr cluster)
 {
   sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2);
   pcl::toROSMsg(*cluster, *msg);
   msg->header.frame_id = "base_link";
   msg->header.stamp = ros::Time::now();
   pub.publish(*msg);
 }
  
 // ------------------- 충돌 예측 함수 ------------------- //
 void checkCollision(ros::Publisher &predPathPub)
 {
   double T_total = 5.0;
   double dt_pred = 0.5;
   visualization_msgs::MarkerArray predArr;
   
   for (int i = 0; i < 6; i++) {
     if (isDummyMeasurement[i])
       continue;
     
     double obj_x = prevObjPositionsUTM[i].x;
     double obj_y = prevObjPositionsUTM[i].y;
     double vx = object_vel_x[i];
     double vy = object_vel_y[i];
     
     double obj_speed = sqrt(vx * vx + vy * vy);
     if (obj_speed < static_threshold)
       continue;
     
     bool collisionPossible = false;
     
     visualization_msgs::Marker lineMarker;
     lineMarker.header.frame_id = "gps_utm";
     lineMarker.header.stamp = ros::Time::now();
     lineMarker.ns = "predicted_path";
     lineMarker.id = i;
     lineMarker.type = visualization_msgs::Marker::LINE_STRIP;
     lineMarker.action = visualization_msgs::Marker::ADD;
     lineMarker.scale.x = 0.2;
     lineMarker.lifetime = ros::Duration(0.1);
     lineMarker.color.a = 1.0;
     lineMarker.color.r = 1.0;
     lineMarker.color.g = 0.0;
     lineMarker.color.b = 0.0;
     
     for (double t = 0.0; t <= T_total; t += dt_pred) {
       geometry_msgs::Point objPred;
       objPred.x = obj_x + vx * t;
       objPred.y = obj_y + vy * t;
       objPred.z = 0.0;
       
       geometry_msgs::Point vehPred;
       vehPred.x = vehicle_pose.x + vehicle_vel_x * t;
       vehPred.y = vehicle_pose.y + vehicle_vel_y * t;
       vehPred.z = 0.0;
       
       double dist = sqrt(pow(objPred.x - vehPred.x, 2) + pow(objPred.y - vehPred.y, 2));
       if (dist < collision_threshold)
         collisionPossible = true;
       
       lineMarker.points.push_back(objPred);
     }
     
     if (collisionPossible)
       predArr.markers.push_back(lineMarker);
   }
   
   if (!predArr.markers.empty())
     predPathPub.publish(predArr);
 }
  
 // ------------------- odom 콜백 함수 ------------------- //
 // /odom/coordinate/gps 토픽을 구독하여 차량의 pose 및 속도(heading 고려)를 계산합니다.
 void odom_cb(const nav_msgs::Odometry::ConstPtr &msg)
 {
   vehicle_pose = msg->pose.pose.position;
   
   double yaw = tf::getYaw(msg->pose.pose.orientation);
   
   double linear_speed = msg->twist.twist.linear.x;
   if (linear_speed >= 0 && linear_speed <= 0.5) linear_speed = 0.5;
   if (linear_speed >= -0.5 && linear_speed < 0) linear_speed = -0.5;
   vehicle_vel_x = linear_speed * cos(yaw);
   vehicle_vel_y = linear_speed * sin(yaw);
   
   ROS_INFO("Vehicle: vel_x = %.3f, vel_y = %.3f, heading (rad) = %.3f", vehicle_vel_x, vehicle_vel_y, yaw);
   
   checkCollision(predPathPub);
 }
  
 // ------------------- cloud_cb 콜백 함수 ------------------- //
 void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input)
 {
   if (firstFrame) {
     // 1) Kalman Filter 초기화
     float dx = 1.0f, dy = 1.0f, dvx = 0.01f, dvy = 0.01f;
     cv::Mat T = (Mat_<float>(4,4) << dx, 0, 1, 0,
                                      0, dy, 0, 1,
                                      0, 0, dvx, 0,
                                      0, 0, 0, dvy);
     KF0.transitionMatrix = T.clone();
     KF1.transitionMatrix = T.clone();
     KF2.transitionMatrix = T.clone();
     KF3.transitionMatrix = T.clone();
     KF4.transitionMatrix = T.clone();
     KF5.transitionMatrix = T.clone();
  
     setIdentity(KF0.measurementMatrix);
     setIdentity(KF1.measurementMatrix);
     setIdentity(KF2.measurementMatrix);
     setIdentity(KF3.measurementMatrix);
     setIdentity(KF4.measurementMatrix);
     setIdentity(KF5.measurementMatrix);
  
     float sigmaP = 0.01f;
     float sigmaQ = 0.1f;
     setIdentity(KF0.processNoiseCov, Scalar::all(sigmaP));
     setIdentity(KF1.processNoiseCov, Scalar::all(sigmaP));
     setIdentity(KF2.processNoiseCov, Scalar::all(sigmaP));
     setIdentity(KF3.processNoiseCov, Scalar::all(sigmaP));
     setIdentity(KF4.processNoiseCov, Scalar::all(sigmaP));
     setIdentity(KF5.processNoiseCov, Scalar::all(sigmaP));
  
     setIdentity(KF0.measurementNoiseCov, Scalar::all(sigmaQ));
     setIdentity(KF1.measurementNoiseCov, Scalar::all(sigmaQ));
     setIdentity(KF2.measurementNoiseCov, Scalar::all(sigmaQ));
     setIdentity(KF3.measurementNoiseCov, Scalar::all(sigmaQ));
     setIdentity(KF4.measurementNoiseCov, Scalar::all(sigmaQ));
     setIdentity(KF5.measurementNoiseCov, Scalar::all(sigmaQ));
  
     // 2) PCL 군집 추출
     pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
     pcl::fromROSMsg(*input, *in_cloud);
     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
     tree->setInputCloud(in_cloud);
     std::vector<pcl::PointIndices> cluster_indices;
     pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
     ec.setClusterTolerance(0.2);
     ec.setMinClusterSize(10);
     ec.setMaxClusterSize(600);
     ec.setSearchMethod(tree);
     ec.setInputCloud(in_cloud);
     ec.extract(cluster_indices);
  
     // 3) median으로 클러스터 대표점 계산
     std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_vec;
     std::vector<pcl::PointXYZ> clusterCentroids;
     for (auto &idxs : cluster_indices) {
       pcl::PointCloud<pcl::PointXYZ>::Ptr c(new pcl::PointCloud<pcl::PointXYZ>);
       for (auto id : idxs.indices) {
         c->points.push_back(in_cloud->points[id]);
       }
       pcl::PointXYZ cent = getMedianPoint(c);
       cluster_vec.push_back(c);
       clusterCentroids.push_back(cent);
     }
     while (clusterCentroids.size() < 6) {
       clusterCentroids.push_back({0, 0, 0});
     }
  
     // 4) KF 초기값 설정
     KF0.statePre.at<float>(0) = clusterCentroids[0].x;
     KF0.statePre.at<float>(1) = clusterCentroids[0].y;
     KF1.statePre.at<float>(0) = clusterCentroids[1].x;
     KF1.statePre.at<float>(1) = clusterCentroids[1].y;
     KF2.statePre.at<float>(0) = clusterCentroids[2].x;
     KF2.statePre.at<float>(1) = clusterCentroids[2].y;
     KF3.statePre.at<float>(0) = clusterCentroids[3].x;
     KF3.statePre.at<float>(1) = clusterCentroids[3].y;
     KF4.statePre.at<float>(0) = clusterCentroids[4].x;
     KF4.statePre.at<float>(1) = clusterCentroids[4].y;
     KF5.statePre.at<float>(0) = clusterCentroids[5].x;
     KF5.statePre.at<float>(1) = clusterCentroids[5].y;
  
     for (int i = 0; i < 6; i++) {
       geometry_msgs::Point p;
       p.x = clusterCentroids[i].x;
       p.y = clusterCentroids[i].y;
       p.z = 0.0;
       prevObjPositions[i] = p;
     }
  
     for (int i = 0; i < 6; i++) {
       geometry_msgs::PointStamped base_pt, utm_pt;
       base_pt.header.frame_id = "base_link";
       base_pt.header.stamp = ros::Time(0);
       base_pt.point.x = clusterCentroids[i].x;
       base_pt.point.y = clusterCentroids[i].y;
       base_pt.point.z = 0.0;
       try {
         tfListener->transformPoint("gps_utm", base_pt, utm_pt);
         prevObjPositionsUTM[i].x = utm_pt.point.x;
         prevObjPositionsUTM[i].y = utm_pt.point.y;
         prevObjPositionsUTM[i].z = 0.0;
         world_x_filtered[i] = utm_pt.point.x;
         world_y_filtered[i] = utm_pt.point.y;
       } catch(tf::TransformException &ex) {
         ROS_WARN("TF exception @ init: %s", ex.what());
         prevObjPositionsUTM[i].x = 0.0;
         prevObjPositionsUTM[i].y = 0.0;
         world_x_filtered[i] = 0.0;
         world_y_filtered[i] = 0.0;
       }
     }
     filter_initialized = true;
     firstFrame = false;
     prevTime = ros::Time::now();
     firstTimeStamp = true;
   }
   else {
     // 2번째 프레임 이후
     pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
     pcl::fromROSMsg(*input, *in_cloud);
     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
     tree->setInputCloud(in_cloud);
     std::vector<pcl::PointIndices> cluster_indices;
     pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
     ec.setClusterTolerance(0.2);
     ec.setMinClusterSize(10);
     ec.setMaxClusterSize(600);
     ec.setSearchMethod(tree);
     ec.setInputCloud(in_cloud);
     ec.extract(cluster_indices);
  
     std::vector<pcl::PointXYZ> centroids;
     centroids.reserve(6);
     std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_vec;
     for (auto &idxs : cluster_indices) {
       pcl::PointCloud<pcl::PointXYZ>::Ptr c(new pcl::PointCloud<pcl::PointXYZ>);
       for (auto id : idxs.indices) {
         c->points.push_back(in_cloud->points[id]);
       }
       pcl::PointXYZ cc = getMedianPoint(c);
       cluster_vec.push_back(c);
       centroids.push_back(cc);
     }
     while (cluster_vec.size() < 6) {
       pcl::PointCloud<pcl::PointXYZ>::Ptr e(new pcl::PointCloud<pcl::PointXYZ>);
       e->points.push_back(pcl::PointXYZ(0, 0, 0));
       cluster_vec.push_back(e);
     }
     while (centroids.size() < 6) {
       pcl::PointXYZ cc;
       cc.x = 0; cc.y = 0; cc.z = 0;
       centroids.push_back(cc);
     }
  
     std_msgs::Float32MultiArray ccMsg;
     for (int i = 0; i < 6; i++) {
       ccMsg.data.push_back(centroids[i].x);
       ccMsg.data.push_back(centroids[i].y);
       ccMsg.data.push_back(centroids[i].z);
     }
     KFT(ccMsg);
  
     visualization_msgs::MarkerArray bboxArr;
     bboxArr.markers.reserve(6);
     for (int i = 0; i < 6; i++) {
       int cid = objID[i];
       publish_cloud(
         ( i == 0 ? pub_cluster0 :
           i == 1 ? pub_cluster1 :
           i == 2 ? pub_cluster2 :
           i == 3 ? pub_cluster3 :
           i == 4 ? pub_cluster4 :
                    pub_cluster5 ),
         cluster_vec[cid]
       );
       geometry_msgs::Pose obbPose;
       geometry_msgs::Vector3 obbScale;
       bool validBox = computeOrientedBoundingBox(cluster_vec[cid], obbPose, obbScale);
       if (validBox) {
         visualization_msgs::Marker bboxMarker;
         bboxMarker.header.frame_id = "base_link";
         bboxMarker.header.stamp = ros::Time::now();
         bboxMarker.ns = "tracked_bounding_boxes";
         bboxMarker.id = i;
         bboxMarker.type = visualization_msgs::Marker::CUBE;
         bboxMarker.action = visualization_msgs::Marker::ADD;
         bboxMarker.pose = obbPose;
         bboxMarker.scale.x = obbScale.x;
         bboxMarker.scale.y = obbScale.y;
         bboxMarker.scale.z = (obbScale.z > 0.01 ? obbScale.z : 0.01);
         bboxMarker.color.a = 0.4;
         bboxMarker.color.r = 1.0;
         bboxMarker.color.g = 0.0;
         bboxMarker.color.b = 0.0;
         bboxArr.markers.push_back(bboxMarker);
       }
     }
     bboxPub.publish(bboxArr);
  
     ros::Time nowT = ros::Time::now();
     double dt = (nowT - prevTime).toSec();
     visualization_msgs::MarkerArray arrowArr;
     arrowArr.markers.reserve(6);
  
     if (firstTimeStamp) {
       firstTimeStamp = false;
       prevTime = nowT;
       for (int i = 0; i < 6; i++) {
         float x_now = (i == 0 ? KF0.statePost.at<float>(0) :
                        i == 1 ? KF1.statePost.at<float>(0) :
                        i == 2 ? KF2.statePost.at<float>(0) :
                        i == 3 ? KF3.statePost.at<float>(0) :
                        i == 4 ? KF4.statePost.at<float>(0) :
                                 KF5.statePost.at<float>(0));
         float y_now = (i == 0 ? KF0.statePost.at<float>(1) :
                        i == 1 ? KF1.statePost.at<float>(1) :
                        i == 2 ? KF2.statePost.at<float>(1) :
                        i == 3 ? KF3.statePost.at<float>(1) :
                        i == 4 ? KF4.statePost.at<float>(1) :
                                 KF5.statePost.at<float>(1));
         prevObjPositions[i].x = x_now;
         prevObjPositions[i].y = y_now;
         prevObjPositions[i].z = 0.0;
  
         geometry_msgs::PointStamped base_pt, utm_pt;
         base_pt.header.frame_id = "base_link";
         base_pt.header.stamp = ros::Time(0);
         base_pt.point.x = x_now;
         base_pt.point.y = y_now;
         base_pt.point.z = 0.0;
         try {
           tfListener->transformPoint("gps_utm", base_pt, utm_pt);
           world_x_filtered[i] = utm_pt.point.x;
           world_y_filtered[i] = utm_pt.point.y;
           prevObjPositionsUTM[i].x = world_x_filtered[i];
           prevObjPositionsUTM[i].y = world_y_filtered[i];
         } catch(tf::TransformException &ex) {
           ROS_WARN("TF exc firstTimeStamp: %s", ex.what());
           prevObjPositionsUTM[i].x = 0;
           prevObjPositionsUTM[i].y = 0;
         }
       }
       filter_initialized = true;
     } else {
       if (dt > 1e-6) {
         ROS_INFO("=============================================================");
         for (int i = 0; i < 6; i++) {
           float x_now = (i == 0 ? KF0.statePost.at<float>(0) :
                          i == 1 ? KF1.statePost.at<float>(0) :
                          i == 2 ? KF2.statePost.at<float>(0) :
                          i == 3 ? KF3.statePost.at<float>(0) :
                          i == 4 ? KF4.statePost.at<float>(0) :
                                   KF5.statePost.at<float>(0));
           float y_now = (i == 0 ? KF0.statePost.at<float>(1) :
                          i == 1 ? KF1.statePost.at<float>(1) :
                          i == 2 ? KF2.statePost.at<float>(1) :
                          i == 3 ? KF3.statePost.at<float>(1) :
                          i == 4 ? KF4.statePost.at<float>(1) :
                                   KF5.statePost.at<float>(1));
           bool isDummy = isDummyMeasurement[i];
   
           float x_prev_b = prevObjPositions[i].x;
           float y_prev_b = prevObjPositions[i].y;
           float vx_b = 0.0f, vy_b = 0.0f;
           if (!isDummy && dt > 1e-6) {
             vx_b = (x_now - x_prev_b) / dt;
             vy_b = (y_now - y_prev_b) / dt;
           }
           prevObjPositions[i].x = x_now;
           prevObjPositions[i].y = y_now;
   
           geometry_msgs::PointStamped base_pt, utm_pt;
           base_pt.header.frame_id = "base_link";
           base_pt.header.stamp = ros::Time(0);
           base_pt.point.x = x_now;
           base_pt.point.y = y_now;
           base_pt.point.z = 0.0;
           double x_now_u = 0.0, y_now_u = 0.0;
           try {
             tfListener->transformPoint("gps_utm", base_pt, utm_pt);
             x_now_u = utm_pt.point.x;
             y_now_u = utm_pt.point.y;
           } catch(tf::TransformException &ex) {
             ROS_WARN("TF exc: %s", ex.what());
           }
   
           if (filter_initialized) {
             if (!isDummy) {
               if (fabs(x_now_u - world_x_filtered[i]) > reset_threshold ||
                   fabs(y_now_u - world_y_filtered[i]) > reset_threshold) {
                 world_x_filtered[i] = x_now_u;
                 world_y_filtered[i] = y_now_u;
               } else {
                 world_x_filtered[i] = alpha * x_now_u + (1.0 - alpha) * world_x_filtered[i];
                 world_y_filtered[i] = alpha * y_now_u + (1.0 - alpha) * world_y_filtered[i];
               }
             }
           } else {
             world_x_filtered[i] = x_now_u;
             world_y_filtered[i] = y_now_u;
           }
   
           double x_f = world_x_filtered[i];
           double y_f = world_y_filtered[i];
   
           double x_prev_u = prevObjPositionsUTM[i].x;
           double y_prev_u = prevObjPositionsUTM[i].y;
           double vx_u = 0.0, vy_u = 0.0;
           // ★ 여기서 현재 트랙의 객체 ID와 이전 객체 ID를 비교하여 전환 시 속도 초기화
           int currentObjID = objID[i];
           if (prevObjID[i] != currentObjID) {
             // 트랙이 다른 객체로 전환되었으므로 속도를 초기화
             object_vel_x[i] = 0;
             object_vel_y[i] = 0;
             prevObjPositionsUTM[i].x = x_f;
             prevObjPositionsUTM[i].y = y_f;
             prevObjID[i] = currentObjID;
             // (원한다면 이 프레임에 대해 arrow 표시를 생략하거나, 0 속도의 화살표를 표시)
             continue;
           }
   
           if (!isDummy && dt > 1e-6) {
             vx_u = (x_f - x_prev_u) / dt;
             vy_u = (y_f - y_prev_u) / dt;
           }
   
           object_vel_x[i] = vx_u;
           object_vel_y[i] = vy_u;
   
           double heading_rad = 0.0, speed = 0.0;
           if (!isDummy) {
             heading_rad = std::atan2(vy_u, vx_u);
             speed = std::sqrt(vx_u * vx_u + vy_u * vy_u);
           }
   
           prevObjPositionsUTM[i].x = x_f;
           prevObjPositionsUTM[i].y = y_f;
   
           ROS_INFO("[Object %d] dt=%.2f s, (isDummy=%d) FilteredPos=(%.2f, %.2f), vel=(%.2f, %.2f), speed=%.2f",
                    i, dt, (int)isDummy, x_f, y_f, vx_u, vy_u, speed);
   
           visualization_msgs::Marker arrow;
           arrow.header.frame_id = "gps_utm";
           arrow.header.stamp = nowT;
           arrow.ns = "heading_arrow";
           arrow.id = i;
           arrow.type = visualization_msgs::Marker::ARROW;
           arrow.action = visualization_msgs::Marker::ADD;
           arrow.pose.position.x = x_f;
           arrow.pose.position.y = y_f;
           arrow.pose.position.z = 0.0;
           arrow.scale.x = 1.0;
           arrow.scale.y = 0.2;
           arrow.scale.z = 0.2;
           tf::Quaternion q;
           q.setRPY(0.0, 0.0, heading_rad);
           arrow.pose.orientation.x = q.x();
           arrow.pose.orientation.y = q.y();
           arrow.pose.orientation.z = q.z();
           arrow.pose.orientation.w = q.w();
           arrow.color.a = 1.0;
           if (speed > static_threshold) {
             arrow.color.r = 0.0;
             arrow.color.g = 1.0;
             arrow.color.b = 0.0;
           } else {
             arrow.color.r = 1.0;
             arrow.color.g = 0.0;
             arrow.color.b = 0.0;
           }
           if (isDummy)
             arrow.color.a = 0.0;
           arrowArr.markers.push_back(arrow);
         }
       }
       filter_initialized = true;
       prevTime = nowT;
     }
     arrowPub.publish(arrowArr);
   }
   
   // 충돌 예측은 odom_cb에서 호출하도록 함.
 }
   
 // ------------------- main 함수 ------------------- //
 int main(int argc, char** argv)
 {
   ros::init(argc, argv, "kf_tracker_world");
   ros::NodeHandle nh;
   tfListener = new tf::TransformListener();
   
   ros::Subscriber subCloud = nh.subscribe("filtered_cloud", 1, cloud_cb);
   ros::Subscriber subOdom  = nh.subscribe("/odom/coordinate/gps", 1, odom_cb);
   
   pub_cluster0 = nh.advertise<sensor_msgs::PointCloud2>("cluster_0", 1);
   pub_cluster1 = nh.advertise<sensor_msgs::PointCloud2>("cluster_1", 1);
   pub_cluster2 = nh.advertise<sensor_msgs::PointCloud2>("cluster_2", 1);
   pub_cluster3 = nh.advertise<sensor_msgs::PointCloud2>("cluster_3", 1);
   pub_cluster4 = nh.advertise<sensor_msgs::PointCloud2>("cluster_4", 1);
   pub_cluster5 = nh.advertise<sensor_msgs::PointCloud2>("cluster_5", 1);
   
   objID_pub = nh.advertise<std_msgs::Int32MultiArray>("obj_id", 1);
   markerPub = nh.advertise<visualization_msgs::MarkerArray>("viz", 1);
   arrowPub  = nh.advertise<visualization_msgs::MarkerArray>("heading_arrows", 1);
   bboxPub   = nh.advertise<visualization_msgs::MarkerArray>("tracked_bboxes", 1);
   predPathPub = nh.advertise<visualization_msgs::MarkerArray>("predicted_collision_path", 1);
   
   ros::spin();
   delete tfListener;
   return 0;
 }
 