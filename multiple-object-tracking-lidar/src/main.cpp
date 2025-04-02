/**********************************************
 * median_based_kf_tracker_with_bbox.cpp
 * 중앙값(median)으로 군집 중심 계산 + 저역통과필터 + 칼만필터 + Bounding Box
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
 
 // TF 변환 (base_link→gps_utm)
 #include <tf/transform_listener.h>
 #include <geometry_msgs/PointStamped.h>
 
 // Heading 계산 시 orientation 변환
 #include <tf/LinearMath/Quaternion.h>
 #include <tf/LinearMath/Matrix3x3.h>
 
 using namespace std;
 using namespace cv;
 
 // ------------------- 전역 변수/객체 ------------------- //
 
 // 최초 프레임 플래그
 static bool firstFrame = true;
 
 // 퍼블리셔
 static ros::Publisher objID_pub;
 static ros::Publisher pub_cluster0;
 static ros::Publisher pub_cluster1;
 static ros::Publisher pub_cluster2;
 static ros::Publisher pub_cluster3;
 static ros::Publisher pub_cluster4;
 static ros::Publisher pub_cluster5;
 static ros::Publisher markerPub;       // CUBE 마커(트래킹된 위치)
 static ros::Publisher arrowPub;        // ARROW 마커(Heading)
 static ros::Publisher bboxPub;         // BBOX 마커(실제 클러스터)
 
 // 칼만 필터(최대 6개 객체)
 static cv::KalmanFilter KF0(4,2,0), KF1(4,2,0), KF2(4,2,0),
                         KF3(4,2,0), KF4(4,2,0), KF5(4,2,0);
 
 // 매칭된 객체 ID
 static std::vector<int> objID(6);
 
 // 이전 프레임 시간 & 초기화용
 static ros::Time prevTime;
 static bool firstTimeStamp = true;
 
 // base_link에서 이전 위치
 static std::vector<geometry_msgs::Point> prevObjPositions(6);
 
 // gps_utm에서 이전 위치
 static std::vector<geometry_msgs::Point> prevObjPositionsUTM(6);
 
 // 저역 통과 필터
 static std::vector<double> world_x_filtered(6, 0.0);
 static std::vector<double> world_y_filtered(6, 0.0);
 static bool filter_initialized = false;
 static const double alpha = 0.2; // 0<alpha<1
 
 // TF
 static tf::TransformListener* tfListener = nullptr;
 
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
 std::pair<int,int> findIndexOfMin(const std::vector<std::vector<float>> &distMat)
 {
   std::pair<int,int> minIndex;
   float minVal = std::numeric_limits<float>::max();
 
   for(int i=0; i<(int)distMat.size(); i++){
     for(int j=0; j<(int)distMat[i].size(); j++){
       if(distMat[i][j]<minVal){
         minVal = distMat[i][j];
         minIndex = std::make_pair(i,j);
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
 
   for(auto &pt : cluster->points){
     xs.push_back(pt.x);
     ys.push_back(pt.y);
   }
 
   pcl::PointXYZ medianPt;
   if(xs.empty()){
     medianPt.x=0; 
     medianPt.y=0; 
     medianPt.z=0;
     return medianPt;
   }
   // 중앙값 index
   size_t mid = xs.size()/2;
 
   // nth_element
   std::nth_element(xs.begin(), xs.begin()+mid, xs.end());
   std::nth_element(ys.begin(), ys.begin()+mid, ys.end());
 
   medianPt.x = xs[mid];
   medianPt.y = ys[mid];
   medianPt.z = 0.0f;
   return medianPt;
 }
 
 // ------------------- 바운딩박스 (Axis-Aligned) 계산 함수 ------------------- //
 /** 
  * @brief cluster(PointCloud)에 대해 min_x,max_x, min_y,max_y, min_z,max_z를 구해 
  *        바운딩 박스 중심(center)과 스케일(scale)을 반환
  */
 bool computeBoundingBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster,
                         geometry_msgs::Point &boxCenter,
                         geometry_msgs::Point &boxScale)
 {
   if(cluster->points.empty()){
     return false;
   }
 
   float min_x = std::numeric_limits<float>::max();
   float max_x = -std::numeric_limits<float>::max();
   float min_y = std::numeric_limits<float>::max();
   float max_y = -std::numeric_limits<float>::max();
   float min_z = std::numeric_limits<float>::max();
   float max_z = -std::numeric_limits<float>::max();
 
   for(const auto &p : cluster->points)
   {
     if(p.x < min_x) min_x = p.x;
     if(p.x > max_x) max_x = p.x;
 
     if(p.y < min_y) min_y = p.y;
     if(p.y > max_y) max_y = p.y;
 
     if(p.z < min_z) min_z = p.z;
     if(p.z > max_z) max_z = p.z;
   }
 
   boxCenter.x = 0.5f*(min_x + max_x);
   boxCenter.y = 0.5f*(min_y + max_y);
   boxCenter.z = 0.5f*(min_z + max_z);
 
   boxScale.x = (max_x - min_x);
   boxScale.y = (max_y - min_y);
   boxScale.z = (max_z - min_z);
 
   return true;
 }
 
 // ------------------- 칼만필터 업데이트 함수 ------------------- //
 void KFT(const std_msgs::Float32MultiArray &ccs)
 {
   // 1) 예측
   std::vector<cv::Mat> pred{
     KF0.predict(),KF1.predict(),KF2.predict(),
     KF3.predict(),KF4.predict(),KF5.predict()
   };
 
   // 2) 측정값 -> clusterCenters
   std::vector<geometry_msgs::Point> clusterCenters;
   clusterCenters.reserve(6);
 
   for(size_t i=0; i<ccs.data.size(); i+=3){
     geometry_msgs::Point p;
     p.x = ccs.data[i];
     p.y = ccs.data[i+1];
     p.z = ccs.data[i+2];
     clusterCenters.push_back(p);
   }
 
   // 3) KF 예측 위치
   std::vector<geometry_msgs::Point> KFpredictions;
   KFpredictions.reserve(6);
 
   for(int i=0; i<6; i++){
     geometry_msgs::Point pt;
     pt.x = pred[i].at<float>(0);
     pt.y = pred[i].at<float>(1);
     pt.z = 0.0;
     KFpredictions.push_back(pt);
   }
 
   // 4) distMat
   std::vector<std::vector<float>> distMat(6, std::vector<float>(6,0.0f));
   for(int i=0; i<6; i++){
     for(int j=0; j<6; j++){
       distMat[i][j] = euclidean_distance(KFpredictions[i], clusterCenters[j]);
     }
   }
 
   // 5) 최소값 -> objID
   for(int c=0; c<6; c++){
     auto minIndex = findIndexOfMin(distMat);
     objID[minIndex.first] = minIndex.second;
     // 해당 row/col은 큰 값으로
     for(int col=0; col<6; col++){
       distMat[minIndex.first][col] = 1e5;
     }
     for(int row=0; row<6; row++){
       distMat[row][minIndex.second] = 1e5;
     }
   }
 
   // 6) base_link 마커 시각화 (CUBE: 칼만필터 예측 위치)
   visualization_msgs::MarkerArray mkArr;
   mkArr.markers.reserve(6);
 
   for(int i=0; i<6; i++){
     visualization_msgs::Marker mk;
     mk.header.frame_id = "base_link";
     mk.header.stamp    = ros::Time::now();
     mk.id = i;
     mk.type = visualization_msgs::Marker::CUBE;
     mk.action = visualization_msgs::Marker::ADD;
     mk.pose.position.x = KFpredictions[i].x;
     mk.pose.position.y = KFpredictions[i].y;
     mk.pose.position.z = 0.0;
 
     // 트래킹 "위치"만 간단히 표시
     mk.scale.x = 0.3;
     mk.scale.y = 0.3;
     mk.scale.z = 0.3;
 
     mk.color.a = 1.0;
     mk.color.r = (i%2)?1:0;
     mk.color.g = (i%3)?1:0;
     mk.color.b = (i%4)?1:0;
 
     mkArr.markers.push_back(mk);
   }
   markerPub.publish(mkArr);
 
   // 7) obj_id 퍼블리시
   std_msgs::Int32MultiArray ids;
   for(int i=0; i<6; i++){
     ids.data.push_back(objID[i]);
   }
   objID_pub.publish(ids);
 
   // 8) 측정값 -> correct
   for(int i=0; i<6; i++){
     int cidx = objID[i];
     float meas[2];
     meas[0] = (float)clusterCenters[cidx].x;
     meas[1] = (float)clusterCenters[cidx].y;
 
     if(!(meas[0]==0.f && meas[1]==0.f)){
       cv::Mat measurement(2,1,CV_32F, meas);
       switch(i){
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
 
 // ------------------- 클라우드 퍼블리시 ------------------- //
 void publish_cloud(ros::Publisher &pub, pcl::PointCloud<pcl::PointXYZ>::Ptr cluster)
 {
   sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2);
   pcl::toROSMsg(*cluster, *msg);
   msg->header.frame_id = "base_link";
   msg->header.stamp    = ros::Time::now();
   pub.publish(*msg);
 }
 
 // ------------------- cloud_cb ------------------- //
 void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input)
 {
   if(firstFrame){
     // 1) Kalman Filter 초기화 (transitionMatrix, measurementMatrix...)
     float dx=1.0f, dy=1.0f, dvx=0.01f, dvy=0.01f;
     cv::Mat T = (Mat_<float>(4,4) << dx,0, 1,0,
                                      0,dy, 0,1,
                                      0,0, dvx,0,
                                      0,0, 0,dvy);
 
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
     ec.setClusterTolerance(0.08);
     ec.setMinClusterSize(10);
     ec.setMaxClusterSize(600);
     ec.setSearchMethod(tree);
     ec.setInputCloud(in_cloud);
     ec.extract(cluster_indices);
 
     // 3) median으로 클러스터 대표점
     std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_vec;
     std::vector<pcl::PointXYZ> clusterCentroids;
 
     for(auto &idxs : cluster_indices){
       pcl::PointCloud<pcl::PointXYZ>::Ptr c(new pcl::PointCloud<pcl::PointXYZ>);
       for(auto id : idxs.indices){
         c->points.push_back(in_cloud->points[id]);
       }
       // 중앙값 계산
       pcl::PointXYZ cent = getMedianPoint(c);
       cluster_vec.push_back(c);
       clusterCentroids.push_back(cent);
     }
     // 6개 미만이면 채우기
     while(clusterCentroids.size()<6){
       clusterCentroids.push_back({0,0,0});
     }
 
     // 4) KF 초기값
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
 
     // base_link 이전 위치
     for(int i=0; i<6; i++){
       geometry_msgs::Point p;
       p.x = clusterCentroids[i].x;
       p.y = clusterCentroids[i].y;
       p.z = 0.0;
       prevObjPositions[i] = p;
     }
 
     // gps_utm 초기화
     {
       for(int i=0; i<6; i++){
         geometry_msgs::PointStamped base_pt, utm_pt;
         base_pt.header.frame_id = "base_link";
         base_pt.header.stamp    = ros::Time(0);
         base_pt.point.x         = clusterCentroids[i].x;
         base_pt.point.y         = clusterCentroids[i].y;
         base_pt.point.z         = 0.0;
 
         try{
           tfListener->transformPoint("gps_utm", base_pt, utm_pt);
           prevObjPositionsUTM[i].x = utm_pt.point.x;
           prevObjPositionsUTM[i].y = utm_pt.point.y;
           prevObjPositionsUTM[i].z = 0.0;
 
           // 저역 통과 필터 초기화
           world_x_filtered[i] = utm_pt.point.x;
           world_y_filtered[i] = utm_pt.point.y;
 
         }catch(tf::TransformException &ex){
           ROS_WARN("TF exception @ init: %s", ex.what());
           prevObjPositionsUTM[i].x = 0.0;
           prevObjPositionsUTM[i].y = 0.0;
           world_x_filtered[i] = 0.0;
           world_y_filtered[i] = 0.0;
         }
       }
       filter_initialized = true;
     }
 
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
     ec.setClusterTolerance(0.08);
     ec.setMinClusterSize(10);
     ec.setMaxClusterSize(600);
     ec.setSearchMethod(tree);
     ec.setInputCloud(in_cloud);
     ec.extract(cluster_indices);
 
     // median 중심
     std::vector<pcl::PointXYZ> centroids;
     centroids.reserve(6);
     std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_vec;
 
     for(auto &idxs : cluster_indices){
       pcl::PointCloud<pcl::PointXYZ>::Ptr c(new pcl::PointCloud<pcl::PointXYZ>);
       for(auto id : idxs.indices){
         c->points.push_back(in_cloud->points[id]);
       }
       // 중앙값
       pcl::PointXYZ cc = getMedianPoint(c);
       cluster_vec.push_back(c);
       centroids.push_back(cc);
     }
 
     while(cluster_vec.size()<6){
       pcl::PointCloud<pcl::PointXYZ>::Ptr e(new pcl::PointCloud<pcl::PointXYZ>);
       e->points.push_back(pcl::PointXYZ(0,0,0));
       cluster_vec.push_back(e);
     }
     while(centroids.size()<6){
       pcl::PointXYZ cc;
       cc.x=0; cc.y=0; cc.z=0;
       centroids.push_back(cc);
     }
 
     // Kalman Filter update
     std_msgs::Float32MultiArray ccMsg;
     for(int i=0; i<6; i++){
       ccMsg.data.push_back(centroids[i].x);
       ccMsg.data.push_back(centroids[i].y);
       ccMsg.data.push_back(centroids[i].z);
     }
     KFT(ccMsg);
 
     // 클러스터 퍼블리시 + 바운딩박스 퍼블리시
     //  (objID[i]로 매칭된 클러스터를 각 pub_cluster(i)에 보냄)
     //  그리고 BBox Marker 생성
     visualization_msgs::MarkerArray bboxArr;
     bboxArr.markers.reserve(6);
 
     for(int i=0; i<6; i++){
       int cid = objID[i];
 
       // (A) 각 클러스터 퍼블리시
       publish_cloud(
         ( i==0? pub_cluster0 :
           i==1? pub_cluster1 :
           i==2? pub_cluster2 :
           i==3? pub_cluster3 :
           i==4? pub_cluster4 :
                 pub_cluster5 ),
         cluster_vec[cid]
       );
 
       // (B) 바운딩박스 계산 & 마커 (base_link에서 축 정렬)
       geometry_msgs::Point boxCenter, boxScale;
       bool validBox = computeBoundingBox(cluster_vec[cid], boxCenter, boxScale);
       if(validBox){
         // 바운딩 박스 마커
         visualization_msgs::Marker bboxMarker;
         bboxMarker.header.frame_id = "base_link";
         bboxMarker.header.stamp    = ros::Time::now();
         bboxMarker.ns = "tracked_bounding_boxes";
         bboxMarker.id = i;  // 트래킹ID 별로 표시
         bboxMarker.type   = visualization_msgs::Marker::CUBE;
         bboxMarker.action = visualization_msgs::Marker::ADD;
 
         bboxMarker.pose.position.x = boxCenter.x;
         bboxMarker.pose.position.y = boxCenter.y;
         bboxMarker.pose.position.z = boxCenter.z;
 
         bboxMarker.scale.x = boxScale.x;
         bboxMarker.scale.y = boxScale.y;
         bboxMarker.scale.z = (boxScale.z > 0.01? boxScale.z : 0.01); // 최소 높이 예외처리
 
         bboxMarker.color.a = 0.4; // 좀 투명하게
         bboxMarker.color.r = 1.0; // 빨강 박스 (원하시면 색상 변경)
         bboxMarker.color.g = 0.0;
         bboxMarker.color.b = 0.0;
 
         bboxArr.markers.push_back(bboxMarker);
       }
     }
     // 바운딩박스 전체 퍼블리시
     bboxPub.publish(bboxArr);
 
     // 월드 좌표계에서 속도/heading
     ros::Time nowT = ros::Time::now();
     double dt = (nowT - prevTime).toSec();
 
     // ARROW MarkerArray
     visualization_msgs::MarkerArray arrowArr;
     arrowArr.markers.reserve(6);
 
     if(firstTimeStamp){
       // 첫 속도 계산은 skip
       firstTimeStamp = false;
       prevTime = nowT;
 
       for(int i=0; i<6; i++){
         float x_now = (i==0? KF0.statePost.at<float>(0) :
                        i==1? KF1.statePost.at<float>(0) :
                        i==2? KF2.statePost.at<float>(0) :
                        i==3? KF3.statePost.at<float>(0) :
                        i==4? KF4.statePost.at<float>(0) :
                              KF5.statePost.at<float>(0));
         float y_now = (i==0? KF0.statePost.at<float>(1) :
                        i==1? KF1.statePost.at<float>(1) :
                        i==2? KF2.statePost.at<float>(1) :
                        i==3? KF3.statePost.at<float>(1) :
                        i==4? KF4.statePost.at<float>(1) :
                              KF5.statePost.at<float>(1));
 
         prevObjPositions[i].x = x_now;
         prevObjPositions[i].y = y_now;
         prevObjPositions[i].z = 0.0;
 
         geometry_msgs::PointStamped base_pt, utm_pt;
         base_pt.header.frame_id = "base_link";
         base_pt.header.stamp    = ros::Time(0);
         base_pt.point.x = x_now;
         base_pt.point.y = y_now;
         base_pt.point.z = 0.0;
 
         try{
           tfListener->transformPoint("gps_utm", base_pt, utm_pt);
 
           if(filter_initialized){
             world_x_filtered[i] = alpha*utm_pt.point.x + (1.0-alpha)*world_x_filtered[i];
             world_y_filtered[i] = alpha*utm_pt.point.y + (1.0-alpha)*world_y_filtered[i];
           } else {
             world_x_filtered[i] = utm_pt.point.x;
             world_y_filtered[i] = utm_pt.point.y;
           }
 
           prevObjPositionsUTM[i].x = world_x_filtered[i];
           prevObjPositionsUTM[i].y = world_y_filtered[i];
         }catch(tf::TransformException &ex){
           ROS_WARN("TF exc firstTimeStamp: %s", ex.what());
           prevObjPositionsUTM[i].x=0;
           prevObjPositionsUTM[i].y=0;
         }
       }
       filter_initialized = true;
     }
     else {
       if(dt>1e-6){
         for(int i=0; i<6; i++){
           float x_now = (i==0? KF0.statePost.at<float>(0) :
                          i==1? KF1.statePost.at<float>(0) :
                          i==2? KF2.statePost.at<float>(0) :
                          i==3? KF3.statePost.at<float>(0) :
                          i==4? KF4.statePost.at<float>(0) :
                                KF5.statePost.at<float>(0));
           float y_now = (i==0? KF0.statePost.at<float>(1) :
                          i==1? KF1.statePost.at<float>(1) :
                          i==2? KF2.statePost.at<float>(1) :
                          i==3? KF3.statePost.at<float>(1) :
                          i==4? KF4.statePost.at<float>(1) :
                                KF5.statePost.at<float>(1));
 
           float x_prev_b = prevObjPositions[i].x;
           float y_prev_b = prevObjPositions[i].y;
 
           float vx_b = (x_now - x_prev_b)/dt;
           float vy_b = (y_now - y_prev_b)/dt;
 
           // update prevObjPositions (base_link)
           prevObjPositions[i].x = x_now;
           prevObjPositions[i].y = y_now;
 
           // base_link -> gps_utm
           geometry_msgs::PointStamped base_pt, utm_pt;
           base_pt.header.frame_id = "base_link";
           base_pt.header.stamp    = ros::Time(0);
           base_pt.point.x = x_now;
           base_pt.point.y = y_now;
           base_pt.point.z = 0.0;
 
           double x_now_u=0.0, y_now_u=0.0;
           try{
             tfListener->transformPoint("gps_utm", base_pt, utm_pt);
             x_now_u = utm_pt.point.x;
             y_now_u = utm_pt.point.y;
           }catch(tf::TransformException &ex){
             ROS_WARN("TF exc: %s", ex.what());
           }
 
           // 저역 통과 필터
           if(filter_initialized){
             world_x_filtered[i] = alpha*x_now_u + (1-alpha)*world_x_filtered[i];
             world_y_filtered[i] = alpha*y_now_u + (1-alpha)*world_y_filtered[i];
           } else {
             world_x_filtered[i] = x_now_u;
             world_y_filtered[i] = y_now_u;
           }
 
           double x_f = world_x_filtered[i];
           double y_f = world_y_filtered[i];
 
           // 속도 계산
           double x_prev_u = prevObjPositionsUTM[i].x;
           double y_prev_u = prevObjPositionsUTM[i].y;
 
           double vx_u=0.0, vy_u=0.0;
           if(dt>1e-6){
             vx_u = (x_f - x_prev_u)/dt;
             vy_u = (y_f - y_prev_u)/dt;
           }
           double heading_rad = std::atan2(vy_u, vx_u);
           double heading_deg = heading_rad * 180.0/M_PI;
 
           double speed = std::sqrt(vx_u*vx_u + vy_u*vy_u);
 
           // update
           prevObjPositionsUTM[i].x = x_f;
           prevObjPositionsUTM[i].y = y_f;
 
           ROS_INFO("[Object Tracker] dt=%.2f s, FilteredPos=(%.2f, %.2f), Vel=(%.2f, %.2f), Heading=%.2f deg, Speed=%.2f",
                    dt, x_f, y_f, vx_u, vy_u, heading_deg, speed);
 
           // ARROW Marker
           visualization_msgs::Marker arrow;
           arrow.header.frame_id = "gps_utm";
           arrow.header.stamp    = nowT;
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
           if(speed>0.5){
             arrow.color.r = 0.0;
             arrow.color.g = 1.0;
             arrow.color.b = 0.0;
           } else {
             arrow.color.r = 1.0;
             arrow.color.g = 0.0;
             arrow.color.b = 0.0;
           }
           arrowArr.markers.push_back(arrow);
         }
       }
       filter_initialized = true;
       prevTime = nowT;
     }
 
     arrowPub.publish(arrowArr);
   }
 }
 
 int main(int argc, char** argv)
 {
   ros::init(argc, argv, "kf_tracker_world");
   ros::NodeHandle nh;
 
   tfListener = new tf::TransformListener();
 
   // 구독
   ros::Subscriber sub = nh.subscribe("filtered_cloud", 1, cloud_cb);
 
   // 퍼블리셔
   pub_cluster0 = nh.advertise<sensor_msgs::PointCloud2>("cluster_0",1);
   pub_cluster1 = nh.advertise<sensor_msgs::PointCloud2>("cluster_1",1);
   pub_cluster2 = nh.advertise<sensor_msgs::PointCloud2>("cluster_2",1);
   pub_cluster3 = nh.advertise<sensor_msgs::PointCloud2>("cluster_3",1);
   pub_cluster4 = nh.advertise<sensor_msgs::PointCloud2>("cluster_4",1);
   pub_cluster5 = nh.advertise<sensor_msgs::PointCloud2>("cluster_5",1);
 
   objID_pub = nh.advertise<std_msgs::Int32MultiArray>("obj_id",1);
   markerPub = nh.advertise<visualization_msgs::MarkerArray>("viz",1);
   arrowPub  = nh.advertise<visualization_msgs::MarkerArray>("heading_arrows",1);
 
   // 새로 추가: 바운딩박스 퍼블리셔
   bboxPub   = nh.advertise<visualization_msgs::MarkerArray>("tracked_bboxes",1);
 
   ros::spin();
 
   delete tfListener;
   return 0;
 }
 