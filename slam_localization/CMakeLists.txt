cmake_minimum_required(VERSION 3.0.2)
project(slam_localization)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
)

find_package(PCL 1.8 REQUIRED COMPONENTS io)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

add_executable(publish_pcd src/publish_pcd.cpp)
add_dependencies(publish_pcd ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(publish_pcd
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
