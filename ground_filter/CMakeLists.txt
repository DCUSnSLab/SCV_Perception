cmake_minimum_required(VERSION 3.16)
project(ground_filter LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)

add_executable(ground_filter_node
  src/ground_filter_node.cpp
  src/grid_ground_filter.cpp)

target_compile_features(ground_filter_node PUBLIC cxx_std_17)
target_include_directories(ground_filter_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

ament_target_dependencies(
  ground_filter_node
  rclcpp
  sensor_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs)

install(TARGETS ground_filter_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
