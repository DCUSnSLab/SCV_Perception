cmake_minimum_required(VERSION 3.16)
project(traversability_grid LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_executable(traversability_node
  src/traversability_node.cpp
  src/elevation_grid.cpp)

target_compile_features(traversability_node PUBLIC cxx_std_17)
target_include_directories(traversability_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

ament_target_dependencies(
  traversability_node
  rclcpp
  sensor_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs)

install(TARGETS traversability_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
