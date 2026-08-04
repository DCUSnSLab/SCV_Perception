cmake_minimum_required(VERSION 3.8)
project(lane_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  lane_detection/lane_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME lane_node
)

install(PROGRAMS
  lane_detection/lane_node_yolo26.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME lane_node_yolo26
)

install(PROGRAMS
  lane_detection/lane_node_yolop.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME lane_node_yolop
)

install(PROGRAMS
  lane_detection/lane_node_yolo26_overlay.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME lane_node_yolo26_overlay
)

install(PROGRAMS
  lane_detection/lane_bev_cloud_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME lane_bev_cloud_node
)

install(PROGRAMS
  lane_detection/bag_to_images.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME bag_to_images
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install weights directory
install(DIRECTORY
  weights
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
