cmake_minimum_required(VERSION 3.8)
project(traffic_light_detection)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  traffic_light_detection/traffic_light_detector.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME traffic_light_detector
)

install(DIRECTORY
  launch
  weights
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
