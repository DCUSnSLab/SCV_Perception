cmake_minimum_required(VERSION 3.8)
project(parking_detector)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(command_center_interfaces REQUIRED)  # 이 라인 추가

# rosidl_generate_interfaces() 부분을 완전히 제거

install(PROGRAMS
  src/main.py
  src/test_T_area.py
  src/test_P_area.py
  src/main_test.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()