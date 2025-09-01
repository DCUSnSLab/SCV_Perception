cmake_minimum_required(VERSION 3.10)
project(memsort_ros2)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# 파이썬 패키지(모듈) 설치
ament_python_install_package(${PROJECT_NAME})

# 실행 래퍼를 lib/<pkg>/ 로 설치 (ros2 run 이 찾음)
install(PROGRAMS
  scripts/memory_sort_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
