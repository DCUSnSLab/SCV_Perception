cmake_minimum_required(VERSION 3.8)
project(panorama_stitcher)

if(NOT CMAKE_CUDA_COMPILER)
  find_program(PANORAMA_CUDA_COMPILER
    NAMES nvcc
    HINTS /usr/local/cuda/bin
  )
  if(PANORAMA_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${PANORAMA_CUDA_COMPILER}" CACHE FILEPATH
      "CUDA compiler used by panorama_stitcher")
  endif()
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
  )
endif()

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(panorama_stitcher_node
  src/panorama_stitcher_node.cpp
)

add_executable(rgbd_panorama_stitcher_node
  src/rgbd_panorama_stitcher_node.cpp
)

if(CMAKE_CUDA_COMPILER)
  target_sources(rgbd_panorama_stitcher_node PRIVATE
    src/cuda_panorama_backend.cu
  )
  target_compile_definitions(rgbd_panorama_stitcher_node PRIVATE
    PANORAMA_WITH_CUDA=1
  )
  set_target_properties(rgbd_panorama_stitcher_node PROPERTIES
    CUDA_ARCHITECTURES 89
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(rgbd_panorama_stitcher_node
    CUDA::cudart
  )
endif()

add_executable(front_rgbd_fusion_node
  src/front_rgbd_fusion_node.cpp
)

target_compile_features(panorama_stitcher_node PUBLIC cxx_std_17)
target_compile_features(rgbd_panorama_stitcher_node PUBLIC cxx_std_17)
target_compile_features(front_rgbd_fusion_node PUBLIC cxx_std_17)

ament_target_dependencies(panorama_stitcher_node
  cv_bridge
  message_filters
  rclcpp
  sensor_msgs
)

ament_target_dependencies(rgbd_panorama_stitcher_node
  cv_bridge
  message_filters
  rclcpp
  sensor_msgs
)

ament_target_dependencies(front_rgbd_fusion_node
  cv_bridge
  message_filters
  rclcpp
  sensor_msgs
)

target_link_libraries(panorama_stitcher_node
  ${OpenCV_LIBS}
)

target_link_libraries(rgbd_panorama_stitcher_node
  ${OpenCV_LIBS}
)

target_link_libraries(front_rgbd_fusion_node
  ${OpenCV_LIBS}
)

install(TARGETS
  panorama_stitcher_node
  rgbd_panorama_stitcher_node
  front_rgbd_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/charuco_rig_calibrator.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME charuco_rig_calibrator
)

install(PROGRAMS
  scripts/multiboard_bundle_calibrator.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME multiboard_bundle_calibrator
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
