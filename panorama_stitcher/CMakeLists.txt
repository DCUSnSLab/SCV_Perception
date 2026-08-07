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
if(
  NOT CMAKE_CUDA_ARCHITECTURES AND
  DEFINED ENV{CUDAARCHS} AND
  NOT "$ENV{CUDAARCHS}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}" CACHE STRING
    "CUDA architectures for panorama_stitcher")
endif()
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(CMAKE_CUDA_ARCHITECTURES)
    message(STATUS
      "panorama_stitcher: CUDA ${CUDAToolkit_VERSION}, "
      "architectures=${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(STATUS
      "panorama_stitcher: CUDA ${CUDAToolkit_VERSION}, compiler-default "
      "architecture (override with CUDAARCHS or "
      "-DCMAKE_CUDA_ARCHITECTURES=<SM>)")
  endif()
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

add_executable(far_hybrid_panorama_stitcher_node
  src/far_hybrid_panorama_stitcher_node.cpp
)

add_executable(panorama_ground_segmentation_node
  src/panorama_ground_segmentation_node.cpp
)

if(CMAKE_CUDA_COMPILER)
  # Build the CUDA implementation once. Keeping it in a standalone library
  # gives both panorama nodes the exact same lifecycle/error handling and
  # makes the backend independently testable under compute-sanitizer.
  add_library(panorama_cuda_backend STATIC
    src/cuda_panorama_backend.cu
  )
  set_target_properties(panorama_cuda_backend PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(panorama_cuda_backend
    CUDA::cudart
    ${OpenCV_LIBS}
  )

  target_compile_definitions(rgbd_panorama_stitcher_node PRIVATE
    PANORAMA_WITH_CUDA=1
  )
  target_link_libraries(rgbd_panorama_stitcher_node
    panorama_cuda_backend
  )

  target_compile_definitions(far_hybrid_panorama_stitcher_node PRIVATE
    PANORAMA_WITH_CUDA=1
  )
  target_link_libraries(far_hybrid_panorama_stitcher_node
    panorama_cuda_backend
  )

  if(BUILD_TESTING)
    add_executable(cuda_panorama_backend_smoke_test
      test/cuda_panorama_backend_smoke_test.cpp
    )
    target_include_directories(cuda_panorama_backend_smoke_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(cuda_panorama_backend_smoke_test
      panorama_cuda_backend
      ${OpenCV_LIBS}
    )
    add_test(
      NAME cuda_panorama_backend_smoke_test
      COMMAND cuda_panorama_backend_smoke_test
    )
    set_tests_properties(cuda_panorama_backend_smoke_test PROPERTIES
      SKIP_RETURN_CODE 77
      TIMEOUT 20
    )
  endif()
endif()

add_executable(front_rgbd_fusion_node
  src/front_rgbd_fusion_node.cpp
)

target_compile_features(panorama_stitcher_node PUBLIC cxx_std_17)
target_compile_features(rgbd_panorama_stitcher_node PUBLIC cxx_std_17)
target_compile_features(far_hybrid_panorama_stitcher_node PUBLIC cxx_std_17)
target_compile_features(panorama_ground_segmentation_node PUBLIC cxx_std_17)
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

ament_target_dependencies(far_hybrid_panorama_stitcher_node
  cv_bridge
  message_filters
  rclcpp
  sensor_msgs
)

ament_target_dependencies(panorama_ground_segmentation_node
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

target_link_libraries(far_hybrid_panorama_stitcher_node
  ${OpenCV_LIBS}
)

target_link_libraries(panorama_ground_segmentation_node
  ${OpenCV_LIBS}
)

target_link_libraries(front_rgbd_fusion_node
  ${OpenCV_LIBS}
)

install(TARGETS
  panorama_stitcher_node
  rgbd_panorama_stitcher_node
  far_hybrid_panorama_stitcher_node
  panorama_ground_segmentation_node
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
