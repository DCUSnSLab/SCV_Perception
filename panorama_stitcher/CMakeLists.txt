cmake_minimum_required(VERSION 3.8)
project(panorama_stitcher)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(
    rgbd_panorama_torch_test
    test/test_rgbd_panorama_torch.py
    TIMEOUT 30
  )
  ament_add_pytest_test(
    panorama_ground_segmentation_test
    test/test_ground_segmentation_python.py
    TIMEOUT 30
  )
endif()

ament_python_install_package(panorama_stitcher_py)

install(PROGRAMS
  scripts/panorama_ground_segmentation_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME panorama_ground_segmentation_node
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

install(PROGRAMS
  scripts/rgbd_panorama_torch_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME rgbd_panorama_torch_node
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

ament_package()
