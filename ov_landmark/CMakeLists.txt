cmake_minimum_required(VERSION 3.8)
project(ov_landmark)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d aruco)
find_package(yaml-cpp REQUIRED)
find_package(ov_core REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ov_landmark_lib SHARED
  src/dummy.cpp
)

target_include_directories(ov_landmark_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(ov_landmark_lib
  ov_core
)

target_link_libraries(ov_landmark_lib
  PUBLIC
    Eigen3::Eigen
    yaml-cpp
    ${OpenCV_LIBRARIES}
)

install(
  TARGETS ov_landmark_lib
  EXPORT export_ov_landmark
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_ov_landmark HAS_LIBRARY_TARGET)
ament_export_dependencies(
  Eigen3
  OpenCV
  yaml-cpp
  ov_core
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
