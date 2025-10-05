cmake_minimum_required(VERSION 3.8)
project(perception_winter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# ---------------------------- Dependencies ---------------------------

# ROS 2 packages
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(dv_msgs REQUIRED)

# Third-party libraries
find_package(Open3D REQUIRED)
find_package(Eigen3 REQUIRED)

# PCL library
find_package(PCL REQUIRED COMPONENTS common io filters segmentation) 
include_directories(${PCL_INCLUDE_DIRS})
include_directories(include)
add_definitions(${PCL_DEFINITIONS})

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------- Executables ---------------------------

# Main process_lidar executable
add_executable(process_lidar
  src/process_lidar.cpp
  src/main.cpp
)

add_executable(cone_visualizer_node
  src/cone_seen_visual_node.cpp
  src/main_1.cpp
)

# Dependencies for process_lidar
ament_target_dependencies(process_lidar 
  rclcpp 
  sensor_msgs 
  visualization_msgs 
  std_msgs
  pcl_conversions 
  dv_msgs
)

ament_target_dependencies(cone_visualizer_node rclcpp visualization_msgs dv_msgs)

# Link libraries for process_lidar
target_link_libraries(process_lidar 
  Open3D::Open3D 
  Eigen3::Eigen 
  ${PCL_LIBRARIES}
)

# Include directories
target_include_directories(process_lidar PRIVATE
  include
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)

# Compile definitions
target_compile_definitions(process_lidar PRIVATE
  ${PCL_DEFINITIONS}
)

# ---------------------------- Installation ---------------------------

# Install executable
install(TARGETS
  process_lidar
  cone_visualizer_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY include/
  DESTINATION include/
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install RViz configuration
install(
  DIRECTORY rviz_config
  DESTINATION share/${PROJECT_NAME}/
)

# Export dependencies
ament_export_dependencies(
  rclcpp
  sensor_msgs
  visualization_msgs
  std_msgs
  pcl_conversions
  dv_msgs
  Open3D
  Eigen3
  PCL
)

ament_export_include_directories(include)

# ---------------------------- Testing ---------------------------

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