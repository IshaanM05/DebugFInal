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

# --- ONNX Runtime setup (EXACTLY matching working package) ---
set(ONNXRUNTIME_DIR "/usr/local/onnxruntime-linux-x64-1.17.1")

if(NOT EXISTS ${ONNXRUNTIME_DIR})
  message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_DIR}")
else()
  message(STATUS "Found ONNX Runtime at ${ONNXRUNTIME_DIR}")
endif()

find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime
  PATHS "${ONNXRUNTIME_DIR}/lib"
  NO_DEFAULT_PATH
  REQUIRED
)

message(STATUS "ONNX Runtime library found: ${ONNXRUNTIME_LIB}")

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

add_executable(filtered_points_visual_node 
    src/filtered_points_visual_node.cpp
    src/filtered_points_visual_main.cpp
)

# Dependencies for process_lidar
ament_target_dependencies(process_lidar 
  rclcpp 
  sensor_msgs 
  visualization_msgs 
  std_msgs
  pcl_conversions 
  dv_msgs
  ament_index_cpp
)

ament_target_dependencies(cone_visualizer_node rclcpp visualization_msgs dv_msgs)

ament_target_dependencies(filtered_points_visual_node
    rclcpp
    visualization_msgs
    std_msgs
)

# Include directories for process_lidar
target_include_directories(process_lidar PRIVATE
  include
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${ONNXRUNTIME_DIR}/include  # ← Use the exact path from working package
)

# Link libraries for process_lidar
target_link_libraries(process_lidar 
  Open3D::Open3D 
  Eigen3::Eigen 
  ${PCL_LIBRARIES}
  ${ONNXRUNTIME_LIB}
)

# Compile definitions
target_compile_definitions(process_lidar PRIVATE
  ${PCL_DEFINITIONS}
)

# --- Install the ONNX model file ---
set(MODEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cone_model.onnx")
if(EXISTS ${MODEL_FILE})
  install(FILES ${MODEL_FILE} DESTINATION share/${PROJECT_NAME})
  message(STATUS "ONNX model file found: ${MODEL_FILE}")
else()
  message(WARNING "ONNX model file not found: ${MODEL_FILE}")
endif()

# ---------------------------- Installation ---------------------------

# Install executable
install(TARGETS
  process_lidar
  cone_visualizer_node
  filtered_points_visual_node
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
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()