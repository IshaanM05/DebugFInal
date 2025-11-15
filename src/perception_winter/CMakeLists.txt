cmake_minimum_required(VERSION 3.8)
project(perception_winter)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ABI compatibility settings for better library interoperability
add_compile_definitions(BUILD_SHARED_LIBS=ON)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# --- Find ROS2 and other dependencies ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(dv_msgs REQUIRED)

# Additional ROS2 packages needed for the C++ code
find_package(ament_index_cpp REQUIRED)
find_package(rcutils REQUIRED)

# Find yaml-cpp for YAML configuration parsing
find_package(yaml-cpp REQUIRED)

# Find third-party libraries
find_package(Open3D REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation)

# TBB parallel processing disabled in code - using sequential implementation
add_compile_definitions(HAVE_TBB=0)
message(STATUS "Enhanced ground removal using sequential processing (TBB disabled in code)")

# --- Manual ONNX Runtime Setup ---
set(ONNXRUNTIME_DIR "/usr/local/onnxruntime-linux-x64-1.17.1")

if(NOT EXISTS ${ONNXRUNTIME_DIR})
    message(FATAL_ERROR "ONNX Runtime not found at the specified path: ${ONNXRUNTIME_DIR}")
else()
    message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_DIR}")
endif()

find_library(ONNXRUNTIME_LIB 
  NAMES onnxruntime
  PATHS "${ONNXRUNTIME_DIR}/lib"
  NO_DEFAULT_PATH
  REQUIRED
)

# --- Executable Definitions ---
add_executable(process_lidar_node
  src/process_lidar.cpp
  src/main.cpp
)

add_executable(cone_visualizer_node
  src/cone_seen_visual_node.cpp
  src/main_1.cpp
)

add_executable(filtered_points_node
  src/filtered_points_visual_node.cpp
  src/filtered_points_visual_main.cpp
)

# NEW: Add cluster visualizer node
add_executable(lidar_clusters_visualizer_node
  src/lidar_clusters_visualizer.cpp
  src/lidar_clusters_visualizer_main.cpp
)

# --- Process Lidar Node ---
ament_target_dependencies(process_lidar_node 
  rclcpp sensor_msgs visualization_msgs std_msgs pcl_conversions dv_msgs
  ament_index_cpp rcutils  # ADD THESE
)

target_include_directories(process_lidar_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${Open3D_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  "${ONNXRUNTIME_DIR}/include"
  ${yaml-cpp_INCLUDE_DIRS}
  # Add these for ament_index_cpp and other ROS2 utilities
  ${ament_index_cpp_INCLUDE_DIRS}
)

target_link_libraries(process_lidar_node
  ${ament_cmake_libraries}
  ${PCL_LIBRARIES}
  Open3D::Open3D 
  Eigen3::Eigen 
  ${ONNXRUNTIME_LIB}
  yaml-cpp
  ament_index_cpp::ament_index_cpp  # ADD THIS
  rcutils::rcutils  # ADD THIS
)

# Add filesystem library for C++17 features (needed for std::filesystem)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
    # GCC 9+ uses std::filesystem in libstdc++
    target_link_libraries(process_lidar_node stdc++fs)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang may need c++fs
    target_link_libraries(process_lidar_node c++fs)
endif()

# --- Cone Visualizer Node ---
ament_target_dependencies(cone_visualizer_node 
  rclcpp visualization_msgs dv_msgs
)
target_include_directories(cone_visualizer_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(cone_visualizer_node
  ${ament_cmake_libraries}
)

# --- Filtered Points Visualizer Node ---
ament_target_dependencies(filtered_points_node 
  rclcpp std_msgs visualization_msgs
)
target_include_directories(filtered_points_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(filtered_points_node
  ${ament_cmake_libraries}
)

# --- NEW: Lidar Clusters Visualizer Node ---
ament_target_dependencies(lidar_clusters_visualizer_node 
  rclcpp std_msgs visualization_msgs
)
target_include_directories(lidar_clusters_visualizer_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(lidar_clusters_visualizer_node
  ${ament_cmake_libraries}
)

# --- Installation ---
install(TARGETS
  process_lidar_node
  cone_visualizer_node
  filtered_points_node
  lidar_clusters_visualizer_node  # NEW: Install the cluster visualizer
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)

# Install the config directory with the YAML file
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}
)

# Install ONNX model file if it exists
set(MODEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cone_model.onnx")
if(EXISTS ${MODEL_FILE})
  install(FILES ${MODEL_FILE} DESTINATION share/${PROJECT_NAME})
endif()

# --- Optional compiler optimizations for performance ---
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Release build detected - enabling optimizations for enhanced ground removal")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(process_lidar_node PRIVATE -O3 -march=native -ffast-math)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(process_lidar_node PRIVATE -O3 -march=native -ffast-math)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(process_lidar_node PRIVATE /O2 /fp:fast)
  endif()
endif()

# Finalize the package configuration
ament_package()