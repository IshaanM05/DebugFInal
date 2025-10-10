# cmake_minimum_required(VERSION 3.8)
# project(perception_winter)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# # find dependencies
# find_package(ament_cmake REQUIRED)

# # ---------------------------- Dependencies ---------------------------

# # ROS 2 packages
# find_package(rclcpp REQUIRED)
# find_package(sensor_msgs REQUIRED)
# find_package(visualization_msgs REQUIRED)
# find_package(std_msgs REQUIRED)
# find_package(pcl_conversions REQUIRED)
# find_package(dv_msgs REQUIRED)
# find_package(onnxruntime REQUIRED)
# # Third-party libraries
# find_package(Open3D REQUIRED)
# find_package(Eigen3 REQUIRED)

# # PCL library
# find_package(PCL REQUIRED COMPONENTS common io filters segmentation) 
# include_directories(${PCL_INCLUDE_DIRS})
# include_directories(include)
# add_definitions(${PCL_DEFINITIONS})

# # C++ standard
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # ---------------------------- Executables ---------------------------

# # Main process_lidar executable
# add_executable(process_lidar
#   src/process_lidar.cpp
#   src/main.cpp
# )

# add_executable(cone_visualizer_node
#   src/cone_seen_visual_node.cpp
#   src/main_1.cpp
# )

# # Dependencies for process_lidar
# ament_target_dependencies(process_lidar 
#   rclcpp 
#   sensor_msgs 
#   visualization_msgs 
#   std_msgs
#   pcl_conversions 
#   dv_msgs
# )

# ament_target_dependencies(cone_visualizer_node rclcpp visualization_msgs dv_msgs)

# # Link libraries for process_lidar
# target_link_libraries(process_lidar 
#   Open3D::Open3D 
#   Eigen3::Eigen 
#   ${onnxruntime_LIBRARIES}
#   ${PCL_LIBRARIES}
# )

# # Include directories
# target_include_directories(process_lidar PRIVATE
#   include
#   ${PCL_INCLUDE_DIRS}
#   ${Eigen3_INCLUDE_DIRS}
# )

# # Compile definitions
# target_compile_definitions(process_lidar PRIVATE
#   ${PCL_DEFINITIONS}
# )

# # ---------------------------- Installation ---------------------------

# # Install executable
# install(TARGETS
#   process_lidar
#   cone_visualizer_node
#   DESTINATION lib/${PROJECT_NAME}
# )

# # Install header files
# install(DIRECTORY include/
#   DESTINATION include/
# )

# # Install launch files
# install(DIRECTORY launch
#   DESTINATION share/${PROJECT_NAME}/
# )

# # Install RViz configuration
# install(
#   DIRECTORY rviz_config
#   DESTINATION share/${PROJECT_NAME}/
# )

# # Export dependencies
# ament_export_dependencies(
#   rclcpp
#   sensor_msgs
#   visualization_msgs
#   std_msgs
#   pcl_conversions
#   dv_msgs
#   Open3D
#   Eigen3
#   PCL
# )

# ament_export_include_directories(include)

# # ---------------------------- Testing ---------------------------

# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   # the following line skips the linter which checks for copyrights
#   # comment the line when a copyright and license is added to all source files
#   set(ament_cmake_copyright_FOUND TRUE)
#   # the following line skips cpplint (only works in a git repo)
#   # comment the line when this package is in a git repo and when
#   # a copyright and license is added to all source files
#   set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()

# ament_package()

cmake_minimum_required(VERSION 3.8)
project(perception_winter)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------------------------- Dependencies ---------------------------

# Find all ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(dv_msgs REQUIRED)

# Find third-party libraries
find_package(Open3D REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation)

# --- Manual ONNX Runtime Setup ---
# ⚠️ UPDATE THIS PATH to the exact location of your extracted ONNX Runtime folder
set(ONNXRUNTIME_DIR "/usr/local/onnxruntime-linux-x64-1.17.1")
# ---

# Check if the path is valid and provide a clear error if it's not
if(NOT EXISTS ${ONNXRUNTIME_DIR})
    message(FATAL_ERROR "ONNX Runtime not found at the specified path: ${ONNXRUNTIME_DIR}")
else()
    message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_DIR}")
endif()

# Find the ONNX library file inside the specified directory
find_library(ONNXRUNTIME_LIB 
  NAMES onnxruntime
  PATHS "${ONNXRUNTIME_DIR}/lib"
  NO_DEFAULT_PATH
  REQUIRED
)

# ---------------------------- Executables ---------------------------

# Main process_lidar executable
add_executable(process_lidar_node
  src/process_lidar.cpp
  src/main.cpp
)

# Cone visualizer executable
add_executable(cone_visualizer_node
  src/cone_seen_visual_node.cpp
  src/main_1.cpp
)

# --- Configure process_lidar_node ---
ament_target_dependencies(process_lidar_node 
  rclcpp sensor_msgs visualization_msgs std_msgs pcl_conversions dv_msgs
)
target_include_directories(process_lidar_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${Open3D_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  "${ONNXRUNTIME_DIR}/include"  # Add ONNX include path
)
target_link_libraries(process_lidar_node
  ${ament_cmake_libraries}
  ${PCL_LIBRARIES}
  Open3D::Open3D 
  Eigen3::Eigen 
  ${ONNXRUNTIME_LIB}  # Link the found ONNX library
)

# --- Configure cone_visualizer_node ---
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

# ---------------------------- Installation ---------------------------

# Install executables
install(TARGETS
  process_lidar_node
  cone_visualizer_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files, launch files, and configs
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY rviz_config DESTINATION share/${PROJECT_NAME}/)

# Install the ONNX model file
set(MODEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cone_model.onnx")
if(EXISTS ${MODEL_FILE})
  install(FILES ${MODEL_FILE} DESTINATION share/${PROJECT_NAME})
endif()

ament_package() 