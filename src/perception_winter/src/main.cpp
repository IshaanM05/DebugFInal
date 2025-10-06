/**
 * @file main.cpp
 * @brief Main entry point for LiDAR processing node
 * @author Siddhesh Phadke
 */

#include "perception_winter/process_lidar.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    // FIX: Add namespace qualification
    auto node = std::make_shared<perception_winter::ProcessLidar>();
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}