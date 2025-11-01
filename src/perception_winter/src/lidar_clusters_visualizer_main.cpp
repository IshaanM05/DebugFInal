#include "perception_winter/lidar_clusters_visualizer.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<perception_winter::LidarClustersVisualizer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}