
#include "perception_winter/filtered_points_visual_node.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<perception_winter::FilteredPointsVisualNode>();
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}