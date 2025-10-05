// src/main.cpp

#include "perception_winter/cone_seen_visual_node.hpp" // Adjust path as needed
#include <rclcpp/rclcpp.hpp>

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ConeSeenVisualNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}