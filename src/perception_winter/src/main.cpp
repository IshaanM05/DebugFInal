#include "perception_winter/process_lidar.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<perception_winter::ProcessLidar>());
    rclcpp::shutdown();
    return 0;
}