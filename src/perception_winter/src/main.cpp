/**
* @name main.cpp
* @brief Entry point of the node process_lidar
* @author Siddhesh Phadke
*/

#include "perception_winter/process_lidar.hpp"

int main(int argc, char**argv){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ProcessLidar>());
    rclcpp::shutdown();
    return 0;
}