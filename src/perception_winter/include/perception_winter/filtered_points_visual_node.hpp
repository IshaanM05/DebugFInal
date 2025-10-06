/**
 * @file filtered_points_visual_node.hpp
 * @brief Visualizer for filtered LiDAR points
 * @author Siddhesh Phadke
 */

#ifndef FILTERED_POINTS_VISUAL_NODE_HPP_
#define FILTERED_POINTS_VISUAL_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

namespace perception_winter {

class FilteredPointsVisualNode : public rclcpp::Node {
public:
    FilteredPointsVisualNode();
    
private:
    void filtered_points_visualisation(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    
    // Subscribers
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr filtered_points_sub_;
    
    // Publishers
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr filtered_points_pub_;
    
    // Parameters
    std::string frame_id_;
    
    // Position variables (if needed for coordinate transformation)
    double x_;
    double y_;
    double yaw_;
};

} // namespace perception_winter

#endif // FILTERED_POINTS_VISUAL_NODE_HPP_