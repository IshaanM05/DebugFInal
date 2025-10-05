// include/your_package_name/cone_seen_visual_node.hpp

#ifndef CONE_SEEN_VISUAL_NODE_HPP_
#define CONE_SEEN_VISUAL_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <dv_msgs/msg/indexed_track.hpp>
#include <dv_msgs/msg/slam_state.hpp>
// #include <eufs_msgs/msg/car_state.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <cmath>

class ConeSeenVisualNode : public rclcpp::Node
{
public:
    ConeSeenVisualNode();

private:
    // Callback functions
    // void car_state_callback(const eufs_msgs::msg::CarState::SharedPtr msg);
    // void slam_state_callback(const dv_msgs::msg::SlamState::SharedPtr msg);
    void cones_seen_visualisation(const dv_msgs::msg::IndexedTrack::SharedPtr msg);
    // void check_for_slam_topic();

    // Helper function
    double quaternion_to_yaw(const geometry_msgs::msg::Quaternion& q);

    // Member Variables
    double x_{0.0};
    double y_{0.0};
    double yaw_{0.0};
    std::string frame_id_;
    bool using_ground_truth_{true};

    // ROS2 Interfaces
    rclcpp::Subscription<dv_msgs::msg::IndexedTrack>::SharedPtr filtered_points_sub_;
    // rclcpp::Subscription<eufs_msgs::msg::CarState>::SharedPtr ground_truth_sub_;
    rclcpp::Subscription<dv_msgs::msg::SlamState>::SharedPtr slam_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr filtered_points_pub_;
    rclcpp::TimerBase::SharedPtr slam_check_timer_;
};

#endif // CONE_SEEN_VISUAL_NODE_HPP_