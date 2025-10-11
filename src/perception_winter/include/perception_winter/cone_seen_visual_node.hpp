#ifndef CONE_SEEN_VISUAL_NODE_HPP_
#define CONE_SEEN_VISUAL_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <dv_msgs/msg/indexed_track.hpp>
#include <dv_msgs/msg/slam_state.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <cmath>

class ConeSeenVisualNode : public rclcpp::Node
{
public:
    ConeSeenVisualNode();

private:
    void cones_seen_visualisation(const dv_msgs::msg::IndexedTrack::SharedPtr msg);

    double quaternion_to_yaw(const geometry_msgs::msg::Quaternion& q);

    double x_{0.0};
    double y_{0.0};
    double yaw_{0.0};
    std::string frame_id_;
    bool using_ground_truth_{true};

    rclcpp::Subscription<dv_msgs::msg::IndexedTrack>::SharedPtr filtered_points_sub_;
    rclcpp::Subscription<dv_msgs::msg::SlamState>::SharedPtr slam_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr filtered_points_pub_;
    rclcpp::TimerBase::SharedPtr slam_check_timer_;
};

#endif