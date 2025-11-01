

#include "perception_winter/filtered_points_visual_node.hpp"
#include <rclcpp/rclcpp.hpp>

using namespace perception_winter;
using std::placeholders::_1;

FilteredPointsVisualNode::FilteredPointsVisualNode() : Node("filtered_points_visuals")
{
    RCLCPP_INFO(this->get_logger(), "FilteredPointsVisuals node has been started.");

    // Declare and get parameters
    this->declare_parameter<std::string>("frame_id", "Fr1A");
    this->get_parameter("frame_id", frame_id_);

    // QoS Profile
    auto qos = rclcpp::QoS(10);

    // Subscribers - use the QoS
    filtered_points_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/perception/filtered_points", qos, 
        std::bind(&FilteredPointsVisualNode::filtered_points_visualisation, this, _1));

    // Publisher - use the QoS
    filtered_points_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/perception/filtered_points_vis", qos);
    
    // Initialize position variables to avoid potential use of uninitialized values
    x_ = 0.0;
    y_ = 0.0;
    yaw_ = 0.0;
}

void FilteredPointsVisualNode::filtered_points_visualisation(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    visualization_msgs::msg::MarkerArray point_array;
    auto timestamp = this->get_clock()->now();

    // First, delete all previous markers with unique ID
    visualization_msgs::msg::Marker delete_all_marker;
    delete_all_marker.header.frame_id = frame_id_;
    delete_all_marker.header.stamp = timestamp;
    delete_all_marker.ns = "filtered_points_visualization";
    delete_all_marker.id = 0;  // Unique ID for DELETEALL marker
    delete_all_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    point_array.markers.push_back(delete_all_marker);

    // Then add new points with unique IDs starting from 1
    int id_counter = 1;  // Start from 1 to avoid conflict with DELETEALL marker (id=0)
    const auto& pts = msg->data;  // 1D list like [x1, y1, z1, x2, y2, z2, ...]

    for (size_t i = 0; i < pts.size(); i += 3) {
        if (i + 2 >= pts.size()) break;  // Ensure we have complete x,y,z triplets

        float x = pts[i];
        float y = pts[i + 1];
        float z = pts[i + 2];

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = timestamp;
        marker.ns = "filtered_points_visualization";
        marker.id = id_counter++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // Marker size
        marker.scale.x = 0.05;
        marker.scale.y = 0.05;
        marker.scale.z = 0.05;

        // Fixed color (solid blue as in Python version)
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;

        // Position and orientation
        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = z;
        marker.pose.orientation.w = 1.0;  // Identity quaternion

        // Remove lifetime since we're deleting all markers on each update
        // marker.lifetime = rclcpp::Duration::from_seconds(2.0);
        // Apply coordinate transformation if needed (similar to cone visualizer)
        if (frame_id_ == "map") {
            // If you need to transform from local to map frame, use the same logic as cones
             marker.pose.position.x = x_ + std::cos(yaw_) * x - std::sin(yaw_) * y;
             marker.pose.position.y = y_ + std::sin(yaw_) * x + std::cos(yaw_) * y;
        } else {
            // Use local frame directly with offset if needed
             marker.pose.position.x = x + 1.532;  // Apply offset if required
             marker.pose.position.y = y;
        }
        
        point_array.markers.push_back(marker);
    }
    
    RCLCPP_INFO(this->get_logger(), "Number of filtered points = %zu", point_array.markers.size() - 1); // Subtract 1 for the DELETEALL marker
    filtered_points_pub_->publish(point_array);
}