// src/cone_seen_visual_node.cpp

#include "perception_winter/cone_seen_visual_node.hpp" // Adjust path as needed

using std::placeholders::_1;

ConeSeenVisualNode::ConeSeenVisualNode() : Node("coneseen_visuals")
{
    RCLCPP_INFO(this->get_logger(), "ConeSeenVisuals node has been started.");

    // Declare and get parameters
    this->declare_parameter<std::string>("frame_id", "Fr1A");
    this->get_parameter("frame_id", frame_id_);

    // QoS Profile - USE IT!
    auto qos = rclcpp::QoS(10);

    // Subscribers - use the QoS
    filtered_points_sub_ = this->create_subscription<dv_msgs::msg::IndexedTrack>(
        "/perception/cones", qos, std::bind(&ConeSeenVisualNode::cones_seen_visualisation, this, _1));

    // Publisher - use the QoS
    filtered_points_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/perception/cones_visualise", qos);
    
    // Initialize position variables to avoid potential use of uninitialized values
    x_ = 0.0;
    y_ = 0.0;
    yaw_ = 0.0;
}

double ConeSeenVisualNode::quaternion_to_yaw(const geometry_msgs::msg::Quaternion& q)
{
    // Yaw (Z-axis rotation)
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
}

// void ConeSeenVisualNode::check_for_slam_topic()
// {
//     if (!using_ground_truth_) {
//         // Already switched, destroy the timer to save resources
//         slam_check_timer_->cancel();
//         return;
//     }

//     auto topics = this->get_topic_names_and_types();
//     for (const auto& topic_pair : topics) {
//         if (topic_pair.first == "/slam_data") {
//             RCLCPP_INFO(this->get_logger(), "SLAM data detected. Switching subscriber...");

//             // Destroy the old ground truth subscriber by resetting the smart pointer
//             ground_truth_sub_.reset();

//             // Create new subscriber to SLAM
//             auto qos = rclcpp::QoS(10);
//             slam_sub_ = this->create_subscription<dv_msgs::msg::SlamState>(
//                 "/slam_data", qos, std::bind(&ConeSeenVisualNode::slam_state_callback, this, _1));
            
//             using_ground_truth_ = false;
//             RCLCPP_INFO(this->get_logger(), "Now subscribed to /slam_data");
            
//             // Stop the timer
//             slam_check_timer_->cancel();
//             break; 
//         }
//     }
// }

// void ConeSeenVisualNode::car_state_callback(const eufs_msgs::msg::CarState::SharedPtr msg)
// {
//     x_ = msg->pose.pose.position.x;
//     y_ = msg->pose.pose.position.y;
//     yaw_ = quaternion_to_yaw(msg->pose.pose.orientation);
// }

// void ConeSeenVisualNode::slam_state_callback(const dv_msgs::msg::SlamState::SharedPtr msg)
// {
//     x_ = msg->pose.pose.position.x;
//     y_ = msg->pose.pose.position.y;
//     yaw_ = quaternion_to_yaw(msg->pose.pose.orientation);
// }

void ConeSeenVisualNode::cones_seen_visualisation(const dv_msgs::msg::IndexedTrack::SharedPtr msg)
{
    visualization_msgs::msg::MarkerArray cones_seen_array;
    auto timestamp = this->get_clock()->now();

    // First, delete all previous markers with unique ID
    visualization_msgs::msg::Marker delete_all_marker;
    delete_all_marker.header.frame_id = frame_id_;
    delete_all_marker.header.stamp = timestamp;
    delete_all_marker.ns = "cone_visualization";
    delete_all_marker.id = 0;  // Unique ID for DELETEALL marker
    delete_all_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    cones_seen_array.markers.push_back(delete_all_marker);

    // Then add new cones with unique IDs starting from 1
    int id_counter = 1;  // Start from 1 to avoid conflict with DELETEALL marker (id=0)
    for (const auto& cone : msg->track) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = timestamp;
        marker.ns = "cone_visualization";
        marker.id = id_counter++;  // Increment ID for each cone
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position.z = 0.0;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = marker.scale.y = marker.scale.z = 0.4;
        
        marker.color.a = 1.0; // Don't forget to set the alpha!
        
        // Set colors based on cone type
        switch (cone.color) {
            case 0: // Blue
                marker.color.r = 0.0; marker.color.g = 0.47; marker.color.b = 1.0;
                break;
            case 1: // Yellow
                marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0;
                break;
            case 2: // Big Orange
                marker.color.r = 1.0; marker.color.g = 0.58; marker.color.b = 0.44;
                break;
            case 3: // Small Orange
                marker.color.r = 0.945; marker.color.g = 0.353; marker.color.b = 0.134;
                break;
            case 4: // Green
                marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0;
                break;
            default: // White (fallback)
                marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0;
                break;
        }

        // Remove lifetime since we're deleting all markers on each update
        // marker.lifetime = rclcpp::Duration::from_seconds(2.0);

        // Convert from local polar to local cartesian
        // double local_x = cone.location.x * std::cos(cone.location.y);
        // double local_y = cone.location.x * std::sin(cone.location.y);

        // cone.location.x is now Cartesian X, cone.location.y is now Cartesian Y.
        double local_x = cone.location.x;
        double local_y = cone.location.y;

        if (frame_id_ == "map") {
            // Rotate to map frame and translate
            marker.pose.position.x = x_ + std::cos(yaw_) * local_x - std::sin(yaw_) * local_y;
            marker.pose.position.y = y_ + std::sin(yaw_) * local_x + std::cos(yaw_) * local_y;
        } else {
            // Use local frame directly
            marker.pose.position.x = local_x + 1.532;
            marker.pose.position.y = local_y;
        }
        
        cones_seen_array.markers.push_back(marker);
    }
    
    RCLCPP_INFO(this->get_logger(), "Number of cones Visualised = %zu", cones_seen_array.markers.size() - 1); // Subtract 1 for the DELETEALL marker
    filtered_points_pub_->publish(cones_seen_array);
}