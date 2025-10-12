#include "perception_winter/cone_seen_visual_node.hpp" 

using std::placeholders::_1;

ConeSeenVisualNode::ConeSeenVisualNode() : Node("coneseen_visuals")
{
    RCLCPP_INFO(this->get_logger(), "ConeSeenVisuals node has been started.");

    this->declare_parameter<std::string>("frame_id", "ouster");
    this->get_parameter("frame_id", frame_id_);

    auto qos = rclcpp::QoS(10);

    filtered_points_sub_ = this->create_subscription<dv_msgs::msg::IndexedTrack>(
        "/perception/cones", qos, std::bind(&ConeSeenVisualNode::cones_seen_visualisation, this, _1));

    filtered_points_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/perception/cones_visualise", qos);
    
    x_ = 0.0;
    y_ = 0.0;
    yaw_ = 0.0;
}

double ConeSeenVisualNode::quaternion_to_yaw(const geometry_msgs::msg::Quaternion& q)
{
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
}

void ConeSeenVisualNode::cones_seen_visualisation(const dv_msgs::msg::IndexedTrack::SharedPtr msg)
{
    visualization_msgs::msg::MarkerArray cones_seen_array;
    auto timestamp = this->get_clock()->now();

    visualization_msgs::msg::Marker delete_all_marker;
    delete_all_marker.header.frame_id = frame_id_;
    delete_all_marker.header.stamp = timestamp;
    delete_all_marker.ns = "cone_visualization";
    delete_all_marker.id = 0; 
    delete_all_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    cones_seen_array.markers.push_back(delete_all_marker);

    int id_counter = 1; 
    for (const auto& cone : msg->track) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = timestamp;
        marker.ns = "cone_visualization";
        marker.id = id_counter++; 
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position.z = 0.0;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = marker.scale.y = marker.scale.z = 0.4;
        
        marker.color.a = 1.0;
        
        switch (cone.color) {
            case 0:
                marker.color.r = 0.0; marker.color.g = 0.47; marker.color.b = 1.0;
                break;
            case 1: 
                marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0;
                break;
            case 2:
                marker.color.r = 1.0; marker.color.g = 0.58; marker.color.b = 0.44;
                break;
            case 3:
                marker.color.r = 0.945; marker.color.g = 0.353; marker.color.b = 0.134;
                break;
            case 4:
                marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0;
                break;
            default:
                marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0;
                break;
        }

        marker.lifetime = rclcpp::Duration::from_seconds(0.5);

        double local_x = cone.location.x * std::cos(cone.location.y);
        double local_y = cone.location.x * std::sin(cone.location.y);

        if (frame_id_ == "map") {
            marker.pose.position.x = x_ + std::cos(yaw_) * local_x - std::sin(yaw_) * local_y;
            marker.pose.position.y = y_ + std::sin(yaw_) * local_x + std::cos(yaw_) * local_y;
        } else {
            marker.pose.position.x = local_x;
            marker.pose.position.y = local_y;
            marker.pose.position.z = -0.5;
        }
        
        cones_seen_array.markers.push_back(marker);
    }
    
    RCLCPP_INFO(this->get_logger(), "Number of cones Visualised = %zu", cones_seen_array.markers.size() - 1);
    filtered_points_pub_->publish(cones_seen_array);
}

// #include "perception_winter/cone_seen_visual_node.hpp"

// #include <cmath>
// #include <functional> // Required for std::bind and std::placeholders

// using std::placeholders::_1;

// ConeSeenVisualNode::ConeSeenVisualNode() : Node("coneseen_visuals")
// {
//     RCLCPP_INFO(this->get_logger(), "ConeSeenVisuals node has been started.");

//     this->declare_parameter<std::string>("frame_id", "ouster");
//     this->get_parameter("frame_id", frame_id_);

//     auto qos = rclcpp::QoS(10);

//     filtered_points_sub_ = this->create_subscription<dv_msgs::msg::IndexedTrack>(
//         "/perception/cones", qos, std::bind(&ConeSeenVisualNode::cones_seen_visualisation, this, _1));

//     filtered_points_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
//         "/perception/cones_visualise", qos);

//     // Initialize state variables
//     x_ = 0.0;
//     y_ = 0.0;
//     yaw_ = 0.0;
// }

// double ConeSeenVisualNode::quaternion_to_yaw(const geometry_msgs::msg::Quaternion& q)
// {
//     // t0 = +2.0 * (w * x + y * z)
//     // t1 = +1.0 - 2.0 * (x * x + y * y)
//     // roll = atan2(t0, t1)
    
//     // t2 = +2.0 * (w * y - z * x)
//     // t2 = t2 > 1.0 ? 1.0 : t2
//     // t2 = t2 < -1.0 ? -1.0 : t2
//     // pitch = asin(t2)
    
//     // t3 = +2.0 * (w * z + x * y)
//     // t4 = +1.0 - 2.0 * (y * y + z * z)
//     // yaw = atan2(t3, t4)
    
//     double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
//     double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
//     return std::atan2(siny_cosp, cosy_cosp);
// }

// void ConeSeenVisualNode::cones_seen_visualisation(const dv_msgs::msg::IndexedTrack::SharedPtr msg)
// {
//     visualization_msgs::msg::MarkerArray cones_seen_array;
//     auto timestamp = this->get_clock()->now();

//     // Add a DELETEALL marker to clear previous visualizations
//     visualization_msgs::msg::Marker delete_all_marker;
//     delete_all_marker.header.frame_id = frame_id_;
//     delete_all_marker.header.stamp = timestamp;
//     delete_all_marker.ns = "cone_visualization";
//     delete_all_marker.id = 0;
//     delete_all_marker.action = visualization_msgs::msg::Marker::DELETEALL;
//     cones_seen_array.markers.push_back(delete_all_marker);

//     int id_counter = 1;
//     for (const auto& cone : msg->track) {
//         visualization_msgs::msg::Marker marker;
//         marker.header.frame_id = frame_id_;
//         marker.header.stamp = timestamp;
//         marker.ns = "cone_visualization";
//         marker.id = id_counter++;
//         marker.type = visualization_msgs::msg::Marker::SPHERE;
//         marker.action = visualization_msgs::msg::Marker::ADD;
        
//         marker.pose.position.z = 0.0;
//         marker.pose.orientation.w = 1.0;

//         marker.scale.x = marker.scale.y = marker.scale.z = 0.4;
        
//         marker.color.a = 1.0;
        
//         switch (cone.color) {
//             case dv_msgs::msg::IndexedCone::BLUE:   // Color 0
//                 marker.color.r = 0.0f; marker.color.g = 0.47f; marker.color.b = 1.0f;
//                 break;
//             case dv_msgs::msg::IndexedCone::YELLOW: // Color 1
//                 marker.color.r = 1.0f; marker.color.g = 1.0f;  marker.color.b = 0.0f;
//                 break;
//             case dv_msgs::msg::IndexedCone::ORANGE_BIG: // Color 2
//                 marker.color.r = 1.0f; marker.color.g = 0.58f; marker.color.b = 0.44f;
//                 break;
//             // Add other cases if needed
//             default:
//                 marker.color.r = 1.0f; marker.color.g = 1.0f; marker.color.b = 1.0f;
//                 break;
//         }

//         marker.lifetime = rclcpp::Duration::from_seconds(0.5);

//         // Convert from polar (range, angle) to Cartesian (x, y) coordinates
//         double local_x = cone.location.x * std::cos(cone.location.y);
//         double local_y = cone.location.x * std::sin(cone.location.y);

//         if (frame_id_ == "map") {
//             // Transform from robot frame to map frame
//             marker.pose.position.x = x_ + std::cos(yaw_) * local_x - std::sin(yaw_) * local_y;
//             marker.pose.position.y = y_ + std::sin(yaw_) * local_x + std::cos(yaw_) * local_y;
//         } else { // Assume robot-centric frame like "ouster"
//             marker.pose.position.x = local_x;
//             marker.pose.position.y = local_y;
//             marker.pose.position.z = -0.5; // Lower the cone slightly for better visibility
//         }
        
//         cones_seen_array.markers.push_back(marker);
//     }
    
//     // Only log if there are cones to visualize
//     if (cones_seen_array.markers.size() > 1) {
//         RCLCPP_INFO(this->get_logger(), "Visualizing %zu cones.", cones_seen_array.markers.size() - 1);
//     }
//     filtered_points_pub_->publish(cones_seen_array);
// }