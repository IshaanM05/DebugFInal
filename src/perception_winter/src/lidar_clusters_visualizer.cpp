#include "perception_winter/lidar_clusters_visualizer.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <chrono>

using namespace perception_winter;

LidarClustersVisualizer::LidarClustersVisualizer() 
    : Node("lidar_clusters_visualizer") {
    
    // Declare parameters
    this->declare_parameter<std::string>("frame_id", "base_footprint");
    frame_id_ = this->get_parameter("frame_id").as_string();
    
    // Create subscriber for cluster centers
    clusters_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/perception/clusters", 10,
        std::bind(&LidarClustersVisualizer::clustersCallback, this, std::placeholders::_1));
    
    // Create publisher for visualization markers
    markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/perception/lidar_clusters_vis", 10);
    
    RCLCPP_INFO(this->get_logger(), "LiDAR Clusters Visualizer node has been started");
    RCLCPP_INFO(this->get_logger(), "Subscribed to: /perception/clusters");
    RCLCPP_INFO(this->get_logger(), "Publishing to: /perception/lidar_clusters_vis");
    RCLCPP_INFO(this->get_logger(), "Frame ID: %s", frame_id_.c_str());
}

void LidarClustersVisualizer::clustersCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    // Create an empty MarkerArray to delete previous markers
    auto delete_array = visualization_msgs::msg::MarkerArray();
    for (auto& marker : previous_markers_) {
        marker.action = visualization_msgs::msg::Marker::DELETE;
        delete_array.markers.push_back(marker);
    }
    
    // Publish the delete command
    markers_pub_->publish(delete_array);
    
    // Clear previous marker list
    previous_markers_.clear();
    
    // Create new markers
    auto marker_array = visualization_msgs::msg::MarkerArray();
    int id = 1;
    
    const auto& cluster_data = msg->data;
    
    // Data format: [x1, y1, x2, y2, ...]
    for (size_t i = 0; i < cluster_data.size(); i += 2) {
        if (i + 1 >= cluster_data.size()) break;
        
        float x = cluster_data[i];
        float y = cluster_data[i + 1];
        
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = this->now();
        marker.ns = "lidar_clusters";
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // Set the scale of the marker (diameter in x and y, height in z)
        marker.scale.x = 0.5;  // Diameter in x direction
        marker.scale.y = 0.5;  // Diameter in y direction  
        marker.scale.z = 0.5;  // Height
        
        // Set color (white for generic clusters)
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;  // Fully opaque
        
        // Set position (using cluster center coordinates)
        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = 0.0;  // On ground level
        
        // Set orientation (identity quaternion)
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        
        // Set lifetime (2 seconds)
        marker.lifetime = rclcpp::Duration::from_seconds(2.0);
        
        marker_array.markers.push_back(marker);
        previous_markers_.push_back(marker);
    }
    
    // Publish the new markers
    markers_pub_->publish(marker_array);
    
    RCLCPP_DEBUG(this->get_logger(), "Published %zu cluster markers", marker_array.markers.size());
}