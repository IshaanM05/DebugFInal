/**
* @name feature_extractor.hpp
* @brief Header file for the data extraction node.
*/

#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/point_cloud.hpp>
#include <vector>
#include <string>
#include <fstream> // Required for writing to a file

class FeatureExtractorNode : public rclcpp::Node {
private:
    // Subscriber to the raw LiDAR data
    rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr lidar_raw_input_sub_;

    // File stream to write the feature data to a .csv file
    std::ofstream feature_file_;

    // Parameter to hold the label for the data being collected (e.g., "BLUE", "SMALL_ORANGE")
    std::string current_cone_label_;

    // The main callback function that processes the point cloud
    void lidar_raw_sub_callback(const sensor_msgs::msg::PointCloud::SharedPtr msg);

    // The function that calculates the feature vector from a cluster
    std::vector<double> extractFeatures(const std::vector<std::vector<double>>& cone_cluster);

public:
    FeatureExtractorNode();
    ~FeatureExtractorNode();
};

#endif // FEATURE_EXTRACTOR_HPP