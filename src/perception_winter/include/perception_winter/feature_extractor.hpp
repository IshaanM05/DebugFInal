/**
* @name feature_extractor.hpp
* @brief Header file for the FeatureExtractor node.
*/

#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/point_cloud.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <Eigen/Dense>

class FeatureExtractorNode : public rclcpp::Node {
private:
    // --- Subscribers ---
    rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr lidar_raw_input_sub;

    // --- Publishers for visualization ---
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ground_points_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr non_ground_points_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr clustered_points_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr classified_cones_output_rviz_pub;

    // --- File writer for extracted features ---
    std::ofstream features_file_;

    // --- Parameters ---
    std::string current_cone_label_;
    std::string fixed_frame = "Fr1A";
    std::string namespace_ = "feature_extractor";

    // --- RANSAC / DBSCAN params ---
    double ransac_threshold = 0.03;
    double min_z_normal_component = 0.80;
    double max_slope_deviation_deg = 10.0;
    double dbscan_epsilon = 0.3;
    int dbscan_minpoints = 2;

      // Topics
    const std::string lidar_raw_input_topic = "/carmaker/pointcloud"; // Lidar data input
  // const std::string lidar_raw_output_rviz_topic = this->namespace_+"/lidar/raw"; // Lidar raw data output topic
    const std::string classified_cones_output_rviz_topic = this->namespace_ + "/classified_cones"; // Final output

    // --- Core Callbacks ---
    void lidar_raw_sub_callback(const sensor_msgs::msg::PointCloud::SharedPtr msg);

    // --- Cone classification (color detection logic) ---
    // int classifyCone(const std::vector<std::vector<double>>& cluster,
    //                  const std::vector<double>& averaged_intensity);

    //bool classifyCone(const std::vector<double> & y_vals, const std::vector<double> & x_vals);
    bool classifyCone(const std::vector<double>& averaged_intensity, const std::vector<double>& z_vals, const std::vector<std::vector<double>>& cluster);


    // --- Helper: Moving average smoother ---
    std::vector<double> movingAverage(const std::vector<double>& data, int kernel);

    // --- Helper: Publish RViz markers ---
    void publishMarkerArray(
        visualization_msgs::msg::Marker::_type_type type,
        std::string ns,
        std::string frame_id,
        std::vector<std::vector<std::vector<double>>> positions_colours,
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher,
        bool del_markers,
        std::vector<double> scales,
        const rclcpp::Time& stamp
    );

public:
    FeatureExtractorNode();
    ~FeatureExtractorNode();
};

#endif // FEATURE_EXTRACTOR_HPP
