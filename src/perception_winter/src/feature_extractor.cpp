/**
 * @file feature_extractor.cpp
 * @brief Node to process LiDAR data and extract cone features to a CSV file.
 */

#include "perception_winter/feature_extractor.hpp"
#include <string>
#include <algorithm>
#include <cmath>
#include <optional>
#include <numeric> 
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>

FeatureExtractorNode::FeatureExtractorNode() : Node("feature_extractor_node")
{
    // Declare and get the parameter for the cone label
    this->declare_parameter<std::string>("cone_label", "UNKNOWN");
    this->get_parameter("cone_label", current_cone_label_);

    RCLCPP_INFO(this->get_logger(), "Feature Extractor Node started. Collecting data for label: '%s'", current_cone_label_.c_str());

    // Open the CSV file and write the header
    feature_file_.open("cone_features.csv", std::ios_base::app); // Open in append mode
    if (feature_file_.tellp() == 0) { // Write header only if file is new/empty
        feature_file_ << "height,point_count,avg_intensity,intensity_std_dev,"
                      << "avg_intensity_bottom,avg_intensity_middle,avg_intensity_top,"
                      << "band_avg_std_dev,cone_type\n";
    }

    // Initialize the subscriber
    lidar_raw_input_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud>(
        "/carmaker/pointcloud", 10, std::bind(&FeatureExtractorNode::lidar_raw_sub_callback, this, std::placeholders::_1));
}

FeatureExtractorNode::~FeatureExtractorNode()
{
    if (feature_file_.is_open()) {
        feature_file_.close();
        RCLCPP_INFO(this->get_logger(), "Feature data saved to cone_features.csv");
    }
}

void FeatureExtractorNode::lidar_raw_sub_callback(const sensor_msgs::msg::PointCloud::SharedPtr msg)
{
    // --- 1. PRE-PROCESSING AND GROUND REMOVAL ---
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    cloud->points.reserve(msg->points.size());
    for (size_t i=0; i<msg->points.size(); ++i) { 
        const auto& pt = msg->points[i];
        if (pt.x > 0) { 
            cloud->points.push_back({pt.x, pt.y, pt.z, msg->channels[0].values[i]}); 
        }
    }

    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y"); pass.setFilterLimits(-4.0, 4.0);
    pass.filter(*cloud);
    
    auto non_ground_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    auto remaining_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>(*cloud);
    
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.03);
    int iterations = 0;
    while (remaining_cloud->points.size() > 150 && iterations < 5)
    {
        seg.setInputCloud(remaining_cloud);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.empty()) { break; }
        Eigen::Vector3f normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        if (normal.z() < 0) { normal = -normal; }
        if (normal.z() < 0.90) { break; }
        
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*remaining_cloud);
        iterations++;
    }
    *non_ground_cloud = *remaining_cloud;

    // --- 2. CLUSTERING ---
    std::vector<std::vector<double>> positions;
    std::vector<double> intensities;
    for (const auto& point : *non_ground_cloud) { positions.push_back({point.x, point.y, point.z}); intensities.push_back(point.intensity); }
    if (positions.empty()) { return; }
    open3d::geometry::PointCloud o3d_pcd;
    for (const auto& p : positions) { o3d_pcd.points_.push_back({p[0], p[1], p[2]}); }
    std::vector<int> labels = o3d_pcd.ClusterDBSCAN(0.3, 2);
    int num_labels = labels.empty() ? 0 : *std::max_element(labels.begin(), labels.end()) + 1;
    std::vector<std::vector<std::vector<double>>> classified_points(num_labels);
    for (size_t i = 0; i < positions.size(); i++) { if (labels[i] != -1) { classified_points[labels[i]].push_back({positions[i][0], positions[i][1], positions[i][2], intensities[i]}); }}

    // --- 3. FEATURE EXTRACTION AND SAVING ---
    RCLCPP_INFO(this->get_logger(), "Found %zu clusters. Extracting features...", classified_points.size());
    for (const auto& cluster : classified_points) {
        if (cluster.size() < 10) continue;
        
        // Calculate the feature vector for this cluster
        std::vector<double> features = extractFeatures(cluster);

        // Write the features and the current label to the file
        for (size_t i = 0; i < features.size(); ++i) {
            feature_file_ << features[i] << ",";
        }
        feature_file_ << current_cone_label_ << "\n";
    }
}

std::vector<double> FeatureExtractorNode::extractFeatures(const std::vector<std::vector<double>>& cone_cluster)
{
    // Sort by Z to easily find height and bands
    auto sorted_cluster = cone_cluster;
    std::sort(sorted_cluster.begin(), sorted_cluster.end(), 
        [](const auto& a, const auto& b){ return a[2] < b[2]; });

    // --- Feature 1 & 2: Height and Point Count ---
    double height = sorted_cluster.back().at(2) - sorted_cluster.front().at(2);
    double point_count = static_cast<double>(cone_cluster.size());

    // --- Feature 3 & 4: Overall Intensity Stats ---
    double intensity_sum = 0.0;
    std::vector<double> intensities;
    intensities.reserve(cone_cluster.size());
    for (const auto& pt : cone_cluster) { 
        intensities.push_back(pt[3]);
        intensity_sum += pt[3];
    }
    double avg_intensity_overall = intensity_sum / cone_cluster.size();

    double intensity_sq_sum = 0.0;
    for (const auto& intensity : intensities) { intensity_sq_sum += (intensity - avg_intensity_overall) * (intensity - avg_intensity_overall); }
    double intensity_std_dev = std::sqrt(intensity_sq_sum / cone_cluster.size());

    // --- Feature 5, 6, 7: Band-based Average Intensities ---
    double z_low = sorted_cluster.front().at(2);
    double band_height = height / 3.0;
    std::vector<std::vector<double>> band_intensities(3);
    for (const auto& pt : cone_cluster) {
        if (pt[2] < z_low + band_height) band_intensities[0].push_back(pt[3]);
        else if (pt[2] < z_low + 2 * band_height) band_intensities[1].push_back(pt[3]);
        else band_intensities[2].push_back(pt[3]);
    }
    
    std::vector<double> band_avg_intensities;
    for(const auto& band : band_intensities){
        if(band.empty()){ band_avg_intensities.push_back(0.0); } 
        else { band_avg_intensities.push_back(std::accumulate(band.begin(), band.end(), 0.0) / band.size()); }
    }
    
    // --- Feature 8: Deviation Across Bands ---
    double avg_of_band_avgs = (band_avg_intensities[0] + band_avg_intensities[1] + band_avg_intensities[2]) / 3.0;
    double band_deviation_sum = 0.0;
    for(const auto& band_avg : band_avg_intensities) {
        band_deviation_sum += (band_avg - avg_of_band_avgs) * (band_avg - avg_of_band_avgs);
    }
    double band_avg_std_dev = std::sqrt(band_deviation_sum / 3.0);

    return {height, point_count, avg_intensity_overall, intensity_std_dev, 
            band_avg_intensities[0], band_avg_intensities[1], band_avg_intensities[2],
            band_avg_std_dev};
}