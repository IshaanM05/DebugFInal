/**
 * @file process_lidar.cpp
 * @brief Source (Definition / Implementation) file for the node
 */

#include "perception_winter/process_lidar.hpp"
#include <string>
#include <algorithm>
#include <cmath>
#include <optional>
#include <fstream>
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>

ProcessLidar::ProcessLidar() : Node("process_lidar")
{
    RCLCPP_INFO(this->get_logger(), "Process Lidar Node started");

    this->lidar_raw_input_sub = this->create_subscription<sensor_msgs::msg::PointCloud>(
        this->lidar_raw_input_topic,
        10,
        std::bind(&ProcessLidar::lidar_raw_sub_callback, this, std::placeholders::_1)
    );

    this->classified_cones_output_rviz_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->classified_cones_output_rviz_topic,
        10
    );

    this->ground_points_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->namespace_ + "/ground_points", 10);
    this->non_ground_points_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->namespace_ + "/non_ground_points", 10);
    this->clustered_points_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->namespace_ + "/clustered_points", 10);

    this->min_z_normal_component = 0.80;
    this->max_slope_deviation_deg = 10.0;

    RCLCPP_INFO(this->get_logger(), "[DEBUG] Constructor finished.");
}

ProcessLidar::~ProcessLidar()
{
    RCLCPP_INFO(this->get_logger(), "[DEBUG] Destructor called. Shutting down.");
}

void ProcessLidar::lidar_raw_sub_callback(const sensor_msgs::msg::PointCloud::SharedPtr msg)
{
    static int frame_idx = 0;

    RCLCPP_INFO(this->get_logger(), "[DEBUG] ---- lidar_raw_sub_callback entered (frame %d) ----", frame_idx);
    RCLCPP_INFO(this->get_logger(), "[DEBUG] Received point cloud with %zu points.", msg->points.size());

    std::vector<std::vector<double>> positions, colors;
    std::vector<double> intensities;

    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    cloud->points.reserve(msg->points.size());
    for (size_t i = 0; i < msg->points.size(); ++i) {
        const auto& pt = msg->points[i];
        if (pt.x > 0) {
            const auto& intensity = msg->channels[0].values[i];
            pcl::PointXYZI p;
            p.x = pt.x;
            p.y = pt.y;
            p.z = pt.z;
            p.intensity = intensity;
            cloud->points.push_back(p);
        }
    }

    auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-3, 3);
    pass.filter(*cloud_filtered);

    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-1.0, 2.0);
    pass.filter(*cloud_filtered);

    auto non_ground_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    auto ground_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    auto remaining_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>(*cloud_filtered);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::ExtractIndices<pcl::PointXYZI> extract;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(this->ransac_threshold);

    int iterations = 0;
    const int max_iterations = 5;
    const size_t min_points_for_plane = 350;
    std::optional<Eigen::Vector3f> reference_normal;

    while (remaining_cloud->points.size() > min_points_for_plane && iterations < max_iterations)
    {
        seg.setInputCloud(remaining_cloud);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.empty()) break;

        Eigen::Vector3f current_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        if (current_normal.z() < 0) current_normal = -current_normal;
        if (current_normal.z() < this->min_z_normal_component) break;

        if (!reference_normal.has_value()) {
            reference_normal = current_normal;
        } else {
            double dot_product = current_normal.dot(reference_normal.value());
            double angle_rad = std::acos(std::clamp(dot_product, -1.0, 1.0));
            double angle_deg = angle_rad * (180.0 / M_PI);
            if (angle_deg > this->max_slope_deviation_deg) break;
        }

        auto current_ground_plane = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*current_ground_plane);
        *ground_cloud += *current_ground_plane;

        extract.setNegative(true);
        auto next_remaining_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        extract.filter(*next_remaining_cloud);
        remaining_cloud = next_remaining_cloud;
        iterations++;
    }

    *non_ground_cloud = *remaining_cloud;

    for (const auto& point : non_ground_cloud->points) {
        positions.push_back({point.x, point.y, point.z});
        intensities.push_back(point.intensity);
    }

    if (positions.empty()) {
        this->publishMarkerArray(visualization_msgs::msg::Marker::CYLINDER, this->namespace_,
            this->fixed_frame, {{}, {}}, this->classified_cones_output_rviz_pub,
            true, {1, 1, 0.5}, msg->header.stamp);
        frame_idx++;
        return;
    }

    auto o3d_pcd = std::make_shared<open3d::geometry::PointCloud>();
    for (const auto& point : positions) {
        o3d_pcd->points_.emplace_back(point[0], point[1], point[2]);
    }

    std::vector<int> labels = o3d_pcd->ClusterDBSCAN(this->dbscan_epsilon, this->dbscan_minpoints);
    int num_labels = *std::max_element(labels.begin(), labels.end()) + 1;

    std::vector<std::vector<std::vector<double>>> classified_points(num_labels);
    for (size_t index = 0; index < positions.size(); index++) {
        int label = labels[index];
        if (label == -1) continue;
        auto point = positions.at(index);
        classified_points[label].push_back({point[0], point[1], point[2], intensities[index]});
    }

    colors.clear();
    positions.clear();

    std::ofstream debug_file("/home/ishaan/Desktop/DebugFInal/lidar_cluster_debug.csv", std::ios::app);
    if (debug_file.is_open()) {
        debug_file << "Frame," << frame_idx << "\n";
    }

    // --- Process each cluster ---
    for (size_t cid = 0; cid < classified_points.size(); ++cid) {
        auto& class_ = classified_points[cid];
        if (class_.size() < 10) continue;

        std::vector<double> intensity_vals, z_vals;
        const double CONE_BASE_RADIUS = 0.12;

        auto min_x_it = std::min_element(class_.begin(), class_.end(),
            [](const std::vector<double>& a, const std::vector<double>& b) {
                return a[0] < b[0];
            });

        double cone_x = (*min_x_it)[0] + CONE_BASE_RADIUS;
        double cone_y = (*min_x_it)[1];
        double cone_z = 0.025;

        double z_min = (*std::min_element(class_.begin(), class_.end(),
                        [](const std::vector<double>& a, const std::vector<double>& b){ return a[2] < b[2]; }))[2];
        double z_max = (*std::max_element(class_.begin(), class_.end(),
                        [](const std::vector<double>& a, const std::vector<double>& b){ return a[2] < b[2]; }))[2];
        double band_height = (z_max - z_min) / 3.0;

        std::vector<std::vector<double>> band_intensities(3); // [band][intensity]

        for (auto& pt : class_) {
            double z = pt[2];
            double intensity = pt[3];
            intensity_vals.push_back(intensity);
            z_vals.push_back(z);

            int band_idx = std::min(2, static_cast<int>((z - z_min) / band_height));
            band_intensities[band_idx].push_back(intensity);

            if (debug_file.is_open()) {
                debug_file << "Raw," << frame_idx << "," << cid << ","
                        << pt[0] << "," << pt[1] << "," << pt[2] << "," << pt[3] << "\n";
            }
        }

        int kernel = std::max(3, static_cast<int>(0.1 * intensity_vals.size()));
        if (kernel % 2 == 0) kernel += 1;
        std::vector<double> averaged_intensity_vals = this->movingAverage(intensity_vals, kernel);

        std::string detected_color;
        if (this->classifyCone(averaged_intensity_vals, z_vals)) {
            colors.push_back({1.0, 1.0, 0.0}); // yellow
            detected_color = "yellow";
        } else {
            colors.push_back({0.0, 0.0, 1.0}); // blue
            detected_color = "blue";
        }
        positions.push_back({cone_x, cone_y, cone_z});

        if (debug_file.is_open()) {
            for (size_t b = 0; b < 3; ++b) {
                for (double inten : band_intensities[b]) {
                    debug_file << "Band," << frame_idx << "," << cid << "," << b
                            << "," << inten << "\n";
                }
            }
            debug_file << "Cluster," << frame_idx << "," << cid << ","
                       << cone_x << "," << cone_y << "," << cone_z << ","
                       << detected_color << "\n";
        }
    }

    this->publishMarkerArray(
        visualization_msgs::msg::Marker::CYLINDER,
        this->namespace_,
        this->fixed_frame,
        {positions, colors},
        this->classified_cones_output_rviz_pub,
        true,
        {0.1, 0.1, 0.5},
        msg->header.stamp
    );

    frame_idx++;
    RCLCPP_INFO(this->get_logger(), "[DEBUG] ---- lidar_raw_sub_callback finished ----");
}

// --- Publish markers function ---
void ProcessLidar::publishMarkerArray(
    visualization_msgs::msg::Marker::_type_type type,
    std::string ns,
    std::string frame_id,
    std::vector<std::vector<std::vector<double>>> positions_colours,
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher,
    bool del_markers,
    std::vector<double> scales,
    const rclcpp::Time& stamp)
{
    if (!publisher) return;

    visualization_msgs::msg::MarkerArray marker_array;

    if (del_markers) {
        visualization_msgs::msg::Marker del_marker;
        del_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(del_marker);
    }

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.ns = ns;
    marker.type = type;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = scales.at(0);
    marker.scale.y = scales.at(1);
    marker.scale.z = scales.at(2);

    for (size_t i = 0; i < positions_colours.at(0).size(); i++) {
        marker.id = i;
        marker.pose.position.x = positions_colours.at(0).at(i).at(0);
        marker.pose.position.y = positions_colours.at(0).at(i).at(1);
        marker.pose.position.z = positions_colours.at(0).at(i).at(2);
        marker.color.a = 1.0;
        marker.color.r = positions_colours.at(1).at(i).at(0);
        marker.color.g = positions_colours.at(1).at(i).at(1);
        marker.color.b = positions_colours.at(1).at(i).at(2);
        marker_array.markers.push_back(marker);
    }

    publisher->publish(marker_array);
}

// --- Cone classification based on intensity profile ---
bool ProcessLidar::classifyCone(const std::vector<double>& y_vals, const std::vector<double>& x_vals)
{
    if (y_vals.size() < 3) return false;

    int n = y_vals.size();
    Eigen::MatrixXd A(n, 3);
    Eigen::VectorXd y(n);

    for (int i = 0; i < n; ++i) {
        double x = x_vals.at(i);
        A(i, 0) = x * x;
        A(i, 1) = x;
        A(i, 2) = 1.0;
        y(i) = y_vals.at(i);
    }

    Eigen::Vector3d coeffs = A.colPivHouseholderQr().solve(y);
    return coeffs(0) > 0; // stub logic: yellow if true, blue otherwise
}

// --- Moving average smoothing ---
std::vector<double> ProcessLidar::movingAverage(const std::vector<double>& data, int kernel)
{
    int n = data.size();
    std::vector<double> result(n, 0.0);
    if (kernel < 1) return data;

    int half = kernel / 2;
    for (int i = 0; i < n; ++i) {
        int start = std::max(0, i - half);
        int end = std::min(n - 1, i + half);
        double sum = 0.0;
        for (int j = start; j <= end; ++j) {
            sum += data[j];
        }
        result[i] = sum / (end - start + 1);
    }
    return result;
}
