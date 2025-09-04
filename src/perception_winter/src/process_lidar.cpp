/**
 * @file process_lidar.cpp
 * @brief Source (Definition / Implementation) file for the node
 * @author Siddhesh Phadke
 */

#include "perception_winter/process_lidar.hpp"
#include <string>
#include <algorithm>
#include <cmath>
#include <optional> // Required for iterative RANSAC
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h> // Required for iterative RANSAC
#include <pcl/filters/passthrough.h>       // Required for ROI filtering
#include <pcl/ModelCoefficients.h>      // Required for iterative RANSAC

ProcessLidar::ProcessLidar() : Node("process_lidar")
{
    // Shout out
    RCLCPP_INFO(this->get_logger(), "Process Lidar Node started");

    // Initializing
    this->lidar_raw_input_sub = this->create_subscription<sensor_msgs::msg::PointCloud>(
        this->lidar_raw_input_topic,
        10,
        std::bind(&ProcessLidar::lidar_raw_sub_callback, this, std::placeholders::_1)
    );

    // For final classified cones
    this->classified_cones_output_rviz_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->classified_cones_output_rviz_topic,
        10
    );

    // Publishers for debugging visualizations
    this->ground_points_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->namespace_ + "/ground_points", 10);
    this->non_ground_points_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->namespace_ + "/non_ground_points", 10);
    this->clustered_points_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->namespace_ + "/clustered_points", 10);

    // Initialize new RANSAC parameters
    this->min_z_normal_component = 0.80;
    this->max_slope_deviation_deg = 10.0;

    // --- DEBUG LOGGER ---
    RCLCPP_INFO(this->get_logger(), "[DEBUG] Constructor finished.");
}

ProcessLidar::~ProcessLidar()
{
    // --- DEBUG LOGGER ---
    RCLCPP_INFO(this->get_logger(), "[DEBUG] Destructor called. Shutting down.");
}

void ProcessLidar::lidar_raw_sub_callback(const sensor_msgs::msg::PointCloud::SharedPtr msg)
{
    RCLCPP_INFO(this->get_logger(), "[DEBUG] ---- lidar_raw_sub_callback entered ----");
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
    // --- DEBUG LOGGER ---
    RCLCPP_INFO(this->get_logger(), "[DEBUG] After X-filter, cloud size is: %zu", cloud->points.size());

    auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PassThrough<pcl::PointXYZI> pass;
    
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-3, 3);
    pass.filter(*cloud_filtered);
    // --- DEBUG LOGGER ---
    RCLCPP_INFO(this->get_logger(), "[DEBUG] After Y-filter, cloud size is: %zu", cloud_filtered->points.size());

    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-1.0, 2.0);
    pass.filter(*cloud_filtered);
    // --- DEBUG LOGGER ---
    RCLCPP_INFO(this->get_logger(), "[DEBUG] After Z-filter, cloud size for RANSAC is: %zu", cloud_filtered->points.size());


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

    // --- DEBUG LOGGER ---
    RCLCPP_INFO(this->get_logger(), "[DEBUG] Starting RANSAC with %zu points.", remaining_cloud->points.size());

    while (remaining_cloud->points.size() > min_points_for_plane && iterations < max_iterations)
    {
        // --- DEBUG LOGGER ---
        RCLCPP_INFO(this->get_logger(), "[DEBUG] RANSAC Iteration %d...", iterations + 1);

        seg.setInputCloud(remaining_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            RCLCPP_INFO(this->get_logger(), "[HEARTBEAT] No more potential ground planes found.");
            break;
        }

        Eigen::Vector3f current_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        if (current_normal.z() < 0) { current_normal = -current_normal; }

        if (current_normal.z() < this->min_z_normal_component) {
            RCLCPP_INFO(this->get_logger(), "[HEARTBEAT] Plane rejected: too vertical (normal Z=%.2f).", current_normal.z());
            break;
        }

        if (!reference_normal.has_value()) {
            reference_normal = current_normal;
            RCLCPP_INFO(this->get_logger(), "[HEARTBEAT] Acquired reference ground plane normal.");
        } else {
            double dot_product = current_normal.dot(reference_normal.value());
            double angle_rad = std::acos(std::clamp(dot_product, -1.0, 1.0));
            double angle_deg = angle_rad * (180.0 / M_PI);
            if (angle_deg > this->max_slope_deviation_deg) {
                RCLCPP_INFO(this->get_logger(), "[HEARTBEAT] Plane rejected: slope deviates by %.2f deg.", angle_deg);
                break;
            }
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
    
    // --- DEBUG LOGGER ---
    RCLCPP_INFO(this->get_logger(), "[DEBUG] RANSAC complete. Ground points: %zu, Non-ground points: %zu", ground_cloud->size(), non_ground_cloud->size());

    // Debugger 1: Visualize ground points
    std::vector<std::vector<double>> ground_positions, ground_colors;
    for (const auto& point : ground_cloud->points) {
        ground_positions.push_back({point.x, point.y, point.z});
        ground_colors.push_back({0.0, 1.0, 0.0});
    }
    this->publishMarkerArray(visualization_msgs::msg::Marker::SPHERE, this->namespace_ + "_ground", 
        this->fixed_frame, {ground_positions, ground_colors}, this->ground_points_pub,
        true, {0.05, 0.05, 0.05}, msg->header.stamp);

    for (const auto& point : non_ground_cloud->points) {
        positions.push_back({point.x, point.y, point.z});
        intensities.push_back(point.intensity);
    }
    
    // Debugger 2: Visualize non-ground points
    std::vector<std::vector<double>> non_ground_colors;
    for (size_t i = 0; i < positions.size(); ++i) {
        non_ground_colors.push_back({1.0, 1.0, 1.0});
    }
    this->publishMarkerArray(visualization_msgs::msg::Marker::SPHERE, this->namespace_ + "_non_ground",
        this->fixed_frame, {positions, non_ground_colors}, this->non_ground_points_pub,
        true, {0.05, 0.05, 0.05}, msg->header.stamp);

    if (positions.empty()) {
        RCLCPP_INFO(this->get_logger(), "[DEBUG] No non-ground points to cluster. Exiting callback.");
        this->publishMarkerArray(visualization_msgs::msg::Marker::CYLINDER, this->namespace_,
            this->fixed_frame, {{}, {}}, this->classified_cones_output_rviz_pub,
            true, {1, 1, 0.5}, msg->header.stamp);
        this->publishMarkerArray(visualization_msgs::msg::Marker::SPHERE, this->namespace_ + "_clustered",
            this->fixed_frame, {{}, {}}, this->clustered_points_pub, true, {0.05, 0.05, 0.05}, msg->header.stamp);
        return;
    }

    // --- DEBUG LOGGER ---
    RCLCPP_INFO(this->get_logger(), "[DEBUG] Starting DBSCAN on %zu points.", positions.size());

    auto o3d_pcd = std::make_shared<open3d::geometry::PointCloud>();
    for (const auto& point : positions) {
        o3d_pcd->points_.emplace_back(point[0], point[1], point[2]);
    }

    std::vector<int> labels = o3d_pcd->ClusterDBSCAN(this->dbscan_epsilon, this->dbscan_minpoints);
    int num_labels = *std::max_element(labels.begin(), labels.end()) + 1;
    // --- DEBUG LOGGER ---
    RCLCPP_INFO(this->get_logger(), "[DEBUG] DBSCAN found %d potential clusters.", num_labels);

    std::vector<std::vector<std::vector<double>>> classified_points(num_labels);
    for (size_t index = 0; index < positions.size(); index++) {
        int label = labels[index];
        if (label == -1) { continue; }
        auto point = positions.at(index);
        classified_points[label].push_back({point[0], point[1], point[2], intensities[index]});
    }

    // Debugger 3: Visualize clustered points
    std::vector<std::vector<double>> clustered_positions, clustered_colors;
    for (const auto& cluster : classified_points) {
        for (const auto& point : cluster) {
            clustered_positions.push_back({point[0] + 2.921, point[1], point[2]});
            clustered_colors.push_back({1.0, 0.0, 1.0});
        }
    }
    this->publishMarkerArray(visualization_msgs::msg::Marker::SPHERE, this->namespace_ + "_clustered",
        this->fixed_frame, {clustered_positions, clustered_colors}, this->clustered_points_pub,
        true, {0.05, 0.05, 0.05}, msg->header.stamp);

    for (auto& cone_class : classified_points) {
        std::sort(cone_class.begin(), cone_class.end(),
            [](const std::vector<double>& v1, const std::vector<double>& v2) -> bool {
                return v1[2] > v2[2];
            });
    }

    colors.clear();
    positions.clear();
    
    // --- DEBUG LOGGER ---
    RCLCPP_INFO(this->get_logger(), "[DEBUG] Processing %zu valid clusters...", classified_points.size());
    for (auto& class_ : classified_points) {
        int class_size = class_.size();
        if (class_size < 10) { continue; }
        
        std::vector<double> intensity_vals, z_vals;
        const double CONE_BASE_RADIUS = 0.12;
        
        auto min_x_it = std::min_element(class_.begin(), class_.end(), 
            [](const std::vector<double>& a, const std::vector<double>& b) {
                return a[0] < b[0];
            });
        
        double cone_x = (*min_x_it)[0] + CONE_BASE_RADIUS + 2.921;
        double cone_y = (*min_x_it)[1];
        double cone_z = 0.1629;
        
        for (auto& pt : class_) {
            intensity_vals.push_back(pt.at(3));
            z_vals.push_back(pt.at(2));
        }

        int kernel = std::max(3, static_cast<int>(0.1 * intensity_vals.size()));
        if (kernel % 2 == 0) kernel += 1;
        
        std::vector<double> averaged_intensity_vals = this->movingAverage(intensity_vals, kernel);
        positions.push_back({cone_x, cone_y, cone_z});
        
        if (this->classifyCone(averaged_intensity_vals, z_vals)) {
            colors.push_back({1.0, 1.0, 0.0});
        } else {
            colors.push_back({0.0, 0.0, 1.0});
        }
    }

    // --- DEBUG LOGGER ---
    RCLCPP_INFO(this->get_logger(), "[DEBUG] Found %zu cones for final publishing.", positions.size());
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

    RCLCPP_INFO(this->get_logger(), "[DEBUG] ---- lidar_raw_sub_callback finished ----");
}

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
    if (!publisher) { return; }

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

bool ProcessLidar::classifyCone(const std::vector<double> & y_vals, const std::vector<double> & x_vals)
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

    return coeffs(0) > 0;
}

// bool ProcessLidar::classifyCone(const std::vector<double>& y_vals, const std::vector<double>& x_vals)
// {
//     if (y_vals.size() < 2) return false; // IRLS benefits from a few more points

//     int n = y_vals.size();
//     Eigen::MatrixXd A(n, 3);
//     Eigen::VectorXd y(n);

//     for (int i = 0; i < n; ++i) {
//         double x = x_vals.at(i);
//         A(i, 0) = x * x;
//         A(i, 1) = x;
//         A(i, 2) = 1.0;
//         y(i) = y_vals.at(i);
//     }

//     // --- Iteratively Reweighted Least Squares (IRLS) Implementation ---

//     // 1. Initial Fit: Perform a standard least-squares fit to get a first guess.
//     Eigen::Vector3d coeffs = A.colPivHouseholderQr().solve(y);

//     const int num_iterations = 10; // 3 iterations is a good balance of speed and accuracy

//     for (int iter = 0; iter < num_iterations; ++iter) {
//         // 2. Calculate Residuals: Find the error of each point from the current curve.
//         Eigen::VectorXd residuals = y - A * coeffs;

//         // 3. Calculate Weights: Down-weight outliers using the Tukey biweight function.
//         // This is the core of the robust fitting.
//         Eigen::VectorXd W = Eigen::VectorXd::Zero(n);
//         double mad = 0; // Median Absolute Deviation
//         std::vector<double> abs_residuals;
//         for(int i=0; i<n; ++i) { abs_residuals.push_back(std::abs(residuals(i))); }
//         std::nth_element(abs_residuals.begin(), abs_residuals.begin() + n / 2, abs_residuals.end());
//         mad = abs_residuals[n/2];
        
//         if (mad < 1e-6) break; // Avoid division by zero if all residuals are zero

//         const double c = 4.685 * mad; // Tukey's tuning constant

//         for (int i = 0; i < n; ++i) {
//             if (std::abs(residuals(i)) < c) {
//                 double term = 1 - std::pow(residuals(i) / c, 2);
//                 W(i) = term * term;
//             } else {
//                 W(i) = 0; // Completely reject extreme outliers
//             }
//         }

//         // 4. Re-fit: Perform a Weighted Least Squares fit with the new weights.
//         Eigen::MatrixXd AtW = A.transpose() * W.asDiagonal();
//         coeffs = (AtW * A).ldlt().solve(AtW * y);
//     }

//     // 5. Final Classification: Return the curvature of the final, robustly-fitted parabola.
//     return coeffs(0) > 0;
// }

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