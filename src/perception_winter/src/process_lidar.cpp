/**
 * @file process_lidar.cpp
 * @brief Optimized LiDAR processing node implementation
 * @author Siddhesh Phadke
 */

#include "perception_winter/process_lidar.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <open3d/Open3D.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>
#include <chrono>

ProcessLidar::ProcessLidar() : Node("process_lidar")
{
    // Initialize reusable containers
    // CHANGE: Commented out to prevent reuse. Fresh clouds will be created in the callback.
    // reusable_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    // reusable_cloud_filtered_ = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    // Subscribers
    lidar_raw_input_sub_ = create_subscription<sensor_msgs::msg::PointCloud>(
        LIDAR_RAW_TOPIC, 10,
        [this](const sensor_msgs::msg::PointCloud::SharedPtr msg) {
            lidarRawCallback(msg);
        });

    lidar_raw_input_sub2_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        LIDAR_RAW_TOPIC2, 10,
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            lidarRawCallback2(msg);
        });

    // Publishers
    detected_cones_pub_ = create_publisher<dv_msgs::msg::IndexedTrack>("/perception/cones", 10);
    // filtered_points_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/perception/filtered_points", 10);
    // lidar_clusters_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/perception/clusters", 10);

    RCLCPP_INFO(get_logger(), "Optimized LiDAR Node started");
}

ProcessLidar::~ProcessLidar()
{
    RCLCPP_INFO(get_logger(), "LiDAR Node shutdown");
}

// PointCloud2 message processing
std::vector<ProcessLidar::Point4D> ProcessLidar::extractPointsFromPointCloud2(
    const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
{
    std::vector<Point4D> points;
    points.reserve(cloud_msg->width * cloud_msg->height);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");
    
    // Check for intensity field once
    bool has_intensity = false;
    for (const auto& field : cloud_msg->fields) {
        if (field.name == "intensity") {
            has_intensity = true;
            break;
        }
    }
    
    // Initialize intensity iterator only if intensity field exists
    std::optional<sensor_msgs::PointCloud2ConstIterator<float>> iter_intensity_opt;
    if (has_intensity) {
        iter_intensity_opt.emplace(*cloud_msg, "intensity");
    }

    // Single pass extraction
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        double intensity = 0.0;
        if (has_intensity && iter_intensity_opt.has_value()) {
            intensity = *(*iter_intensity_opt);
            ++(*iter_intensity_opt);
        }
        points.push_back({*iter_x, *iter_y, *iter_z, intensity});
    }

    return points;
}

// PointCloud message processing
std::vector<ProcessLidar::Point4D> ProcessLidar::extractPointsFromPointCloud(
    const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg)
{
    std::vector<Point4D> points;
    points.reserve(cloud_msg->points.size());

    bool has_intensity = !cloud_msg->channels.empty() && 
                        cloud_msg->channels[0].values.size() == cloud_msg->points.size();

    // Single pass extraction with bounds checking
    for (size_t i = 0; i < cloud_msg->points.size(); ++i) {
        const auto& pt = cloud_msg->points[i];
        points.push_back({pt.x, pt.y, pt.z, has_intensity ? cloud_msg->channels[0].values[i] : 0.0});
    }

    return points;
}

// Main processing pipeline
void ProcessLidar::processPointCloudData(std::vector<Point4D>& points, const std_msgs::msg::Header& header)
{
    (void)header; // Mark as unused to suppress warning
    
    if (points.empty()) return;

    // --- CHANGE HERE ---
    // Create fresh, local point clouds for this specific scan to avoid stale metadata issues.
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    auto pipeline_start = std::chrono::steady_clock::now();
    
    // Stage 1: Filtering
    // reusable_cloud_->clear(); // No longer needed
    if (!filterCarBodyAndROI(points, cloud)) { // Use local 'cloud'
        RCLCPP_DEBUG(get_logger(), "No points after car body and ROI filtering");
        return;
    }

    // Stage 2: Ground removal
    // reusable_cloud_filtered_->clear(); // No longer needed
    if (!removeGroundPlane(cloud, cloud_filtered)) { // Use local 'cloud' and 'cloud_filtered'
        RCLCPP_DEBUG(get_logger(), "No points after ground removal");
        return;
    }

    // Publish filtered points for visualization
    // publishFilteredPoints(cloud_filtered);

    // Stage 3: Clustering
    auto clusters = clusterPoints(cloud_filtered); // Use local 'cloud_filtered'
    if (clusters.empty()) {
        RCLCPP_DEBUG(get_logger(), "No clusters found");
        return;
    }

    // Stage 4: Cluster filtering
    auto filtered_clusters = filterClustersBySize(clusters);
    if (filtered_clusters.empty()) {
        RCLCPP_DEBUG(get_logger(), "No valid clusters after size filtering");
        return;
    }

    // Stage 5: Cone detection
    std::vector<Point3D> cone_positions;
    std::vector<int> cone_colors;
    detectConesInClusters(filtered_clusters, cone_positions, cone_colors);

    // Publish results
    // publishLidarClusters(cone_positions);
    publishDetectedCones(cone_positions, cone_colors);

    auto pipeline_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(pipeline_end - pipeline_start);
    RCLCPP_DEBUG(get_logger(), "Processing completed in %ld ms", duration.count());
}

// Optimized filtering stage - MATCHING WORKING CODE EXACTLY
bool ProcessLidar::filterCarBodyAndROI(const std::vector<Point4D>& input_points, 
                                      pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud)
{
    output_cloud->reserve(input_points.size());
    
    // First pass: Car body filtering
    for (const auto& point : input_points) {
        bool is_valid_point = (point[0] > CAR_FRONT_X) || (std::abs(point[1]) > CAR_SIDE_Y);

        if (point[0] > 0 && is_valid_point) {
            pcl::PointXYZI pcl_point;
            pcl_point.x = point[0];
            pcl_point.y = point[1];
            pcl_point.z = point[2];
            pcl_point.intensity = point[3];
            output_cloud->push_back(pcl_point);
        }
    }

    // Second pass: ROI filtering using PassThrough (matching working code)
    auto cloud_filtered_pass = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PassThrough<pcl::PointXYZI> pass;

    pass.setInputCloud(output_cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(ROI_Y_MIN, ROI_Y_MAX);
    pass.filter(*cloud_filtered_pass);

    pass.setInputCloud(cloud_filtered_pass);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(ROI_Z_MIN, ROI_Z_MAX);
    pass.filter(*output_cloud);
    
    return !output_cloud->empty();
}

// Optimized ground removal - MATCHING WORKING CODE EXACTLY
bool ProcessLidar::removeGroundPlane(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloud)
{
    if (cloud->empty()) return false;

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::ExtractIndices<pcl::PointXYZI> extract;

    // Configure segmentation
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(RANSAC_THRESHOLD);

    auto remaining_cloud = cloud;
    auto ground_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    
    std::optional<Eigen::Vector3f> reference_normal;
    int iterations = 0;

    while (remaining_cloud->size() > MIN_POINTS_FOR_PLANE && iterations < MAX_GROUND_ITERATIONS) {
        // Dynamic max iterations like working code
        int dynamic_max_iter = std::min(static_cast<int>(remaining_cloud->size() / 200), MAX_GROUND_ITERATIONS);
        if (iterations >= dynamic_max_iter) break;

        seg.setInputCloud(remaining_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) break;

        Eigen::Vector3f current_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        if (current_normal.z() < 0) current_normal = -current_normal;

        if (current_normal.z() < MIN_Z_NORMAL_COMPONENT) break;

        if (reference_normal.has_value()) {
            double dot_product = current_normal.dot(reference_normal.value());
            double angle_rad = std::acos(std::clamp(dot_product, -1.0, 1.0));
            double angle_deg = angle_rad * (180.0 / M_PI);
            if (angle_deg > MAX_SLOPE_DEVIATION_DEG) break;
        } else {
            reference_normal = current_normal;
        }

        // Extract ground plane
        auto current_ground_plane = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*current_ground_plane);
        *ground_cloud += *current_ground_plane;

        // Extract remaining points
        auto next_remaining = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        extract.setNegative(true);
        extract.filter(*next_remaining);
        remaining_cloud = next_remaining;
        iterations++;
    }

    *non_ground_cloud = *remaining_cloud;
    return !non_ground_cloud->empty();
}

// Optimized clustering
std::vector<ProcessLidar::Cluster> ProcessLidar::clusterPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
    if (cloud->empty()) return {};

    // Convert to Open3D format efficiently
    auto o3d_pcd = std::make_shared<open3d::geometry::PointCloud>();
    o3d_pcd->points_.reserve(cloud->size());
    
    for (const auto& point : cloud->points) {
        o3d_pcd->points_.emplace_back(point.x, point.y, point.z);
    }

    auto labels = o3d_pcd->ClusterDBSCAN(DBSCAN_EPSILON, DBSCAN_MINPOINTS, false);
    
    // Group points by cluster efficiently
    int max_label = 0;
    if (!labels.empty()) {
        max_label = *std::max_element(labels.begin(), labels.end());
    }
    
    std::vector<Cluster> clusters(max_label + 1);
    for (size_t i = 0; i < labels.size(); ++i) {
        int label = labels[i];
        if (label >= 0) {
            const auto& point = cloud->points[i];
            clusters[label].push_back({point.x, point.y, point.z, point.intensity});
        }
    }

    // Remove empty clusters
    clusters.erase(std::remove_if(clusters.begin(), clusters.end(),
        [](const Cluster& c) { return c.empty(); }), clusters.end());

    return clusters;
}

// Optimized cluster filtering - MATCHING WORKING CODE EXACTLY
std::vector<ProcessLidar::Cluster> ProcessLidar::filterClustersBySize(const std::vector<Cluster>& clusters)
{
    std::vector<Cluster> valid_clusters;
    valid_clusters.reserve(clusters.size());

    for (const auto& cluster : clusters) {
        if (cluster.size() < 4) continue; // Using 4 to match stable code logic

        // Compute bounding box exactly like working code
        double min_x = cluster[0][0], max_x = cluster[0][0];
        double min_y = cluster[0][1], max_y = cluster[0][1];
        double min_z = cluster[0][2], max_z = cluster[0][2];

        for (const auto& point : cluster) {
            min_x = std::min(min_x, point[0]); max_x = std::max(max_x, point[0]);
            min_y = std::min(min_y, point[1]); max_y = std::max(max_y, point[1]);
            min_z = std::min(min_z, point[2]); max_z = std::max(max_z, point[2]);
        }

        double height = max_z - min_z;
        double width = std::max(max_x - min_x, max_y - min_y);

        // Use the EXACT same pruning criteria as working code
        if (height >= 0.15 && height <= 0.4 && width <= 0.45) {
            valid_clusters.push_back(cluster);
        }
    }

    return valid_clusters;
}

// Optimized cone detection - MATCHING WORKING CODE EXACTLY
void ProcessLidar::detectConesInClusters(const std::vector<Cluster>& clusters,
                                        std::vector<Point3D>& positions,
                                        std::vector<int>& colors)
{
    positions.reserve(clusters.size());
    colors.reserve(clusters.size());

    for (const auto& cluster : clusters) {
        // Sort by Z for intensity analysis (like working code)
        auto sorted_cluster = cluster;
        std::sort(sorted_cluster.begin(), sorted_cluster.end(),
            [](const Point4D& a, const Point4D& b) { return a[2] > b[2]; });

        // Calculate cone position using working code's method
        auto cone_pos = calculateConePosition(sorted_cluster);
        positions.push_back(cone_pos);

        // Extract intensities and z-values
        std::vector<double> intensities, z_values;
        intensities.reserve(sorted_cluster.size());
        z_values.reserve(sorted_cluster.size());
        
        for (const auto& point : sorted_cluster) {
            intensities.push_back(point[3]);
            z_values.push_back(point[2]);
        }

        // Apply moving average like working code
        int kernel = std::max(3, static_cast<int>(0.1 * intensities.size()));
        if (kernel % 2 == 0) kernel += 1;
        std::vector<double> averaged_intensities = movingAverage(intensities, kernel);

        // Classify cone with smoothed intensities
        colors.push_back(classifyCone(averaged_intensities, z_values) ? 
                        dv_msgs::msg::IndexedCone::YELLOW : 
                        dv_msgs::msg::IndexedCone::BLUE);
    }
}

// Optimized cone position calculation - MATCHING WORKING CODE EXACTLY
ProcessLidar::Point3D ProcessLidar::calculateConePosition(const Cluster& cluster)
{
    // Compute min/max like working code
    double min_x = cluster[0][0], max_x = cluster[0][0];
    double min_y = cluster[0][1], max_y = cluster[0][1];
    for (const auto& point : cluster) {
        min_x = std::min(min_x, point[0]);
        max_x = std::max(max_x, point[0]);
        min_y = std::min(min_y, point[1]);
        max_y = std::max(max_y, point[1]);
    }

    // Compute median using working code's method
    double median_x = getMedian(cluster, 0);
    double median_y = getMedian(cluster, 1);

    // Use EXACT same weighting as working code
    constexpr double w_median = 0.7;
    constexpr double w_min_x = 0.3;
    constexpr double w_min_y = 0.3;

    double cone_x = w_median * median_x + w_min_x * (min_x + CONE_BASE_RADIUS);
    double cone_y = w_median * median_y + w_min_y * (min_y + CONE_BASE_RADIUS);

    return {cone_x, cone_y, CONE_HEIGHT};
}

// Optimized median calculation - MATCHING WORKING CODE EXACTLY
double ProcessLidar::getMedian(const Cluster& points, size_t idx) const
{
    if (points.empty()) return 0.0;

    // Create indices and use nth_element like working code
    std::vector<size_t> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0);

    size_t mid = indices.size() / 2;
    std::nth_element(indices.begin(), indices.begin() + mid, indices.end(),
        [&points, idx](size_t a, size_t b) { return points[a][idx] < points[b][idx]; });

    double median = points[indices[mid]][idx];

    // For even-sized clusters, average middle two (like working code)
    if (indices.size() % 2 == 0 && mid > 0) {
        auto max_it = std::max_element(indices.begin(), indices.begin() + mid,
            [&points, idx](size_t a, size_t b) { return points[a][idx] < points[b][idx]; });
        median = 0.5 * (median + points[*max_it][idx]);
    }

    return median;
}

// Cone classification using quadratic fitting - MATCHING WORKING CODE EXACTLY
bool ProcessLidar::classifyCone(const std::vector<double> &y_vals, const std::vector<double> &x_vals)
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

// Moving average filter - MATCHING WORKING CODE EXACTLY
std::vector<double> ProcessLidar::movingAverage(const std::vector<double> &data, int kernel)
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

// Publish filtered points for visualization
// void ProcessLidar::publishFilteredPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
// {
//     if (!filtered_points_pub_ || cloud->points.empty()) return;

//     auto message = std_msgs::msg::Float32MultiArray();
//     for (const auto& point : cloud->points) {
//         message.data.push_back(static_cast<float>(point.x));
//         message.data.push_back(static_cast<float>(point.y));
//         message.data.push_back(static_cast<float>(point.z));
//     }
//     filtered_points_pub_->publish(message);
// }

// // Publish cluster centers
// void ProcessLidar::publishLidarClusters(const std::vector<Point3D>& cluster_centers)
// {
//     if (!lidar_clusters_pub_ || cluster_centers.empty()) return;

//     auto message = std_msgs::msg::Float32MultiArray();
//     for (const auto& center : cluster_centers) {
//         message.data.push_back(static_cast<float>(center[0]));
//         message.data.push_back(static_cast<float>(center[1]));
//     }
//     lidar_clusters_pub_->publish(message);
// }

// Publish detected cones - PRESERVING CURRENT OUTPUT FORMAT
void ProcessLidar::publishDetectedCones(const std::vector<Point3D>& positions, const std::vector<int>& colors)
{
    if (!detected_cones_pub_ || positions.empty()) return;

    dv_msgs::msg::IndexedTrack track_msg;

    int yellow_count = 0;
    int blue_count = 0;

    for (size_t i = 0; i < positions.size(); ++i) {
        dv_msgs::msg::IndexedCone cone_msg;
        double x = positions[i][0];
        double y = positions[i][1];
        double z = positions[i][2];

        if (x < 3.35) continue; // skip this marker
        if (x > 10) continue; // Ignore very far cones

        
        // Convert to polar coordinates (range and angle) as in current code
        double range = sqrt(x * x + y * y);
        double angle = atan2(y, x);
        
        cone_msg.location.x = range;
        cone_msg.location.y = angle;
        // --- CHANGE HERE ---
        // Publish the final Cartesian coordinates directly, just like the stable code.
        // cone_msg.location.x = x;
        // cone_msg.location.y = y;
        cone_msg.location.z = z;
        cone_msg.color = colors[i];
        cone_msg.index = i;
        track_msg.track.push_back(cone_msg);

        // Count colors
        if (colors[i] == dv_msgs::msg::IndexedCone::YELLOW) yellow_count++;
        else if (colors[i] == dv_msgs::msg::IndexedCone::BLUE) blue_count++;

    }
    detected_cones_pub_->publish(track_msg);

    // Print counts to terminal
    RCLCPP_INFO(get_logger(), "Detected cones - Yellow: %d, Blue: %d", yellow_count, blue_count);
}

// Callbacks
void ProcessLidar::lidarRawCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg)
{
    try {
        auto points = extractPointsFromPointCloud(msg);
        processPointCloudData(points, msg->header);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "PointCloud processing error: %s", e.what());
    }
}

void ProcessLidar::lidarRawCallback2(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    try {
        auto points = extractPointsFromPointCloud2(msg);
        processPointCloudData(points, msg->header);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "PointCloud2 processing error: %s", e.what());
    }
}