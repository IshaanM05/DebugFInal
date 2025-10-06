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
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>

using namespace perception_winter;

// --- NEW: ACCURACY METRICS ---
// Enum to represent the calculated ground truth color
enum class GroundTruthColor { BLUE, YELLOW };

// --- CONE CLASSIFIER IMPLEMENTATION ---

ConeClassifier::ConeClassifier(Ort::Env& env) : env_(env) {}

bool ConeClassifier::initialize(const std::string& model_path) {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
        input_node_names_.push_back(std::string(input_name_ptr.get()));

        auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
        output_node_names_.push_back(std::string(output_name_ptr.get()));

        return true;
    }
    catch (const Ort::Exception &e) {
        return false;
    }
}

// Change return type to std::optional<int> to match first code
std::optional<int> ConeClassifier::classify(const Cluster& cluster) {
    if (cluster.empty()) return std::nullopt;

    auto feature_vector = extractFeatures(cluster);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> concrete_input_shape = {1, lidar_constants::FEATURE_SIZE, 1};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, feature_vector.data(), feature_vector.size(),
        concrete_input_shape.data(), concrete_input_shape.size());

    std::vector<const char *> input_names_char;
    input_names_char.reserve(input_node_names_.size());
    for (const auto &s : input_node_names_) {
        input_names_char.push_back(s.c_str());
    }

    std::vector<const char *> output_names_char;
    output_names_char.reserve(output_node_names_.size());
    for (const auto &s : output_node_names_) {
        output_names_char.push_back(s.c_str());
    }

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                        input_names_char.data(), &input_tensor, 1, 
                                        output_names_char.data(), 1);

    float prediction_probability = *output_tensors[0].GetTensorMutableData<float>();

    // EXACT REPLICA OF FIRST CODE:
    if (prediction_probability > confidence_threshold_) {
        return 1; // Confidently class 1 (blue)
    }
    else if (prediction_probability < (1.0 - confidence_threshold_)) {
        return 0; // Confidently class 0 (yellow)
    }
    else {
        return std::nullopt; // Low confidence, unclassified
    }
}

std::vector<float> ConeClassifier::extractFeatures(const Cluster& cluster) const {
    float min_intensity = std::numeric_limits<float>::max();
    float max_intensity = std::numeric_limits<float>::lowest();
    
    for (const auto& point : cluster) {
        min_intensity = std::min(min_intensity, static_cast<float>(point[3]));
        max_intensity = std::max(max_intensity, static_cast<float>(point[3]));
    }
    float intensity_range = max_intensity - min_intensity;

    std::vector<float> sum_intensity(lidar_constants::NUM_BINS, 0.0f);
    std::vector<int> point_count(lidar_constants::NUM_BINS, 0);
    
    for (const auto& point : cluster) {
        float z = static_cast<float>(point[2]);
        if (z >= lidar_constants::Z_MIN && z < lidar_constants::Z_MAX) {
            float norm_intensity = (intensity_range > 1e-6) ? 
                (static_cast<float>(point[3]) - min_intensity) / intensity_range : 0.0f;
            int bin_index = static_cast<int>((z - lidar_constants::Z_MIN) / lidar_constants::BIN_WIDTH);
            if (bin_index >= 0 && bin_index < lidar_constants::NUM_BINS) {
                sum_intensity[bin_index] += norm_intensity;
                point_count[bin_index]++;
            }
        }
    }
    
    std::vector<float> feature_vector(lidar_constants::FEATURE_SIZE);
    for (int i = 0; i < lidar_constants::NUM_BINS; ++i) {
        feature_vector[i] = static_cast<float>(point_count[i]);
        feature_vector[i + lidar_constants::NUM_BINS] = (point_count[i] > 0) ? 
            sum_intensity[i] / point_count[i] : 0.0f;
    }

    return feature_vector;
}

// --- POINT CLOUD PROCESSOR IMPLEMENTATION ---

bool PointCloudProcessor::filterCarBodyAndROI(const std::vector<Point4D>& input_points, 
                                              PointCloudPtr output_cloud) {
    output_cloud->reserve(input_points.size());
    
    for (const auto& point : input_points) {
        bool is_valid_point = (point[0] > lidar_constants::CAR_FRONT_X) || 
                             (std::abs(point[1]) > lidar_constants::CAR_SIDE_Y);

        if (point[0] > 0 && is_valid_point) {
            pcl::PointXYZI pcl_point;
            pcl_point.x = point[0];
            pcl_point.y = point[1];
            pcl_point.z = point[2];
            pcl_point.intensity = point[3];
            output_cloud->push_back(pcl_point);
        }
    }

    auto cloud_filtered_pass = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PassThrough<pcl::PointXYZI> pass;

    pass.setInputCloud(output_cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(lidar_constants::ROI_Y_MIN, lidar_constants::ROI_Y_MAX);
    pass.filter(*cloud_filtered_pass);

    pass.setInputCloud(cloud_filtered_pass);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(lidar_constants::ROI_Z_MIN, lidar_constants::ROI_Z_MAX);
    pass.filter(*output_cloud);
    
    return !output_cloud->empty();
}

bool PointCloudProcessor::removeGroundPlane(PointCloudPtr cloud, PointCloudPtr non_ground_cloud) {
    if (cloud->empty()) return false;

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::ExtractIndices<pcl::PointXYZI> extract;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(lidar_constants::RANSAC_THRESHOLD);

    auto remaining_cloud = cloud;
    auto ground_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    
    std::optional<Eigen::Vector3f> reference_normal;
    int iterations = 0;

    while (remaining_cloud->size() > lidar_constants::MIN_POINTS_FOR_PLANE && 
           iterations < lidar_constants::MAX_GROUND_ITERATIONS) {
        int dynamic_max_iter = std::min(static_cast<int>(remaining_cloud->size() / 200), 
                                       lidar_constants::MAX_GROUND_ITERATIONS);
        if (iterations >= dynamic_max_iter) break;

        seg.setInputCloud(remaining_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) break;

        Eigen::Vector3f current_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        if (current_normal.z() < 0) current_normal = -current_normal;

        if (!isValidGroundPlane(current_normal, reference_normal)) break;

        if (!reference_normal.has_value()) {
            reference_normal = current_normal;
        }

        auto current_ground_plane = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*current_ground_plane);
        *ground_cloud += *current_ground_plane;

        auto next_remaining = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        extract.setNegative(true);
        extract.filter(*next_remaining);
        remaining_cloud = next_remaining;
        iterations++;
    }

    *non_ground_cloud = *remaining_cloud;
    return !non_ground_cloud->empty();
}

bool PointCloudProcessor::isValidGroundPlane(const Eigen::Vector3f& normal, 
                                            const std::optional<Eigen::Vector3f>& reference_normal) const {
    if (normal.z() < lidar_constants::MIN_Z_NORMAL_COMPONENT) return false;

    if (reference_normal.has_value()) {
        double dot_product = normal.dot(reference_normal.value());
        double angle_rad = std::acos(std::clamp(dot_product, -1.0, 1.0));
        double angle_deg = angle_rad * (180.0 / M_PI);
        if (angle_deg > lidar_constants::MAX_SLOPE_DEVIATION_DEG) return false;
    }

    return true;
}

// --- ACCURACY METRICS IMPLEMENTATION ---

void AccuracyMetrics::updateMetrics(bool true_is_yellow, bool predicted_is_yellow) {
    if (true_is_yellow) {
        if (predicted_is_yellow) {
            metrics_.true_positives_yellow++;
        } else {
            metrics_.false_positives_blue++;
        }
    } else {
        if (predicted_is_yellow) {
            metrics_.false_positives_yellow++;
        } else {
            metrics_.true_positives_blue++;
        }
    }
}

void AccuracyMetrics::printReport(rclcpp::Logger logger) const {
    long total_yellow_predictions = metrics_.true_positives_yellow + metrics_.false_positives_yellow;
    long total_blue_predictions = metrics_.true_positives_blue + metrics_.false_positives_blue;
    long total_true_yellow = metrics_.true_positives_yellow + metrics_.false_positives_blue;
    long total_true_blue = metrics_.true_positives_blue + metrics_.false_positives_yellow;
    long total_correct = metrics_.true_positives_yellow + metrics_.true_positives_blue;
    long total_all = total_yellow_predictions + total_blue_predictions;

    double yellow_precision = (total_yellow_predictions > 0) ? 
        (double)metrics_.true_positives_yellow / total_yellow_predictions : 0.0;
    double yellow_recall = (total_true_yellow > 0) ? 
        (double)metrics_.true_positives_yellow / total_true_yellow : 0.0;
    double blue_precision = (total_blue_predictions > 0) ? 
        (double)metrics_.true_positives_blue / total_blue_predictions : 0.0;
    double blue_recall = (total_true_blue > 0) ? 
        (double)metrics_.true_positives_blue / total_true_blue : 0.0;
    double overall_accuracy = (total_all > 0) ? 
        (double)total_correct / total_all : 0.0;

    RCLCPP_INFO(logger, "--- LiDAR Perception Accuracy Report ---");
    RCLCPP_INFO(logger, "Overall Accuracy: %.2f%% (%ld / %ld)", overall_accuracy * 100.0, total_correct, total_all);
    RCLCPP_INFO(logger, "----------------------------------------");
    RCLCPP_INFO(logger, "Yellow Cone Metrics:");
    RCLCPP_INFO(logger, "  - Precision: %.2f%%", yellow_precision * 100.0);
    RCLCPP_INFO(logger, "  - Recall:    %.2f%%", yellow_recall * 100.0);
    RCLCPP_INFO(logger, "  - Counts (TP/FP): %ld / %ld", metrics_.true_positives_yellow, metrics_.false_positives_yellow);
    RCLCPP_INFO(logger, "----------------------------------------");
    RCLCPP_INFO(logger, "Blue Cone Metrics:");
    RCLCPP_INFO(logger, "  - Precision: %.2f%%", blue_precision * 100.0);
    RCLCPP_INFO(logger, "  - Recall:    %.2f%%", blue_recall * 100.0);
    RCLCPP_INFO(logger, "  - Counts (TP/FP): %ld / %ld", metrics_.true_positives_blue, metrics_.false_positives_blue);
    RCLCPP_INFO(logger, "----------------------------------------");
}

// --- CLUSTER PROCESSOR IMPLEMENTATION ---

std::vector<Cluster> ClusterProcessor::clusterPoints(const PointCloudPtr cloud) {
    if (cloud->empty()) return {};

    auto o3d_pcd = std::make_shared<open3d::geometry::PointCloud>();
    o3d_pcd->points_.reserve(cloud->size());
    
    for (const auto& point : cloud->points) {
        o3d_pcd->points_.emplace_back(point.x, point.y, point.z);
    }

    auto labels = o3d_pcd->ClusterDBSCAN(lidar_constants::DBSCAN_EPSILON, 
                                        lidar_constants::DBSCAN_MINPOINTS, false);
    
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

    clusters.erase(std::remove_if(clusters.begin(), clusters.end(),
        [](const Cluster& c) { return c.empty(); }), clusters.end());

    return clusters;
}

std::vector<Cluster> ClusterProcessor::filterClustersBySize(const std::vector<Cluster>& clusters) {
    std::vector<Cluster> valid_clusters;
    valid_clusters.reserve(clusters.size());

    for (const auto& cluster : clusters) {
        if (cluster.size() < 4) continue;

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

        if (height >= 0.15 && height <= 0.4 && width <= 0.45) {
            valid_clusters.push_back(cluster);
        }
    }

    return valid_clusters;
}

// Original method with accuracy metrics
// void ClusterProcessor::detectConesInClusters(const std::vector<Cluster>& clusters,
//                                             std::vector<Point3D>& positions,
//                                             std::vector<int>& colors,
//                                             ConeClassifier& classifier,
//                                             AccuracyMetrics& accuracy_metrics) {
//     positions.reserve(clusters.size());
//     colors.reserve(clusters.size());
//
//     for (const auto& cluster : clusters) {
//         auto sorted_cluster = cluster;
//         std::sort(sorted_cluster.begin(), sorted_cluster.end(),
//             [](const Point4D& a, const Point4D& b) { return a[2] > b[2]; });
//
//         auto cone_pos = calculateConePosition(sorted_cluster);
//         
//         // Get classification with optional handling (matches first code)
//         std::optional<int> classification = classifier.classify(cluster);
//         
//         // Skip low confidence clusters (std::nullopt) - same as first code
//         if (!classification.has_value()) {
//             continue; // Don't publish this cone
//         }
//
//         // Convert to proper color codes (0=Yellow, 1=Blue)
//         int predicted_color = (classification.value() == 1) ? 
//                               dv_msgs::msg::IndexedCone::BLUE : 
//                               dv_msgs::msg::IndexedCone::YELLOW;
//         
//         positions.push_back(cone_pos);
//         colors.push_back(predicted_color);
//
//         // Accuracy metrics calculation
//         double total_intensity = 0.0;
//         for (const auto& point : cluster) {
//             total_intensity += point[3];
//         }
//         double avg_intensity = cluster.empty() ? 0.0 : total_intensity / cluster.size();
//         
//         bool true_is_yellow = (avg_intensity <= 1e6); // Adjust threshold as needed for your sensor
//         bool predicted_is_yellow = (predicted_color == dv_msgs::msg::IndexedCone::YELLOW);
//         
//         accuracy_metrics.updateMetrics(true_is_yellow, predicted_is_yellow);
//     }
// }

// Without accuracy metrics
void ClusterProcessor::detectConesInClusters(const std::vector<Cluster>& clusters,
                                            std::vector<Point3D>& positions,
                                            std::vector<int>& colors,
                                            ConeClassifier& classifier) {
    positions.reserve(clusters.size());
    colors.reserve(clusters.size());

    for (const auto& cluster : clusters) {
        auto sorted_cluster = cluster;
        std::sort(sorted_cluster.begin(), sorted_cluster.end(),
            [](const Point4D& a, const Point4D& b) { return a[2] > b[2]; });

        auto cone_pos = calculateConePosition(sorted_cluster);
        
        // Get classification with optional handling
        std::optional<int> classification = classifier.classify(cluster);
        
        // Skip low confidence clusters (std::nullopt)
        if (!classification.has_value()) {
            continue; // Don't publish this cone
        }

        // Convert to proper color codes
        int predicted_color = (classification.value() == 1) ? 
                              dv_msgs::msg::IndexedCone::BLUE : 
                              dv_msgs::msg::IndexedCone::YELLOW;
        
        positions.push_back(cone_pos);
        colors.push_back(predicted_color);
    }
}

Point3D ClusterProcessor::calculateConePosition(const Cluster& cluster) {
    double min_x = cluster[0][0], max_x = cluster[0][0];
    double min_y = cluster[0][1], max_y = cluster[0][1];
    for (const auto& point : cluster) {
        min_x = std::min(min_x, point[0]);
        max_x = std::max(max_x, point[0]);
        min_y = std::min(min_y, point[1]);
        max_y = std::max(max_y, point[1]);
    }

    double median_x = getMedian(cluster, 0);
    double median_y = getMedian(cluster, 1);

    constexpr double w_median = 0.7;
    constexpr double w_min_x = 0.3;
    constexpr double w_min_y = 0.3;

    double cone_x = w_median * median_x + w_min_x * (min_x + lidar_constants::CONE_BASE_RADIUS);
    double cone_y = w_median * median_y + w_min_y * (min_y + lidar_constants::CONE_BASE_RADIUS);

    return {cone_x, cone_y, lidar_constants::CONE_HEIGHT};
}

double ClusterProcessor::getMedian(const Cluster& points, size_t idx) const {
    if (points.empty()) return 0.0;

    std::vector<size_t> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0);

    size_t mid = indices.size() / 2;
    std::nth_element(indices.begin(), indices.begin() + mid, indices.end(),
        [&points, idx](size_t a, size_t b) { return points[a][idx] < points[b][idx]; });

    double median = points[indices[mid]][idx];

    if (indices.size() % 2 == 0 && mid > 0) {
        auto max_it = std::max_element(indices.begin(), indices.begin() + mid,
            [&points, idx](size_t a, size_t b) { return points[a][idx] < points[b][idx]; });
        median = 0.5 * (median + points[*max_it][idx]);
    }

    return median;
}

// --- POINT CLOUD EXTRACTOR IMPLEMENTATION ---

std::vector<Point4D> PointCloudExtractor::fromPointCloud(const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg) {
    std::vector<Point4D> points;
    points.reserve(cloud_msg->points.size());

    bool has_intensity = !cloud_msg->channels.empty() && 
                        cloud_msg->channels[0].values.size() == cloud_msg->points.size();

    for (size_t i = 0; i < cloud_msg->points.size(); ++i) {
        const auto& pt = cloud_msg->points[i];
        points.push_back({pt.x, pt.y, pt.z, has_intensity ? cloud_msg->channels[0].values[i] : 0.0});
    }

    return points;
}

std::vector<Point4D> PointCloudExtractor::fromPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    std::vector<Point4D> points;
    points.reserve(cloud_msg->width * cloud_msg->height);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");
    
    bool has_intensity = false;
    for (const auto& field : cloud_msg->fields) {
        if (field.name == "intensity") {
            has_intensity = true;
            break;
        }
    }
    
    std::optional<sensor_msgs::PointCloud2ConstIterator<float>> iter_intensity_opt;
    if (has_intensity) {
        iter_intensity_opt.emplace(*cloud_msg, "intensity");
    }

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

// --- MAIN PROCESS LIDAR IMPLEMENTATION ---

ProcessLidar::ProcessLidar() : 
    Node("process_lidar"), 
    env_(ORT_LOGGING_LEVEL_WARNING, "ONNX_INFERENCE") {
    
    initializeComponents();
    
    // Subscribers
    lidar_raw_input_sub_ = create_subscription<sensor_msgs::msg::PointCloud>(
        lidar_constants::LIDAR_RAW_TOPIC, 10,
        [this](const sensor_msgs::msg::PointCloud::SharedPtr msg) {
            lidarRawCallback(msg);
        });

    lidar_raw_input_sub2_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_constants::LIDAR_RAW_TOPIC2, 10,
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            lidarRawCallback2(msg);
        });

    // Publishers
    detected_cones_pub_ = create_publisher<dv_msgs::msg::IndexedTrack>("/perception/cones", 10);
    filtered_points_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/perception/filtered_points", 10);
    lidar_clusters_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/perception/clusters", 10);

    RCLCPP_INFO(get_logger(), "Optimized LiDAR Node started");
}

ProcessLidar::~ProcessLidar() {
    // Commented to remove accuracy metrics printing on shutdown
    // if (accuracy_metrics_) {
    //     accuracy_metrics_->printReport(get_logger());
    // }
    RCLCPP_INFO(get_logger(), "LiDAR Node shutdown");
}

void ProcessLidar::initializeComponents() {
    cone_classifier_ = std::make_unique<ConeClassifier>(env_);
    point_cloud_processor_ = std::make_unique<PointCloudProcessor>();
    cluster_processor_ = std::make_unique<ClusterProcessor>();
    //accuracy_metrics_ = std::make_unique<AccuracyMetrics>();
    
    loadONNXModel();
}

void ProcessLidar::loadONNXModel() {
    std::string package_share_directory;
    try {
        package_share_directory = ament_index_cpp::get_package_share_directory("perception_winter");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to get package share directory: %s", e.what());
        package_share_directory = ".";
    }
    
    std::string model_path = package_share_directory + "/cone_model.onnx";
    
    if (cone_classifier_->initialize(model_path)) {
        RCLCPP_INFO(get_logger(), "Successfully loaded ONNX model from: %s", model_path.c_str());
    } else {
        RCLCPP_FATAL(get_logger(), "Failed to load ONNX model: %s", model_path.c_str());
        rclcpp::shutdown();
    }
}

void ProcessLidar::processPointCloudData(std::vector<Point4D>& points, const std_msgs::msg::Header& header) {
    (void)header;
    
    if (points.empty()) return;

    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    auto pipeline_start = std::chrono::steady_clock::now();
    
    // Stage 1: Filtering
    if (!point_cloud_processor_->filterCarBodyAndROI(points, cloud)) {
        RCLCPP_DEBUG(get_logger(), "No points after car body and ROI filtering");
        return;
    }

    // Stage 2: Ground removal
    if (!point_cloud_processor_->removeGroundPlane(cloud, cloud_filtered)) {
        RCLCPP_DEBUG(get_logger(), "No points after ground removal");
        return;
    }
    //  Optional: Publish filtered points for debugging
    publishFilteredPoints(cloud_filtered);

    // Stage 3: Clustering
    auto clusters = cluster_processor_->clusterPoints(cloud_filtered);
    if (clusters.empty()) {
        RCLCPP_DEBUG(get_logger(), "No clusters found");
        return;
    }

    // Stage 4: Cluster filtering
    auto filtered_clusters = cluster_processor_->filterClustersBySize(clusters);
    if (filtered_clusters.empty()) {
        RCLCPP_DEBUG(get_logger(), "No valid clusters after size filtering");
        return;
    }

    // Stage 5: Cone detection
    std::vector<Point3D> cone_positions;
    std::vector<int> cone_colors;
    
    //With accuracy metrics
    // cluster_processor_->detectConesInClusters(filtered_clusters, cone_positions, cone_colors,
    //                                          *cone_classifier_, *accuracy_metrics_);

    //Without accuracy metrics
    cluster_processor_->detectConesInClusters(filtered_clusters, cone_positions, cone_colors,
                                         *cone_classifier_);

    // PUBLISH LIDAR CLUSTERS FOR DEBUGGING
    // publishLidarClusters(cone_positions);

    // Publish results
    publishDetectedCones(cone_positions, cone_colors);

    auto pipeline_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(pipeline_end - pipeline_start);
    RCLCPP_DEBUG(get_logger(), "Processing completed in %ld ms", duration.count());
}

void ProcessLidar::publishDetectedCones(const std::vector<Point3D>& positions, const std::vector<int>& colors) {
    dv_msgs::msg::IndexedTrack track_msg;

    int yellow_count = 0;
    int blue_count = 0;

    for (size_t i = 0; i < positions.size(); ++i) {
        dv_msgs::msg::IndexedCone cone_msg;
        double x = positions[i][0];
        double y = positions[i][1];
        double z = positions[i][2];

        if (x < 3.35) continue;
        if (x > 12) continue;
        
        double range = sqrt(x * x + y * y);
        double angle = atan2(y, x);
        
        cone_msg.location.x = range;
        cone_msg.location.y = angle;
        cone_msg.location.z = z;
        cone_msg.color = colors[i];
        cone_msg.index = i;
        track_msg.track.push_back(cone_msg);

        if (colors[i] == dv_msgs::msg::IndexedCone::YELLOW) yellow_count++;
        else if (colors[i] == dv_msgs::msg::IndexedCone::BLUE) blue_count++;
    }
    
    if (!track_msg.track.empty()) {
        detected_cones_pub_->publish(track_msg);
    }

    RCLCPP_INFO(get_logger(), "Detected cones - Yellow: %d, Blue: %d", yellow_count, blue_count);
}

void ProcessLidar::lidarRawCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg) {
    try {
        auto points = PointCloudExtractor::fromPointCloud(msg);
        processPointCloudData(points, msg->header);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "PointCloud processing error: %s", e.what());
    }
}

void ProcessLidar::lidarRawCallback2(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    try {
        auto points = PointCloudExtractor::fromPointCloud2(msg);
        processPointCloudData(points, msg->header);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "PointCloud processing error: %s", e.what());
    }
}

// Publishers (commented out - kept for reference)
void ProcessLidar::publishFilteredPoints(const PointCloudPtr cloud)
{
    if (!filtered_points_pub_ || cloud->points.empty()) return;

    auto message = std_msgs::msg::Float32MultiArray();
    for (const auto& point : cloud->points) {
        message.data.push_back(static_cast<float>(point.x));
        message.data.push_back(static_cast<float>(point.y));
        message.data.push_back(static_cast<float>(point.z));
    }
    filtered_points_pub_->publish(message);
}
//
// void ProcessLidar::publishLidarClusters(const std::vector<Point3D>& cluster_centers)
// {
//     if (!lidar_clusters_pub_ || cluster_centers.empty()) return;
//
//     auto message = std_msgs::msg::Float32MultiArray();
//     for (const auto& center : cluster_centers) {
//         message.data.push_back(static_cast<float>(center[0]));
//         message.data.push_back(static_cast<float>(center[1]));
//     }
//     lidar_clusters_pub_->publish(message);
// }