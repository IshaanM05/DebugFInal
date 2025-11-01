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
#include <pcl/features/normal_3d.h>
#include <chrono>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <chrono>
#include <atomic>

using namespace perception_winter;

static std::atomic<long long> g_true_positives_blue{0};
static std::atomic<long long> g_true_positives_yellow{0};
static std::atomic<long long> g_false_positives_blue{0};     // Predicted Blue, but was Yellow
static std::atomic<long long> g_false_positives_yellow{0};   // Predicted Yellow, but was Blue
static std::atomic<long long> g_rejected_as_blue{0};       // Was Blue, but rejected
static std::atomic<long long> g_rejected_as_yellow{0};     // Was Yellow, but rejected

// =============================================
// ML CLASSIFIER IMPLEMENTATION
// =============================================

ConeClassifier::ConeClassifier(Ort::Env& env) : env_(env) {}

bool ConeClassifier::initialize(const std::string& model_path) {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        // Get input names
        auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
        input_node_names_.push_back(input_name_ptr.get());

        // Get output names  
        auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
        output_node_names_.push_back(output_name_ptr.get());

        // Get input dimensions
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_node_dims_ = input_tensor_info.GetShape();

        RCLCPP_INFO(rclcpp::get_logger("cone_classifier"), 
                   "ML Classifier initialized successfully with confidence threshold: %.2f", 
                   confidence_threshold_);
        return true;
    } 
    catch (const Ort::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_classifier"), 
                    "ONNX initialization failed: %s", e.what());
        return false;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_classifier"), 
                    "ML Classifier initialization failed: %s", e.what());
        return false;
    }
}

std::vector<float> ConeClassifier::createFeatureVector(const Cluster& cluster) const {
    std::vector<float> feature_vector(lidar_constants::NUM_BINS * 2, 0.0f);
    if (cluster.empty()) return feature_vector;

    // Calculate normalization parameters
    double min_z = cluster[0][2], max_z = cluster[0][2];
    double min_intensity = cluster[0][3], max_intensity = cluster[0][3];
    
    for (const auto& point : cluster) {
        min_z = std::min(min_z, point[2]); 
        max_z = std::max(max_z, point[2]);
        min_intensity = std::min(min_intensity, point[3]); 
        max_intensity = std::max(max_intensity, point[3]);
    }

    double z_range = max_z - min_z;
    double intensity_range = max_intensity - min_intensity;
    double bin_width = (z_range > 1e-6) ? z_range / lidar_constants::NUM_BINS : 0.0;

    // Bin points and calculate statistics
    std::vector<double> sum_intensity(lidar_constants::NUM_BINS, 0.0);
    std::vector<int> point_count(lidar_constants::NUM_BINS, 0);

    for (const auto& point : cluster) {
        double norm_intensity = (intensity_range > 1e-6) ? 
                               (point[3] - min_intensity) / intensity_range : 0.0;
        int bin_index = (bin_width > 0) ? 
                       static_cast<int>((point[2] - min_z) / bin_width) : 0;
        bin_index = std::min(bin_index, lidar_constants::NUM_BINS - 1);
        
        sum_intensity[bin_index] += norm_intensity;
        point_count[bin_index]++;
    }

    // Normalize counts
    auto min_max_it = std::minmax_element(point_count.begin(), point_count.end());
    float min_c = static_cast<float>(*min_max_it.first);
    float max_c = static_cast<float>(*min_max_it.second);
    float count_range = max_c - min_c;

    for (int i = 0; i < lidar_constants::NUM_BINS; ++i) {
        // Normalized point count
        float normalized_count = 0.0f;
        if (count_range > 0) {
            normalized_count = (static_cast<float>(point_count[i]) - min_c) / count_range;
        } else {
            normalized_count = (min_c > 0) ? 1.0f : 0.0f;
        }
        feature_vector[i * 2 + 0] = normalized_count;
        
        // Average normalized intensity
        feature_vector[i * 2 + 1] = (point_count[i] > 0) ? 
                                   static_cast<float>(sum_intensity[i] / point_count[i]) : 0.0f;
    }

    return feature_vector;
}

std::optional<int> ConeClassifier::classify(const Cluster& cluster) {
    if (cluster.empty()) {
        return std::nullopt;
    }

    try {
        std::vector<float> feature_vector = createFeatureVector(cluster);
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> concrete_shape = input_node_dims_;
        if (!concrete_shape.empty() && concrete_shape[0] == -1) {
            concrete_shape[0] = 1; 
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, feature_vector.data(), feature_vector.size(),
            concrete_shape.data(), concrete_shape.size());

        std::vector<const char*> input_names_char;
        input_names_char.reserve(input_node_names_.size());
        for (const auto& s : input_node_names_) {
            input_names_char.push_back(s.c_str());
        }

        std::vector<const char*> output_names_char;
        output_names_char.reserve(output_node_names_.size());
        for (const auto& s : output_node_names_) {
            output_names_char.push_back(s.c_str());
        }

        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                           input_names_char.data(), &input_tensor, 1,
                                           output_names_char.data(), 1);
        
        float prediction_prob = *output_tensors[0].GetTensorMutableData<float>();

        // Apply confidence thresholding
        if (prediction_prob > confidence_threshold_) {
            return dv_msgs::msg::IndexedCone::BLUE;
        } else if ((1.0 - prediction_prob) > confidence_threshold_) {
            return dv_msgs::msg::IndexedCone::YELLOW;
        }
        
        return std::nullopt;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_classifier"), 
                    "Classification failed: %s", e.what());
        return std::nullopt;
    }
}

// =============================================
// HEURISTIC CLASSIFIER IMPLEMENTATION
// =============================================

bool HeuristicClassifier::classifyCone(const std::vector<double> &y_vals, const std::vector<double> &x_vals) {
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
    return coeffs(0) > 0; // Positive quadratic coefficient indicates upward curve (yellow)
}

std::vector<double> HeuristicClassifier::movingAverage(const std::vector<double> &data, int kernel) {
    int n = data.size();
    std::vector<double> result(n, 0.0);
    if (kernel < 1 || n == 0) return data;

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

std::optional<int> HeuristicClassifier::classify(const Cluster& cluster) {
    if (cluster.size() < 3) return std::nullopt;

    try {
        // Sort by height for consistent analysis
        auto sorted_cluster = cluster;
        std::sort(sorted_cluster.begin(), sorted_cluster.end(),
                  [](const Point4D& a, const Point4D& b) { return a[2] > b[2]; });

        // Extract intensity and height profiles
        std::vector<double> intensity_vals;
        std::vector<double> z_vals;
        intensity_vals.reserve(sorted_cluster.size());
        z_vals.reserve(sorted_cluster.size());

        for (const auto& pt : sorted_cluster) {
            intensity_vals.push_back(pt[3]);
            z_vals.push_back(pt[2]);
        }

        // Apply smoothing for noise reduction
        int kernel = std::max(3, static_cast<int>(0.1 * intensity_vals.size()));
        if (kernel % 2 == 0) kernel += 1;
        std::vector<double> averaged_intensities = this->movingAverage(intensity_vals, kernel);
        
        // Classify based on intensity profile curvature
        bool is_yellow_heuristic = this->classifyCone(averaged_intensities, z_vals);
        return is_yellow_heuristic ? dv_msgs::msg::IndexedCone::YELLOW : dv_msgs::msg::IndexedCone::BLUE;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("heuristic_classifier"), 
                    "Heuristic classification failed: %s", e.what());
        return std::nullopt;
    }
}

// =============================================
// POINT CLOUD PROCESSOR IMPLEMENTATION
// =============================================

bool PointCloudProcessor::filterCarBodyAndROI(const std::vector<Point4D>& input_points, 
                                              PointCloudPtr output_cloud,
                                              rclcpp::Logger logger) {
    if (input_points.empty()) {
        RCLCPP_WARN(logger, "Empty input point cloud for filtering");
        return false;
    }

    output_cloud->reserve(input_points.size());
    
    int car_body_points = 0;
    int behind_car_points = 0;
    int outside_roi_points = 0;
    int valid_points = 0;
    
    RCLCPP_DEBUG(logger, "Starting ROI filtering with %zu input points", input_points.size());
    
    for (const auto& point : input_points) {
        double x = point[0];
        double y = point[1];
        double z = point[2];
        
        // Skip points behind the car (redundant safety check)
        if (x <= 0) {
            behind_car_points++;
            continue;
        }
        
        // Skip points that are on the car body (primary exclusion)
        bool on_car_body = (x <= lidar_constants::CAR_FRONT_X) && 
                          (std::abs(y) <= lidar_constants::CAR_SIDE_Y);
        if (on_car_body) {
            car_body_points++;
            continue;
        }
        
        // Apply ROI filtering (secondary validation)
        bool in_roi_y = (y >= lidar_constants::ROI_Y_MIN) && (y <= lidar_constants::ROI_Y_MAX);
        bool in_roi_z = (z >= lidar_constants::ROI_Z_MIN) && (z <= lidar_constants::ROI_Z_MAX);
        
        if (in_roi_y && in_roi_z) {
            pcl::PointXYZI pcl_point;
            pcl_point.x = x;
            pcl_point.y = y;
            pcl_point.z = z;
            pcl_point.intensity = point[3];
            output_cloud->push_back(pcl_point);
            valid_points++;
        } else {
            outside_roi_points++;
        }
    }
    
    RCLCPP_INFO(logger,
               "ROI filtering: %zu -> %zu points (behind: %d, car_body: %d, outside_roi: %d, valid: %d)", 
               input_points.size(), output_cloud->size(), behind_car_points, 
               car_body_points, outside_roi_points, valid_points);
    
    return !output_cloud->empty();
}

bool PointCloudProcessor::removeGroundPlane(PointCloudPtr cloud,
                                           PointCloudPtr non_ground_cloud,
                                           rclcpp::Logger logger) {
    if (cloud->empty()) {
        RCLCPP_WARN(logger, "Empty cloud for ground removal");
        return false;
    }

    RCLCPP_DEBUG(logger, "Starting ground removal with %zu points", cloud->size());

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::ExtractIndices<pcl::PointXYZI> extract;

    // Configure RANSAC parameters
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(lidar_constants::RANSAC_THRESHOLD);

    auto remaining_cloud = cloud;
    auto ground_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    
    std::optional<Eigen::Vector3f> reference_normal;
    int iterations = 0;

    // Iterative ground plane removal
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

        // Extract current ground plane
        auto current_ground_plane = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*current_ground_plane);
        *ground_cloud += *current_ground_plane;

        // Update remaining cloud
        auto next_remaining = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        extract.setNegative(true);
        extract.filter(*next_remaining);
        remaining_cloud = next_remaining;
        iterations++;
    }

    *non_ground_cloud = *remaining_cloud;
    
    RCLCPP_INFO(logger,
               "Ground removal: %zu -> %zu points (%zu ground points removed in %d iterations)",
               cloud->size(), non_ground_cloud->size(), ground_cloud->size(), iterations);
    
    return !non_ground_cloud->empty();
}

bool PointCloudProcessor::isValidGroundPlane(const Eigen::Vector3f& normal, 
                                            const std::optional<Eigen::Vector3f>& reference_normal) const {
    // Check normal orientation
    if (normal.z() < lidar_constants::MIN_Z_NORMAL_COMPONENT) return false;

    // Check consistency with reference normal
    if (reference_normal.has_value()) {
        double dot_product = normal.dot(reference_normal.value());
        double angle_rad = std::acos(std::clamp(dot_product, -1.0, 1.0));
        double angle_deg = angle_rad * (180.0 / M_PI);
        if (angle_deg > lidar_constants::MAX_SLOPE_DEVIATION_DEG) return false;
    }

    return true;
}

// =============================================
// CLUSTER PROCESSOR IMPLEMENTATION
// =============================================

std::vector<Cluster> ClusterProcessor::clusterPoints(const PointCloudPtr cloud, rclcpp::Logger logger) {
    if (cloud->empty()) {
        RCLCPP_WARN(logger, "Empty cloud for clustering");
        return {};
    }

    RCLCPP_DEBUG(logger, "Starting clustering with %zu points", cloud->size());

    try {
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
        int noise_points = 0;
        
        for (size_t i = 0; i < labels.size(); ++i) {
            int label = labels[i];
            if (label >= 0) {
                const auto& point = cloud->points[i];
                clusters[label].push_back({point.x, point.y, point.z, point.intensity});
            } else {
                noise_points++;
            }
        }

        // Remove empty clusters
        clusters.erase(std::remove_if(clusters.begin(), clusters.end(),
            [](const Cluster& c) { return c.empty(); }), clusters.end());

        RCLCPP_INFO(logger,
                   "Clustering: %zu points -> %zu clusters (noise points: %d)",
                   cloud->size(), clusters.size(), noise_points);
        
        return clusters;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "Clustering failed: %s", e.what());
        return {};
    }
}

std::vector<Cluster> ClusterProcessor::filterClustersBySize(const std::vector<Cluster>& clusters, 
                                                           std::vector<bool>& orange_candidates,
                                                           rclcpp::Logger logger) {
    std::vector<Cluster> valid_clusters;
    valid_clusters.reserve(clusters.size());
    orange_candidates.clear();
    orange_candidates.reserve(clusters.size());

    int total_points_before = 0;
    int total_points_after = 0;
    int rejected_by_points = 0;
    int rejected_by_height = 0;
    int rejected_by_width = 0;
    int rejected_by_x_dist = 0; // Counter for the new filter

    for (const auto& cluster : clusters) {
        total_points_before += cluster.size();
        
        // Minimum points check
        if (cluster.size() < lidar_constants::MIN_CLUSTER_POINTS) {
            rejected_by_points++;
            orange_candidates.push_back(false);
            continue;
        }

        // Calculate cluster properties
        double min_x = cluster[0][0], max_x = cluster[0][0];
        double min_y = cluster[0][1], max_y = cluster[0][1];
        double min_z = cluster[0][2], max_z = cluster[0][2];
        double sum_x = 0.0, sum_y = 0.0;

        for (const auto& point : cluster) {
            min_x = std::min(min_x, point[0]); max_x = std::max(max_x, point[0]);
            min_y = std::min(min_y, point[1]); max_y = std::max(max_y, point[1]);
            min_z = std::min(min_z, point[2]); max_z = std::max(max_z, point[2]);
            sum_x += point[0];
            sum_y += point[1];
        }

        double height = max_z - min_z;
        double width = std::max(max_x - min_x, max_y - min_y);
        double centroid_x = sum_x / cluster.size();
        double centroid_y = sum_y / cluster.size();

        // Check for orange cone candidate during filtering
        bool is_orange_candidate = isOrangeConeCandidate(cluster);
        orange_candidates.push_back(is_orange_candidate);

        // Regular cone filtering
        bool valid_height = (height >= lidar_constants::MIN_CLUSTER_HEIGHT && height <= lidar_constants::MAX_CLUSTER_HEIGHT);
        bool valid_width = (width <= lidar_constants::MAX_CLUSTER_WIDTH);
        bool valid_x_pos = (centroid_x <= lidar_constants::ROI_X_MAX);

        if (!valid_height) rejected_by_height++;
        if (!valid_width) rejected_by_width++;
        if (!valid_x_pos) rejected_by_x_dist++;

        if (valid_height && valid_width && valid_x_pos) {
            valid_clusters.push_back(cluster);
            total_points_after += cluster.size();
        }
    }

    RCLCPP_INFO(rclcpp::get_logger("cluster_processor"),
                "Cluster filtering: %zu -> %zu clusters, points: %d -> %d "
                "(rejected: points=%d, height=%d, width=%d, x_dist=%d)",
                clusters.size(), valid_clusters.size(), total_points_before, total_points_after,
                rejected_by_points, rejected_by_height, rejected_by_width, rejected_by_x_dist);

    return valid_clusters;
}

void ClusterProcessor::detectConesInClusters(const std::vector<Cluster>& clusters,
                                            const std::vector<bool>& orange_candidates,
                                            std::vector<Point3D>& positions, 
                                            std::vector<int>& colors,
                                            ConeClassifier& ml_classifier,
                                            HeuristicClassifier& heuristic_classifier,
                                            rclcpp::Logger logger) {
    positions.reserve(clusters.size());
    colors.reserve(clusters.size());

    int accepted_cones = 0;
    int orange_cones = 0;

    for (size_t i = 0; i < clusters.size(); ++i) {
        const auto& cluster = clusters[i];
        if (cluster.empty()) continue;

        // --- 1. Handle Orange Cones (Assumed correct, not part of Blue/Yellow metrics) ---
        if (orange_candidates[i]) {
            auto cone_pos = calculateConePosition(cluster);
            positions.push_back(cone_pos);
            colors.push_back(dv_msgs::msg::IndexedCone::ORANGE_BIG);
            accepted_cones++;
            orange_cones++;
            continue;
        }

        // --- 2. Determine Ground Truth from Simulator Intensity ---
        double total_intensity = 0.0;
        for (const auto& point : cluster) {
            total_intensity += point[3]; // Intensity is the 4th element
        }
        double avg_intensity = total_intensity / cluster.size();
        
        // Assumption: Simulator uses very high intensity for blue cones.
        bool is_ground_truth_blue = (avg_intensity > 1e6);

        // --- 3. Get Predictions from Both Classifiers ---
        auto heuristic_color_opt = heuristic_classifier.classify(cluster);
        auto ml_color_opt = ml_classifier.classify(cluster);

        // --- 4. Compare and Update Metrics ---
        int final_color = -1; // -1 indicates no decision

        if (ml_color_opt.has_value() && heuristic_color_opt.has_value()) {
            // Decision Rule: Only accept if both classifiers agree.
            if (ml_color_opt.value() == heuristic_color_opt.value()) {
                final_color = ml_color_opt.value();
                auto cone_pos = calculateConePosition(cluster);
                positions.push_back(cone_pos);
                colors.push_back(final_color);
                accepted_cones++;
                
                // Update True/False Positive counters
                if (final_color == dv_msgs::msg::IndexedCone::BLUE) {
                    if (is_ground_truth_blue) g_true_positives_blue++;
                    else g_false_positives_blue++; // Predicted Blue, was Yellow
                } else { // Predicted Yellow
                    if (!is_ground_truth_blue) g_true_positives_yellow++;
                    else g_false_positives_yellow++; // Predicted Yellow, was Blue
                }
            }
        }
        
        // --- 5. Handle Rejections ---
        if (final_color == -1) {
            // If no decision was made (disagreement, low confidence, etc.)
            if (is_ground_truth_blue) g_rejected_as_blue++;
            else g_rejected_as_yellow++;
        }
    }

    // Update the log to show the count of accepted cones only
    RCLCPP_INFO(rclcpp::get_logger("cluster_processor"), 
                "Cone detection: Accepted cones (ML+Heuristic agreement): %d, Orange cones: %d",
                accepted_cones, orange_cones);
}

void ClusterProcessor::printClusterStats(const std::vector<Cluster>& clusters, rclcpp::Logger logger) const {
    RCLCPP_INFO(logger, "Cluster Statistics:");
    RCLCPP_INFO(logger, "  - Total clusters: %zu", clusters.size());
    
    for (size_t i = 0; i < clusters.size(); ++i) {
        const auto& cluster = clusters[i];
        double avg_intensity = 0.0;
        double min_z = std::numeric_limits<double>::max();
        double max_z = std::numeric_limits<double>::lowest();
        double min_x = cluster[0][0], max_x = cluster[0][0];
        double min_y = cluster[0][1], max_y = cluster[0][1];
        
        for (const auto& point : cluster) {
            avg_intensity += point[3];
            min_z = std::min(min_z, point[2]);
            max_z = std::max(max_z, point[2]);
            min_x = std::min(min_x, point[0]);
            max_x = std::max(max_x, point[0]);
            min_y = std::min(min_y, point[1]);
            max_y = std::max(max_y, point[1]);
        }
        avg_intensity /= cluster.size();
        
        RCLCPP_INFO(logger, "  Cluster %zu: %zu points, height: %.3fm, intensity: %.3f", 
                   i, cluster.size(), max_z - min_z, avg_intensity);
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

    // Weighted combination for robust position estimation
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

bool ClusterProcessor::isOrangeConeCandidate(const Cluster& cluster) const {
    if (cluster.size() < lidar_constants::ORANGE_CONE_MIN_POINTS) {
        return false;
    }

    // Calculate centroid distance
    double sum_x = 0.0, sum_y = 0.0;
    for (const auto& point : cluster) {
        sum_x += point[0];
        sum_y += point[1];
    }
    double centroid_x = sum_x / cluster.size();
    double centroid_y = sum_y / cluster.size();
    double distance = std::sqrt(centroid_x * centroid_x + centroid_y * centroid_y);

    return (distance > lidar_constants::ORANGE_CONE_DISTANCE_THRESHOLD);
}

// =============================================
// POINT CLOUD EXTRACTOR IMPLEMENTATION
// =============================================

std::vector<Point4D> PointCloudExtractor::fromPointCloud(const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg) {
    if (!validatePointCloud(cloud_msg)) {
        return {};
    }

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
    if (!validatePointCloud2(cloud_msg)) {
        return {};
    }

    std::vector<Point4D> points;
    points.reserve(cloud_msg->width * cloud_msg->height);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");
    
    bool has_intensity = false;
    std::optional<sensor_msgs::PointCloud2ConstIterator<float>> iter_intensity;
    
    // Check for intensity field with redundancy
    for (const auto& field : cloud_msg->fields) {
        if (field.name == "intensity" || field.name == "intensities") {
            has_intensity = true;
            iter_intensity = sensor_msgs::PointCloud2ConstIterator<float>(*cloud_msg, field.name);
            break;
        }
    }

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        double intensity = 0.0;
        if (has_intensity && iter_intensity.has_value()) {
            intensity = *(*iter_intensity);
            ++(*iter_intensity);
        }
        points.push_back({*iter_x, *iter_y, *iter_z, intensity});
    }

    RCLCPP_INFO(rclcpp::get_logger("point_cloud_extractor"), 
               "Extracted %zu points from PointCloud2 (has_intensity: %d)", 
               points.size(), has_intensity);

    return points;
}

bool PointCloudExtractor::validatePointCloud(const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg) {
    if (!cloud_msg) {
        RCLCPP_ERROR(rclcpp::get_logger("point_cloud_extractor"), "Null PointCloud message");
        return false;
    }
    if (cloud_msg->points.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("point_cloud_extractor"), "Empty PointCloud message");
        return false;
    }
    return true;
}

bool PointCloudExtractor::validatePointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    if (!cloud_msg) {
        RCLCPP_ERROR(rclcpp::get_logger("point_cloud_extractor"), "Null PointCloud2 message");
        return false;
    }
    if (cloud_msg->width * cloud_msg->height == 0) {
        RCLCPP_WARN(rclcpp::get_logger("point_cloud_extractor"), "Empty PointCloud2 message");
        return false;
    }
    return true;
}

// =============================================
// MAIN PROCESS LIDAR IMPLEMENTATION
// =============================================

ProcessLidar::ProcessLidar() : 
    Node("process_lidar"), 
    env_(ORT_LOGGING_LEVEL_WARNING, "ONNX_INFERENCE") {
    
    // Enable comprehensive logging
    auto debug_logger = this->get_logger();
    auto result = rcutils_logging_set_logger_level(debug_logger.get_name(), RCUTILS_LOG_SEVERITY_DEBUG);
    (void)result;

    // Initialize modular components
    initializeComponents();
    
    // Dual input subscribers for redundancy
    lidar_raw_input_sub_ = create_subscription<sensor_msgs::msg::PointCloud>(
        lidar_constants::LIDAR_RAW_TOPIC, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::PointCloud::SharedPtr msg) {
            RCLCPP_DEBUG(this->get_logger(), "Received PointCloud with %zu points", msg->points.size());
            lidarRawCallback(msg);
        });

    lidar_raw_input_sub2_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_constants::LIDAR_RAW_TOPIC2, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            RCLCPP_DEBUG(this->get_logger(), "Received PointCloud2 with %dx%d points", msg->width, msg->height);
            lidarRawCallback2(msg);
        });

    // Output publishers
    detected_cones_pub_ = create_publisher<dv_msgs::msg::IndexedTrack>("/perception/cones", 10);
    filtered_points_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/perception/filtered_points", 10);

    RCLCPP_INFO(get_logger(), "=== MODULAR LIDAR PROCESSING NODE STARTED ===");
    RCLCPP_INFO(get_logger(), "Dual input topics: %s, %s", 
                lidar_constants::LIDAR_RAW_TOPIC, lidar_constants::LIDAR_RAW_TOPIC2);
    RCLCPP_INFO(get_logger(), "Output topic: /perception/cones");
    RCLCPP_INFO(get_logger(), "Filtered points visualization topic: /perception/filtered_points");
}

ProcessLidar::~ProcessLidar() {
    RCLCPP_INFO(get_logger(), "------------------------------------");
    RCLCPP_INFO(get_logger(), "--- Final Model Accuracy Report ---");

    long long tp_blue = g_true_positives_blue.load();
    long long tp_yellow = g_true_positives_yellow.load();
    long long fp_blue = g_false_positives_blue.load();
    long long fp_yellow = g_false_positives_yellow.load();
    long long rej_blue = g_rejected_as_blue.load();
    long long rej_yellow = g_rejected_as_yellow.load();

    long long total_actual_blue = tp_blue + fp_yellow + rej_blue;
    long long total_actual_yellow = tp_yellow + fp_blue + rej_yellow;
    long long total_clusters = total_actual_blue + total_actual_yellow;

    RCLCPP_INFO(get_logger(), "Total Blue/Yellow Clusters Encountered: %lld", total_clusters);
    RCLCPP_INFO(get_logger(), " ");

    if (total_actual_blue > 0) {
        RCLCPP_INFO(get_logger(), "--- BLUE CONES (Actual: %lld) ---", total_actual_blue);
        RCLCPP_INFO(get_logger(), "Correctly Detected (TP): %lld", tp_blue);
        RCLCPP_INFO(get_logger(), "Incorrectly Detected as Yellow (FN): %lld", fp_yellow);
        RCLCPP_INFO(get_logger(), "Rejected by Classifiers: %lld", rej_blue);
        double blue_recall = 100.0 * tp_blue / total_actual_blue;
        RCLCPP_INFO(get_logger(), "Recall (Sensitivity): %.2f%%", blue_recall);
        if ((tp_blue + fp_blue) > 0) {
            double blue_precision = 100.0 * tp_blue / (tp_blue + fp_blue);
            RCLCPP_INFO(get_logger(), "Precision: %.2f%%", blue_precision);
        }
    }
    
    RCLCPP_INFO(get_logger(), " ");

    if (total_actual_yellow > 0) {
        RCLCPP_INFO(get_logger(), "--- YELLOW CONES (Actual: %lld) ---", total_actual_yellow);
        RCLCPP_INFO(get_logger(), "Correctly Detected (TP): %lld", tp_yellow);
        RCLCPP_INFO(get_logger(), "Incorrectly Detected as Blue (FN): %lld", fp_blue);
        RCLCPP_INFO(get_logger(), "Rejected by Classifiers: %lld", rej_yellow);
        double yellow_recall = 100.0 * tp_yellow / total_actual_yellow;
        RCLCPP_INFO(get_logger(), "Recall (Sensitivity): %.2f%%", yellow_recall);
        if ((tp_yellow + fp_yellow) > 0) {
            double yellow_precision = 100.0 * tp_yellow / (tp_yellow + fp_yellow);
            RCLCPP_INFO(get_logger(), "Precision: %.2f%%", yellow_precision);
        }
    }
    
    RCLCPP_INFO(get_logger(), " ");

    if (total_clusters > 0) {
        long long total_correct = tp_blue + tp_yellow;
        double overall_accuracy = 100.0 * total_correct / total_clusters;
        RCLCPP_INFO(get_logger(), "Overall Accuracy (Correct / (TP+FN+Rejected)): %.2f%%", overall_accuracy);
    }
    
    RCLCPP_INFO(get_logger(), "------------------------------------");
    RCLCPP_INFO(get_logger(), "LiDAR Node shutdown complete.");
}

void ProcessLidar::initializeComponents() {
    try {
        ml_classifier_ = std::make_unique<ConeClassifier>(env_);
        heuristic_classifier_ = std::make_unique<HeuristicClassifier>();
        point_cloud_processor_ = std::make_unique<PointCloudProcessor>();
        cluster_processor_ = std::make_unique<ClusterProcessor>();
        
        loadONNXModel();
        
        if (validateComponents()) {
            RCLCPP_INFO(get_logger(), "All modular components initialized successfully");
        } else {
            RCLCPP_FATAL(get_logger(), "Component validation failed");
            rclcpp::shutdown();
        }
    }
    catch (const std::exception& e) {
        RCLCPP_FATAL(get_logger(), "Component initialization failed: %s", e.what());
        rclcpp::shutdown();
    }
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
    
    // Comprehensive model path search with redundancy
    std::vector<std::string> possible_paths = {
        model_path,
        package_share_directory + "/share/perception_winter/cone_model.onnx",
        package_share_directory + "/cone_model.onnx", 
        "./cone_model.onnx",
        "/home/ishaan/Desktop/DebugFInal/install/perception_winter/share/perception_winter/cone_model.onnx"
    };
    
    bool model_found = false;
    for (const auto& path : possible_paths) {
        if (std::filesystem::exists(path)) {
            model_path = path;
            model_found = true;
            RCLCPP_INFO(get_logger(), "Found model at: %s", path.c_str());
            break;
        }
    }
    
    if (!model_found) {
        RCLCPP_ERROR(get_logger(), "ONNX model file not found in any search path");
        return;
    }
    
    if (ml_classifier_ && ml_classifier_->initialize(model_path)) {
        RCLCPP_INFO(get_logger(), "ONNX model loaded successfully: %s", model_path.c_str());
    } else {
        RCLCPP_ERROR(get_logger(), "Failed to initialize ONNX model: %s", model_path.c_str());
    }
}

bool ProcessLidar::validateComponents() const {
    bool all_valid = true;
    
    if (!ml_classifier_) {
        RCLCPP_ERROR(get_logger(), "ML Classifier component missing");
        all_valid = false;
    }
    if (!heuristic_classifier_) {
        RCLCPP_ERROR(get_logger(), "Heuristic Classifier component missing");
        all_valid = false;
    }
    if (!point_cloud_processor_) {
        RCLCPP_ERROR(get_logger(), "Point Cloud Processor component missing");
        all_valid = false;
    }
    if (!cluster_processor_) {
        RCLCPP_ERROR(get_logger(), "Cluster Processor component missing");
        all_valid = false;
    }
    
    return all_valid;
}

void ProcessLidar::processPointCloudData(std::vector<Point4D>& points, const std_msgs::msg::Header& header) {
    (void)header; // Currently unused
    
    if (points.empty()) {
        RCLCPP_WARN(get_logger(), "Empty point cloud data received");
        return;
    }

    total_processed_frames_++;
    
    auto pipeline_start = std::chrono::steady_clock::now();
    bool success = executeProcessingPipeline(points);
    auto pipeline_end = std::chrono::steady_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(pipeline_end - pipeline_start);
    
    if (success) {
        publishProcessingMetrics(duration);
    } else {
        failed_processing_attempts_++;
        RCLCPP_WARN(get_logger(), "Processing pipeline failed for frame %lld", total_processed_frames_.load());
    }
}

bool ProcessLidar::executeProcessingPipeline(const std::vector<Point4D>& points) {
    try {
        RCLCPP_INFO(get_logger(), "=== STARTING PROCESSING PIPELINE ===");
        RCLCPP_INFO(get_logger(), "Input points: %zu", points.size());

        // Stage 1: Point Cloud Filtering
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        if (!point_cloud_processor_->filterCarBodyAndROI(points, cloud, get_logger())) {
            RCLCPP_WARN(get_logger(), "Stage 1 failed: No points after filtering");
            return false;
        }

        // Stage 2: Ground Removal
        auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        if (!point_cloud_processor_->removeGroundPlane(cloud, cloud_filtered, get_logger())) {
            RCLCPP_WARN(get_logger(), "Stage 2 failed: No points after ground removal");
            return false;
        }

        // Stage 3: Clustering
        auto clusters = cluster_processor_->clusterPoints(cloud_filtered, get_logger());
        if (clusters.empty()) {
            RCLCPP_WARN(get_logger(), "Stage 3 failed: No clusters found");
            publishConeClusterPoints(clusters);
            return true; // No clusters is not necessarily a failure
        }

        // Stage 4: Cluster Filtering
        std::vector<bool> orange_candidates;
        auto filtered_clusters = cluster_processor_->filterClustersBySize(clusters, orange_candidates, get_logger());

        // Stage 5: Cone Detection
        std::vector<Point3D> cone_positions;
        std::vector<int> cone_colors;
        
        if (!filtered_clusters.empty()) {
            cluster_processor_->detectConesInClusters(filtered_clusters, orange_candidates, 
                                                     cone_positions, cone_colors,
                                                     *ml_classifier_, *heuristic_classifier_, get_logger());

            // Stage 6: Publish Results
            publishDetectedCones(cone_positions, cone_colors);
        } else {
            RCLCPP_WARN(get_logger(), "Stage 5: No valid clusters after filtering");
            publishDetectedCones({}, {});
        }
        
        // Always publish clustered points for visualization
        // This is crucial for the FilteredPointsVisualNode to work
        publishConeClusterPoints(clusters);

        RCLCPP_INFO(get_logger(), "=== PROCESSING PIPELINE COMPLETED SUCCESSFULLY ===");
        return true;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Processing pipeline exception: %s", e.what());
        return false;
    }
}

void ProcessLidar::publishDetectedCones(const std::vector<Point3D>& positions, const std::vector<int>& colors) {
    dv_msgs::msg::IndexedTrack track_msg;

    int yellow_count = 0;
    int blue_count = 0;
    int orange_count = 0;

    for (size_t i = 0; i < positions.size(); ++i) {
        double x = positions[i][0];
        double y = positions[i][1];
        double z = positions[i][2];

        // Distance and angle validation
        if (x < 3.35 || x > 12) continue;
        
        double range = sqrt(x * x + y * y);
        double angle = atan2(y, x);
        
        dv_msgs::msg::IndexedCone cone_msg;
        cone_msg.location.x = range;
        cone_msg.location.y = angle;
        cone_msg.location.z = z;
        cone_msg.color = colors[i];
        cone_msg.index = i;
        track_msg.track.push_back(cone_msg);

        // Color counting for diagnostics
        if (colors[i] == dv_msgs::msg::IndexedCone::YELLOW) {
            yellow_count++;
        } else if (colors[i] == dv_msgs::msg::IndexedCone::BLUE) {
            blue_count++;
        } else if (colors[i] == dv_msgs::msg::IndexedCone::ORANGE_BIG) {
            orange_count++;
        }
    }
    
    if (!track_msg.track.empty()) {
        detected_cones_pub_->publish(track_msg);
        RCLCPP_DEBUG(get_logger(), "Published %zu cones to /perception/cones", track_msg.track.size());
    }

    RCLCPP_INFO(get_logger(), "Detected cones - Yellow: %d, Blue: %d, Orange: %d", 
                yellow_count, blue_count, orange_count);
}

void ProcessLidar::publishConeClusterPoints(const std::vector<Cluster>& cone_clusters) {
    if (!filtered_points_pub_) {
        RCLCPP_WARN(get_logger(), "Filtered points publisher not available");
        return;
    }

    auto message = std_msgs::msg::Float32MultiArray();
    
    // Publish all cluster points for visualization
    // Format: [x1, y1, z1, x2, y2, z2, ...] for compatibility with FilteredPointsVisualNode
    size_t total_points = 0;
    for (const auto& cluster : cone_clusters) {
        total_points += cluster.size();
        for (const auto& point : cluster) {
            message.data.push_back(static_cast<float>(point[0])); // x
            message.data.push_back(static_cast<float>(point[1])); // y  
            message.data.push_back(static_cast<float>(point[2])); // z
            // Note: We're not including intensity to match the visualizer's expectation of 3 floats per point
        }
    }
    
    filtered_points_pub_->publish(message);
    
    RCLCPP_DEBUG(get_logger(), "Published %zu clustered points (from %zu clusters) for visualization", 
                 total_points, cone_clusters.size());
}

void ProcessLidar::publishProcessingMetrics(const std::chrono::milliseconds& duration) {
    RCLCPP_INFO(get_logger(), "Frame processing time: %ld ms", duration.count());
}

void ProcessLidar::lidarRawCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg) {
    auto start_time = std::chrono::steady_clock::now();

    try {
        auto points = PointCloudExtractor::fromPointCloud(msg);
        processPointCloudData(points, msg->header);
    } 
    catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "PointCloud processing error: %s", e.what());
        failed_processing_attempts_++;
    }
    
    auto end_time = std::chrono::steady_clock::now();  
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    RCLCPP_DEBUG(get_logger(), "PointCloud callback completed in %ld ms", duration.count());
}

void ProcessLidar::lidarRawCallback2(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto start_time = std::chrono::steady_clock::now();

    try {
        auto points = PointCloudExtractor::fromPointCloud2(msg);
        processPointCloudData(points, msg->header);
    } 
    catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "PointCloud2 processing error: %s", e.what());
        failed_processing_attempts_++;
    }
    
    auto end_time = std::chrono::steady_clock::now();  
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    RCLCPP_DEBUG(get_logger(), "PointCloud2 callback completed in %ld ms", duration.count());
}