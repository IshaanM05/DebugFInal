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
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
        input_node_names_.push_back(input_name_ptr.get());

        auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
        output_node_names_.push_back(output_name_ptr.get());

        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_node_dims_ = input_tensor_info.GetShape();

        return true;
    } catch (const Ort::Exception& e) {
        return false;
    }
}

std::vector<float> ConeClassifier::createFeatureVector(const Cluster& cluster) const {
    std::vector<float> feature_vector(lidar_constants::NUM_BINS * 2, 0.0f);
    if (cluster.empty()) return feature_vector;

    // --- Find min/max for normalization ---
    double min_z = cluster[0][2], max_z = cluster[0][2];
    double min_intensity = cluster[0][3], max_intensity = cluster[0][3];
    for (const auto& point : cluster) {
        min_z = std::min(min_z, point[2]); max_z = std::max(max_z, point[2]);
        min_intensity = std::min(min_intensity, point[3]); max_intensity = std::max(max_intensity, point[3]);
    }

    double z_range = max_z - min_z;
    double intensity_range = max_intensity - min_intensity;
    double bin_width = (z_range > 1e-6) ? z_range / lidar_constants::NUM_BINS : 0.0;

    std::vector<double> sum_intensity(lidar_constants::NUM_BINS, 0.0);
    std::vector<int> point_count(lidar_constants::NUM_BINS, 0);

    for (const auto& point : cluster) {
        double norm_intensity = (intensity_range > 1e-6) ? (point[3] - min_intensity) / intensity_range : 0.0;
        int bin_index = (bin_width > 0) ? static_cast<int>((point[2] - min_z) / bin_width) : 0;
        bin_index = std::min(bin_index, lidar_constants::NUM_BINS - 1);
        sum_intensity[bin_index] += norm_intensity;
        point_count[bin_index]++;
    }

    auto min_max_it = std::minmax_element(point_count.begin(), point_count.end());
    float min_c = static_cast<float>(*min_max_it.first);
    float max_c = static_cast<float>(*min_max_it.second);
    float count_range = max_c - min_c;

    for (int i = 0; i < lidar_constants::NUM_BINS; ++i) {
        float normalized_count = 0.0f;
        if (count_range > 0) {
            normalized_count = (static_cast<float>(point_count[i]) - min_c) / count_range;
        } else {
            normalized_count = (min_c > 0) ? 1.0f : 0.0f;
        }
        feature_vector[i * 2 + 0] = normalized_count;
        
        feature_vector[i * 2 + 1] = (point_count[i] > 0) ? static_cast<float>(sum_intensity[i] / point_count[i]) : 0.0f;
    }

    return feature_vector;
}

std::optional<int> ConeClassifier::classify(const Cluster& cluster) {
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

    if (prediction_prob > confidence_threshold_) {
        return dv_msgs::msg::IndexedCone::BLUE;
    } else if ((1.0 - prediction_prob) > confidence_threshold_) {
        return dv_msgs::msg::IndexedCone::YELLOW;
    }
    
    return std::nullopt;
}

// =============================================
// HEURISTIC CLASSIFIER IMPLEMENTATION
// =============================================

bool HeuristicClassifier::classifyCone(const std::vector<double> &y_vals, const std::vector<double> &x_vals)
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

std::vector<double> HeuristicClassifier::movingAverage(const std::vector<double> &data, int kernel)
{
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

    auto sorted_cluster = cluster;
    std::sort(sorted_cluster.begin(), sorted_cluster.end(),
              [](const Point4D& a, const Point4D& b) { return a[2] > b[2]; });

    std::vector<double> intensity_vals;
    std::vector<double> z_vals;
    intensity_vals.reserve(sorted_cluster.size());
    z_vals.reserve(sorted_cluster.size());

    for (const auto& pt : sorted_cluster) {
        intensity_vals.push_back(pt[3]);
        z_vals.push_back(pt[2]);
    }

    int kernel = std::max(3, static_cast<int>(0.1 * intensity_vals.size()));
    if (kernel % 2 == 0) kernel += 1;
    std::vector<double> averaged_intensities = this->movingAverage(intensity_vals, kernel);
    
    bool is_yellow_heuristic = this->classifyCone(averaged_intensities, z_vals);
    return is_yellow_heuristic ? dv_msgs::msg::IndexedCone::YELLOW : dv_msgs::msg::IndexedCone::BLUE;
}

// =============================================
// POINT CLOUD PROCESSOR IMPLEMENTATION
// =============================================

bool PointCloudProcessor::filterCarBodyAndROI(const std::vector<Point4D>& input_points, 
                                              PointCloudPtr output_cloud) {
    output_cloud->reserve(input_points.size());
    
    int car_body_points = 0;
    int behind_car_points = 0;
    int outside_roi_points = 0;
    int valid_points = 0;
    
    RCLCPP_DEBUG(rclcpp::get_logger("point_cloud_processor"),
                "Starting ROI filtering with %zu input points", input_points.size());
    
    for (const auto& point : input_points) {
        double x = point[0];
        double y = point[1];
        double z = point[2];
        
        // Skip points behind the car
        if (x <= 0) {
            behind_car_points++;
            continue;
        }
        
        // Skip points that are on the car body
        bool on_car_body = (x <= lidar_constants::CAR_FRONT_X) && 
                          (std::abs(y) <= lidar_constants::CAR_SIDE_Y);
        if (on_car_body) {
            car_body_points++;
            continue;
        }
        
        // Apply ROI filtering in one step
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
    
    RCLCPP_INFO(rclcpp::get_logger("point_cloud_processor"),
                "ROI filtering: %zu -> %zu points (behind: %d, car_body: %d, outside_roi: %d, valid: %d)", 
                input_points.size(), output_cloud->size(), behind_car_points, car_body_points, outside_roi_points, valid_points);
    
    return !output_cloud->empty();
}

bool PointCloudProcessor::removeGroundPlane(PointCloudPtr cloud,
                                    PointCloudPtr non_ground_cloud) {
    if (cloud->empty()) return false;

    RCLCPP_DEBUG(rclcpp::get_logger("point_cloud_processor"),
                "Starting ground removal with %zu points", cloud->size());

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
    
    RCLCPP_INFO(rclcpp::get_logger("point_cloud_processor"),
                "Ground removal: %zu -> %zu points (%zu ground points removed in %d iterations)",
                cloud->size(), non_ground_cloud->size(), ground_cloud->size(), iterations);
    
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

// =============================================
// CLUSTER PROCESSOR IMPLEMENTATION
// =============================================

std::vector<Cluster> ClusterProcessor::clusterPoints(const PointCloudPtr cloud) {
    if (cloud->empty()) return {};

    RCLCPP_DEBUG(rclcpp::get_logger("cluster_processor"),
                "Starting clustering with %zu points", cloud->size());

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

    RCLCPP_INFO(rclcpp::get_logger("cluster_processor"),
                "Clustering: %zu points -> %zu clusters (noise points: %d)",
                cloud->size(), clusters.size(), noise_points);
    
    return clusters;
}

std::vector<Cluster> ClusterProcessor::filterClustersBySize(const std::vector<Cluster>& clusters, 
                                                           std::vector<bool>& orange_candidates) {
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
        
        if (cluster.size() < 4) {
            rejected_by_points++;
            orange_candidates.push_back(false);
            continue;
        }

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
        double distance = std::sqrt(centroid_x * centroid_x + centroid_y * centroid_y);

        // Check for orange cone candidate during filtering
        bool is_orange_candidate = false;
        orange_candidates.push_back(is_orange_candidate);

        // Regular cone filtering
        bool valid_height = (height >= 0.15 && height <= 0.5);
        bool valid_width = (width <= 0.75);
        bool valid_x_pos = (centroid_x <= lidar_constants::ROI_X_MAX); // Using the constant

        if (!valid_height) rejected_by_height++;
        if (!valid_width) rejected_by_width++;
        if (!valid_x_pos) rejected_by_x_dist++; // Increment the new counter

        if (valid_height && valid_width && valid_x_pos) { // Added the new condition
            valid_clusters.push_back(cluster);
            total_points_after += cluster.size();
        }
    }

    RCLCPP_INFO(rclcpp::get_logger("cluster_processor"),
                "Cluster filtering: %zu -> %zu clusters, points: %d -> %d "
                "(rejected: points=%d, height=%d, width=%d, x_dist=%d)", // Updated log
                clusters.size(), valid_clusters.size(), total_points_before, total_points_after,
                rejected_by_points, rejected_by_height, rejected_by_width, rejected_by_x_dist);

    return valid_clusters;
}

void ClusterProcessor::detectConesInClusters(const std::vector<Cluster>& clusters,
                                            const std::vector<bool>& orange_candidates,
                                            std::vector<Point3D>& positions,
                                            std::vector<int>& colors,
                                            ConeClassifier& ml_classifier,
                                            HeuristicClassifier& heuristic_classifier) {
    positions.reserve(clusters.size());
    colors.reserve(clusters.size());

    int orange_cones = 0;

    for (size_t i = 0; i < clusters.size(); ++i) {
        const auto& cluster = clusters[i];
        if (cluster.empty()) continue;

        // --- 1. Handle Orange Cones (Assumed correct, not part of Blue/Yellow metrics) ---
        if (orange_candidates[i]) {
            auto cone_pos = calculateConePosition(cluster);
            positions.push_back(cone_pos);
            colors.push_back(dv_msgs::msg::IndexedCone::ORANGE_BIG);
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
    long long accepted_cones = g_true_positives_blue + g_false_positives_yellow + g_true_positives_yellow + g_false_positives_blue;
    RCLCPP_INFO(rclcpp::get_logger("cluster_processor"), 
                "Cone detection: Accepted cones (ML+Heuristic agreement): %lld, Orange cones: %d",
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
        double height = max_z - min_z;
        double width_x = max_x - min_x;
        double width_y = max_y - min_y;
        
        RCLCPP_INFO(logger, "  Cluster %zu: points=%zu, height=%.3f, width=(%.3f,%.3f), avg_intensity=%.3f", 
                   i, cluster.size(), height, width_x, width_y, avg_intensity);
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

// =============================================
// POINT CLOUD EXTRACTOR IMPLEMENTATION
// =============================================

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
    std::optional<sensor_msgs::PointCloud2ConstIterator<float>> iter_intensity;
    
    // Check if intensity field exists and create iterator
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

// =============================================
// MAIN PROCESS LIDAR IMPLEMENTATION
// =============================================

ProcessLidar::ProcessLidar() : 
    Node("process_lidar"), 
    env_(ORT_LOGGING_LEVEL_WARNING, "ONNX_INFERENCE") {
    
    // Enable debug logging to see all messages
    auto debug_logger = this->get_logger();
    auto result = rcutils_logging_set_logger_level(debug_logger.get_name(), RCUTILS_LOG_SEVERITY_DEBUG);
    (void)result; // Explicitly ignore the result
    
    initializeComponents();
    
    // Subscribers
    lidar_raw_input_sub_ = create_subscription<sensor_msgs::msg::PointCloud>(
        lidar_constants::LIDAR_RAW_TOPIC, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::PointCloud::SharedPtr msg) {
            RCLCPP_DEBUG(this->get_logger(), "Received PointCloud message with %zu points", msg->points.size());
            lidarRawCallback(msg);
        });

    lidar_raw_input_sub2_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_constants::LIDAR_RAW_TOPIC2, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            RCLCPP_DEBUG(this->get_logger(), "Received PointCloud2 message with %dx%d points", msg->width, msg->height);
            lidarRawCallback2(msg);
        });

    // Publishers
    detected_cones_pub_ = create_publisher<dv_msgs::msg::IndexedTrack>("/perception/cones", 10);
    filtered_points_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/perception/filtered_points", 10);

    RCLCPP_INFO(get_logger(), "Optimized LiDAR Node with Hybrid ML/Heuristic classification started");
    RCLCPP_INFO(get_logger(), "Subscribed to both PointCloud (%s) and PointCloud2 (%s) topics", 
                lidar_constants::LIDAR_RAW_TOPIC, lidar_constants::LIDAR_RAW_TOPIC2);
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
    ml_classifier_ = std::make_unique<ConeClassifier>(env_);
    heuristic_classifier_ = std::make_unique<HeuristicClassifier>();
    point_cloud_processor_ = std::make_unique<PointCloudProcessor>();
    cluster_processor_ = std::make_unique<ClusterProcessor>();
    
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
    
    // Check if model file exists
    if (!std::filesystem::exists(model_path)) {
        RCLCPP_ERROR(get_logger(), "ONNX model file not found at: %s", model_path.c_str());
        RCLCPP_ERROR(get_logger(), "Current working directory: %s", std::filesystem::current_path().c_str());
        
        // Try to find the model file
        std::vector<std::string> possible_paths = {
            package_share_directory + "/share/perception_winter/cone_model.onnx",
            package_share_directory + "/cone_model.onnx", 
            "./cone_model.onnx",
            "/home/ishaan/Desktop/DebugFInal/install/perception_winter/share/perception_winter/cone_model.onnx"
        };
        
        for (const auto& path : possible_paths) {
            if (std::filesystem::exists(path)) {
                model_path = path;
                RCLCPP_INFO(get_logger(), "Found model at: %s", path.c_str());
                break;
            }
        }
    }
    
    if (ml_classifier_) {
        if (ml_classifier_->initialize(model_path)) {
            RCLCPP_INFO(get_logger(), "Successfully loaded ONNX model from: %s", model_path.c_str());
        } else {
            RCLCPP_ERROR(get_logger(), "Failed to initialize ONNX model from: %s", model_path.c_str());
        }
    } else {
        RCLCPP_FATAL(get_logger(), "Failed to initialize ML classifier");
        rclcpp::shutdown();
    }
}

void ProcessLidar::processPointCloudData(std::vector<Point4D>& points, const std_msgs::msg::Header& header) {
    (void)header;
    
    if (points.empty()) {
        RCLCPP_WARN(get_logger(), "No points in input cloud");
        return;
    }

    RCLCPP_INFO(get_logger(), "=== STARTING LIDAR PROCESSING PIPELINE ===");
    RCLCPP_INFO(get_logger(), "Processing point cloud with %zu points", points.size());

    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    auto pipeline_start = std::chrono::steady_clock::now();
    
    // Stage 1: Filtering
    RCLCPP_INFO(get_logger(), "--- STAGE 1: ROI and Car Body Filtering ---");
    if (!point_cloud_processor_->filterCarBodyAndROI(points, cloud)) {
        RCLCPP_WARN(get_logger(), "No points after car body and ROI filtering");
        return;
    }

    // Stage 2: Ground removal
    RCLCPP_INFO(get_logger(), "--- STAGE 2: Ground Removal ---");
    if (!point_cloud_processor_->removeGroundPlane(cloud, cloud_filtered)) {
        RCLCPP_WARN(get_logger(), "No points after ground removal");
        return;
    }

    // Stage 3: Clustering
    RCLCPP_INFO(get_logger(), "--- STAGE 3: Clustering ---");
    auto clusters = cluster_processor_->clusterPoints(cloud_filtered);
    if (clusters.empty()) {
        RCLCPP_WARN(get_logger(), "No clusters found");
        publishConeClusterPoints(clusters);
        return;
    }

    // Stage 4: Cluster filtering with orange cone candidate detection
    RCLCPP_INFO(get_logger(), "--- STAGE 4: Cluster Filtering with Orange Detection ---");
    std::vector<bool> orange_candidates;
    auto filtered_clusters = cluster_processor_->filterClustersBySize(clusters, orange_candidates);

    // Stage 5: Cone detection with hybrid ML+Heuristic classification
    RCLCPP_INFO(get_logger(), "--- STAGE 5: Hybrid ML+Heuristic Classification ---");
    std::vector<Point3D> cone_positions;
    std::vector<int> cone_colors;
    
    if (!filtered_clusters.empty()) {
        cluster_processor_->detectConesInClusters(filtered_clusters, orange_candidates, 
                                                 cone_positions, cone_colors,
                                                 *ml_classifier_, *heuristic_classifier_);

        // Stage 6: Publish results
        RCLCPP_INFO(get_logger(), "--- STAGE 6: Publishing Results ---");
        publishDetectedCones(cone_positions, cone_colors);
    } else {
        RCLCPP_WARN(get_logger(), "No valid clusters after size filtering");
        publishDetectedCones({}, {});
    }
    
    // Publish clustered points for visualization
    RCLCPP_INFO(get_logger(), "--- PUBLISHING CLUSTERED POINTS FOR VISUALIZATION ---");
    publishConeClusterPoints(clusters);

    auto pipeline_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(pipeline_end - pipeline_start);
    RCLCPP_INFO(get_logger(), "=== PROCESSING COMPLETED in %ld ms ===", duration.count());
    RCLCPP_INFO(get_logger(), "Detected %zu clusters total", clusters.size());
}

void ProcessLidar::publishDetectedCones(const std::vector<Point3D>& positions, const std::vector<int>& colors) {
    dv_msgs::msg::IndexedTrack track_msg;

    int yellow_count = 0;
    int blue_count = 0;
    int orange_count = 0;

    for (size_t i = 0; i < positions.size(); ++i) {
        dv_msgs::msg::IndexedCone cone_msg;
        double x = positions[i][0];
        double y = positions[i][1];
        double z = positions[i][2];

        if (x < 3.35 || x > 12) continue;
        
        double range = sqrt(x * x + y * y);
        double angle = atan2(y, x);
        
        cone_msg.location.x = range;
        cone_msg.location.y = angle;
        cone_msg.location.z = z;
        
        // Convert internal color to message format
        cone_msg.color = colors[i];
        cone_msg.index = i;
        track_msg.track.push_back(cone_msg);

        // Count by color
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
        RCLCPP_INFO(get_logger(), "Published %zu cones to /perception/cones", track_msg.track.size());
    }

    RCLCPP_INFO(get_logger(), "Detected cones - Yellow: %d, Blue: %d, Orange: %d", yellow_count, blue_count, orange_count);
}

void ProcessLidar::publishConeClusterPoints(const std::vector<Cluster>& cone_clusters) {
    if (!filtered_points_pub_) {
        RCLCPP_WARN(get_logger(), "Filtered points publisher not available");
        return;
    }

    auto message = std_msgs::msg::Float32MultiArray();
    
    // Publish ALL points from ALL clusters (before filtering)
    size_t total_points = 0;
    for (const auto& cluster : cone_clusters) {
        total_points += cluster.size();
        for (const auto& point : cluster) {
            message.data.push_back(static_cast<float>(point[0])); // x
            message.data.push_back(static_cast<float>(point[1])); // y  
            message.data.push_back(static_cast<float>(point[2])); // z
            message.data.push_back(static_cast<float>(point[3])); // intensity
        }
    }
    
    filtered_points_pub_->publish(message);
    
    RCLCPP_INFO(get_logger(), "Published %zu clustered points (from %zu clusters) to /perception/filtered_points", 
                 total_points, cone_clusters.size());
}

void ProcessLidar::lidarRawCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg) {
    auto start_time = std::chrono::steady_clock::now();

    try {
        RCLCPP_INFO(get_logger(), "Processing PointCloud with %zu points", msg->points.size());
        auto points = PointCloudExtractor::fromPointCloud(msg);
        processPointCloudData(points, msg->header);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "PointCloud processing error: %s", e.what());
    }
    
    auto end_time = std::chrono::steady_clock::now();  
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    RCLCPP_DEBUG(get_logger(), "PointCloud processing completed in %ld ms", duration.count());
}

void ProcessLidar::lidarRawCallback2(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto start_time = std::chrono::steady_clock::now();

    try {
        RCLCPP_INFO(get_logger(), "Processing PointCloud2 with %dx%d points", msg->width, msg->height);
        auto points = PointCloudExtractor::fromPointCloud2(msg);
        processPointCloudData(points, msg->header);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "PointCloud2 processing error: %s", e.what());
    }
    
    auto end_time = std::chrono::steady_clock::now();  
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    RCLCPP_DEBUG(get_logger(), "PointCloud2 processing completed in %ld ms", duration.count());
}