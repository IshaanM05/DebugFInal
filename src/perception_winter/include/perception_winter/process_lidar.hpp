#ifndef PROCESS_LIDAR_HPP_
#define PROCESS_LIDAR_HPP_

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "dv_msgs/msg/indexed_track.hpp"
#include "dv_msgs/msg/indexed_cone.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <array>
#include <memory>
#include <optional>
#include <Eigen/Core>
#include <string>
#include <atomic>

// ONNX Runtime for ML inference
#include <onnxruntime_cxx_api.h>

namespace pcl {
    class PointXYZI;
}

namespace perception_winter {

// =============================================
// TYPE ALIASES FOR CODE CLARITY
// =============================================

using Point4D = std::array<double, 4>; // x, y, z, intensity
using Point3D = std::array<double, 3>; // x, y, z
using Cluster = std::vector<Point4D>;
using PointCloudPtr = pcl::PointCloud<pcl::PointXYZI>::Ptr;

// =============================================
// LIDAR CONSTANTS CONFIGURATION STRUCTURE
// =============================================

/**
 * @brief Centralized configuration structure for LiDAR processing parameters
 * @note All values are loaded from YAML config file for runtime flexibility
 */
struct LidarConfig {
    // Topic Configuration
    std::string lidar_raw_topic;
    std::string lidar_raw_topic2;
    
    // Ground Removal Parameters (Optimized for RANSAC)
    double ransac_threshold;
    double min_z_normal_component;
    double max_slope_deviation_deg;
    int max_ground_iterations;
    size_t min_points_for_plane;
    
    // Clustering Parameters (Optimized for DBSCAN)
    double dbscan_epsilon;
    int dbscan_minpoints;
    
    // Region of Interest (ROI) Boundaries
    double roi_y_min;
    double roi_y_max;
    double roi_z_min;
    double roi_z_max;
    double roi_x_max;  // Optimized from second code
    
    // Vehicle Body Exclusion Zone
    double car_front_x;
    double car_side_y;
    
    // Cone Physical Properties
    double cone_base_radius;
    double lidar_offset;
    double cone_height;
    
    // ML Model Configuration
    int num_bins;
    int feature_size;
    float z_min;
    float z_max;
    float bin_width;
    double confidence_threshold;
    
    // Cluster Filtering Parameters (Optimized from second code)
    double min_cluster_height;
    double max_cluster_height;
    double max_cluster_width;
    size_t min_cluster_points;
    
    // Orange Cone Detection
    double orange_cone_distance_threshold;
    size_t orange_cone_min_points;
    
    // Visualization Control
    bool publish_cluster_centers;
    bool publish_filtered_points;
    
    // ONNX Model Configuration
    std::vector<std::string> onnx_model_paths;
    std::string default_model_path;
    
    // Cone Detection Parameters
    double cone_distance_x_min;
    double cone_distance_x_max;
    double intensity_threshold_blue;
    
    // Cone Position Calculation Weights
    double cone_position_w_median;
    double cone_position_w_min_x;
    double cone_position_w_min_y;
    
    // Heuristic Classifier Parameters
    double moving_average_factor;
    int min_kernel_size;
};

// =============================================
// ACCURACY METRICS TRACKING
// =============================================

/**
 * @brief Thread-safe accuracy metrics for cone classification
 */
class AccuracyMetrics {
public:
    struct Metrics {
        std::atomic<long long> true_positives_blue{0};
        std::atomic<long long> true_positives_yellow{0};
        std::atomic<long long> false_positives_blue{0};
        std::atomic<long long> false_positives_yellow{0};
        std::atomic<long long> rejected_as_blue{0};
        std::atomic<long long> rejected_as_yellow{0};
    };

    static Metrics& getGlobalMetrics() {
        static Metrics instance;
        return instance;
    }

    static void updateMetrics(bool is_ground_truth_blue, int final_color, bool was_rejected = false);
    static void printComprehensiveReport(rclcpp::Logger logger);
};

// =============================================
// MACHINE LEARNING CLASSIFIER
// =============================================

/**
 * @brief ONNX-based ML classifier for cone color detection
 * @details Uses neural network inference for robust cone classification
 */
class ConeClassifier {
public:
    ConeClassifier(Ort::Env& env, const LidarConfig& config);
    ~ConeClassifier() = default;

    // Non-copyable, non-movable
    ConeClassifier(const ConeClassifier&) = delete;
    ConeClassifier& operator=(const ConeClassifier&) = delete;
    ConeClassifier(ConeClassifier&&) = delete;
    ConeClassifier& operator=(ConeClassifier&&) = delete;

    bool initialize(const std::string& model_path);
    std::optional<int> classify(const Cluster& cluster);

    // Getters for configuration
    double getConfidenceThreshold() const { return config_.confidence_threshold; }

private:
    std::vector<float> createFeatureVector(const Cluster& cluster) const;
    
    Ort::Env& env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<int64_t> input_node_dims_;
    const LidarConfig& config_;
};

// =============================================
// HEURISTIC CLASSIFIER
// =============================================

/**
 * @brief Curve-fitting based heuristic classifier
 * @details Provides redundancy and validation for ML classification
 */
class HeuristicClassifier {
public:
    HeuristicClassifier(const LidarConfig& config);
    ~HeuristicClassifier() = default;

    // Non-copyable, non-movable
    HeuristicClassifier(const HeuristicClassifier&) = delete;
    HeuristicClassifier& operator=(const HeuristicClassifier&) = delete;
    HeuristicClassifier(HeuristicClassifier&&) = delete;
    HeuristicClassifier& operator=(HeuristicClassifier&&) = delete;

    std::optional<int> classify(const Cluster& cluster);

private:
    bool classifyCone(const std::vector<double> &y_vals, const std::vector<double> &x_vals);
    std::vector<double> movingAverage(const std::vector<double> &data, int kernel);
    
    const LidarConfig& config_;
};

// =============================================
// POINT CLOUD PROCESSOR
// =============================================

/**
 * @brief Handles point cloud filtering and ground removal
 * @details Optimized for real-time processing with multiple redundancy checks
 */
class PointCloudProcessor {
public:
    PointCloudProcessor(const LidarConfig& config);
    ~PointCloudProcessor() = default;

    // Non-copyable, non-movable
    PointCloudProcessor(const PointCloudProcessor&) = delete;
    PointCloudProcessor& operator=(const PointCloudProcessor&) = delete;
    PointCloudProcessor(PointCloudProcessor&&) = delete;
    PointCloudProcessor& operator=(PointCloudProcessor&&) = delete;

    bool filterCarBodyAndROI(const std::vector<Point4D>& input_points, 
                            PointCloudPtr output_cloud,
                            rclcpp::Logger logger);
    
    bool removeGroundPlane(PointCloudPtr cloud,
                          PointCloudPtr non_ground_cloud,
                          rclcpp::Logger logger);

private:
    bool isValidGroundPlane(const Eigen::Vector3f& normal, 
                          const std::optional<Eigen::Vector3f>& reference_normal) const;
    
    const LidarConfig& config_;
};

// =============================================
// CLUSTER PROCESSOR
// =============================================

/**
 * @brief Handles clustering, filtering, and cone detection
 * @details Uses DBSCAN clustering with multiple validation layers
 */
class ClusterProcessor {
public:
    ClusterProcessor(const LidarConfig& config);
    ~ClusterProcessor() = default;

    // Non-copyable, non-movable
    ClusterProcessor(const ClusterProcessor&) = delete;
    ClusterProcessor& operator=(const ClusterProcessor&) = delete;
    ClusterProcessor(ClusterProcessor&&) = delete;
    ClusterProcessor& operator=(ClusterProcessor&&) = delete;

    std::vector<Cluster> clusterPoints(const PointCloudPtr cloud, rclcpp::Logger logger);
    
    std::vector<Cluster> filterClustersBySize(const std::vector<Cluster>& clusters, 
                                             std::vector<bool>& orange_candidates,
                                             rclcpp::Logger logger);
    
    void detectConesInClusters(const std::vector<Cluster>& clusters,
                              const std::vector<bool>& orange_candidates,
                              std::vector<Point3D>& positions, 
                              std::vector<int>& colors,
                              ConeClassifier& ml_classifier,
                              HeuristicClassifier& heuristic_classifier,
                              rclcpp::Logger logger);

    void printClusterStats(const std::vector<Cluster>& clusters, rclcpp::Logger logger) const;

    // New method to calculate cluster centers for visualization
    std::vector<Point3D> calculateClusterCenters(const std::vector<Cluster>& clusters) const;

private:
    Point3D calculateConePosition(const Cluster& cluster);
    double getMedian(const Cluster& points, size_t idx) const;
    bool isOrangeConeCandidate(const Cluster& cluster) const;
    
    const LidarConfig& config_;
};

// =============================================
// POINT CLOUD EXTRACTOR
// =============================================

/**
 * @brief Handles multiple point cloud format conversions
 * @details Supports both PointCloud and PointCloud2 messages with redundancy
 */
class PointCloudExtractor {
public:
    PointCloudExtractor() = delete; // Static class

    static std::vector<Point4D> fromPointCloud(const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg);
    static std::vector<Point4D> fromPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg);

private:
    static bool validatePointCloud(const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg);
    static bool validatePointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg);
};

// =============================================
// MAIN LIDAR PROCESSING NODE
// =============================================

/**
 * @brief Main LiDAR processing node with modular pipeline architecture
 * @details Implements redundant processing paths and comprehensive error handling
 */
class ProcessLidar : public rclcpp::Node {
public:
    ProcessLidar();
    ~ProcessLidar();

    // Non-copyable, non-movable
    ProcessLidar(const ProcessLidar&) = delete;
    ProcessLidar& operator=(const ProcessLidar&) = delete;
    ProcessLidar(ProcessLidar&&) = delete;
    ProcessLidar& operator=(ProcessLidar&&) = delete;

private:
    // Message Callbacks with Redundancy
    void lidarRawCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg);
    void lidarRawCallback2(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // Core Processing Pipeline
    void processPointCloudData(std::vector<Point4D>& points, const std_msgs::msg::Header& header);
    bool executeProcessingPipeline(const std::vector<Point4D>& points);

    // Component Management
    void initializeComponents();
    void loadONNXModel();
    bool validateComponents() const;
    void loadLidarConfig();

    // Publication Methods
    void publishDetectedCones(const std::vector<Point3D>& positions, const std::vector<int>& colors);
    void publishConeClusterPoints(const std::vector<Cluster>& cone_clusters);
    void publishClusterCenters(const std::vector<Cluster>& filtered_clusters);
    void publishProcessingMetrics(const std::chrono::milliseconds& duration);

    // Subscribers (Dual input for redundancy)
    rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr lidar_raw_input_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_raw_input_sub2_;

    // Publishers
    rclcpp::Publisher<dv_msgs::msg::IndexedTrack>::SharedPtr detected_cones_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr filtered_points_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr cluster_centers_pub_; // New publisher for cluster centers

    // Modular Processing Components
    std::unique_ptr<ConeClassifier> ml_classifier_;
    std::unique_ptr<HeuristicClassifier> heuristic_classifier_;
    std::unique_ptr<PointCloudProcessor> point_cloud_processor_;
    std::unique_ptr<ClusterProcessor> cluster_processor_;

    // ONNX Runtime Environment
    Ort::Env env_;

    // Configuration
    LidarConfig lidar_config_;

    // Performance Monitoring
    std::atomic<long long> total_processed_frames_{0};
    std::atomic<long long> failed_processing_attempts_{0};
};

} // namespace perception_winter

#endif // PROCESS_LIDAR_HPP_