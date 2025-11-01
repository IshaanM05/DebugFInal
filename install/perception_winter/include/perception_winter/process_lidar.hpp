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
// CONSTANTS CONFIGURATION
// =============================================

/**
 * @brief Centralized constants for LiDAR processing parameters
 * @note All values are optimized for performance and accuracy
 */
namespace lidar_constants {
    // Topic Configuration
    constexpr auto LIDAR_RAW_TOPIC = "/carmaker/pointcloud";
    constexpr auto LIDAR_RAW_TOPIC2 = "/velodyne_points";

    // Ground Removal Parameters (Optimized for RANSAC)
    constexpr double RANSAC_THRESHOLD = 0.015;
    constexpr double MIN_Z_NORMAL_COMPONENT = 0.80;
    constexpr double MAX_SLOPE_DEVIATION_DEG = 40.0;
    constexpr int MAX_GROUND_ITERATIONS = 8;
    constexpr size_t MIN_POINTS_FOR_PLANE = 150;

    // Clustering Parameters (Optimized for DBSCAN)
    constexpr double DBSCAN_EPSILON = 0.20;
    constexpr int DBSCAN_MINPOINTS = 3;

    // Region of Interest (ROI) Boundaries
    constexpr double ROI_Y_MIN = -3.50;
    constexpr double ROI_Y_MAX = 3.50;
    constexpr double ROI_Z_MIN = -0.63;
    constexpr double ROI_Z_MAX = 0.40;
    constexpr double ROI_X_MAX = 12.0;  // Optimized from second code

    // Vehicle Body Exclusion Zone
    constexpr double CAR_FRONT_X = 1.15;
    constexpr double CAR_SIDE_Y = 1.25;

    // Cone Physical Properties
    constexpr double CONE_BASE_RADIUS = 0.12;
    constexpr double LIDAR_OFFSET = 1.532;
    constexpr double CONE_HEIGHT = 0.1629;

    // ML Model Configuration
    constexpr int NUM_BINS = 10;
    constexpr int FEATURE_SIZE = 20;
    constexpr float Z_MIN = -0.640f;
    constexpr float Z_MAX = -0.300f;
    constexpr float BIN_WIDTH = (Z_MAX - Z_MIN) / NUM_BINS;
    constexpr double CONFIDENCE_THRESHOLD = 0.95;

    // Cluster Filtering Parameters (Optimized from second code)
    constexpr double MIN_CLUSTER_HEIGHT = 0.15;
    constexpr double MAX_CLUSTER_HEIGHT = 0.60;
    constexpr double MAX_CLUSTER_WIDTH = 0.75;
    constexpr int MIN_CLUSTER_POINTS = 4;

    // Orange Cone Detection
    constexpr double ORANGE_CONE_DISTANCE_THRESHOLD = 10.0;
    constexpr int ORANGE_CONE_MIN_POINTS = 20;
}

// =============================================
// TYPE ALIASES FOR CODE CLARITY
// =============================================

using Point4D = std::array<double, 4>; // x, y, z, intensity
using Point3D = std::array<double, 3>; // x, y, z
using Cluster = std::vector<Point4D>;
using PointCloudPtr = pcl::PointCloud<pcl::PointXYZI>::Ptr;

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
    explicit ConeClassifier(Ort::Env& env);
    ~ConeClassifier() = default;

    // Non-copyable, non-movable
    ConeClassifier(const ConeClassifier&) = delete;
    ConeClassifier& operator=(const ConeClassifier&) = delete;
    ConeClassifier(ConeClassifier&&) = delete;
    ConeClassifier& operator=(ConeClassifier&&) = delete;

    bool initialize(const std::string& model_path);
    std::optional<int> classify(const Cluster& cluster);

    // Getters for configuration
    double getConfidenceThreshold() const { return confidence_threshold_; }

private:
    std::vector<float> createFeatureVector(const Cluster& cluster) const;
    
    Ort::Env& env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<int64_t> input_node_dims_;
    double confidence_threshold_ = lidar_constants::CONFIDENCE_THRESHOLD;
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
    HeuristicClassifier() = default;
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
    PointCloudProcessor() = default;
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
    ClusterProcessor() = default;
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

    // Performance Monitoring
    std::atomic<long long> total_processed_frames_{0};
    std::atomic<long long> failed_processing_attempts_{0};

    // Visualization control parameters
    bool publish_cluster_centers_ = true; // Control flag for cluster centers visualization
    bool publish_filtered_points_ = true; // Control flag for filtered points visualization
};

} // namespace perception_winter

#endif // PROCESS_LIDAR_HPP_