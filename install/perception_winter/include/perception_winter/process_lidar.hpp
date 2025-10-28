/**
 * @file process_lidar.hpp
 * @brief Optimized LiDAR processing node implementation
 * @author Siddhesh Phadke
 */

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

// Add ONNX Runtime include
#include <onnxruntime_cxx_api.h>

// Forward declarations for better compilation
namespace pcl {
    class PointXYZI;
}

namespace perception_winter {

// Constants namespace for better organization
namespace lidar_constants {
    // Topic names
    constexpr auto LIDAR_RAW_TOPIC = "/carmaker/pointcloud";
    constexpr auto LIDAR_RAW_TOPIC2 = "/carmaker/pointcloud2";

    // Ground removal parameters from working code
    constexpr double RANSAC_THRESHOLD = 0.015;
    constexpr double MIN_Z_NORMAL_COMPONENT = 0.80;
    constexpr double MAX_SLOPE_DEVIATION_DEG = 40.0;
    constexpr int MAX_GROUND_ITERATIONS = 8;
    constexpr size_t MIN_POINTS_FOR_PLANE = 150;

    // Clustering parameters from working code
    constexpr double DBSCAN_EPSILON = 0.20;
    constexpr int DBSCAN_MINPOINTS = 3;

    // ROI values from working code
    constexpr double ROI_Y_MIN = -3.50;
    constexpr double ROI_Y_MAX = 3.50;
    constexpr double ROI_Z_MIN = -0.63;
    constexpr double ROI_Z_MAX = 0.40;

    // Car body dimensions from working code
    constexpr double CAR_FRONT_X = 1.5;
    constexpr double CAR_SIDE_Y = 1.25;

    // Cone parameters from working code
    constexpr double CONE_BASE_RADIUS = 0.12;
    constexpr double LIDAR_OFFSET = 1.532;
    constexpr double CONE_HEIGHT = 0.1629;

    // ML Model parameters
    constexpr int NUM_BINS = 10;
    constexpr int FEATURE_SIZE = 20;
    constexpr float Z_MIN = -0.640f;
    constexpr float Z_MAX = -0.300f;
    constexpr float BIN_WIDTH = (Z_MAX - Z_MIN) / NUM_BINS;
    constexpr double CONFIDENCE_THRESHOLD = 0.995;
}

// Type aliases for better readability
using Point4D = std::array<double, 4>; // x, y, z, intensity
using Point3D = std::array<double, 3>; // x, y, z
using Cluster = std::vector<Point4D>;
using PointCloudPtr = pcl::PointCloud<pcl::PointXYZI>::Ptr;

// --- MODULAR COMPONENTS ---

/**
 * @brief ML Classifier for cone color detection
 */
class ConeClassifier {
public:
    explicit ConeClassifier(Ort::Env& env);
    bool initialize(const std::string& model_path);
    std::optional<int> classify(const Cluster& cluster);
    
private:
    std::vector<float> createFeatureVector(const Cluster& cluster) const;
    
    Ort::Env& env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<int64_t> input_node_dims_;
    double confidence_threshold_ = lidar_constants::CONFIDENCE_THRESHOLD;
};

/**
 * @brief Heuristic Classifier using curve fitting
 */
class HeuristicClassifier {
public:
    bool classifyCone(const std::vector<double> &y_vals, const std::vector<double> &x_vals);
    std::vector<double> movingAverage(const std::vector<double> &data, int kernel);
    std::optional<int> classify(const Cluster& cluster);
};

/**
 * @brief Point Cloud Processor for filtering and ground removal
 */
class PointCloudProcessor {
public:
    PointCloudProcessor() = default;
    
    bool filterCarBodyAndROI(const std::vector<Point4D>& input_points, PointCloudPtr output_cloud);
    bool removeGroundPlane(PointCloudPtr cloud, PointCloudPtr non_ground_cloud);
    
private:
    bool isValidGroundPlane(const Eigen::Vector3f& normal, const std::optional<Eigen::Vector3f>& reference_normal) const;
};

/**
 * @brief Cluster Processor for DBSCAN and cone detection
 */
class ClusterProcessor {
public:
    ClusterProcessor() = default;
    
    std::vector<Cluster> clusterPoints(const PointCloudPtr cloud);
    std::vector<Cluster> filterClustersBySize(const std::vector<Cluster>& clusters, std::vector<bool>& orange_candidates);
    
    void detectConesInClusters(const std::vector<Cluster>& clusters, 
                              const std::vector<bool>& orange_candidates,
                              std::vector<Point3D>& positions, 
                              std::vector<int>& colors,
                              ConeClassifier& ml_classifier,
                              HeuristicClassifier& heuristic_classifier);
    
    void printClusterStats(const std::vector<Cluster>& clusters, rclcpp::Logger logger) const;
    
private:
    Point3D calculateConePosition(const Cluster& cluster);
    double getMedian(const Cluster& points, size_t idx) const;
};

/**
 * @brief Data Extractor for different point cloud formats
 */
class PointCloudExtractor {
public:
    static std::vector<Point4D> fromPointCloud(const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg);
    static std::vector<Point4D> fromPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg);
};

// --- MAIN LIDAR PROCESSING CLASS ---

class ProcessLidar : public rclcpp::Node {
public:
    ProcessLidar();
    ~ProcessLidar();

private:
    // Callbacks
    void lidarRawCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg);
    void lidarRawCallback2(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // Main processing pipeline
    void processPointCloudData(std::vector<Point4D>& points, const std_msgs::msg::Header& header);

    // Component initialization
    void initializeComponents();
    void loadONNXModel();

    // Publishers
    void publishDetectedCones(const std::vector<Point3D>& positions, const std::vector<int>& colors);
    void publishConeClusterPoints(const std::vector<Cluster>& cone_clusters);

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr lidar_raw_input_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_raw_input_sub2_;

    // Publisher objects
    rclcpp::Publisher<dv_msgs::msg::IndexedTrack>::SharedPtr detected_cones_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr filtered_points_pub_;

    // --- MODULAR COMPONENTS ---
    std::unique_ptr<ConeClassifier> ml_classifier_;
    std::unique_ptr<HeuristicClassifier> heuristic_classifier_;
    std::unique_ptr<PointCloudProcessor> point_cloud_processor_;
    std::unique_ptr<ClusterProcessor> cluster_processor_;

    // ONNX Environment (must outlive the classifier)
    Ort::Env env_;

    RCLCPP_DISABLE_COPY(ProcessLidar)
};

} // namespace perception_winter

#endif // PROCESS_LIDAR_HPP_