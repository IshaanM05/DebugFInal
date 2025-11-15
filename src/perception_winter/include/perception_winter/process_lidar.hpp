#ifndef PROCESS_LIDAR_HPP_
#define PROCESS_LIDAR_HPP_
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <dv_msgs/msg/indexed_track.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <onnxruntime_cxx_api.h>
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <atomic>
#include <memory>
#include <vector>
#include <optional>
#include <string>
#include <utility> // for std::pair

// =============================================
// LIDAR CONFIG STRUCTURE
// =============================================

namespace perception_winter {

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
    double roi_x_max;

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

    // Cluster Filtering Parameters
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

    // Enhanced ground removal parameters
    struct GroundRemovalConfig {
        // Preprocessing
        double voxel_downsample_size;
        double single_axis_smoothing_cell_size;
        double ground_smoothing_threshold;
        
        // Angular Sector Fitting
        int num_angular_sectors;
        int min_points_per_sector;
        double lowest_points_ratio;
        
        // Zonal PCA Fitting  
        std::vector<std::pair<double, double>> concentric_rings;
        int segments_per_ring;
        int min_points_per_zone;
        
        // Elevation Grid Filtering
        double grid_cell_size;
        int min_points_per_cell;
        double adaptive_variance_base;
        double adaptive_variance_scale;
        
        // Cone Restoration
        double base_cylinder_radius;
        double base_cylinder_height;
        int min_cone_cluster_points;
        
        // Performance
        bool use_parallel_processing;
        int max_threads;
    } ground_removal;
};

// =============================================
// TYPE DEFINITIONS (PRESERVED)
// =============================================

using Point4D = std::array<double, 4>; // x, y, z, intensity
using Point3D = std::array<double, 3>; // x, y, z  
using Cluster = std::vector<Point4D>;
using PointCloudPtr = pcl::PointCloud<pcl::PointXYZI>::Ptr;

// =============================================
// ORIGINAL CLASS DEFINITIONS (PRESERVED - MOVED UP)
// =============================================

/**
 * @brief Point cloud processing utilities
 *        ORIGINAL IMPLEMENTATION - PRESERVED
 */
class PointCloudProcessor {
public:
    explicit PointCloudProcessor(const LidarConfig& config);
    
    bool filterCarBodyAndROI(const std::vector<Point4D>& input_points, 
                            PointCloudPtr output_cloud,
                            rclcpp::Logger logger);
                            
    bool removeGroundPlane(PointCloudPtr cloud,
                          PointCloudPtr non_ground_cloud, 
                          rclcpp::Logger logger);

protected:
    bool isValidGroundPlane(const Eigen::Vector3f& normal, 
                          const std::optional<Eigen::Vector3f>& reference_normal) const;
    
    const LidarConfig& config_;
};

// =============================================
// ENHANCED GROUND REMOVAL COMPONENTS
// =============================================

/**
 * @brief Multi-stage ground removal pipeline combining angular sector fitting,
 *        zonal PCA, elevation grid filtering, and cone base restoration.
 *        Based on research from LineFit, Patchwork, and GroundGrid methods.
 */
class MultiStageGroundRemover {
public:
    explicit MultiStageGroundRemover(const LidarConfig& config);
    
    /**
     * @brief Main entry point for enhanced ground removal
     * @param points Input point cloud after ROI filtering
     * @param logger ROS logger for debugging
     * @return Non-ground points with cone bases preserved
     */
    std::vector<Point4D> removeGround(const std::vector<Point4D>& points, rclcpp::Logger logger);

private:
    // Preprocessing stage
    std::vector<Point4D> preprocessPoints(const std::vector<Point4D>& points);
    std::vector<Point4D> singleAxisGroundSmoothing(const std::vector<Point4D>& points);
    
    // Hybrid plane fitting stage
    std::vector<Point4D> fitGroundPlanesHybrid(const std::vector<Point4D>& points);
    std::vector<Point4D> fitAngularSectors(const std::vector<Point4D>& points);
    std::vector<Point4D> fitZonalPCA(const std::vector<Point4D>& points);
    
    // Elevation grid filtering stage  
    std::vector<Point4D> filterByElevationGrid(const std::vector<Point4D>& points);
    
    // Cone restoration stage
    std::vector<Point4D> restoreConeBases(const std::vector<Point4D>& non_ground_points,
                                         const std::vector<Point4D>& ground_points,
                                         const std::vector<Point4D>& all_points);
    
    // Utility methods
    struct PlaneModel {
        Eigen::Vector3d normal;
        double d;
        std::vector<Point4D> inliers;
    };
    
    PlaneModel fitPlaneToLowestPoints(const std::vector<Point4D>& points);
    double computeAdaptiveThreshold(const std::vector<Point4D>& points);
    
    const LidarConfig& config_;
};

/**
 * @brief Enhanced point cloud processor with multi-stage ground removal
 *        and automatic fallback to original RANSAC
 */
class EnhancedPointCloudProcessor : public PointCloudProcessor {
public:
    explicit EnhancedPointCloudProcessor(const LidarConfig& config);
    
    /**
     * @brief Enhanced ground removal using multi-stage pipeline with fallback
     * @param cloud Input point cloud
     * @param non_ground_cloud Output non-ground points
     * @param logger ROS logger
     * @return True if successful (either enhanced or fallback)
     */
    bool removeGroundPlaneEnhanced(PointCloudPtr cloud,
                                  PointCloudPtr non_ground_cloud,
                                  rclcpp::Logger logger);

private:
    MultiStageGroundRemover ground_remover_;
};

/**
 * @brief ML-based cone classifier using ONNX runtime
 *        ORIGINAL IMPLEMENTATION - PRESERVED
 */
class ConeClassifier {
public:
    ConeClassifier(Ort::Env& env, const LidarConfig& config);
    bool initialize(const std::string& model_path);
    std::optional<int> classify(const Cluster& cluster);

private:
    std::vector<float> createFeatureVector(const Cluster& cluster) const;
    
    Ort::Env& env_;
    const LidarConfig& config_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<int64_t> input_node_dims_;
};

/**
 * @brief Heuristic cone classifier using intensity profile analysis
 *        ORIGINAL IMPLEMENTATION - PRESERVED  
 */
class HeuristicClassifier {
public:
    explicit HeuristicClassifier(const LidarConfig& config);
    std::optional<int> classify(const Cluster& cluster);

private:
    bool classifyCone(const std::vector<double> &y_vals, const std::vector<double> &x_vals);
    std::vector<double> movingAverage(const std::vector<double> &data, int kernel);
    
    const LidarConfig& config_;
};

/**
 * @brief Cluster processing and cone detection
 *        ORIGINAL IMPLEMENTATION - PRESERVED
 */
class ClusterProcessor {
public:
    explicit ClusterProcessor(const LidarConfig& config);
    
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
    Point3D calculateConePosition(const Cluster& cluster);
    std::vector<Point3D> calculateClusterCenters(const std::vector<Cluster>& clusters) const;

private:
    double getMedian(const Cluster& points, size_t idx) const;
    bool isOrangeConeCandidate(const Cluster& cluster) const;
    
    const LidarConfig& config_;
};

/**
 * @brief Point cloud data extraction from ROS messages
 *        ORIGINAL IMPLEMENTATION - PRESERVED
 */
class PointCloudExtractor {
public:
    static std::vector<Point4D> fromPointCloud(const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg);
    static std::vector<Point4D> fromPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg);

private:
    static bool validatePointCloud(const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg);
    static bool validatePointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg);
};

/**
 * @brief Main LiDAR processing node
 *        ENHANCED WITH MULTI-STAGE GROUND REMOVAL AND FALLBACK SYSTEM
 */
class ProcessLidar : public rclcpp::Node {
public:
    ProcessLidar();
    ~ProcessLidar();

private:
    // Configuration
    LidarConfig lidar_config_;
    void loadLidarConfig();
    void initializeComponents();
    void loadONNXModel();
    bool validateComponents() const;
    
    // Enhanced ground removal flag - automatically falls back to RANSAC if needed
    bool use_enhanced_ground_removal_ = true;
    
    // Processing pipeline
    void processPointCloudData(std::vector<Point4D>& points, const std_msgs::msg::Header& header);
    bool executeProcessingPipeline(const std::vector<Point4D>& points);
    
    // Enhanced pipeline with multi-stage ground removal
    bool executeEnhancedProcessingPipeline(const std::vector<Point4D>& points);
    
    // Original pipeline as fallback (proven RANSAC implementation)
    bool executeOriginalProcessingPipeline(const std::vector<Point4D>& points);
    
    // Publication methods
    void publishDetectedCones(const std::vector<Point3D>& positions, const std::vector<int>& colors);
    void publishConeClusterPoints(const std::vector<Cluster>& cone_clusters);
    void publishClusterCenters(const std::vector<Cluster>& filtered_clusters);
    void publishProcessingMetrics(const std::chrono::milliseconds& duration);
    
    // Callbacks
    void lidarRawCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg);
    void lidarRawCallback2(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    
    // Components - ORIGINAL PRESERVED + ENHANCED
    Ort::Env env_;
    std::unique_ptr<ConeClassifier> ml_classifier_;
    std::unique_ptr<HeuristicClassifier> heuristic_classifier_;
    std::unique_ptr<PointCloudProcessor> point_cloud_processor_;
    std::unique_ptr<EnhancedPointCloudProcessor> enhanced_point_cloud_processor_; // NEW: Enhanced ground removal
    std::unique_ptr<ClusterProcessor> cluster_processor_;
    
    // Subscribers and publishers - ORIGINAL PRESERVED
    rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr lidar_raw_input_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_raw_input_sub2_;
    rclcpp::Publisher<dv_msgs::msg::IndexedTrack>::SharedPtr detected_cones_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr filtered_points_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr cluster_centers_pub_;
    
    // Statistics - ORIGINAL PRESERVED
    std::atomic<long long> total_processed_frames_{0};
    std::atomic<long long> failed_processing_attempts_{0};
};

} // namespace perception_winter

#endif // PROCESS_LIDAR_HPP_