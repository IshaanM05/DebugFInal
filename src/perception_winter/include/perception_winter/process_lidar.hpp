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

// Constants used in the cpp file
const std::string LIDAR_RAW_TOPIC = "/carmaker/pointcloud";
const std::string LIDAR_RAW_TOPIC2 = "/carmaker/pointcloud2";

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
constexpr double ROI_Z_MAX = 0.50;

// Car body dimensions from working code
constexpr double CAR_FRONT_X = 1.5;
constexpr double CAR_SIDE_Y = 1.25;

// Cone parameters from working code
constexpr double CONE_BASE_RADIUS = 0.12;
constexpr double LIDAR_OFFSET = 1.532;
constexpr double CONE_HEIGHT = 0.1629;

class ProcessLidar : public rclcpp::Node {
public:
    ProcessLidar();
    ~ProcessLidar();

private:
    // Type aliases for clarity
    using Point4D = std::array<double, 4>; // x, y, z, intensity
    using Point3D = std::array<double, 3>; // x, y, z
    using Cluster = std::vector<Point4D>;

    // Callbacks
    void lidarRawCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg);
    void lidarRawCallback2(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // Main processing pipeline
    void processPointCloudData(std::vector<Point4D>& points, const std_msgs::msg::Header& header);

    // Helper functions for data extraction
    std::vector<Point4D> extractPointsFromPointCloud(const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg);
    std::vector<Point4D> extractPointsFromPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg);

    // Processing stages
    bool filterCarBodyAndROI(const std::vector<Point4D>& input_points, pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud);
    bool removeGroundPlane(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloud);
    std::vector<Cluster> clusterPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
    std::vector<Cluster> filterClustersBySize(const std::vector<Cluster>& clusters);
    void detectConesInClusters(const std::vector<Cluster>& clusters, std::vector<Point3D>& positions, std::vector<int>& colors);
    
    // Cone position calculation
    Point3D calculateConePosition(const Cluster& cluster);
    
    // Math and classification helpers
    double getMedian(const Cluster& points, size_t idx) const;
    bool classifyCone(const std::vector<double>& y_vals, const std::vector<double>& x_vals);
    std::vector<double> movingAverage(const std::vector<double>& data, int kernel);

    // Publishers
    void publishFilteredPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
    void publishLidarClusters(const std::vector<Point3D>& cluster_centers);
    void publishDetectedCones(const std::vector<Point3D>& positions, const std::vector<int>& colors);

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr lidar_raw_input_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_raw_input_sub2_;

    // Publisher objects
    rclcpp::Publisher<dv_msgs::msg::IndexedTrack>::SharedPtr detected_cones_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr filtered_points_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr lidar_clusters_pub_;

    // Reusable PCL clouds
    pcl::PointCloud<pcl::PointXYZI>::Ptr reusable_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr reusable_cloud_filtered_;

    // --- CHANGES REQUIRED FOR ACCURACY METRICS ---
    long true_positives_yellow_;
    long false_positives_yellow_;
    long true_positives_blue_;
    long false_positives_blue_;
};

#endif // PROCESS_LIDAR_HPP_