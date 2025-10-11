#ifndef PROCESS_LIDAR_HPP_
#define PROCESS_LIDAR_HPP_

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "dv_msgs/msg/indexed_track.hpp"
#include "dv_msgs/msg/indexed_cone.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <array>
#include <optional>

#include <onnxruntime_cxx_api.h>

const std::string LIDAR_RAW_TOPIC = "/carmaker/pointcloud";
const std::string LIDAR_RAW_TOPIC2 = "points";

constexpr double RANSAC_THRESHOLD = 0.015;
constexpr double MIN_Z_NORMAL_COMPONENT = 0.80;
constexpr double MAX_SLOPE_DEVIATION_DEG = 40.0;
constexpr int MAX_GROUND_ITERATIONS = 8;
constexpr size_t MIN_POINTS_FOR_PLANE = 150;

constexpr double DBSCAN_EPSILON = 0.20;
constexpr int DBSCAN_MINPOINTS = 3;

constexpr double ROI_Y_MIN = -7.50;
constexpr double ROI_Y_MAX = 7.50;
constexpr double ROI_Z_MIN = -10.00;
constexpr double ROI_Z_MAX = 10.00;

constexpr double CAR_FRONT_X = 0.0;
constexpr double CAR_SIDE_Y = 0.0;

constexpr double CONE_BASE_RADIUS = 0.12;
constexpr double LIDAR_OFFSET = 0.0;
constexpr double CONE_HEIGHT = 0.1629;


class ProcessLidar : public rclcpp::Node {
public:
    ProcessLidar();
    ~ProcessLidar();

    using Point4D = std::array<double, 4>; 
    using Point3D = std::array<double, 3>;
    using Cluster = std::vector<Point4D>;

private:
    void lidarRawCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg);
    void lidarRawCallback2(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    void processPointCloudData(std::vector<Point4D>& points, const std_msgs::msg::Header& header);

    std::vector<Point4D> extractPointsFromPointCloud(const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg);
    std::vector<Point4D> extractPointsFromPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg);
    bool filterCarBodyAndROI(const std::vector<Point4D>& input_points, pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud);
    bool removeGroundPlane(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloud);
    std::vector<Cluster> clusterPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
    std::vector<Cluster> filterClustersBySize(const std::vector<Cluster>& clusters);
    void detectConesInClusters(const std::vector<Cluster>& clusters, std::vector<Point3D>& positions, std::vector<int>& colors);
    Point3D calculateConePosition(const Cluster& cluster);
    double getMedian(const Cluster& points, size_t idx) const;
    void publishDetectedCones(const std::vector<Point3D>& positions, const std::vector<int>& colors);

    bool classifyCone(const std::vector<double> &y_vals, const std::vector<double> &x_vals);
    std::vector<double> movingAverage(const std::vector<double> &data, int kernel);

    void loadOnnxModel();
    std::vector<float> createFeatureVector(const Cluster& cluster);
    std::optional<int> predictColor(const Cluster& cluster);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr lidar_raw_input_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_raw_input_sub2_;
    rclcpp::Publisher<dv_msgs::msg::IndexedTrack>::SharedPtr detected_cones_pub_;

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<int64_t> input_node_dims_;
};

#endif 