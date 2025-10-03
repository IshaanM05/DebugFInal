/**
 * @file process_lidar.hpp
 * @brief Header (Declaration / Interface) file for the LiDAR processing node
 * @author Siddhesh Phadke
 */

#ifndef PROCESS_LIDAR_HPP
#define PROCESS_LIDAR_HPP

#include "rclcpp/rclcpp.hpp"                         // Node class inheritance
#include "visualization_msgs/msg/marker_array.hpp"   // Marker array for visualization
#include "visualization_msgs/msg/marker.hpp"
#include <sensor_msgs/msg/point_cloud.hpp>           // PointCloud message type
#include "dv_msgs/msg/indexed_track.hpp"             // Track message for detected cones
#include "dv_msgs/msg/indexed_cone.hpp"              // Cone message definition
#include "std_msgs/msg/float32_multi_array.hpp"      // For filtered points and clusters

/**
 * @brief LiDAR Processing Node Class
 * 
 * This node processes raw LiDAR point cloud data to detect and classify cones
 * using RANSAC ground removal, DBSCAN clustering, and intensity-based classification.
 */
class ProcessLidar : public rclcpp::Node {
private:
  // Node configuration
  const std::string namespace_ = "process_lidar";
  const std::string fixed_frame = "Fr1A"; // Coordinate frame for all outputs

  // DBSCAN clustering parameters
  double dbscan_epsilon = 0.20;
  int dbscan_minpoints = 3;

  // LiDAR geometry parameters (relative to LiDAR sensor in meters)
  const double ground_z = -0.625212;
  const double rear_end_x = -1.532;
  const double lidar_z_threshold = this->ground_z + 0.05;

  // RANSAC ground removal parameters
  const double ransac_threshold = 0.015;
  double min_z_normal_component;
  double max_slope_deviation_deg;

  // ROS 2 topic names
  const std::string lidar_raw_input_topic = "/carmaker/pointcloud";
  const std::string classified_cones_output_rviz_topic = this->namespace_ + "/classified_cones";

  // ROS 2 Publishers and Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr lidar_raw_input_sub;
  
  // Main output publishers
  rclcpp::Publisher<dv_msgs::msg::IndexedTrack>::SharedPtr detected_cones_pub;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr filtered_points_pub;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr lidar_clusters_pub;

  // Visualization publishers (optional - kept for compatibility)
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr classified_cones_output_rviz_pub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr reference_vehicle_rviz_pub;

  // Private member functions
  /**
   * @brief Main callback for processing incoming LiDAR point cloud data
   * @param msg Shared pointer to the PointCloud message
   */
  void lidar_raw_sub_callback(const sensor_msgs::msg::PointCloud::SharedPtr msg);

  /**
   * @brief Calculate median value for a specific coordinate index in point cloud
   * @param points Vector of points (each point is vector of coordinates)
   * @param idx Coordinate index (0=x, 1=y, 2=z)
   * @return Median value of the specified coordinate
   */
  double getMedian(const std::vector<std::vector<double>> &points, int idx) const;

  /**
   * @brief Publish filtered points (non-ground) for visualization
   * @param cloud Point cloud containing filtered points after RANSAC
   */
  void publishFilteredPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

  /**
   * @brief Publish cluster centers for visualization
   * @param cluster_centers Vector of cluster center coordinates (x,y)
   */
  void publishLidarClusters(const std::vector<std::vector<double>>& cluster_centers);

  /**
   * @brief Classify cone as yellow or blue based on intensity profile
   * @param y_vals Intensity values (normalized)
   * @param x_vals Z-coordinate values
   * @return true if yellow cone, false if blue cone
   */
  bool classifyCone(const std::vector<double>& y_vals, const std::vector<double>& x_vals);

  /**
   * @brief Apply moving average filter to smooth intensity data
   * @param data Input data vector
   * @param kernel Kernel size for moving average
   * @return Smoothed data vector
   */
  std::vector<double> movingAverage(const std::vector<double>& data, int kernel);

public:
  /**
   * @brief Constructor - initializes node, parameters, publishers and subscribers
   */
  ProcessLidar();

  /**
   * @brief Destructor - cleans up resources
   */
  ~ProcessLidar();
};

#endif // PROCESS_LIDAR_HPP