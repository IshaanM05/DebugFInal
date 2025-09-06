/**
 * @name process_lidar.hpp
 * @brief Header (Declaration / Interface) file for the node
 * @author 
 */

#ifndef PROCESS_LIDAR_HPP
#define PROCESS_LIDAR_HPP

#include "rclcpp/rclcpp.hpp"                         // Node class inheritance
#include "visualization_msgs/msg/marker_array.hpp"   // Marker array for visualization
#include "visualization_msgs/msg/marker.hpp"
#include <sensor_msgs/msg/point_cloud.hpp>           // Only use PointCloud2

// Note: <fstream> is no longer needed and has been removed.

/**
 * @brief Node class Interface
 */
class ProcessLidar : public rclcpp::Node {
private:
  // Constants
  const std::string namespace_ = "process_lidar";
  const std::string fixed_frame = "Fr1A"; // Can change this later
  double dbscan_epsilon = 0.3;
  int dbscan_minpoints = 5;

  // Relative to Lidar, in metres
  const double ground_z = -0.625212;
  const double rear_end_x = -1.532;
  const double lidar_z_threshhold = this->ground_z + 0.05;

  // --- CHANGE: Updated RANSAC threshold for better robustness ---
  const double ransac_threshold = 0.03;

  // Parameters for iterative RANSAC
  double min_z_normal_component;
  double max_slope_deviation_deg;

  // Topics
  const std::string lidar_raw_input_topic = "/carmaker/pointcloud"; // Lidar data input
  // const std::string lidar_raw_output_rviz_topic = this->namespace_+"/lidar/raw"; // Lidar raw data output topic
  const std::string classified_cones_output_rviz_topic = this->namespace_ + "/classified_cones"; // Final output
  const std::string reference_vehicle_rviz_topic = this->namespace_ + "/reference_vehicle";      // Reference vehicle

  // Publishers and Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr lidar_raw_input_sub; // switched to PointCloud
  // rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr lidar_raw_output_rviz_pub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr reference_vehicle_rviz_pub;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr classified_cones_output_rviz_pub;

  // --- ADDITION: Declarations for the new visual debuggers ---
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ground_points_pub;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr non_ground_points_pub;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr clustered_points_pub;

  // Private Functions
  /**
   * @brief Callback function for lidar's raw input subscription
   * @param msg Shared pointer object of the topic's msg
   * @return void
   */
  void lidar_raw_sub_callback(const sensor_msgs::msg::PointCloud::SharedPtr msg); // keep as PointCloud

  /**
   * @brief Publishes a marker array
   * @param type Type of marker
   * @param ns namespace
   * @param frame_id frame_id
   * @param position_colours a vector with shape (2,n,3), where 1st dimension 1st(0th index) row contains (x,y,z) and 2nd(1st index) row contains (r,g,b) of n data points
   * @param publisher publisher object
   * @param del_markers Bool, true if markers to be deleted after each publish, else false
   * @param scales dimensions of the marker
   * @return void
   */
  void publishMarkerArray(
    visualization_msgs::msg::Marker::_type_type type,
    std::string ns,
    std::string frame_id,
    std::vector<std::vector<std::vector<double>>> positions_colours,
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher,
    bool del_markers,
    std::vector<double> scales,
    const rclcpp::Time& stamp
  );

  
  bool classifyCone(const std::vector<double>& y_vals, const std::vector<double>& x_vals);

  std::vector<double> movingAverage(const std::vector<double>& data, int kernel);

public:
  /**
   * @brief constructor
   */
  ProcessLidar();

  /**
   * @brief destructor
   */
  ~ProcessLidar();
};

#endif // PROCESS_LIDAR_HPP
