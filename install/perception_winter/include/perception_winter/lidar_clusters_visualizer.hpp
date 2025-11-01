#ifndef LIDAR_CLUSTERS_VISUALIZER_HPP_
#define LIDAR_CLUSTERS_VISUALIZER_HPP_

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <vector>
#include <memory>

namespace perception_winter {

/**
 * @brief Visualizes LiDAR clusters as markers in RViz
 * @details Converts cluster center positions to cylinder markers for visualization
 */
class LidarClustersVisualizer : public rclcpp::Node {
public:
    LidarClustersVisualizer();
    ~LidarClustersVisualizer() = default;

    // Non-copyable, non-movable
    LidarClustersVisualizer(const LidarClustersVisualizer&) = delete;
    LidarClustersVisualizer& operator=(const LidarClustersVisualizer&) = delete;
    LidarClustersVisualizer(LidarClustersVisualizer&&) = delete;
    LidarClustersVisualizer& operator=(LidarClustersVisualizer&&) = delete;

private:
    void clustersCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr clusters_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
    
    std::vector<visualization_msgs::msg::Marker> previous_markers_;
    std::string frame_id_;
};

} // namespace perception_winter

#endif // LIDAR_CLUSTERS_VISUALIZER_HPP_