/**
 * @file process_lidar.cpp
 * @brief Source (Definition / Implementation) file for the node
 * @author Siddhesh Phadke
 */

#include "perception_winter/process_lidar.hpp"
#include <string>
#include <algorithm>
#include <cmath>
#include <optional> // Required for iterative RANSAC
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h> // Required for iterative RANSAC
#include <pcl/filters/passthrough.h>     // Required for ROI filtering
#include <pcl/ModelCoefficients.h>       // Required for iterative RANSAC
#include <chrono> // For timing

ProcessLidar::ProcessLidar() : Node("process_lidar")
{
    // Shout out
    RCLCPP_INFO(this->get_logger(), "Process Lidar Node started");

    // Initializing
    this->lidar_raw_input_sub = this->create_subscription<sensor_msgs::msg::PointCloud>(
        this->lidar_raw_input_topic,
        10,
        std::bind(&ProcessLidar::lidar_raw_sub_callback, this, std::placeholders::_1)
    );

    // For final classified cones
    this->classified_cones_output_rviz_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->classified_cones_output_rviz_topic,
        10
    );
    this->detected_cones_pub = this->create_publisher<dv_msgs::msg::IndexedTrack>(
    "/detected_cones",
    10
    );


    // // Publishers for debugging visualizations
    // this->ground_points_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    //     this->namespace_ + "/ground_points", 10);
    // this->non_ground_points_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    //     this->namespace_ + "/non_ground_points", 10);
    // this->clustered_points_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    //     this->namespace_ + "/clustered_points", 10);

    // Initialize new RANSAC parameters
    this->min_z_normal_component = 0.80;
    this->max_slope_deviation_deg = 40.0;

    // --- DEBUG LOGGER ---
    RCLCPP_INFO(this->get_logger(), "[DEBUG] Constructor finished.");
}

// double ProcessLidar::getMedian(const std::vector<std::vector<double>> &points, int idx) const {
//     std::vector<double> vals;
//     vals.reserve(points.size());
//     for (const auto &pt : points) vals.push_back(pt[idx]);
//     std::nth_element(vals.begin(), vals.begin() + vals.size()/2, vals.end());
//     return vals[vals.size()/2];
// }

double ProcessLidar::getMedian(const std::vector<std::vector<double>> &points, int idx) const {
    if (points.empty()) return 0.0;

    std::vector<size_t> indices(points.size());
    for (size_t i = 0; i < points.size(); ++i) indices[i] = i;

    // Use nth_element on indices comparing the actual values
    size_t mid = indices.size() / 2;
    std::nth_element(indices.begin(), indices.begin() + mid, indices.end(),
                     [&points, idx](size_t a, size_t b) { return points[a][idx] < points[b][idx]; });

    double median = points[indices[mid]][idx];

    // Optional: for even-sized clusters, average middle two values
    if (indices.size() % 2 == 0) {
        size_t max_lower_idx = std::max_element(indices.begin(), indices.begin() + mid,
                                                [&points, idx](size_t a, size_t b) { return points[a][idx] < points[b][idx]; }) - indices.begin();
        median = 0.5 * (median + points[indices[max_lower_idx]][idx]);
    }

    return median;
}

ProcessLidar::~ProcessLidar()
{
    // --- DEBUG LOGGER ---
    RCLCPP_INFO(this->get_logger(), "[DEBUG] Destructor called. Shutting down.");
}

void ProcessLidar::lidar_raw_sub_callback(const sensor_msgs::msg::PointCloud::SharedPtr msg)
{
    try {
        if (!msg) {
            RCLCPP_WARN(this->get_logger(), "Received null PointCloud message, skipping.");
            return;
        }
        if (msg->points.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty PointCloud, skipping.");
            return;
        }
        if (msg->channels.empty() || msg->channels[0].values.size() != msg->points.size()) {
            RCLCPP_WARN(this->get_logger(), "PointCloud message missing intensity channel or mismatch in size, skipping.");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "[DEBUG] ---- lidar_raw_sub_callback entered ----");
        auto pipeline_start = std::chrono::steady_clock::now();
        // RCLCPP_INFO(this->get_logger(), "[DEBUG] Received point cloud with %zu points.", msg->points.size());

        std::vector<std::vector<double>> positions, colors;
        std::vector<double> intensities;
        positions.reserve(msg->points.size());
        intensities.reserve(msg->points.size());

        // --- New Code with Exclusion Box ---
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        cloud->points.reserve(msg->points.size());

        // Define the dimensions of the car's body relative to the LiDAR sensor
        // Tune these values to match your car's geometry
        const double CAR_FRONT_X = 1.5;   // Ignore points closer than 1.5m in front
        const double CAR_SIDE_Y  = 1.25;  // Ignore points within 0.9m to the left/right
        //std::cout << "Car y = " << CAR_SIDE_Y << std::endl;

        for (size_t i = 0; i < msg->points.size(); ++i) {
            const auto &pt = msg->points[i];
            bool is_valid_point = (pt.x > CAR_FRONT_X) || (std::abs(pt.y) > CAR_SIDE_Y);

            if (pt.x > 0 && is_valid_point) {
                pcl::PointXYZI p;
                p.x = pt.x;
                p.y = pt.y;
                p.z = pt.z;
                p.intensity = msg->channels[0].values[i];
                cloud->points.push_back(p);
            }
        }

        auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        pcl::PassThrough<pcl::PointXYZI> pass;

        pass.setInputCloud(cloud);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-3.50, 3.50);
        pass.filter(*cloud_filtered);

        pass.setInputCloud(cloud_filtered);
        // pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-0.63, 0.50);
        pass.filter(*cloud_filtered);

        auto non_ground_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        auto ground_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        auto remaining_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>(*cloud_filtered);

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZI> seg;
        pcl::ExtractIndices<pcl::PointXYZI> extract;

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(this->ransac_threshold);

        int iterations = 0;
        const int max_iterations = 8;
        const size_t min_points_for_plane = 150;
        std::optional<Eigen::Vector3f> reference_normal;

        while (remaining_cloud->points.size() > min_points_for_plane && iterations < max_iterations)
    {
        // Dynamically clamp max iterations based on remaining points
        int dynamic_max_iter = std::min(static_cast<int>(remaining_cloud->points.size() / 200), max_iterations);
        if (iterations >= dynamic_max_iter) break;

        seg.setInputCloud(remaining_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) break;

        Eigen::Vector3f current_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        if (current_normal.z() < 0) current_normal = -current_normal;

        if (current_normal.z() < this->min_z_normal_component) break;

        if (!reference_normal.has_value()) reference_normal = current_normal;
        else {
            double dot_product = current_normal.dot(reference_normal.value());
            double angle_rad = std::acos(std::clamp(dot_product, -1.0, 1.0));
            double angle_deg = angle_rad * (180.0 / M_PI);
            if (angle_deg > this->max_slope_deviation_deg) break;
        }

        auto current_ground_plane = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*current_ground_plane);
        *ground_cloud += *current_ground_plane;

        extract.setNegative(true);
        auto next_remaining_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        extract.filter(*next_remaining_cloud);
        remaining_cloud = next_remaining_cloud;
        iterations++;
    }

        *non_ground_cloud = *remaining_cloud;
        // --- DEBUG LOGGER ---
    // RCLCPP_INFO(this->get_logger(), "[DEBUG] RANSAC complete. Ground points: %zu, Non-ground points: %zu", ground_cloud->size(), non_ground_cloud->size());

    // Debugger 1: Visualize ground points
    // std::vector<std::vector<double>> ground_positions, ground_colors;
    // for (const auto& point : ground_cloud->points) {
    //     ground_positions.push_back({point.x, point.y, point.z});
    //     ground_colors.push_back({0.0, 1.0, 0.0});
    // }
    // this->publishMarkerArray(visualization_msgs::msg::Marker::SPHERE, this->namespace_ + "_ground", 
    //     this->fixed_frame, {ground_positions, ground_colors}, this->ground_points_pub,
    //     true, {0.05, 0.05, 0.05}, msg->header.stamp);

        for (const auto &point : non_ground_cloud->points) {
            positions.push_back({point.x, point.y, point.z});
            intensities.push_back(point.intensity);
        }
        
        // Debugger 2: Visualize non-ground points
        // std::vector<std::vector<double>> non_ground_colors;
        // for (size_t i = 0; i < positions.size(); ++i) {
        //     non_ground_colors.push_back({1.0, 1.0, 1.0});
        // }
        // this->publishMarkerArray(visualization_msgs::msg::Marker::SPHERE, this->namespace_ + "_non_ground",
        //     this->fixed_frame, {positions, non_ground_colors}, this->non_ground_points_pub,
        //     true, {0.05, 0.05, 0.05}, msg->header.stamp);

        // if (positions.empty()) {
        //     // RCLCPP_INFO(this->get_logger(), "[DEBUG] No non-ground points to cluster. Exiting callback.");
        //     this->publishMarkerArray(visualization_msgs::msg::Marker::CYLINDER, this->namespace_,
        //         this->fixed_frame, {{}, {}}, this->classified_cones_output_rviz_pub,
        //         true, {1, 1, 0.5}, msg->header.stamp);
        //     this->publishMarkerArray(visualization_msgs::msg::Marker::SPHERE, this->namespace_ + "_clustered",
        //         this->fixed_frame, {{}, {}}, this->clustered_points_pub, true, {0.05, 0.05, 0.05}, msg->header.stamp);
        //     return;
        // }


        auto o3d_pcd = std::make_shared<open3d::geometry::PointCloud>();
        o3d_pcd->points_.reserve(positions.size());
        for (const auto &point : positions) {
            o3d_pcd->points_.emplace_back(point[0], point[1], point[2]);
        }

        std::vector<int> labels = o3d_pcd->ClusterDBSCAN(this->dbscan_epsilon, this->dbscan_minpoints);
        int num_labels = labels.empty() ? 0 : (*std::max_element(labels.begin(), labels.end()) + 1);

        std::vector<std::vector<std::vector<double>>> classified_points(num_labels);
        for (size_t index = 0; index < positions.size(); index++) {
            int label = labels[index];
            if (label == -1) continue;
            classified_points[label].push_back({positions[index][0], positions[index][1], positions[index][2], intensities[index]});
        }

    // ------------------- ADD PRUNING HERE -------------------
    for (auto it = classified_points.begin(); it != classified_points.end();) {
        auto &cluster = *it;

        // Compute bounding box
        double min_x = cluster[0][0], max_x = cluster[0][0];
        double min_y = cluster[0][1], max_y = cluster[0][1];
        double min_z = cluster[0][2], max_z = cluster[0][2];

        for (auto &pt : cluster) {
            min_x = std::min(min_x, pt[0]); max_x = std::max(max_x, pt[0]);
            min_y = std::min(min_y, pt[1]); max_y = std::max(max_y, pt[1]);
            min_z = std::min(min_z, pt[2]); max_z = std::max(max_z, pt[2]);
        }

        double height = max_z - min_z;
        double width  = std::max(max_x - min_x, max_y - min_y);

        // Reject clusters outside expected cone size
        if (height < 0.15 || height > 0.4 || width > 0.45) {
            it = classified_points.erase(it);
            continue;
        }

        ++it;
    }
    // ------------------- PRUNING DONE -------------------

        std::vector<std::vector<double>> clustered_positions, clustered_colors;
        for (const auto &cluster : classified_points) {
            for (const auto &point : cluster) {
                clustered_positions.push_back({point[0] + 2.921, point[1], point[2]});
                clustered_colors.push_back({1.0, 0.0, 1.0});
            }
        }

        for (auto &cone_class : classified_points) {
            std::sort(cone_class.begin(), cone_class.end(),
                      [](const std::vector<double> &v1, const std::vector<double> &v2) -> bool {
                          return v1[2] > v2[2];
                      });
        }

        colors.clear();
        positions.clear();

        for (auto &class_ : classified_points) {
            int class_size = class_.size();
            if (class_size < 4) continue;

            std::vector<double> intensity_vals;
            std::vector<double> z_vals;
            intensity_vals.reserve(class_size);
            z_vals.reserve(class_size);

            const double CONE_BASE_RADIUS = 0.12;

            // auto min_x_it = std::min_element(class_.begin(), class_.end(),
            //                                  [](const std::vector<double> &a, const std::vector<double> &b) {
            //                                      return a[0] < b[0];
            //                                  });

            // double cone_x = (*min_x_it)[0] + CONE_BASE_RADIUS + 1.532;
            // double cone_y = (*min_x_it)[1];
            // double cone_z = 0.1629;

            // Compute min X and min/max Y
            double min_x = class_[0][0];
            double min_y = class_[0][1], max_y = class_[0][1];
            for (auto &pt : class_) {
                min_x = std::min(min_x, pt[0]);
                min_y = std::min(min_y, pt[1]);
                max_y = std::max(max_y, pt[1]);
            }

            // Compute median
            double median_x = getMedian(class_, 0);
            double median_y = getMedian(class_, 1);

            // Weighted blend for X: median_x and min_x + CONE_BASE_RADIUS
            const double w_median = 0.7;
            const double w_min_x  = 0.3;
            const double w_min_y  = 0.3;

            double cone_x = w_median * median_x + w_min_x * (min_x + CONE_BASE_RADIUS) + 1.532; // LiDAR offset
            double cone_y = w_median * median_y + w_min_y * (min_y + CONE_BASE_RADIUS);   
            double cone_z = 0.1629;     // cones assumed on ground



            for (auto &pt : class_) {
                intensity_vals.push_back(pt.at(3));
                z_vals.push_back(pt.at(2));
            }

            int kernel = std::max(3, static_cast<int>(0.1 * intensity_vals.size()));
            if (kernel % 2 == 0) kernel += 1;

            std::vector<double> averaged_intensity_vals = this->movingAverage(intensity_vals, kernel);
            positions.push_back({cone_x, cone_y, cone_z});

            if (this->classifyCone(averaged_intensity_vals, z_vals)) {
                colors.push_back({1.0, 1.0, 0.0});
            } else {
                colors.push_back({0.0, 0.0, 1.0});
            }
        }

        this->publishMarkerArray(
            visualization_msgs::msg::Marker::CYLINDER,
            this->namespace_,
            this->fixed_frame,
            {positions, colors},
            this->classified_cones_output_rviz_pub,
            true,
            {0.1, 0.1, 0.5},
            msg->header.stamp);

        // ---------------- Publish IndexedTrack ----------------
        if (this->detected_cones_pub) {
            dv_msgs::msg::IndexedTrack track_msg;
            track_msg.track.clear();
            for (size_t i = 0; i < positions.size(); ++i) {
                dv_msgs::msg::IndexedCone cone_msg;
                cone_msg.location.x = positions[i][0];
                cone_msg.location.y = positions[i][1];
                cone_msg.location.z = positions[i][2];

                if (colors[i][0] == 1.0 && colors[i][1] == 1.0)
                    cone_msg.color = dv_msgs::msg::IndexedCone::YELLOW;
                else if (colors[i][0] == 0.0 && colors[i][1] == 0.0 && colors[i][2] == 1.0)
                    cone_msg.color = dv_msgs::msg::IndexedCone::BLUE;
                else
                    cone_msg.color = dv_msgs::msg::IndexedCone::UNKNOWN;

                cone_msg.index = i;
                track_msg.track.push_back(cone_msg);
            }
            this->detected_cones_pub->publish(track_msg);
        }

        RCLCPP_INFO(this->get_logger(), "[DEBUG] ---- lidar_raw_sub_callback finished ----");
        auto pipeline_end = std::chrono::steady_clock::now();
        RCLCPP_INFO(this->get_logger(), "[TIMER] Total pipeline took %ld ms",
                    std::chrono::duration_cast<std::chrono::milliseconds>(pipeline_end - pipeline_start).count());

    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in lidar_raw_sub_callback: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown exception in lidar_raw_sub_callback");
    }
}

void ProcessLidar::publishMarkerArray(
    visualization_msgs::msg::Marker::_type_type type,
    std::string ns,
    std::string frame_id,
    std::vector<std::vector<std::vector<double>>> positions_colours,
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher,
    bool del_markers,
    std::vector<double> scales,
    const rclcpp::Time &stamp)
{
    if (!publisher) { return; }

    visualization_msgs::msg::MarkerArray marker_array;

    if (del_markers) {
        visualization_msgs::msg::Marker del_marker;
        del_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(del_marker);
    }

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.ns = ns;
    marker.type = type;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = scales.at(0);
    marker.scale.y = scales.at(1);
    marker.scale.z = scales.at(2);

    for (size_t i = 0; i < positions_colours.at(0).size(); i++) {
        double x = positions_colours.at(0).at(i).at(0);
        double y = positions_colours.at(0).at(i).at(1);
        double z = positions_colours.at(0).at(i).at(2);

        if ((x < 4.0)) {
            continue; // skip this marker
        }

        marker.id = i;
        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = z;
        marker.color.a = 1.0;
        marker.color.r = positions_colours.at(1).at(i).at(0);
        marker.color.g = positions_colours.at(1).at(i).at(1);
        marker.color.b = positions_colours.at(1).at(i).at(2);
        // marker.color.r = 0.0;  // red
        // marker.color.g = 1.0;  // green
        // marker.color.b = 0.0;  // blue

        marker_array.markers.push_back(marker);

        // std::cout << "Published marker at ("
        //           << marker.pose.position.x << ", "
        //           << marker.pose.position.y << ", "
        //           << marker.pose.position.z << ") with color ("
        //           << marker.color.r << ", "
        //           << marker.color.g << ", "
        //           << marker.color.b << ")\n";
    }

    publisher->publish(marker_array);
}

bool ProcessLidar::classifyCone(const std::vector<double> &y_vals, const std::vector<double> &x_vals)
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

std::vector<double> ProcessLidar::movingAverage(const std::vector<double> &data, int kernel)
{
    int n = data.size();
    std::vector<double> result(n, 0.0);
    if (kernel < 1) return data;

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
