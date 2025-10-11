// #include "perception_winter/process_lidar.hpp"
// #include <algorithm>
// #include <cmath>
// #include <numeric>
// #include <optional>
// #include <open3d/Open3D.h>
// #include <pcl/segmentation/sac_segmentation.h>
// #include <pcl/filters/extract_indices.h>
// #include <pcl/filters/passthrough.h>
// #include <pcl/ModelCoefficients.h>
// #include <chrono>
// #include <Eigen/Dense>

// constexpr int NUM_BINS = 10;
// constexpr int NUM_CHANNELS = 2; 
// constexpr float CONFIDENCE_THRESHOLD = 0.9f;

// ProcessLidar::ProcessLidar() : Node("process_lidar"), env_(ORT_LOGGING_LEVEL_WARNING, "process_lidar_onnx")
// {
//     lidar_raw_input_sub_ = create_subscription<sensor_msgs::msg::PointCloud>(
//         LIDAR_RAW_TOPIC, 10,
//         [this](const sensor_msgs::msg::PointCloud::SharedPtr msg) {
//             lidarRawCallback(msg);
//         });

//     lidar_raw_input_sub2_ = create_subscription<sensor_msgs::msg::PointCloud2>(
//         LIDAR_RAW_TOPIC2, 10,
//         [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
//             lidarRawCallback2(msg);
//         });

//     detected_cones_pub_ = create_publisher<dv_msgs::msg::IndexedTrack>("/perception/cones", 10);

//     loadOnnxModel();

//     RCLCPP_INFO(get_logger(), "Optimized LiDAR Node with Hybrid ML/Heuristic classification started");
// }

// ProcessLidar::~ProcessLidar()
// {
//     RCLCPP_INFO(get_logger(), "LiDAR Node shutdown");
// }

// void ProcessLidar::loadOnnxModel() {
//     this->declare_parameter<std::string>("onnx_model_path", "src/perception_winter/cone_model.onnx");
//     std::string model_path = this->get_parameter("onnx_model_path").as_string();

//     try {
//         Ort::SessionOptions session_options;
//         session_options.SetIntraOpNumThreads(1);
//         session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

//         Ort::AllocatorWithDefaultOptions allocator;

//         auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
//         input_node_names_.push_back(input_name_ptr.get());

//         auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
//         output_node_names_.push_back(output_name_ptr.get());

//         Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
//         auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
//         input_node_dims_ = input_tensor_info.GetShape();

//         RCLCPP_INFO(this->get_logger(), "Successfully loaded ONNX model from: %s", model_path.c_str());
//         RCLCPP_INFO(this->get_logger(), "Model Input Name: %s", input_node_names_[0].c_str());
//         RCLCPP_INFO(this->get_logger(), "Model expects input shape: [%ld, %ld, %ld]",
//             input_node_dims_[0], input_node_dims_[1], input_node_dims_[2]);

//     } catch (const Ort::Exception& e) {
//         RCLCPP_FATAL(this->get_logger(), "Failed to load ONNX model: %s", e.what());
//         rclcpp::shutdown();
//     }
// }

// std::vector<ProcessLidar::Point4D> ProcessLidar::extractPointsFromPointCloud2(
//     const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
// {
//     std::vector<Point4D> points;
//     points.reserve(cloud_msg->width * cloud_msg->height);

//     sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
//     sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
//     sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");
    
//     bool has_intensity = false;
//     for (const auto& field : cloud_msg->fields) {
//         if (field.name == "intensity") {
//             has_intensity = true;
//             break;
//         }
//     }
    
//     std::optional<sensor_msgs::PointCloud2ConstIterator<float>> iter_intensity_opt;
//     if (has_intensity) {
//         iter_intensity_opt.emplace(*cloud_msg, "intensity");
//     }

//     for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
//         double intensity = 0.0;
//         if (has_intensity && iter_intensity_opt.has_value()) {
//             intensity = *(*iter_intensity_opt);
//             ++(*iter_intensity_opt);
//         }
//         points.push_back({*iter_x, *iter_y, *iter_z, intensity});
//     }
//     return points;
// }

// std::vector<ProcessLidar::Point4D> ProcessLidar::extractPointsFromPointCloud(
//     const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg)
// {
//     std::vector<Point4D> points;
//     points.reserve(cloud_msg->points.size());

//     bool has_intensity = !cloud_msg->channels.empty() && 
//                         cloud_msg->channels[0].values.size() == cloud_msg->points.size();

//     for (size_t i = 0; i < cloud_msg->points.size(); ++i) {
//         const auto& pt = cloud_msg->points[i];
//         points.push_back({pt.x, pt.y, pt.z, has_intensity ? cloud_msg->channels[0].values[i] : 0.0});
//     }
//     return points;
// }

// void ProcessLidar::processPointCloudData(std::vector<Point4D>& points, const std_msgs::msg::Header& header)
// {
//     (void)header; // Mark as unused
//     if (points.empty()) return;

//     auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
//     auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

//     if (!filterCarBodyAndROI(points, cloud)) return;
//     if (!removeGroundPlane(cloud, cloud_filtered)) return;

//     auto clusters = clusterPoints(cloud_filtered);
//     if (clusters.empty()) return;

//     auto filtered_clusters = filterClustersBySize(clusters);
//     if (filtered_clusters.empty()) return;

//     std::vector<Point3D> cone_positions;
//     std::vector<int> cone_colors;
//     detectConesInClusters(filtered_clusters, cone_positions, cone_colors);

//     publishDetectedCones(cone_positions, cone_colors);
// }

// bool ProcessLidar::filterCarBodyAndROI(const std::vector<Point4D>& input_points, 
//                                       pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud)
// {
//     output_cloud->reserve(input_points.size());
    
//     for (const auto& point : input_points) {
//         bool is_valid_point = (point[0] > CAR_FRONT_X) || (std::abs(point[1]) > CAR_SIDE_Y);
//         if (point[0] > 0 && is_valid_point) {
//             pcl::PointXYZI pcl_point;
//             pcl_point.x = point[0]; pcl_point.y = point[1];
//             pcl_point.z = point[2]; pcl_point.intensity = point[3];
//             output_cloud->push_back(pcl_point);
//         }
//     }

//     auto cloud_filtered_pass = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
//     pcl::PassThrough<pcl::PointXYZI> pass;
//     pass.setInputCloud(output_cloud);
//     pass.setFilterFieldName("y");
//     pass.setFilterLimits(ROI_Y_MIN, ROI_Y_MAX);
//     pass.filter(*cloud_filtered_pass);

//     pass.setInputCloud(cloud_filtered_pass);
//     pass.setFilterFieldName("z");
//     pass.setFilterLimits(ROI_Z_MIN, ROI_Z_MAX);
//     pass.filter(*output_cloud);
    
//     return !output_cloud->empty();
// }

// bool ProcessLidar::removeGroundPlane(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
//                                     pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloud)
// {
//     if (cloud->empty()) return false;

//     pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
//     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//     pcl::SACSegmentation<pcl::PointXYZI> seg;
//     pcl::ExtractIndices<pcl::PointXYZI> extract;

//     seg.setOptimizeCoefficients(true);
//     seg.setModelType(pcl::SACMODEL_PLANE);
//     seg.setMethodType(pcl::SAC_RANSAC);
//     seg.setDistanceThreshold(RANSAC_THRESHOLD);

//     auto remaining_cloud = cloud;
//     int iterations = 0;
//     std::optional<Eigen::Vector3f> reference_normal;

//     while (remaining_cloud->size() > MIN_POINTS_FOR_PLANE && iterations < MAX_GROUND_ITERATIONS) {
//         seg.setInputCloud(remaining_cloud);
//         seg.segment(*inliers, *coefficients);

//         if (inliers->indices.empty()) {
//             break;
//         }

//         Eigen::Vector3f current_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
//         if (current_normal.z() < 0) {
//             current_normal = -current_normal;
//         }
//         if (current_normal.z() < MIN_Z_NORMAL_COMPONENT) {
//             break;
//         }

//         if (!reference_normal.has_value()) {
//             reference_normal = current_normal;
//         } else {
//             double angle_rad = std::acos(std::clamp(static_cast<float>(current_normal.dot(reference_normal.value())), -1.0f, 1.0f));
//             double angle_deg = angle_rad * (180.0 / M_PI);
//             if (angle_deg > MAX_SLOPE_DEVIATION_DEG) {
//                 break;
//             }
//         }
        
//         extract.setInputCloud(remaining_cloud);
//         extract.setIndices(inliers);
//         extract.setNegative(true);
//         auto next_remaining = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
//         extract.filter(*next_remaining);
//         remaining_cloud = next_remaining;
        
//         iterations++;
//     }

//     *non_ground_cloud = *remaining_cloud;
//     return !non_ground_cloud->empty();
// }

// std::vector<ProcessLidar::Cluster> ProcessLidar::clusterPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
// {
//     if (cloud->empty()) return {};

//     auto o3d_pcd = std::make_shared<open3d::geometry::PointCloud>();
//     o3d_pcd->points_.reserve(cloud->size());
//     for (const auto& point : cloud->points) {
//         o3d_pcd->points_.emplace_back(point.x, point.y, point.z);
//     }
//     auto labels = o3d_pcd->ClusterDBSCAN(DBSCAN_EPSILON, DBSCAN_MINPOINTS, false);
    
//     int max_label = labels.empty() ? -1 : *std::max_element(labels.begin(), labels.end());
//     if (max_label < 0) return {};
    
//     std::vector<Cluster> clusters(max_label + 1);
//     for (size_t i = 0; i < labels.size(); ++i) {
//         if (labels[i] >= 0) {
//             const auto& point = cloud->points[i];
//             clusters[labels[i]].push_back({point.x, point.y, point.z, point.intensity});
//         }
//     }
//     clusters.erase(std::remove_if(clusters.begin(), clusters.end(),
//         [](const Cluster& c) { return c.empty(); }), clusters.end());

//     return clusters;
// }

// std::vector<ProcessLidar::Cluster> ProcessLidar::filterClustersBySize(const std::vector<Cluster>& clusters)
// {
//     std::vector<Cluster> valid_clusters;
//     valid_clusters.reserve(clusters.size());
//     for (const auto& cluster : clusters) {
//         if (cluster.size() < 4) continue;
//         double min_z = cluster[0][2], max_z = cluster[0][2];
//         for (const auto& point : cluster) {
//             min_z = std::min(min_z, point[2]);
//             max_z = std::max(max_z, point[2]);
//         }
//         double height = max_z - min_z;
//         if (height >= 0.15 && height <= 0.4) {
//             valid_clusters.push_back(cluster);
//         }
//     }
//     return valid_clusters;
// }

// std::vector<float> ProcessLidar::createFeatureVector(const Cluster& cluster) {
//     std::vector<float> feature_vector(NUM_BINS * NUM_CHANNELS, 0.0f);
//     if (cluster.empty()) return feature_vector;

//     // --- 1. Find min/max for normalization ---
//     double min_z = cluster[0][2], max_z = cluster[0][2];
//     double min_intensity = cluster[0][3], max_intensity = cluster[0][3];
//     for (const auto& point : cluster) {
//         min_z = std::min(min_z, point[2]); max_z = std::max(max_z, point[2]);
//         min_intensity = std::min(min_intensity, point[3]); max_intensity = std::max(max_intensity, point[3]);
//     }

//     double z_range = max_z - min_z;
//     double intensity_range = max_intensity - min_intensity;
//     double bin_width = (z_range > 1e-6) ? z_range / NUM_BINS : 0.0;

//     std::vector<double> sum_intensity(NUM_BINS, 0.0);
//     std::vector<int> point_count(NUM_BINS, 0);

//     for (const auto& point : cluster) {
//         double norm_intensity = (intensity_range > 1e-6) ? (point[3] - min_intensity) / intensity_range : 0.0;
//         int bin_index = (bin_width > 0) ? static_cast<int>((point[2] - min_z) / bin_width) : 0;
//         bin_index = std::min(bin_index, NUM_BINS - 1);
//         sum_intensity[bin_index] += norm_intensity;
//         point_count[bin_index]++;
//     }

//     auto min_max_it = std::minmax_element(point_count.begin(), point_count.end());
//     float min_c = static_cast<float>(*min_max_it.first);
//     float max_c = static_cast<float>(*min_max_it.second);
//     float count_range = max_c - min_c;

//     for (int i = 0; i < NUM_BINS; ++i) {
//         float normalized_count = 0.0f;
//         if (count_range > 0) {
//             normalized_count = (static_cast<float>(point_count[i]) - min_c) / count_range;
//         } else {
//             normalized_count = (min_c > 0) ? 1.0f : 0.0f;
//         }
//         feature_vector[i * NUM_CHANNELS + 0] = normalized_count;
        
//         feature_vector[i * NUM_CHANNELS + 1] = (point_count[i] > 0) ? static_cast<float>(sum_intensity[i] / point_count[i]) : 0.0f;
//     }

//     return feature_vector;
// }

// std::optional<int> ProcessLidar::predictColor(const Cluster& cluster) {
//     std::vector<float> feature_vector = createFeatureVector(cluster);
//     Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

//     std::vector<int64_t> concrete_shape = input_node_dims_;
//     if (!concrete_shape.empty() && concrete_shape[0] == -1) {
//         concrete_shape[0] = 1; 
//     }

//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, feature_vector.data(), feature_vector.size(),
//         concrete_shape.data(), concrete_shape.size());

//     std::vector<const char*> input_names_char;
//     input_names_char.reserve(input_node_names_.size());
//     for (const auto& s : input_node_names_) {
//         input_names_char.push_back(s.c_str());
//     }

//     std::vector<const char*> output_names_char;
//     output_names_char.reserve(output_node_names_.size());
//     for (const auto& s : output_node_names_) {
//         output_names_char.push_back(s.c_str());
//     }

//     auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
//                                          input_names_char.data(), &input_tensor, 1,
//                                          output_names_char.data(), 1);
    
//     float prediction_prob = *output_tensors[0].GetTensorMutableData<float>();

//     if (prediction_prob > CONFIDENCE_THRESHOLD) {
//         return dv_msgs::msg::IndexedCone::BLUE;
//     } else if ((1.0 - prediction_prob) > CONFIDENCE_THRESHOLD) {
//         return dv_msgs::msg::IndexedCone::YELLOW;
//     }
    
//     return std::nullopt;
// }

// bool ProcessLidar::classifyCone(const std::vector<double> &y_vals, const std::vector<double> &x_vals)
// {
//     if (y_vals.size() < 3) return false;

//     int n = y_vals.size();
//     Eigen::MatrixXd A(n, 3);
//     Eigen::VectorXd y(n);

//     for (int i = 0; i < n; ++i) {
//         double x = x_vals.at(i);
//         A(i, 0) = x * x;
//         A(i, 1) = x;
//         A(i, 2) = 1.0;
//         y(i) = y_vals.at(i);
//     }

//     Eigen::Vector3d coeffs = A.colPivHouseholderQr().solve(y);

//     return coeffs(0) > 0;
// }

// std::vector<double> ProcessLidar::movingAverage(const std::vector<double> &data, int kernel)
// {
//     int n = data.size();
//     std::vector<double> result(n, 0.0);
//     if (kernel < 1 || n == 0) return data;

//     int half = kernel / 2;
//     for (int i = 0; i < n; ++i) {
//         int start = std::max(0, i - half);
//         int end = std::min(n - 1, i + half);
//         double sum = 0.0;
//         for (int j = start; j <= end; ++j) {
//             sum += data[j];
//         }
//         result[i] = sum / (end - start + 1);
//     }
//     return result;
// }

// void ProcessLidar::detectConesInClusters(const std::vector<Cluster>& clusters,
//                                         std::vector<Point3D>& positions,
//                                         std::vector<int>& colors)
// {
//     positions.reserve(clusters.size());
//     colors.reserve(clusters.size());

//     for (const auto& cluster : clusters) {
//         auto sorted_cluster = cluster;
//         std::sort(sorted_cluster.begin(), sorted_cluster.end(),
//                   [](const Point4D& a, const Point4D& b) { return a[2] > b[2]; });

//         std::vector<double> intensity_vals;
//         std::vector<double> z_vals;
//         intensity_vals.reserve(sorted_cluster.size());
//         z_vals.reserve(sorted_cluster.size());

//         for (const auto& pt : sorted_cluster) {
//             intensity_vals.push_back(pt[3]);
//             z_vals.push_back(pt[2]);
//         }

//         int kernel = std::max(3, static_cast<int>(0.1 * intensity_vals.size()));
//         if (kernel % 2 == 0) kernel += 1;
//         std::vector<double> averaged_intensities = this->movingAverage(intensity_vals, kernel);
        
//         bool is_yellow_heuristic = this->classifyCone(averaged_intensities, z_vals);
//         int heuristic_color = is_yellow_heuristic ? dv_msgs::msg::IndexedCone::YELLOW : dv_msgs::msg::IndexedCone::BLUE;

//         auto ml_color_opt = predictColor(cluster);

//         if (ml_color_opt.has_value()) {
//             int ml_color = ml_color_opt.value();

//             if (ml_color == heuristic_color) {
//                 auto cone_pos = calculateConePosition(cluster);
//                 positions.push_back(cone_pos);
//                 colors.push_back(ml_color); 
//             }
//         }
//     }
// }

// ProcessLidar::Point3D ProcessLidar::calculateConePosition(const Cluster& cluster)
// {
//     double min_x = cluster[0][0], max_x = cluster[0][0];
//     double min_y = cluster[0][1], max_y = cluster[0][1];
//     for (const auto& point : cluster) {
//         min_x = std::min(min_x, point[0]); max_x = std::max(max_x, point[0]);
//         min_y = std::min(min_y, point[1]); max_y = std::max(max_y, point[1]);
//     }

//     double median_x = getMedian(cluster, 0);
//     double median_y = getMedian(cluster, 1);

//     constexpr double w_median = 0.7;
//     constexpr double w_min_x = 0.3;
//     double cone_x = w_median * median_x + w_min_x * (min_x + CONE_BASE_RADIUS);
//     double cone_y = w_median * median_y + w_min_x * (min_y + CONE_BASE_RADIUS);

//     return {cone_x, cone_y, CONE_HEIGHT};
// }

// double ProcessLidar::getMedian(const Cluster& points, size_t idx) const
// {
//     if (points.empty()) return 0.0;
//     std::vector<double> values;
//     values.reserve(points.size());
//     for(const auto& p : points) {
//         values.push_back(p[idx]);
//     }
//     std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
//     return values[values.size() / 2];
// }

// void ProcessLidar::publishDetectedCones(const std::vector<Point3D>& positions, const std::vector<int>& colors)
// {
//     if (!detected_cones_pub_ || positions.empty()) return;

//     dv_msgs::msg::IndexedTrack track_msg;
//     int yellow_count = 0;
//     int blue_count = 0;

//     for (size_t i = 0; i < positions.size(); ++i) {
//         dv_msgs::msg::IndexedCone cone_msg;
//         double x = positions[i][0];
//         double y = positions[i][1];
//         double z = positions[i][2];

//         if (x < 3.35 || x > 10) continue; 
        
//         double range = sqrt(x * x + y * y);
//         double angle = atan2(y, x);
        
//         cone_msg.location.x = range;
//         cone_msg.location.y = angle;
//         cone_msg.location.z = z;
//         cone_msg.color = colors[i];
//         cone_msg.index = i;
//         track_msg.track.push_back(cone_msg);

//         if (colors[i] == dv_msgs::msg::IndexedCone::YELLOW) yellow_count++;
//         else if (colors[i] == dv_msgs::msg::IndexedCone::BLUE) blue_count++;
//     }

//     if (!track_msg.track.empty()) {
//         detected_cones_pub_->publish(track_msg);
//         RCLCPP_INFO(get_logger(), "Detected cones (Hybrid Method) - Yellow: %d, Blue: %d", yellow_count, blue_count);
//     }
// }

// void ProcessLidar::lidarRawCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg)
// {
//     try {
//         auto points = extractPointsFromPointCloud(msg);
//         processPointCloudData(points, msg->header);
//     } catch (const std::exception& e) {
//         RCLCPP_ERROR(get_logger(), "PointCloud processing error: %s", e.what());
//     }
// }

// void ProcessLidar::lidarRawCallback2(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
// {
//     try {
//         auto points = extractPointsFromPointCloud2(msg);
//         processPointCloudData(points, msg->header);
//     } catch (const std::exception& e) {
//         RCLCPP_ERROR(get_logger(), "PointCloud2 processing error: %s", e.what());
//     }
// }

#include "perception_winter/process_lidar.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <open3d/Open3D.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>
#include <chrono>
#include <Eigen/Dense>

// --- Constants for the ML Model Structure ---
constexpr int NUM_BINS = 10;
constexpr int NUM_CHANNELS = 2; 
constexpr float CONFIDENCE_THRESHOLD = 0.95f;

ProcessLidar::ProcessLidar() : Node("process_lidar"), env_(ORT_LOGGING_LEVEL_WARNING, "process_lidar_onnx")
{
    lidar_raw_input_sub_ = create_subscription<sensor_msgs::msg::PointCloud>(
        LIDAR_RAW_TOPIC, 10,
        [this](const sensor_msgs::msg::PointCloud::SharedPtr msg) {
            lidarRawCallback(msg);
        });

    lidar_raw_input_sub2_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        LIDAR_RAW_TOPIC2, 10,
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            lidarRawCallback2(msg);
        });

    detected_cones_pub_ = create_publisher<dv_msgs::msg::IndexedTrack>("/perception/cones", 10);

    loadOnnxModel();

    RCLCPP_INFO(get_logger(), "Optimized LiDAR Node with Hybrid ML/Heuristic classification started");
}

ProcessLidar::~ProcessLidar()
{
    RCLCPP_INFO(get_logger(), "LiDAR Node shutdown");
}

void ProcessLidar::loadOnnxModel() {
    this->declare_parameter<std::string>("onnx_model_path", "src/perception_winter/cone_model.onnx");
    std::string model_path = this->get_parameter("onnx_model_path").as_string();

    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
        input_node_names_.push_back(input_name_ptr.get());

        auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
        output_node_names_.push_back(output_name_ptr.get());

        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_node_dims_ = input_tensor_info.GetShape();

        RCLCPP_INFO(this->get_logger(), "Successfully loaded ONNX model from: %s", model_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Model Input Name: %s", input_node_names_[0].c_str());
        RCLCPP_INFO(this->get_logger(), "Model expects input shape: [%ld, %ld, %ld]",
            input_node_dims_[0], input_node_dims_[1], input_node_dims_[2]);

    } catch (const Ort::Exception& e) {
        RCLCPP_FATAL(this->get_logger(), "Failed to load ONNX model: %s", e.what());
        rclcpp::shutdown();
    }
}

std::vector<ProcessLidar::Point4D> ProcessLidar::extractPointsFromPointCloud2(
    const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
{
    std::vector<Point4D> points;
    points.reserve(cloud_msg->width * cloud_msg->height);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");
    
    bool has_intensity = false;
    for (const auto& field : cloud_msg->fields) {
        if (field.name == "intensity") {
            has_intensity = true;
            break;
        }
    }
    
    std::optional<sensor_msgs::PointCloud2ConstIterator<float>> iter_intensity_opt;
    if (has_intensity) {
        iter_intensity_opt.emplace(*cloud_msg, "intensity");
    }

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        double intensity = 0.0;
        if (has_intensity && iter_intensity_opt.has_value()) {
            intensity = *(*iter_intensity_opt);
            ++(*iter_intensity_opt);
        }
        points.push_back({*iter_x, *iter_y, *iter_z, intensity});
    }
    return points;
}

std::vector<ProcessLidar::Point4D> ProcessLidar::extractPointsFromPointCloud(
    const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg)
{
    std::vector<Point4D> points;
    points.reserve(cloud_msg->points.size());

    bool has_intensity = !cloud_msg->channels.empty() && 
                        cloud_msg->channels[0].values.size() == cloud_msg->points.size();

    for (size_t i = 0; i < cloud_msg->points.size(); ++i) {
        const auto& pt = cloud_msg->points[i];
        points.push_back({pt.x, pt.y, pt.z, has_intensity ? cloud_msg->channels[0].values[i] : 0.0});
    }
    return points;
}

void ProcessLidar::processPointCloudData(std::vector<Point4D>& points, const std_msgs::msg::Header& header)
{
    (void)header; // Mark as unused
    if (points.empty()) return;

    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    if (!filterCarBodyAndROI(points, cloud)) return;
    if (!removeGroundPlane(cloud, cloud_filtered)) return;

    auto clusters = clusterPoints(cloud_filtered);
    if (clusters.empty()) return;

    auto filtered_clusters = filterClustersBySize(clusters);
    if (filtered_clusters.empty()) return;

    std::vector<Point3D> cone_positions;
    std::vector<int> cone_colors;
    detectConesInClusters(filtered_clusters, cone_positions, cone_colors);

    publishDetectedCones(cone_positions, cone_colors);
}

bool ProcessLidar::filterCarBodyAndROI(const std::vector<Point4D>& input_points, 
                                      pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud)
{
    output_cloud->reserve(input_points.size());
    
    for (const auto& point : input_points) {
        bool is_valid_point = (point[0] > CAR_FRONT_X) || (std::abs(point[1]) > CAR_SIDE_Y);
        if (point[0] > 0 && is_valid_point) {
            pcl::PointXYZI pcl_point;
            pcl_point.x = point[0]; pcl_point.y = point[1];
            pcl_point.z = point[2]; pcl_point.intensity = point[3];
            output_cloud->push_back(pcl_point);
        }
    }

    auto cloud_filtered_pass = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(output_cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(ROI_Y_MIN, ROI_Y_MAX);
    pass.filter(*cloud_filtered_pass);

    pass.setInputCloud(cloud_filtered_pass);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(ROI_Z_MIN, ROI_Z_MAX);
    pass.filter(*output_cloud);
    
    return !output_cloud->empty();
}

bool ProcessLidar::removeGroundPlane(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloud)
{
    if (cloud->empty()) return false;

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::ExtractIndices<pcl::PointXYZI> extract;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(RANSAC_THRESHOLD);

    auto remaining_cloud = cloud;
    int iterations = 0;
    std::optional<Eigen::Vector3f> reference_normal;

    while (remaining_cloud->size() > MIN_POINTS_FOR_PLANE && iterations < MAX_GROUND_ITERATIONS) {
        seg.setInputCloud(remaining_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            break;
        }

        Eigen::Vector3f current_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        if (current_normal.z() < 0) {
            current_normal = -current_normal;
        }
        if (current_normal.z() < MIN_Z_NORMAL_COMPONENT) {
            break;
        }

        if (!reference_normal.has_value()) {
            reference_normal = current_normal;
        } else {
            double angle_rad = std::acos(std::clamp(static_cast<float>(current_normal.dot(reference_normal.value())), -1.0f, 1.0f));
            double angle_deg = angle_rad * (180.0 / M_PI);
            if (angle_deg > MAX_SLOPE_DEVIATION_DEG) {
                break;
            }
        }
        
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        auto next_remaining = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        extract.filter(*next_remaining);
        remaining_cloud = next_remaining;
        
        iterations++;
    }

    *non_ground_cloud = *remaining_cloud;
    return !non_ground_cloud->empty();
}

std::vector<ProcessLidar::Cluster> ProcessLidar::clusterPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
    if (cloud->empty()) return {};

    auto o3d_pcd = std::make_shared<open3d::geometry::PointCloud>();
    o3d_pcd->points_.reserve(cloud->size());
    for (const auto& point : cloud->points) {
        o3d_pcd->points_.emplace_back(point.x, point.y, point.z);
    }
    auto labels = o3d_pcd->ClusterDBSCAN(DBSCAN_EPSILON, DBSCAN_MINPOINTS, false);
    
    int max_label = labels.empty() ? -1 : *std::max_element(labels.begin(), labels.end());
    if (max_label < 0) return {};
    
    std::vector<Cluster> clusters(max_label + 1);
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] >= 0) {
            const auto& point = cloud->points[i];
            clusters[labels[i]].push_back({point.x, point.y, point.z, point.intensity});
        }
    }
    clusters.erase(std::remove_if(clusters.begin(), clusters.end(),
        [](const Cluster& c) { return c.empty(); }), clusters.end());

    return clusters;
}

std::vector<ProcessLidar::Cluster> ProcessLidar::filterClustersBySize(const std::vector<Cluster>& clusters)
{
    std::vector<Cluster> valid_clusters;
    valid_clusters.reserve(clusters.size());
    for (const auto& cluster : clusters) {
        if (cluster.size() < 4) continue;
        double min_z = cluster[0][2], max_z = cluster[0][2];
        for (const auto& point : cluster) {
            min_z = std::min(min_z, point[2]);
            max_z = std::max(max_z, point[2]);
        }
        double height = max_z - min_z;
        if (height >= 0.15 && height <= 0.5) { // Increased height to allow for orange cones
            valid_clusters.push_back(cluster);
        }
    }
    return valid_clusters;
}

std::vector<float> ProcessLidar::createFeatureVector(const Cluster& cluster) {
    std::vector<float> feature_vector(NUM_BINS * NUM_CHANNELS, 0.0f);
    if (cluster.empty()) return feature_vector;

    double min_z = cluster[0][2], max_z = cluster[0][2];
    double min_intensity = cluster[0][3], max_intensity = cluster[0][3];
    for (const auto& point : cluster) {
        min_z = std::min(min_z, point[2]); max_z = std::max(max_z, point[2]);
        min_intensity = std::min(min_intensity, point[3]); max_intensity = std::max(max_intensity, point[3]);
    }

    double z_range = max_z - min_z;
    double intensity_range = max_intensity - min_intensity;
    double bin_width = (z_range > 1e-6) ? z_range / NUM_BINS : 0.0;

    std::vector<double> sum_intensity(NUM_BINS, 0.0);
    std::vector<int> point_count(NUM_BINS, 0);

    for (const auto& point : cluster) {
        double norm_intensity = (intensity_range > 1e-6) ? (point[3] - min_intensity) / intensity_range : 0.0;
        int bin_index = (bin_width > 0) ? static_cast<int>((point[2] - min_z) / bin_width) : 0;
        bin_index = std::min(bin_index, NUM_BINS - 1);
        sum_intensity[bin_index] += norm_intensity;
        point_count[bin_index]++;
    }

    auto min_max_it = std::minmax_element(point_count.begin(), point_count.end());
    float min_c = static_cast<float>(*min_max_it.first);
    float max_c = static_cast<float>(*min_max_it.second);
    float count_range = max_c - min_c;

    for (int i = 0; i < NUM_BINS; ++i) {
        float normalized_count = 0.0f;
        if (count_range > 0) {
            normalized_count = (static_cast<float>(point_count[i]) - min_c) / count_range;
        } else {
            normalized_count = (min_c > 0) ? 1.0f : 0.0f;
        }
        feature_vector[i * NUM_CHANNELS + 0] = normalized_count;
        
        feature_vector[i * NUM_CHANNELS + 1] = (point_count[i] > 0) ? static_cast<float>(sum_intensity[i] / point_count[i]) : 0.0f;
    }

    return feature_vector;
}

std::optional<int> ProcessLidar::predictColor(const Cluster& cluster) {
    std::vector<float> feature_vector = createFeatureVector(cluster);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> concrete_shape = input_node_dims_;
    if (!concrete_shape.empty() && concrete_shape[0] == -1) {
        concrete_shape[0] = 1; 
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, feature_vector.data(), feature_vector.size(),
        concrete_shape.data(), concrete_shape.size());

    std::vector<const char*> input_names_char;
    input_names_char.reserve(input_node_names_.size());
    for (const auto& s : input_node_names_) {
        input_names_char.push_back(s.c_str());
    }

    std::vector<const char*> output_names_char;
    output_names_char.reserve(output_node_names_.size());
    for (const auto& s : output_node_names_) {
        output_names_char.push_back(s.c_str());
    }

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                         input_names_char.data(), &input_tensor, 1,
                                         output_names_char.data(), 1);
    
    float prediction_prob = *output_tensors[0].GetTensorMutableData<float>();

    if (prediction_prob > CONFIDENCE_THRESHOLD) {
        return dv_msgs::msg::IndexedCone::BLUE;
    } else if ((1.0 - prediction_prob) > CONFIDENCE_THRESHOLD) {
        return dv_msgs::msg::IndexedCone::YELLOW;
    }
    
    return std::nullopt;
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
    if (kernel < 1 || n == 0) return data;

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

// --- MODIFIED: Implemented Orange Cone and Hybrid Classification Logic ---
void ProcessLidar::detectConesInClusters(const std::vector<Cluster>& clusters,
                                        std::vector<Point3D>& positions,
                                        std::vector<int>& colors)
{
    positions.reserve(clusters.size());
    colors.reserve(clusters.size());

    for (const auto& cluster : clusters) {
        // --- 1. Calculate Cluster Properties (Height and Distance) ---
        double min_z = cluster[0][2], max_z = cluster[0][2];
        double sum_x = 0.0, sum_y = 0.0;
        for (const auto& point : cluster) {
            min_z = std::min(min_z, point[2]);
            max_z = std::max(max_z, point[2]);
            sum_x += point[0];
            sum_y += point[1];
        }
        double z_range = max_z - min_z;
        double centroid_x = sum_x / cluster.size();
        double centroid_y = sum_y / cluster.size();
        double distance = std::sqrt(centroid_x * centroid_x + centroid_y * centroid_y);

        // --- 2. Check for Big Orange Cones ---
        if (z_range > 0.35 && distance < 7.5) {
            auto cone_pos = calculateConePosition(cluster);
            positions.push_back(cone_pos);
            colors.push_back(2); // Color code 2 for Orange
            continue; // Skip other classification for this cluster
        }

        // --- 3. Apply Exclusion Rules ---
        if (z_range > 0.32 && distance >= 7.5) {
            continue; // Ignore tall cones that are far away
        }

        // --- 4. Perform Heuristic Classification for Blue/Yellow ---
        auto sorted_cluster = cluster;
        std::sort(sorted_cluster.begin(), sorted_cluster.end(),
                  [](const Point4D& a, const Point4D& b) { return a[2] > b[2]; });

        std::vector<double> intensity_vals, z_vals;
        intensity_vals.reserve(sorted_cluster.size());
        z_vals.reserve(sorted_cluster.size());

        for (const auto& pt : sorted_cluster) {
            intensity_vals.push_back(pt[3]);
            z_vals.push_back(pt[2]);
        }

        int kernel = std::max(3, static_cast<int>(0.1 * intensity_vals.size()));
        if (kernel % 2 == 0) kernel += 1;
        std::vector<double> averaged_intensities = this->movingAverage(intensity_vals, kernel);
        
        bool is_yellow_heuristic = this->classifyCone(averaged_intensities, z_vals);
        int heuristic_color = is_yellow_heuristic ? dv_msgs::msg::IndexedCone::YELLOW : dv_msgs::msg::IndexedCone::BLUE;

        // --- 5. Perform ML Classification for Blue/Yellow ---
        auto ml_color_opt = predictColor(cluster);

        // --- 6. Compare Results: Only accept if both methods agree ---
        if (ml_color_opt.has_value()) {
            if (ml_color_opt.value() == heuristic_color) {
                auto cone_pos = calculateConePosition(cluster);
                positions.push_back(cone_pos);
                colors.push_back(ml_color_opt.value()); 
            }
        }
    }
}


ProcessLidar::Point3D ProcessLidar::calculateConePosition(const Cluster& cluster)
{
    double min_x = cluster[0][0], max_x = cluster[0][0];
    double min_y = cluster[0][1], max_y = cluster[0][1];
    for (const auto& point : cluster) {
        min_x = std::min(min_x, point[0]); max_x = std::max(max_x, point[0]);
        min_y = std::min(min_y, point[1]); max_y = std::max(max_y, point[1]);
    }

    double median_x = getMedian(cluster, 0);
    double median_y = getMedian(cluster, 1);

    constexpr double w_median = 0.7;
    constexpr double w_min_x = 0.3;
    double cone_x = w_median * median_x + w_min_x * (min_x + CONE_BASE_RADIUS);
    double cone_y = w_median * median_y + w_min_x * (min_y + CONE_BASE_RADIUS);

    return {cone_x, cone_y, CONE_HEIGHT};
}

double ProcessLidar::getMedian(const Cluster& points, size_t idx) const
{
    if (points.empty()) return 0.0;
    std::vector<double> values;
    values.reserve(points.size());
    for(const auto& p : points) {
        values.push_back(p[idx]);
    }
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2];
}

// --- MODIFIED: Handle Orange Cones in Publisher Log ---
void ProcessLidar::publishDetectedCones(const std::vector<Point3D>& positions, const std::vector<int>& colors)
{
    if (!detected_cones_pub_ || positions.empty()) return;

    dv_msgs::msg::IndexedTrack track_msg;
    int yellow_count = 0;
    int blue_count = 0;
    int orange_count = 0;

    for (size_t i = 0; i < positions.size(); ++i) {
        dv_msgs::msg::IndexedCone cone_msg;
        double x = positions[i][0];
        double y = positions[i][1];
        double z = positions[i][2];

        if (x < 3.35 || x > 10) continue; 
        
        double range = sqrt(x * x + y * y);
        double angle = atan2(y, x);
        
        cone_msg.location.x = range;
        cone_msg.location.y = angle;
        cone_msg.location.z = z;
        cone_msg.color = colors[i];
        cone_msg.index = i;
        track_msg.track.push_back(cone_msg);

        if (colors[i] == dv_msgs::msg::IndexedCone::YELLOW) {
            yellow_count++;
        } else if (colors[i] == dv_msgs::msg::IndexedCone::BLUE) {
            blue_count++;
        } else if (colors[i] == 2) { // Color code 2 for Orange
            orange_count++;
        }
    }

    if (!track_msg.track.empty()) {
        detected_cones_pub_->publish(track_msg);
        RCLCPP_INFO(get_logger(), "Detected cones - Yellow: %d, Blue: %d, Orange: %d", yellow_count, blue_count, orange_count);
    }
}

// Callbacks
void ProcessLidar::lidarRawCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg)
{
    try {
        auto points = extractPointsFromPointCloud(msg);
        processPointCloudData(points, msg->header);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "PointCloud processing error: %s", e.what());
    }
}

void ProcessLidar::lidarRawCallback2(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    try {
        auto points = extractPointsFromPointCloud2(msg);
        processPointCloudData(points, msg->header);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "PointCloud2 processing error: %s", e.what());
    }
}