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
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <chrono>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <chrono>
#include <atomic>
#include <yaml-cpp/yaml.h>
#include <unordered_map>

using namespace perception_winter;

// =============================================
// GLOBAL METRICS (PRESERVED)
// =============================================

static std::atomic<long long> g_true_positives_blue{0};
static std::atomic<long long> g_true_positives_yellow{0};
static std::atomic<long long> g_false_positives_blue{0};     // Predicted Blue, but was Yellow
static std::atomic<long long> g_false_positives_yellow{0};   // Predicted Yellow, but was Blue
static std::atomic<long long> g_rejected_as_blue{0};       // Was Blue, but rejected
static std::atomic<long long> g_rejected_as_yellow{0};     // Was Yellow, but rejected

// =============================================
// ENHANCED GROUND REMOVAL IMPLEMENTATION
// =============================================

MultiStageGroundRemover::MultiStageGroundRemover(const LidarConfig& config) 
    : config_(config) {}

std::vector<Point4D> MultiStageGroundRemover::removeGround(const std::vector<Point4D>& points, 
                                                          rclcpp::Logger logger) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    RCLCPP_DEBUG(logger, "Starting multi-stage ground removal with %zu points", points.size());
    
    try {
        // Stage 1: Preprocessing
        auto preprocessed = preprocessPoints(points);
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // Stage 2: Hybrid Plane Fitting
        auto plane_ground = fitGroundPlanesHybrid(preprocessed);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        // Stage 3: Elevation Grid Filtering
        auto grid_ground = filterByElevationGrid(preprocessed);
        auto t3 = std::chrono::high_resolution_clock::now();
        
        // Combine ground points from all methods
        std::vector<Point4D> all_ground_points;
        all_ground_points.insert(all_ground_points.end(), plane_ground.begin(), plane_ground.end());
        all_ground_points.insert(all_ground_points.end(), grid_ground.begin(), grid_ground.end());
        
        // Remove duplicates
        std::sort(all_ground_points.begin(), all_ground_points.end());
        auto last = std::unique(all_ground_points.begin(), all_ground_points.end());
        all_ground_points.erase(last, all_ground_points.end());
        
        // Remove ground points to get initial non-ground
        std::vector<Point4D> initial_non_ground;
        std::copy_if(preprocessed.begin(), preprocessed.end(), 
                    std::back_inserter(initial_non_ground),
                    [&](const Point4D& point) {
                        return std::find(all_ground_points.begin(), all_ground_points.end(), point) == all_ground_points.end();
                    });
        
        // Stage 4: Cone Base Restoration
        auto final_non_ground = restoreConeBases(initial_non_ground, all_ground_points, preprocessed);
        auto t4 = std::chrono::high_resolution_clock::now();
        
        // Performance logging
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(t4 - start_time);
        auto preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - start_time);
        auto plane_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        auto grid_time = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
        auto restoration_time = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3);
        
        RCLCPP_DEBUG(logger, 
                    "Multi-stage ground removal completed in %ld μs (preprocess: %ld, plane: %ld, grid: %ld, restoration: %ld)",
                    total_duration.count(), preprocess_time.count(), plane_time.count(), grid_time.count(), restoration_time.count());
        RCLCPP_DEBUG(logger, "Ground removal: %zu -> %zu points (%zu ground removed)",
                    points.size(), final_non_ground.size(), all_ground_points.size());
        
        return final_non_ground;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "Multi-stage ground removal failed: %s", e.what());
        return points; // Fallback: return all points as non-ground
    }
}

std::vector<Point4D> MultiStageGroundRemover::preprocessPoints(const std::vector<Point4D>& points) {
    // Voxel downsampling for density normalization
    std::unordered_map<std::string, Point4D> voxel_map;
    double voxel_size = config_.ground_removal.voxel_downsample_size;
    
    for (const auto& point : points) {
        int voxel_x = static_cast<int>(point[0] / voxel_size);
        int voxel_y = static_cast<int>(point[1] / voxel_size);
        int voxel_z = static_cast<int>(point[2] / voxel_size);
        
        std::string key = std::to_string(voxel_x) + "_" + std::to_string(voxel_y) + "_" + std::to_string(voxel_z);
        
        if (!voxel_map.count(key) || point[3] > voxel_map[key][3]) {
            voxel_map[key] = point; // Keep point with highest intensity in each voxel
        }
    }
    
    std::vector<Point4D> downsampled;
    downsampled.reserve(voxel_map.size());
    for (const auto& [key, point] : voxel_map) {
        downsampled.push_back(point);
    }
    
    // Apply single-axis ground smoothing
    return singleAxisGroundSmoothing(downsampled);
}

std::vector<Point4D> MultiStageGroundRemover::singleAxisGroundSmoothing(const std::vector<Point4D>& points) {
    struct GridKey {
        int x, y;
        bool operator==(const GridKey& other) const { return x == other.x && y == other.y; }
    };
    
    struct GridKeyHash {
        std::size_t operator()(const GridKey& k) const { return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1); }
    };
    
    std::unordered_map<GridKey, double, GridKeyHash> min_height_per_cell;
    double cell_size = config_.ground_removal.single_axis_smoothing_cell_size;
    
    // Find minimum height per grid cell
    for (const auto& point : points) {
        GridKey key = {static_cast<int>(point[0] / cell_size), static_cast<int>(point[1] / cell_size)};
        if (!min_height_per_cell.count(key) || point[2] < min_height_per_cell[key]) {
            min_height_per_cell[key] = point[2];
        }
    }
    
    // Keep only points close to the minimum height (ground points)
    std::vector<Point4D> smoothed;
    double ground_threshold = config_.ground_removal.ground_smoothing_threshold;
    
    for (const auto& point : points) {
        GridKey key = {static_cast<int>(point[0] / cell_size), static_cast<int>(point[1] / cell_size)};
        if (min_height_per_cell.count(key) && 
            point[2] - min_height_per_cell[key] < ground_threshold) {
            smoothed.push_back(point);
        }
    }
    
    return smoothed;
}

std::vector<Point4D> MultiStageGroundRemover::fitGroundPlanesHybrid(const std::vector<Point4D>& points) {
    std::vector<Point4D> all_ground_points;
    
    // Run both methods sequentially (TBB removed for compatibility)
    auto angular_ground = fitAngularSectors(points);
    auto zonal_ground = fitZonalPCA(points);
    
    all_ground_points.insert(all_ground_points.end(), angular_ground.begin(), angular_ground.end());
    all_ground_points.insert(all_ground_points.end(), zonal_ground.begin(), zonal_ground.end());
    
    return all_ground_points;
}

std::vector<Point4D> MultiStageGroundRemover::fitAngularSectors(const std::vector<Point4D>& points) {
    const int num_sectors = config_.ground_removal.num_angular_sectors;
    std::vector<std::vector<Point4D>> sector_points(num_sectors);
    std::vector<PlaneModel> sector_planes(num_sectors);
    
    // Distribute points to angular sectors
    for (const auto& point : points) {
        double angle = std::atan2(point[1], point[0]); // -π to π
        int sector = static_cast<int>((angle + M_PI) / (2 * M_PI) * num_sectors);
        sector = std::clamp(sector, 0, num_sectors - 1);
        sector_points[sector].push_back(point);
    }
    
    // Sequential plane fitting per sector (TBB parallel_for replaced)
    for (int sector = 0; sector < num_sectors; ++sector) {
        if (static_cast<int>(sector_points[sector].size()) >= config_.ground_removal.min_points_per_sector) {
            sector_planes[sector] = fitPlaneToLowestPoints(sector_points[sector]);
        }
    }
    
    // Collect ground points from all sectors
    std::vector<Point4D> ground_points;
    for (int sector = 0; sector < num_sectors; ++sector) {
        ground_points.insert(ground_points.end(),
                           sector_planes[sector].inliers.begin(),
                           sector_planes[sector].inliers.end());
    }
    
    return ground_points;
}

std::vector<Point4D> MultiStageGroundRemover::fitZonalPCA(const std::vector<Point4D>& points) {
    const auto& rings = config_.ground_removal.concentric_rings;
    const int segments_per_ring = config_.ground_removal.segments_per_ring;
    const int total_zones = rings.size() * segments_per_ring;
    
    std::vector<PlaneModel> zone_planes(total_zones);
    
    // Sequential processing (TBB parallel_for replaced)
    for (int ring_idx = 0; ring_idx < static_cast<int>(rings.size()); ++ring_idx) {
        auto [ring_min, ring_max] = rings[ring_idx];
        
        for (int segment = 0; segment < segments_per_ring; ++segment) {
            double angle_start = (2 * M_PI / segments_per_ring) * segment;
            double angle_end = (2 * M_PI / segments_per_ring) * (segment + 1);
            
            // Extract points in this zone
            std::vector<Point4D> zone_points;
            for (const auto& point : points) {
                double distance = std::hypot(point[0], point[1]);
                double angle = std::atan2(point[1], point[0]);
                if (angle < 0) angle += 2 * M_PI; // Normalize to [0, 2π]
                
                if (distance >= ring_min && distance < ring_max &&
                    angle >= angle_start && angle < angle_end) {
                    zone_points.push_back(point);
                }
            }
            
            if (static_cast<int>(zone_points.size()) >= config_.ground_removal.min_points_per_zone) {
                int zone_idx = ring_idx * segments_per_ring + segment;
                zone_planes[zone_idx] = fitPlaneToLowestPoints(zone_points);
            }
        }
    }
    
    // Merge ground points from all zones
    std::vector<Point4D> ground_points;
    for (const auto& plane : zone_planes) {
        ground_points.insert(ground_points.end(),
                           plane.inliers.begin(), plane.inliers.end());
    }
    
    return ground_points;
}

MultiStageGroundRemover::PlaneModel MultiStageGroundRemover::fitPlaneToLowestPoints(const std::vector<Point4D>& points) {
    // Take lowest points for ground plane fitting
    auto sorted_points = points;
    std::sort(sorted_points.begin(), sorted_points.end(),
             [](const Point4D& a, const Point4D& b) { return a[2] < b[2]; });
    
    int num_lowest = std::max(config_.ground_removal.min_points_per_sector, 
                             (int)(sorted_points.size() * config_.ground_removal.lowest_points_ratio));
    std::vector<Point4D> lowest_points(sorted_points.begin(), 
                                      sorted_points.begin() + num_lowest);
    
    // Fit plane using PCA
    Eigen::MatrixXd points_matrix(num_lowest, 3);
    for (int i = 0; i < num_lowest; ++i) {
        points_matrix(i, 0) = lowest_points[i][0];
        points_matrix(i, 1) = lowest_points[i][1];
        points_matrix(i, 2) = lowest_points[i][2];
    }
    
    Eigen::Vector3d centroid = points_matrix.colwise().mean();
    Eigen::MatrixXd centered = points_matrix.rowwise() - centroid.transpose();
    Eigen::Matrix3d cov = centered.transpose() * centered;
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    Eigen::Vector3d normal = eig.eigenvectors().col(0); // Smallest eigenvalue
    
    // Ensure normal points upward
    if (normal.z() < 0) normal = -normal;
    
    double d = -normal.dot(centroid);
    
    // Find inliers within adaptive threshold
    double threshold = computeAdaptiveThreshold(lowest_points);
    std::vector<Point4D> inliers;
    
    for (const auto& point : points) {
        Eigen::Vector3d pt(point[0], point[1], point[2]);
        double distance = std::abs(normal.dot(pt) + d);
        if (distance < threshold) {
            inliers.push_back(point);
        }
    }
    
    return {normal, d, inliers};
}

double MultiStageGroundRemover::computeAdaptiveThreshold(const std::vector<Point4D>& points) {
    // Adaptive threshold based on point density and distribution
    if (points.empty()) return 0.02; // Default 2cm
    
    double avg_distance = 0.0;
    int count = 0;
    
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < std::min(i + 10, points.size()); ++j) {
            double dx = points[i][0] - points[j][0];
            double dy = points[i][1] - points[j][1];
            double dz = points[i][2] - points[j][2];
            avg_distance += std::sqrt(dx*dx + dy*dy + dz*dz);
            count++;
        }
    }
    
    if (count > 0) {
        avg_distance /= count;
        return std::clamp(avg_distance * 0.5, 0.01, 0.05); // 1-5cm threshold
    }
    
    return 0.02;
}

std::vector<Point4D> MultiStageGroundRemover::filterByElevationGrid(const std::vector<Point4D>& points) {
    struct GridCell {
        std::vector<double> heights;
        double mean_height = 0.0;
        double height_variance = 0.0;
        bool is_ground = false;
    };
    
    double cell_size = config_.ground_removal.grid_cell_size;
    int grid_x = std::ceil(12.0 / cell_size);  // ROI x_max
    int grid_y = std::ceil(7.0 / cell_size);   // ROI y_max - y_min
    
    std::vector<std::vector<GridCell>> grid(grid_x, std::vector<GridCell>(grid_y));
    
    // Populate grid with point heights
    for (const auto& point : points) {
        int x_idx = point[0] / cell_size;
        int y_idx = (point[1] + 3.5) / cell_size; // Adjust for ROI y_min = -3.5
        
        if (x_idx >= 0 && x_idx < grid_x && y_idx >= 0 && y_idx < grid_y) {
            grid[x_idx][y_idx].heights.push_back(point[2]);
        }
    }
    
    // Compute statistics and classify ground cells (sequential - TBB removed)
    for (int x = 0; x < grid_x; ++x) {
        for (int y = 0; y < grid_y; ++y) {
            auto& cell = grid[x][y];
            if (static_cast<int>(cell.heights.size()) >= config_.ground_removal.min_points_per_cell) {
                // Compute mean
                cell.mean_height = std::accumulate(cell.heights.begin(), cell.heights.end(), 0.0) 
                                 / cell.heights.size();
                
                // Compute variance
                double sum_sq = 0.0;
                for (double h : cell.heights) {
                    sum_sq += (h - cell.mean_height) * (h - cell.mean_height);
                }
                cell.height_variance = sum_sq / cell.heights.size();
                
                // Adaptive variance threshold based on distance
                double distance = (x + 0.5) * cell_size;
                double max_variance = config_.ground_removal.adaptive_variance_base + 
                                    (distance / 12.0) * config_.ground_removal.adaptive_variance_scale;
                
                cell.is_ground = (cell.height_variance < max_variance);
            }
        }
    }
    
    // Extract ground points
    std::vector<Point4D> ground_points;
    for (const auto& point : points) {
        int x_idx = point[0] / cell_size;
        int y_idx = (point[1] + 3.5) / cell_size;
        
        if (x_idx >= 0 && x_idx < grid_x && y_idx >= 0 && y_idx < grid_y) {
            const auto& cell = grid[x_idx][y_idx];
            if (cell.is_ground && std::abs(point[2] - cell.mean_height) < 0.05) { // 5cm tolerance
                ground_points.push_back(point);
            }
        }
    }
    
    return ground_points;
}

std::vector<Point4D> MultiStageGroundRemover::restoreConeBases(const std::vector<Point4D>& non_ground_points,
                                                              const std::vector<Point4D>& ground_points,
                                                              const std::vector<Point4D>& all_points) {
    (void)all_points; // Mark parameter as unused to suppress warning
    
    // Simple clustering for cone candidates (using your existing DBSCAN would be better)
    std::vector<std::vector<Point4D>> cone_candidates;
    double cluster_epsilon = 0.2; // Similar to your dbscan_epsilon
    
    // Simple distance-based clustering (replace with your DBSCAN for production)
    std::vector<bool> processed(non_ground_points.size(), false);
    for (size_t i = 0; i < non_ground_points.size(); ++i) {
        if (processed[i]) continue;
        
        std::vector<Point4D> cluster;
        std::vector<size_t> queue = {i};
        processed[i] = true;
        
        while (!queue.empty()) {
            size_t current = queue.back();
            queue.pop_back();
            cluster.push_back(non_ground_points[current]);
            
            for (size_t j = 0; j < non_ground_points.size(); ++j) {
                if (!processed[j]) {
                    double dx = non_ground_points[current][0] - non_ground_points[j][0];
                    double dy = non_ground_points[current][1] - non_ground_points[j][1];
                    double dz = non_ground_points[current][2] - non_ground_points[j][2];
                    double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
                    
                    if (distance < cluster_epsilon) {
                        queue.push_back(j);
                        processed[j] = true;
                    }
                }
            }
        }
        
        if (static_cast<int>(cluster.size()) >= config_.ground_removal.min_cone_cluster_points) {
            cone_candidates.push_back(cluster);
        }
    }
    
    // Restore cone base points
    std::vector<Point4D> restored_points = non_ground_points;
    double cylinder_radius = config_.ground_removal.base_cylinder_radius;
    double cylinder_height = config_.ground_removal.base_cylinder_height;
    
    for (const auto& cluster : cone_candidates) {
        // Compute cluster centroid
        Eigen::Vector3d centroid(0, 0, 0);
        for (const auto& point : cluster) {
            centroid += Eigen::Vector3d(point[0], point[1], point[2]);
        }
        centroid /= cluster.size();
        
        // Find ground points within the cylinder around cluster base
        for (const auto& ground_point : ground_points) {
            double dx = ground_point[0] - centroid.x();
            double dy = ground_point[1] - centroid.y();
            double horizontal_distance = std::sqrt(dx*dx + dy*dy);
            
            if (horizontal_distance < cylinder_radius && 
                std::abs(ground_point[2] - centroid.z()) < cylinder_height/2) {
                restored_points.push_back(ground_point);
            }
        }
    }
    
    return restored_points;
}

// =============================================
// ENHANCED POINT CLOUD PROCESSOR IMPLEMENTATION
// =============================================

EnhancedPointCloudProcessor::EnhancedPointCloudProcessor(const LidarConfig& config)
    : PointCloudProcessor(config), ground_remover_(config) {}

bool EnhancedPointCloudProcessor::removeGroundPlaneEnhanced(PointCloudPtr cloud,
                                                           PointCloudPtr non_ground_cloud,
                                                           rclcpp::Logger logger) {
    if (cloud->empty()) {
        RCLCPP_WARN(logger, "Empty cloud for enhanced ground removal");
        return false;
    }

    RCLCPP_DEBUG(logger, "Starting enhanced ground removal with %zu points", cloud->size());

    try {
        // Convert to Point4D format
        std::vector<Point4D> points;
        points.reserve(cloud->size());
        for (const auto& pcl_point : cloud->points) {
            points.push_back({pcl_point.x, pcl_point.y, pcl_point.z, pcl_point.intensity});
        }
        
        // Use multi-stage ground removal
        auto non_ground_points = ground_remover_.removeGround(points, logger);
        
        // Convert back to PCL
        non_ground_cloud->points.clear();
        non_ground_cloud->points.reserve(non_ground_points.size());
        
        for (const auto& point : non_ground_points) {
            pcl::PointXYZI pcl_point;
            pcl_point.x = point[0];
            pcl_point.y = point[1];
            pcl_point.z = point[2];
            pcl_point.intensity = point[3];
            non_ground_cloud->points.push_back(pcl_point);
        }
        
        RCLCPP_INFO(logger,
                   "Enhanced ground removal: %zu -> %zu points (%zu ground points removed)",
                   cloud->size(), non_ground_cloud->size(), points.size() - non_ground_points.size());
        
        return !non_ground_cloud->empty();
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "Enhanced ground removal failed: %s", e.what());
        // Fallback to original RANSAC using temporary instance
        PointCloudProcessor fallback_processor(config_);
        return fallback_processor.removeGroundPlane(cloud, non_ground_cloud, logger);
    }
}

// =============================================
// ML CLASSIFIER IMPLEMENTATION
// =============================================

ConeClassifier::ConeClassifier(Ort::Env& env, const LidarConfig& config) : env_(env), config_(config) {}

bool ConeClassifier::initialize(const std::string& model_path) {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        // Get input names
        auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
        input_node_names_.push_back(input_name_ptr.get());

        // Get output names  
        auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
        output_node_names_.push_back(output_name_ptr.get());

        // Get input dimensions - FIXED: Properly declare the variable first
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo(); // FIXED: Use input_type_info, not input_tensor_info
        input_node_dims_ = input_tensor_info.GetShape();

        RCLCPP_INFO(rclcpp::get_logger("cone_classifier"), 
                   "ML Classifier initialized successfully with confidence threshold: %.2f", 
                   config_.confidence_threshold);
        return true;
    } 
    catch (const Ort::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_classifier"), 
                    "ONNX initialization failed: %s", e.what());
        return false;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_classifier"), 
                    "ML Classifier initialization failed: %s", e.what());
        return false;
    }
}

std::vector<float> ConeClassifier::createFeatureVector(const Cluster& cluster) const {
    std::vector<float> feature_vector(config_.num_bins * 2, 0.0f);
    if (cluster.empty()) return feature_vector;

    // Calculate normalization parameters
    double min_z = cluster[0][2], max_z = cluster[0][2];
    double min_intensity = cluster[0][3], max_intensity = cluster[0][3];
    
    for (const auto& point : cluster) {
        min_z = std::min(min_z, point[2]); 
        max_z = std::max(max_z, point[2]);
        min_intensity = std::min(min_intensity, point[3]); 
        max_intensity = std::max(max_intensity, point[3]);
    }

    double z_range = max_z - min_z;
    double intensity_range = max_intensity - min_intensity;
    double bin_width = (z_range > 1e-6) ? z_range / config_.num_bins : 0.0;

    // Bin points and calculate statistics
    std::vector<double> sum_intensity(config_.num_bins, 0.0);
    std::vector<int> point_count(config_.num_bins, 0);

    for (const auto& point : cluster) {
        double norm_intensity = (intensity_range > 1e-6) ? 
                               (point[3] - min_intensity) / intensity_range : 0.0;
        int bin_index = (bin_width > 0) ? 
                       static_cast<int>((point[2] - min_z) / bin_width) : 0;
        bin_index = std::min(bin_index, config_.num_bins - 1);
        
        sum_intensity[bin_index] += norm_intensity;
        point_count[bin_index]++;
    }

    // Normalize counts
    auto min_max_it = std::minmax_element(point_count.begin(), point_count.end());
    float min_c = static_cast<float>(*min_max_it.first);
    float max_c = static_cast<float>(*min_max_it.second);
    float count_range = max_c - min_c;

    for (int i = 0; i < config_.num_bins; ++i) {
        // Normalized point count
        float normalized_count = 0.0f;
        if (count_range > 0) {
            normalized_count = (static_cast<float>(point_count[i]) - min_c) / count_range;
        } else {
            normalized_count = (min_c > 0) ? 1.0f : 0.0f;
        }
        feature_vector[i * 2 + 0] = normalized_count;
        
        // Average normalized intensity
        feature_vector[i * 2 + 1] = (point_count[i] > 0) ? 
                                   static_cast<float>(sum_intensity[i] / point_count[i]) : 0.0f;
    }

    return feature_vector;
}

std::optional<int> ConeClassifier::classify(const Cluster& cluster) {
    if (cluster.empty()) {
        return std::nullopt;
    }

    try {
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

        // Apply confidence thresholding
        if (prediction_prob > config_.confidence_threshold) {
            return dv_msgs::msg::IndexedCone::BLUE;
        } else if ((1.0 - prediction_prob) > config_.confidence_threshold) {
            return dv_msgs::msg::IndexedCone::YELLOW;
        }
        
        return std::nullopt;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_classifier"), 
                    "Classification failed: %s", e.what());
        return std::nullopt;
    }
}

// =============================================
// HEURISTIC CLASSIFIER IMPLEMENTATION
// =============================================

HeuristicClassifier::HeuristicClassifier(const LidarConfig& config) : config_(config) {}

bool HeuristicClassifier::classifyCone(const std::vector<double> &y_vals, const std::vector<double> &x_vals) {
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
    return coeffs(0) > 0; // Positive quadratic coefficient indicates upward curve (yellow)
}

std::vector<double> HeuristicClassifier::movingAverage(const std::vector<double> &data, int kernel) {
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

std::optional<int> HeuristicClassifier::classify(const Cluster& cluster) {
    if (cluster.size() < 3) return std::nullopt;

    try {
        // Sort by height for consistent analysis
        auto sorted_cluster = cluster;
        std::sort(sorted_cluster.begin(), sorted_cluster.end(),
                  [](const Point4D& a, const Point4D& b) { return a[2] > b[2]; });

        // Extract intensity and height profiles
        std::vector<double> intensity_vals;
        std::vector<double> z_vals;
        intensity_vals.reserve(sorted_cluster.size());
        z_vals.reserve(sorted_cluster.size());

        for (const auto& pt : sorted_cluster) {
            intensity_vals.push_back(pt[3]);
            z_vals.push_back(pt[2]);
        }

        // Apply smoothing for noise reduction
        int kernel = std::max(config_.min_kernel_size, static_cast<int>(config_.moving_average_factor * intensity_vals.size()));
        if (kernel % 2 == 0) kernel += 1;
        std::vector<double> averaged_intensities = this->movingAverage(intensity_vals, kernel);
        
        // Classify based on intensity profile curvature
        bool is_yellow_heuristic = this->classifyCone(averaged_intensities, z_vals);
        return is_yellow_heuristic ? dv_msgs::msg::IndexedCone::YELLOW : dv_msgs::msg::IndexedCone::BLUE;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("heuristic_classifier"), 
                    "Heuristic classification failed: %s", e.what());
        return std::nullopt;
    }
}

// =============================================
// POINT CLOUD PROCESSOR IMPLEMENTATION
// =============================================

PointCloudProcessor::PointCloudProcessor(const LidarConfig& config) : config_(config) {}

bool PointCloudProcessor::filterCarBodyAndROI(const std::vector<Point4D>& input_points, 
                                              PointCloudPtr output_cloud,
                                              rclcpp::Logger logger) {
    if (input_points.empty()) {
        RCLCPP_WARN(logger, "Empty input point cloud for filtering");
        return false;
    }

    output_cloud->reserve(input_points.size());
    
    int car_body_points = 0;
    int behind_car_points = 0;
    int outside_roi_points = 0;
    int valid_points = 0;
    
    RCLCPP_DEBUG(logger, "Starting ROI filtering with %zu input points", input_points.size());
    
    for (const auto& point : input_points) {
        double x = point[0];
        double y = point[1];
        double z = point[2];
        
        // Skip points behind the car (redundant safety check)
        if (x <= 0) {
            behind_car_points++;
            continue;
        }
        
        // Skip points that are on the car body (primary exclusion)
        bool on_car_body = (x <= config_.car_front_x) && 
                          (std::abs(y) <= config_.car_side_y);
        if (on_car_body) {
            car_body_points++;
            continue;
        }
        
        // Apply ROI filtering (secondary validation)
        bool in_roi_y = (y >= config_.roi_y_min) && (y <= config_.roi_y_max);
        bool in_roi_z = (z >= config_.roi_z_min) && (z <= config_.roi_z_max);
        
        if (in_roi_y && in_roi_z) {
            pcl::PointXYZI pcl_point;
            pcl_point.x = x;
            pcl_point.y = y;
            pcl_point.z = z;
            pcl_point.intensity = point[3];
            output_cloud->push_back(pcl_point);
            valid_points++;
        } else {
            outside_roi_points++;
        }
    }
    
    RCLCPP_INFO(logger,
               "ROI filtering: %zu -> %zu points (behind: %d, car_body: %d, outside_roi: %d, valid: %d)", 
               input_points.size(), output_cloud->size(), behind_car_points, 
               car_body_points, outside_roi_points, valid_points);
    
    return !output_cloud->empty();
}

bool PointCloudProcessor::removeGroundPlane(PointCloudPtr cloud,
                                           PointCloudPtr non_ground_cloud,
                                           rclcpp::Logger logger) {
    if (cloud->empty()) {
        RCLCPP_WARN(logger, "Empty cloud for ground removal");
        return false;
    }

    RCLCPP_DEBUG(logger, "Starting ground removal with %zu points", cloud->size());

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::ExtractIndices<pcl::PointXYZI> extract;

    // Configure RANSAC parameters
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(config_.ransac_threshold);

    auto remaining_cloud = cloud;
    auto ground_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    
    std::optional<Eigen::Vector3f> reference_normal;
    int iterations = 0;

    // Iterative ground plane removal
    while (remaining_cloud->size() > config_.min_points_for_plane && 
           iterations < config_.max_ground_iterations) {
        
        int dynamic_max_iter = std::min(static_cast<int>(remaining_cloud->size() / 200), 
                                       config_.max_ground_iterations);
        if (iterations >= dynamic_max_iter) break;

        seg.setInputCloud(remaining_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) break;

        Eigen::Vector3f current_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        if (current_normal.z() < 0) current_normal = -current_normal;

        if (!isValidGroundPlane(current_normal, reference_normal)) break;

        if (!reference_normal.has_value()) {
            reference_normal = current_normal;
        }

        // Extract current ground plane
        auto current_ground_plane = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*current_ground_plane);
        *ground_cloud += *current_ground_plane;

        // Update remaining cloud
        auto next_remaining = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        extract.setNegative(true);
        extract.filter(*next_remaining);
        remaining_cloud = next_remaining;
        iterations++;
    }

    *non_ground_cloud = *remaining_cloud;
    
    RCLCPP_INFO(logger,
               "Ground removal: %zu -> %zu points (%zu ground points removed in %d iterations)",
               cloud->size(), non_ground_cloud->size(), ground_cloud->size(), iterations);
    
    return !non_ground_cloud->empty();
}

bool PointCloudProcessor::isValidGroundPlane(const Eigen::Vector3f& normal, 
                                            const std::optional<Eigen::Vector3f>& reference_normal) const {
    // Check normal orientation
    if (normal.z() < config_.min_z_normal_component) return false;

    // Check consistency with reference normal
    if (reference_normal.has_value()) {
        double dot_product = normal.dot(reference_normal.value());
        double angle_rad = std::acos(std::clamp(dot_product, -1.0, 1.0));
        double angle_deg = angle_rad * (180.0 / M_PI);
        if (angle_deg > config_.max_slope_deviation_deg) return false;
    }

    return true;
}

// =============================================
// CLUSTER PROCESSOR IMPLEMENTATION
// =============================================

ClusterProcessor::ClusterProcessor(const LidarConfig& config) : config_(config) {}

std::vector<Cluster> ClusterProcessor::clusterPoints(const PointCloudPtr cloud, rclcpp::Logger logger) {
    if (cloud->empty()) {
        RCLCPP_WARN(logger, "Empty cloud for clustering");
        return {};
    }

    RCLCPP_DEBUG(logger, "Starting clustering with %zu points", cloud->size());

    try {
        auto o3d_pcd = std::make_shared<open3d::geometry::PointCloud>();
        o3d_pcd->points_.reserve(cloud->size());
        
        for (const auto& point : cloud->points) {
            o3d_pcd->points_.emplace_back(point.x, point.y, point.z);
        }

        auto labels = o3d_pcd->ClusterDBSCAN(config_.dbscan_epsilon, 
                                            config_.dbscan_minpoints, false);
        
        int max_label = 0;
        if (!labels.empty()) {
            max_label = *std::max_element(labels.begin(), labels.end());
        }
        
        std::vector<Cluster> clusters(max_label + 1);
        int noise_points = 0;
        
        for (size_t i = 0; i < labels.size(); ++i) {
            int label = labels[i];
            if (label >= 0) {
                const auto& point = cloud->points[i];
                clusters[label].push_back({point.x, point.y, point.z, point.intensity});
            } else {
                noise_points++;
            }
        }

        // Remove empty clusters
        clusters.erase(std::remove_if(clusters.begin(), clusters.end(),
            [](const Cluster& c) { return c.empty(); }), clusters.end());

        RCLCPP_INFO(logger,
                   "Clustering: %zu points -> %zu clusters (noise points: %d)",
                   cloud->size(), clusters.size(), noise_points);
        
        return clusters;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "Clustering failed: %s", e.what());
        return {};
    }
}

std::vector<Cluster> ClusterProcessor::filterClustersBySize(const std::vector<Cluster>& clusters, 
                                                           std::vector<bool>& orange_candidates,
                                                           rclcpp::Logger logger) {
    std::vector<Cluster> valid_clusters;
    valid_clusters.reserve(clusters.size());
    orange_candidates.clear();
    orange_candidates.reserve(clusters.size());

    int total_points_before = 0;
    int total_points_after = 0;
    int rejected_by_points = 0;
    int rejected_by_height = 0;
    int rejected_by_width = 0;
    int rejected_by_x_dist = 0;

    for (const auto& cluster : clusters) {
        total_points_before += cluster.size();
        
        // Minimum points check
        if (cluster.size() < config_.min_cluster_points) {
            rejected_by_points++;
            orange_candidates.push_back(false);
            continue;
        }

        // Calculate cluster properties
        double min_x = cluster[0][0], max_x = cluster[0][0];
        double min_y = cluster[0][1], max_y = cluster[0][1];
        double min_z = cluster[0][2], max_z = cluster[0][2];
        double sum_x = 0.0; // Removed sum_y since centroid_y is unused

        for (const auto& point : cluster) {
            min_x = std::min(min_x, point[0]); max_x = std::max(max_x, point[0]);
            min_y = std::min(min_y, point[1]); max_y = std::max(max_y, point[1]);
            min_z = std::min(min_z, point[2]); max_z = std::max(max_z, point[2]);
            sum_x += point[0];
            // Removed: sum_y += point[1]; since centroid_y is unused
        }

        double height = max_z - min_z;
        double width = std::max(max_x - min_x, max_y - min_y);
        double centroid_x = sum_x / cluster.size();
        // Removed unused centroid_y variable

        // Check for orange cone candidate during filtering
        bool is_orange_candidate = isOrangeConeCandidate(cluster);
        orange_candidates.push_back(is_orange_candidate);

        // Regular cone filtering
        bool valid_height = (height >= config_.min_cluster_height && height <= config_.max_cluster_height);
        bool valid_width = (width <= config_.max_cluster_width);
        bool valid_x_pos = (centroid_x <= config_.roi_x_max);

        if (!valid_height) rejected_by_height++;
        if (!valid_width) rejected_by_width++;
        if (!valid_x_pos) rejected_by_x_dist++;

        if (valid_height && valid_width && valid_x_pos) {
            valid_clusters.push_back(cluster);
            total_points_after += cluster.size();
        }
    }

    RCLCPP_INFO(logger,  // Use the passed logger parameter
                "Cluster filtering: %zu -> %zu clusters, points: %d -> %d "
                "(rejected: points=%d, height=%d, width=%d, x_dist=%d)",
                clusters.size(), valid_clusters.size(), total_points_before, total_points_after,
                rejected_by_points, rejected_by_height, rejected_by_width, rejected_by_x_dist);

    return valid_clusters;
}

void ClusterProcessor::detectConesInClusters(const std::vector<Cluster>& clusters,
                                            const std::vector<bool>& orange_candidates,
                                            std::vector<Point3D>& positions, 
                                            std::vector<int>& colors,
                                            ConeClassifier& ml_classifier,
                                            HeuristicClassifier& heuristic_classifier,
                                            rclcpp::Logger logger) {  // Use this logger parameter
    positions.reserve(clusters.size());
    colors.reserve(clusters.size());

    int accepted_cones = 0;
    int orange_cones = 0;

    for (size_t i = 0; i < clusters.size(); ++i) {
        const auto& cluster = clusters[i];
        if (cluster.empty()) continue;

        // --- 1. Handle Orange Cones ---
        if (orange_candidates[i]) {
            auto cone_pos = calculateConePosition(cluster);
            positions.push_back(cone_pos);
            colors.push_back(dv_msgs::msg::IndexedCone::ORANGE_BIG);
            accepted_cones++;
            orange_cones++;
            continue;
        }

        // --- 2. Determine Ground Truth from Simulator Intensity ---
        double total_intensity = 0.0;
        for (const auto& point : cluster) {
            total_intensity += point[3];
        }
        double avg_intensity = total_intensity / cluster.size();
        
        bool is_ground_truth_blue = (avg_intensity > config_.intensity_threshold_blue);

        // --- 3. Get Predictions from Both Classifiers ---
        auto heuristic_color_opt = heuristic_classifier.classify(cluster);
        auto ml_color_opt = ml_classifier.classify(cluster);

        // --- 4. Compare and Update Metrics ---
        int final_color = -1;

        if (ml_color_opt.has_value() && heuristic_color_opt.has_value()) {
            if (ml_color_opt.value() == heuristic_color_opt.value()) {
                final_color = ml_color_opt.value();
                auto cone_pos = calculateConePosition(cluster);
                positions.push_back(cone_pos);
                colors.push_back(final_color);
                accepted_cones++;
                
                // Update True/False Positive counters
                if (final_color == dv_msgs::msg::IndexedCone::BLUE) {
                    if (is_ground_truth_blue) g_true_positives_blue++;
                    else g_false_positives_blue++;
                } else {
                    if (!is_ground_truth_blue) g_true_positives_yellow++;
                    else g_false_positives_yellow++;
                }
            }
        }
        
        // --- 5. Handle Rejections ---
        if (final_color == -1) {
            if (is_ground_truth_blue) g_rejected_as_blue++;
            else g_rejected_as_yellow++;
        }
    }

    // Use the passed logger parameter
    RCLCPP_INFO(logger, 
                "Cone detection: Accepted cones (ML+Heuristic agreement): %d, Orange cones: %d",
                accepted_cones, orange_cones);
}

void ClusterProcessor::printClusterStats(const std::vector<Cluster>& clusters, rclcpp::Logger logger) const {
    RCLCPP_INFO(logger, "Cluster Statistics:");
    RCLCPP_INFO(logger, "  - Total clusters: %zu", clusters.size());
    
    for (size_t i = 0; i < clusters.size(); ++i) {
        const auto& cluster = clusters[i];
        double avg_intensity = 0.0;
        double min_z = std::numeric_limits<double>::max();
        double max_z = std::numeric_limits<double>::lowest();
        double min_x = cluster[0][0], max_x = cluster[0][0];
        double min_y = cluster[0][1], max_y = cluster[0][1];
        
        for (const auto& point : cluster) {
            avg_intensity += point[3];
            min_z = std::min(min_z, point[2]);
            max_z = std::max(max_z, point[2]);
            min_x = std::min(min_x, point[0]);
            max_x = std::max(max_x, point[0]);
            min_y = std::min(min_y, point[1]);
            max_y = std::max(max_y, point[1]);
        }
        avg_intensity /= cluster.size();
        
        RCLCPP_INFO(logger, "  Cluster %zu: %zu points, height: %.3fm, intensity: %.3f", 
                   i, cluster.size(), max_z - min_z, avg_intensity);
    }
}

Point3D ClusterProcessor::calculateConePosition(const Cluster& cluster) {
    double min_x = cluster[0][0], max_x = cluster[0][0];
    double min_y = cluster[0][1], max_y = cluster[0][1];
    
    for (const auto& point : cluster) {
        min_x = std::min(min_x, point[0]);
        max_x = std::max(max_x, point[0]);
        min_y = std::min(min_y, point[1]);
        max_y = std::max(max_y, point[1]);
    }

    double median_x = getMedian(cluster, 0);
    double median_y = getMedian(cluster, 1);

    // Weighted combination for robust position estimation using config weights
    double cone_x = config_.cone_position_w_median * median_x + config_.cone_position_w_min_x * (min_x + config_.cone_base_radius);
    double cone_y = config_.cone_position_w_median * median_y + config_.cone_position_w_min_y * (min_y + config_.cone_base_radius);

    return {cone_x, cone_y, config_.cone_height};
}

double ClusterProcessor::getMedian(const Cluster& points, size_t idx) const {
    if (points.empty()) return 0.0;

    std::vector<size_t> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0);

    size_t mid = indices.size() / 2;
    std::nth_element(indices.begin(), indices.begin() + mid, indices.end(),
        [&points, idx](size_t a, size_t b) { return points[a][idx] < points[b][idx]; });

    double median = points[indices[mid]][idx];

    if (indices.size() % 2 == 0 && mid > 0) {
        auto max_it = std::max_element(indices.begin(), indices.begin() + mid,
            [&points, idx](size_t a, size_t b) { return points[a][idx] < points[b][idx]; });
        median = 0.5 * (median + points[*max_it][idx]);
    }

    return median;
}

bool ClusterProcessor::isOrangeConeCandidate(const Cluster& cluster) const {
    if (cluster.size() < config_.orange_cone_min_points) {  // Now both are size_t
        return false;
    }

    // Calculate centroid distance
    double sum_x = 0.0, sum_y = 0.0;
    for (const auto& point : cluster) {
        sum_x += point[0];
        sum_y += point[1];
    }
    double centroid_x = sum_x / cluster.size();
    double centroid_y = sum_y / cluster.size();
    double distance = std::sqrt(centroid_x * centroid_x + centroid_y * centroid_y);

    return (distance > config_.orange_cone_distance_threshold);
}

std::vector<Point3D> ClusterProcessor::calculateClusterCenters(const std::vector<Cluster>& clusters) const {
    std::vector<Point3D> centers;
    centers.reserve(clusters.size());

    for (const auto& cluster : clusters) {
        if (cluster.empty()) continue;

        // Calculate centroid of the cluster
        double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
        for (const auto& point : cluster) {
            sum_x += point[0];
            sum_y += point[1];
            sum_z += point[2];
        }

        Point3D center = {
            sum_x / cluster.size(),
            sum_y / cluster.size(),
            sum_z / cluster.size()
        };
        centers.push_back(center);
    }

    return centers;
}

// =============================================
// POINT CLOUD EXTRACTOR IMPLEMENTATION
// =============================================

std::vector<Point4D> PointCloudExtractor::fromPointCloud(const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg) {
    if (!validatePointCloud(cloud_msg)) {
        return {};
    }

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

std::vector<Point4D> PointCloudExtractor::fromPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    if (!validatePointCloud2(cloud_msg)) {
        return {};
    }

    std::vector<Point4D> points;
    points.reserve(cloud_msg->width * cloud_msg->height);

    // Convert PointCloud2 to PCL format and then extract points (fixed implementation)
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*cloud_msg, pcl_pc2);
    
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::fromPCLPointCloud2(pcl_pc2, cloud);

    for (const auto& point : cloud.points) {
        Point4D pcl_point;
        pcl_point[0] = point.x;
        pcl_point[1] = point.y;
        pcl_point[2] = point.z;
        pcl_point[3] = point.intensity;
        points.push_back(pcl_point);
    }

    RCLCPP_INFO(rclcpp::get_logger("point_cloud_extractor"), 
               "Extracted %zu points from PointCloud2", 
               points.size());

    return points;
}

bool PointCloudExtractor::validatePointCloud(const sensor_msgs::msg::PointCloud::SharedPtr cloud_msg) {
    if (!cloud_msg) {
        RCLCPP_ERROR(rclcpp::get_logger("point_cloud_extractor"), "Null PointCloud message");
        return false;
    }
    if (cloud_msg->points.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("point_cloud_extractor"), "Empty PointCloud message");
        return false;
    }
    return true;
}

bool PointCloudExtractor::validatePointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    if (!cloud_msg) {
        RCLCPP_ERROR(rclcpp::get_logger("point_cloud_extractor"), "Null PointCloud2 message");
        return false;
    }
    if (cloud_msg->width * cloud_msg->height == 0) {
        RCLCPP_WARN(rclcpp::get_logger("point_cloud_extractor"), "Empty PointCloud2 message");
        return false;
    }
    return true;
}

// =============================================
// MAIN PROCESS LIDAR IMPLEMENTATION
// =============================================

ProcessLidar::ProcessLidar() : 
    Node("process_lidar"), 
    env_(ORT_LOGGING_LEVEL_WARNING, "ONNX_INFERENCE") {
    
    // Enable comprehensive logging
    auto debug_logger = this->get_logger();
    auto result = rcutils_logging_set_logger_level(debug_logger.get_name(), RCUTILS_LOG_SEVERITY_DEBUG);
    (void)result;

    // Load configuration first
    loadLidarConfig();

    // Initialize modular components
    initializeComponents();
    
    // Dual input subscribers for redundancy
    lidar_raw_input_sub_ = create_subscription<sensor_msgs::msg::PointCloud>(
        lidar_config_.lidar_raw_topic, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::PointCloud::SharedPtr msg) {
            RCLCPP_DEBUG(this->get_logger(), "Received PointCloud with %zu points", msg->points.size());
            lidarRawCallback(msg);
        });

    lidar_raw_input_sub2_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_config_.lidar_raw_topic2, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            RCLCPP_DEBUG(this->get_logger(), "Received PointCloud2 with %dx%d points", msg->width, msg->height);
            lidarRawCallback2(msg);
        });

    // Output publishers
    detected_cones_pub_ = create_publisher<dv_msgs::msg::IndexedTrack>("/perception/cones", 10);
    filtered_points_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/perception/filtered_points", 10);
    cluster_centers_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/perception/clusters", 10); // New publisher for cluster centers

    RCLCPP_INFO(get_logger(), "=== ENHANCED MODULAR LIDAR PROCESSING NODE STARTED ===");
    RCLCPP_INFO(get_logger(), "Using multi-stage ground removal: %s", use_enhanced_ground_removal_ ? "ENABLED" : "DISABLED");
    RCLCPP_INFO(get_logger(), "Dual input topics: %s, %s", 
                lidar_config_.lidar_raw_topic.c_str(), lidar_config_.lidar_raw_topic2.c_str());
    RCLCPP_INFO(get_logger(), "Output topic: /perception/cones");
    RCLCPP_INFO(get_logger(), "Filtered points visualization topic: /perception/filtered_points");
    RCLCPP_INFO(get_logger(), "Cluster centers visualization topic: /perception/clusters");
}

ProcessLidar::~ProcessLidar() {
    RCLCPP_INFO(get_logger(), "------------------------------------");
    RCLCPP_INFO(get_logger(), "--- Final Model Accuracy Report ---");

    long long tp_blue = g_true_positives_blue.load();
    long long tp_yellow = g_true_positives_yellow.load();
    long long fp_blue = g_false_positives_blue.load();
    long long fp_yellow = g_false_positives_yellow.load();
    long long rej_blue = g_rejected_as_blue.load();
    long long rej_yellow = g_rejected_as_yellow.load();

    long long total_actual_blue = tp_blue + fp_yellow + rej_blue;
    long long total_actual_yellow = tp_yellow + fp_blue + rej_yellow;
    long long total_clusters = total_actual_blue + total_actual_yellow;

    RCLCPP_INFO(get_logger(), "Total Blue/Yellow Clusters Encountered: %lld", total_clusters);
    RCLCPP_INFO(get_logger(), " ");

    if (total_actual_blue > 0) {
        RCLCPP_INFO(get_logger(), "--- BLUE CONES (Actual: %lld) ---", total_actual_blue);
        RCLCPP_INFO(get_logger(), "Correctly Detected (TP): %lld", tp_blue);
        RCLCPP_INFO(get_logger(), "Incorrectly Detected as Yellow (FN): %lld", fp_yellow);
        RCLCPP_INFO(get_logger(), "Rejected by Classifiers: %lld", rej_blue);
        double blue_recall = 100.0 * tp_blue / total_actual_blue;
        RCLCPP_INFO(get_logger(), "Recall (Sensitivity): %.2f%%", blue_recall);
        if ((tp_blue + fp_blue) > 0) {
            double blue_precision = 100.0 * tp_blue / (tp_blue + fp_blue);
            RCLCPP_INFO(get_logger(), "Precision: %.2f%%", blue_precision);
        }
    }
    
    RCLCPP_INFO(get_logger(), " ");

    if (total_actual_yellow > 0) {
        RCLCPP_INFO(get_logger(), "--- YELLOW CONES (Actual: %lld) ---", total_actual_yellow);
        RCLCPP_INFO(get_logger(), "Correctly Detected (TP): %lld", tp_yellow);
        RCLCPP_INFO(get_logger(), "Incorrectly Detected as Blue (FN): %lld", fp_blue);
        RCLCPP_INFO(get_logger(), "Rejected by Classifiers: %lld", rej_yellow);
        double yellow_recall = 100.0 * tp_yellow / total_actual_yellow;
        RCLCPP_INFO(get_logger(), "Recall (Sensitivity): %.2f%%", yellow_recall);
        if ((tp_yellow + fp_yellow) > 0) {
            double yellow_precision = 100.0 * tp_yellow / (tp_yellow + fp_yellow);
            RCLCPP_INFO(get_logger(), "Precision: %.2f%%", yellow_precision);
        }
    }
    
    RCLCPP_INFO(get_logger(), " ");

    if (total_clusters > 0) {
        long long total_correct = tp_blue + tp_yellow;
        double overall_accuracy = 100.0 * total_correct / total_clusters;
        RCLCPP_INFO(get_logger(), "Overall Accuracy (Correct / (TP+FN+Rejected)): %.2f%%", overall_accuracy);
    }
    
    RCLCPP_INFO(get_logger(), "------------------------------------");
    RCLCPP_INFO(get_logger(), "LiDAR Node shutdown complete.");
}

void ProcessLidar::loadLidarConfig() {
    try {
        // Get package share directory
        std::string package_share_directory = ament_index_cpp::get_package_share_directory("perception_winter");
        std::string config_path = package_share_directory + "/config/perception_config.yaml";
        
        // Load YAML config
        YAML::Node config = YAML::LoadFile(config_path);
        
        // Load lidar_only section-defaults for CarMaker
        auto lidar_config = config["lidar_only"]["carmaker"];
        
        // Topic Configuration
        lidar_config_.lidar_raw_topic = lidar_config["lidar_raw_topic"].as<std::string>();
        lidar_config_.lidar_raw_topic2 = lidar_config["lidar_raw_topic2"].as<std::string>();
        
        // Ground Removal Parameters (Optimized for RANSAC)
        lidar_config_.ransac_threshold = lidar_config["ransac_threshold"].as<double>();
        lidar_config_.min_z_normal_component = lidar_config["min_z_normal_component"].as<double>();
        lidar_config_.max_slope_deviation_deg = lidar_config["max_slope_deviation_deg"].as<double>();
        lidar_config_.max_ground_iterations = lidar_config["max_ground_iterations"].as<int>();
        lidar_config_.min_points_for_plane = lidar_config["min_points_for_plane"].as<size_t>();
        
        // Clustering Parameters (Optimized for DBSCAN)
        lidar_config_.dbscan_epsilon = lidar_config["dbscan_epsilon"].as<double>();
        lidar_config_.dbscan_minpoints = lidar_config["dbscan_minpoints"].as<int>();
        
        // Region of Interest (ROI) Boundaries
        lidar_config_.roi_y_min = lidar_config["roi_y_min"].as<double>();
        lidar_config_.roi_y_max = lidar_config["roi_y_max"].as<double>();
        lidar_config_.roi_z_min = lidar_config["roi_z_min"].as<double>();
        lidar_config_.roi_z_max = lidar_config["roi_z_max"].as<double>();
        lidar_config_.roi_x_max = lidar_config["roi_x_max"].as<double>();
        
        // Vehicle Body Exclusion Zone
        lidar_config_.car_front_x = lidar_config["car_front_x"].as<double>();
        lidar_config_.car_side_y = lidar_config["car_side_y"].as<double>();
        
        // Cone Physical Properties
        lidar_config_.cone_base_radius = lidar_config["cone_base_radius"].as<double>();
        lidar_config_.lidar_offset = lidar_config["lidar_offset"].as<double>();
        lidar_config_.cone_height = lidar_config["cone_height"].as<double>();
        
        // ML Model Configuration
        lidar_config_.num_bins = lidar_config["num_bins"].as<int>();
        lidar_config_.feature_size = lidar_config["feature_size"].as<int>();
        lidar_config_.z_min = lidar_config["z_min"].as<float>();
        lidar_config_.z_max = lidar_config["z_max"].as<float>();
        lidar_config_.bin_width = lidar_config["bin_width"].as<float>();
        lidar_config_.confidence_threshold = lidar_config["confidence_threshold"].as<double>();
        
        // Cluster Filtering Parameters (Optimized from second code)
        lidar_config_.min_cluster_height = lidar_config["min_cluster_height"].as<double>();
        lidar_config_.max_cluster_height = lidar_config["max_cluster_height"].as<double>();
        lidar_config_.max_cluster_width = lidar_config["max_cluster_width"].as<double>();
        lidar_config_.min_cluster_points = lidar_config["min_cluster_points"].as<size_t>();
        
        // Orange Cone Detection
        lidar_config_.orange_cone_distance_threshold = lidar_config["orange_cone_distance_threshold"].as<double>();
        lidar_config_.orange_cone_min_points = lidar_config["orange_cone_min_points"].as<size_t>();
        
        // Visualization Control
        lidar_config_.publish_cluster_centers = lidar_config["publish_cluster_centers"].as<bool>();
        lidar_config_.publish_filtered_points = lidar_config["publish_filtered_points"].as<bool>();
        
        // ONNX Model Configuration
        lidar_config_.onnx_model_paths.clear();
        for (const auto& path : lidar_config["onnx_model_paths"]) {
            lidar_config_.onnx_model_paths.push_back(path.as<std::string>());
        }
        lidar_config_.default_model_path = lidar_config["default_model_path"].as<std::string>();
        
        // Cone Detection Parameters
        lidar_config_.cone_distance_x_min = lidar_config["cone_distance_x_min"].as<double>();
        lidar_config_.cone_distance_x_max = lidar_config["cone_distance_x_max"].as<double>();
        lidar_config_.intensity_threshold_blue = lidar_config["intensity_threshold_blue"].as<double>();
        
        // Cone Position Calculation Weights
        lidar_config_.cone_position_w_median = lidar_config["cone_position_weights"]["w_median"].as<double>();
        lidar_config_.cone_position_w_min_x = lidar_config["cone_position_weights"]["w_min_x"].as<double>();
        lidar_config_.cone_position_w_min_y = lidar_config["cone_position_weights"]["w_min_y"].as<double>();
        
        // Heuristic Classifier Parameters
        lidar_config_.moving_average_factor = lidar_config["heuristic_classifier"]["moving_average_factor"].as<double>();
        lidar_config_.min_kernel_size = lidar_config["heuristic_classifier"]["min_kernel_size"].as<int>();
        
        // Enhanced ground removal parameters
        if (lidar_config["ground_removal"]) {
            auto ground_config = lidar_config["ground_removal"];
            lidar_config_.ground_removal.voxel_downsample_size = ground_config["voxel_downsample_size"].as<double>();
            lidar_config_.ground_removal.single_axis_smoothing_cell_size = ground_config["single_axis_smoothing_cell_size"].as<double>();
            lidar_config_.ground_removal.ground_smoothing_threshold = ground_config["ground_smoothing_threshold"].as<double>();
            lidar_config_.ground_removal.num_angular_sectors = ground_config["num_angular_sectors"].as<int>();
            lidar_config_.ground_removal.min_points_per_sector = ground_config["min_points_per_sector"].as<int>();
            lidar_config_.ground_removal.lowest_points_ratio = ground_config["lowest_points_ratio"].as<double>();
            lidar_config_.ground_removal.segments_per_ring = ground_config["segments_per_ring"].as<int>();
            lidar_config_.ground_removal.min_points_per_zone = ground_config["min_points_per_zone"].as<int>();
            lidar_config_.ground_removal.grid_cell_size = ground_config["grid_cell_size"].as<double>();
            lidar_config_.ground_removal.min_points_per_cell = ground_config["min_points_per_cell"].as<int>();
            lidar_config_.ground_removal.adaptive_variance_base = ground_config["adaptive_variance_base"].as<double>();
            lidar_config_.ground_removal.adaptive_variance_scale = ground_config["adaptive_variance_scale"].as<double>();
            lidar_config_.ground_removal.base_cylinder_radius = ground_config["base_cylinder_radius"].as<double>();
            lidar_config_.ground_removal.base_cylinder_height = ground_config["base_cylinder_height"].as<double>();
            lidar_config_.ground_removal.min_cone_cluster_points = ground_config["min_cone_cluster_points"].as<int>();
            lidar_config_.ground_removal.use_parallel_processing = ground_config["use_parallel_processing"].as<bool>();
            lidar_config_.ground_removal.max_threads = ground_config["max_threads"].as<int>();
            
            // Load concentric rings
            lidar_config_.ground_removal.concentric_rings.clear();
            for (const auto& ring : ground_config["concentric_rings"]) {
                lidar_config_.ground_removal.concentric_rings.push_back(
                    {ring[0].as<double>(), ring[1].as<double>()});
            }
            
            RCLCPP_INFO(get_logger(), "Enhanced ground removal parameters loaded successfully");
        } else {
            RCLCPP_WARN(get_logger(), "Enhanced ground removal parameters not found - using defaults");
        }
        
        RCLCPP_INFO(get_logger(), "LiDAR configuration loaded successfully from YAML");
    }
    catch (const std::exception& e) {
        RCLCPP_FATAL(get_logger(), "Failed to load LiDAR configuration: %s", e.what());
        rclcpp::shutdown();
    }
}

void ProcessLidar::initializeComponents() {
    try {
        ml_classifier_ = std::make_unique<ConeClassifier>(env_, lidar_config_);
        heuristic_classifier_ = std::make_unique<HeuristicClassifier>(lidar_config_);
        point_cloud_processor_ = std::make_unique<PointCloudProcessor>(lidar_config_);
        
        // Try to create enhanced processor, but handle potential failures
        try {
            enhanced_point_cloud_processor_ = std::make_unique<EnhancedPointCloudProcessor>(lidar_config_);
        } catch (const std::exception& e) {
            RCLCPP_WARN(get_logger(), "Enhanced ground removal unavailable: %s", e.what());
            use_enhanced_ground_removal_ = false;
        }
        
        cluster_processor_ = std::make_unique<ClusterProcessor>(lidar_config_);
        
        loadONNXModel();
        
        if (validateComponents()) {
            RCLCPP_INFO(get_logger(), "All modular components initialized successfully");
            RCLCPP_INFO(get_logger(), "Enhanced ground removal: %s", 
                       use_enhanced_ground_removal_ ? "ENABLED" : "DISABLED");
        } else {
            RCLCPP_FATAL(get_logger(), "Component validation failed");
            rclcpp::shutdown();
        }
    }
    catch (const std::exception& e) {
        RCLCPP_FATAL(get_logger(), "Component initialization failed: %s", e.what());
        rclcpp::shutdown();
    }
}

void ProcessLidar::loadONNXModel() {
    std::string package_share_directory;
    try {
        package_share_directory = ament_index_cpp::get_package_share_directory("perception_winter");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to get package share directory: %s", e.what());
        package_share_directory = ".";
    }
    
    // Use the model paths from configuration
    bool model_found = false;
    std::string model_path;
    
    for (const auto& search_path : lidar_config_.onnx_model_paths) {
        // Replace package placeholder with actual path
        std::string full_path = search_path;
        size_t pos = full_path.find("${PACKAGE_SHARE_DIR}");
        if (pos != std::string::npos) {
            full_path.replace(pos, 21, package_share_directory);
        }
        
        if (std::filesystem::exists(full_path)) {
            model_path = full_path;
            model_found = true;
            RCLCPP_INFO(get_logger(), "Found model at: %s", full_path.c_str());
            break;
        }
    }
    
    if (!model_found) {
        RCLCPP_ERROR(get_logger(), "ONNX model file not found in any search path");
        return;
    }
    
    if (ml_classifier_ && ml_classifier_->initialize(model_path)) {
        RCLCPP_INFO(get_logger(), "ONNX model loaded successfully: %s", model_path.c_str());
    } else {
        RCLCPP_ERROR(get_logger(), "Failed to initialize ONNX model: %s", model_path.c_str());
    }
}

bool ProcessLidar::validateComponents() const {
    bool all_valid = true;
    
    if (!ml_classifier_) {
        RCLCPP_ERROR(get_logger(), "ML Classifier component missing");
        all_valid = false;
    }
    if (!heuristic_classifier_) {
        RCLCPP_ERROR(get_logger(), "Heuristic Classifier component missing");
        all_valid = false;
    }
    if (!point_cloud_processor_) {
        RCLCPP_ERROR(get_logger(), "Point Cloud Processor component missing");
        all_valid = false;
    }
    if (!cluster_processor_) {
        RCLCPP_ERROR(get_logger(), "Cluster Processor component missing");
        all_valid = false;
    }
    
    // Enhanced component is optional - just log warning (no assignment in const method)
    if (!enhanced_point_cloud_processor_) {
        RCLCPP_WARN(get_logger(), "Enhanced Point Cloud Processor component missing - using fallback");
    }
    
    return all_valid;
}

void ProcessLidar::processPointCloudData(std::vector<Point4D>& points, const std_msgs::msg::Header& header) {
    (void)header; // Currently unused
    
    if (points.empty()) {
        RCLCPP_WARN(get_logger(), "Empty point cloud data received");
        return;
    }

    total_processed_frames_++;
    
    auto pipeline_start = std::chrono::steady_clock::now();
    bool success = executeProcessingPipeline(points);
    auto pipeline_end = std::chrono::steady_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(pipeline_end - pipeline_start);
    
    if (success) {
        publishProcessingMetrics(duration);
    } else {
        failed_processing_attempts_++;
        RCLCPP_WARN(get_logger(), "Processing pipeline failed for frame %lld", total_processed_frames_.load());
    }
}

bool ProcessLidar::executeProcessingPipeline(const std::vector<Point4D>& points) {
    // Use enhanced pipeline if available, otherwise fallback to original
    if (use_enhanced_ground_removal_ && enhanced_point_cloud_processor_) {
        return executeEnhancedProcessingPipeline(points);
    } else {
        // Fallback to original pipeline
        RCLCPP_WARN(get_logger(), "Using original ground removal pipeline (enhanced not available)");
        return executeOriginalProcessingPipeline(points);
    }
}

bool ProcessLidar::executeEnhancedProcessingPipeline(const std::vector<Point4D>& points) {
    try {
        RCLCPP_INFO(get_logger(), "=== STARTING ENHANCED PROCESSING PIPELINE ===");
        RCLCPP_INFO(get_logger(), "Input points: %zu", points.size());

        // Stage 1: Point Cloud Filtering (ORIGINAL)
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        if (!point_cloud_processor_->filterCarBodyAndROI(points, cloud, get_logger())) {
            RCLCPP_WARN(get_logger(), "Stage 1 failed: No points after filtering");
            return false;
        }

        // Stage 2: Enhanced Ground Removal (NEW)
        auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        if (!enhanced_point_cloud_processor_->removeGroundPlaneEnhanced(cloud, cloud_filtered, get_logger())) {
            RCLCPP_WARN(get_logger(), "Stage 2 failed: No points after enhanced ground removal");
            return false;
        }

        // Stage 3: Clustering
        auto clusters = cluster_processor_->clusterPoints(cloud_filtered, get_logger());
        if (clusters.empty()) {
            RCLCPP_WARN(get_logger(), "Stage 3 failed: No clusters found");
            if (lidar_config_.publish_filtered_points) {
                publishConeClusterPoints(clusters);
            }
            if (lidar_config_.publish_cluster_centers) {
                publishClusterCenters(clusters);
            }
            return true; // No clusters is not necessarily a failure
        }

        // Stage 4: Cluster Filtering
        std::vector<bool> orange_candidates;
        auto filtered_clusters = cluster_processor_->filterClustersBySize(clusters, orange_candidates, get_logger());

        // Publish cluster centers for visualization (before color classification)
        if (lidar_config_.publish_cluster_centers) {
            publishClusterCenters(filtered_clusters);
        }

        // Stage 5: Cone Detection
        std::vector<Point3D> cone_positions;
        std::vector<int> cone_colors;
        
        if (!filtered_clusters.empty()) {
            cluster_processor_->detectConesInClusters(filtered_clusters, orange_candidates, 
                                                     cone_positions, cone_colors,
                                                     *ml_classifier_, *heuristic_classifier_, get_logger());

            // Stage 6: Publish Results
            publishDetectedCones(cone_positions, cone_colors);
        } else {
            RCLCPP_WARN(get_logger(), "Stage 5: No valid clusters after filtering");
            publishDetectedCones({}, {});
        }
        
        // Always publish clustered points for visualization if enabled
        // This is crucial for the FilteredPointsVisualNode to work
        if (lidar_config_.publish_filtered_points) {
            publishConeClusterPoints(clusters);
        }

        RCLCPP_INFO(get_logger(), "=== ENHANCED PROCESSING PIPELINE COMPLETED SUCCESSFULLY ===");
        return true;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Enhanced processing pipeline exception: %s", e.what());
        return false;
    }
}

bool ProcessLidar::executeOriginalProcessingPipeline(const std::vector<Point4D>& points) {
    try {
        RCLCPP_INFO(get_logger(), "=== STARTING ORIGINAL PROCESSING PIPELINE ===");
        RCLCPP_INFO(get_logger(), "Input points: %zu", points.size());

        // Stage 1: Point Cloud Filtering
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        if (!point_cloud_processor_->filterCarBodyAndROI(points, cloud, get_logger())) {
            RCLCPP_WARN(get_logger(), "Stage 1 failed: No points after filtering");
            return false;
        }

        // Stage 2: Ground Removal (ORIGINAL RANSAC)
        auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        if (!point_cloud_processor_->removeGroundPlane(cloud, cloud_filtered, get_logger())) {
            RCLCPP_WARN(get_logger(), "Stage 2 failed: No points after ground removal");
            return false;
        }

        // Stage 3: Clustering
        auto clusters = cluster_processor_->clusterPoints(cloud_filtered, get_logger());
        if (clusters.empty()) {
            RCLCPP_WARN(get_logger(), "Stage 3 failed: No clusters found");
            if (lidar_config_.publish_filtered_points) {
                publishConeClusterPoints(clusters);
            }
            if (lidar_config_.publish_cluster_centers) {
                publishClusterCenters(clusters);
            }
            return true; // No clusters is not necessarily a failure
        }

        // Stage 4: Cluster Filtering
        std::vector<bool> orange_candidates;
        auto filtered_clusters = cluster_processor_->filterClustersBySize(clusters, orange_candidates, get_logger());

        // Publish cluster centers for visualization (before color classification)
        if (lidar_config_.publish_cluster_centers) {
            publishClusterCenters(filtered_clusters);
        }

        // Stage 5: Cone Detection
        std::vector<Point3D> cone_positions;
        std::vector<int> cone_colors;
        
        if (!filtered_clusters.empty()) {
            cluster_processor_->detectConesInClusters(filtered_clusters, orange_candidates, 
                                                     cone_positions, cone_colors,
                                                     *ml_classifier_, *heuristic_classifier_, get_logger());

            // Stage 6: Publish Results
            publishDetectedCones(cone_positions, cone_colors);
        } else {
            RCLCPP_WARN(get_logger(), "Stage 5: No valid clusters after filtering");
            publishDetectedCones({}, {});
        }
        
        // Always publish clustered points for visualization if enabled
        // This is crucial for the FilteredPointsVisualNode to work
        if (lidar_config_.publish_filtered_points) {
            publishConeClusterPoints(clusters);
        }

        RCLCPP_INFO(get_logger(), "=== ORIGINAL PROCESSING PIPELINE COMPLETED SUCCESSFULLY ===");
        return true;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Original processing pipeline exception: %s", e.what());
        return false;
    }
}

void ProcessLidar::publishDetectedCones(const std::vector<Point3D>& positions, const std::vector<int>& colors) {
    dv_msgs::msg::IndexedTrack track_msg;

    int yellow_count = 0;
    int blue_count = 0;
    int orange_count = 0;

    for (size_t i = 0; i < positions.size(); ++i) {
        double x = positions[i][0];
        double y = positions[i][1];
        double z = positions[i][2];

        // Distance and angle validation using config values
        if (x < lidar_config_.cone_distance_x_min || x > lidar_config_.cone_distance_x_max) continue;
        
        double range = sqrt(x * x + y * y);
        double angle = atan2(y, x);
        
        dv_msgs::msg::IndexedCone cone_msg;
        cone_msg.location.x = range;
        cone_msg.location.y = angle;
        cone_msg.location.z = z;
        cone_msg.color = colors[i];
        cone_msg.index = i;
        track_msg.track.push_back(cone_msg);

        // Color counting for diagnostics
        if (colors[i] == dv_msgs::msg::IndexedCone::YELLOW) {
            yellow_count++;
        } else if (colors[i] == dv_msgs::msg::IndexedCone::BLUE) {
            blue_count++;
        } else if (colors[i] == dv_msgs::msg::IndexedCone::ORANGE_BIG) {
            orange_count++;
        }
    }
    
    if (!track_msg.track.empty()) {
        detected_cones_pub_->publish(track_msg);
        RCLCPP_DEBUG(get_logger(), "Published %zu cones to /perception/cones", track_msg.track.size());
    }

    RCLCPP_INFO(get_logger(), "Detected cones - Yellow: %d, Blue: %d, Orange: %d", 
                yellow_count, blue_count, orange_count);
}

void ProcessLidar::publishConeClusterPoints(const std::vector<Cluster>& cone_clusters) {
    if (!filtered_points_pub_) {
        RCLCPP_WARN(get_logger(), "Filtered points publisher not available");
        return;
    }

    auto message = std_msgs::msg::Float32MultiArray();
    
    // Publish all cluster points for visualization
    // Format: [x1, y1, z1, x2, y2, z2, ...] for compatibility with FilteredPointsVisualNode
    size_t total_points = 0;
    for (const auto& cluster : cone_clusters) {
        total_points += cluster.size();
        for (const auto& point : cluster) {
            message.data.push_back(static_cast<float>(point[0])); // x
            message.data.push_back(static_cast<float>(point[1])); // y  
            message.data.push_back(static_cast<float>(point[2])); // z
            // Note: We're not including intensity to match the visualizer's expectation of 3 floats per point
        }
    }
    
    filtered_points_pub_->publish(message);
    
    RCLCPP_DEBUG(get_logger(), "Published %zu clustered points (from %zu clusters) for visualization", 
                 total_points, cone_clusters.size());
}

void ProcessLidar::publishClusterCenters(const std::vector<Cluster>& filtered_clusters) {
    if (!cluster_centers_pub_) {
        RCLCPP_WARN(get_logger(), "Cluster centers publisher not available");
        return;
    }

    auto message = std_msgs::msg::Float32MultiArray();
    
    // Calculate cluster centers using the new method
    auto cluster_centers = cluster_processor_->calculateClusterCenters(filtered_clusters);
    
    // Publish cluster centers in format [x1, y1, x2, y2, ...] for compatibility with Python visualizer
    for (const auto& center : cluster_centers) {
        message.data.push_back(static_cast<float>(center[0])); // x
        message.data.push_back(static_cast<float>(center[1])); // y
        // Note: We're only publishing x and y coordinates to match the Python visualizer expectation
    }
    
    cluster_centers_pub_->publish(message);
    
    RCLCPP_DEBUG(get_logger(), "Published %zu cluster centers for visualization", 
                 cluster_centers.size());
}

void ProcessLidar::publishProcessingMetrics(const std::chrono::milliseconds& duration) {
    RCLCPP_INFO(get_logger(), "Frame processing time: %ld ms", duration.count());
}

void ProcessLidar::lidarRawCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg) {
    auto start_time = std::chrono::steady_clock::now();

    try {
        auto points = PointCloudExtractor::fromPointCloud(msg);
        processPointCloudData(points, msg->header);
    } 
    catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "PointCloud processing error: %s", e.what());
        failed_processing_attempts_++;
    }
    
    auto end_time = std::chrono::steady_clock::now();  
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    RCLCPP_DEBUG(get_logger(), "PointCloud callback completed in %ld ms", duration.count());
}

void ProcessLidar::lidarRawCallback2(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto start_time = std::chrono::steady_clock::now();

    try {
        auto points = PointCloudExtractor::fromPointCloud2(msg);
        processPointCloudData(points, msg->header);
    } 
    catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "PointCloud2 processing error: %s", e.what());
        failed_processing_attempts_++;
    }
    
    auto end_time = std::chrono::steady_clock::now();  
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    RCLCPP_DEBUG(get_logger(), "PointCloud2 callback completed in %ld ms", duration.count());
}