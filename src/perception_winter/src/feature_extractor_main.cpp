// Create this new file at src/feature_extractor_main.cpp
#include "perception_winter/feature_extractor.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FeatureExtractorNode>());
  rclcpp::shutdown();
  return 0;
}