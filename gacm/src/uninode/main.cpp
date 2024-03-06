#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

#include "ros/ros.h"

#include "gacm/parameters.h"
#include "gacm/uninode/feature_tracking.hpp"
#include "gacm/uninode/file_enumeration.hpp"
#include "gacm/uninode/odometry.hpp"
#include "gacm/uninode/submap_management.hpp"

int main(int argc, char * argv[])
{
  ros::init(argc, argv, "gacm_uninode");
  ros::NodeHandle nh_p("~");
  ros::NodeHandle nh;
  readParameters(nh_p);

  std::vector<std::string> left_image_directories;
  std::vector<std::string> point_cloud_directories;
  std::vector<std::string> odometry_files;
  nh_p.getParam("left_image_directories", left_image_directories);
  nh_p.getParam("point_cloud_directories", point_cloud_directories);
  nh_p.getParam("odometry_files", odometry_files);
  if (left_image_directories.size() != point_cloud_directories.size()) {
    std::fputs("input size does not match\n", stderr);
    return 1;
  }
  ROS_INFO("#robots = %zu", left_image_directories.size());

  std::filesystem::path gacm_output_dir =
    std::filesystem::path(std::getenv("HOME")) / "gacm_output";
  std::filesystem::path data_dir = gacm_output_dir / "data";
  std::filesystem::create_directories(data_dir / "testSaveMap" / "fullcloud");
  std::filesystem::create_directories(data_dir / "air_experiment");
  std::filesystem::create_directories(gacm_output_dir / "timecost");
  std::filesystem::create_directories(
    gacm_output_dir / "pictures" /
    "submap_img");
  std::filesystem::create_directories(
    gacm_output_dir / "pictures" /
    "eachframe");
  std::filesystem::create_directories(gacm_output_dir / "cache");

  auto feature_tracking = gacm::spawn_feature_tracking_object(CAM_NAME, &nh);
  auto odometry = gacm::spawn_odometry_object(CAM_NAME, &nh);
  auto submap_management = gacm::spawn_submap_management_object(nh);

  for (size_t i = 0; i < left_image_directories.size(); ++i) {
    // enumerate input files
    std::vector<std::filesystem::path> left_image_files = gacm::enumerate_file(
      std::filesystem::directory_iterator(left_image_directories[i]), ".png");
    std::vector<std::filesystem::path> point_cloud_files = gacm::enumerate_file(
      std::filesystem::directory_iterator(point_cloud_directories[i]),
      ".pcd");
    bool estimate_odometry = odometry_files[i].empty();
    std::vector<gacm::Odom> odometry_list;
    if (!estimate_odometry) {
      odometry_list = gacm::load_odometry(odometry_files[i]);
    }
    ROS_INFO(
      R"(==============================
Robot #%zu:
  images:   %s (%zu frames)
  points:   %s (%zu frames)
  odometry: %s (%zu frames)
------------------------------
)",
      i,
      left_image_directories[i].c_str(), left_image_files.size(),
      point_cloud_directories[i].c_str(), point_cloud_files.size(),
      estimate_odometry ? "<estimate>" : odometry_files[i].c_str(),
      odometry_list.size());

    auto left_image_it = left_image_files.cbegin();
    auto point_cloud_it = point_cloud_files.cbegin();
    auto odometry_it = odometry_list.cbegin();

    while (left_image_it != left_image_files.cend() &&
      point_cloud_it != point_cloud_files.cend() &&
      (estimate_odometry || odometry_it != odometry_list.cend()))
    {
      int64_t left_image_timestamp =
        gacm::filename_to_timestamp(*left_image_it);
      int64_t point_cloud_timestamp =
        gacm::filename_to_timestamp(*point_cloud_it);
      int64_t odometry_timestamp = estimate_odometry ?
        std::numeric_limits<int64_t>::max() :
        odometry_it->timestamp_nanoseconds;
      if (std::abs(left_image_timestamp - point_cloud_timestamp) < 1'000'000 &&
        (estimate_odometry ||
        (std::abs(point_cloud_timestamp - odometry_timestamp) < 1'000'000 &&
        std::abs(odometry_timestamp - point_cloud_timestamp) <
        1'000'000)))
      {
        // data is synchronized, run
        pcl::PointCloud<gacm::PointType>::Ptr point_cloud(
          new pcl::PointCloud<gacm::PointType>());
        pcl::io::loadPCDFile(*point_cloud_it, *point_cloud);
        ROS_INFO("timestamp=%" PRId64, left_image_timestamp);
        gacm::FeatureTrackingInput feature_tracking_input{
          .image = cv::imread(*left_image_it, cv::IMREAD_COLOR),
          .point_cloud = std::move(point_cloud),
          .timestamp_nanoseconds = left_image_timestamp,
        };
        gacm::FeatureTrackingOutput feature_tracking_output =
          gacm::track_feature(
          std::move(feature_tracking_input),
          feature_tracking);

        Eigen::Vector3d position(0.0, 0.0, 0.0);
        Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
        if (!estimate_odometry) {
          Eigen::Transform<double, 3, Eigen::TransformTraits::Isometry> Twb;
          Twb.linear() = odometry_it->pose.orientation.toRotationMatrix();
          Twb.translation() = odometry_it->pose.position;

          // TODO load this from the configuration file
          Eigen::Transform<double, 3, Eigen::TransformTraits::Isometry> Tbc;
          Tbc.matrix() << -0.99995218, -0.00522539, -0.00826584,  0.21743355,
                          -0.00827815,  0.00233898,  0.999963  ,  0.04673507,
                          -0.00520586,  0.99998361, -0.00238212, -0.01191991,
                           0.        ,  0.        ,  0.        ,  1.;
          Eigen::Transform<double, 3, Eigen::TransformTraits::Isometry> Twc
            = Twb * Tbc;
          position = Twc.translation();
          orientation = Twc.rotation();
        }

        gacm::OdometryInput odometry_input{
          .feature_image = std::move(feature_tracking_output.feature_image),
          .keypoints = std::move(feature_tracking_output.feature_points),
          .laser_cloud_sharp =
            std::move(feature_tracking_output.laser_cloud_sharp),
          .laser_cloud_less_sharp =
            std::move(feature_tracking_output.laser_cloud_less_sharp),
          .laser_cloud_flat =
            std::move(feature_tracking_output.laser_cloud_flat),
          .laser_cloud_less_flat =
            std::move(feature_tracking_output.laser_cloud_less_flat),
          .laser_full = std::move(feature_tracking_output.laser_full),
          .pose =
          {
            .position = position,
            .orientation = orientation,
          },
          .timestamp_nanoseconds = left_image_timestamp,
        };
        gacm::OdometryOutput odometry_output =
          gacm::odometry(std::move(odometry_input), odometry);

        gacm::SubMapManagementInput submap_management_input{
          .laser_cloud_corner_last =
            std::move(odometry_output.laser_cloud_corner_last),
          .laser_cloud_surf_last =
            std::move(odometry_output.laser_cloud_surf_last),
          .laser_full_3 = std::move(odometry_output.laser_full_3),
          .laser_odom_to_init =
          {
            .position =
              odometry_output.laser_odom_to_init.position,
            .orientation =
              odometry_output.laser_odom_to_init.orientation,
          },
          .timestamp_nanoseconds = left_image_timestamp,
        };
        gacm::manage_submap(
          std::move(submap_management_input),
          submap_management);

        ++left_image_it;
        ++point_cloud_it;
        if (!estimate_odometry) {
          ++odometry_it;
        }
        ros::spinOnce();
      } else {
        // data is not synchronized
        std::array<int64_t, 3> timestamps{
          left_image_timestamp, point_cloud_timestamp, odometry_timestamp};
        auto * it = std::min_element(timestamps.begin(), timestamps.end());
        if (it - timestamps.begin() == 0) {
          ++left_image_it;
        } else if (it - timestamps.begin() == 1) {
          ++point_cloud_it;
        } else {
          ++odometry_it;
        }
      }
    }

    // finish one sequence
    ++CONFIG_ID;
    readParameters(nh_p);
    gacm::new_session(feature_tracking, CAM_NAME);
    gacm::new_session(odometry, CAM_NAME);
    gacm::new_session(submap_management);
  }
  ROS_INFO("All sequences finished");
  ros::shutdown();
  return 0;
}
