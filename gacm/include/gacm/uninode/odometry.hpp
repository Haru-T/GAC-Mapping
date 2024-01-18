#ifndef GACM__UNINODE__ODOMETRY_HPP_
#define GACM__UNINODE__ODOMETRY_HPP_

#include <cstdint>
#include <memory>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/core/mat.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "gacm/featureDefinition.h"

namespace ros {
class NodeHandle;
}

namespace gacm {
using PointType = pcl::PointXYZI;
struct OdometryInput {
  cv::Mat feature_image;
  pcl::PointCloud<ImagePoint>::Ptr keypoints;
  pcl::PointCloud<PointType>::Ptr laser_cloud_sharp;
  pcl::PointCloud<PointType>::Ptr laser_cloud_less_sharp;
  pcl::PointCloud<PointType>::Ptr laser_cloud_flat;
  pcl::PointCloud<PointType>::Ptr laser_cloud_less_flat;
  pcl::PointCloud<PointType>::Ptr laser_full;
  struct {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
  } pose;
  int64_t timestamp_nanoseconds;
};

struct OdometryOutput {
  pcl::PointCloud<PointType>::Ptr laser_cloud_corner_last;
  pcl::PointCloud<PointType>::Ptr laser_cloud_surf_last;
  pcl::PointCloud<PointType>::Ptr laser_full_3;
  struct {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
  } laser_odom_to_init;
};

struct OdometryObject;
std::shared_ptr<OdometryObject>
spawn_odometry_object(const std::string & calib_file, ros::NodeHandle *nh_ptr = nullptr);

void new_session(const std::shared_ptr<OdometryObject> &obj, const std::string & calib_file);

OdometryOutput odometry(OdometryInput &&input,
                        const std::shared_ptr<OdometryObject> &obj);
} // namespace gacm

#endif // GACM__UNINODE__ODOMETRY_HPP_
