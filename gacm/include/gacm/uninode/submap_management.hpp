#ifndef GACM__UNINODE__SUBMAP_MANAGEMENT_HPP_
#define GACM__UNINODE__SUBMAP_MANAGEMENT_HPP_

#include <cstdint>
#include <memory>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace ros {
class NodeHandle;
}

namespace gacm {
using PointType = pcl::PointXYZI;

struct SubMapManagementInput {
  pcl::PointCloud<PointType>::Ptr laser_cloud_corner_last;
  pcl::PointCloud<PointType>::Ptr laser_cloud_surf_last;
  pcl::PointCloud<PointType>::Ptr laser_full_3;
  struct {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
  } laser_odom_to_init;
  int64_t timestamp_nanoseconds;
};

struct SubMapManagementObject;
std::shared_ptr<SubMapManagementObject>
spawn_submap_management_object(ros::NodeHandle &nh);

void new_session(const std::shared_ptr<SubMapManagementObject> & obj);

void manage_submap(SubMapManagementInput &&input,
                   const std::shared_ptr<SubMapManagementObject> &obj);
} // namespace gacm

#endif // GACM__UNINODE__SUBMAP_MANAGEMENT_HPP_
