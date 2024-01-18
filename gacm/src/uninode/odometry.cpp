#include "gacm/uninode/odometry.hpp"

#include <memory>

#include "nav_msgs/Path.h"

#include "gacm/poseEstimator.h"

namespace gacm
{
struct OdometryObject : std::enable_shared_from_this<OdometryObject>
{
  std::shared_ptr<PoseEstimator> pose_estimator;
  std::shared_ptr<ros::Publisher> pubLaserPath;
  std::shared_ptr<ros::Publisher> pubLaserPath2;
  std::shared_ptr<ros::Publisher> pubPointFeatures;
  std::shared_ptr<ros::Publisher> pubDepthImageRGB;

  OdometryObject(
    std::shared_ptr<PoseEstimator> pose_estimator,
    std::shared_ptr<ros::Publisher> pubLaserPath,
    std::shared_ptr<ros::Publisher> pubLaserPath2,
    std::shared_ptr<ros::Publisher> pubPointFeatures,
    std::shared_ptr<ros::Publisher> pubDepthImageRGB)
  : pose_estimator(std::move(pose_estimator)),
    pubLaserPath(std::move(pubLaserPath)),
    pubLaserPath2(std::move(pubLaserPath2)),
    pubPointFeatures(std::move(pubPointFeatures)),
    pubDepthImageRGB(std::move(pubDepthImageRGB)) {}

  OdometryOutput odometry(OdometryInput && input);
};

std::shared_ptr<OdometryObject>
spawn_odometry_object(const std::string & calib_file, ros::NodeHandle * nh_ptr)
{
  return std::make_shared<OdometryObject>(
    std::make_shared<PoseEstimator>(calib_file),
    nh_ptr != nullptr ?
    std::make_shared<ros::Publisher>(
      nh_ptr->advertise<nav_msgs::Path>("laser_odom_path", 10)) :
    nullptr,
    nh_ptr != nullptr ?
    std::make_shared<ros::Publisher>(
      nh_ptr->advertise<nav_msgs::Path>("laser_odom_path2", 10)) :
    nullptr,
    nh_ptr != nullptr ?
    std::make_shared<ros::Publisher>(
      nh_ptr->advertise<sensor_msgs::PointCloud2>("point_3d", 10)) :
    nullptr,
    nh_ptr != nullptr ?
    std::make_shared<ros::Publisher>(
      nh_ptr->advertise<sensor_msgs::Image>("depth_rgb", 10)) :
    nullptr);
}

void new_session(const std::shared_ptr<OdometryObject> & obj, const std::string & calib_file)
{
  obj->pose_estimator = std::make_shared<PoseEstimator>(calib_file);
}

OdometryOutput odometry(
  OdometryInput && input,
  const std::shared_ptr<OdometryObject> & obj)
{
  return obj->odometry(std::move(input));
}

OdometryOutput OdometryObject::odometry(OdometryInput && input)
{
  pose_estimator->handleImage(
    input.feature_image, input.keypoints, input.laser_cloud_sharp,
    input.laser_cloud_less_sharp, input.laser_cloud_flat,
    input.laser_cloud_less_flat, input.laser_full, input.pose.position,
    input.pose.orientation, input.timestamp_nanoseconds);
  if (pubLaserPath != nullptr) {
    pubLaserPath->publish(pose_estimator->getLaserPath());
  }
  if (pubLaserPath2 != nullptr) {
    pubLaserPath2->publish(pose_estimator->getLaserPath2());
  }
  if (pubPointFeatures != nullptr) {
    pubPointFeatures->publish(pose_estimator->getMapPointCloud());
  }
  if (pubDepthImageRGB != nullptr) {
    pubDepthImageRGB->publish(pose_estimator->getDepthImageRGB());
  }

  nav_msgs::Odometry odom = pose_estimator->getLaserOdometry();

  return {
    .laser_cloud_corner_last = input.laser_cloud_less_sharp,
    .laser_cloud_surf_last = input.laser_cloud_less_flat,
    .laser_full_3 = input.laser_full,
    .laser_odom_to_init =
    {.position = Eigen::Vector3d(
        odom.pose.pose.position.x,
        odom.pose.pose.position.y,
        odom.pose.pose.position.z),
      .orientation = Eigen::Quaterniond(
        odom.pose.pose.orientation.w, odom.pose.pose.orientation.x,
        odom.pose.pose.orientation.y, odom.pose.pose.orientation.z)},
  };
}
} // namespace gacm
