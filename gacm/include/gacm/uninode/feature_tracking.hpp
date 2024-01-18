#ifndef GACM__UNINODE__FEATURE_TRACKING_HPP_
#define GACM__UNINODE__FEATURE_TRACKING_HPP_

#include <cstdint>
#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "gacm/featureDefinition.h"

namespace ros {
class NodeHandle;
}

namespace gacm {
using PointType = pcl::PointXYZI;

struct FeatureTrackingInput {
  cv::Mat image;
  pcl::PointCloud<PointType>::Ptr point_cloud;
  int64_t timestamp_nanoseconds;
};

struct FeatureTrackingOutput {
  cv::Mat feature_image;                 // pub_image
  pcl::PointCloud<ImagePoint>::Ptr feature_points;        // pub_keypoints
  pcl::PointCloud<PointType>::Ptr laser_full;             // pubFullResPoints
  pcl::PointCloud<PointType>::Ptr laser_cloud_sharp;      // pubCornerPointsSharp
  pcl::PointCloud<PointType>::Ptr laser_cloud_less_sharp; // pubCornerPointsLessSharp
  pcl::PointCloud<PointType>::Ptr laser_cloud_flat;       // pubSurfPointsFlat
  pcl::PointCloud<PointType>::Ptr laser_cloud_less_flat;  // pubSurfPointsLessFlat
};

struct FeatureTrackingObject;
std::shared_ptr<FeatureTrackingObject>
spawn_feature_tracking_object(const std::string & calib_file, ros::NodeHandle *nh_ptr = nullptr);

void new_session(const std::shared_ptr<FeatureTrackingObject> & obj, const std::string & calib_file);

FeatureTrackingOutput track_feature(FeatureTrackingInput &&input,
                                    const std::shared_ptr<FeatureTrackingObject> & obj);
} // namespace gacm

#endif // GACM__UNINODE__FEATURE_TRACKING_HPP_
