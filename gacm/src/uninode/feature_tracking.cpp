#include "gacm/uninode/feature_tracking.hpp"

#include <memory>
#include <utility>

#include "ros/ros.h"
#include "sensor_msgs/Image.h"

#include "gacm/featureExtractor.h"
#include "gacm/scanRegistrator.h"

namespace gacm
{
struct FeatureTrackingObject
  : std::enable_shared_from_this<FeatureTrackingObject>
{
  FeatureTracker feature_tracker;
  ScanRegistrator scan_registrator;
  std::shared_ptr<ros::Publisher> pub_feature_image_with_text;

  FeatureTrackingObject(std::shared_ptr<ros::Publisher> pub_feature_image_with_text)
  : feature_tracker(), scan_registrator(),
    pub_feature_image_with_text(std::move(pub_feature_image_with_text)) {}

  FeatureTrackingOutput track_feature(FeatureTrackingInput && input);
};

std::shared_ptr<FeatureTrackingObject>
spawn_feature_tracking_object(const std::string & calib_file, ros::NodeHandle * nh_ptr)
{
  auto ptr = std::make_shared<FeatureTrackingObject>(
    nh_ptr != nullptr ? std::make_shared<ros::Publisher>(
      nh_ptr->advertise<sensor_msgs::Image>(
        "feature_image_with_text", 10)) :
    nullptr);
  ptr->feature_tracker.readIntrinsicParameter(calib_file);
  ptr->scan_registrator.readIntrinsicParameter(calib_file);
  return ptr;
}

FeatureTrackingOutput
track_feature(
  FeatureTrackingInput && input,
  const std::shared_ptr<FeatureTrackingObject> & obj)
{
  return obj->track_feature(std::move(input));
}

void new_session(const std::shared_ptr<FeatureTrackingObject> & obj, const std::string & calib_file)
{
  obj->feature_tracker = FeatureTracker();
  obj->feature_tracker.readIntrinsicParameter(calib_file);
  obj->scan_registrator = ScanRegistrator();
  obj->scan_registrator.readIntrinsicParameter(calib_file);
}

FeatureTrackingOutput FeatureTrackingObject::track_feature(
  FeatureTrackingInput && input
)
{
  feature_tracker.imageDataCallback(
    input.image,
    input.timestamp_nanoseconds * 1e-9);
  if (pub_feature_image_with_text != nullptr) {
    cv::Mat feature_image_text = feature_tracker.getKeyPointImageLast();
    cv_bridge::CvImage bridge;
    bridge.header.stamp =
      ros::Time(
      input.timestamp_nanoseconds / 1'000'000'000,
      input.timestamp_nanoseconds % 1'000'000'000);
    bridge.header.frame_id = "camera";
    bridge.image = feature_image_text;
    bridge.encoding = "bgr8";
    sensor_msgs::Image::Ptr feature_image_ptr = bridge.toImageMsg();
    pub_feature_image_with_text->publish(feature_image_ptr);
  }
  scan_registrator.laserCloudHandler(input.point_cloud);

  return {
    .feature_image = std::move(input.image),
    .feature_points = feature_tracker.getPointToPub(),
    .laser_full = scan_registrator.getFullResCloud(),
    .laser_cloud_sharp = scan_registrator.getCornerPointsSharp(),
    .laser_cloud_less_sharp =
      scan_registrator.getCornerPointsLessSharp(),
    .laser_cloud_flat = scan_registrator.getSurfPointsFlat(),
    .laser_cloud_less_flat = scan_registrator.getSurfPointsLessFlat(),
  };
}

} // namespace gacm
