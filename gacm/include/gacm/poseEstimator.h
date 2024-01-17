/**
 * This file is part of GAC-Mapping.
 *
 * Copyright (C) 2020-2022 JinHao He, Yilin Zhu / RAPID Lab, Sun Yat-Sen
 * University
 *
 * For more information see <https://github.com/SYSU-RoboticsLab/GAC-Mapping>
 *
 * GAC-Mapping is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the license, or
 * (at your option) any later version.
 *
 * GAC-Mapping is distributed to support research and development of
 * Ground-Aerial heterogeneous multi-agent system, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. In no event will the authors be held liable for any
 * damages arising from the use of this software. See the GNU General Public
 * License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GAC-Mapping. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef GACM__POSE_ESTIMATOR_H_
#define GACM__POSE_ESTIMATOR_H_

#include <algorithm>
#include <functional>
#include <map>
#include <mutex>
#include <queue>
#include <vector>
#include <cstdint>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/mat.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "camodocal/camera_models/Camera.h"
#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/image_encodings.h"

#include "gacm/featureDefinition.h"
#include "gacm/parameters.h"

typedef pcl::PointXYZI PointType;

class PoseEstimator
{

private:
  camodocal::CameraPtr m_camera;
  std::vector<double> m_intrinsic_params;

  // image features
  cv::Mat rgb_cur, rgb_last;
  pcl::PointCloud<ImagePoint>::Ptr imagePointsCur, imagePointsLast;

  // full lidar cloud and enhanced depth map
  pcl::PointCloud<PointType>::Ptr laserCloudFullResCur;
  pcl::PointCloud<PointType>::Ptr laserCloudFullResLast;
  cv::Mat depth_map;
  cv::Mat laser_map;

  // lidar features
  pcl::PointCloud<PointType>::Ptr cornerPointsSharp;
  pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
  pcl::PointCloud<PointType>::Ptr surfPointsFlat;
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;

  pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;

  // KD Tree to search
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast;

  // from feature extractor
  std::queue<sensor_msgs::ImageConstPtr> imageRawBuf;
  std::queue<sensor_msgs::PointCloud2ConstPtr> featurePointsBuf;
  // from scan registrator
  std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
  std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
  std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
  std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
  std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
  std::queue<geometry_msgs::PoseStampedConstPtr> poseStampedBuf;
  std::mutex mBuf; // beacause there is only one access thread , a single mutex
                   // is adequate

  // std::mutex mutex_pointmap;

  double timeImageRaw;
  double timeFeaturePoints;

  double timeCornerPointsSharp;
  double timeCornerPointsLessSharp;
  double timeSurfPointsFlat;
  double timeSurfPointsLessFlat;
  double timeLaserCloudFullRes;

  double timePoseStamped;

  int skipFrameNum;
  int laserFrameCount;
  int laserFrameCountFromBegin;
  // int lastOptimizeFrame;
  int frameTobeMap;
  bool systemInited;
  bool need_pub_odom;
  bool need_pub_cloud;
  bool mapping_ready;
  bool lidar_only;

  int corner_correspondence;
  int plane_correspondence;
  int point_correspondence;

  int laserCloudCornerLastNum;
  int laserCloudSurfLastNum;

  int imagePointsLastNum, imagePointsCurNum;

  // Transformation from current frame to world frame
  Eigen::Quaterniond q_w_curr;       // (1, 0, 0, 0);
  Eigen::Vector3d t_w_curr;          // (0, 0, 0);
  Eigen::Quaterniond q_w_curr_hfreq; // (1, 0, 0, 0);
  Eigen::Vector3d t_w_curr_hfreq;    // (0, 0, 0);

  Eigen::Matrix<double, 7, 1> se3_last_curr; // x, y, z, w,; x, y, z
  Eigen::Matrix<double, 7, 1>
  se3_last_curr_hfreq;     // image high frequency odometry
  std::vector<PosePerFrame, Eigen::aligned_allocator<PosePerFrame>>
  poseSequence;     // (frameid, pose)
  std::deque<MappingDataPerFrame>
  mapDataPerFrameBuf;     // buffer storing feature ids in each frame

  // global point feature map
  // std::map<int, PointFeaturePerId> globalFeatureMap;
  std::map<int, PointFeaturePerId, std::less<int>,
    Eigen::aligned_allocator<std::pair<const int, PointFeaturePerId>>>
  idPointFeatureMap;

  // for mapping node
  sensor_msgs::Image::Ptr rgb_image_ptr;
  sensor_msgs::Image::Ptr depth_image_ptr;
  geometry_msgs::PoseStamped mapping_pose;
  sensor_msgs::Image::Ptr depth_rgb_ptr;

  nav_msgs::Odometry laserOdometry;
  nav_msgs::Odometry laserOdometryHfreq;
  nav_msgs::Path laserPath;  // path after odom
  nav_msgs::Path laserPath2; // path after optimization

  sensor_msgs::PointCloud2 laserCloudCornerLastMsg;
  sensor_msgs::PointCloud2 laserCloudSurfLastMsg;
  sensor_msgs::PointCloud2 laserCloudFullResMsg;

  // visualization_msgs::Marker line3dMsg;
  sensor_msgs::PointCloud2 pointFeatureCloudMsg;

  void allocateMemory();

  void readIntrinsicParameter(const std::string & calib_file)
  {
    if (DEBUG) {
      ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    }
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
      calib_file);
    m_camera->writeParameters(m_intrinsic_params);
  }

  // undistort lidar point
  // Tranform points to the last frame (to find correspondence)
  void TransformToStart(PointType const * const pi, PointType * const po) const;

  // transform all lidar points to the start of the next frame
  void TransformToEnd(PointType const * const pi, PointType * const po) const;

public
  :
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PoseEstimator(
    const std::string &
    calib_file)             /*:
                               q_last_curr(Eigen::Map<Eigen::Quaterniond>(NULL)),
                               t_last_curr(Eigen::Map<Eigen::Vector3d>(NULL)) ,
                               se3_last_curr(Eigen::Map<Eigen::Matrix<double,6,1>>(NULL))*/
  {
    allocateMemory();
    if (DEBUG) {
      ROS_WARN_STREAM("finish allocate memory");
    }
    readIntrinsicParameter(calib_file);
    if (DEBUG) {
      ROS_WARN_STREAM("finish read intrinsic");
    }
    se3_last_curr << 0, 0, 0, 1, 0, 0, 0;
  }

  void imageHandler(const sensor_msgs::ImageConstPtr & image_msg)
  {
    std::lock_guard<std::mutex> lk(mBuf);
    imageRawBuf.push(image_msg);
  }

  void featurePointsHandler(
    const sensor_msgs::PointCloud2ConstPtr & feature_points_msg)
  {
    std::lock_guard<std::mutex> lk(mBuf);
    featurePointsBuf.push(feature_points_msg);
  }

  void laserCloudSharpHandler(
    const sensor_msgs::PointCloud2ConstPtr & corner_points_sharp)
  {
    std::lock_guard<std::mutex> lk(mBuf);
    cornerSharpBuf.push(corner_points_sharp);
  }

  void laserCloudLessSharpHandler(
    const sensor_msgs::PointCloud2ConstPtr & corner_points_less_sharp)
  {
    std::lock_guard<std::mutex> lk(mBuf);
    cornerLessSharpBuf.push(corner_points_less_sharp);
  }

  void laserCloudFlatHandler(
    const sensor_msgs::PointCloud2ConstPtr & surf_points_flat)
  {
    std::lock_guard<std::mutex> lk(mBuf);
    surfFlatBuf.push(surf_points_flat);
  }

  void laserCloudLessFlatHandler(
    const sensor_msgs::PointCloud2ConstPtr & surf_points_less_flat)
  {
    std::lock_guard<std::mutex> lk(mBuf);
    surfLessFlatBuf.push(surf_points_less_flat);
  }

  void laserCloudFullResHandler(
    const sensor_msgs::PointCloud2ConstPtr & laser_cloud_fullres)
  {
    std::lock_guard<std::mutex> lk(mBuf);
    fullPointsBuf.push(laser_cloud_fullres);
  }

  void
  poseStampedHandler(const geometry_msgs::PoseStampedConstPtr & pose_stamped)
  {
    std::lock_guard<std::mutex> lk(mBuf);
    poseStampedBuf.push(pose_stamped);
  }

  void fetchAllFromBuf();

  // TODO: try finish in new thread
  void generateDepthMap();

  // TODO: try finish in new thread
  void updatePointFeatureMap(bool highfreq = false);

  void optimizeMap(bool lidar_only); // check outliers

  void updateLandmark();

  void fetchAllTimeStamp()
  {
    timeImageRaw = imageRawBuf.front()->header.stamp.toSec();
    timeFeaturePoints = featurePointsBuf.front()->header.stamp.toSec();
    timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
    timeCornerPointsLessSharp =
      cornerLessSharpBuf.front()->header.stamp.toSec();
    timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
    timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
    timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();
    if (!RUN_ODOMETRY) {
      timePoseStamped = poseStampedBuf.front()->header.stamp.toSec();
    }
  }

  void saveHistoryMessage(bool highfreq = false);

  void generateOdomMessage(bool highfreq = false);

  void updatePoseSequence(bool highfreq = false);

  void generateLaserMessage();

  bool checkAllMessageSynced()
  {
    if (fabs(timeImageRaw - timeLaserCloudFullRes) > 0.03 ||
      fabs(timeFeaturePoints - timeLaserCloudFullRes) > 0.03 ||
      fabs(timeCornerPointsSharp - timeLaserCloudFullRes) > 0.01 ||
      fabs(timeCornerPointsLessSharp - timeLaserCloudFullRes) > 0.01 ||
      fabs(timeSurfPointsFlat - timeLaserCloudFullRes) > 0.01 ||
      fabs(timeSurfPointsLessFlat - timeLaserCloudFullRes) > 0.01 ||
      (!RUN_ODOMETRY &&
      fabs(timePoseStamped - timeLaserCloudFullRes) > 0.03))
    {

      // ROS_WARN_STREAM("\ntimeImageRaw " << std::setprecision(10) <<
      // timeImageRaw*100000 << "\n"<< "timeFeaturePoints " <<
      // timeFeaturePoints*100000 << "\n"<< "timeLaserCloudFullRes " <<
      // timeLaserCloudFullRes*100000 << "\n"
      // << timeImageRaw - timeLaserCloudFullRes << "\n"
      // << timeFeaturePoints - timeLaserCloudFullRes << "\n"
      // << timeCornerPointsSharp - timeLaserCloudFullRes<< "\n"
      // << timeCornerPointsLessSharp - timeLaserCloudFullRes<< "\n"
      // << timeSurfPointsFlat - timeLaserCloudFullRes<< "\n"
      // << timeSurfPointsLessFlat - timeLaserCloudFullRes<< "\n");
      ROS_WARN(
        "Message unsynced: %f, %f, %f, %f, %f, %f, %f", timeImageRaw,
        timeFeaturePoints, timeCornerPointsSharp,
        timeCornerPointsLessSharp, timeSurfPointsFlat,
        timeSurfPointsLessFlat, timePoseStamped);
      return false;
    }
    return true;
  }

  void estimatePoseWithImageAndLaser(bool need_update = true);

  void handleImage();
  void handleImage(
    const cv::Mat & feature_image, pcl::PointCloud<ImagePoint>::Ptr keypoints,
    pcl::PointCloud<PointType>::Ptr laser_cloud_sharp,
    pcl::PointCloud<PointType>::Ptr laser_cloud_less_sharp,
    pcl::PointCloud<PointType>::Ptr laser_cloud_flat,
    pcl::PointCloud<PointType>::Ptr laser_cloud_less_flat,
    pcl::PointCloud<PointType>::Ptr laser_full,
    const Eigen::Vector3d & position,
    const Eigen::Quaterniond & orientation,
    int64_t timestamp_ns
  );

  const nav_msgs::Odometry & getLaserOdometry() const {return laserOdometry;}

  const nav_msgs::Path & getLaserPath() const {return laserPath;}

  const nav_msgs::Path & getLaserPath2();

  const sensor_msgs::PointCloud2 & getMapPointCloud();

  const sensor_msgs::PointCloud2 & getLaserCloudCornerLastMsg()
  {
    return laserCloudCornerLastMsg;
  }
  const sensor_msgs::PointCloud2 & getLaserCloudSurfLastMsg()
  {
    return laserCloudSurfLastMsg;
  }
  const sensor_msgs::PointCloud2 & getLaserCloudFullResMsg()
  {
    return laserCloudFullResMsg;
  }
  const bool cloudNeedPub() {return need_pub_cloud;}
  const bool odomNeedPub()
  {
    // bool temp = need_pub_odom;
    // if (temp){
    //     need_pub_odom = false;
    // }
    return need_pub_odom;
  }

  const bool mappingReady() {return mapping_ready;}
  const sensor_msgs::Image::Ptr & getRGBImage() {return rgb_image_ptr;}
  const sensor_msgs::Image::Ptr & getDepthImage() {return depth_image_ptr;}
  const geometry_msgs::PoseStamped & getMappingPose() {return mapping_pose;}
  const sensor_msgs::Image::Ptr & getDepthImageRGB() {return depth_rgb_ptr;}

  void publishMappingTopics();
};

#endif
