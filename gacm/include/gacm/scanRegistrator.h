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

#ifndef GACM__SCAN_REGISTRATOR_H_
#define GACM__SCAN_REGISTRATOR_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>

#include "camodocal/camera_models/Camera.h"
#include "cloud_msgs/cloud_info.h"
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"

#include "gacm/parameters.h"

typedef pcl::PointXYZI PointType;

struct smoothness_t
{
  float value;
  size_t ind;
};

struct by_value
{
  bool operator()(smoothness_t const & left, smoothness_t const & right)
  {
    return left.value < right.value;
  }
};

inline pcl::PointCloud<PointType>::Ptr
toCameraFrame(pcl::PointCloud<PointType>::Ptr & inCloud)
{
  pcl::PointCloud<PointType>::Ptr outCloud;
  outCloud.reset(new pcl::PointCloud<PointType>());
  // Eigen::Quaterniond q_extrinsic(0.704888,    0.709282, 0.007163,0.000280);
  // Eigen::Vector3d t_extrinsic(0.287338, -0.371119, -0.071403);
  outCloud->points.reserve(inCloud->points.size());
  for (auto point : inCloud->points) {
    PointType temp = point;
    Eigen::Vector3d temp_P(temp.x, temp.y, temp.z);
    temp_P = RCL * temp_P + TCL; // q_extrinsic * temp_P + t_extrinsic;
    temp.x = temp_P(0);
    temp.y = temp_P(1);
    temp.z = temp_P(2);
    temp.intensity = point.intensity;
    // temp.x = -1*point.y;
    // temp.y = -1*point.z;
    // temp.z = point.x;
    outCloud->push_back(temp);
  }
  return outCloud;
}

class ScanRegistrator
{
private:
  camodocal::CameraPtr m_camera;

  std_msgs::Header cloudHeader;
  pcl::PointCloud<PointType>::Ptr laserCloudIn;

  pcl::PointCloud<PointType>::Ptr fullCloudImage;
  pcl::PointCloud<PointType>::Ptr fullRangeCloudImage;

  //  std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudScans;
  pcl::PointCloud<PointType>::Ptr laserCloud;
  pcl::PointCloud<PointType>::Ptr laserRangeCloud;

  pcl::PointCloud<PointType>::Ptr cornerPointsSharp;
  pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
  pcl::PointCloud<PointType>::Ptr surfPointsFlat;
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;

  pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan;
  pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScanDS;

  pcl::VoxelGrid<PointType> downSizeFilter;

  cloud_msgs::cloud_info segMsg;

  cv::Mat rangeMat;
  PointType nanPoint;
  std::vector<std::pair<uint8_t, uint8_t>> neighborIterator;

  std::vector<float> cloudCurvature;
  std::vector<int> cloudSortInd;
  std::vector<smoothness_t> cloudSmoothness;
  std::vector<int> cloudNeighborPicked;
  std::vector<int> cloudLabel;

  template<typename PointT>
  void removeClosedPointCloud(
    const pcl::PointCloud<PointT> & cloud_in,
    pcl::PointCloud<PointT> & cloud_out, float thres)
  {
    if (&cloud_in != &cloud_out) {
      cloud_out.header = cloud_in.header;
      cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i) {
      if (cloud_in.points[i].x * cloud_in.points[i].x +
        cloud_in.points[i].y * cloud_in.points[i].y +
        cloud_in.points[i].z * cloud_in.points[i].z <
        thres * thres)
      {
        continue;
      }
      cloud_out.points[j] = cloud_in.points[i];
      j++;
    }
    if (j != cloud_in.points.size()) {
      cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
  }

public:
  ScanRegistrator()
  {
    nanPoint.x = std::numeric_limits<float>::quiet_NaN();
    nanPoint.y = std::numeric_limits<float>::quiet_NaN();
    nanPoint.z = std::numeric_limits<float>::quiet_NaN();
    nanPoint.intensity = -1;

    allocateMemory();
    resetParameters();
  }

  void allocateMemory()
  {
    laserCloudIn.reset(new pcl::PointCloud<PointType>());

    fullCloudImage.reset(new pcl::PointCloud<PointType>());
    fullRangeCloudImage.reset(new pcl::PointCloud<PointType>());

    cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
    cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
    surfPointsFlat.reset(new pcl::PointCloud<PointType>());
    surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());

    surfPointsLessFlatScan.reset(new pcl::PointCloud<PointType>());
    surfPointsLessFlatScanDS.reset(new pcl::PointCloud<PointType>());

    fullCloudImage->points.resize(N_SCANS * HORIZON_SCANS);
    fullRangeCloudImage->points.resize(N_SCANS * HORIZON_SCANS);

    // for (auto pcptr: laserCloudScans) {
    //     pcptr.reset(new pcl::PointCloud<PointType>());
    // }
    // for (size_t i = 0; i < N_SCANS; i++) {
    //     laserCloudScans.push_back(pcl::PointCloud<PointType>::Ptr(new
    //     pcl::PointCloud<PointType>()));
    // }

    laserCloud.reset(new pcl::PointCloud<PointType>());
    laserRangeCloud.reset(new pcl::PointCloud<PointType>());
    cloudCurvature.assign(N_SCANS * HORIZON_SCANS, -1);
    cloudNeighborPicked.assign(N_SCANS * HORIZON_SCANS, -1);
    cloudSortInd.assign(N_SCANS * HORIZON_SCANS, -1);
    cloudLabel.assign(N_SCANS * HORIZON_SCANS, 0);
    cloudSmoothness.resize(N_SCANS * HORIZON_SCANS);

    segMsg.startRingIndex.assign(N_SCANS, 0);
    segMsg.endRingIndex.assign(N_SCANS, 0);

    segMsg.segmentedCloudGroundFlag.assign(N_SCANS * HORIZON_SCANS, false);
    segMsg.segmentedCloudColInd.assign(N_SCANS * HORIZON_SCANS, 0);
    segMsg.segmentedCloudRange.assign(N_SCANS * HORIZON_SCANS, 0);

    std::pair<int8_t, int8_t> neighbor;
    neighbor.first = -1;
    neighbor.second = 0;
    neighborIterator.push_back(neighbor);
    neighbor.first = 0;
    neighbor.second = 1;
    neighborIterator.push_back(neighbor);
    neighbor.first = 0;
    neighbor.second = -1;
    neighborIterator.push_back(neighbor);
    neighbor.first = 1;
    neighbor.second = 0;
    neighborIterator.push_back(neighbor);

    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
  }

  void resetParameters()
  {
    laserCloudIn->clear();
    // ROS_INFO_STREAM("N_SCANS " << N_SCANS << " HS " << HORIZON_SCANS);
    rangeMat = cv::Mat(N_SCANS, HORIZON_SCANS, CV_32F, cv::Scalar::all(-1));

    std::fill(
      fullCloudImage->points.begin(), fullCloudImage->points.end(),
      nanPoint);
    std::fill(
      fullRangeCloudImage->points.begin(),
      fullRangeCloudImage->points.end(), nanPoint);
  }

  void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr & laserCloudMsg);

  // project into a 16*1800 range image
  void projectPointCloud();

  void organizePointCloud();

  void calculateSmoothness();

  void markOccludedPoints();

  void extractFeatures();

  // 这个函数根本没有被调用过
  void pointCloudToImage(pcl::PointCloud<PointType>::Ptr pointcloud);

  pcl::PointCloud<PointType>::Ptr getFullResCloud()
  {
    // return laserCloud;
    return toCameraFrame(laserCloud);
  }

  pcl::PointCloud<PointType>::Ptr getCornerPointsSharp()
  {
    // ROS_INFO_STREAM("Return Corner points Sharp " <<
    // cornerPointsSharp->points.size()); return cornerPointsSharp;
    return toCameraFrame(cornerPointsSharp);
  }

  pcl::PointCloud<PointType>::Ptr getCornerPointsLessSharp()
  {
    // ROS_INFO_STREAM("Return Corner points less Sharp " <<
    // cornerPointsLessSharp->points.size()); return cornerPointsLessSharp;
    return toCameraFrame(cornerPointsLessSharp);
  }
  pcl::PointCloud<PointType>::Ptr getSurfPointsFlat()
  {
    // ROS_INFO_STREAM("Return Surf points Flat " <<
    // surfPointsFlat->points.size()); return surfPointsFlat;
    return toCameraFrame(surfPointsFlat);
  }
  pcl::PointCloud<PointType>::Ptr getSurfPointsLessFlat()
  {
    // ROS_INFO_STREAM("Return Surf points Less Flat " <<
    // surfPointsLessFlat->points.size()); return surfPointsLessFlat;
    return toCameraFrame(surfPointsLessFlat);
  }

  double getStamp() {return cloudHeader.stamp.toSec();}

  void readIntrinsicParameter(const std::string & calib_file)
  {
    if (DEBUG) {
      ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    }
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
      calib_file);
  }

  void laserCloudHandler(const pcl::PointCloud<PointType>::Ptr & point_cloud);
  void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr & laser_msg);
};

#endif
