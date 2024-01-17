#include "gacm/scanRegistrator.h"

#include <cmath>

#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <pcl/PointIndices.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>

#include "pcl_conversions/pcl_conversions.h"
#include "sensor_msgs/PointCloud2.h"

#include "gacm/parameters.h"
#include "gacm/util/ip_basic.h"
#include "gacm/util/iputil.h"

void ScanRegistrator::copyPointCloud(
  const sensor_msgs::PointCloud2ConstPtr & laserCloudMsg)
{
  cloudHeader = laserCloudMsg->header;
  pcl::PointCloud<PointType>::Ptr templaser(new pcl::PointCloud<PointType>());

  pcl::fromROSMsg(*laserCloudMsg, *templaser);

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  for (int i = 0; i < templaser->size(); i++) {
    Eigen::Vector3d temp;
    temp << templaser->points[i].x, templaser->points[i].y,
      templaser->points[i].z;
    if (temp.norm() > CROP_NEAR && temp.norm() < CROP_FAR) {
      inliers->indices.push_back(i);
    }
  }
  pcl::ExtractIndices<PointType> extract;
  extract.setInputCloud(templaser); // input cloud
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*laserCloudIn);
}

void ScanRegistrator::projectPointCloud()
{
  float verticalAngle, horizonAngle, range;
  size_t rowIdn, columnIdn, index, cloudSize;
  PointType thisPoint;
  PointType thisPointProjected;

  cloudSize = laserCloudIn->points.size();

  for (size_t i = 0; i < cloudSize; ++i) {
    thisPoint.x = laserCloudIn->points[i].x;
    thisPoint.y = laserCloudIn->points[i].y;
    thisPoint.z = laserCloudIn->points[i].z;

    if (FRAME == FrameType::LASER) {
      verticalAngle =
        std::atan2(thisPoint.z, std::hypot(thisPoint.x, thisPoint.y)) * 180 /
        M_PI;
    } else {
      verticalAngle =
        std::atan2(thisPoint.y, std::hypot(thisPoint.x, thisPoint.z)) * 180 /
        M_PI;
    }
    rowIdn = (verticalAngle + ANGLE_BOTTOM) / ANGLE_RES_Y;
    if (rowIdn < 0 || rowIdn >= N_SCANS) {
      continue;
    }

    if (FRAME == FrameType::LASER) {
      horizonAngle = std::atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
    } else {
      horizonAngle = std::atan2(thisPoint.z, thisPoint.x) * 180 / M_PI;
    }

    columnIdn =
      -std::round((horizonAngle - 90.0) / ANGLE_RES_X) + HORIZON_SCANS / 2;
    if (columnIdn >= HORIZON_SCANS) {
      columnIdn -= HORIZON_SCANS;
    }

    if (columnIdn < 0 || columnIdn >= HORIZON_SCANS) {
      continue;
    }

    range = std::hypot(thisPoint.x, std::hypot(thisPoint.y, thisPoint.z));
    // ROS_INFO_STREAM("Processing point " << rowIdn << ", " << columnIdn);
    rangeMat.at<float>(rowIdn, columnIdn) = range;
    // ROS_INFO_STREAM("Stack in range Image");

    thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

    thisPointProjected.x = (float)rowIdn;
    thisPointProjected.y = (float)columnIdn;
    thisPointProjected.z = range;
    thisPointProjected.intensity = range;

    index = columnIdn + rowIdn * HORIZON_SCANS;

    fullCloudImage->points[index] = thisPoint;
    fullRangeCloudImage->points[index] = thisPointProjected;
  }
}

void ScanRegistrator::organizePointCloud()
{
  laserCloud->clear();
  laserRangeCloud->clear();

  PointType thisPointProjected;
  for (int i = 0; i < N_SCANS; i++) {
    segMsg.startRingIndex[i] = laserCloud->size() + 5;
    for (int j = 0; j < HORIZON_SCANS; j++) {
      float range = rangeMat.at<float>(i, j);
      int index = j + i * HORIZON_SCANS;
      if (rangeMat.at<float>(i, j) > 0) {
        laserRangeCloud->points.push_back(fullRangeCloudImage->points[index]);
        laserCloud->push_back(fullCloudImage->points[index]);
      }
    }
    segMsg.endRingIndex[i] = laserCloud->size() - 6;
  }
}

void ScanRegistrator::calculateSmoothness()
{
  int cloudSize = laserCloud->points.size();
  for (int i = 5; i < cloudSize - 5; i++) {
    float diffRange = laserRangeCloud->points[i - 5].intensity +
      laserRangeCloud->points[i - 4].intensity +
      laserRangeCloud->points[i - 3].intensity +
      laserRangeCloud->points[i - 2].intensity +
      laserRangeCloud->points[i - 1].intensity -
      laserRangeCloud->points[i].intensity * 10 +
      laserRangeCloud->points[i + 1].intensity +
      laserRangeCloud->points[i + 2].intensity +
      laserRangeCloud->points[i + 3].intensity +
      laserRangeCloud->points[i + 4].intensity +
      laserRangeCloud->points[i + 5].intensity;

    cloudCurvature[i] = diffRange * diffRange;

    cloudNeighborPicked[i] = 0;
    cloudLabel[i] = 0;

    cloudSmoothness[i].value = cloudCurvature[i];
    cloudSmoothness[i].ind = i;
  }
}

void ScanRegistrator::markOccludedPoints()
{
  int cloudSize = laserCloud->points.size();

  for (int i = 5; i < cloudSize - 6; ++i) {
    float depth1 = laserRangeCloud->points[i].intensity;
    float depth2 = laserRangeCloud->points[i + 1].intensity;
    float depth3 = laserRangeCloud->points[i - 1].intensity;
    int columnDiff = std::abs(
      int(laserRangeCloud->points[i + 1].y - laserRangeCloud->points[i].y));

    if (columnDiff < 10) {
      // points far away are occluded
      if (depth1 - depth2 > OCCLUDE_THRESHOLD) {
        cloudNeighborPicked[i - 5] = 1;
        cloudNeighborPicked[i - 4] = 1;
        cloudNeighborPicked[i - 3] = 1;
        cloudNeighborPicked[i - 2] = 1;
        cloudNeighborPicked[i - 1] = 1;
        cloudNeighborPicked[i] = 1;
      } else if (depth2 - depth1 > OCCLUDE_THRESHOLD) {
        cloudNeighborPicked[i + 1] = 1;
        cloudNeighborPicked[i + 2] = 1;
        cloudNeighborPicked[i + 3] = 1;
        cloudNeighborPicked[i + 4] = 1;
        cloudNeighborPicked[i + 5] = 1;
        cloudNeighborPicked[i + 6] = 1;
      }
    }

    float diff1 = std::abs(depth3 - depth1);
    float diff2 = std::abs(depth2 - depth1);

    if (diff1 > 0.02 * depth1 && diff2 > 0.02 * depth1) {
      cloudNeighborPicked[i] = 1;
    }
  }
}

void ScanRegistrator::extractFeatures()
{
  cornerPointsSharp->clear();
  cornerPointsLessSharp->clear();
  surfPointsFlat->clear();
  surfPointsLessFlat->clear();

  for (int i = 0; i < N_SCANS; i++) {
    surfPointsLessFlatScan->clear();

    for (int j = 0; j < 6; j++) {
      int sp =
        (segMsg.startRingIndex[i] * (6 - j) + segMsg.endRingIndex[i] * j) / 6;
      int ep = (segMsg.startRingIndex[i] * (5 - j) +
        segMsg.endRingIndex[i] * (j + 1)) /
        6 -
        1;
      if (sp >= ep) {
        continue;
      }

      std::sort(
        cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep,
        by_value());

      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSmoothness[k].ind;
        if (cloudNeighborPicked[ind] == 0 &&
          cloudCurvature[ind] > EDGE_THRESHOLD)
        {
          largestPickedNum++;
          if (largestPickedNum <= 2) {
            cloudLabel[ind] = 2;
            cornerPointsSharp->push_back(laserCloud->points[ind]);
            cornerPointsLessSharp->push_back(laserCloud->points[ind]);
          } else if (largestPickedNum <= 30) {
            cloudLabel[ind] = 1;
            cornerPointsLessSharp->push_back(laserCloud->points[ind]);
          } else {
            break;
          }

          cloudNeighborPicked[ind] = 1;
          for (int l = 1; l <= 5; l++) {
            int columnDiff =
              std::abs(
              int(laserRangeCloud->points[ind + l].y -
              laserRangeCloud->points[ind + l - 1].y));
            if (columnDiff > 10) {
              break;
            }
            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            int columnDiff =
              std::abs(
              int(laserRangeCloud->points[ind + l].y -
              laserRangeCloud->points[ind + l + 1].y));
            if (columnDiff > 10) {
              break;
            }
            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      int smallestPickedNum = 0;
      for (int k = sp; k <= ep; k++) {
        int ind = cloudSmoothness[k].ind;
        if (cloudNeighborPicked[ind] == 0 &&
          cloudCurvature[ind] < SURF_THRESHOLD)
        {
          cloudLabel[ind] = -1;
          surfPointsFlat->push_back(laserCloud->points[ind]);

          smallestPickedNum++;
          if (smallestPickedNum >= 4) {
            break;
          }

          cloudNeighborPicked[ind] = 1;
          for (int l = 1; l <= 5; l++) {
            int columnDiff =
              std::abs(
              int(laserRangeCloud->points[ind + l].y -
              laserRangeCloud->points[ind + l - 1].y));
            if (columnDiff > 10) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            int columnDiff =
              std::abs(
              int(laserRangeCloud->points[ind + l].y -
              laserRangeCloud->points[ind + l + 1].y));
            if (columnDiff > 10) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
        // 增加特征点数量
        else if (cloudCurvature[ind] < SURF_THRESHOLD) {
          cloudLabel[ind] = -1;
        }
      }

      for (int k = sp; k <= ep; k++) {
        // 这里只取小于0的点
        if (cloudLabel[k] < 0) {
          surfPointsLessFlatScan->push_back(laserCloud->points[k]);
        }
      }
    }

    surfPointsLessFlatScanDS->clear();
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);
    downSizeFilter.filter(*surfPointsLessFlatScanDS);

    *surfPointsLessFlat += *surfPointsLessFlatScanDS;
  }
}

void ScanRegistrator::pointCloudToImage(
  pcl::PointCloud<PointType>::Ptr pointcloud)
{
  cv::Mat depthmap(ROW, COL, CV_16UC1, cv::Scalar(0));
  // int valid_cnt = 0;

  /* 谁他妈写的，居然把外参写死了 */
  // Eigen::Quaterniond q_extrinsic(0.704888,    0.709282, 0.007163,0.000280);
  // Eigen::Vector3d t_extrinsic(0.287338, -0.371119, -0.071403);

  // R_extrinsic << 0.9999, 0.0098, 0.0105, 0.0106, -0.0062, -0.9999, -0.0097,
  // 0.9999,-0.0063; q_extrinsic = R_extrinsic; std::cout << "Check extrinsic
  // \n" << R_extrinsic << std::endl;

  for (auto point : pointcloud->points) {
    // Eigen::Vector3d P(-point.y, -point.z, point.x); // under camera
    // coordinate PointType temp = point;
    Eigen::Vector3d P(point.x, point.y, point.z);
    P = RCL * P + TCL;

    Eigen::Vector2d p;
    m_camera->spaceToPlane(P, p);

    if (p[1] > 0 && p[0] > 0 && p[1] < ROW && p[0] < COL && P[2] > 0) {
      depthmap.at<ushort>(p[1], p[0]) = (ushort)P[2];
      // valid_cnt++;
    } else {
      // ROS_INFO_STREAM("PCL point" << point.x << " ;" << point.y << " ;" <<
      // point.z << " Current point " << P[0]<<" ;" << P[1]<<" ;" <<P[2] << "
      // Project point " << p[0]<<" ;"<<p[1]);
    }
  }
  displayFalseColors(depthmap, "laser origin");
  // ROS_INFO_STREAM("Depth count in range " << valid_cnt << "/" <<
  // pointcloud->size());
  customDilate(depthmap, depthmap, 5, KERNEL_TYPE_DIAMOND);
  customDilate(depthmap, depthmap, 3, KERNEL_TYPE_DIAMOND);
  displayFalseColors(depthmap, "laser map");
  cv::waitKey(5);
}

void ScanRegistrator::laserCloudHandler(
  const sensor_msgs::PointCloud2ConstPtr & laser_msg)
{
  copyPointCloud(laser_msg);
  // ROS_INFO_STREAM("Finish copy point cloud");
  projectPointCloud();
  // ROS_INFO_STREAM("Finish project point cloud");
  organizePointCloud();
  calculateSmoothness();
  // ROS_INFO_STREAM("Finish calculate smoothness");
  markOccludedPoints();
  // ROS_INFO_STREAM("Finish mark Occluded point");
  extractFeatures();
  // ROS_INFO_STREAM("Finish extract feature");
  // pointCloudToImage(laserCloudIn);
  resetParameters();
}

void ScanRegistrator::laserCloudHandler(const pcl::PointCloud<PointType>::Ptr & point_cloud)
{
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  for (int i = 0; i < point_cloud->size(); i++) {
    Eigen::Vector3d temp;
    temp << point_cloud->points[i].x, point_cloud->points[i].y,
      point_cloud->points[i].z;
    if (temp.norm() > CROP_NEAR && temp.norm() < CROP_FAR) {
      inliers->indices.push_back(i);
    }
  }
  pcl::ExtractIndices<PointType> extract;
  extract.setInputCloud(point_cloud); // input cloud
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*laserCloudIn);

  projectPointCloud();
  // ROS_INFO_STREAM("Finish project point cloud");
  organizePointCloud();
  calculateSmoothness();
  // ROS_INFO_STREAM("Finish calculate smoothness");
  markOccludedPoints();
  // ROS_INFO_STREAM("Finish mark Occluded point");
  extractFeatures();
  // ROS_INFO_STREAM("Finish extract feature");
  // pointCloudToImage(laserCloudIn);
  resetParameters();
}
