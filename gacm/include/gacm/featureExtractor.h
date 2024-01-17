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

#ifndef GACM__FEATURE_EXTRACTOR_H_
#define GACM__FEATURE_EXTRACTOR_H_

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/fast_math.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <pcl/point_cloud.h>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "cv_bridge/cv_bridge.h"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"

#include "gacm/featureDefinition.h"
#include "gacm/parameters.h"

struct sort_descriptor_by_queryIdx
{
  inline bool operator()(
    const std::vector<cv::DMatch> & a,
    const std::vector<cv::DMatch> & b)
  {
    return a[0].queryIdx < b[0].queryIdx;
  }
};

struct sort_points_by_response
{
  inline bool operator()(const cv::KeyPoint & a, const cv::KeyPoint & b)
  {
    return a.response > b.response;
  }
};

class FeatureTracker
{

private:
  int temp_counter = 0;
  ros::Subscriber imageDataSub;
  camodocal::CameraPtr m_camera;

  cv::Mat image_cur, image_last;
  cv::Mat image_cur_with_text, image_last_with_text;
  double time_cur = 0, time_last = 0;

  // point feature
  // cv::Ptr<cv::ORB> orb;
  std::vector<cv::Point2f> keypoints_cur, keypoints_last;
  std::vector<cv::Point2f> keypoints_add;
  cv::Mat point_descriptors_cur, point_descriptors_last;

  // point feature info
  std::vector<int> keypoints_id;
  std::vector<int> keypoints_track_count;
  cv::Mat point_mask;
  int keypoint_id_cur = 0;

  // publish data
  pcl::PointCloud<ImagePoint>::Ptr point_to_pub;

public:
  FeatureTracker()
  {

    point_to_pub =
      pcl::PointCloud<ImagePoint>::Ptr(new pcl::PointCloud<ImagePoint>());
  }

  void imageDataCallback(const cv::Mat & image, double timestamp);
  void imageDataCallback(const sensor_msgs::Image::ConstPtr & image_msg);

  bool inBorder(const cv::Point2f & pt)
  {
    const int BORDER_SIZE = 2;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE &&
           BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
  }

  void addPoints(
    std::vector<cv::Point2f> & keypoints_add,
    std::vector<cv::Point2f> & keypoints_cur,
    std::vector<int> & track_cnt, std::vector<int> & ids,
    int & id_cur)
  {
    for (auto & p : keypoints_add) {
      keypoints_cur.push_back(p);
      ids.push_back(id_cur++);
      track_cnt.push_back(1);
    }
  }

  void rejectWithFundamentalMat(
    std::vector<cv::Point2f> & keypoints_last,
    std::vector<cv::Point2f> & keypoints_cur,
    std::vector<int> & track_cnt,
    std::vector<int> & ids);

  void readIntrinsicParameter(const std::string & calib_file)
  {
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
      calib_file);
  }

  void displayFeature(
    const std::string & window_name, cv::Mat & image,
    std::vector<cv::Point2f> & points,
    std::vector<int> & track_cnt, std::vector<int> & ids,
    cv::Mat & image_out) const;

  void generatePointCloud();

  pcl::PointCloud<ImagePoint>::Ptr getPointToPub() {return point_to_pub;}

  double getTimeStampLast() {return time_last;}

  cv::Mat getImageLast() {return image_last.clone();}

  cv::Mat getKeyPointImageLast() {return image_last_with_text.clone();}
};

#endif  // GACM__FEATURE_EXTRACTOR_H_
