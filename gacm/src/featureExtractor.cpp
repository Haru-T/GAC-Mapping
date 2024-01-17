#include "gacm/featureExtractor.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/Image.h"

#include "gacm/parameters.h"
#include "gacm/tic_toc.h"

namespace
{
template<typename T>
void reduceVector(std::vector<T> & v, const std::vector<uchar> & status)
{
  int j = 0;
  for (int i = 0; i < int(v.size()); i++) {
    if (status[i]) {
      v[j++] = v[i];
    }
  }
  v.resize(j);
}

void setMask(
  cv::Mat & mask, std::vector<cv::Point2f> & points,
  std::vector<int> & track_cnt, std::vector<int> & ids,
  const int MIN_DIST)
{
  mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

  // prefer to keep features that are tracked for long time
  std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cnt_pts_id;

  for (unsigned int i = 0; i < points.size(); i++) {
    cnt_pts_id.push_back(
      std::make_pair(track_cnt[i], std::make_pair(points[i], ids[i])));
  }

  sort(
    cnt_pts_id.begin(), cnt_pts_id.end(),
    [](const std::pair<int, std::pair<cv::Point2f, int>> & a,
    const std::pair<int, std::pair<cv::Point2f, int>> & b) {
      return a.first > b.first;
    });

  points.clear();
  ids.clear();
  track_cnt.clear();

  for (auto & it : cnt_pts_id) {
    if (mask.at<uchar>(it.second.first) == 255) {
      points.push_back(it.second.first);
      ids.push_back(it.second.second);
      track_cnt.push_back(it.first);
      cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
    }
  }
}
} // namespace

void FeatureTracker::imageDataCallback(const cv::Mat & image, double timestamp)
{
  image_cur = image.clone();
  time_cur = timestamp;
  if (DOWN_SAMPLE_NEED) {
    cv::resize(image_cur, image_cur, cv::Size(COL, ROW), cv::INTER_LINEAR);
  }
  image_cur_with_text = image_cur.clone();
  if (image_cur.empty()) {
    image_last = image_cur.clone();
    image_last_with_text = image_cur_with_text.clone();
  }
  if (keypoints_last.size() > 0) {
    // ROS_INFO_STREAM("Detect optical flow");
    std::vector<unsigned char> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(
      image_last, image_cur, keypoints_last,
      keypoints_cur, status, error, cv::Size(21, 21), 3);
    for (int i = 0; i < int(keypoints_cur.size()); i++) {
      if (status[i] && !inBorder(keypoints_cur[i])) {
        status[i] = 0;
      }
    }
    reduceVector(keypoints_last, status);
    reduceVector(keypoints_cur, status);
    reduceVector(keypoints_id, status);
    reduceVector(keypoints_track_count, status);
  }

  for (auto & n : keypoints_track_count) {
    n++;
  }
  // ROS_ERROR_STREAM("done1");
  // if (keypoints_last.size() > 0) {
  //     displayFeature("feature
  //     last",image_last,keypoints_last,keypoints_track_count, keypoints_id);
  // }
  rejectWithFundamentalMat(
    keypoints_last, keypoints_cur, keypoints_track_count,
    keypoints_id);
  setMask(
    point_mask, keypoints_cur, keypoints_track_count, keypoints_id,
    MIN_DIST_CORNER);
  int max_count_allow = MAX_CNT_CORNER - static_cast<int>(keypoints_cur.size());
  if (max_count_allow > 0) {
    keypoints_add.clear();
    cv::Mat image_cur_gray;
    cv::cvtColor(image_cur, image_cur_gray, cv::COLOR_RGB2GRAY);
    cv::goodFeaturesToTrack(
      image_cur_gray, keypoints_add, max_count_allow,
      0.01, MIN_DIST_CORNER, point_mask);
    addPoints(
      keypoints_add, keypoints_cur, keypoints_track_count, keypoints_id,
      keypoint_id_cur);
  }
  // ROS_ERROR_STREAM("done2");

  displayFeature(
    "feature cur", image_cur, keypoints_cur, keypoints_track_count,
    keypoints_id, image_cur_with_text);
  // ROS_ERROR_STREAM("done3");
  // cv::waitKey(5);

  time_last = time_cur;
  image_last = image_cur.clone();
  image_last_with_text = image_cur_with_text.clone();
  keypoints_last = keypoints_cur;
  keypoints_cur.clear();
  point_descriptors_last = point_descriptors_cur;

  // ROS_INFO_STREAM("GeneratePointCloud");
  generatePointCloud(); // publish features
  // ROS_INFO_STREAM("GenerateLineCloud");
  // generateLineCloud(); // publish features
  // ROS_INFO_STREAM("FinishGenerateLineCloud");
}

void FeatureTracker::imageDataCallback(
  const sensor_msgs::Image::ConstPtr & image_msg)
{
  time_cur = image_msg->header.stamp.toSec();
  cv_bridge::CvImageConstPtr ptr;
  ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
  imageDataCallback(ptr->image, time_cur);
}

void FeatureTracker::rejectWithFundamentalMat(
  std::vector<cv::Point2f> & keypoints_last,
  std::vector<cv::Point2f> & keypoints_cur, std::vector<int> & track_cnt,
  std::vector<int> & ids)
{
  if (keypoints_cur.size() >= 8) {
    ROS_DEBUG("FM ransac begins");
    TicToc t_f;
    std::vector<cv::Point2f> un_cur_pts(keypoints_last.size()),
    un_forw_pts(keypoints_cur.size());
    for (unsigned int i = 0; i < keypoints_last.size(); i++) {
      Eigen::Vector3d tmp_p;
      m_camera->liftProjective(
        Eigen::Vector2d(keypoints_last[i].x, keypoints_last[i].y), tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
      un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

      m_camera->liftProjective(
        Eigen::Vector2d(keypoints_cur[i].x, keypoints_cur[i].y), tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
      un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }

    std::vector<uchar> status;
    cv::findFundamentalMat(
      un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD,
      0.99, status);
    int size_a = keypoints_last.size();
    reduceVector(keypoints_last, status);
    reduceVector(keypoints_cur, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    ROS_DEBUG(
      "FM ransac: %d -> %lu: %f", size_a, keypoints_cur.size(),
      1.0 * keypoints_cur.size() / size_a);
    ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
  }
}

void FeatureTracker::displayFeature(
  const std::string & window_name,
  cv::Mat & image,
  std::vector<cv::Point2f> & points,
  std::vector<int> & track_cnt,
  std::vector<int> & ids,
  cv::Mat & image_out) const
{
  cv::Mat temp_img = image.clone();
  for (size_t i = 0; i < points.size(); i++) {
    double len = std::min(1.0, 1.0 * track_cnt[i] / WINDOW_SIZE);
    cv::circle(
      temp_img, points[i], 2,
      cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    char name[20];
    std::sprintf(name, "%d(%d)", ids[i], track_cnt[i]);
    cv::putText(
      temp_img, name, points[i], cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(0, 0, 0));
  }
  image_out = temp_img.clone();
}

void FeatureTracker::generatePointCloud()
{
  point_to_pub->clear();
  ImagePoint point;
  for (size_t i = 0; i < keypoints_last.size(); i++) {
    point.index = keypoints_id[i];
    point.track_cnt = keypoints_track_count[i];
    // point.u = keypoints_cur[i].x; // in OpenCV x is correspond to cols
    // point.v = keypoints_cur[i].y;
    // Eigen::Vector2d a(keypoints_last[i].x, keypoints_last[i].y);
    // Eigen::Vector3d b;
    // m_camera->liftProjective(a,b);
    point.u = keypoints_last[i].x; // in OpenCV x correspond to cols, that is u
    point.v = keypoints_last[i].y; // in OpenCV y correspond to rows, that is v
    // point.x = b[0];
    // point.y = b[1];
    // point.z = b[2];
    point_to_pub->push_back(point);
  }
}
