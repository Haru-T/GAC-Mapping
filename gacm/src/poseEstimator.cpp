#include "gacm/poseEstimator.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "pcl_conversions/pcl_conversions.h"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/image_encodings.h"

#include "gacm/featureDefinition.h"
#include "gacm/ga_posegraph/poseFactor.h"
#include "gacm/lidarFactor.h"
#include "gacm/parameters.h"
#include "gacm/util/godec.h"
#include "gacm/util/ip_basic.h"
#include "gacm/util/iputil.h"
#include "gacm/util/jbf_filter.h"

void PoseEstimator::allocateMemory()
{
  rgb_cur = cv::Mat(ROW, COL, CV_8UC3, cv::Scalar(0, 0, 0));
  rgb_last = rgb_cur.clone();
  depth_map = cv::Mat(ROW, COL, CV_16UC1, cv::Scalar(0));
  laser_map = cv::Mat(ROW, COL, CV_16UC1, cv::Scalar(0));

  imagePointsCur.reset(new pcl::PointCloud<ImagePoint>());
  imagePointsLast.reset(new pcl::PointCloud<ImagePoint>());

  laserCloudFullResCur.reset(new pcl::PointCloud<PointType>());
  laserCloudFullResLast.reset(new pcl::PointCloud<PointType>());

  cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
  cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
  surfPointsFlat.reset(new pcl::PointCloud<PointType>());
  surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());

  laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
  laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());

  kdtreeCornerLast.reset(new pcl::KdTreeFLANN<PointType>());
  kdtreeSurfLast.reset(new pcl::KdTreeFLANN<PointType>());

  timeImageRaw = 0;
  timeFeaturePoints = 0;

  timeCornerPointsSharp = 0;
  timeCornerPointsLessSharp = 0;
  timeSurfPointsFlat = 0;
  timeSurfPointsLessFlat = 0;
  timeLaserCloudFullRes = 0;

  skipFrameNum = 1;
  laserFrameCount = 0;
  laserFrameCountFromBegin = 0;
  frameTobeMap = 0;
  // lastOptimizeFrame = 0;
  systemInited = false;
  need_pub_cloud = false;
  need_pub_odom = false;
  mapping_ready = false;

  corner_correspondence = 0;
  plane_correspondence = 0;
  point_correspondence = 0;

  laserCloudCornerLastNum = 0;
  laserCloudSurfLastNum = 0;

  imagePointsLastNum = 0;
  imagePointsCurNum = 0;

  Eigen::Vector3d q_v(0, 0, 0);

  q_w_curr.vec() = q_v;
  q_w_curr.w() = 1;
  t_w_curr << 0, 0, 0;

  q_w_curr_hfreq.vec() = q_v;
  q_w_curr_hfreq.w() = 1;
  t_w_curr_hfreq << 0, 0, 0;
}

void PoseEstimator::TransformToStart(
  PointType const * const pi,
  PointType * const po) const
{
  // interpolation ratio
  double s;
  if (DISTORTION) {
    s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
  } else {
    s = 1.0;
  }
  // s = 1;
  Eigen::Quaterniond q_last_curr(se3_last_curr.head(4).data());
  Eigen::Vector3d t_last_curr(se3_last_curr.tail(3).data());
  Eigen::Quaterniond q_point_last =
    Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
  Eigen::Vector3d t_point_last = s * t_last_curr;
  Eigen::Vector3d point(pi->x, pi->y, pi->z);
  Eigen::Vector3d un_point = q_point_last * point + t_point_last;

  po->x = un_point.x();
  po->y = un_point.y();
  po->z = un_point.z();
  po->intensity = pi->intensity;
}

void PoseEstimator::TransformToEnd(
  PointType const * const pi,
  PointType * const po) const
{
  // undistort point first
  pcl::PointXYZI un_point_tmp;
  TransformToStart(pi, &un_point_tmp);

  Eigen::Quaterniond q_last_curr(se3_last_curr.head(4).data());
  Eigen::Vector3d t_last_curr(se3_last_curr.tail(3).data());

  Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
  Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

  po->x = point_end.x();
  po->y = point_end.y();
  po->z = point_end.z();

  // Remove distortion time info
  po->intensity = int(pi->intensity);
}

void PoseEstimator::fetchAllFromBuf()
{
  std::lock_guard<std::mutex> lk(mBuf);

  cv_bridge::CvImageConstPtr ptr;
  ptr = cv_bridge::toCvCopy(
    *imageRawBuf.front(),
    sensor_msgs::image_encodings::RGB8);
  rgb_cur = ptr->image;
  imageRawBuf.pop();

  imagePointsCur->clear();
  pcl::fromROSMsg(*featurePointsBuf.front(), *imagePointsCur);
  imagePointsCurNum = imagePointsCur->size();
  featurePointsBuf.pop();

  cornerPointsSharp->clear();
  pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
  cornerSharpBuf.pop();

  cornerPointsLessSharp->clear();
  pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
  cornerLessSharpBuf.pop();

  surfPointsFlat->clear();
  pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
  surfFlatBuf.pop();

  surfPointsLessFlat->clear();
  pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
  surfLessFlatBuf.pop();

  laserCloudFullResCur->clear();
  pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullResCur);
  fullPointsBuf.pop();

  if (!RUN_ODOMETRY) {
    const auto & msg = poseStampedBuf.front();
    t_w_curr = Eigen::Vector3d(
      msg->pose.position.x, msg->pose.position.y,
      msg->pose.position.z);
    q_w_curr =
      Eigen::Quaterniond(
      msg->pose.orientation.w, msg->pose.orientation.x,
      msg->pose.orientation.y, msg->pose.orientation.z);
    poseStampedBuf.pop();
  }
}

void PoseEstimator::generateDepthMap()
{
  depth_map = cv::Mat(ROW, COL, CV_16UC1, cv::Scalar(0));

  // TODO: project triangulate points to image
  // laser point project to image
  for (auto point : laserCloudFullResLast->points) {
    // Eigen::Vector3d P(-point.y, -point.z, point.x); // under camera
    // coordinate
    Eigen::Vector3d P(point.x, point.y, point.z); // under camera coordinate
    Eigen::Vector2d p;
    m_camera->spaceToPlane(P, p);
    if (p[1] > 0 && p[0] > 0 && p[1] < ROW && p[0] < COL && P[2] > 0 &&
      P[2] <= 65)
    {
      // Scale: P[2] -> [0, 60] ushort->[0, 65535]
      depth_map.at<ushort>(p[1], p[0]) = (ushort)(1000.0 * P[2]);
      // ROS_WARN_STREAM("##Check depth " << P[2] << " ; " <<
      // depth_map.at<ushort>(p[1], p[0]));
    }
  }
  // save laser depth image as mapping node input
  // laser_map = depth_map;

  // TODO: try other method
  // displayFalseColors(depth_map, "before diliate");
  // cv::waitKey(0);
  customDilate(depth_map, depth_map, 7, KERNEL_TYPE_DIAMOND);
  // customDilate(depth_map, depth_map, 5, KERNEL_TYPE_DIAMOND);

  // displayFalseColors(depth_map, "falsecolor");
  cv::Mat bgr;
  generateFalseColors(depth_map, bgr);
  // cv::waitKey(5);
  cv_bridge::CvImage bridge;
  bridge.image = bgr;
  bridge.encoding = sensor_msgs::image_encodings::BGR8;
  depth_rgb_ptr = bridge.toImageMsg();
  depth_rgb_ptr->header.stamp =
    mapDataPerFrameBuf.empty() ? ros::Time().fromSec(timeCornerPointsLessSharp) :
    ros::Time().fromSec(mapDataPerFrameBuf[0].timestamp);
}

void PoseEstimator::updatePointFeatureMap(bool highfreq)
{
  for (ImagePoint point : imagePointsLast->points) {
    auto it = idPointFeatureMap.find(point.index);
    Eigen::Matrix<double, 3, 1> eigen_point;
    double inverse_depth = 0;
    unsigned char depth_type = 0;
    if (!highfreq) {
      double depth = depth_map.at<ushort>(point.v, point.u);
      depth /= 1000.0;
      depth_type =
        depth > 0 ? 1 : 0;   // 0 no depth 1 from lidar 2 from triangulation
      if (depth > 0) {
        inverse_depth = 1.0 / depth;
      }
    }
    eigen_point << inverse_depth, point.u, point.v;

    // if (!highfreq) {
    // record feature in each frame for mapping use
    mapDataPerFrameBuf[laserFrameCountFromBegin - frameTobeMap]
    .point_feature_ids.push_back(point.index);
    // }

    if (it != idPointFeatureMap.end()) {
      // ROS_WARN_STREAM("###Check feature track " << it->second.endFrame() <<
      // " now " << laserFrameCountFromBegin);
      it->second.feature_per_frame.push_back(
        PointFeaturePerFrame(
          eigen_point, depth_type, laserFrameCountFromBegin));
      it->second.setDepthType(depth_type); // TODO: maybe not correct
      // it->second.setDepth(depth);
    } else {
      PointFeaturePerId pfpid(
        point.index,
        laserFrameCountFromBegin);   // create a new structure with
                                     // (featureid, startfreame)
      pfpid.feature_per_frame.push_back(
        PointFeaturePerFrame(
          eigen_point, depth_type, laserFrameCountFromBegin));
      pfpid.setDepthType(depth_type);
      // pfpid.setInitialEstimation(0); // TODO: maybe useless
      // pfpid.setDepth(depth);
      idPointFeatureMap.insert(
        std::pair<int, PointFeaturePerId>(point.index, pfpid));
    }
  }
}

void PoseEstimator::optimizeMap(bool lidar_only)
{
  // ROS_WARN_STREAM("### Begin OptimizeMap ###");
  // if (laserFrameCountFromBegin < 5) return;
  int win_size = 7;
  int keep_constant_len = 0;
  int min_track_len = 2;
  ceres::Problem problem;
  ceres::LossFunction * loss_function = new ceres::HuberLoss(0.1);
  ceres::LocalParameterization * q_parameterization =
    new ceres::EigenQuaternionParameterization();
  ceres::LocalParameterization * t_parameterization =
    new ceres::IdentityParameterization(3);
  ceres::LocalParameterization * pose_parameterization =
    new ceres::ProductParameterization(
    q_parameterization,
    t_parameterization);
  for (size_t i = 0; i < poseSequence.size(); i++) {
    problem.AddParameterBlock(
      poseSequence[i].pose.data(), 7,
      pose_parameterization);
    // problem.AddParameterBlock(poseSequence[i].q.coeffs().data(),4,q_parameterization);
    // problem.AddParameterBlock(poseSequence[i].t.data(),3);
    if (i > 0) {
      Eigen::Matrix<double, 6, 6> sqrt =
        Eigen::Matrix<double, 6, 6>::Identity();

      ceres::CostFunction * cost_function =
        PoseGraph3dErrorTerm2::Create(poseSequence[i].rel, sqrt * 100);
      problem.AddResidualBlock(
        cost_function, loss_function,
        poseSequence[i - 1].pose.data(),
        poseSequence[i].pose.data());
    }
    if (lidar_only) {
      problem.SetParameterBlockConstant(poseSequence[i].pose.data());
      continue;
    }
    if (i <= 5 || i < laserFrameCountFromBegin -
      win_size)                     // set the first frame to be fix
    {
      problem.SetParameterBlockConstant(poseSequence[i].pose.data());
    }
  }

  // add residual block for feature point
  for (auto it = idPointFeatureMap.begin(); it != idPointFeatureMap.end();
    it++)
  {

    if (it->second.depth_type == 0 ||
      it->second.endFrame() < laserFrameCountFromBegin - win_size)
    {
      continue;
    }

    size_t trackcount = it->second.feature_per_frame.size();
    if (trackcount < min_track_len) {
      continue;
    }

    problem.AddParameterBlock(it->second.inverse_depth, 1);

    int start_frame = it->second.startFrame();
    int depth_frame = it->second.depthFrame();
    int depth_frame_index = it->second.depthFrameIndex();
    Eigen::Vector2d p_first_depth =
      it->second.feature_per_frame[depth_frame_index].uv;
    Eigen::Vector3d P_first_depth;
    Eigen::Quaterniond q_first_depth(
      poseSequence[depth_frame].pose.head(4).data());
    Eigen::Vector3d t_first_depth(
      poseSequence[depth_frame].pose.tail(3).data());
    m_camera->liftProjective(p_first_depth, P_first_depth);

    for (size_t j = 0; j < trackcount; j++) {

      int frame_id = it->second.getFrameIdAt(j);
      if (frame_id == depth_frame) {
        continue;
      }
      Eigen::Vector2d cp;
      cp = it->second.feature_per_frame[j].uv;
      ceres::CostFunction * cost_function =
        ReprojectionPoint32Factor3new::Create(
        m_intrinsic_params, cp,
        P_first_depth);
      problem.AddResidualBlock(
        cost_function, loss_function,
        poseSequence[depth_frame].pose.data(),
        poseSequence[frame_id].pose.data(),
        // it->second.estimated_depth
        it->second.inverse_depth);
      // it->second.estimated_point.data());
    }
  }

  ceres::Solver::Options options;
  options.gradient_tolerance = 1e-16;
  options.function_tolerance = 1e-16;
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  // options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  // options.minimizer_type = ceres::TRUST_REGION;
  // options.trust_region_strategy_type = ceres::DOGLEG;
  // options.dogleg_type = ceres::SUBSPACE_DOGLEG;
  options.dynamic_sparsity = true;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 2;
  options.max_num_iterations = 3;

  ceres::Solver::Summary summary;
  // ROS_WARN_STREAM("### Begin BA ###");
  ceres::Solve(options, &problem, &summary);
  // std::cout << summary.BriefReport() << "\n";
}

void PoseEstimator::updateLandmark()
{
  for (auto it = idPointFeatureMap.begin(); it != idPointFeatureMap.end(); ) {

    if (it->second.isOutlier()) {
      bool is_active = it->second.endFrame() >= laserFrameCountFromBegin - 7;

      bool end_track = it->second.endFrame() < laserFrameCountFromBegin;
      if (end_track && it->second.feature_per_frame.size() < 5) {
        it = idPointFeatureMap.erase(it);
      } else if (is_active) {
        bool next_depth_sucess = it->second.useNextDepth();
        if (next_depth_sucess) {
          it++;
        } else {
          it = idPointFeatureMap.erase(it);
        }
      } else {
        // TODO:  maginalization
        it = idPointFeatureMap.erase(it);
      }
    } else {
      // not a outlier
      it++;
    }
  }
}

void PoseEstimator::saveHistoryMessage(bool highfreq)
{

  // image feature and picture
  rgb_last = rgb_cur.clone();
  pcl::PointCloud<ImagePoint>::Ptr imagePointsTemp = imagePointsCur;
  imagePointsCur = imagePointsLast;
  imagePointsLast = imagePointsTemp;

  // feature number
  laserCloudCornerLastNum = laserCloudCornerLast->points.size();
  laserCloudSurfLastNum = laserCloudSurfLast->points.size();
  imagePointsLastNum = imagePointsCurNum;

  if (highfreq) {
    return;
  }

  // laser feature and cloud
  // std::swap(cornerPointsLessSharp, laserCloudCornerLast);
  pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
  cornerPointsLessSharp = laserCloudCornerLast;
  laserCloudCornerLast = laserCloudTemp;

  // std::swap(surfPointsLessFlat, laserCloudSurfLast);
  laserCloudTemp = surfPointsLessFlat;
  surfPointsLessFlat = laserCloudSurfLast;
  laserCloudSurfLast = laserCloudTemp;

  // std::swap(laserCloudfullResLast, laserCloudFullResCur);
  laserCloudTemp = laserCloudFullResLast;
  laserCloudFullResLast = laserCloudFullResCur;
  laserCloudFullResCur = laserCloudTemp;

  // laser features set into kd tree
  if (laserCloudCornerLast != nullptr && !laserCloudCornerLast->empty()) {
    kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
  }
  if (laserCloudSurfLast != nullptr && !laserCloudSurfLast->empty()) {
    kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
  }
}

void PoseEstimator::generateOdomMessage(bool highfreq)
{
  // ROS_INFO_STREAM("Generating message");
  if (highfreq) {
    laserOdometry.header.frame_id = "camera"; // need check
    laserOdometry.child_frame_id = "laser_odom";
    laserOdometry.header.stamp = ros::Time().fromSec(timeImageRaw);
    laserOdometry.pose.pose.orientation.x = q_w_curr_hfreq.x();
    laserOdometry.pose.pose.orientation.y = q_w_curr_hfreq.y();
    laserOdometry.pose.pose.orientation.z = q_w_curr_hfreq.z();
    laserOdometry.pose.pose.orientation.w = q_w_curr_hfreq.w();
    laserOdometry.pose.pose.position.x = t_w_curr_hfreq.x();
    laserOdometry.pose.pose.position.y = t_w_curr_hfreq.y();
    laserOdometry.pose.pose.position.z = t_w_curr_hfreq.z();
  } else {
    // generate odometry message and add to path message
    laserOdometry.header.frame_id = "camera"; // need check
    laserOdometry.child_frame_id = "laser_odom";
    laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
    // ROS_ERROR_STREAM("Generate odom " <<
    // ros::Time().fromSec(timeSurfPointsLessFlat).toNSec());
    laserOdometry.pose.pose.orientation.x = q_w_curr.x();
    laserOdometry.pose.pose.orientation.y = q_w_curr.y();
    laserOdometry.pose.pose.orientation.z = q_w_curr.z();
    laserOdometry.pose.pose.orientation.w = q_w_curr.w();
    laserOdometry.pose.pose.position.x = t_w_curr.x();
    laserOdometry.pose.pose.position.y = t_w_curr.y();
    laserOdometry.pose.pose.position.z = t_w_curr.z();
    // need_pub_odom = true;
  }

  geometry_msgs::PoseStamped laserPose;
  laserPose.header = laserOdometry.header;
  laserPose.pose = laserOdometry.pose.pose;
  laserPath.header.stamp = laserOdometry.header.stamp;
  laserPath.poses.push_back(laserPose);
  laserPath.header.frame_id = "camera"; // need check
}

void PoseEstimator::updatePoseSequence(bool highfreq)
{
  Eigen::Matrix<double, 7, 1> temp_pose;
  // Eigen::Matrix<double,7,1> temp_rel;
  if (highfreq) {
    temp_pose << q_w_curr_hfreq.x(), q_w_curr_hfreq.y(), q_w_curr_hfreq.z(),
      q_w_curr_hfreq.w(), t_w_curr_hfreq.x(), t_w_curr_hfreq.y(),
      t_w_curr_hfreq.z();
    // temp_rel = se3_last_curr;
  } else {
    temp_pose << q_w_curr.x(), q_w_curr.y(), q_w_curr.z(), q_w_curr.w(),
      t_w_curr.x(), t_w_curr.y(), t_w_curr.z();
    // temp_rel = se3_last_curr;
  }
  poseSequence.push_back(
    PosePerFrame(laserFrameCountFromBegin, temp_pose, se3_last_curr));
}

void PoseEstimator::generateLaserMessage()
{
  if (laserFrameCount % skipFrameNum == 0) {
    laserFrameCount = 0;
    need_pub_cloud = true;
    need_pub_odom = true;

    pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLastMsg);
    laserCloudCornerLastMsg.header.stamp =
      ros::Time().fromSec(timeSurfPointsLessFlat);
    laserCloudCornerLastMsg.header.frame_id = "camera";
    // pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

    pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLastMsg);
    laserCloudSurfLastMsg.header.stamp =
      ros::Time().fromSec(timeSurfPointsLessFlat);
    laserCloudSurfLastMsg.header.frame_id = "camera";
    // pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

    pcl::toROSMsg(*laserCloudFullResCur, laserCloudFullResMsg);
    laserCloudFullResMsg.header.stamp =
      ros::Time().fromSec(timeSurfPointsLessFlat);
    laserCloudFullResMsg.header.frame_id = "camera";
    // pubLaserCloudFullRes.publish(laserCloudFullRes3);
  } else {
    need_pub_cloud = false;
    need_pub_odom = false;
  }
  laserFrameCount++;
}

void PoseEstimator::estimatePoseWithImageAndLaser(bool need_update)
{
  clock_t start, end;
  start = clock();
  // generate depth image
  generateDepthMap();
  end = clock();
  std::ofstream outfile;
  outfile.open(
    (std::string(std::getenv("HOME")) +
    std::string("/gacm_output/timecost/depth_completion_time.txt"))
    .c_str(),
    ios::app);
  outfile << (double)(end - start) / CLOCKS_PER_SEC << "\n";
  outfile.close();

  start = clock();
  if (RUN_ODOMETRY && need_update) {
    int cornerPointsSharpNum = cornerPointsSharp->points.size();
    int surfPointsFlatNum = surfPointsFlat->points.size();

    for (size_t opti_counter = 0; opti_counter < 8; ++opti_counter) {
      corner_correspondence = 0;
      plane_correspondence = 0;
      point_correspondence = 0;

      ceres::LossFunction * loss_function = new ceres::HuberLoss(0.1);
      ceres::LocalParameterization * q_parameterization =
        new ceres::EigenQuaternionParameterization();
      ceres::LocalParameterization * t_parameterization =
        new ceres::IdentityParameterization(3);
      ceres::Problem::Options problem_options;
      ceres::Problem problem(problem_options);
      problem.AddParameterBlock(
        se3_last_curr.head(4).data(), 4,
        q_parameterization);
      problem.AddParameterBlock(se3_last_curr.tail(3).data(), 3);

      pcl::PointXYZI pointSel;
      std::vector<int> pointSearchInd;
      std::vector<float> pointSearchSqDis;

      // find correspondence for corner features
      for (int i = 0; i < cornerPointsSharpNum; ++i) {
        TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
        kdtreeCornerLast->nearestKSearch(
          pointSel, 1, pointSearchInd,
          pointSearchSqDis);

        int closestPointInd = -1, minPointInd2 = -1;
        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
          closestPointInd = pointSearchInd[0];
          int closestPointScanID =
            int(laserCloudCornerLast->points[closestPointInd].intensity);

          double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
          // search in the direction of increasing scan line
          for (int j = closestPointInd + 1;
            j < (int)laserCloudCornerLast->points.size(); ++j)
          {
            // if in the same scan line, continue
            if (int(laserCloudCornerLast->points[j].intensity) <=
              closestPointScanID)
            {
              continue;
            }

            // if not in nearby scans, end the loop
            if (int(laserCloudCornerLast->points[j].intensity) >
              (closestPointScanID + NEARBY_SCAN))
            {
              break;
            }

            double pointSqDis =
              (laserCloudCornerLast->points[j].x - pointSel.x) *
              (laserCloudCornerLast->points[j].x - pointSel.x) +
              (laserCloudCornerLast->points[j].y - pointSel.y) *
              (laserCloudCornerLast->points[j].y - pointSel.y) +
              (laserCloudCornerLast->points[j].z - pointSel.z) *
              (laserCloudCornerLast->points[j].z - pointSel.z);

            if (pointSqDis < minPointSqDis2) {
              // find nearer point
              minPointSqDis2 = pointSqDis;
              minPointInd2 = j;
            }
          }

          // search in the direction of decreasing scan line
          for (int j = closestPointInd - 1; j >= 0; --j) {
            // if in the same scan line, continue
            if (int(laserCloudCornerLast->points[j].intensity) >=
              closestPointScanID)
            {
              continue;
            }

            // if not in nearby scans, end the loop
            if (int(laserCloudCornerLast->points[j].intensity) <
              (closestPointScanID - NEARBY_SCAN))
            {
              break;
            }

            double pointSqDis =
              (laserCloudCornerLast->points[j].x - pointSel.x) *
              (laserCloudCornerLast->points[j].x - pointSel.x) +
              (laserCloudCornerLast->points[j].y - pointSel.y) *
              (laserCloudCornerLast->points[j].y - pointSel.y) +
              (laserCloudCornerLast->points[j].z - pointSel.z) *
              (laserCloudCornerLast->points[j].z - pointSel.z);

            if (pointSqDis < minPointSqDis2) {
              // find nearer point
              minPointSqDis2 = pointSqDis;
              minPointInd2 = j;
            }
          }
        }

        // both closestPointInd and minPointInd2 is valid
        if (minPointInd2 >= 0) {
          Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
            cornerPointsSharp->points[i].y,
            cornerPointsSharp->points[i].z);
          Eigen::Vector3d last_point_a(
            laserCloudCornerLast->points[closestPointInd].x,
            laserCloudCornerLast->points[closestPointInd].y,
            laserCloudCornerLast->points[closestPointInd].z);
          Eigen::Vector3d last_point_b(
            laserCloudCornerLast->points[minPointInd2].x,
            laserCloudCornerLast->points[minPointInd2].y,
            laserCloudCornerLast->points[minPointInd2].z);

          double s;
          if (DISTORTION) {
            s = (cornerPointsSharp->points[i].intensity -
              int(cornerPointsSharp->points[i].intensity)) /
              SCAN_PERIOD;
          } else {
            s = 1.0;
          }

          ceres::CostFunction * cost_function = LidarEdgeFactor::Create(
            curr_point, last_point_a, last_point_b, s);
          problem.AddResidualBlock(
            cost_function, loss_function,
            se3_last_curr.head(4).data(),
            se3_last_curr.tail(3).data());

          corner_correspondence++;
        }
      } // find correspondence for corner features

      // find correspondence for plane features
      for (int i = 0; i < surfPointsFlatNum; ++i) {
        TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
        kdtreeSurfLast->nearestKSearch(
          pointSel, 1, pointSearchInd,
          pointSearchSqDis);

        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
          closestPointInd = pointSearchInd[0];

          // get closest point's scan ID
          int closestPointScanID =
            int(laserCloudSurfLast->points[closestPointInd].intensity);
          double minPointSqDis2 = DISTANCE_SQ_THRESHOLD,
            minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

          // search in the direction of increasing scan line
          for (int j = closestPointInd + 1;
            j < (int)laserCloudSurfLast->points.size(); ++j)
          {
            // if not in nearby scans, end the loop
            if (int(laserCloudSurfLast->points[j].intensity) >
              (closestPointScanID + NEARBY_SCAN))
            {
              break;
            }

            double pointSqDis =
              (laserCloudSurfLast->points[j].x - pointSel.x) *
              (laserCloudSurfLast->points[j].x - pointSel.x) +
              (laserCloudSurfLast->points[j].y - pointSel.y) *
              (laserCloudSurfLast->points[j].y - pointSel.y) +
              (laserCloudSurfLast->points[j].z - pointSel.z) *
              (laserCloudSurfLast->points[j].z - pointSel.z);

            // if in the same or lower scan line
            if (int(laserCloudSurfLast->points[j].intensity) <=
              closestPointScanID &&
              pointSqDis < minPointSqDis2)
            {
              minPointSqDis2 = pointSqDis;
              minPointInd2 = j;
            }
            // if in the higher scan line
            else if (int(laserCloudSurfLast->points[j].intensity) >
              closestPointScanID &&
              pointSqDis < minPointSqDis3)
            {
              minPointSqDis3 = pointSqDis;
              minPointInd3 = j;
            }
          }

          // search in the direction of decreasing scan line
          for (int j = closestPointInd - 1; j >= 0; --j) {
            // if not in nearby scans, end the loop
            if (int(laserCloudSurfLast->points[j].intensity) <
              (closestPointScanID - NEARBY_SCAN))
            {
              break;
            }

            double pointSqDis =
              (laserCloudSurfLast->points[j].x - pointSel.x) *
              (laserCloudSurfLast->points[j].x - pointSel.x) +
              (laserCloudSurfLast->points[j].y - pointSel.y) *
              (laserCloudSurfLast->points[j].y - pointSel.y) +
              (laserCloudSurfLast->points[j].z - pointSel.z) *
              (laserCloudSurfLast->points[j].z - pointSel.z);

            // if in the same or higher scan line
            if (int(laserCloudSurfLast->points[j].intensity) >=
              closestPointScanID &&
              pointSqDis < minPointSqDis2)
            {
              minPointSqDis2 = pointSqDis;
              minPointInd2 = j;
            } else if (int(laserCloudSurfLast->points[j].intensity) <
              closestPointScanID &&
              pointSqDis < minPointSqDis3)
            {
              // find nearer point
              minPointSqDis3 = pointSqDis;
              minPointInd3 = j;
            }
          }

          if (minPointInd2 >= 0 && minPointInd3 >= 0) {

            Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
              surfPointsFlat->points[i].y,
              surfPointsFlat->points[i].z);
            Eigen::Vector3d last_point_a(
              laserCloudSurfLast->points[closestPointInd].x,
              laserCloudSurfLast->points[closestPointInd].y,
              laserCloudSurfLast->points[closestPointInd].z);
            Eigen::Vector3d last_point_b(
              laserCloudSurfLast->points[minPointInd2].x,
              laserCloudSurfLast->points[minPointInd2].y,
              laserCloudSurfLast->points[minPointInd2].z);
            Eigen::Vector3d last_point_c(
              laserCloudSurfLast->points[minPointInd3].x,
              laserCloudSurfLast->points[minPointInd3].y,
              laserCloudSurfLast->points[minPointInd3].z);

            double s;
            if (DISTORTION) {
              s = (surfPointsFlat->points[i].intensity -
                int(surfPointsFlat->points[i].intensity)) /
                SCAN_PERIOD;
            } else {
              s = 1.0;
            }
            ceres::CostFunction * cost_function = LidarPlaneFactor::Create(
              curr_point, last_point_a, last_point_b, last_point_c, s);
            problem.AddResidualBlock(
              cost_function, loss_function,
              se3_last_curr.head(4).data(),
              se3_last_curr.tail(3).data());
            plane_correspondence++;
          }
        }
      } // find correspondence for plane

      lidar_only = true; // already tested, scale weird if set false
      float visual_weight =
        std::min(
        150.0 / (corner_correspondence + plane_correspondence),
        (double)VISUAL_WEIGHT_MAX);
      if ((corner_correspondence + plane_correspondence) < 20) {
        printf(
          "less  lidar correspondence! "
          "*************************************************\n");
        // lidar_only = false;
      } // test
      // find correspondence for each cur point features
      if (imagePointsCurNum > 10) {
        for (int i = 0; i < imagePointsCurNum; ++i) {

          ImagePoint pointSelCur = imagePointsCur->points[i];
          // ImagePoint pointSelLast;// = imagePointsLast->points[i];
          Eigen::Vector3d lP;
          Eigen::Vector3d dP;
          Eigen::Quaterniond lq(poseSequence.back().pose.head(4).data());
          Eigen::Vector3d lt(poseSequence.back().pose.tail(3).data());
          bool is_found = false;
          // for (int j = 0; j < imagePointsLastNum; ++j) {
          auto it = idPointFeatureMap.find(pointSelCur.index);
          if (it != idPointFeatureMap.end() && it->second.depth_type > 0 &&
            it->second.feature_per_frame.size() > 3)
          {
            // if (pointSelCur.index == imagePointsLast->points[j].index) {
            int ef = it->second.endFrame();
            int df = it->second.depthFrame();
            int dfi = it->second.depthFrameIndex();
            int ndi = it->second.nextDepthIndex();

            double d = it->second.inverse_depth[0];
            if (d > 0) {
              d = 1.0 / d;
            } else {
              d = 0;
            }

            if (/*ef-df <= 60 &&*/ d > 2 && d < 60) {
              // TODO: is it necessary?
              Eigen::Quaterniond dq(poseSequence[df].pose.head(4).data());
              Eigen::Vector3d dt(poseSequence[df].pose.tail(3).data());
              Eigen::Vector2d duv = it->second.feature_per_frame[dfi].uv;
              m_camera->liftProjective(duv, dP);
              dP *= d;
              dP = dq * dP + dt;             // transform to world frame
              lP = lq.inverse() * (dP - lt); // transform to camera frame
              // lP = d>0 ? lq.inverse()*(dP-lt) : Eigen::Vector3d::Zero(); //
              // transform to camera frame
            } else if (ndi > 0) {

              int ndf = it->second.feature_per_frame[ndi].frame_id;
              Eigen::Quaterniond ndq(poseSequence[ndf].pose.head(4).data());
              Eigen::Vector3d ndt(poseSequence[ndf].pose.tail(3).data());
              Eigen::Vector2d nduv = it->second.feature_per_frame[ndi].uv;
              m_camera->liftProjective(nduv, dP);
              d = it->second.feature_per_frame[ndi].inverse_depth;
              if (d > 0) {
                d = 1.0 / d;
              } else {
                d = 0;
              }

              dP *= d;
              dP = ndq * dP + ndt; // transform to world frame
              lP = (d > 0 && d < 60) ?
                lq.inverse() * (dP - lt) :
                Eigen::Vector3d::Zero();          // transform to camera frame

            } else {
              lP = Eigen::Vector3d::Zero();
              // lP = it->second.feature_per_frame[ef-sf].point;
            }

            if (lP[2] > 0 && lP[2] < 60) {
              is_found = true;
            } else {
              // ROS_ERROR_STREAM("check point " << lP.transpose() << " ; " <<
              // d << " ; " << dP.transpose());
            }
            // break;
            // }
          }

          if (is_found) {
            // 3d-2d case

            Eigen::Vector2d cp;
            cp(0) = pointSelCur.u;
            cp(1) = pointSelCur.v;
            Eigen::Vector3d cP;
            m_camera->liftProjective(cp, cP);
            std::vector<double> m_intrinsic_params;
            m_camera->writeParameters(m_intrinsic_params);

            ceres::CostFunction * cost_function =
              PointPosition33FactorQT::Create(lP, cP, visual_weight);
            problem.AddResidualBlock(
              cost_function, loss_function,
              se3_last_curr.head(4).data(),
              se3_last_curr.tail(3).data());
            point_correspondence++;
          }
        }
      } else {
        if (DEBUG) {
          ROS_INFO_STREAM("Few point feature. ignore!");
        }
      }
      //} // end lidar only

      if ((corner_correspondence + plane_correspondence +
        point_correspondence) < 10)
      {
        printf(
          "less correspondence! "
          "*************************************************\n");
      } else {
      }

      // solve ceres problem
      // ROS_INFO_STREAM("Begin slove pose");
      ceres::Solver::Options options;

      options.dynamic_sparsity = true;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      // options.linear_solver_type = ceres::DENSE_QR;
      // options.trust_region_strategy_type = ceres::DOGLEG;
      // options.dogleg_type = ceres::SUBSPACE_DOGLEG;
      options.max_num_iterations = 8;
      options.minimizer_progress_to_stdout = false;

      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      // std::cout << summary.BriefReport() << "\n";

    } // optimization loop

    Eigen::Quaterniond q_last_curr(se3_last_curr.head(4).data());
    Eigen::Vector3d t_last_curr(se3_last_curr.tail(3).data());
    Eigen::Quaterniond last_q(
      poseSequence[poseSequence.size() - 1].pose.head(4).data());
    Eigen::Vector3d last_t(
      poseSequence[poseSequence.size() - 1].pose.tail(3).data());
    t_w_curr = last_t + last_q * t_last_curr;
    q_w_curr = last_q * q_last_curr;

    // PoseType temp_pose = PoseType(t_w_curr, q_w_curr);
  }

  end = clock();
  std::string home_dir = std::getenv("HOME");
  outfile.open(
    (home_dir + std::string("/gacm_output/timecost/pose_estimate_time.txt"))
    .c_str(),
    ios::app);
  outfile << (double)(end - start) / CLOCKS_PER_SEC << "\n";
  outfile.close();

  // generate odometry message to be published
  generateOdomMessage();

  // save history result
  saveHistoryMessage();

  // update posegraph
  updatePoseSequence();

  mapDataPerFrameBuf.push_back(
    MappingDataPerFrame(
      laserFrameCountFromBegin, timeSurfPointsLessFlat, rgb_last, depth_map));

  updatePointFeatureMap();

  if (!RUN_ODOMETRY) {
    start = clock();

    optimizeMap(lidar_only);

    end = clock();
    outfile.open(
      (home_dir + std::string("/gacm_output/timecost/landmark_time.txt"))
      .c_str(),
      ios::app);
    outfile << (double)(end - start) / CLOCKS_PER_SEC << "\n";
    outfile.close();
  }

  q_w_curr_hfreq = q_w_curr;
  t_w_curr_hfreq = t_w_curr;

  // check outliers after optimization
  updateLandmark();

  generateLaserMessage();

  laserFrameCountFromBegin++;
}

void PoseEstimator::handleImage()
{
  need_pub_odom = false;
  if (!systemInited) {
    // should wait for the first laser frame and image frame together

    // if all buf have message
    if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
      !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
      !fullPointsBuf.empty() && !imageRawBuf.empty() &&
      !featurePointsBuf.empty() &&
      (RUN_ODOMETRY || !poseStampedBuf.empty()))
    {
      if (DEBUG) {
        ROS_ERROR_STREAM("Try initialization");
      }
      // fetch timestamp from messages
      fetchAllTimeStamp();

      // if message not synced
      if (!checkAllMessageSynced()) {
        // drop the first few messages
        if (timeImageRaw < timeLaserCloudFullRes) {
          mBuf.lock();
          imageRawBuf.pop();
          featurePointsBuf.pop();
          mBuf.unlock();
        } else if (timeImageRaw > timeLaserCloudFullRes) {
          mBuf.lock();
          cornerSharpBuf.pop();
          cornerLessSharpBuf.pop();
          surfFlatBuf.pop();
          surfLessFlatBuf.pop();
          fullPointsBuf.pop();
          mBuf.unlock();
        }
        if (!RUN_ODOMETRY && timePoseStamped < timeLaserCloudFullRes) {
          mBuf.lock();
          poseStampedBuf.pop();
          mBuf.unlock();
        } else if (!RUN_ODOMETRY && timePoseStamped > timeLaserCloudFullRes) {
          mBuf.lock();
          cornerSharpBuf.pop();
          cornerLessSharpBuf.pop();
          surfFlatBuf.pop();
          surfLessFlatBuf.pop();
          fullPointsBuf.pop();
          if (timeImageRaw < timeLaserCloudFullRes) {
            imageRawBuf.pop();
            featurePointsBuf.pop();
          }
          mBuf.unlock();
        }
        return; // time not synced wait for next loop
      }
      fetchAllFromBuf();
      estimatePoseWithImageAndLaser(false);

      systemInited = true;
      need_pub_odom = true;
      if (DEBUG) {
        ROS_ERROR_STREAM("Initialization finished");
      }
    } // end if all buf have message
  }   // end if system not initialized
  else {
    // system is initialized, do pose estimation
    if (!imageRawBuf.empty() && !featurePointsBuf.empty()) {
      if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
        !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
        !fullPointsBuf.empty() && (RUN_ODOMETRY || !poseStampedBuf.empty()))
      {
        // fetch timestamp from messages
        fetchAllTimeStamp();
        // check synced
        if (!checkAllMessageSynced()) {
          if (timeImageRaw < timeLaserCloudFullRes) {
            mBuf.lock();
            imageRawBuf.pop();
            featurePointsBuf.pop();
            mBuf.unlock();
          } else if (timeImageRaw > timeLaserCloudFullRes) {
            mBuf.lock();
            cornerSharpBuf.pop();
            cornerLessSharpBuf.pop();
            surfFlatBuf.pop();
            surfLessFlatBuf.pop();
            fullPointsBuf.pop();
            mBuf.unlock();
          }
          if (!RUN_ODOMETRY && timePoseStamped < timeLaserCloudFullRes) {
            mBuf.lock();
            poseStampedBuf.pop();
            mBuf.unlock();
          } else if (!RUN_ODOMETRY && timePoseStamped > timeLaserCloudFullRes) {
            mBuf.lock();
            cornerSharpBuf.pop();
            cornerLessSharpBuf.pop();
            surfFlatBuf.pop();
            surfLessFlatBuf.pop();
            fullPointsBuf.pop();
            if (timeImageRaw < timeLaserCloudFullRes) {
              imageRawBuf.pop();
              featurePointsBuf.pop();
            }
            mBuf.unlock();
          }
        } else {
          // pose estimation using laser & image
          fetchAllFromBuf();
          estimatePoseWithImageAndLaser(true);
          // ROS_ERROR_STREAM("estimatePoseWithImageAndLaser");
          need_pub_odom = true;
        }
      }
    } // end if image not empty
  }
}

void PoseEstimator::handleImage(
  const cv::Mat & feature_image, pcl::PointCloud<ImagePoint>::Ptr keypoints,
  pcl::PointCloud<PointType>::Ptr laser_cloud_sharp,
  pcl::PointCloud<PointType>::Ptr laser_cloud_less_sharp,
  pcl::PointCloud<PointType>::Ptr laser_cloud_flat,
  pcl::PointCloud<PointType>::Ptr laser_cloud_less_flat,
  pcl::PointCloud<PointType>::Ptr laser_full, const Eigen::Vector3d & position,
  const Eigen::Quaterniond & orientation, int64_t timestamp_ns)
{
  rgb_cur = feature_image.clone();
  imagePointsCur = std::move(keypoints);
  imagePointsCurNum = imagePointsCur->size();
  cornerPointsSharp = std::move(laser_cloud_sharp);
  cornerPointsLessSharp = std::move(laser_cloud_less_sharp);
  surfPointsFlat = std::move(laser_cloud_flat);
  surfPointsLessFlat = std::move(laser_cloud_less_flat);
  laserCloudFullResCur = std::move(laser_full);
  if (!RUN_ODOMETRY) {
    t_w_curr = position;
    q_w_curr = orientation;
  }

  double timestamp_s = timestamp_ns * 1e-9;
  timeImageRaw = timeCornerPointsSharp = timeCornerPointsLessSharp =
    timeSurfPointsFlat = timeSurfPointsLessFlat = timeLaserCloudFullRes =
    timestamp_s;

  if (systemInited) {
    estimatePoseWithImageAndLaser(true);
    need_pub_odom = true;
  } else {
    estimatePoseWithImageAndLaser(false);
    systemInited = true;
    need_pub_odom = true;
  }
}

const nav_msgs::Path & PoseEstimator::getLaserPath2()
{
  laserPath2.poses.clear();
  for (size_t i = 0; i < poseSequence.size(); i++) {
    geometry_msgs::PoseStamped laserPose2;

    Eigen::Quaterniond q(poseSequence[i].pose.head(4).data());
    Eigen::Vector3d t(poseSequence[i].pose.tail(3).data());

    laserPose2.header = laserPath.poses[i].header;
    laserPose2.pose.orientation.x = q.x();
    laserPose2.pose.orientation.y = q.y();
    laserPose2.pose.orientation.z = q.z();
    laserPose2.pose.orientation.w = q.w();
    laserPose2.pose.position.x = t.x();
    laserPose2.pose.position.y = t.y();
    laserPose2.pose.position.z = t.z();
    // laserPose2.header.stamp = laserPath.poses[i].header.stamp;
    // laserPose2.header = laserOdometry.header;
    // laserPose2.header.child_frame_id = "/laser_odom";
    laserPath2.header.stamp = laserOdometry.header.stamp;
    laserPath2.poses.push_back(laserPose2);
    laserPath2.header.frame_id = "camera";
  }

  return laserPath2;
}

const sensor_msgs::PointCloud2 & PoseEstimator::getMapPointCloud()
{
  // sensor_msgs::PointCloud2 points;//(new sensor_msgs::PointCloud2);
  pointFeatureCloudMsg.header.stamp = ros::Time().fromSec(timeImageRaw);
  pointFeatureCloudMsg.header.frame_id = "camera";

  pointFeatureCloudMsg.data.clear();

  pointFeatureCloudMsg.height = 1;
  pointFeatureCloudMsg.width = idPointFeatureMap.size();
  pointFeatureCloudMsg.fields.resize(4);
  pointFeatureCloudMsg.fields[0].name = "x";
  pointFeatureCloudMsg.fields[0].offset = 0;
  pointFeatureCloudMsg.fields[0].count = 1;
  pointFeatureCloudMsg.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
  pointFeatureCloudMsg.fields[1].name = "y";
  pointFeatureCloudMsg.fields[1].offset = 4;
  pointFeatureCloudMsg.fields[1].count = 1;
  pointFeatureCloudMsg.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
  pointFeatureCloudMsg.fields[2].name = "z";
  pointFeatureCloudMsg.fields[2].offset = 8;
  pointFeatureCloudMsg.fields[2].count = 1;
  pointFeatureCloudMsg.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
  pointFeatureCloudMsg.fields[3].name = "rgb";
  pointFeatureCloudMsg.fields[3].offset = 12;
  pointFeatureCloudMsg.fields[3].count = 1;
  pointFeatureCloudMsg.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
  // pointFeatureCloudMsg.is_bigendian = false; ???
  pointFeatureCloudMsg.point_step = 16;
  pointFeatureCloudMsg.row_step =
    pointFeatureCloudMsg.point_step * pointFeatureCloudMsg.width;
  pointFeatureCloudMsg.data.resize(
    pointFeatureCloudMsg.row_step *
    pointFeatureCloudMsg.height);
  pointFeatureCloudMsg.is_dense = false; // there may be invalid points

  float bad_point = std::numeric_limits<float>::quiet_NaN(); // for invalid part
  int counter = 0;
  for (auto it = idPointFeatureMap.begin(); it != idPointFeatureMap.end();
    it++, counter++)
  {
    if (it->second.depth_type > 0) {
      int32_t rgb = ((uint8_t)78 << 16) | ((uint8_t)238 << 8) | (uint8_t)148;
      double d = it->second.inverse_depth[0];
      if (d > 0) {
        d = 1.0 / d;
      } else {
        d = 0;
        continue;
      }
      int df = it->second.depthFrame();
      int dfi = it->second.depthFrameIndex();
      Eigen::Quaterniond dq(poseSequence[df].pose.head(4).data());
      Eigen::Vector3d dt(poseSequence[df].pose.tail(3).data());
      Eigen::Vector2d duv = it->second.feature_per_frame[dfi].uv;
      Eigen::Vector3d dP;
      m_camera->liftProjective(duv, dP);
      dP *= d;
      dP = dq * dP + dt;
      float x = (float)dP(0);
      float y = (float)dP(1);
      float z = (float)dP(2);
      memcpy(
        &pointFeatureCloudMsg
        .data[counter * pointFeatureCloudMsg.point_step + 0],
        &x, sizeof(float));
      memcpy(
        &pointFeatureCloudMsg
        .data[counter * pointFeatureCloudMsg.point_step + 4],
        &y, sizeof(float));
      memcpy(
        &pointFeatureCloudMsg
        .data[counter * pointFeatureCloudMsg.point_step + 8],
        &z, sizeof(float));
      memcpy(
        &pointFeatureCloudMsg
        .data[counter * pointFeatureCloudMsg.point_step + 12],
        &rgb, sizeof(int32_t));
    } else {
      memcpy(
        &pointFeatureCloudMsg
        .data[counter * pointFeatureCloudMsg.point_step + 0],
        &bad_point, sizeof(float));
      memcpy(
        &pointFeatureCloudMsg
        .data[counter * pointFeatureCloudMsg.point_step + 4],
        &bad_point, sizeof(float));
      memcpy(
        &pointFeatureCloudMsg
        .data[counter * pointFeatureCloudMsg.point_step + 8],
        &bad_point, sizeof(float));
      memcpy(
        &pointFeatureCloudMsg
        .data[counter * pointFeatureCloudMsg.point_step + 12],
        &bad_point, sizeof(float));
    }
  }

  return pointFeatureCloudMsg;
}

void PoseEstimator::publishMappingTopics()
{
  // mapping buffer size bigger than pose optimization window size
  if (mapDataPerFrameBuf.size() > 10) {
    // ROS_ERROR_STREAM("#########MappingTopics###########");
    cv_bridge::CvImage bridge;

    bridge.image = mapDataPerFrameBuf[0].rgb_image;
    bridge.encoding = sensor_msgs::image_encodings::RGB8;
    rgb_image_ptr = bridge.toImageMsg();
    rgb_image_ptr->header.stamp =
      ros::Time().fromSec(mapDataPerFrameBuf[0].timestamp);
    int mf = mapDataPerFrameBuf[0].frame_id;
    Eigen::Quaterniond q_mf(poseSequence[mf].pose.head(4).data());
    Eigen::Vector3d t_mf(poseSequence[mf].pose.tail(3).data());

    cv::Mat depthmap = mapDataPerFrameBuf[0].depth_image;

    bridge.image = depthmap;
    bridge.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
    depth_image_ptr = bridge.toImageMsg();
    depth_image_ptr->header.stamp = rgb_image_ptr->header.stamp;

    mapping_pose.header.frame_id = "camera";
    mapping_pose.header.stamp = rgb_image_ptr->header.stamp;
    mapping_pose.pose.position.x = t_mf(0);
    mapping_pose.pose.position.y = t_mf(1);
    mapping_pose.pose.position.z = t_mf(2);
    mapping_pose.pose.orientation.x = q_mf.x();
    mapping_pose.pose.orientation.y = q_mf.y();
    mapping_pose.pose.orientation.z = q_mf.z();
    mapping_pose.pose.orientation.w = q_mf.w();

    mapDataPerFrameBuf.pop_front();
    frameTobeMap++;
    mapping_ready = true;
    // end if
  } else {
    mapping_ready = false;
  }
}
