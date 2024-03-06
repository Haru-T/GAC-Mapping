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

#ifndef GACM__MAPPING__SUB_MAP_H_
#define GACM__MAPPING__SUB_MAP_H_

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "sensor_msgs/PointCloud2.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "ros/ros.h"

#include "gacm/ga_posegraph/poseGraphStructure.h"

using PointType = pcl::PointXYZI;

class SubMap : public std::enable_shared_from_this<SubMap>
{
  // private:

public:
  int robot_id;  // which session
  int submap_id; // which submap
  bool is_drone;
  ros::NodeHandle nh;

  int frameCount = 0;
  int access_status = 0; // -1 cached, >=0 active(recently accessed by submap N)

  int laserCloudCenWidth = 52; // 52
  int laserCloudCenHeight = 52;
  int laserCloudCenDepth = 52;
  const int laserCloudWidth = 105; // 105
  const int laserCloudHeight = 105;
  const int laserCloudDepth = 105;
  float halfCubeSize = 5.0;     // 5
  float cubeSize = 10.0;        // 10
  int halfSurroundCubeNum = 13; // 13
  int halfCubeForMapping = 10;

  const int laserCloudNum =
    laserCloudWidth * laserCloudHeight * laserCloudDepth;
  int laserCloudSurroundNum = 0;
  int laserCloudValidInd[15625];
  int laserCloudSurroundInd[15625];

  // 点云ptr初始化形式参考pose estimator
  // input: from odom
  pcl::PointCloud<PointType>::Ptr
    laserCloudCornerLast;   //(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr
    laserCloudSurfLast;   // (new pcl::PointCloud<PointType>());

  // ouput: all visualble cube points
  pcl::PointCloud<PointType>::Ptr
    laserCloudSurround;   //(new pcl::PointCloud<PointType>());

  // surround points in map to build tree
  pcl::PointCloud<PointType>::Ptr
    laserCloudCornerFromMap;   //(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr
    laserCloudSurfFromMap;   //(new pcl::PointCloud<PointType>());

  // input & output: points in one frame. local --> global
  pcl::PointCloud<PointType>::Ptr
    laserCloudFullRes;   //(new pcl::PointCloud<PointType>());

  // 这里初始化要看一下
  // points in every cube
  std::vector<pcl::PointCloud<PointType>::Ptr>
  laserCloudCornerArray;     // [laserCloudNum];
  std::vector<pcl::PointCloud<PointType>::Ptr>
  laserCloudSurfArray;     //[laserCloudNum];

  // kd-tree
  pcl::KdTreeFLANN<PointType>::Ptr
    kdtreeCornerFromMap;   //(new pcl::KdTreeFLANN<PointType>());
  pcl::KdTreeFLANN<PointType>::Ptr
    kdtreeSurfFromMap;   //(new pcl::KdTreeFLANN<PointType>());

  double stampLatestOdom = 0;
  double timeLaserOdometry = 0;
  double timeLaserOdometryLast = 0;
  // int frame_counter = 0;

  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  pcl::VoxelGrid<PointType> downSizeFilterSurf;

  std::vector<int> pointSearchInd;
  std::vector<float> pointSearchSqDis;

  PointType pointOri, pointSel;

  ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes,
    pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath,
    pubMatchSurround, pubMatchPose;
  ros::Publisher pubSubmapThumbnail;

  nav_msgs::Path laserAfterMappedPath;

  mutable std::mutex mBuf;
  mutable std::mutex mCloudGrid;

  bool needNewSubMap = false;
  bool alignedToGlobal = false; // first loop arrived?
  bool initWithPose = false;
  bool adjust_finish = false; // shift cube flag
  bool thumbnailGenerated = false;

  std::shared_ptr<SubMap> next_submap_ptr;
  // pcl::PointCloud<PointType>::Ptr thumbnailLast;
  pcl::PointCloud<PointType>::Ptr thumbnailCur; // ground
  // pcl::PointCloud<PointType>::Ptr thumbnailBCur; // building
  Eigen::Vector4f coeffs; // plane norm

  // public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Quaterniond q_world_base; //(parameters);
  Eigen::Vector3d t_world_base;    //(parameters + 4);

  // transformation from base node to current frame
  Eigen::Quaterniond q_base_curr; //(1, 0, 0, 0);
  Eigen::Vector3d t_base_curr;    //(0, 0, 0);

  // relative transformation from last frame[received]
  Eigen::Quaterniond q_last_curr; //(1, 0, 0, 0);
  Eigen::Vector3d t_last_curr;    //(0, 0, 0);

  // relative transformation from last frame[received]
  Eigen::Quaterniond q_odom_last; //(1, 0, 0, 0);
  Eigen::Vector3d t_odom_last;    //(0, 0, 0);

  // transformation from world to current frame
  Eigen::Quaterniond q_world_curr; //(1, 0, 0, 0);
  Eigen::Vector3d t_world_curr;    //(0, 0, 0);

  // translation from odometry frame to map frame
  Eigen::Quaterniond q_odom_map;
  Eigen::Vector3d t_odom_map;

  // 内部维护一段posegraph
  std::vector<PoseNode, Eigen::aligned_allocator<PoseNode>> pose_graph_local;
  std::vector<MeasurementEdge, Eigen::aligned_allocator<MeasurementEdge>>
  edges_local;

  std::vector<std::pair<int, std::string>> thumbnails_db;
  std::vector<std::pair<int, Eigen::VectorXf>> descriptor_db;

  // pcl::PointCloud<PointType>::Ptr laserCloudSurfStack;
  // pcl::PointCloud<PointType>::Ptr laserCloudCornerStack;

  SubMap(ros::NodeHandle nh_);

  void initParameters(
    int robot_id_, int submap_id_, bool is_drone_,
    double base_stamp_);

  // check, is parameter enough?
  SubMap(
    int robot_id_, int submap_id_, bool is_drone_,
    const Eigen::Quaterniond & q_world_base_,
    const Eigen::Vector3d & t_world_base_,
    const Eigen::Quaterniond & q_odom_last_,
    const Eigen::Vector3d & t_odom_last_, double base_stamp_,
    ros::NodeHandle nh_);

  SubMap(
    const std::string & map_path, int robot_id_, int submap_id_,
    bool is_drone_, ros::NodeHandle nh_);

  // deconstructor
  ~SubMap()
  {
    // todo?
  }
  void setActive(int idx);

  void setCache(bool force = false);

  void saveMapToFile(const std::string & map_path) const;

  void loadMapFromFile(const std::string & map_path);

  void allocateMemory();

  std::shared_ptr<SubMap> createNewSubMap();

  void pointAssociateToMap(
    PointType const * const pi,
    PointType * const po) const;

  void pointAssociateTobeMapped(
    PointType const * const pi,
    PointType * const po) const;

  bool checkNeedNewSubMap()
  {
    // return false; //  skip map spilt
    return needNewSubMap;
  }

  void process(
    sensor_msgs::PointCloud2ConstPtr cornerLastBufFrontPtr,
    sensor_msgs::PointCloud2ConstPtr surfLastBufFrontPtr,
    sensor_msgs::PointCloud2ConstPtr fullResBufFrontPtr,
    nav_msgs::Odometry::ConstPtr odometryBufFrontPtr);

  void process(
    pcl::PointCloud<PointType>::Ptr laser_cloud_corner_last,
    pcl::PointCloud<PointType>::Ptr laser_cloud_surf_last,
    pcl::PointCloud<PointType>::Ptr laser_full_3,
    const Eigen::Vector3d & position,
    const Eigen::Quaterniond & orientation, int64_t timestamp_ns);

  void publishMap(int laserCloudSurroundNum = -1);

  pcl::PointCloud<PointType>::Ptr getMapCloud(
    bool in_world_frame = false,
    bool get_surround = false) const;

  void publishSurroundMap(int laserCloudSurroundNum);

  /**
   * @brief
   *
   * @param headless disable visualization, maybe useful when load from file
   * @return ** void
   */
  void generateThumbnail(bool headless = false);

  void getWorldFramePoseAt(int fid, Eigen::Quaterniond & q, Eigen::Vector3d & t)
  {
    q = pose_graph_local[fid].q * q_world_base;
    t = t_world_base + q_world_base * pose_graph_local[fid].t;
  }
};

#endif  // GACM__MAPPING__SUB_MAP_H_
