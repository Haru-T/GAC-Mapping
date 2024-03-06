#include "gacm/uninode/submap_management.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>

#include "cv_bridge/cv_bridge.h"
#include "nav_msgs/Path.h"
#include "netvlad_tf_test/CompactImg.h"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Empty.h"
#include "visualization_msgs/Marker.h"

#include "gacm/ga_posegraph/poseGraphStructure.h"
#include "gacm/loop_closure/loop_score.hpp"
#include "gacm/mapping/sub_map.h"
#include "gacm/parameters.h"

namespace gacm
{
struct SubMapManagementObject
  : std::enable_shared_from_this<SubMapManagementObject>
{
  size_t cur_robot_id = 0;
  std::array<std::vector<std::shared_ptr<SubMap>>, 5> MapDatabase;
  std::array<bool, 6> is_drone{};
  std::size_t cur_map_id = 0;
  std::size_t cur_thumbnail_id = 0;
  int cur_loop_test_id = 1;
  std::int64_t timestamp_nanoseconds = 0;

  std::mutex m_loop{};

  ros::NodeHandle * nh_ref;

  bool need_init_coord;
  bool updating_pg;
  bool updating_map;

  std::vector<LoopScore> loop_scores;
  std::vector<MeasurementEdge, Eigen::aligned_allocator<MeasurementEdge>>
  loopEdgeBuf;
  std::vector<MeasurementEdge, Eigen::aligned_allocator<MeasurementEdge>>
  dropEdgeBuf;

  std::array<ros::Publisher, 6> pubPathOpt;
  ros::Publisher pubRelEdge;
  std::array<ros::Publisher, 6> pubOptimizedMap;
  ros::Publisher pubQuery;
  ros::Publisher pubDetectL;
  ros::ServiceClient netvlad_client;

  pcl::KdTreeFLANN<PointType>::Ptr kdtree;
  std::array<pcl::PointCloud<PointType>::Ptr, 6> submap_base_map;

  bool is_first = true;
  int last_loop_size = 0;

  std::thread cache_thread;
  std::atomic_bool continue_cache_thread;

  SubMapManagementObject(ros::NodeHandle & nh)
  : is_drone{static_cast<bool>(NEED_CHECK_DIRECTION)}, nh_ref(&nh),
    pubRelEdge(nh.advertise<visualization_msgs::Marker>(
        "/relative_observation", 100)),
    pubQuery(nh.advertise<sensor_msgs::Image>("/query_thumb", 10)),
    pubDetectL(nh.advertise<sensor_msgs::Image>("/database_thumb", 10)),
    netvlad_client(
      nh.serviceClient<netvlad_tf_test::CompactImg>("/compact_image")),
    kdtree(new pcl::KdTreeFLANN<PointType>()),
    continue_cache_thread{true}
  {
    MapDatabase[0].push_back(std::make_shared<SubMap>(nh));
    MapDatabase[0][0]->initParameters(
      0, 0, static_cast<bool>(NEED_CHECK_DIRECTION), 0);
    for (size_t i = 0; i < pubPathOpt.size(); ++i) {
      std::stringstream path_topic;
      path_topic << "/path_opt" << i;
      pubPathOpt[i] = nh.advertise<nav_msgs::Path>(path_topic.str(), 100);
    }
    for (size_t i = 0; i < pubOptimizedMap.size(); ++i) {
      std::stringstream map_topic;
      map_topic << "/optimized_map" << i;
      pubOptimizedMap[i] = nh.advertise<sensor_msgs::PointCloud2>(map_topic.str(), 100);
    }
    submap_base_map[0].reset(new pcl::PointCloud<PointType>());
  }

  ~SubMapManagementObject() noexcept
  {
    continue_cache_thread = false;
    if (cache_thread.joinable()) {cache_thread.join();}
  }

  void process(SubMapManagementInput && input);
  void map_process(SubMapManagementInput && input);
  void thumbnail_process();
  void cache_process();

  void optimize();
  void update_submapbase_cloud();
  void publish_opt_posegraph(bool pubmap = false);
  void publish_opt_map(bool pubmap = false, bool pubmerge = false);
  void optimize_globally();

  void export_odom();
  void save_all();
  void pub_loop_image(
    cv::Mat & query_tn, cv::Mat & response_tn,
    bool accepted) const;
  void test_base_score(
    const pcl::PointCloud<PointType>::Ptr & all_submap_bases,
    std::vector<int> & knn_index,
    std::vector<float> & knn_distance);
  std::pair<float, Eigen::Matrix4f>
  test_match_cloud(
    const pcl::PointCloud<PointType>::Ptr & cloud_src,
    const pcl::PointCloud<PointType>::Ptr & cloud_target,
    int ri_test, int si_test, const Eigen::Matrix4f & try_guess);

  void check_cachable_map();
};

std::shared_ptr<SubMapManagementObject>
spawn_submap_management_object(ros::NodeHandle & nh)
{
  auto p = std::make_shared<SubMapManagementObject>(nh);
  p->cache_thread = std::thread(&SubMapManagementObject::cache_process, p.get());
  return p;
}

void new_session(const std::shared_ptr<SubMapManagementObject> & obj)
{
  std::lock_guard<std::mutex> lk_loop{obj->m_loop};
  obj->thumbnail_process();
  obj->thumbnail_process();
  obj->optimize();
  obj->save_all();
  if (obj->cur_robot_id > 0) {
    if (DEBUG) {
      std::cout << "\033[1;31m\nodom before opt exported, please "
        "check\n\033[0m";
    }
    obj->optimize_globally();
  }
  obj->export_odom();
  std::cout << "\033[1;32m\nSave odom and global optimization Done! "
    "\n\033[0m";
  std::cout << "\033[1;32m\nCurrent robot[" << obj->cur_robot_id
            << "] end, ready to receive robot[" << obj->cur_robot_id + 1
            << "] data!\n\033[0m";

  ++obj->cur_robot_id;
  obj->is_drone[obj->cur_robot_id] = static_cast<bool>(NEED_CHECK_DIRECTION);
  obj->MapDatabase[obj->cur_robot_id].clear();
  obj->MapDatabase[obj->cur_robot_id].push_back(
    std::make_shared<SubMap>(*obj->nh_ref));
  obj->kdtree.reset(new pcl::KdTreeFLANN<PointType>());
  obj->submap_base_map[obj->cur_robot_id].reset(
    new pcl::PointCloud<PointType>());
  obj->cur_map_id = 0;
  obj->cur_thumbnail_id = 0;
  obj->cur_loop_test_id = 0;
  obj->MapDatabase[obj->cur_robot_id][obj->cur_map_id]->initParameters(
    obj->cur_robot_id, 0, static_cast<bool>(NEED_CHECK_DIRECTION), 0);
  obj->need_init_coord = true;
}

void manage_submap(
  SubMapManagementInput && input,
  const std::shared_ptr<SubMapManagementObject> & obj)
{
  obj->process(std::move(input));
}

void SubMapManagementObject::export_odom()
{
  if (NEED_PUB_ODOM == 0) {
    return;
  }
  // std::fstream fs;
  for (int r = 0; r <= cur_robot_id; r++) {
    std::fstream fs;
    std::stringstream ss;
    ss << r;
    fs.open(OUT_ODOM_PATH + ss.str() + ".txt", std::ios::out);
    int database_size = MapDatabase[r].size();

    for (int i = 0; i < database_size - 1; i++) { // the last submap is isolated
      int max_size = MapDatabase[r][i]->pose_graph_local.size();
      for (int j = 1; j < max_size; j++) { // start from 1, skip first node

        Eigen::Quaterniond q = MapDatabase[r][i]->pose_graph_local[j].q;
        Eigen::Vector3d t = MapDatabase[r][i]->pose_graph_local[j].t;

        t = MapDatabase[r][i]->t_world_base +
          MapDatabase[r][i]->q_world_base * t;
        q = MapDatabase[r][i]->q_world_base * q;

        // 输出tum格式数据
        fs << std::fixed << std::setprecision(6)
           << MapDatabase[r][i]->pose_graph_local[j].stamp << " "
           << std::setprecision(7) << t.x() << " " << t.y() << " " << t.z()
           << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
           << "\n";
      }
    }

    fs.close();

    if (DEBUG) {
      std::cout << "Done export\n";
    }
  }
}

void SubMapManagementObject::save_all()
{
  std::puts("\033[1;32m\nBegin save map\n\033[0m");
  if (DEBUG) {
    std::cout << "Current Edges " << loopEdgeBuf.size() << " : "
              << dropEdgeBuf.size() << "\n";
  }

  int database_size = MapDatabase[cur_robot_id].size();
  for (int i = 0; i < database_size; i++) {
    MapDatabase[cur_robot_id][i]->saveMapToFile(
      std::string(std::getenv("HOME")) + "/gacm_output/data/testSavemap/");
  }
  std::fstream fs;
  // 打开并清空文件
  fs.open(
    std::string(std::getenv("HOME")) +
    "/gacm_output/data/testSavemap/robot" +
    std::to_string(cur_robot_id) + "_loop.cfg",
    std::ios::ate | std::ios::out);
  fs << cur_robot_id << "\n";
  fs << is_drone[cur_robot_id] << "\n";
  fs << database_size << "\n";
  fs << loopEdgeBuf.size() << "\n";
  for (auto edge : loopEdgeBuf) {
    fs << std::fixed << edge.stamp_from << " " << edge.stamp_to << " "
       << edge.robot_from << " " << edge.robot_to << " " << edge.submap_from
       << " " << edge.submap_to << " " << edge.index_from << " "
       << edge.index_to << " " << edge.q.w() << " " << edge.q.x() << " "
       << edge.q.y() << " " << edge.q.z() << " " << edge.t.x() << " "
       << edge.t.y() << " " << edge.t.z() << "\n";
  }

  fs << dropEdgeBuf.size() << "\n";
  for (auto edge : dropEdgeBuf) {
    fs << std::fixed << edge.stamp_from << " " << edge.stamp_to << " "
       << edge.robot_from << " " << edge.robot_to << " " << edge.submap_from
       << " " << edge.submap_to << " " << edge.index_from << " "
       << edge.index_to << " " << edge.q.w() << " " << edge.q.x() << " "
       << edge.q.y() << " " << edge.q.z() << " " << edge.t.x() << " "
       << edge.t.y() << " " << edge.t.z() << "\n";
  }
  fs.close();
  std::cout << "\033[1;32m\nSave data Done!\n\033[0m";
}

void SubMapManagementObject::pub_loop_image(
  cv::Mat & query_tn,
  cv::Mat & response_tn,
  bool accepted) const
{
  cv_bridge::CvImage bridge;
  if (accepted) {
    cv::copyMakeBorder(
      query_tn, query_tn, 5, 5, 5, 5, cv::BORDER_CONSTANT,
      cv::Scalar(0, 255, 0));
    cv::copyMakeBorder(
      response_tn, response_tn, 5, 5, 5, 5,
      cv::BORDER_CONSTANT, cv::Scalar(0, 255, 0));
  } else {
    cv::copyMakeBorder(
      query_tn, query_tn, 5, 5, 5, 5, cv::BORDER_CONSTANT,
      cv::Scalar(0, 0, 255));
    cv::copyMakeBorder(
      response_tn, response_tn, 5, 5, 5, 5,
      cv::BORDER_CONSTANT, cv::Scalar(0, 0, 255));
  }
  bridge.image = query_tn;
  bridge.encoding = "bgr8";
  sensor_msgs::Image::Ptr query_image_ptr = bridge.toImageMsg();
  pubQuery.publish(query_image_ptr);
  bridge.image = response_tn;
  sensor_msgs::Image::Ptr response_image_ptr = bridge.toImageMsg();
  pubDetectL.publish(response_image_ptr);
}

void SubMapManagementObject::process(SubMapManagementInput && input)
{
  map_process(std::move(input));
  thumbnail_process();
}

void SubMapManagementObject::map_process(SubMapManagementInput && input)
{
  timestamp_nanoseconds = input.timestamp_nanoseconds;
  {
    std::lock_guard<std::mutex> lk_loop{m_loop};
    MapDatabase[cur_robot_id][cur_map_id]->process(
      input.laser_cloud_corner_last, input.laser_cloud_surf_last,
      input.laser_full_3, input.laser_odom_to_init.position,
      input.laser_odom_to_init.orientation, input.timestamp_nanoseconds);
  }
  if (MapDatabase[cur_robot_id][cur_map_id]->checkNeedNewSubMap()) {
    std::lock_guard<std::mutex> lk_loop{m_loop};
    MapDatabase[cur_robot_id].push_back(
      MapDatabase[cur_robot_id][cur_map_id]->createNewSubMap());
    ++cur_map_id;
  }
}

void SubMapManagementObject::test_base_score(
  const pcl::PointCloud<PointType>::Ptr & all_submap_bases,
  std::vector<int> & knn_index, std::vector<float> & /* knn_distance */)
{
  for (size_t knnid = 0; knnid < knn_index.size(); knnid++) {
    size_t ri = 0;
    size_t i = all_submap_bases->points[knn_index[knnid]].intensity;
    ri = i / 100; // robot id
    i = i % 100;  // submap_id

    if (ri == cur_robot_id && i >= cur_loop_test_id - 5) {
      if (DEBUG) {
        ROS_ERROR_STREAM(
          "Skip test " << i << " (homo) too close to "
                       << cur_loop_test_id);
      }
      continue; // future submap && neighbour submap
    }

    // BUG，补充检查描述子是否为空
    if (MapDatabase[ri][i]->descriptor_db.empty()) {
      continue;
    }

    // descriptor distance
    float desc_score =
      (MapDatabase[ri][i]->descriptor_db[0].second -
      MapDatabase[cur_robot_id][cur_loop_test_id]->descriptor_db[0].second)
      .norm();
    if (DEBUG) {
      ROS_ERROR_STREAM(
        "Desc test " << ri << ":" << i << " and " << cur_robot_id
                     << ":" << cur_loop_test_id << " dist "
                     << desc_score);
    }
    // diff_matrix(cur_loop_test_id, i) = desc_score;
    // base transform
    Eigen::Quaterniond q_rel =
      MapDatabase[cur_robot_id][cur_loop_test_id]->q_world_base.inverse() *
      MapDatabase[ri][i]->q_world_base;
    Eigen::Vector3d t_rel =
      MapDatabase[cur_robot_id][cur_loop_test_id]->q_world_base.inverse() *
      (MapDatabase[ri][i]->t_world_base -
      MapDatabase[cur_robot_id][cur_loop_test_id]->t_world_base);
    double angle_rel = q_rel.angularDistance(Eigen::Quaterniond::Identity());
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f try_guess = Eigen::Matrix4f::Identity();
    try_guess.block(0, 0, 3, 3) =
      q_rel.normalized().toRotationMatrix().cast<float>();
    try_guess.block(0, 3, 3, 1) = t_rel.cast<float>();
    LoopScore loop_score(desc_score, (float)angle_rel, ri, i,
      init_guess /*update_guess*/, try_guess);
    loop_scores.push_back(loop_score);
  }
}

std::pair<float, Eigen::Matrix4f> SubMapManagementObject::test_match_cloud(
  const pcl::PointCloud<PointType>::Ptr & cloud_src,
  const pcl::PointCloud<PointType>::Ptr & cloud_target, int ri_test,
  int si_test, const Eigen::Matrix4f & try_guess)
{

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_src_d1(
    new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_target_d1(
    new pcl::PointCloud<pcl::PointXYZI>);
  // downsample
  pcl::ApproximateVoxelGrid<PointType> avg;
  avg.setLeafSize(0.8F, 0.8F, 0.8F); // trade off
  // avg.setLeafSize(0.5f, 0.5f, 0.5f); // trade off
  avg.setInputCloud(cloud_src);
  avg.filter(*cloud_src);
  avg.setInputCloud(cloud_target);
  avg.filter(*cloud_target);

  avg.setLeafSize(1.8F, 1.8F, 1.8F); // trade off
  avg.setInputCloud(cloud_src);
  avg.filter(*cloud_src_d1);
  avg.setInputCloud(cloud_target);
  avg.filter(*cloud_target_d1);

  pcl::StatisticalOutlierRemoval<PointType> sor;
  sor.setMeanK(5);
  sor.setStddevMulThresh(1.0);
  sor.setInputCloud(cloud_src_d1);
  sor.filter(*cloud_src_d1);
  sor.setInputCloud(cloud_target_d1);
  sor.filter(*cloud_target_d1);

  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();

  // compute init guess using plane norm
  Eigen::Vector4f norm_src = MapDatabase[ri_test][si_test]->coeffs;
  Eigen::Vector4f norm_src_tmp = norm_src;
  norm_src_tmp(3) = 0;
  norm_src_tmp.normalize();
  norm_src.head(3) = norm_src_tmp.head(3);
  Eigen::Vector4f norm_target =
    MapDatabase[cur_robot_id][cur_loop_test_id]->coeffs;
  Eigen::Vector4f norm_target_tmp = norm_target;
  norm_target_tmp(3) = 0;
  norm_target_tmp.normalize();
  norm_target.head(3) = norm_target_tmp.head(3);
  // angle between two ground plane normal
  Eigen::Quaternionf q_temp;
  q_temp.setFromTwoVectors(norm_src.head(3), norm_target.head(3));

  // if you equipment is installed 45 degree down facing, this will help to find
  // the real norm direction.
  if (is_drone[cur_robot_id] == true && is_drone[ri_test] == false) {
    Eigen::Quaternionf q_check;
    q_check.setFromTwoVectors(norm_target.head(3), Eigen::Vector3f(0, 0, 1));
    if (q_check.angularDistance(Eigen::Quaternionf::Identity()) * 57.3 < 90) {
      q_temp.setFromTwoVectors(norm_src.head(3), -1 * norm_target.head(3));
    }
  } else if (is_drone[cur_robot_id] == true && is_drone[ri_test] == true) {
    Eigen::Quaternionf q_check_src;
    Eigen::Quaternionf q_check_tar;
    q_check_src.setFromTwoVectors(norm_src.head(3), Eigen::Vector3f(0, 0, 1));
    q_check_tar.setFromTwoVectors(
      norm_target.head(3),
      Eigen::Vector3f(0, 0, 1));
    int filp_src =
      q_check_src.angularDistance(Eigen::Quaternionf::Identity()) * 57.3 < 90 ?
      -1 :
      1;
    int filp_tar =
      q_check_tar.angularDistance(Eigen::Quaternionf::Identity()) * 57.3 < 90 ?
      -1 :
      1;
    q_temp.setFromTwoVectors(
      filp_src * norm_src.head(3),
      filp_tar * norm_target.head(3));
  }

  // ROS_WARN_STREAM("Q src tar " << norm_src.transpose() << " " <<
  // norm_target.transpose());

  // translation part using pc centroid
  Eigen::Matrix<float, 4, 1> centroid_src;
  pcl::compute3DCentroid(*cloud_src, centroid_src);
  centroid_src =
    centroid_src -
    norm_src_tmp * centroid_src.dot(norm_src);   // project centroid on plane
  centroid_src.head(3) = q_temp * centroid_src.head(3);

  Eigen::Matrix<float, 4, 1> centroid_target;
  pcl::compute3DCentroid(*cloud_target, centroid_target);
  centroid_target =
    centroid_target - norm_target_tmp * centroid_target.dot(norm_target);

  init_guess.topLeftCorner(3, 3) = q_temp.toRotationMatrix();
  init_guess.topRightCorner(3, 1) = (centroid_target - centroid_src).head(3);
  // ROS_WARN_STREAM("init guess \n" << init_guess);

  // try geometry guess
  pcl::NormalDistributionsTransform<PointType, PointType> ndt1;
  ndt1.setResolution(NDT_RESOLUTION_MATCH);
  // ndt1.setMaximumIterations(20);
  ndt1.setInputSource(cloud_src_d1);
  ndt1.setInputTarget(cloud_target_d1);
  pcl::PointCloud<PointType>::Ptr cloud_final_ndt(
    new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr cloud_final_ndt1(
    new pcl::PointCloud<PointType>);
  ndt1.align(*cloud_final_ndt, init_guess);
  Eigen::Matrix4f cur_guess = ndt1.getFinalTransformation();
  float cur_score = ndt1.getFitnessScore();

  // try world frame guess
  // 如果还没有对齐的时候，使用单位阵进行ndt结果很差，可以不用这个猜测进行对齐
  float cur_score_w = 1000;
  Eigen::Matrix4f cur_guess_w = Eigen::Matrix4f::Identity();
  if (!need_init_coord) {
    ndt1.align(*cloud_final_ndt1, try_guess);
    cur_guess_w = ndt1.getFinalTransformation();
    cur_score_w = ndt1.getFitnessScore();
  }

  float final_score = cur_score < cur_score_w ? cur_score : cur_score_w;
  Eigen::Matrix4f final_guess =
    cur_score < cur_score_w ? cur_guess : cur_guess_w;

  pcl::PointCloud<PointType>::Ptr cloud_final_icp(
    new pcl::PointCloud<PointType>);
  pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp2;
  icp2.setInputSource(cloud_src_d1);
  icp2.setInputTarget(cloud_target_d1);
  icp2.setMaximumIterations(50);
  if (!need_init_coord || !is_drone[cur_robot_id]) {
    icp2.setMaxCorrespondenceDistance(5);
  }

  icp2.align(*cloud_final_icp, final_guess);

  if (icp2.hasConverged() && icp2.getFitnessScore() < final_score) {
    final_guess = icp2.getFinalTransformation();
    final_score = icp2.getFitnessScore();
  }

  if (DEBUG) {
    ROS_INFO_STREAM(
      "init guess ndt score: "
        << cur_score << ", try guess score: " << cur_score_w
        << ", icp score: " << icp2.getFitnessScore());
  }
  return std::make_pair(final_score, final_guess);
}

void SubMapManagementObject::thumbnail_process()
{
  if (cur_thumbnail_id < cur_map_id) {
    // mTProcess.lock();
    clock_t start, end;
    start = clock();
    MapDatabase[cur_robot_id][cur_thumbnail_id]->generateThumbnail(true);

    // BUG，补充检查是否包含缩略图
    if (MapDatabase[cur_robot_id][cur_thumbnail_id]->thumbnails_db[0].first ==
      -1)
    {
      if (DEBUG) {
        ROS_WARN_STREAM("no thumbnail get!");
      }
      return;
    }

    end = clock();
    std::ofstream outfile;
    outfile.open(
      std::string(std::getenv("HOME")) +
      "/gacm_output/timecost/thumbnail_time.txt",
      std::ios::app);
    outfile << (double)(end - start) / CLOCKS_PER_SEC << "\n";
    outfile.close();

    // call netvlad service
    netvlad_tf_test::CompactImg img_srv;
    img_srv.request.req_img_name =
      MapDatabase[cur_robot_id][cur_thumbnail_id]->thumbnails_db[0].second;
    int id_tn =
      MapDatabase[cur_robot_id][cur_thumbnail_id]->thumbnails_db[0].first;
    start = clock();
    if (netvlad_client.call(img_srv)) {
      if (DEBUG) {
        ROS_ERROR("Succeed to call service");
      }
      Eigen::VectorXf desc(img_srv.response.res_des.size());
      for (size_t i = 0; i < img_srv.response.res_des.size(); ++i) {
        desc(i) = img_srv.response.res_des[i];
      }

      MapDatabase[cur_robot_id][cur_thumbnail_id]->descriptor_db.emplace_back(
        id_tn, desc);
    } else {
      ROS_ERROR("Failed to call service");
    }
    end = clock();
    outfile.open(
      std::string(std::getenv("HOME")) +
      "/gacm_output/timecost/descriptor_time.txt",
      std::ios::app);
    outfile << (double)(end - start) / CLOCKS_PER_SEC << "\n";
    outfile.close();

    // add submap base for search
    PointType temp_base;
    temp_base.x =
      (float)(MapDatabase[cur_robot_id][cur_thumbnail_id]->t_world_base.x());
    temp_base.y =
      (float)(MapDatabase[cur_robot_id][cur_thumbnail_id]->t_world_base.y());
    temp_base.z =
      (float)(MapDatabase[cur_robot_id][cur_thumbnail_id]->t_world_base.z());
    temp_base.intensity = (float)(100 * cur_robot_id + cur_thumbnail_id);
    submap_base_map[cur_robot_id]->points.push_back(temp_base);

    if (DEBUG) {
      ROS_ERROR_STREAM(
        "\n\n\n\n\n\n++++++ thumbnail "
          << cur_thumbnail_id << "\n is published \nsubmap num "
          << submap_base_map[cur_robot_id]->points.size()
          << "\n\n\n\n");
    }
    cur_thumbnail_id++;

    // mTProcess.unlock();
  } else {

    // continue;
    // ROS_ERROR_STREAM("process loop ok");
    if (cur_loop_test_id < cur_thumbnail_id) {
      int big_loop_num = 0; // homo + hetero
      // search for submaps to detect
      update_submapbase_cloud();
      pcl::PointCloud<PointType>::Ptr all_submap_bases(
        new pcl::PointCloud<PointType>());

      clock_t start, end;
      start = clock();

      // first deal with homogeneous loop detection
      std::vector<int> knn_idx;
      std::vector<float> knn_dist;
      PointType temp_base;
      temp_base.x = (float)(MapDatabase[cur_robot_id][cur_loop_test_id]
        ->t_world_base.x());
      temp_base.y = (float)(MapDatabase[cur_robot_id][cur_loop_test_id]
        ->t_world_base.y());
      temp_base.z = (float)(MapDatabase[cur_robot_id][cur_loop_test_id]
        ->t_world_base.z());
      temp_base.intensity = (float)(cur_loop_test_id);

      // int k = need_init_coord ? 10 : 5;
      if (DEBUG) {
        ROS_INFO_STREAM("start homo test");
      }
      *all_submap_bases += *submap_base_map[cur_robot_id];
      int k = 5; // for homogeneous
      if (!all_submap_bases->empty()) {
        kdtree->setInputCloud(all_submap_bases);
        kdtree->nearestKSearch(temp_base, k, knn_idx, knn_dist);
      }
      loop_scores.clear();
      test_base_score(all_submap_bases, knn_idx, knn_dist);

      // try find possible loop (NDT+GICP)
      std::sort(loop_scores.begin(), loop_scores.end()); // increase order
      // test best match, once loop found, early stop
      bool find_homo_loop = false;

      for (int i = 0; i < 1 && i < loop_scores.size(); i++) {
        // todo decide loop score threshold
        // 飞机可以放弃inter-loop？感觉不是很可靠
        if (find_homo_loop == false && loop_scores[i].loop_score < 0.9 &&
          loop_scores[i].angle_rel * 57.3 < 60)
        {

          if (loop_scores.size() >= 3) {
            if (loop_scores[i].loop_score /
              loop_scores[loop_scores.size() - 1].loop_score >
              0.9)
            {
              break;
            }
          }
          // accept

          int ri_test = loop_scores[i].robot_id;
          int si_test = loop_scores[i].submap_id;
          if (DEBUG) {
            ROS_ERROR_STREAM(
              "Homo Matching "
                << ri_test << ":" << si_test << " ; "
                << cur_loop_test_id << " with score "
                << loop_scores[i].loop_score << " with angle "
                << loop_scores[i].angle_rel * 57.3);
          }
          // visualization
          cv::Mat query_tn =
            cv::imread(
            MapDatabase[cur_robot_id][cur_loop_test_id]
            ->thumbnails_db[0]
            .second);
          cv::Mat response_tn = cv::imread(
            MapDatabase[ri_test][si_test]->thumbnails_db[0].second);

          pcl::PointCloud<PointType>::Ptr cloud_src =
            MapDatabase[ri_test][si_test]->getMapCloud();
          pcl::PointCloud<PointType>::Ptr cloud_target =
            MapDatabase[cur_robot_id][cur_loop_test_id]->getMapCloud();

          clock_t start_icp;
          clock_t end_icp;
          start_icp = clock();

          auto [final_score, final_guess] =
            test_match_cloud(
            cloud_src, cloud_target, ri_test, si_test,
            loop_scores[i].try_guess);

          pcl::PointCloud<PointType>::Ptr cloud_final_icp(
            new pcl::PointCloud<PointType>);
          pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp2;
          icp2.setInputSource(cloud_src);
          icp2.setInputTarget(cloud_target);
          icp2.setMaximumIterations(50);
          icp2.setMaxCorrespondenceDistance(5);
          icp2.align(*cloud_final_icp, final_guess);

          end_icp = clock();
          std::ofstream outfile1;
          outfile1.open(
            std::string(std::getenv("HOME")) +
            "/gacm_output/timecost/single_match_homo.txt",
            std::ios::app);
          if ((double)(end_icp - start_icp) / CLOCKS_PER_SEC > 0.5) {
            outfile1 << (double)(end_icp - start_icp) / CLOCKS_PER_SEC << "\n";
          }
          outfile1.close();

          if (icp2.hasConverged()) {
            final_score = icp2.getFitnessScore(5);
            if (DEBUG) {
              std::cout << "\nFitness score final" << final_score << std::endl;
            }
            // save loop closure edge
            Eigen::Matrix3d R_rel = icp2.getFinalTransformation()
              .topLeftCorner(3, 3)
              .cast<double>();
            Eigen::Quaterniond q_rel;
            q_rel = R_rel;
            Eigen::Vector3d t_rel = icp2.getFinalTransformation()
              .topRightCorner(3, 1)
              .cast<double>();

            MeasurementEdge measurement_edge(
              MapDatabase[cur_robot_id][cur_loop_test_id]
              ->pose_graph_local[0]
              .stamp,
              MapDatabase[ri_test][si_test]->pose_graph_local[0].stamp,
              cur_robot_id, ri_test, cur_loop_test_id, si_test, 0,
              0,   // base node to base node
              q_rel, t_rel);

            // 这里无人机可以严格一点，因为好像inter-loop不是很可靠
            if (final_score < (is_drone[cur_robot_id] ? 0.3 : 0.5)) {
              {
                std::lock_guard<std::mutex> lk_loop{m_loop};
                loopEdgeBuf.push_back(measurement_edge);
              }

              find_homo_loop = true;
              big_loop_num++;
              if (cur_robot_id != ri_test) {
                need_init_coord = false;
              }
              std::cout << "\033[1;32m\nDone add edge homo! robot[" << ri_test
                        << ":" << si_test << "] => robot[" << cur_robot_id
                        << ":" << cur_loop_test_id << "]\n\033[0m\n";

              pub_loop_image(query_tn, response_tn, true);

            } else { // reject by icp score
              dropEdgeBuf.push_back(measurement_edge);
              if (DEBUG) {
                ROS_WARN_STREAM("drop edge score");
              }
              pub_loop_image(query_tn, response_tn, false);
            }
          } else { // reject by icp convergence
            if (DEBUG) {
              ROS_WARN_STREAM("Not converge");
            }
            MeasurementEdge measurement_edge(
              MapDatabase[cur_robot_id][cur_loop_test_id]
              ->pose_graph_local[0]
              .stamp,
              MapDatabase[ri_test][si_test]->pose_graph_local[0].stamp,
              cur_robot_id, ri_test, cur_loop_test_id, si_test, 0,
              0,   // base node to base node
              Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
            dropEdgeBuf.push_back(measurement_edge);
            if (DEBUG) {
              ROS_WARN_STREAM(
                "drop edge converge("
                  << loop_scores[i].angle_rel * 57.3 << ")");
            }
            pub_loop_image(query_tn, response_tn, false);
          }

        } else { // reject by geometry
          if (loop_scores[i].loop_score < 0.9) {

            int ri_test = loop_scores[i].robot_id;
            int si_test = loop_scores[i].submap_id;
            MeasurementEdge measurement_edge(
              MapDatabase[cur_robot_id][cur_loop_test_id]
              ->pose_graph_local[0]
              .stamp,
              MapDatabase[ri_test][si_test]->pose_graph_local[0].stamp,
              cur_robot_id, ri_test, cur_loop_test_id, si_test, 0,
              0,   // base node to base node
              Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
            dropEdgeBuf.push_back(measurement_edge);
            if (DEBUG) {
              ROS_WARN_STREAM("drop edge orientation");
            }
            cv::Mat query_tn =
              cv::imread(
              MapDatabase[cur_robot_id][cur_loop_test_id]
              ->thumbnails_db[0]
              .second);
            cv::Mat response_tn = cv::imread(
              MapDatabase[ri_test][si_test]->thumbnails_db[0].second);
            pub_loop_image(query_tn, response_tn, false);
          }
        } // end if candidate accept
      }
      // end homo loopscores

      if (DEBUG) {
        ROS_INFO_STREAM("start hetro test");
      }
      // for heterogeneous
      all_submap_bases->clear();
      for (int r = 0; r < cur_robot_id; r++) {
        // if (is_drone[r] != is_drone[cur_robot_id]) {
        *all_submap_bases += *submap_base_map[r];
        // }
      }
      knn_idx.clear();
      knn_dist.clear();

      k = 20;
      if (need_init_coord) {
        k = 200;
      }
      if (!all_submap_bases->empty()) {
        kdtree->setInputCloud(all_submap_bases);
        kdtree->nearestKSearch(temp_base, k, knn_idx, knn_dist);
      }
      loop_scores.clear();
      test_base_score(all_submap_bases, knn_idx, knn_dist);

      // try find possible loop (NDT+GICP)
      std::sort(loop_scores.begin(), loop_scores.end()); // increase order
      // test best match, once loop found, early stop
      bool find_hetero_loop = false;
      for (int i = 0; i < 1 && i < loop_scores.size(); i++) {
        // todo decide loop score threshold
        // 飞机可以在角度上放宽要求
        if (find_hetero_loop == false && loop_scores[i].loop_score < 0.9 &&
          (need_init_coord || is_drone[cur_robot_id] ||
          loop_scores[i].angle_rel * 57.3 < 60))
        {
          // accept

          if (loop_scores.size() >= 3) {
            if (loop_scores[i].loop_score /
              loop_scores[loop_scores.size() - 1].loop_score >
              0.9)
            {
              break;
            }
          }

          int ri_test = loop_scores[i].robot_id;
          int si_test = loop_scores[i].submap_id;
          if (DEBUG) {
            ROS_ERROR_STREAM(
              "Hetero Matching "
                << ri_test << " : " << si_test << ";"
                << cur_loop_test_id << " with score "
                << loop_scores[i].loop_score << " with angle "
                << loop_scores[i].angle_rel * 57.3);
          }
          // visualization
          cv::Mat query_tn =
            cv::imread(
            MapDatabase[cur_robot_id][cur_loop_test_id]
            ->thumbnails_db[0]
            .second);
          cv::Mat response_tn = cv::imread(
            MapDatabase[ri_test][si_test]->thumbnails_db[0].second);

          // debug icp
          pcl::PointCloud<PointType>::Ptr cloud_src =
            MapDatabase[ri_test][si_test]->getMapCloud();
          pcl::PointCloud<PointType>::Ptr cloud_target =
            MapDatabase[cur_robot_id][cur_loop_test_id]->getMapCloud();

          clock_t start_icp, end_icp;
          start_icp = clock();

          auto [final_score, final_guess] =
            test_match_cloud(
            cloud_src, cloud_target, ri_test, si_test,
            loop_scores[i].try_guess);

          pcl::PointCloud<PointType>::Ptr cloud_final_icp(
            new pcl::PointCloud<PointType>);
          pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp3;
          if (need_init_coord == false || !is_drone[cur_robot_id]) {
            icp3.setMaximumIterations(60);
            icp3.setMaxCorrespondenceDistance(
              3);   // hetero test shouldn't set distance threshold
            icp3.setRANSACIterations(10);
          }
          icp3.setInputSource(cloud_src);
          icp3.setInputTarget(cloud_target);
          icp3.align(*cloud_final_icp, final_guess);

          end_icp = clock();
          std::ofstream outfile1;
          outfile1.open(
            std::string(std::getenv("HOME")) +
            "/gacm_output/timecost/single_match_hetero.txt",
            std::ios::app);
          if ((double)(end_icp - start_icp) / CLOCKS_PER_SEC > 0.5) {
            outfile1 << (double)(end_icp - start_icp) / CLOCKS_PER_SEC << "\n";
          }
          outfile1.close();

          if (icp3.hasConverged()) {
            final_score = icp3.getFitnessScore(5);
            if (DEBUG) {
              ROS_INFO_STREAM(
                "robot[" << ri_test << ":" << si_test << "] => robot["
                         << cur_robot_id << ":" << cur_loop_test_id
                         << "] Fitness score final: " << final_score << "\n");
            }
            // save loop closure edge
            Eigen::Matrix3d R_rel = icp3.getFinalTransformation()
              .topLeftCorner(3, 3)
              .cast<double>();
            Eigen::Quaterniond q_rel;
            q_rel = R_rel;
            Eigen::Vector3d t_rel = icp3.getFinalTransformation()
              .topRightCorner(3, 1)
              .cast<double>();

            MeasurementEdge measurement_edge(
              MapDatabase[cur_robot_id][cur_loop_test_id]
              ->pose_graph_local[0]
              .stamp,
              MapDatabase[ri_test][si_test]->pose_graph_local[0].stamp,
              cur_robot_id, ri_test, cur_loop_test_id, si_test, 0,
              0,   // base node to base node
              q_rel, t_rel);

            // 无人机初始化的时候可以放松要求
            // 除了初始化，如果构成异构回环的都是无人机，那么还需要更严格的要求
            if (final_score <
              (is_drone[cur_robot_id] && need_init_coord ?
              (is_drone[ri_test] ? 0.8 : 0.9) :
              (is_drone[cur_robot_id] && is_drone[ri_test] ? 0.65 :
              0.75)))
            {

              {
                std::lock_guard<std::mutex> lk_loop{m_loop};
                // int accept;
                // std::cin >> accept;
                // if (accept == 1) {
                // }
                loopEdgeBuf.push_back(measurement_edge);
              }
              // performOptimization();
              find_hetero_loop = true;
              big_loop_num++;
              std::cout << "\033[1;32m\nDone add edge hetero! robot[" << ri_test
                        << ":" << si_test << "] => robot[" << cur_robot_id
                        << ":" << cur_loop_test_id << "]\n\033[0m\n";
              need_init_coord = false;

              std::string file_pefix = std::getenv("HOME");
              file_pefix +=
                "/gacm_output/cache/r" + std::to_string(cur_robot_id) + "s" +
                std::to_string(cur_loop_test_id) + "-r" +
                std::to_string(ri_test) + "s" + std::to_string(si_test) + "-";
              pcl::io::savePCDFileASCII(
                file_pefix + "target.pcd",
                *cloud_target);
              pcl::io::savePCDFileASCII(file_pefix + "source.pcd", *cloud_src);
              pcl::io::savePCDFileASCII(
                file_pefix + "final.pcd",
                *cloud_final_icp);

              pub_loop_image(query_tn, response_tn, true);

            } else { // reject by icp score
              dropEdgeBuf.push_back(measurement_edge);
              if (DEBUG) {
                ROS_WARN_STREAM("(hetero loop)drop edge score");
              }
              pub_loop_image(query_tn, response_tn, false);
            }
          } else { // reject by icp convergence

            // int ri_test = loop_scores[i].robot_id;
            // int si_test = loop_scores[i].submap_id;
            MeasurementEdge measurement_edge(
              MapDatabase[cur_robot_id][cur_loop_test_id]
              ->pose_graph_local[0]
              .stamp,
              MapDatabase[ri_test][si_test]->pose_graph_local[0].stamp,
              cur_robot_id, ri_test, cur_loop_test_id, si_test, 0,
              0,   // base node to base node
              Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
            if (DEBUG) {
              ROS_WARN_STREAM("(hetero loop)Not converge");
            }
            pub_loop_image(query_tn, response_tn, false);
          }

        } else { // reject by geometry
          if (loop_scores[i].loop_score < 1.0) {
            int ri_test = loop_scores[i].robot_id;
            int si_test = loop_scores[i].submap_id;
            MeasurementEdge measurement_edge(
              MapDatabase[cur_robot_id][cur_loop_test_id]
              ->pose_graph_local[0]
              .stamp,
              MapDatabase[ri_test][si_test]->pose_graph_local[0].stamp,
              cur_robot_id, ri_test, cur_loop_test_id, si_test, 0,
              0,   // base node to base node
              Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
            dropEdgeBuf.push_back(measurement_edge);
            if (DEBUG) {
              ROS_WARN_STREAM(
                "(hetero loop)drop edge orientation("
                  << loop_scores[i].angle_rel * 57.3 << ")");
            }
            cv::Mat query_tn =
              cv::imread(
              MapDatabase[cur_robot_id][cur_loop_test_id]
              ->thumbnails_db[0]
              .second);
            cv::Mat response_tn = cv::imread(
              MapDatabase[ri_test][si_test]->thumbnails_db[0].second);
            pub_loop_image(query_tn, response_tn, false);
          }
        } // end if candidate accept
      }
      // end hetero loopscores

      end = clock();
      std::ofstream outfile;
      outfile.open(
        std::string(std::getenv("HOME")) +
        "/gacm_output/timecost/loop_time.txt",
        std::ios::app);
      if ((double)(end - start) / CLOCKS_PER_SEC > 0.5) {
        outfile << (double)(end - start) / CLOCKS_PER_SEC << "\n";
      }
      outfile.close();

      if (DEBUG) {
        ROS_INFO_STREAM("start adjacent test");
      }
      // for adjacent submap
      if (RUN_ODOMETRY && cur_loop_test_id > 1) {
        clock_t start_icp, end_icp;
        start_icp = clock();

        pcl::PointCloud<PointType>::Ptr cloud_final_icp3(
          new pcl::PointCloud<PointType>);
        pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp3;
        pcl::PointCloud<PointType>::Ptr cloud_src =
          MapDatabase[cur_robot_id][cur_loop_test_id - 1]->getMapCloud();
        pcl::PointCloud<PointType>::Ptr cloud_target =
          MapDatabase[cur_robot_id][cur_loop_test_id]->getMapCloud();
        if (DEBUG) {
          ROS_INFO_STREAM(
            "done fetch " << cloud_src->size() << " : "
                          << cloud_target->size());
        }
        pcl::ApproximateVoxelGrid<PointType> avg;
        avg.setLeafSize(0.8f, 0.8f, 0.8f); // trade off
        // avg.setLeafSize(0.5f, 0.5f, 0.5f); // trade off
        avg.setInputCloud(cloud_src);
        avg.filter(*cloud_src);
        avg.setInputCloud(cloud_target);
        avg.filter(*cloud_target);
        icp3.setInputSource(cloud_src);
        icp3.setInputTarget(cloud_target);
        icp3.setMaxCorrespondenceDistance(5);
        int ri_test = cur_robot_id;
        int si_test = cur_loop_test_id - 1;
        Eigen::Quaterniond q_rel = MapDatabase[cur_robot_id][cur_loop_test_id]
          ->q_world_base.inverse() *
          MapDatabase[ri_test][si_test]->q_world_base;
        Eigen::Vector3d t_rel =
          MapDatabase[cur_robot_id][cur_loop_test_id]
          ->q_world_base.inverse() *
          (MapDatabase[ri_test][si_test]->t_world_base -
          MapDatabase[cur_robot_id][cur_loop_test_id]->t_world_base);
        Eigen::Matrix4f try_guess = Eigen::Matrix4f::Identity();
        try_guess.block(0, 0, 3, 3) =
          q_rel.normalized().toRotationMatrix().cast<float>();
        try_guess.block(0, 3, 3, 1) = t_rel.cast<float>();
        if (DEBUG) {
          ROS_INFO_STREAM("done guess");
        }
        icp3.align(*cloud_final_icp3, try_guess);
        if (DEBUG) {
          ROS_INFO_STREAM("done icp");
        }
        end_icp = clock();
        std::ofstream outfile1;
        outfile1.open(
          std::string(std::getenv("HOME")) +
          "/gacm_output/timecost/single_match_adj.txt",
          std::ios::app);
        outfile1 << (double)(end_icp - start_icp) / CLOCKS_PER_SEC << "\n";
        outfile1.close();

        if (icp3.hasConverged()) {
          float final_score = icp3.getFitnessScore(5);
          if (DEBUG) {
            std::cout << "\nFitness score final adj" << final_score
                      << std::endl;
          }
          // save loop closure edge
          Eigen::Matrix3d R_rel =
            icp3.getFinalTransformation().topLeftCorner(3, 3).cast<double>();
          Eigen::Quaterniond q_rel_;
          q_rel_ = R_rel;
          Eigen::Vector3d t_rel_ =
            icp3.getFinalTransformation().topRightCorner(3, 1).cast<double>();

          MeasurementEdge measurement_edge(
            MapDatabase[cur_robot_id][cur_loop_test_id]
            ->pose_graph_local[0]
            .stamp,
            MapDatabase[ri_test][si_test]->pose_graph_local[0].stamp,
            cur_robot_id, ri_test, cur_loop_test_id, si_test, 0,
            0,   // base node to base node
            q_rel_, t_rel_);

          if (final_score < 0.9) {
            std::lock_guard<std::mutex> lk_loop{m_loop};
            loopEdgeBuf.push_back(measurement_edge);
            // performOptimization();
            if (DEBUG) {
              ROS_WARN_STREAM("done add edge adjacent");
            }
          } else { // reject by icp score
            if (DEBUG) {
              ROS_WARN_STREAM("drop edge adj");
            }
          }
        } else { // reject by icp convergence
          if (DEBUG) {
            ROS_WARN_STREAM("Not converge");
          }
        }
      }
      // end adjacent test

      std::thread t(&SubMapManagementObject::publish_opt_posegraph, this,
        false);
      t.detach();
      publish_opt_posegraph(false);
      if (big_loop_num > 0 || loopEdgeBuf.size() >= 3) {
        std::lock_guard<std::mutex> lk_loop{m_loop};
        optimize(); // check need optimize inside function
      }
      cur_loop_test_id++;
      if (DEBUG) {
        ROS_ERROR_STREAM("done thumnail loop");
      }
    }
  }
}

void SubMapManagementObject::update_submapbase_cloud()
{
  if (DEBUG) {
    std::cout << "\nStart update submapbase\n";
  }

  for (int ri = 0; ri <= cur_robot_id; ri++) {
    int temp_submap_size = MapDatabase[ri].size();
    if (ri == cur_robot_id && temp_submap_size > cur_loop_test_id) {
      temp_submap_size = cur_loop_test_id;
    }
    submap_base_map[ri].reset(new pcl::PointCloud<PointType>());
    // submap_base_map[ri]->points.clear();
    for (int si = 0; si < temp_submap_size; si++) {
      // don't use future submap or lastnode for loop detection
      if (MapDatabase[ri][si]->thumbnailGenerated == false) {
        continue;
      }
      PointType temp_base;
      temp_base.x = (float)(MapDatabase[ri][si]->t_world_base.x());
      temp_base.y = (float)(MapDatabase[ri][si]->t_world_base.y());
      temp_base.z = (float)(MapDatabase[ri][si]->t_world_base.z());
      temp_base.intensity = (float)(100 * ri + si);
      submap_base_map[ri]->points.push_back(temp_base);
      // std::cout << "add " << si <<"\n";
    }
    // std::cout << "\ndone " << ri << "\n";
  }
  if (DEBUG) {
    std::cout << "\nDone update submapbase\n";
  }
}

void SubMapManagementObject::publish_opt_posegraph(bool pubmap)
{

  if (updating_pg == true) {
    return;
  }
  updating_pg = true;

  if (DEBUG) {
    std::cout << "Publish posegraph start \n";
  }
  int edge_acc = 0;
  int edge_adj = 0;
  int edge_rej = 0;
  clock_t start, end;
  start = clock();
  // pose graph and loop edge visualization , see ga_posegrph folder
  for (int r = 0; r <= cur_robot_id; r++) {
    nav_msgs::Path optPath; // path after odom
    optPath.header.frame_id = "camera";
    optPath.header.stamp = ros::Time::now();
    int database_size = MapDatabase[r].size();
    // if(DEBUG) ROS_WARN_STREAM("robot[" << r << "] has " << database_size << "
    // submaps");
    for (int i = 0; i < database_size - 1; i++) { // the last submap is isolated
      int max_size = MapDatabase[r][i]->pose_graph_local.size();
      // std::cout << "Max size " << max_size << "\n";
      for (int j = 0; j < max_size; j++) {
        // nav_msgs::Odometry laserOdometry;
        geometry_msgs::PoseStamped pose_opt;
        pose_opt.header.frame_id = "camera";
        pose_opt.header.stamp =
          ros::Time().fromSec(MapDatabase[r][i]->pose_graph_local[j].stamp);
        Eigen::Quaterniond q = MapDatabase[r][i]->pose_graph_local[j].q;
        Eigen::Vector3d t = MapDatabase[r][i]->pose_graph_local[j].t;

        t = MapDatabase[r][i]->t_world_base +
          MapDatabase[r][i]->q_world_base * t;
        q = MapDatabase[r][i]->q_world_base * q;

        if (!display_frame_cam) {
          Eigen::Affine3d T_Cworld_robot =
            Eigen::Translation3d(t) * q.toRotationMatrix();
          Eigen::Affine3d T_Lworld_robot =
            Eigen::Affine3d(T_LC) * T_Cworld_robot;
          q = T_Lworld_robot.rotation();
          t = T_Lworld_robot.translation();
        }

        pose_opt.pose.orientation.w = q.w();
        pose_opt.pose.orientation.x = q.x();
        pose_opt.pose.orientation.y = q.y();
        pose_opt.pose.orientation.z = q.z();
        pose_opt.pose.position.x = t.x();
        pose_opt.pose.position.y = t.y();
        pose_opt.pose.position.z = t.z();
        optPath.poses.push_back(pose_opt);
      }

      // publish posegraph edges

      visualization_msgs::Marker edge_line_msg;
      edge_line_msg.header.frame_id = "camera";
      edge_line_msg.header.stamp = ros::Time::now();
      edge_line_msg.id = 0;
      edge_line_msg.type = visualization_msgs::Marker::LINE_LIST;
      edge_line_msg.scale.x = 0.3;
      edge_line_msg.color.r = 1.0;
      edge_line_msg.color.g = 1.0;
      edge_line_msg.color.b = 0.0;
      edge_line_msg.color.a = 1.0;
      visualization_msgs::Marker edge_node_msg;
      edge_node_msg.header.frame_id = "camera";
      edge_node_msg.header.stamp = ros::Time::now();
      edge_node_msg.id = 1;
      edge_node_msg.type = visualization_msgs::Marker::SPHERE_LIST;
      edge_node_msg.scale.x = 3;
      edge_node_msg.color.r = 0.0;
      edge_node_msg.color.g = 1.0;
      edge_node_msg.color.b = 0.0;
      edge_node_msg.color.a = 1.0;
      visualization_msgs::Marker edge_line_msg1;
      edge_line_msg1.header.frame_id = "camera";
      edge_line_msg1.header.stamp = ros::Time::now();
      edge_line_msg1.id = 2;
      edge_line_msg1.type = visualization_msgs::Marker::LINE_LIST;
      edge_line_msg1.scale.x = 0.15;
      edge_line_msg1.color.r = 1.0;
      edge_line_msg1.color.g = 0.0;
      edge_line_msg1.color.b = 0.0;
      edge_line_msg1.color.a = 1.0;
      visualization_msgs::Marker edge_node_msg1;
      edge_node_msg1.header.frame_id = "camera";
      edge_node_msg1.header.stamp = ros::Time::now();
      edge_node_msg1.id = 3;
      edge_node_msg1.type = visualization_msgs::Marker::SPHERE_LIST;
      edge_node_msg1.scale.x = 1.5;
      edge_node_msg1.color.r = 1.0;
      edge_node_msg1.color.g = 0.2;
      edge_node_msg1.color.b = 0.0;
      edge_node_msg1.color.a = 1.0;
      for (int index = 0; index < loopEdgeBuf.size(); index++) {
        int ri_from = loopEdgeBuf[index].robot_from;
        int ri_to = loopEdgeBuf[index].robot_to;
        int si_from = loopEdgeBuf[index].submap_from;
        int si_to = loopEdgeBuf[index].submap_to;
        if (ri_from != ri_to || si_from != si_to + 1) {
          Eigen::Vector3d t_from = MapDatabase[ri_from][si_from]->t_world_base;
          Eigen::Vector3d t_to = MapDatabase[ri_to][si_to]->t_world_base;
          edge_acc++;

          if (!display_frame_cam) {
            t_from =
              (T_LC * Eigen::Vector4d(t_from[0], t_from[1], t_from[2], 1))
              .head(3);
            t_to =
              (T_LC * Eigen::Vector4d(t_to[0], t_to[1], t_to[2], 1)).head(3);
          }

          geometry_msgs::Point p;
          p.x = t_from.x();
          p.y = t_from.y();
          p.z = t_from.z();
          edge_line_msg.points.push_back(p);
          edge_node_msg.points.push_back(p);
          p.x = t_to.x();
          p.y = t_to.y();
          p.z = t_to.z();
          edge_line_msg.points.push_back(p);
          edge_node_msg.points.push_back(p);
        } else {
          edge_adj++;
        }
      }

      for (int index = 0; index < dropEdgeBuf.size(); index++) {
        int ri_from = dropEdgeBuf[index].robot_from;
        int ri_to = dropEdgeBuf[index].robot_to;
        int si_from = dropEdgeBuf[index].submap_from;
        int si_to = dropEdgeBuf[index].submap_to;

        if (ri_from != ri_to || si_from != si_to + 1) {
          Eigen::Vector3d t_from = MapDatabase[ri_from][si_from]->t_world_base;
          Eigen::Vector3d t_to = MapDatabase[ri_to][si_to]->t_world_base;
          edge_rej++;

          if (!display_frame_cam) {
            t_from =
              (T_LC * Eigen::Vector4d(t_from[0], t_from[1], t_from[2], 1))
              .head(3);
            t_to =
              (T_LC * Eigen::Vector4d(t_to[0], t_to[1], t_to[2], 1)).head(3);
          }

          geometry_msgs::Point p;
          p.x = t_from.x();
          p.y = t_from.y();
          p.z = t_from.z();
          edge_line_msg1.points.push_back(p);
          edge_node_msg1.points.push_back(p);
          p.x = t_to.x();
          p.y = t_to.y();
          p.z = t_to.z();
          edge_line_msg1.points.push_back(p);
          edge_node_msg1.points.push_back(p);
        }
      }

      pubRelEdge.publish(edge_line_msg);
      pubRelEdge.publish(edge_node_msg);
      // pubRelEdge.publish(edge_line_msg1);
      pubRelEdge.publish(edge_node_msg1);

      MapDatabase[r][i]->publishMap(0); // don't pub map, update TF only
    }
    pubPathOpt[r].publish(optPath);
  }
  end = clock();
  if (DEBUG) {
    std::cout << "Publish posegraph use "
              << (double)(end - start) / CLOCKS_PER_SEC << "\n";
  }
  updating_pg = false;
}

void SubMapManagementObject::publish_opt_map(bool pubmap, bool pubmerge)
{
  if (updating_map == true) {
    return;
  }
  updating_map = true;

  if (DEBUG) {
    std::cout << "\nstart visualization\n ";
  }
  pcl::PointCloud<PointType>::Ptr mergeCloudMapG(
    new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr mergeCloudMapA(
    new pcl::PointCloud<PointType>());
  clock_t start, end;
  start = clock();
  // pose graph and loop edge visualization , see ga_posegrph folder
  for (int r = 0; r <= cur_robot_id; r++) {
    pcl::PointCloud<PointType>::Ptr laserCloudMap(
      new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr tempCloud(new pcl::PointCloud<PointType>());

    int database_size = MapDatabase[r].size();
    for (int i = 0; i < database_size - 1; i++) { // the last submap is isolated
      tempCloud->clear();
      *tempCloud += *(MapDatabase[r][i]->getMapCloud(true));

      if (tempCloud->size() > 0) {

        // 这里不要用去除外点，要不然实在太慢了
        if (0) {
          pcl::ApproximateVoxelGrid<PointType> approximate_voxel_filter;
          approximate_voxel_filter.setLeafSize(
            0.05, 0.05,
            0.05);                                    // downsample map speed up
          approximate_voxel_filter.setInputCloud(tempCloud);
          approximate_voxel_filter.filter(*tempCloud);

          pcl::StatisticalOutlierRemoval<PointType> sor; // remove outlier
          sor.setMeanK(5);
          sor.setStddevMulThresh(1.0);
          sor.setInputCloud(tempCloud);
          sor.filter(*tempCloud);
        }

        *laserCloudMap += *tempCloud;
        tempCloud->clear();
      }
      if (pubmap == true) {
        std::string ss = "0";
        ss[0] += r;
        if (DEBUG) {
          std::cout << "start exporting " << laserCloudMap->size() << std::endl;
        }
      }
      // ROS_WARN_STREAM("pub opt posegraph");
    }

    if (pubmerge && laserCloudMap->points.size() > 100) {

      pcl::ApproximateVoxelGrid<PointType> approximate_voxel_filter;
      approximate_voxel_filter.setLeafSize(
        0.1, 0.1,
        0.1);                                    // downsample map speed up
      approximate_voxel_filter.setInputCloud(laserCloudMap);
      approximate_voxel_filter.filter(*laserCloudMap);
      // std::cout << "Down sampled\n";
      pcl::StatisticalOutlierRemoval<PointType> sor; // remove outlier
      sor.setMeanK(10);
      sor.setStddevMulThresh(1.0);
      sor.setInputCloud(laserCloudMap);
      sor.filter(*laserCloudMap);
      std::string ss = "0";
      ss[0] += r;
      pcl::io::savePCDFileASCII(
        std::string(std::getenv("HOME")) +
        "/gacm_output/data/testSavemap/fullcloud/" +
        ss + "full.pcd",
        *laserCloudMap);
      if (r < 3) {
        *mergeCloudMapG += *laserCloudMap;
        pcl::io::savePCDFileASCII(
          std::string(std::getenv("HOME")) +
          "/gacm_output/data/testSavemap/fullcloud/mergeG.pcd",
          *mergeCloudMapG);
      } else {
        *mergeCloudMapA += *laserCloudMap;
        pcl::io::savePCDFileASCII(
          std::string(std::getenv("HOME")) +
          "/gacm_output/data/testSavemap/fullcloud/mergeA.pcd",
          *mergeCloudMapA);
      }
      std::cout << "\nSave merge to pcd\n";
    }
    // std::cout << "Sor sampled\n";
    sensor_msgs::PointCloud2 laserCloudMsg;

    if (!display_frame_cam) {
      pcl::transformPointCloud(*laserCloudMap, *laserCloudMap, T_LC);
    }

    pcl::toROSMsg(*laserCloudMap, laserCloudMsg);
    laserCloudMsg.header.stamp =
      ros::Time().fromNSec(static_cast<uint64_t>(timestamp_nanoseconds));
    laserCloudMsg.header.frame_id = "camera";
    pubOptimizedMap[r].publish(laserCloudMsg);
    laserCloudMap->clear();
    if (DEBUG) {
      ROS_INFO_STREAM("pub map [" << r << "]");
    }
  }
  end = clock();
  if (DEBUG) {
    std::cout << "Publish optimized map use "
              << (double)(end - start) / CLOCKS_PER_SEC << "\n";
  }
  updating_map = false;
}

void SubMapManagementObject::optimize()
{
  int cur_loop_size = loopEdgeBuf.size();
  if (cur_loop_size - last_loop_size == 0) {
    return;
  }

  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);
  ceres::LocalParameterization * q_parameterization =
    new ceres::EigenQuaternionParameterization();
  Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
  // add_loop_num = 0;
  for (int ri = 0; ri <= cur_robot_id; ri++) {
    int current_database_size = MapDatabase[ri].size();
    for (int i = 0; i < current_database_size; i++) {
      int current_node_size, current_edge_size;

      // parameter block for base node
      problem.AddParameterBlock(
        MapDatabase[ri][i]->q_world_base.coeffs().data(), 4,
        q_parameterization);
      problem.AddParameterBlock(MapDatabase[ri][i]->t_world_base.data(), 3);
      if (ri == 0 && i == 0) {
        // set fix the first base
        problem.SetParameterBlockConstant(
          MapDatabase[ri][i]->q_world_base.coeffs().data());
        problem.SetParameterBlockConstant(
          MapDatabase[ri][i]->t_world_base.data());
      }
      // parameter block for local node
      // fetch node and add parameter block
      current_node_size = MapDatabase[ri][i]->pose_graph_local.size();
      for (int node_id = 0; node_id < current_node_size; node_id++) {
        problem.AddParameterBlock(
          MapDatabase[ri][i]->pose_graph_local[node_id].q.coeffs().data(), 4,
          q_parameterization);
        problem.AddParameterBlock(
          MapDatabase[ri][i]->pose_graph_local[node_id].t.data(), 3);
        if (node_id == 0) {
          // set fix every start node
          problem.SetParameterBlockConstant(
            MapDatabase[ri][i]->pose_graph_local[node_id].q.coeffs().data());
          problem.SetParameterBlockConstant(
            MapDatabase[ri][i]->pose_graph_local[node_id].t.data());
        }
      }

      // add residual block for consistancy, the last node of the former submap
      // shuoud be the same as the first node of current submap
      if (i > 0) {
        ceres::CostFunction * cost_function = PoseGraph3dErrorTermWorld::Create(
          Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero(), cov);
        problem.AddResidualBlock(
          cost_function, NULL,
          MapDatabase[ri][i - 1]->pose_graph_local.back().q.coeffs().data(),
          MapDatabase[ri][i - 1]->pose_graph_local.back().t.data(),
          MapDatabase[ri][i - 1]->q_world_base.coeffs().data(),
          MapDatabase[ri][i - 1]->t_world_base.data(),
          MapDatabase[ri][i]->pose_graph_local.front().q.coeffs().data(),
          MapDatabase[ri][i]->pose_graph_local.front().t.data(),
          MapDatabase[ri][i]->q_world_base.coeffs().data(),
          MapDatabase[ri][i]->t_world_base.data());
      }

      // fetch edge and add residual block
      current_edge_size = MapDatabase[ri][i]->edges_local.size();
      for (int edge_id = 0; edge_id < current_edge_size; edge_id++) {
        ceres::CostFunction * cost_function = PoseGraph3dErrorTerm::Create(
          MapDatabase[ri][i]->edges_local[edge_id].q,
          MapDatabase[ri][i]->edges_local[edge_id].t, cov);
        problem.AddResidualBlock(
          cost_function, NULL,
          MapDatabase[ri][i]
          ->pose_graph_local
          [MapDatabase[ri][i]->edges_local[edge_id].index_from]
          .q.coeffs()
          .data(),
          MapDatabase[ri][i]
          ->pose_graph_local
          [MapDatabase[ri][i]->edges_local[edge_id].index_from]
          .t.data(),
          MapDatabase[ri][i]
          ->pose_graph_local
          [MapDatabase[ri][i]->edges_local[edge_id].index_to]
          .q.coeffs()
          .data(),
          MapDatabase[ri][i]
          ->pose_graph_local
          [MapDatabase[ri][i]->edges_local[edge_id].index_to]
          .t.data());
      }
    }
  }

  for (int i = 0; i < cur_loop_size; i++) {
    int r_id_f = loopEdgeBuf[i].robot_from;
    int r_id_t = loopEdgeBuf[i].robot_to;
    int s_id_f = loopEdgeBuf[i].submap_from;
    int s_id_t = loopEdgeBuf[i].submap_to;
    ceres::CostFunction * cost_function = PoseGraph3dErrorTermWorld::Create(
      loopEdgeBuf[i].q, loopEdgeBuf[i].t, cov);
    problem.AddResidualBlock(
      cost_function, NULL,
      MapDatabase[r_id_f][s_id_f]->pose_graph_local[0].q.coeffs().data(),
      MapDatabase[r_id_f][s_id_f]->pose_graph_local[0].t.data(),
      MapDatabase[r_id_f][s_id_f]->q_world_base.coeffs().data(),
      MapDatabase[r_id_f][s_id_f]->t_world_base.data(),
      MapDatabase[r_id_t][s_id_t]->pose_graph_local[0].q.coeffs().data(),
      MapDatabase[r_id_t][s_id_t]->pose_graph_local[0].t.data(),
      MapDatabase[r_id_t][s_id_t]->q_world_base.coeffs().data(),
      MapDatabase[r_id_t][s_id_t]->t_world_base.data());
  }

  if (cur_loop_size >= 1) {
    clock_t start, end;
    start = clock();
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.dynamic_sparsity = true;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = false;

    if (DEBUG) {
      ROS_WARN_STREAM("Start Optimizing");
    }
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (DEBUG) {
      std::cout << summary.BriefReport() << std::endl;
      ROS_WARN_STREAM("Pose graph solved");
    }
    last_loop_size = cur_loop_size;
    end = clock();

    std::ofstream outfile;
    outfile.open(
      std::string(std::getenv("HOME")) +
      "/gacm_output/timecost/pgo_time.txt",
      std::ios::app);
    outfile << (double)(end - start) / CLOCKS_PER_SEC << "\n";
    outfile.close();
  }
}

void SubMapManagementObject::optimize_globally()
{
  if (cur_robot_id < 1) {
    return;
  }

  int cmd;

  std::cout << "\033[1;36mNeed Optimization? (\"0\" for no, other key for "
    "yes)\n\033[0m";
  std::cin >> cmd;
  if (cmd == 0) {
    return;
  }

  // 先统计一共要处理多少数据
  int data_length = 0, now_length = 0;
  for (int i = 0; i <= cur_robot_id; i++) {
    data_length += (MapDatabase[i].size() - 1) * 2;
  }

  std::cout << "\033[1;32mBegin global optimization!\033[0m\n";
  int cnt = 0;
  for (int ri = 1; ri <= cur_robot_id; ri++) {
    for (int si = 0; si < MapDatabase[ri].size() - 1; si++) {
      for (int is_homo = 0; is_homo <= 1; is_homo++) {
        // if (ri_t == ri) {
        //      continue;
        // }
        // 显示处理进度
        now_length++;
        if (!DEBUG) {
          status(40, (1.0f * now_length) / data_length);
        }

        pcl::PointCloud<PointType>::Ptr all_submap_bases(
          new pcl::PointCloud<PointType>());
        all_submap_bases->clear();
        // 拿到所有机器人子图信息
        for (int r = 0; r <= cur_robot_id; r++) {
          if (ri != r && (is_homo == 1 ? is_drone[ri] == is_drone[r] :
            is_drone[ri] != is_drone[r]))
          {

            *all_submap_bases += *submap_base_map[r];
          }
        }
        if (all_submap_bases->empty()) {
          continue;
        }
        // search neighbour
        std::vector<int> knn_idx;
        std::vector<float> knn_dist;
        PointType temp_base;
        temp_base.x = (float)(MapDatabase[ri][si]->t_world_base.x());
        temp_base.y = (float)(MapDatabase[ri][si]->t_world_base.y());
        temp_base.z = (float)(MapDatabase[ri][si]->t_world_base.z());
        temp_base.intensity = (float)(si);
        knn_idx.clear();
        knn_dist.clear();
        int k = 1;
        kdtree->setInputCloud(all_submap_bases);
        kdtree->nearestKSearch(temp_base, k, knn_idx, knn_dist);

        for (int knnid = 0; knnid < knn_idx.size(); knnid++) {
          int ri_test = 0;
          int si_test = all_submap_bases->points[knn_idx[knnid]].intensity;
          ri_test = si_test / 100; // robot id
          si_test = si_test % 100; // submap_id

          pcl::PointCloud<PointType>::Ptr cloud_final_icp3(
            new pcl::PointCloud<PointType>);
          pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp3;
          pcl::PointCloud<PointType>::Ptr cloud_src =
            MapDatabase[ri_test][si_test]->getMapCloud();
          pcl::PointCloud<PointType>::Ptr cloud_target =
            MapDatabase[ri][si]->getMapCloud();
          if (DEBUG) {
            ROS_INFO_STREAM(
              "done fetch " << cloud_src->size() << " : "
                            << cloud_target->size());
          }
          pcl::ApproximateVoxelGrid<PointType> avg;
          avg.setLeafSize(0.8F, 0.8F, 0.8F); // trade off
          avg.setInputCloud(cloud_src);
          avg.filter(*cloud_src);
          avg.setInputCloud(cloud_target);
          avg.filter(*cloud_target);
          icp3.setInputSource(cloud_src);
          icp3.setInputTarget(cloud_target);
          icp3.setMaxCorrespondenceDistance(
            (is_drone[ri] ? 5 : 3));   // 这里也需要更严格的限制，除了无人机

          Eigen::Quaterniond q_rel =
            MapDatabase[ri][si]->q_world_base.inverse() *
            MapDatabase[ri_test][si_test]->q_world_base;
          Eigen::Vector3d t_rel = MapDatabase[ri][si]->q_world_base.inverse() *
            (MapDatabase[ri_test][si_test]->t_world_base -
            MapDatabase[ri][si]->t_world_base);
          if (t_rel.norm() > 50) {
            continue;
          }
          Eigen::Matrix4f try_guess = Eigen::Matrix4f::Identity();
          try_guess.block(0, 0, 3, 3) =
            q_rel.normalized().toRotationMatrix().cast<float>();
          try_guess.block(0, 3, 3, 1) = t_rel.cast<float>();
          if (DEBUG) {
            ROS_INFO_STREAM("done guess");
          }
          icp3.align(*cloud_final_icp3, try_guess);
          if (DEBUG) {
            ROS_INFO_STREAM("done icp");
          }

          if (icp3.hasConverged()) {
            float final_score = icp3.getFitnessScore(5);
            if (DEBUG) {
              std::cout << "\nFitness score final adj" << final_score
                        << std::endl;
            }
            // save loop closure edge
            Eigen::Matrix3d R_rel = icp3.getFinalTransformation()
              .topLeftCorner(3, 3)
              .cast<double>();
            Eigen::Quaterniond q_rel_;
            q_rel_ = R_rel;
            Eigen::Vector3d t_rel_ = icp3.getFinalTransformation()
              .topRightCorner(3, 1)
              .cast<double>();

            MeasurementEdge measurement_edge(
              MapDatabase[ri][si]->pose_graph_local[0].stamp,
              MapDatabase[ri_test][si_test]->pose_graph_local[0].stamp, ri,
              ri_test, si, si_test, 0, 0,   // base node to base node
              q_rel_, t_rel_);
            if (final_score <
              (is_drone[ri] ?
              (is_drone[ri_test] ? 0.65 : 0.8) :
              0.65))          // 0.9 // 这里需要更加严格的条件，除了无人机
            // int accept;
            // std::cin >> accept;
            {
              if (std::find(
                  loopEdgeBuf.begin(), loopEdgeBuf.end(),
                  measurement_edge) == loopEdgeBuf.end())
              {
                m_loop.lock();
                loopEdgeBuf.push_back(measurement_edge);
                cnt++;

                // if (accept == 1) {
                // }
                m_loop.unlock();
                // performOptimization();
                if (DEBUG) {
                  ROS_WARN_STREAM("done add edge");
                }
                publish_opt_posegraph();
                break; // 添加回环边后即可退出
              }
            }
          }
        }
      }
    }
  }
  if (!DEBUG) {
    status(40, 1);
    std::cout << std::endl;
  }
  if (DEBUG) {
    ROS_WARN_STREAM(cnt << " global Edges added");
  }
  optimize();
  publish_opt_map(false, true);
}

void SubMapManagementObject::check_cachable_map()
{
  if (DEBUG) {
    std::cout << "\n Check cache\n ";
  }
  for (int ri = 0; ri <= cur_robot_id; ri++) {
    int temp_submap_size = MapDatabase[ri].size();
    if (ri == cur_robot_id) {
      temp_submap_size = cur_loop_test_id;
    }

    if (DEBUG) {
      std::cout << "\033[1;32m\nrobot[" << ri
                << "] submap num: " << MapDatabase[ri].size()
                << ", cur loop test id: " << cur_loop_test_id << "\n\033[0m";
    }
    for (int si = 0; si < temp_submap_size; si++) {
      if (si < temp_submap_size - 2 &&
        MapDatabase[ri][si]->access_status < temp_submap_size - 2)
      {
        MapDatabase[ri][si]->setCache();
      }
      // std::cout << "\n Check memory " << sizeof(*MapDatabase[ri][si]) <<
      // "\n";
    }
  }
}

void SubMapManagementObject::cache_process()
{
  unsigned int cnt = 0;
  while (continue_cache_thread) {
    if (cnt % 3 == 0) {
      check_cachable_map();
      publish_opt_map(false);
      cnt = 1;
    } else {
      cnt++;
    }

    publish_opt_posegraph(false);
    std::chrono::milliseconds dura(2000);
    std::this_thread::sleep_for(dura);
  }
}

} // namespace gacm
