cmake_minimum_required(VERSION 3.8)
project(gacm)

# set(CMAKE_BUILD_TYPE debug)

IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF ()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g  -Wall  -O3 -march=native ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g -Wall   -O3 -march=native")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O1 -g -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-common -fsanitize=address")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -g -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-common -fsanitize=address")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=address")

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

find_package(catkin REQUIRED COMPONENTS
        tf
        roscpp
        rospy
        cv_bridge
        image_transport

        #pcl_ros
        #pcl_conversions

        std_msgs
        sensor_msgs
        geometry_msgs
        nav_msgs
        cloud_msgs

        camera_model
        netvlad_tf_test
)


find_package(OpenCV QUIET)
# find_package(Sophus REQUIRED)
# find_package(Eigen3 REQUIRED)

catkin_package(
        INCLUDE_DIRS include
        CATKIN_DEPENDS cloud_msgs
)


include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}

)

link_directories(
        include
        ${OpenCV_LIBRARY_DIRS}
)

add_library(gacm
  src/mapping/sub_map.cpp
  src/parameters.cpp
  src/featureExtractor.cpp
  src/scanRegistrator.cpp
  src/poseEstimator.cpp
  src/util/iputil.cpp
  src/util/ip_basic.cpp)
target_link_libraries(gacm
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRAIRES}
  Eigen3::Eigen
  Ceres::ceres)

add_executable(featureTrackingNode src/featureTrackingNode.cpp)
add_dependencies(featureTrackingNode
  ${catkin_EXPORTED_TARGETS})
target_link_libraries(featureTrackingNode
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
  gacm)

add_executable(plOdometryNode src/plOdometryNode.cpp)
add_dependencies(plOdometryNode
  ${catkin_EXPORTED_TARGETS})
target_link_libraries(plOdometryNode
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  Eigen3::Eigen
  Ceres::ceres
  gacm)

add_executable(submapManagerNode src/submapManagerNode.cpp)
add_dependencies(submapManagerNode
  ${catkin_EXPORTED_TARGETS})
target_link_libraries(submapManagerNode
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  Eigen3::Eigen
  Ceres::ceres
  gacm)

add_executable(gacm_uninode
  src/uninode/main.cpp
  src/uninode/feature_tracking.cpp
  src/uninode/odometry.cpp
  src/uninode/submap_management.cpp
  src/uninode/file_enumeration.cpp)
add_dependencies(gacm_uninode
  ${catkin_EXPORTED_TARGETS})
target_link_libraries(gacm_uninode
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  Eigen3::Eigen
  Ceres::ceres
  gacm)
