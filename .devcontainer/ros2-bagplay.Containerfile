# syntax=docker.io/docker/dockerfile:1

FROM docker.io/library/ros:humble-ros-base-jammy

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ros-humble-rosbag2-storage-mcap
