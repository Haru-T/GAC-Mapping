# syntax=docker.io/docker/dockerfile:1

FROM docker.io/library/ubuntu:20.04 AS ceres-builder

ARG DEBIAN_FRONTEND=noninteractive
ADD https://github.com/ceres-solver/ceres-solver.git#2.0.0 /ceres-solver
WORKDIR /ceres-bin
RUN apt-get update \
&& apt-get install -y --no-install-recommends \
build-essential g++ cmake ninja-build libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev \
&& cmake ../ceres-solver -GNinja && cmake --build .

FROM docker.io/osrf/ros:noetic-desktop-full
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
&& apt-get install -y --no-install-recommends \
ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-tf ros-${ROS_DISTRO}-message-filters ros-${ROS_DISTRO}-image-transport \
libgoogle-glog-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev libopencv-dev \
cmake ninja-build python3-catkin-tools ca-certificates curl gzip sudo clangd gdb

RUN --mount=type=bind,from=ceres-builder,source=/ceres-solver,target=/ceres-solver,rw \
    --mount=type=bind,from=ceres-builder,source=/ceres-bin,target=/ceres-bin,rw cd /ceres-bin && ninja install

RUN groupadd --gid ${USER_GID} ${USERNAME} \
&& useradd -s /usr/bin/bash --uid ${USER_UID} --gid ${USERNAME} -m ${USERNAME} \
&& echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
&& chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}

ADD --chown=${USERNAME}:${USERNAME} https://github.com/mitsuhiko/rye/releases/latest/download/rye-x86_64-linux.gz /tmp/rye.gz
RUN gunzip /tmp/rye.gz && chmod +x /tmp/rye && /tmp/rye self install --yes && rm /tmp/rye && . ${HOME}/.rye/env && rye fetch 3.8 \
&& sudo mkdir -p /workspaces/src && sudo chown -R ${USERNAME}:${USERNAME} /workspaces
