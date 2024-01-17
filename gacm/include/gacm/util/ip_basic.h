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

#ifndef GACM__UTIL__IP_BASIC_H_
#define GACM__UTIL__IP_BASIC_H_

#include <opencv2/core/mat.hpp>

enum KernelType { KERNEL_TYPE_CROSS = 1, KERNEL_TYPE_DIAMOND = 2 };

void getInvertDepth(cv::Mat src, cv::Mat & dst, const cv::Mat & mask);

void generateAllMask(
  cv::Mat src, cv::Mat & valid, cv::Mat & invalid,
  cv::Mat & far, cv::Mat & med, cv::Mat & near);

void generateValidMask(cv::Mat src, cv::Mat & valid);

void generateInvalidMask(cv::Mat src, cv::Mat & invalid);

void extendTop(cv::Mat src, cv::Mat & dst);

void customDilate(cv::Mat src, cv::Mat & dst, int kernel_size, KernelType kernel_type);

#endif // GACM__UTIL__IP_BASIC_H_
