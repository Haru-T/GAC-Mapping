/**
* This file is part of GAC-Mapping.
*
* Copyright (C) 2020-2022 JinHao He, Yilin Zhu / RAPID Lab, Sun Yat-Sen University
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
* without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
* PURPOSE. In no event will the authors be held liable for any damages
* arising from the use of this software. See the GNU General Public
* License for more details.
*
* You should have received a copy of the GNU General Public License
* along with GAC-Mapping. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef GACM__UTIL__IPUTIL_H_
#define GACM__UTIL__IPUTIL_H_

#include <opencv2/core/mat.hpp>
#include <string>

//将深度图映射到彩色域中
void generateFalseColors(cv::Mat src, cv::Mat & dst);

void generateFalseColors1(cv::Mat src, cv::Mat & dst, double maxv);

// int tmpcnt = 0;
void displayFalseColors(cv::Mat src, const std::string & name);

#endif  // GACM__UTIL__IPUTIL_H_
