#include "gacm/util/ip_basic.h"

#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/hal/interface.h>

namespace {
const cv::Mat FULL_KERNEL_3 = cv::Mat::ones(3, 3, CV_8U);
const cv::Mat FULL_KERNEL_5 = cv::Mat::ones(5, 5, CV_8U);
const cv::Mat FULL_KERNEL_7 = cv::Mat::ones(7, 7, CV_8U);
const cv::Mat FULL_KERNEL_9 = cv::Mat::ones(9, 9, CV_8U);
const cv::Mat FULL_KERNEL_31 = cv::Mat::ones(31, 31, CV_8U);

// 3*3 cross kernel
const cv::Mat CROSS_KERNEL_3 = (cv::Mat_<uchar>(3, 3) << 0, 1, 0,
  1, 1, 1,
  0, 1, 0);

// 5*5 cross kernel
const cv::Mat CROSS_KERNEL_5 = (cv::Mat_<uchar>(5, 5) << 0, 0, 1, 0, 0,
  0, 0, 1, 0, 0,
  1, 1, 1, 1, 1,
  0, 0, 1, 0, 0,
  0, 0, 1, 0, 0);


// 7*7 cross kernel
const cv::Mat CROSS_KERNEL_7 = (cv::Mat_<uchar>(7, 7) << 0, 0, 0, 1, 0, 0, 0,
  0, 0, 0, 1, 0, 0, 0,
  0, 0, 0, 1, 0, 0, 0,
  1, 1, 1, 1, 1, 1, 1,
  0, 0, 0, 1, 0, 0, 0,
  0, 0, 0, 1, 0, 0, 0,
  0, 0, 0, 1, 0, 0, 0);

const cv::Mat TALL_KERNEL_5 = (cv::Mat_<uchar>(5, 5) << 0, 0, 1, 0, 0,
  0, 1, 1, 1, 0,
  0, 1, 1, 1, 0,
  0, 1, 1, 1, 0,
  0, 0, 1, 0, 0);

const cv::Mat TALL_KERNEL_7 = (cv::Mat_<uchar>(7, 7) << 0, 0, 0, 1, 0, 0, 0,
  0, 0, 1, 1, 1, 0, 0,
  0, 0, 1, 1, 1, 0, 0,
  0, 0, 1, 1, 1, 0, 0,
  0, 0, 1, 1, 1, 0, 0,
  0, 0, 1, 1, 1, 0, 0,
  0, 0, 0, 1, 0, 0, 0);


// 5*5 diamond kernel
const cv::Mat DIAMOND_KERNEL_5 = (cv::Mat_<uchar>(5, 5) << 0, 0, 1, 0, 0,
  0, 1, 1, 1, 0,
  1, 1, 1, 1, 1,
  0, 1, 1, 1, 0,
  0, 0, 1, 0, 0);

// 7*7 diamond kernel
const cv::Mat DIAMOND_KERNEL_7 = (cv::Mat_<uchar>(7, 7) << 0, 0, 0, 1, 0, 0, 0,
  0, 0, 1, 1, 1, 0, 0,
  0, 1, 1, 1, 1, 1, 0,
  1, 1, 1, 1, 1, 1, 1,
  0, 1, 1, 1, 1, 1, 0,
  0, 0, 1, 1, 1, 0, 0,
  0, 0, 0, 1, 0, 0, 0);
}

void getInvertDepth(cv::Mat src, cv::Mat & dst, const cv::Mat & mask)
{
  int type = src.type();
  if (type != CV_64F) {
    src.convertTo(src, CV_64F);
  }

  cv::Mat max = cv::Mat(src.rows, src.cols, CV_64F, cv::Scalar(100));
  cv::subtract(max, src, dst, mask);
  dst.convertTo(dst, type);
}


void generateAllMask(
  cv::Mat src, cv::Mat & valid, cv::Mat & invalid, cv::Mat & far, cv::Mat & med,
  cv::Mat & near)
{
  int type = src.type();
  if (type != CV_64F) {
    src.convertTo(src, CV_64F);
  }

  valid = cv::Mat::zeros(src.rows, src.cols, CV_8U);
  invalid = cv::Mat::zeros(src.rows, src.cols, CV_8U);
  far = cv::Mat::zeros(src.rows, src.cols, CV_8U);
  med = cv::Mat::zeros(src.rows, src.cols, CV_8U);
  near = cv::Mat::zeros(src.rows, src.cols, CV_8U);

  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      double cur_val = src.at<double>(i, j);
      if (cur_val == 0 || cur_val >= 256 * 200) {
        invalid.at<uchar>(i, j) = 1;
      } else {
        valid.at<uchar>(i, j) = 1;
        if (cur_val < 15 * 256) {
          near.at<uchar>(i, j) = 1;
        } else if (cur_val < 30 * 256) {
          med.at<uchar>(i, j) = 1;
        } else if (cur_val < 100 * 256) {
          far.at<uchar>(i, j) = 1;
        }
      }
    }
  }
}

void generateValidMask(cv::Mat src, cv::Mat & valid)
{
  int type = src.type();
  if (type != CV_64F) {
    src.convertTo(src, CV_64F);
  }

  valid = cv::Mat::zeros(src.rows, src.cols, CV_8U);

  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      double cur_val = src.at<double>(i, j);
      if (cur_val > 0) {
        valid.at<uchar>(i, j) = 1;
      }
    }
  }
}

void generateInvalidMask(cv::Mat src, cv::Mat & invalid)
{
  int type = src.type();
  if (type != CV_64F) {
    src.convertTo(src, CV_64F);
  }

  invalid = cv::Mat::zeros(src.rows, src.cols, CV_8U);

  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      double cur_val = src.at<double>(i, j);
      if (cur_val < 0.1) {
        invalid.at<uchar>(i, j) = 1;
      }
    }
  }
}

void extendTop(cv::Mat src, cv::Mat & dst)
{
  int type = src.type();
  if (type != CV_64F) {
    src.convertTo(src, CV_64F);
  }
  std::vector<int> top_indices;
  std::vector<double> top_values;

  for (int j = 0; j < src.cols; j++) {
    for (int i = 0; i < src.rows; i++) {
      double cur_val = src.at<double>(i, j);
      // std::cout << " " << cur_val;
      if (cur_val > 0.1) {
        top_indices.push_back(i);
        // std::cout << "Top " << i << std::endl;
        top_values.push_back(cur_val);
        break;
      }
    }
  }
  // std::cout<<"\n\n" << top_indices.size() << " " << src.cols << "\n";
  if (top_indices.size() != src.cols) {
    return;
  }
  for (int j = 0; j < src.cols; j++) {
    if (0 < top_indices[j] && top_indices[j] < src.cols) {
      cv::Mat(
        top_indices[j] + 1, 1, CV_64F,
        cv::Scalar(top_values[j])).copyTo(src.col(j).rowRange(0, top_indices[j] + 1));
    }
  }
  src.convertTo(dst, type);
}

void customDilate(cv::Mat src, cv::Mat & dst, int kernel_size, KernelType kernel_type)
{
  int type = src.type();
  if (type != CV_64F) {
    src.convertTo(src, CV_64F);
  }
  src = src / 1000.0;
  cv::Mat mask;
  generateValidMask(src, mask);
  getInvertDepth(src, src, mask);
  // displayFalseColors(src, "inverse depth");
  // cv::waitKey(0);
  cv::Mat kernel;
  switch (kernel_size) {
    case 3:
      if (kernel_type == KERNEL_TYPE_CROSS) {
        kernel = CROSS_KERNEL_3;
      }
      break;
    case 5:
      if (kernel_type == KERNEL_TYPE_CROSS) {
        kernel = CROSS_KERNEL_5;
      } else if (kernel_type == KERNEL_TYPE_DIAMOND) {
        kernel = DIAMOND_KERNEL_5;
      }
      break;
    case 7:
      if (kernel_type == KERNEL_TYPE_CROSS) {
        kernel = CROSS_KERNEL_7;
      } else if (kernel_type == KERNEL_TYPE_DIAMOND) {
        kernel = DIAMOND_KERNEL_7;
      }
      break;
  }

  cv::dilate(src, src, kernel);
  cv::dilate(src, src, TALL_KERNEL_7);
  // displayFalseColors(src, "after dilate");

  cv::morphologyEx(src, src, cv::MORPH_CLOSE, TALL_KERNEL_7);
  // displayFalseColors(src, "after mop close");

  cv::Mat tempdilate;
  cv::dilate(src, tempdilate, FULL_KERNEL_7);
  generateInvalidMask(src, mask);
  cv::Mat temphole;
  tempdilate.copyTo(temphole, mask);
  // cv::imshow("temphole", temphole);

  // displayFalseColors(temphole, "medium hole");
  // cv::Mat temp(src.rows, src.cols, CV_64F, cv::Scalar(500));

  // double minv = 0.0, maxv = 0.0;
  // double* minp = &minv;
  // double* maxp = &maxv;

  // minMaxIdx(src,minp,maxp);

  // std::cout << "Mat minv = " << minv << std::endl;
  // std::cout << "Mat maxv = " << maxv << std::endl;


  src = src + temphole;
  // displayFalseColors(src, "after add hole");
  // cv::waitKey(0);
  // displayFalseColors(temphole, "medium hole");

  // extendTop(src,src);
  // displayFalseColors(src, "after extend top");


  cv::dilate(src, tempdilate, FULL_KERNEL_31);
  generateInvalidMask(src, mask);
  cv::Mat bighole;
  tempdilate.copyTo(bighole, mask);
  // // displayFalseColors(bighole, "big hole");
  src = src + bighole;
  // displayFalseColors(src, "after add big hole");

  cv::Mat src_;
  src.convertTo(src_, CV_32F);
  cv::Mat median;
  cv::medianBlur(src_, median, 5);
  median.convertTo(median, CV_64F);
  src = median.clone();

  // generateValidMask(src, mask);
  // cv::Mat blur;
  // cv::GaussianBlur(src,blur,cv::Size(5,5),0);
  // cv::Mat blurvalid;
  // blur.copyTo(blurvalid, mask);
  // src = blurvalid.clone();

  generateValidMask(src, mask);
  getInvertDepth(src, dst, mask);
  dst = dst * 1000.0;
  dst.convertTo(dst, type);
}
