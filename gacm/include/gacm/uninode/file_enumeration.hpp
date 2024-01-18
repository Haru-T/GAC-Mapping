#ifndef GACM__UNINODE__FILE_ENUMERATION_HPP_
#define GACM__UNINODE__FILE_ENUMERATION_HPP_

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <string_view>
#include <type_traits>
#include <vector>

#include <Eigen/Geometry>

namespace gacm {
int64_t filename_to_timestamp(const std::filesystem::path &filename);

template <class DirectoryIterator>
std::vector<std::filesystem::path> enumerate_file(DirectoryIterator iter,
                                                  std::string_view extension) {
  std::vector<std::filesystem::path> files;
  for (; iter != DirectoryIterator(); ++iter) {
    const auto &path = iter->path();
    if (path.extension() == extension) {
      files.push_back(path);
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

struct Odom {
  int64_t timestamp_nanoseconds;
  struct {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
  } pose;
};

std::vector<Odom> load_odometry(const std::string &path);
} // namespace gacm

#endif
