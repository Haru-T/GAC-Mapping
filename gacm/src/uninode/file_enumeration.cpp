#include "gacm/uninode/file_enumeration.hpp"

#include <charconv>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <system_error>

#include <Eigen/Geometry>

namespace gacm
{
int64_t filename_to_timestamp(const std::filesystem::path & filename)
{
  auto stem = filename.stem().native();
  int64_t timestamp{};
  auto result =
    std::from_chars(stem.c_str(), stem.c_str() + stem.size(), timestamp);
  if (result.ptr == stem.c_str()) {
    throw std::system_error(std::make_error_code(result.ec));
  }
  return timestamp;
}

std::vector<Odom> load_odometry(const std::string & path)
{
  std::vector<Odom> odometry;

  std::FILE * fp = std::fopen(path.c_str(), "r");
  char c{};
  if ((c = std::fgetc(fp)) == '#') {
    std::fscanf(fp, "%*[^\n]\n");
  } else {
    std::ungetc(c, fp);
  }

  while ((c = std::fgetc(fp)) != EOF) {
    std::ungetc(c, fp);
    int64_t timestamp;
    double x, y, z, qw, qx, qy, qz;
    if (std::fscanf(
        fp, "%" SCNd64 ".%*d %lf %lf %lf %lf %lf %lf %lf",
        &timestamp, &x, &y, &z, &qx, &qy, &qz, &qw) != 8)
    {
      break;
    }
    odometry.push_back(
      {
        timestamp,
        {
          Eigen::Vector3d(x, y, z),
          Eigen::Quaterniond(qw, qx, qy, qz),
        },
      });
  }
  std::fclose(fp);
  return odometry;
}
} // namespace gacm
