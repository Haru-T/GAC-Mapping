#ifndef GACM__LOOP_CLOSURE__LOOP_SCORE_HPP__
#define GACM__LOOP_CLOSURE__LOOP_SCORE_HPP__

#include <Eigen/Core>
#include <Eigen/Dense>
#include <utility>

struct LoopScore {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  float loop_score;
  float angle_rel;
  int robot_id;
  int submap_id;
  Eigen::Matrix4f guess;
  Eigen::Matrix4f try_guess;

  LoopScore(float loop_score_, float angle_rel_, int robot_id_, int submap_id_,
            Eigen::Matrix4f guess_, Eigen::Matrix4f try_guess_)
      : loop_score(loop_score_), angle_rel(angle_rel_), robot_id(robot_id_),
        submap_id(submap_id_), guess(std::move(guess_)),
        try_guess(std::move(try_guess_)) {}
};

inline bool operator<(const LoopScore &lhs, const LoopScore &rhs) {
  return lhs.loop_score < rhs.loop_score ||
         (lhs.loop_score == rhs.loop_score && lhs.angle_rel < rhs.angle_rel);
}

#endif
