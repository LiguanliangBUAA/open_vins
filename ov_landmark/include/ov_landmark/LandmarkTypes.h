#ifndef OV_LANDMARK_LANDMARK_TYPES_H
#define OV_LANDMARK_LANDMARK_TYPES_H

#include <Eigen/Core>

#include <cstdint>
#include <unordered_map>

namespace ov_landmark {
    using LandmarkId = std::uint32_t;
    using LandmarkMap = std::unordered_map<LandmarkId, Eigen::Vector3d>; // 3D point in map frame
} // namespace ov_landmark

#endif //OV_LANDMARK_LANDMARK_TYPES_H
