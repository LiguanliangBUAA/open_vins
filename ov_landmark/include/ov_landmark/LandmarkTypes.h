#ifndef OV_LANDMARK_LANDMARK_TYPES_H
#define OV_LANDMARK_LANDMARK_TYPES_H

#include <Eigen/Core>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ov_landmark {
    using LandmarkId = std::uint32_t;
    using LandmarkMap = std::unordered_map<LandmarkId, Eigen::Vector3d>; // 3D point in map frame

    // A detectable object (e.g. a gate): its type is the detector class, ids are in keypoint order (ascending id)
    struct LandmarkObject {
        std::string name;
        std::string type;
        std::vector<LandmarkId> ids;
    };
} // namespace ov_landmark

#endif //OV_LANDMARK_LANDMARK_TYPES_H
