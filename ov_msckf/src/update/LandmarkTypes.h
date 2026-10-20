# ifndef OV_MSCKF_LANDMARK_TYPES_H
# define OV_MSCKF_LANDMARK_TYPES_H

#include <ov_landmark/LandmarkTypes.h>

#include <Eigen/Core>

#include <cstdint>
#include <map>
#include <vector>

namespace ov_msckf {

using LandmarkId = ov_landmark::LandmarkId;
using LandmarkMap = ov_landmark::LandmarkMap;

// Landmark category types
enum class LandmarkType { ARUCO = 0, WINDOW, PIPE, HOTBOX, COPPER_DISCS, GENERIC };

// Observation of a landmark in camera frame
struct LandmarkObservation {
    double timestamp = -1.0;
    size_t cam_id = 0;

    int landmark_id = -1;
    std::vector<Eigen::Vector2d> points_cam;

    double sigma_px = 1.0;
};

// Update config for each landmark type
struct LandmarkTypeConfig {
    bool enabled = false;
    double sigma_px = 1.0;
    double chi2_multiplier = 5.0;
};


} // namespace ov_msckf

# endif
