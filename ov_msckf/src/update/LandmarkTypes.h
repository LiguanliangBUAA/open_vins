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

// One landmark point seen in the image, raw pixel coordinates
struct LandmarkPoint {
    LandmarkId id;
    double u;
    double v;
};

// All landmark points seen in one image
struct LandmarkObservation {
    double timestamp = -1.0;
    std::vector<LandmarkPoint> points;
};

// Update config for each landmark type
struct LandmarkTypeConfig {
    bool enabled = false;
    double sigma_px = 1.0;
    double chi2_multiplier = 5.0;
};


} // namespace ov_msckf

# endif
