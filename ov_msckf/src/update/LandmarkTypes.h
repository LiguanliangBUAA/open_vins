# ifndef OV_MSCKF_LANDMARK_TYPES_H
# define OV_MSCKF_LANDMARK_TYPES_H

#include <ov_landmark/LandmarkTypes.h>

#include <Eigen/Core>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ov_msckf {

using LandmarkId = ov_landmark::LandmarkId;
using LandmarkMap = ov_landmark::LandmarkMap;
using LandmarkObject = ov_landmark::LandmarkObject;

// Landmark category types
enum class LandmarkType { ARUCO = 0, WINDOW, PIPE, HOTBOX, COPPER_DISCS, GENERIC };

// One landmark point seen in the image, raw pixel coordinates
struct LandmarkPoint {
    LandmarkId id;
    double u;
    double v;
};

// One detected object without identity (e.g. YOLO): class = landmark object type, keypoints in that type's order
struct ObjectDetection {
    std::string type;
    std::vector<LandmarkPoint> keypoints; // id unused
    std::vector<double> scores;           // per keypoint, 0 = not visible
    double box_area = 0.0;                // pixels^2
};

// All landmark points seen in one image. points have known ids (e.g. ArUco), objects are matched to the map in the filter.
struct LandmarkObservation {
    double timestamp = -1.0;
    std::vector<LandmarkPoint> points;
    std::vector<ObjectDetection> objects;
};

// Update config for each landmark type
struct LandmarkTypeConfig {
    bool enabled = false;
    double sigma_px = 1.0;
    double chi2_multiplier = 5.0;
};


} // namespace ov_msckf

# endif
