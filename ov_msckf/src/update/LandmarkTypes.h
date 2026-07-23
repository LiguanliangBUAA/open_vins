# ifndef OV_MSCKF_LANDMARK_TYPES_H
# define OV_MSCKF_LANDMARK_TYPES_H

# include <Eigen/Eigen>
# include <array>
# include <map>
# include <string>
# include <vector>

namespace ov_msckf {

/// Landmark category types
enum class LandmarkType { ARUCO = 0, WINDOW, PIPE, HOTBOX, COPPER_DISCS, GENERIC };

/// Landmark -> a set of 3D points in map/earth frame. (ground truth)
/// ARUCO: 4 points (corners); WINDOW: 4 points (corners); PIPE: 2 points (endpoints);
/// HOTBOX: 1 point (center); COPPER_DISC: 1 point (center);
struct MapLandmark {
    int id;
    LandmarkType type;
    std::vector<Eigen::Vector3d> points_map;
};

/// Observation of a landmark in camera frame
struct LandmarkObservation {
    double timestamp = -1.0;
    size_t cam_id = 0;
    int landmark_id = -1;
    std::vector<Eigen::Vector2d> points_cam;
    // (u,v) same order | (-1,-1) if not observed
    // 4 points: left-top, right-top, right-bottom, left-bottom
    double sigma_px = 1.0; // pixel noise stddev
};

/// Update config for each landmark type
struct LandmarkTypeConfig {
    bool enabled = false;
    double sigma_px = 1.0;
    double chi2_multiplier = 5.0;
};


} // namespace ov_msckf

# endif