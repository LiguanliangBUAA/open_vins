# ifndef OV_MSCKF_ARUCO_MAP_DETECTOR_H
# define OV_MSCKF_ARUCO_MAP_DETECTOR_H

# include "LandmarkTypes.h"

# include <opencv2/aruco.hpp>
# include <opencv2/opencv.hpp>

namespace ov_msckf {

class ArucoMapDetector {
public:
    /// landmarks: aruco-type entries of the global map (filtered by caller)
    ArucoMapDetector(const std::map<int, MapLandmark> &landmarks, double sigma_px);
    /// Returns observations of known aruco markers only, in the unified format
    std::vector<LandmarkObservation> detect(double timestamp, size_t cam_id, const cv::Mat &img);

private:
    std::map<int, MapLandmark> _landmarks;
    double _sigma_px;
    cv::Ptr<cv::aruco::Dictionary> _dict;
    cv::Ptr<cv::aruco::DetectorParameters> _params;
};

} // namespace ov_msckf

# endif