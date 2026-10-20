#ifndef OV_LANDMARK_ARUCO_IDS_H
#define OV_LANDMARK_ARUCO_IDS_H

#include "ov_landmark/LandmarkTypes.h"

namespace ov_landmark {

constexpr LandmarkId ARUCO_ID_BASE = 1000;
constexpr int MAX_ARUCO_MARKER_ID = 249;
constexpr LandmarkId INVALID_LANDMARK_ID = 0;

inline LandmarkId aruco_corner_id (int marker_id, int corner_idx, int corner_offset = 0) {
    if (marker_id < 0 || marker_id > MAX_ARUCO_MARKER_ID || corner_idx < 0 || corner_idx > 3) {
        return INVALID_LANDMARK_ID;
    }

    const int normalized_offset = ((corner_offset % 4) + 4) % 4;
    const int mapped_corner = (corner_idx + normalized_offset) % 4;
    return ARUCO_ID_BASE + static_cast<LandmarkId>(marker_id * 4 + mapped_corner);
}

} // namespace ov_landmark

#endif //OV_LANDMARK_ARUCO_IDS_H