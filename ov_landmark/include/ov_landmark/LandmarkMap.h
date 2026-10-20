#ifndef OV_LANDMARK_LANDMARK_MAP_H
#define OV_LANDMARK_LANDMARK_MAP_H

#include "ov_landmark/LandmarkTypes.h"

#include <string>

namespace ov_landmark {

// Resolve the landmark map path given in the estimator config
std::string resolve_landmark_map_path(const std::string &yaml_path, const std::string &config_folder);

// Load known landmark points from the yaml file.
LandmarkMap load_landmark_map(const std::string &yaml_path);

} // namespace ov_landmark

#endif //OV_LANDMARK_LANDMARK_MAP_H
