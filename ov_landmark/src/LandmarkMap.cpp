#include "ov_landmark/LandmarkMap.h"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ov_landmark {

namespace {

std::runtime_error yaml_error(const std::string &path, const std::string &message) {
    return std::runtime_error("invalid landmark map '" + path + "': " + message);
}

} // namespace

std::string resolve_landmark_map_path(const std::string &yaml_path, const std::string &config_folder) {
    if (yaml_path.empty()) {
        return ament_index_cpp::get_package_share_directory("ov_landmark") + "/config/landmarks.yaml";
    }
    if (yaml_path.front() == '/' || config_folder.empty()) {
        return yaml_path;
    }
    return config_folder + (config_folder.back() == '/' ? "" : "/") + yaml_path;
}

LandmarkMap load_landmark_map(const std::string &yaml_path) {
    if (yaml_path.empty()) {
        throw std::invalid_argument("landmark map path is empty");
    }

    YAML::Node root;

    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const YAML::Exception &e) {
        throw yaml_error(yaml_path, e.what());
    }

    const YAML::Node landmarks = root["landmarks"];

    if (!landmarks || !landmarks.IsSequence()) {
        throw yaml_error(yaml_path, "top-level 'landmarks' must be a sequence");
    }

    LandmarkMap result;
    result.reserve(landmarks.size());

    for (std::size_t index = 0; index < landmarks.size(); ++index) {
        const YAML::Node item = landmarks[index];

        if (!item.IsMap()) {
            throw yaml_error(yaml_path, 
                "landmark at index " + std::to_string(index) + " must be a mapping");
        }

        if (!item["id"]) {
            throw yaml_error(yaml_path, 
                "landmark at index " + std::to_string(index) + " is missing 'id'");
        }

        if (!item["xyz"]) {
            throw yaml_error(yaml_path, 
                "landmark at index " + std::to_string(index) + " is missing 'xyz'");
        }

        const std::uint64_t id64 = item["id"].as<std::uint64_t>();

        if (id64 > std::numeric_limits<LandmarkId>::max()) {
            throw yaml_error(yaml_path, 
                "landmark at index " + std::to_string(index) + " has 'id' out of range");
        }

        const LandmarkId id = static_cast<LandmarkId>(id64);
        const YAML::Node xyz = item["xyz"];

        if (!xyz.IsSequence() || xyz.size() != 3) {
            throw yaml_error(yaml_path, 
                "landmark at index " + std::to_string(index) + " has invalid 'xyz'");
        }

        Eigen::Vector3d p_M;

        try {
            p_M << xyz[0].as<double>(), xyz[1].as<double>(), xyz[2].as<double>();
        } catch (const YAML::Exception &e) {
            throw yaml_error(yaml_path, 
                "landmark at index " + std::to_string(index) + " has invalid 'xyz': " + e.what());
        }

        const auto inserted = result.emplace(id, p_M);

        if (!inserted.second) {
            throw yaml_error(yaml_path, 
                "duplicate landmark id " + std::to_string(id) + " at index " + std::to_string(index));
        }
    }

    if (result.empty()) {
        throw yaml_error(yaml_path, "no landmarks found");
    }

    return result;
}

std::vector<LandmarkObject> load_landmark_objects(const std::string &yaml_path) {
    // load_landmark_map() has already validated id and xyz
    const YAML::Node landmarks = YAML::LoadFile(yaml_path)["landmarks"];

    std::map<std::string, LandmarkObject> by_name;
    for (std::size_t index = 0; index < landmarks.size(); ++index) {
        const YAML::Node item = landmarks[index];
        if (!item["object"] || !item["type"]) {
            continue;
        }
        LandmarkObject &obj = by_name[item["object"].as<std::string>()];
        const std::string type = item["type"].as<std::string>();
        if (!obj.type.empty() && obj.type != type) {
            throw yaml_error(yaml_path, "object '" + item["object"].as<std::string>() + "' has more than one type");
        }
        obj.name = item["object"].as<std::string>();
        obj.type = type;
        obj.ids.push_back(item["id"].as<LandmarkId>());
    }

    std::vector<LandmarkObject> result;
    for (auto &entry : by_name) {
        std::sort(entry.second.ids.begin(), entry.second.ids.end());
        result.push_back(entry.second);
    }
    return result;
}

} // namespace ov_landmark
