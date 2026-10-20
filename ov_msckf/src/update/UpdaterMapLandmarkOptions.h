#ifndef OV_MSCKF_UPDATER_MAP_LANDMARK_OPTIONS_H
#define OV_MSCKF_UPDATER_MAP_LANDMARK_OPTIONS_H

#include "LandmarkTypes.h"

#include <Eigen/Eigen>
#include <cmath>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "ov_landmark/LandmarkMap.h"
#include "utils/colors.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

namespace ov_msckf {

/**
 * @brief Options for the map-to-global transform T_MtoG and the map landmark update
 *
 * T_MtoG is stored as a PoseJPL (q_MtoG, p_GinM), i.e. the pose of the OpenVINS global frame G in the map frame M.
 * Its orientation error is the JPL left error expressed in G, so components x/y are roll/pitch (G is gravity aligned)
 * and component z is yaw. Roll/pitch and yaw are given separate noise so the tilt can be kept tight.
 */
struct UpdaterMapLandmarkOptions {
    std::string map_landmarks_yaml; // Known landmark points in the map frame, loaded from a yaml file
    std::map<LandmarkType, LandmarkTypeConfig> type_cfg; // per-type config - noise/gating/enable
    int init_min_obs = 3; // observation times before delayed init
    double init_max_dist = 1e4; // PnP conditioning gate at init
    double max_obs_age = 0.15; // drop observations older than newest clone by this [s]

    /// Prior pose of the body frame B in the map M at the time VIO initializes (the takeoff pad)
    Eigen::Matrix4d T_M_B_init = Eigen::Matrix4d::Identity();

    /// Pose of the body frame B in the IMU frame I (kalibr imu0.T_i_b), maps p_B to p_I
    Eigen::Matrix4d T_I_B = Eigen::Matrix4d::Identity();

    /// Initial standard deviation of T_MtoG (roll/pitch [deg], yaw [deg], position [m])
    double sigma0_rp_deg = 2.0;
    double sigma0_yaw_deg = 5.0;
    double sigma0_pos_m = 1.0;

    /// Random-walk of T_MtoG (roll/pitch [deg/sqrt(s)], yaw [deg/sqrt(s)], position [m/sqrt(s)])
    double rw_rp_deg_sqrt_s = 0.01;
    double rw_yaw_deg_sqrt_s = 0.05;
    double rw_pos_m_sqrt_s = 0.03;

    /// Optional base_link navigation outputs (PoseStamped / TwistStamped), empty = not published
    std::string nav_pose_topic = "";
    std::string nav_twist_topic = "";

    /// Initial covariance diagonal of T_MtoG in [rot, pos] order
    Eigen::Matrix<double, 6, 1> sigma0_vec() const {
        Eigen::Matrix<double, 6, 1> s;
        double rp = sigma0_rp_deg * M_PI / 180.0;
        double yaw = sigma0_yaw_deg * M_PI / 180.0;
        s << rp, rp, yaw, sigma0_pos_m, sigma0_pos_m, sigma0_pos_m;
        return s;
    }

    /// Random-walk power spectral density diagonal of T_MtoG in [rot, pos] order
    Eigen::Matrix<double, 6, 1> rw_psd_vec() const {
        Eigen::Matrix<double, 6, 1> q;
        double rp = rw_rp_deg_sqrt_s * M_PI / 180.0;
        double yaw = rw_yaw_deg_sqrt_s * M_PI / 180.0;
        q << rp * rp, rp * rp, yaw * yaw, rw_pos_m_sqrt_s * rw_pos_m_sqrt_s, rw_pos_m_sqrt_s * rw_pos_m_sqrt_s,
            rw_pos_m_sqrt_s * rw_pos_m_sqrt_s;
        return q;
    }

    /// Aruco perception input
    std::string map_aruco_topic = "/drone0/detections/aruco/front";
    std::string map_aruco_cam = "cam3";
    int map_aruco_corner_offset = 0;
    double map_aruco_timeshift_cam_imu = 0.0;

    /// ArUco camera calibration from relative_config_imucam, p_C = T_C_I * p_I
    Eigen::Matrix4d aruco_T_C_I = Eigen::Matrix4d::Identity();
    std::shared_ptr<ov_core::CamBase> aruco_camera;

    /// Landmark update
    bool map_update_dry_run = true; // only compute and print the residuals
    bool map_check_jacobian = false; // compare the Jacobian with numerical differences on the first frames
    double map_max_clone_dt = 0.001; // use an observation only if a clone is this close in time [s]
    double map_sigma_px = 2.0; // pixel noise of a detected corner
    double map_sigma_landmark_pos_m = 0.05; // position uncertainty of a surveyed landmark point
    double map_chi2_multiplier = 1.0; // gate: chi2 < multiplier * chi2_0.95(2N)

    void print(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
        PRINT_DEBUG("MAP LANDMARK PARAMETERS:\n");
        if (parser != nullptr) {
            // Row-major 4x4, same layout as initial_T_M_B in map_alignment_ekf_*.yaml
            std::vector<double> T_flat;
            parser->parse_config("map_initial_T_M_B", T_flat);
            if (T_flat.size() != 16) {
                PRINT_ERROR(RED "map_initial_T_M_B must contain exactly 16 numbers (got %d)\n" RESET, (int)T_flat.size());
                std::exit(EXIT_FAILURE);
            }
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    T_M_B_init(r, c) = T_flat.at(4 * r + c);
                }
            }
            check_transform("map_initial_T_M_B", T_M_B_init);

            // Landmark map: empty = ov_landmark's config/landmarks.yaml, relative = relative to this config file
            parser->parse_config("map_landmarks_yaml", map_landmarks_yaml, false);
            map_landmarks_yaml = ov_landmark::resolve_landmark_map_path(map_landmarks_yaml, parser->get_config_folder());

            // Optional: stays identity (body == IMU) if the imu chain has no T_i_b
            parser->parse_external("relative_config_imu", "imu0", "T_i_b", T_I_B, false);
            check_transform("imu0.T_i_b", T_I_B);

            parser->parse_config("map_sigma0_rp_deg", sigma0_rp_deg, false);
            parser->parse_config("map_sigma0_yaw_deg", sigma0_yaw_deg, false);
            parser->parse_config("map_sigma0_pos_m", sigma0_pos_m, false);
            parser->parse_config("map_rw_rp_deg_sqrt_s", rw_rp_deg_sqrt_s, false);
            parser->parse_config("map_rw_yaw_deg_sqrt_s", rw_yaw_deg_sqrt_s, false);
            parser->parse_config("map_rw_pos_m_sqrt_s", rw_pos_m_sqrt_s, false);
            parser->parse_config("map_nav_pose_topic", nav_pose_topic, false);
            parser->parse_config("map_nav_twist_topic", nav_twist_topic, false);

            parser->parse_config("map_aruco_topic", map_aruco_topic, false);
            parser->parse_config("map_aruco_cam", map_aruco_cam, false);
            parser->parse_config("map_aruco_corner_offset", map_aruco_corner_offset, false);

            if (map_aruco_corner_offset < -3 || map_aruco_corner_offset > 3) {
                PRINT_WARNING(YELLOW "map_aruco_corner_offset = %d is outside [-3, 3], will be wrapped modulo 4\n" RESET,
                              map_aruco_corner_offset);
            }
            map_aruco_corner_offset = ((map_aruco_corner_offset % 4) + 4) % 4;
            
            parser->parse_config("map_aruco_timeshift_cam_imu", map_aruco_timeshift_cam_imu, false);
            load_aruco_camera(parser);

            parser->parse_config("map_update_dry_run", map_update_dry_run, false);
            parser->parse_config("map_check_jacobian", map_check_jacobian, false);
            parser->parse_config("map_max_clone_dt", map_max_clone_dt, false);
            parser->parse_config("map_sigma_px", map_sigma_px, false);
            parser->parse_config("map_sigma_landmark_pos_m", map_sigma_landmark_pos_m, false);
            parser->parse_config("map_chi2_multiplier", map_chi2_multiplier, false);
        }
        std::stringstream ss;
        ss << "  - map_initial_T_M_B:" << std::endl << T_M_B_init << std::endl;
        ss << "  - imu0.T_i_b:" << std::endl << T_I_B << std::endl;
        PRINT_DEBUG(ss.str().c_str());
        PRINT_DEBUG("  - map_landmarks_yaml: %s\n", map_landmarks_yaml.c_str());
        PRINT_DEBUG("  - map_sigma0: rp %.3f deg | yaw %.3f deg | pos %.3f m\n", sigma0_rp_deg, sigma0_yaw_deg, sigma0_pos_m);
        PRINT_DEBUG("  - map_rw: rp %.4f deg/sqrt(s) | yaw %.4f deg/sqrt(s) | pos %.4f m/sqrt(s)\n", rw_rp_deg_sqrt_s, rw_yaw_deg_sqrt_s,
                    rw_pos_m_sqrt_s);
        PRINT_DEBUG("  - map_nav_pose_topic: %s\n", nav_pose_topic.c_str());
        PRINT_DEBUG("  - map_nav_twist_topic: %s\n", nav_twist_topic.c_str());
        PRINT_DEBUG("  - map_aruco_topic: %s\n", map_aruco_topic.c_str());
        PRINT_DEBUG("  - map_aruco_cam: %s | corner_offset %d | timeshift %.4f s\n", map_aruco_cam.c_str(), map_aruco_corner_offset,
                    map_aruco_timeshift_cam_imu);
        if (aruco_camera != nullptr) {
            std::stringstream ss_cam;
            ss_cam << "  - aruco T_C_I:" << std::endl << aruco_T_C_I << std::endl;
            ss_cam << "  - aruco intrinsics (fx fy cx cy d0..d3): " << aruco_camera->get_value().transpose() << " | " << aruco_camera->w()
                   << "x" << aruco_camera->h() << std::endl;
            PRINT_DEBUG(ss_cam.str().c_str());
        }
        PRINT_DEBUG("  - map_update_dry_run: %d | map_check_jacobian: %d | map_max_clone_dt: %.4f s\n", map_update_dry_run,
                    map_check_jacobian, map_max_clone_dt);
        PRINT_DEBUG("  - map_sigma_px: %.2f | map_sigma_landmark_pos_m: %.3f | map_chi2_multiplier: %.2f\n", map_sigma_px,
                    map_sigma_landmark_pos_m, map_chi2_multiplier);
    }

private:
    void load_aruco_camera(const std::shared_ptr<ov_core::YamlParser> &parser) {
        parser->parse_external("relative_config_imucam", map_aruco_cam, "T_cam_imu", aruco_T_C_I);
        check_transform(map_aruco_cam + ".T_cam_imu", aruco_T_C_I);

        std::string dist_model = "radtan";
        std::vector<double> k = {1, 1, 0, 0};
        std::vector<double> d = {0, 0, 0, 0};
        std::vector<int> wh = {1, 1};
        parser->parse_external("relative_config_imucam", map_aruco_cam, "distortion_model", dist_model);
        parser->parse_external("relative_config_imucam", map_aruco_cam, "intrinsics", k);
        parser->parse_external("relative_config_imucam", map_aruco_cam, "distortion_coeffs", d);
        parser->parse_external("relative_config_imucam", map_aruco_cam, "resolution", wh);

        Eigen::VectorXd calib(8);
        calib << k.at(0), k.at(1), k.at(2), k.at(3), d.at(0), d.at(1), d.at(2), d.at(3);
        if (dist_model == "equidistant") {
            aruco_camera = std::make_shared<ov_core::CamEqui>(wh.at(0), wh.at(1));
        } else {
            aruco_camera = std::make_shared<ov_core::CamRadtan>(wh.at(0), wh.at(1));
        }
        aruco_camera->set_value(calib);
    }

    static void check_transform(const std::string &name, const Eigen::Matrix4d &T) {
        Eigen::Matrix3d R = T.block(0, 0, 3, 3);
        bool ok = T.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-8) &&
                  (R.transpose() * R).isApprox(Eigen::Matrix3d::Identity(), 1e-5) && std::abs(R.determinant() - 1.0) < 1e-5;
        if (!ok) {
            PRINT_ERROR(RED "%s is not a valid SE(3) transform (orthonormal, det(R)=+1, last row [0 0 0 1])\n" RESET, name.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
};

} // namespace ov_msckf

#endif
