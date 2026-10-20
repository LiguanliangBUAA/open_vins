#include "UpdaterMapLandmark.h"

#include "state/State.h"
#include "state/StateHelper.h"
#include "types/PoseJPL.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterMapLandmark::UpdaterMapLandmark(UpdaterMapLandmarkOptions opts, LandmarkMap landmark_map)
    : _opts(std::move(opts)), _map(std::move(landmark_map)) {
        if (_map.empty()) {
            throw std::invalid_argument("landmark map is empty");
        }

        LandmarkId min_id = std::numeric_limits<LandmarkId>::max();
        LandmarkId max_id = std::numeric_limits<LandmarkId>::lowest();

        for (const auto &entry : _map) {
            min_id = std::min(min_id, entry.first);
            max_id = std::max(max_id, entry.first);
        }

        PRINT_INFO(GREEN "[MAP]: loaded %zu landmarks from %s, id range [%u, %u]\n" 
            RESET, _map.size(), _opts.map_landmarks_yaml.c_str(), 
            static_cast<unsigned int>(min_id), static_cast<unsigned int>(max_id));
    }

void UpdaterMapLandmark::feed_observation(const LandmarkObservation &obs) {
    LandmarkObservation known;
    known.timestamp = obs.timestamp;
    for (const auto &p : obs.points) {
        if (_map.count(p.id) > 0) {
            known.points.push_back(p);
        }
    }
    if (known.points.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lck(_pending_mtx);
    _pending.push_back(known);
    if (_pending.size() > 100) {
        _pending.erase(_pending.begin());
    }
}

void UpdaterMapLandmark::update(std::shared_ptr<State> state) {
    std::vector<LandmarkObservation> ready = take_ready(state->_timestamp);
    if (!_initialized) {
        _initialized = try_initialize(state);
        return;
    }
    propagate(state);

    for (const auto &obs : ready) {
        print_observation(state, obs);
        // TODO: build_landmark_system + StateHelper::EKFUpdate
    }
}

std::vector<LandmarkObservation> UpdaterMapLandmark::take_ready(double t) {
    std::lock_guard<std::mutex> lck(_pending_mtx);
    std::vector<LandmarkObservation> ready, waiting;
    for (const auto &obs : _pending) {
        (obs.timestamp <= t ? ready : waiting).push_back(obs);
    }
    _pending.swap(waiting);
    return ready;
}

void UpdaterMapLandmark::print_observation(std::shared_ptr<State> state, const LandmarkObservation &obs) {
    double dt_ms = std::numeric_limits<double>::infinity();
    for (const auto &clone : state->_clones_IMU) {
        double dt = 1000.0 * (obs.timestamp - clone.first);
        if (std::abs(dt) < std::abs(dt_ms)) {
            dt_ms = dt;
        }
    }
    bool in_window = !state->_clones_IMU.empty() && obs.timestamp >= state->_clones_IMU.begin()->first;

    std::stringstream ids;
    for (const auto &p : obs.points) {
        ids << " " << p.id;
    }
    PRINT_DEBUG("[MAP]: obs t = %.3f | %zu pts | nearest clone dt = %.1f ms%s | ids:%s\n", obs.timestamp, obs.points.size(), dt_ms,
                in_window ? "" : " (older than window)", ids.str().c_str());
}

bool UpdaterMapLandmark::try_initialize(std::shared_ptr<State> state) {
    if (state->_T_MtoG != nullptr) {
        return true;
    }

    // ^G T_I from the current IMU state (q_GtoI, p_IinG)
    Eigen::Matrix4d T_G_I = Eigen::Matrix4d::Identity();
    T_G_I.block(0, 0, 3, 3) = state->_imu->Rot().transpose();
    T_G_I.block(0, 3, 3, 1) = state->_imu->pos();

    // ^M T_G = ^M T_B * ^B T_I * (^G T_I)^-1
    Eigen::Matrix4d T_M_I = _opts.T_M_B_init * Inv_se3(_opts.T_I_B);
    Eigen::Matrix4d T_M_G = T_M_I * Inv_se3(T_G_I);

    // PoseJPL (q_MtoG, p_GinM): R_MtoG maps vectors from M into G
    Eigen::Matrix<double, 7, 1> x;
    x.block(0, 0, 4, 1) = rot_2_quat(T_M_G.block<3, 3>(0, 0).transpose());
    x.block(4, 0, 3, 1) = T_M_G.block(0, 3, 3, 1);
    auto T_MtoG = std::make_shared<PoseJPL>();
    T_MtoG->set_value(x);
    T_MtoG->set_fej(x);

    // Prior is independent of the VIO state. initialize_invertible() needs isotropic noise, so whiten:
    // H_L = diag(1/sigma0), R = I  ->  P_LL = diag(sigma0^2), and H_R = 0 gives zero cross-covariance.
    Eigen::Matrix<double, 6, 1> sigma0 = _opts.sigma0_vec();
    Eigen::MatrixXd H_L = sigma0.cwiseInverse().asDiagonal();
    Eigen::MatrixXd H_R = Eigen::MatrixXd::Zero(6, 6);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(6, 6);
    Eigen::VectorXd res = Eigen::VectorXd::Zero(6);
    std::vector<std::shared_ptr<Type>> H_order = {state->_imu->pose()};
    StateHelper::initialize_invertible(state, T_MtoG, H_order, H_R, H_L, R, res);

    state->_T_MtoG = T_MtoG;
    _last_prop_time = state->_timestamp;

    std::stringstream ss;
    ss << "[MAP]: T_MtoG initialized from map_initial_T_M_B at t = " << std::fixed << state->_timestamp << std::endl;
    ss << "^M T_G =" << std::endl << T_M_G << std::endl;
    PRINT_INFO(GREEN "%s" RESET, ss.str().c_str());
    return true;
}

void UpdaterMapLandmark::propagate(std::shared_ptr<State> state) {
    double dt = state->_timestamp - _last_prop_time;
    if (dt <= 0.0) {
        return;
    }
    _last_prop_time = state->_timestamp;

    Eigen::Matrix<double, 6, 1> psd = _opts.rw_psd_vec();
    if (psd.isZero()) {
        return;
    }
    std::vector<std::shared_ptr<Type>> order = {state->_T_MtoG};
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(6, 6);
    Eigen::MatrixXd Q = (psd * dt).asDiagonal();
    StateHelper::EKFPropagation(state, order, order, Phi, Q);
}
