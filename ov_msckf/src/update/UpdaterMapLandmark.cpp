#include "UpdaterMapLandmark.h"

#include "state/State.h"
#include "state/StateHelper.h"
#include "types/PoseJPL.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/math/distributions/chi_squared.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
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
        ClonePair cp;
        if (!find_clones(state, obs.timestamp, cp)) {
            _n_no_clone++;
            continue;
        }

        Eigen::VectorXd uv;
        Eigen::MatrixXd H;
        if (!predict(state, cp, obs, uv, &H)) {
            continue;
        }
        if (cp.c1 != nullptr) {
            _n_interpolated++;
        }
        Eigen::VectorXd res(uv.rows());
        for (size_t i = 0; i < obs.points.size(); i++) {
            res(2 * i) = obs.points.at(i).u - uv(2 * i);
            res(2 * i + 1) = obs.points.at(i).v - uv(2 * i + 1);
        }
        if (_opts.map_check_jacobian && _n_jacobian_checks < 6) {
            check_jacobian(state, cp, obs, H);
            _n_jacobian_checks++;
        }

        // Pixel noise plus the landmark position uncertainty projected into the image (dp_I/dp_M = -dp_I/dp_GinM)
        int n2 = (int)res.rows();
        Eigen::MatrixXd R = std::pow(_opts.map_sigma_px, 2) * Eigen::MatrixXd::Identity(n2, n2);
        for (int i = 0; i < n2 / 2; i++) {
            Eigen::Matrix<double, 2, 3> J_pM = -H.block<2, 3>(2 * i, H.cols() - 3);
            R.block<2, 2>(2 * i, 2 * i) += std::pow(_opts.map_sigma_landmark_pos_m, 2) * J_pM * J_pM.transpose();
        }

        // Chi2 gate on the whole frame
        std::vector<std::shared_ptr<Type>> H_order = h_order(state, cp);
        Eigen::MatrixXd P = StateHelper::get_marginal_covariance(state, H_order);
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        double chi2 = res.dot(S.llt().solve(res));
        boost::math::chi_squared chi2_dist(n2);
        double chi2_check = _opts.map_chi2_multiplier * boost::math::quantile(chi2_dist, 0.95);
        bool accepted = chi2 < chi2_check;
        if (accepted) {
            _n_accepted++;
            _n_consecutive_rejected = 0;
        } else {
            _n_rejected++;
            _n_consecutive_rejected++;
        }
        print_residual(obs, cp, res, chi2, chi2_check, accepted);
        if (_n_consecutive_rejected > 0 && _n_consecutive_rejected % 10 == 0) {
            PRINT_WARNING(YELLOW "[MAP]: %d frames rejected in a row, check map_initial_T_M_B and the landmark map\n" RESET,
                          _n_consecutive_rejected);
        }

        if (accepted && !_opts.map_update_dry_run) {
            StateHelper::EKFUpdate(state, H_order, H, res, R);
        }
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

bool UpdaterMapLandmark::find_clones(std::shared_ptr<State> state, double t, ClonePair &cp) const {
    const auto &clones = state->_clones_IMU;
    if (clones.empty() || t < clones.begin()->first - _opts.map_max_clone_dt || t > clones.rbegin()->first + _opts.map_max_clone_dt) {
        return false;
    }
    for (auto it = clones.begin(); it != clones.end(); it++) {
        if (std::abs(t - it->first) <= _opts.map_max_clone_dt) {
            cp = {it->second, nullptr, 0.0};
            return true;
        }
        auto next = std::next(it);
        if (next != clones.end() && it->first < t && t < next->first) {
            cp = {it->second, next->second, (t - it->first) / (next->first - it->first)};
            return true;
        }
    }
    return false;
}

std::vector<std::shared_ptr<Type>> UpdaterMapLandmark::h_order(std::shared_ptr<State> state, const ClonePair &cp) const {
    if (cp.c1 == nullptr) {
        return {cp.c0, state->_T_MtoG};
    }
    return {cp.c0, cp.c1, state->_T_MtoG};
}

bool UpdaterMapLandmark::predict(std::shared_ptr<State> state, const ClonePair &cp, const LandmarkObservation &obs, Eigen::VectorXd &uv,
                                 Eigen::MatrixXd *H) const {
    // R = Exp(lambda * Log(R1 R0^T)) R0, matching the JPL left error; position linear
    Eigen::Matrix3d R_GtoI = cp.c0->Rot();
    Eigen::Vector3d p_IinG = cp.c0->pos();
    if (cp.c1 != nullptr) {
        R_GtoI = exp_so3(cp.lambda * log_so3(cp.c1->Rot() * cp.c0->Rot().transpose())) * cp.c0->Rot();
        p_IinG = (1.0 - cp.lambda) * cp.c0->pos() + cp.lambda * cp.c1->pos();
    }
    Eigen::Matrix3d R_MtoG = state->_T_MtoG->Rot();
    Eigen::Vector3d p_GinM = state->_T_MtoG->pos();
    Eigen::Matrix3d R_ItoC = _opts.aruco_T_C_I.block<3, 3>(0, 0);
    Eigen::Vector3d p_IinC = _opts.aruco_T_C_I.block<3, 1>(0, 3);

    size_t n = obs.points.size();
    uv.resize(2 * n);
    int col_T = (cp.c1 == nullptr) ? 6 : 12;
    if (H != nullptr) {
        H->setZero(2 * n, col_T + 6);
    }
    for (size_t i = 0; i < n; i++) {
        Eigen::Vector3d p_M = _map.at(obs.points.at(i).id);
        Eigen::Vector3d p_G = R_MtoG * (p_M - p_GinM);
        Eigen::Vector3d p_I = R_GtoI * (p_G - p_IinG);
        Eigen::Vector3d p_C = R_ItoC * p_I + p_IinC;
        if (p_C(2) < 0.1) {
            return false;
        }
        Eigen::Vector2d uv_norm(p_C(0) / p_C(2), p_C(1) / p_C(2));
        uv.segment<2>(2 * i) = _opts.aruco_camera->distort_d(uv_norm);

        if (H != nullptr) {
            Eigen::MatrixXd H_dz_dzn, H_dz_dzeta;
            _opts.aruco_camera->compute_distort_jacobian(uv_norm, H_dz_dzn, H_dz_dzeta);
            Eigen::Matrix<double, 2, 3> dzn_dpc;
            dzn_dpc << 1 / p_C(2), 0, -p_C(0) / (p_C(2) * p_C(2)), 0, 1 / p_C(2), -p_C(1) / (p_C(2) * p_C(2));
            Eigen::Matrix<double, 2, 3> J = H_dz_dzn * dzn_dpc * R_ItoC;

            // JPL left error R = (I - [dtheta]x) R_hat, order [theta, p]
            Eigen::Matrix<double, 2, 6> H_pose;
            H_pose << J * skew_x(p_I), -J * R_GtoI;
            if (cp.c1 == nullptr) {
                H->block<2, 6>(2 * i, 0) = H_pose;
            } else {
                // Exact for the position, first order in the relative rotation of the two clones for the attitude
                H->block<2, 6>(2 * i, 0) = (1.0 - cp.lambda) * H_pose;
                H->block<2, 6>(2 * i, 6) = cp.lambda * H_pose;
            }
            H->block<2, 3>(2 * i, col_T) = J * R_GtoI * skew_x(p_G);
            H->block<2, 3>(2 * i, col_T + 3) = -J * R_GtoI * R_MtoG;
        }
    }
    return true;
}

void UpdaterMapLandmark::print_residual(const LandmarkObservation &obs, const ClonePair &cp, const Eigen::VectorXd &res, double chi2,
                                        double chi2_check, bool accepted) const {
    std::vector<double> norms;
    for (int i = 0; i < res.rows() / 2; i++) {
        norms.push_back(res.segment<2>(2 * i).norm());
    }
    std::sort(norms.begin(), norms.end());
    PRINT_INFO("[MAP]: t = %.3f | %zu pts | lambda %.2f | |r| median %.1f px, max %.1f px | chi2 %.1f / %.1f %s | accepted %d, rejected %d, "
               "interpolated %d, no clone %d\n",
               obs.timestamp, obs.points.size(), cp.lambda, norms.at(norms.size() / 2), norms.back(), chi2, chi2_check,
               accepted ? "ok" : "REJECT", _n_accepted, _n_rejected, _n_interpolated, _n_no_clone);
}

void UpdaterMapLandmark::check_jacobian(std::shared_ptr<State> state, const ClonePair &cp, const LandmarkObservation &obs,
                                        const Eigen::MatrixXd &H) const {
    const double eps = 1e-3; // distort_d() works in float, a smaller step is dominated by rounding
    Eigen::MatrixXd H_num = Eigen::MatrixXd::Zero(H.rows(), H.cols());
    std::vector<std::shared_ptr<PoseJPL>> poses = {cp.c0, state->_T_MtoG};
    if (cp.c1 != nullptr) {
        poses.insert(poses.begin() + 1, cp.c1);
    }
    for (size_t k = 0; k < poses.size(); k++) {
        Eigen::MatrixXd x0 = poses.at(k)->value();
        for (int j = 0; j < 6; j++) {
            Eigen::VectorXd dx = Eigen::VectorXd::Zero(6);
            Eigen::VectorXd uv_plus, uv_minus;
            dx(j) = eps;
            poses.at(k)->update(dx);
            bool ok_plus = predict(state, cp, obs, uv_plus, nullptr);
            poses.at(k)->set_value(x0);
            poses.at(k)->update(-dx);
            bool ok_minus = predict(state, cp, obs, uv_minus, nullptr);
            poses.at(k)->set_value(x0);
            if (!ok_plus || !ok_minus) {
                return;
            }
            H_num.col(6 * k + j) = (uv_plus - uv_minus) / (2 * eps);
        }
    }
    double err = (H - H_num).cwiseAbs().maxCoeff();
    double scale = H.cwiseAbs().maxCoeff();
    PRINT_INFO("[MAP]: Jacobian check t = %.3f | lambda %.2f | max |H - H_num| = %.3e | max |H| = %.3e | relative %.3e\n", obs.timestamp,
               cp.lambda, err, scale, err / scale);
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
