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
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterMapLandmark::UpdaterMapLandmark(UpdaterMapLandmarkOptions opts, LandmarkMap landmark_map, std::vector<LandmarkObject> objects)
    : _opts(std::move(opts)), _map(std::move(landmark_map)), _objects(std::move(objects)) {
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

        std::map<std::string, int> per_type;
        for (const auto &obj : _objects) {
            for (LandmarkId id : obj.ids) {
                if (_map.count(id) == 0) {
                    throw std::invalid_argument("object " + obj.name + " uses unknown landmark id " + std::to_string(id));
                }
            }
            per_type[obj.type]++;
        }
        for (const auto &entry : per_type) {
            PRINT_INFO(GREEN "[MAP]: object type %s: %d objects\n" RESET, entry.first.c_str(), entry.second);
        }
    }

void UpdaterMapLandmark::feed_observation(const LandmarkObservation &obs) {
    LandmarkObservation known;
    known.timestamp = obs.timestamp;
    for (const auto &p : obs.points) {
        if (_map.count(p.id) > 0) {
            known.points.push_back(p);
        }
    }
    for (const auto &det : obs.objects) {
        for (const auto &obj : _objects) {
            if (obj.type == det.type) {
                known.objects.push_back(det);
                break;
            }
        }
    }
    if (known.points.empty() && known.objects.empty()) {
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

    for (const auto &obs_in : ready) {
        ClonePair cp;
        if (!find_clones(state, obs_in.timestamp, cp)) {
            _n_no_clone++;
            continue;
        }

        // Objects without identity become landmark points once matched to the map
        LandmarkObservation obs = obs_in;
        int n_matched = associate(state, cp, obs_in.objects, obs.points);
        _n_objects_matched += n_matched;
        _n_objects_unmatched += (int)obs_in.objects.size() - n_matched;
        if (!obs_in.objects.empty()) {
            PRINT_DEBUG("[MAP]: t = %.3f | objects matched %d / %zu (total %d matched, %d unmatched)\n", obs_in.timestamp, n_matched,
                        obs_in.objects.size(), _n_objects_matched, _n_objects_unmatched);
        }
        if (obs.points.empty()) {
            continue;
        }

        Eigen::VectorXd uv;
        Eigen::MatrixXd H;
        std::vector<bool> in_front;
        predict(state, cp, obs, uv, &H, &in_front);
        if (cp.c1 != nullptr) {
            _n_interpolated++;
        }
        int n = (int)obs.points.size();
        int n_front = (int)std::count(in_front.begin(), in_front.end(), true);
        _n_points_behind += n - n_front;

        // No point visible: initialization, calibration or frame convention is wrong
        if (n_front == 0) {
            _n_consecutive_all_behind++;
            PRINT_WARNING(YELLOW "[MAP]: t = %.3f all %d points behind the camera or far outside the image (%d/%d)\n" RESET, obs.timestamp, n,
                          _n_consecutive_all_behind, _opts.map_all_behind_limit);
            if (_opts.map_fatal_on_all_behind && _n_consecutive_all_behind >= _opts.map_all_behind_limit) {
                PRINT_ERROR(RED "[MAP]: landmarks predicted behind the camera or far outside the image %d frames in a row, check map_initial_T_M_B, "
                                "the camera extrinsics and the landmark map\n" RESET,
                            _n_consecutive_all_behind);
                // std::exit() would run destructors and unload libraries under the running ROS executor threads (segfault)
                std::fflush(stdout);
                std::fflush(stderr);
                std::_Exit(EXIT_FAILURE);
            }
            continue;
        }
        _n_consecutive_all_behind = 0;

        if (_opts.map_check_jacobian && _n_jacobian_checks < 6 && n_front == n) {
            check_jacobian(state, cp, obs, H);
            _n_jacobian_checks++;
        }

        // Per-point chi2 gate, as map_alignment_ekf: a bad corner is dropped, the rest of the frame is kept
        std::vector<std::shared_ptr<Type>> H_order = h_order(state, cp);
        Eigen::MatrixXd P = StateHelper::get_marginal_covariance(state, H_order);
        std::vector<int> kept;
        std::vector<Eigen::Matrix2d> R_kept;
        std::vector<double> r_norms;
        double max_chi2 = 0.0;
        for (int i = 0; i < n; i++) {
            if (!in_front.at(i)) {
                continue;
            }
            Eigen::Vector2d r_i(obs.points.at(i).u - uv(2 * i), obs.points.at(i).v - uv(2 * i + 1));
            Eigen::MatrixXd H_i = H.middleRows(2 * i, 2);
            // Pixel noise plus the landmark position uncertainty projected into the image (dp_I/dp_M = -dp_I/dp_GinM)
            Eigen::Matrix<double, 2, 3> J_pM = -H_i.rightCols(3);
            Eigen::Matrix2d R_i = std::pow(_opts.map_sigma_px, 2) * Eigen::Matrix2d::Identity() +
                                  std::pow(_opts.map_sigma_landmark_pos_m, 2) * J_pM * J_pM.transpose();
            Eigen::Matrix2d S_i = H_i * P * H_i.transpose() + R_i;
            double chi2 = r_i.dot(S_i.llt().solve(r_i));
            r_norms.push_back(r_i.norm());
            max_chi2 = std::max(max_chi2, chi2);
            if (chi2 < _opts.map_chi2_2d_threshold) {
                kept.push_back(i);
                R_kept.push_back(R_i);
            } else {
                _n_points_rejected++;
            }
        }
        std::sort(r_norms.begin(), r_norms.end());
        print_frame(obs, cp, n_front, (int)kept.size(), r_norms.at(r_norms.size() / 2), max_chi2);

        // Every visible point rejected: after map_full_reject_limit such frames, inflate the T_MtoG covariance (capped)
        if (kept.empty()) {
            _n_rejected++;
            _n_consecutive_rejected++;
            PRINT_WARNING(YELLOW "[MAP]: all %d visible points rejected by chi2 (%d/%d)\n" RESET, n_front, _n_consecutive_rejected,
                          _opts.map_full_reject_limit);
            if (_opts.map_full_reject_limit > 0 && _n_consecutive_rejected >= _opts.map_full_reject_limit) {
                _n_consecutive_rejected = 0;
                if (_n_consecutive_inflations < _opts.map_max_consecutive_cov_inflations) {
                    inflate_T_MtoG(state, _opts.map_reject_cov_inflation);
                    _n_consecutive_inflations++;
                    _n_cov_inflations++;
                    PRINT_WARNING(YELLOW "[MAP]: reject lock, T_MtoG covariance inflated by %.1fx (%d/%d in a row)\n" RESET,
                                  _opts.map_reject_cov_inflation, _n_consecutive_inflations, _opts.map_max_consecutive_cov_inflations);
                } else {
                    PRINT_WARNING(YELLOW "[MAP]: reject lock persists, inflation cap of %d reached, check map_initial_T_M_B\n" RESET,
                                  _opts.map_max_consecutive_cov_inflations);
                }
            }
            continue;
        }
        _n_accepted++;
        _n_consecutive_rejected = 0;
        _n_consecutive_inflations = 0;

        if (_opts.map_update_dry_run) {
            continue;
        }
        int m = (int)kept.size();
        Eigen::MatrixXd H_up(2 * m, H.cols());
        Eigen::VectorXd r_up(2 * m);
        Eigen::MatrixXd R_up = Eigen::MatrixXd::Zero(2 * m, 2 * m);
        for (int k = 0; k < m; k++) {
            int i = kept.at(k);
            H_up.middleRows(2 * k, 2) = H.middleRows(2 * i, 2);
            r_up(2 * k) = obs.points.at(i).u - uv(2 * i);
            r_up(2 * k + 1) = obs.points.at(i).v - uv(2 * i + 1);
            R_up.block<2, 2>(2 * k, 2 * k) = R_kept.at(k);
        }
        StateHelper::EKFUpdate(state, H_order, H_up, r_up, R_up);
    }
}

void UpdaterMapLandmark::inflate_T_MtoG(std::shared_ptr<State> state, double factor) const {
    // P <- D P D with D = sqrt(factor) on T_MtoG: scales its block by factor and its cross terms by sqrt(factor), stays PSD
    std::vector<std::shared_ptr<Type>> order = {state->_T_MtoG};
    Eigen::MatrixXd Phi = std::sqrt(factor) * Eigen::MatrixXd::Identity(6, 6);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6, 6);
    StateHelper::EKFPropagation(state, order, order, Phi, Q);
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

int UpdaterMapLandmark::associate(std::shared_ptr<State> state, const ClonePair &cp, const std::vector<ObjectDetection> &dets,
                                  std::vector<LandmarkPoint> &points) const {
    if (dets.empty()) {
        return 0;
    }

    // Project every map object of a detected type, with the std of its predicted pixels (grows while no landmark is seen)
    struct Candidate {
        const LandmarkObject *obj;
        std::vector<Eigen::Vector2d> uv;
        std::vector<bool> in_front;
        double sigma_px = 0.0;
    };
    Eigen::MatrixXd P = StateHelper::get_marginal_covariance(state, h_order(state, cp));
    std::vector<Candidate> cands;
    for (const auto &obj : _objects) {
        bool wanted = std::any_of(dets.begin(), dets.end(), [&](const ObjectDetection &d) { return d.type == obj.type; });
        if (!wanted) {
            continue;
        }
        LandmarkObservation o;
        for (LandmarkId id : obj.ids) {
            o.points.push_back({id, 0.0, 0.0});
        }
        Eigen::VectorXd uv;
        Eigen::MatrixXd H;
        Candidate c{&obj, {}, {}};
        predict(state, cp, o, uv, &H, &c.in_front);
        std::vector<Eigen::Vector2d> front;
        for (size_t i = 0; i < obj.ids.size(); i++) {
            c.uv.emplace_back(uv(2 * i), uv(2 * i + 1));
            if (c.in_front.at(i)) {
                front.push_back(c.uv.back());
                Eigen::MatrixXd H_i = H.middleRows(2 * i, 2);
                c.sigma_px += std::sqrt((H_i * P * H_i.transpose()).trace() / 2.0);
            }
        }
        if (!front.empty()) {
            c.sigma_px /= (double)front.size();
        }
        // A polygon (3+ points, TL TR BR BL) that winds the wrong way in the image is seen from behind
        double area = 0.0;
        for (size_t i = 0; i < front.size(); i++) {
            const Eigen::Vector2d &a = front.at(i), &b = front.at((i + 1) % front.size());
            area += 0.5 * (a(0) * b(1) - b(0) * a(1));
        }
        if (front.empty() || (front.size() >= 3 && area < 0.0)) {
            continue;
        }
        cands.push_back(c);
    }

    // Best candidate per detection: mean keypoint distance, a 2-point object (bar) is tried in both orders
    struct Match {
        double cost;
        int det, cand;
        std::vector<int> order;
    };
    std::vector<Match> matches;
    for (int d = 0; d < (int)dets.size(); d++) {
        const ObjectDetection &det = dets.at(d);
        std::vector<Match> costs;
        for (int c = 0; c < (int)cands.size(); c++) {
            const Candidate &cand = cands.at(c);
            if (cand.obj->type != det.type) {
                continue;
            }
            int n = (int)cand.obj->ids.size();
            std::vector<std::vector<int>> orders = {std::vector<int>(n)};
            std::iota(orders.at(0).begin(), orders.at(0).end(), 0);
            if (n == 2) {
                orders.push_back({1, 0});
            }
            for (const auto &order : orders) {
                double sum = 0.0;
                int cnt = 0;
                for (int k = 0; k < std::min(n, (int)det.keypoints.size()); k++) {
                    if (!cand.in_front.at(order.at(k)) || det.scores.at(k) < _opts.map_keypoint_conf) {
                        continue;
                    }
                    sum += (Eigen::Vector2d(det.keypoints.at(k).u, det.keypoints.at(k).v) - cand.uv.at(order.at(k))).norm();
                    cnt++;
                }
                if (cnt > 0) {
                    costs.push_back({sum / cnt, d, c, order});
                }
            }
        }
        std::sort(costs.begin(), costs.end(), [](const Match &a, const Match &b) { return a.cost < b.cost; });
        // Drop the second order of the same candidate before the ambiguity check
        auto second = std::find_if(costs.begin() + std::min<size_t>(1, costs.size()), costs.end(),
                                   [&](const Match &m) { return m.cand != costs.front().cand; });
        // Pixel gate as map_alignment_ekf, widened by 3 sigma of the predicted position so the match recovers after drifting
        double gate = _opts.map_match_px + _opts.map_match_scale * std::sqrt(std::max(0.0, det.box_area));
        if (!costs.empty()) {
            gate += 3.0 * cands.at(costs.front().cand).sigma_px;
        }
        if (costs.empty() || costs.front().cost > gate) {
            PRINT_DEBUG("[MAP]: %s not matched: %zu candidates, best %.1f px > gate %.1f px\n", det.type.c_str(), costs.size(),
                        costs.empty() ? -1.0 : costs.front().cost, gate);
            continue;
        }
        // Two alike objects side by side: a wrong id is worse than none
        if (second != costs.end() && second->cost < _opts.map_ambiguity_ratio * costs.front().cost) {
            PRINT_DEBUG("[MAP]: %s not matched: ambiguous, best %s %.1f px, second %s %.1f px\n", det.type.c_str(),
                        cands.at(costs.front().cand).obj->name.c_str(), costs.front().cost, cands.at(second->cand).obj->name.c_str(),
                        second->cost);
            continue;
        }
        matches.push_back(costs.front());
    }

    // One object per detection and one detection per object, cheapest first
    std::sort(matches.begin(), matches.end(), [](const Match &a, const Match &b) { return a.cost < b.cost; });
    std::vector<bool> taken(cands.size(), false);
    int n_matched = 0;
    for (const auto &m : matches) {
        if (taken.at(m.cand)) {
            continue;
        }
        taken.at(m.cand) = true;
        n_matched++;
        const ObjectDetection &det = dets.at(m.det);
        const Candidate &cand = cands.at(m.cand);
        // A pose error shifts all keypoints alike, so a single bad keypoint is one that deviates from the mean shift
        std::vector<int> valid;
        Eigen::Vector2d mean_err = Eigen::Vector2d::Zero();
        for (int k = 0; k < (int)m.order.size() && k < (int)det.keypoints.size(); k++) {
            if (cand.in_front.at(m.order.at(k)) && det.scores.at(k) >= _opts.map_keypoint_conf) {
                valid.push_back(k);
                mean_err += Eigen::Vector2d(det.keypoints.at(k).u, det.keypoints.at(k).v) - cand.uv.at(m.order.at(k));
            }
        }
        if (valid.empty()) {
            continue;
        }
        mean_err /= (double)valid.size();
        for (int k : valid) {
            int idx = m.order.at(k);
            Eigen::Vector2d z(det.keypoints.at(k).u, det.keypoints.at(k).v);
            if ((z - cand.uv.at(idx) - mean_err).norm() > _opts.map_keypoint_match_px) {
                continue;
            }
            points.push_back({cand.obj->ids.at(idx), z(0), z(1)});
        }
    }
    return n_matched;
}

bool UpdaterMapLandmark::predict(std::shared_ptr<State> state, const ClonePair &cp, const LandmarkObservation &obs, Eigen::VectorXd &uv,
                                 Eigen::MatrixXd *H, std::vector<bool> *in_front) const {
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
    uv.setZero(2 * n);
    int col_T = (cp.c1 == nullptr) ? 6 : 12;
    if (H != nullptr) {
        H->setZero(2 * n, col_T + 6);
    }
    if (in_front != nullptr) {
        in_front->assign(n, true);
    }
    for (size_t i = 0; i < n; i++) {
        Eigen::Vector3d p_M = _map.at(obs.points.at(i).id);
        Eigen::Vector3d p_G = R_MtoG * (p_M - p_GinM);
        Eigen::Vector3d p_I = R_GtoI * (p_G - p_IinG);
        Eigen::Vector3d p_C = R_ItoC * p_I + p_IinC;
        // Not visible: behind the camera, or predicted further than half the image size outside it (the linearization is meaningless)
        bool visible = p_C(2) >= 0.1;
        if (visible) {
            Eigen::Vector2d uv_norm(p_C(0) / p_C(2), p_C(1) / p_C(2));
            uv.segment<2>(2 * i) = _opts.aruco_camera->distort_d(uv_norm);
            double w = _opts.aruco_camera->w(), h = _opts.aruco_camera->h();
            visible = uv(2 * i) > -0.5 * w && uv(2 * i) < 1.5 * w && uv(2 * i + 1) > -0.5 * h && uv(2 * i + 1) < 1.5 * h;
        }
        if (!visible) {
            uv.segment<2>(2 * i).setZero();
            if (in_front == nullptr) {
                return false;
            }
            in_front->at(i) = false;
            continue;
        }
        Eigen::Vector2d uv_norm(p_C(0) / p_C(2), p_C(1) / p_C(2));

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

void UpdaterMapLandmark::print_frame(const LandmarkObservation &obs, const ClonePair &cp, int n_front, int n_kept, double median_r,
                                     double max_chi2) const {
    PRINT_INFO("[MAP]: t = %.3f | lambda %.2f | pts %zu, in front %d, kept %d | |r| median %.1f px | max chi2 %.1f / %.1f | frames: "
               "accepted %d, rejected %d | points: rejected %d, behind %d | inflations %d | no clone %d\n",
               obs.timestamp, cp.lambda, obs.points.size(), n_front, n_kept, median_r, max_chi2, _opts.map_chi2_2d_threshold, _n_accepted,
               _n_rejected, _n_points_rejected, _n_points_behind, _n_cov_inflations, _n_no_clone);
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
