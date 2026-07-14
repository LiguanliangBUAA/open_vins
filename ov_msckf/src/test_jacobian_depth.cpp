#include <Eigen/Eigen>
#include <iostream>
#include <memory>
#include <random>

#include "state/State.h"
#include "state/StateOptions.h"
#include "update/UpdaterHelper.h"
#include "types/PoseJPL.h"
#include "utils/quat_ops.h"

#include "cam/CamRadtan.h"
#include "types/Vec.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

// ---------- helpers ----------

// Build a state with N clones at given timestamps, identity-ish calib, one camera
std::shared_ptr<State> build_state(const std::vector<double> &times, int max_cams = 1) {
  StateOptions opts;
  opts.num_cameras = max_cams;
  opts.max_clone_size = (int)times.size() + 2;
  opts.do_fej = false;              // IMPORTANT: linearization point == estimate for FD check
  opts.do_calib_camera_pose = false;
  opts.do_calib_camera_intrinsics = false;
  opts.feat_rep_msckf = LandmarkRepresentation::Representation::GLOBAL_3D;
  auto state = std::make_shared<State>(opts);

  // Camera intrinsics: simple pinhole, zero distortion (radtan with zeros)
  Eigen::MatrixXd cam_calib = Eigen::MatrixXd::Zero(8, 1);
  cam_calib << 458.0, 457.0, 367.0, 248.0, 0, 0, 0, 0;
  auto camera = std::make_shared<ov_core::CamRadtan>(752, 480);
  camera->set_value(cam_calib);
  state->_cam_intrinsics_cameras.insert({0, camera});
  auto intr = std::make_shared<Vec>(8);
  intr->set_value(cam_calib);
  state->_cam_intrinsics.insert({0, intr});

  // Extrinsic IMU->CAM: small rotation + offset (non-trivial to catch frame errors)
  Eigen::Matrix<double, 7, 1> ext;
  Eigen::Vector3d rotvec(0.02, -0.01, 0.03);
  Eigen::Matrix3d R_ItoC = ov_core::exp_so3(rotvec);
  ext.block(0, 0, 4, 1) = ov_core::rot_2_quat(R_ItoC);
  ext.block(4, 0, 3, 1) << 0.05, -0.02, 0.01;
  auto calib = std::make_shared<PoseJPL>();
  calib->set_value(ext);
  calib->set_fej(ext);
  state->_calib_IMUtoCAM.insert({0, calib});

  // Clones: a small trajectory with rotation + translation
  std::mt19937 gen(42);
  std::normal_distribution<double> nd(0.0, 0.1);
  for (size_t i = 0; i < times.size(); i++) {
    Eigen::Matrix<double, 7, 1> pose;
    Eigen::Vector3d rv(0.05 * i + nd(gen) * 0.01, -0.03 * i, 0.02 * i);
    pose.block(0, 0, 4, 1) = ov_core::rot_2_quat(ov_core::exp_so3(rv));
    pose.block(4, 0, 3, 1) << 0.3 * i, 0.1 * i + nd(gen) * 0.01, 0.05 * i;
    auto clone = std::make_shared<PoseJPL>();
    clone->set_value(pose);
    clone->set_fej(pose);
    state->_clones_IMU.insert({times[i], clone});
  }
  return state;
}

// Compute residual only (re-calls the full function, discards Jacobians)
Eigen::VectorXd compute_res(std::shared_ptr<State> state, UpdaterHelper::UpdaterHelperFeature feat,
                            double sigma_pix, bool use_depth) {
  Eigen::MatrixXd H_f, H_x;
  Eigen::VectorXd res;
  std::vector<std::shared_ptr<Type>> order;
  UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, order, sigma_pix, use_depth);
  return res;
}

int main() {
  const double eps = 1e-5;
  const double tol = 1e-4;   // FD truncation ~eps^2 rel; whitened units make 1e-4 a safe gate
  const double sigma_pix = 1.5;   // deliberately != 1 to catch sigma/sigma^2 bugs!
  bool use_depth = true;

  // ---------- build scene ----------
  std::vector<double> times = {0.1, 0.2, 0.3, 0.4};
  auto state = build_state(times);

  // True feature position
  Eigen::Vector3d p_FinG(1.2, -0.4, 3.5);

  // Build feature with synthetic measurements = perfect projection + tiny noise
  UpdaterHelper::UpdaterHelperFeature feat;
  feat.featid = 1;
  feat.feat_representation = LandmarkRepresentation::Representation::GLOBAL_3D;
  feat.p_FinG = p_FinG + Eigen::Vector3d(0.01, -0.02, 0.03); // linearization point != truth
  feat.p_FinG_fej = feat.p_FinG;

  std::mt19937 gen(7);
  std::normal_distribution<double> px_noise(0.0, 0.5);
  auto calib = state->_calib_IMUtoCAM.at(0);
  for (double t : times) {
    auto clone = state->_clones_IMU.at(t);
    Eigen::Vector3d p_FinI = clone->Rot() * (p_FinG - clone->pos());
    Eigen::Vector3d p_FinC = calib->Rot() * p_FinI + calib->pos();
    Eigen::Vector2d uvn(p_FinC(0) / p_FinC(2), p_FinC(1) / p_FinC(2));
    Eigen::Vector2d uvd = state->_cam_intrinsics_cameras.at(0)->distort_d(uvn);
    Eigen::VectorXf uv(2), uvn_f(2);
    uv << (float)(uvd(0) + px_noise(gen)), (float)(uvd(1) + px_noise(gen));
    uvn_f << (float)uvn(0), (float)uvn(1);
    feat.uvs[0].push_back(uv);
    feat.uvs_norm[0].push_back(uvn_f);
    feat.timestamps[0].push_back(t);
    // depth on observations 0 and 2 only (mixed 2/3-row layout on purpose)
    size_t m = feat.timestamps[0].size() - 1;
    if (m == 0 || m == 2) {
      feat.depths[0].push_back((float)(p_FinC(2) + 0.02));
      float sz = (float)(0.01 * p_FinC(2) * p_FinC(2));   // sigma_z ~ z^2 model
      feat.depth_vars[0].push_back(sz * sz);
    } else {
      feat.depths[0].push_back(-1.0f);
      feat.depth_vars[0].push_back(-1.0f);
    }
  }

  // ---------- analytic ----------
  Eigen::MatrixXd H_f, H_x;
  Eigen::VectorXd res0;
  std::vector<std::shared_ptr<Type>> order;
  UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res0, order, sigma_pix, use_depth);
  std::cout << "rows = " << res0.rows() << " (expect 2*4 + 2 depth = 10)\n";

  int failures = 0;
  auto check = [&](const std::string &name, int col, const Eigen::VectorXd &Hcol_analytic,
                   const Eigen::VectorXd &res_p, const Eigen::VectorXd &res_m) {
    Eigen::VectorXd Hcol_num = -(res_p - res_m) / (2 * eps);
    double err = (Hcol_num - Hcol_analytic).norm() / std::max(1.0, Hcol_analytic.norm());
    bool ok = err < tol;
    if (!ok) {
      failures++;
      Eigen::VectorXd diff = (Hcol_num - Hcol_analytic).cwiseAbs();
      std::cout << name << " col " << col << " : rel_err = " << err << "  <-- FAIL\n"
                << "    per-row |diff|: " << diff.transpose() << "\n"
                << "    analytic:       " << Hcol_analytic.transpose() << "\n";
    } else {
      printf("%-28s col %d : rel_err = %.3e  OK\n", name.c_str(), col, err);
    }
  };

  // ---- H_f: perturb p_FinG ----
  for (int j = 0; j < 3; j++) {
    auto fp = feat; fp.p_FinG(j) += eps; fp.p_FinG_fej = fp.p_FinG;
    auto fm = feat; fm.p_FinG(j) -= eps; fm.p_FinG_fej = fm.p_FinG;
    check("H_f (p_FinG)", j, H_f.col(j),
          compute_res(state, fp, sigma_pix, use_depth),
          compute_res(state, fm, sigma_pix, use_depth));
  }

  // ---- H_x: perturb each clone (JPL: theta uses left-mult exp on quat; pos additive) ----
  // locate column offset of each variable in H_x via `order`
  size_t col0 = 0;
  for (auto &var : order) {
    auto pose = std::dynamic_pointer_cast<PoseJPL>(var);
    // orientation dofs (JPL error: q_pert = quat_multiply(rot_2_quat(exp_so3(dtheta)), q))
    for (int j = 0; j < 3; j++) {
      Eigen::VectorXd dtp = Eigen::VectorXd::Zero(3); dtp(j) = eps;
      Eigen::Matrix<double, 7, 1> vp = pose->value(), vm = pose->value();
      vp.block(0,0,4,1) = ov_core::quat_multiply(ov_core::rot_2_quat(ov_core::exp_so3(-dtp)), pose->quat());
      vm.block(0,0,4,1) = ov_core::quat_multiply(ov_core::rot_2_quat(ov_core::exp_so3( dtp)), pose->quat());
      Eigen::Matrix<double,7,1> orig = pose->value();
      pose->set_value(vp); pose->set_fej(vp);
      Eigen::VectorXd rp = compute_res(state, feat, sigma_pix, use_depth);
      pose->set_value(vm); pose->set_fej(vm);
      Eigen::VectorXd rm = compute_res(state, feat, sigma_pix, use_depth);
      pose->set_value(orig); pose->set_fej(orig);
      check("H_x theta", (int)(col0 + j), H_x.col(col0 + j), rp, rm);
    }
    // position dofs
    for (int j = 0; j < 3; j++) {
      Eigen::Matrix<double,7,1> orig = pose->value();
      Eigen::Matrix<double,7,1> vp = orig, vm = orig;
      vp(4 + j) += eps; vm(4 + j) -= eps;
      pose->set_value(vp); pose->set_fej(vp);
      Eigen::VectorXd rp = compute_res(state, feat, sigma_pix, use_depth);
      pose->set_value(vm); pose->set_fej(vm);
      Eigen::VectorXd rm = compute_res(state, feat, sigma_pix, use_depth);
      pose->set_value(orig); pose->set_fej(orig);
      check("H_x pos", (int)(col0 + 3 + j), H_x.col(col0 + 3 + j), rp, rm);
    }
    col0 += var->size();
  }

  printf("\n========== %s (%d failures) ==========\n", failures == 0 ? "ALL PASS" : "FAILED", failures);
  return failures == 0 ? 0 : 1;
}