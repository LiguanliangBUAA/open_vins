/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "UpdaterHelper.h"

#include "state/State.h"

#include "utils/quat_ops.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

void UpdaterHelper::get_feature_jacobian_representation(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                        std::vector<Eigen::MatrixXd> &H_x, std::vector<std::shared_ptr<Type>> &x_order) {

  // Global XYZ representation
  if (feature.feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D) {
    H_f.resize(3, 3);
    H_f.setIdentity();
    return;
  }

  // Global inverse depth representation
  if (feature.feat_representation == LandmarkRepresentation::Representation::GLOBAL_FULL_INVERSE_DEPTH) {

    // Get the feature linearization point
    Eigen::Matrix<double, 3, 1> p_FinG = (state->_options.do_fej) ? feature.p_FinG_fej : feature.p_FinG;

    // Get inverse depth representation (should match what is in Landmark.cpp)
    double g_rho = 1 / p_FinG.norm();
    double g_phi = std::acos(g_rho * p_FinG(2));
    // double g_theta = std::asin(g_rho*p_FinG(1)/std::sin(g_phi));
    double g_theta = std::atan2(p_FinG(1), p_FinG(0));
    Eigen::Matrix<double, 3, 1> p_invFinG;
    p_invFinG(0) = g_theta;
    p_invFinG(1) = g_phi;
    p_invFinG(2) = g_rho;

    // Get inverse depth bearings
    double sin_th = std::sin(p_invFinG(0, 0));
    double cos_th = std::cos(p_invFinG(0, 0));
    double sin_phi = std::sin(p_invFinG(1, 0));
    double cos_phi = std::cos(p_invFinG(1, 0));
    double rho = p_invFinG(2, 0);

    // Construct the Jacobian
    H_f.resize(3, 3);
    H_f << -(1.0 / rho) * sin_th * sin_phi, (1.0 / rho) * cos_th * cos_phi, -(1.0 / (rho * rho)) * cos_th * sin_phi,
        (1.0 / rho) * cos_th * sin_phi, (1.0 / rho) * sin_th * cos_phi, -(1.0 / (rho * rho)) * sin_th * sin_phi, 0.0,
        -(1.0 / rho) * sin_phi, -(1.0 / (rho * rho)) * cos_phi;
    return;
  }

  //======================================================================
  //======================================================================
  //======================================================================

  // Assert that we have an anchor pose for this feature
  assert(feature.anchor_cam_id != -1);

  // Anchor pose orientation and position, and camera calibration for our anchor camera
  Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
  Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
  Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
  Eigen::Vector3d p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
  Eigen::Vector3d p_FinA = feature.p_FinA;

  // If I am doing FEJ, I should FEJ the anchor states (should we fej calibration???)
  // Also get the FEJ position of the feature if we are
  if (state->_options.do_fej) {
    // "Best" feature in the global frame
    Eigen::Vector3d p_FinG_best = R_GtoI.transpose() * R_ItoC.transpose() * (feature.p_FinA - p_IinC) + p_IinG;
    // Transform the best into our anchor frame using FEJ
    R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot_fej();
    p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos_fej();
    p_FinA = (R_GtoI.transpose() * R_ItoC.transpose()).transpose() * (p_FinG_best - p_IinG) + p_IinC;
  }
  Eigen::Matrix3d R_CtoG = R_GtoI.transpose() * R_ItoC.transpose();

  // Jacobian for our anchor pose
  Eigen::Matrix<double, 3, 6> H_anc;
  H_anc.block(0, 0, 3, 3).noalias() = -R_GtoI.transpose() * skew_x(R_ItoC.transpose() * (p_FinA - p_IinC));
  H_anc.block(0, 3, 3, 3).setIdentity();

  // Add anchor Jacobians to our return vector
  x_order.push_back(state->_clones_IMU.at(feature.anchor_clone_timestamp));
  H_x.push_back(H_anc);

  // Get calibration Jacobians (for anchor clone)
  if (state->_options.do_calib_camera_pose) {
    Eigen::Matrix<double, 3, 6> H_calib;
    H_calib.block(0, 0, 3, 3).noalias() = -R_CtoG * skew_x(p_FinA - p_IinC);
    H_calib.block(0, 3, 3, 3) = -R_CtoG;
    x_order.push_back(state->_calib_IMUtoCAM.at(feature.anchor_cam_id));
    H_x.push_back(H_calib);
  }

  // If we are doing anchored XYZ feature
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_3D) {
    H_f = R_CtoG;
    return;
  }

  // If we are doing full inverse depth
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_FULL_INVERSE_DEPTH) {

    // Get inverse depth representation (should match what is in Landmark.cpp)
    double a_rho = 1 / p_FinA.norm();
    double a_phi = std::acos(a_rho * p_FinA(2));
    double a_theta = std::atan2(p_FinA(1), p_FinA(0));
    Eigen::Matrix<double, 3, 1> p_invFinA;
    p_invFinA(0) = a_theta;
    p_invFinA(1) = a_phi;
    p_invFinA(2) = a_rho;

    // Using anchored inverse depth
    double sin_th = std::sin(p_invFinA(0, 0));
    double cos_th = std::cos(p_invFinA(0, 0));
    double sin_phi = std::sin(p_invFinA(1, 0));
    double cos_phi = std::cos(p_invFinA(1, 0));
    double rho = p_invFinA(2, 0);
    // assert(p_invFinA(2,0)>=0.0);

    // Jacobian of anchored 3D position wrt inverse depth parameters
    Eigen::Matrix<double, 3, 3> d_pfinA_dpinv;
    d_pfinA_dpinv << -(1.0 / rho) * sin_th * sin_phi, (1.0 / rho) * cos_th * cos_phi, -(1.0 / (rho * rho)) * cos_th * sin_phi,
        (1.0 / rho) * cos_th * sin_phi, (1.0 / rho) * sin_th * cos_phi, -(1.0 / (rho * rho)) * sin_th * sin_phi, 0.0,
        -(1.0 / rho) * sin_phi, -(1.0 / (rho * rho)) * cos_phi;
    H_f = R_CtoG * d_pfinA_dpinv;
    return;
  }

  // If we are doing the MSCKF version of inverse depth
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH) {

    // Get inverse depth representation (should match what is in Landmark.cpp)
    Eigen::Matrix<double, 3, 1> p_invFinA_MSCKF;
    p_invFinA_MSCKF(0) = p_FinA(0) / p_FinA(2);
    p_invFinA_MSCKF(1) = p_FinA(1) / p_FinA(2);
    p_invFinA_MSCKF(2) = 1 / p_FinA(2);

    // Using the MSCKF version of inverse depth
    double alpha = p_invFinA_MSCKF(0, 0);
    double beta = p_invFinA_MSCKF(1, 0);
    double rho = p_invFinA_MSCKF(2, 0);

    // Jacobian of anchored 3D position wrt inverse depth parameters
    Eigen::Matrix<double, 3, 3> d_pfinA_dpinv;
    d_pfinA_dpinv << (1.0 / rho), 0.0, -(1.0 / (rho * rho)) * alpha, 0.0, (1.0 / rho), -(1.0 / (rho * rho)) * beta, 0.0, 0.0,
        -(1.0 / (rho * rho));
    H_f = R_CtoG * d_pfinA_dpinv;
    return;
  }

  /// CASE: Estimate single depth of the feature using the initial bearing
  if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

    // Get inverse depth representation (should match what is in Landmark.cpp)
    double rho = 1.0 / p_FinA(2);
    Eigen::Vector3d bearing = rho * p_FinA;

    // Jacobian of anchored 3D position wrt inverse depth parameters
    Eigen::Vector3d d_pfinA_drho;
    d_pfinA_drho << -(1.0 / (rho * rho)) * bearing;
    H_f = R_CtoG * d_pfinA_drho;
    return;
  }

  // Failure, invalid representation that is not programmed
  assert(false);
}

void UpdaterHelper::get_feature_jacobian_full(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                              Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order,
                                              double sigma_pix, bool use_depth_residual) {

  // Total number of measurements for this feature
  int total_hx = 0;
  int total_rows = 0;
  std::unordered_map<std::shared_ptr<Type>, size_t> map_hx;
  
  for (auto const &pair : feature.timestamps) {
    // total_meas += (int)pair.second.size();
    std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(pair.first);
    std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(pair.first);

    // If doing calibration intrinsics
    if (state->_options.do_calib_camera_pose && map_hx.find(calibration) == map_hx.end()) {
      map_hx.insert({calibration, total_hx});
      x_order.push_back(calibration);
      total_hx += calibration->size();
    }

    if (state->_options.do_calib_camera_intrinsics && map_hx.find(distortion) == map_hx.end()) {
      map_hx.insert({distortion, total_hx});
      x_order.push_back(distortion);
      total_hx += distortion->size();
    }

    for (size_t m = 0; m < pair.second.size(); m++) {
      std::shared_ptr<PoseJPL> clone_Ci = state->_clones_IMU.at(pair.second.at(m));
      if (map_hx.find(clone_Ci) == map_hx.end()) {
        map_hx.insert({clone_Ci, total_hx});
        x_order.push_back(clone_Ci);
        total_hx += clone_Ci->size();
      }

      bool has_d = false;
      if (use_depth_residual && 
          feature.depths.find(pair.first) != feature.depths.end() && feature.depths.at(pair.first).size() > m &&
          feature.depth_vars.find(pair.first) != feature.depth_vars.end() && feature.depth_vars.at(pair.first).size() > m) {
        if (feature.depths.at(pair.first).at(m) > 0.0f && feature.depth_vars.at(pair.first).at(m) > 0.0f) {
          has_d = true;
        }
      }
      total_rows += has_d ? 3 : 2; // 2 for uv, 1 for depth
    }
  }
  
  // Calculate the position of this feature in the global frame
  // If anchored, then we need to calculate the position of the feature in the global
  Eigen::Vector3d p_FinG = feature.p_FinG;
  if (LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    // Assert that we have an anchor pose for this feature
    assert(feature.anchor_cam_id != -1);
    // Get calibration for our anchor camera
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
    // Anchor pose orientation and position
    Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
    Eigen::Vector3d p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
    // Feature in the global frame
    p_FinG = R_GtoI.transpose() * R_ItoC.transpose() * (feature.p_FinA - p_IinC) + p_IinG;
  }

  // Calculate the position of this feature in the global frame FEJ
  // If anchored, then we can use the "best" p_FinG since the value of p_FinA does not matter
  Eigen::Vector3d p_FinG_fej = feature.p_FinG_fej;
  if (LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    p_FinG_fej = p_FinG;
  }

  // Allocate our residual and Jacobians
  int jacobsize = (feature.feat_representation != LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 3 : 1;
  res = Eigen::VectorXd::Zero(2 * total_meas);
  H_f = Eigen::MatrixXd::Zero(2 * total_meas, jacobsize);
  H_x = Eigen::MatrixXd::Zero(2 * total_meas, total_hx);

  // Derivative of p_FinG in respect to feature representation.
  // This only needs to be computed once and thus we pull it out of the loop
  Eigen::MatrixXd dpfg_dlambda;
  std::vector<Eigen::MatrixXd> dpfg_dx;
  std::vector<std::shared_ptr<Type>> dpfg_dx_order;
  UpdaterHelper::get_feature_jacobian_representation(state, feature, dpfg_dlambda, dpfg_dx, dpfg_dx_order);

  int row = 0;

  for (auto const &pair : feature.timestamps) {
    std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(pair.first);
    std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(pair.first);
    Eigen::Matrix3d R_ItoC = calibration->Rot();
    Eigen::Vector3d p_IinC = calibration->pos();

    for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {

      bool has_d = false;
      float z_meas = -1.0f, z_var = -1.0f;
      if (use_depth_residual && feature.depths.find(pair.first) != feature.depths.end() && feature.depths.at(pair.first).size() > m) {
        z_meas = feature.depths.at(pair.first).at(m);
        z_var = feature.depth_vars.at(pair.first).at(m);
        if (z_meas > 0.0f && z_var > 0.0f) has_d = true;
      }

      std::shared_ptr<PoseJPL> clone_Ii = state->_clones_IMU.at(feature.timestamps[pair.first].at(m));
      Eigen::Matrix3d R_GtoIi = clone_Ii->Rot();
      Eigen::Vector3d p_IiinG = clone_Ii->pos();

      Eigen::Vector3d p_FinIi = R_GtoIi * (p_FinG - p_IiinG);
      Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
      Eigen::Vector3d p_FinCi_meas = p_FinCi; // Save the measured value for depth residuals

      Eigen::Vector2d uv_norm(p_FinCi(0) / p_FinCi(2), p_FinCi(1) / p_FinCi(2));
      Eigen::Vector2d uv_dist = state->_cam_intrinsics_cameras.at(pair.first)->distort_d(uv_norm);

      // Compute the residual (uv_meas - uv_pred) and weight it by the pixel noise
      Eigen::Vector2d uv_m((double)feature.uvs[pair.first].at(m)(0), (double)feature.uvs[pair.first].at(m)(1));
      
      double weight_pix = 1.0 / sigma_pix;
      res.block(row, 0, 2, 1) = (uv_m - uv_dist) * weight_pix;

      // If FEJ, recompute the feature position
      if (state->_options.do_fej) {
        R_GtoIi = clone_Ii->Rot_fej();
        p_IiinG = clone_Ii->pos_fej();
        p_FinIi = R_GtoIi * (p_FinG_fej - p_IiinG);
        p_FinCi = R_ItoC * p_FinIi + p_IinC;
      }

      Eigen::MatrixXd dz_dzn, dz_dzeta;
      state->_cam_intrinsics_cameras.at(pair.first)->compute_distort_jacobian(uv_norm, dz_dzn, dz_dzeta);

      Eigen::MatrixXd dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
      dzn_dpfc << 1/p_FinCi(2), 0, -p_FinCi(0)/(p_FinCi(2)*p_FinCi(2)), 0, 1/p_FinCi(2), -p_FinCi(1)/(p_FinCi(2)*p_FinCi(2));

      Eigen::MatrixXd dpfc_dpfg = R_ItoC * R_GtoIi;
      Eigen::MatrixXd dpfc_dclone = Eigen::MatrixXd::Zero(3, 6);
      dpfc_dclone.block(0, 0, 3, 3).noalias() = R_ItoC * skew_x(p_FinIi);
      dpfc_dclone.block(0, 3, 3, 3) = -dpfc_dpfg;

      Eigen::MatrixXd dz_dpfc = dz_dzn * dzn_dpfc;
      Eigen::MatrixXd dz_dpfg = dz_dpfc * dpfc_dpfg;

      // Compute the Jacobians and weight them by the pixel noise
      H_f.block(row, 0, 2, H_f.cols()).noalias() = (dz_dpfg * dpfg_dlambda) * weight_pix;
      H_x.block(row, map_hx[clone_Ii], 2, clone_Ii->size()).noalias() = (dz_dpfc * dpfc_dclone) * weight_pix;

      for (size_t i = 0; i < dpfg_dx_order.size(); i++) {
        H_x.block(row, map_hx[dpfg_dx_order.at(i)], 2, dpfg_dx_order.at(i)->size()).noalias() += (dz_dpfg * dpfg_dx.at(i)) * weight_pix;
      }

      if (state->_options.do_calib_camera_pose) {
        Eigen::MatrixXd dpfc_dcalib = Eigen::MatrixXd::Zero(3, 6);
        dpfc_dcalib.block(0, 0, 3, 3) = skew_x(p_FinCi - p_IinC);
        dpfc_dcalib.block(0, 3, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
        H_x.block(row, map_hx[calibration], 2, calibration->size()).noalias() += (dz_dpfc * dpfc_dcalib) * weight_pix;
      }

      if (state->_options.do_calib_camera_intrinsics) {
        H_x.block(row, map_hx[distortion], 2, distortion->size()) = dz_dzeta * weight_pix;
      }

      // Depth row injection and whitening
      if (has_d) {
        double sigma_z = std::sqrt(z_var);
        double weight_z = 1.0 / sigma_z;

        // Depth residual: z_meas - z_pred
        res(row + 2) = (z_meas - p_FinCi_meas(2)) * weight_z;

        // Projection Matrix e3^T = [0, 0, 1]
        Eigen::MatrixXd dzd_dpfc = Eigen::MatrixXd::Zero(1, 3);
        dzd_dpfc(0, 2) = 1.0; 
        Eigen::MatrixXd dzd_dpfg = dzd_dpfc * dpfc_dpfg;

        // Compute the Jacobians and weight them by the depth noise
        H_f.block(row + 2, 0, 1, H_f.cols()).noalias() = (dzd_dpfg * dpfg_dlambda) * weight_z;
        H_x.block(row + 2, map_hx[clone_Ii], 1, clone_Ii->size()).noalias() = (dzd_dpfc * dpfc_dclone) * weight_z;

        for (size_t i = 0; i < dpfg_dx_order.size(); i++) {
          H_x.block(row + 2, map_hx[dpfg_dx_order.at(i)], 1, dpfg_dx_order.at(i)->size()).noalias() += (dzd_dpfg * dpfg_dx.at(i)) * weight_z;
        }

        if (state->_options.do_calib_camera_pose) {
          Eigen::MatrixXd dpfc_dcalib = Eigen::MatrixXd::Zero(3, 6);
          dpfc_dcalib.block(0, 0, 3, 3) = skew_x(p_FinCi - p_IinC);
          dpfc_dcalib.block(0, 3, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
          H_x.block(row + 2, map_hx[calibration], 1, 6).noalias() += (dzd_dpfc * dpfc_dcalib) * weight_z;
        }
      }

      // Cursor advancement
      row += has_d ? 3 : 2;
    }
  }
}

void UpdaterHelper::nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

  // Apply the left nullspace of H_f to all variables
  // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
  // See page 252, Algorithm 5.2.4 for how these two loops work
  // They use "matlab" index notation, thus we need to subtract 1 from all index
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_f.cols(); ++n) {
    for (int m = (int)H_f.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_x.block(m - 1, 0, 2, H_x.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // The H_f jacobian max rank is 3 if it is a 3d position, thus size of the left nullspace is Hf.rows()-3
  // NOTE: need to eigen3 eval here since this experiences aliasing!
  // H_f = H_f.block(H_f.cols(),0,H_f.rows()-H_f.cols(),H_f.cols()).eval();
  H_x = H_x.block(H_f.cols(), 0, H_x.rows() - H_f.cols(), H_x.cols()).eval();
  res = res.block(H_f.cols(), 0, res.rows() - H_f.cols(), res.cols()).eval();

  // Sanity check
  assert(H_x.rows() == res.rows());
}

void UpdaterHelper::measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

  // Return if H_x is a fat matrix (there is no need to compress in this case)
  if (H_x.rows() <= H_x.cols())
    return;

  // Do measurement compression through givens rotations
  // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
  // See page 252, Algorithm 5.2.4 for how these two loops work
  // They use "matlab" index notation, thus we need to subtract 1 from all index
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_x.cols(); n++) {
    for (int m = (int)H_x.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_x(m - 1, n), H_x(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_x.block(m - 1, n, 2, H_x.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // If H is a fat matrix, then use the rows
  // Else it should be same size as our state
  int r = std::min(H_x.rows(), H_x.cols());

  // Construct the smaller jacobian and residual after measurement compression
  assert(r <= H_x.rows());
  H_x.conservativeResize(r, H_x.cols());
  res.conservativeResize(r, res.cols());
}
