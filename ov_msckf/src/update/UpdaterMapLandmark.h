# ifndef OV_MSCKF_UPDATER_MAP_LANDMARK_H
# define OV_MSCKF_UPDATER_MAP_LANDMARK_H

# include "LandmarkTypes.h"
# include "UpdaterMapLandmarkOptions.h"
# include <memory>
# include <mutex>
# include <vector>

/// Forward declaration
namespace ov_type {
class Type;
class PoseJPL;
} // namespace ov_type

namespace ov_msckf {
/// Forward declaration
class State;

/**
 * @brief Keeps T_MtoG in the state and updates it with map landmark observations
 *
 * T_MtoG (q_MtoG, p_GinM) is added to the covariance once VIO is initialized, from the prior map_initial_T_M_B.
 * Between updates it follows a random walk. Landmark points that pass a per-point chi2 gate update the clones and T_MtoG.
 */
class UpdaterMapLandmark {
public:
    UpdaterMapLandmark(UpdaterMapLandmarkOptions opts, LandmarkMap landmark_map, std::vector<LandmarkObject> objects = {});

    /// Thread-safe. Shifts the stamp to the cam0 clock; points whose id is not in the map and objects of an unknown type are dropped.
    void feed_observation(const LandmarkObservation &obs);

    /// Call once per image after the state is propagated to it: initializes T_MtoG, then propagates it
    void update(std::shared_ptr<State> state);

    bool is_initialized() const { return _initialized; }

private:
    bool try_initialize(std::shared_ptr<State> state);
    /// Random walk of T_MtoG from the last propagation time to the state time
    void propagate(std::shared_ptr<State> state);
    /// Removes and returns the pending observations not newer than t
    std::vector<LandmarkObservation> take_ready(double t);
    /// IMU pose at an observation time: one clone (c1 == nullptr), or interpolated between c0 and c1 with weight lambda on c1
    struct ClonePair {
        std::shared_ptr<ov_type::PoseJPL> c0, c1;
        double lambda = 0.0;
    };
    /// False if t is outside the clone window
    bool find_clones(std::shared_ptr<State> state, double t, ClonePair &cp) const;
    /// Clones of cp followed by T_MtoG, the column order of H
    std::vector<std::shared_ptr<ov_type::Type>> h_order(std::shared_ptr<State> state, const ClonePair &cp) const;
    /// Stacked predicted pixels of obs, and H = d(uv)/d[h_order] (2N x 12 or 18) if H != nullptr.
    /// in_front == nullptr: false if a point is not visible (behind the camera or far outside the image).
    /// Otherwise those points get in_front[i] = false and zero rows.
    bool predict(std::shared_ptr<State> state, const ClonePair &cp, const LandmarkObservation &obs, Eigen::VectorXd &uv,
                 Eigen::MatrixXd *H, std::vector<bool> *in_front = nullptr) const;
    /// Matches object detections to map objects of the same type by projecting them with the current state,
    /// appends the keypoints of the matched ones to points. Returns the number of matched objects.
    int associate(std::shared_ptr<State> state, const ClonePair &cp, const LandmarkObservation &obs,
                  std::vector<LandmarkPoint> &points) const;
    /// Scales the T_MtoG covariance (and its cross terms) by factor
    void inflate_T_MtoG(std::shared_ptr<State> state, double factor) const;
    void print_frame(const LandmarkObservation &obs, const ClonePair &cp, int n_front, int n_kept, double median_r,
                     double max_chi2) const;
    /// Compares H with central differences through PoseJPL::update(), i.e. the error state OpenVINS uses
    void check_jacobian(std::shared_ptr<State> state, const ClonePair &cp, const LandmarkObservation &obs, const Eigen::MatrixXd &H) const;
    UpdaterMapLandmarkOptions _opts;
    LandmarkMap _map;
    std::vector<LandmarkObject> _objects;
    std::vector<LandmarkObservation> _pending;
    std::mutex _pending_mtx;
    bool _initialized = false;
    double _last_prop_time = -1.0;
    int _n_no_clone = 0;
    int _n_objects_matched = 0;
    int _n_objects_unmatched = 0;
    int _n_interpolated = 0;
    int _n_jacobian_checks = 0;
    int _n_accepted = 0; // frames used in an update
    int _n_rejected = 0; // frames with every point rejected
    int _n_points_rejected = 0;
    int _n_points_behind = 0;
    int _n_cov_inflations = 0;
    int _n_consecutive_rejected = 0;
    int _n_consecutive_inflations = 0;
    int _n_consecutive_all_behind = 0;
};

} // namespace ov_msckf

# endif