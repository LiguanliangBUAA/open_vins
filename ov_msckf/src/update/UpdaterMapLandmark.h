# ifndef OV_MSCKF_UPDATER_MAP_LANDMARK_H
# define OV_MSCKF_UPDATER_MAP_LANDMARK_H

# include "LandmarkTypes.h"
# include "UpdaterMapLandmarkOptions.h"
# include <map>
# include <memory>
# include <mutex>
# include <vector>

/// Forward declaration
namespace ov_type {
class Type;
} // namespace ov_type

namespace ov_msckf {
/// Forward declaration
class State;

/**
 * @brief Keeps T_MtoG in the state and updates it with map landmark observations
 *
 * T_MtoG (q_MtoG, p_GinM) is added to the covariance once VIO is initialized, from the prior map_initial_T_M_B.
 * Between updates it follows a random walk. Landmark observations are not used yet.
 */
class UpdaterMapLandmark {
public:
    UpdaterMapLandmark(UpdaterMapLandmarkOptions opts, std::map<int, MapLandmark> landmark_map);

    /// Thread-safe: called by ant source (in-process detector or external topic subscriber)
    void feed_observations(const std::vector<LandmarkObservation> &obs);

    /// Call once per image after the state is propagated to it: initializes T_MtoG, then propagates it
    void update(std::shared_ptr<State> state);

    bool is_initialized() const { return _initialized; }

private:
    bool try_initialize(std::shared_ptr<State> state);
    /// Random walk of T_MtoG from the last propagation time to the state time
    void propagate(std::shared_ptr<State> state);
    /// Observation to Landmark
    void build_landmark_system(std::shared_ptr<State> state, const LandmarkObservation &ob,
                               Eigen::MatrixXd &H_x, Eigen::VectorXd &res, 
                               std::vector<std::shared_ptr<ov_type::Type>> &x_order);
    UpdaterMapLandmarkOptions _opts;
    std::map<int, MapLandmark> _map;
    std::vector<LandmarkObservation> _pending;
    std::mutex _pending_mtx;
    bool _initialized = false;
    double _last_prop_time = -1.0;
};

} // namespace ov_msckf

# endif