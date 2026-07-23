# ifndef OV_MSCKF_UPDATER_MAP_LANDMARK_H
# define OV_MSCKF_UPDATER_MAP_LANDMARK_H

# include "LandmarkTypes.h"
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

struct UpdaterMapLandmarkOptions {
    std::map<LandmarkType, LandmarkTypeConfig> type_cfg; // per-type config - noise/gating/enable
    int init_min_obs = 3; // observation times before delayed init
    double init_max_dist = 1e4; // PnP conditioning gate at init
    double rw_ori = 0.0; // T_map_odom random-walk PSD
    double rw_pos = 0.0;
    double max_obs_age = 0.15; // drop observations older than newest clone by this [s]
};

class UpdaterMapLandmark {
public:
    UpdaterMapLandmark(UpdaterMapLandmarkOptions opts, std::map<int, MapLandmark> landmark_map);

    /// Thread-safe: called by ant source (in-process detector or external topic subscriber)
    void feed_observations(const std::vector<LandmarkObservation> &obs);

    void update(std::shared_ptr<State> state);

    bool is_initialized() const { return _initialized; }

private:
    bool try_initialize(std::shared_ptr<State> state);
    /// Observation to Landmark
    void build_landmark_system(std::shared_ptr<State> state, const LandmarkObservation &ob,
                               Eigen::MatrixXd &H_x, Eigen::VectorXd &res, 
                               std::vector<std::shared_ptr<ov_type::Type>> &x_order);
    UpdaterMapLandmarkOptions _opts;
    std::map<int, MapLandmark> _map;
    std::vector<LandmarkObservation> _pending;
    std::mutex _pending_mtx;
    bool _initialized = false;
};

} // namespace ov_msckf

# endif