#pragma once

// =============================================================================
// TrajectoryBatch - Batch Operations for Multiple Trajectories
// =============================================================================
//
// This class manages multiple trajectories and provides efficient batch
// operations that keep loops on the C++ side for performance.
//
// Key design principles:
// - Batch operations return vectors of results
// - No per-trajectory Python callbacks during batch operations
// - Efficient memory management with unique_ptr
//
// Usage (from Python):
//   batch = ballistic.TrajectoryBatch([
//       (epoch1, pos1, vel1),
//       (epoch2, pos2, vel2),
//   ], config)
//   batch.propagate_all(3600.0)
//   states = batch.get_states_at(query_epoch)
//
// =============================================================================

#include <ballistic/trajectory.hpp>
#include <ballistic/trajectory_interpolator.hpp>
#include <memory>
#include <vector>
#include <optional>
#include <stdexcept>

namespace ballistic {
namespace python {

// Result from batch interpolation query
struct BatchInterpolatedState {
    size_t trajectory_index;      // Which trajectory this came from
    bool valid;                   // Whether the query succeeded
    Epoch epoch;                  // Query time
    Vector3d position;            // Position in output frame
    Vector3d velocity;            // Velocity in output frame
    Vector3d acceleration;        // Acceleration in output frame
    bool was_propagated;          // True if history was extended
};

// Result from batch altitude propagation
struct BatchAltitudeResult {
    size_t trajectory_index;      // Which trajectory this came from
    bool condition_met;           // Did we reach the target altitude?
    size_t steps_taken;           // Number of steps added to history
    double starting_altitude_m;   // Geodetic altitude at starting point
    double ending_altitude_m;     // Geodetic altitude at final point
};

// =============================================================================
// TrajectoryBatch Class
// =============================================================================

class TrajectoryBatch {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    // Non-copyable (contains unique_ptr members)
    TrajectoryBatch(const TrajectoryBatch&) = delete;
    TrajectoryBatch& operator=(const TrajectoryBatch&) = delete;

    // Movable
    TrajectoryBatch(TrajectoryBatch&&) = default;
    TrajectoryBatch& operator=(TrajectoryBatch&&) = default;

    // Construct with initial states and optional configuration
    TrajectoryBatch(
        const std::vector<ballistic::InitialState>& states,
        const TrajectoryConfig& config = {})
        : config_(config)
    {
        trajectories_.reserve(states.size());
        interpolators_.reserve(states.size());

        for (const auto& state : states) {
            trajectories_.push_back(std::make_unique<Trajectory>(state, config));
        }

        // Create interpolators lazily when needed
    }

    // =========================================================================
    // Size and Access
    // =========================================================================

    // Get number of trajectories
    [[nodiscard]] size_t size() const noexcept {
        return trajectories_.size();
    }

    // Check if batch is empty
    [[nodiscard]] bool empty() const noexcept {
        return trajectories_.empty();
    }

    // Access individual trajectory (for advanced usage)
    [[nodiscard]] Trajectory& trajectory(size_t index) {
        if (index >= trajectories_.size()) {
            throw std::out_of_range("Trajectory index out of range");
        }
        return *trajectories_[index];
    }

    [[nodiscard]] const Trajectory& trajectory(size_t index) const {
        if (index >= trajectories_.size()) {
            throw std::out_of_range("Trajectory index out of range");
        }
        return *trajectories_[index];
    }

    // Access individual interpolator (creates if needed)
    [[nodiscard]] TrajectoryInterpolator& interpolator(size_t index) {
        ensure_interpolators();
        if (index >= interpolators_.size()) {
            throw std::out_of_range("Interpolator index out of range");
        }
        return *interpolators_[index];
    }

    // =========================================================================
    // Batch Propagation
    // =========================================================================

    // Propagate every member by duration_seconds from ITS OWN current epoch,
    // landing exactly there (members may start at different epochs). Forward
    // runs record history (batch interpolation reads it).
    void propagate_all(double duration_seconds) {
        for (auto& traj : trajectories_) advance(*traj, traj->epoch() + duration_seconds);
        invalidate_interpolators();
    }

    // Propagate every member to the same epoch, landing exactly on it.
    void propagate_all_to(const Epoch& target) {
        for (auto& traj : trajectories_) advance(*traj, target);
        invalidate_interpolators();
    }

    // Take N steps for all trajectories
    void step_all(size_t n_steps = 1) {
        for (auto& traj : trajectories_) {
            for (size_t i = 0; i < n_steps; ++i) {
                (void)traj->step();
            }
        }
        invalidate_interpolators();
    }

    // =========================================================================
    // Batch Queries
    // =========================================================================

    // Query all trajectories at a single epoch
    // Returns one result per trajectory
    [[nodiscard]] std::vector<BatchInterpolatedState> get_states_at(const Epoch& t) {
        ensure_interpolators();
        std::vector<BatchInterpolatedState> results;
        results.reserve(interpolators_.size());

        for (size_t i = 0; i < interpolators_.size(); ++i) {
            auto state_opt = interpolators_[i]->get_state(t);
            results.push_back(make_batch_state(i, state_opt, t));
        }

        return results;
    }

    // Query all trajectories at multiple epochs
    // Returns [trajectory_index][time_index] results
    [[nodiscard]] std::vector<std::vector<BatchInterpolatedState>> get_states_at_times(
        const std::vector<Epoch>& times)
    {
        ensure_interpolators();

        if (!times.empty()) {
            // Pre-compute time range
            Epoch t_min = times[0];
            Epoch t_max = times[0];
            for (const auto& t : times) {
                if (t < t_min) t_min = t;
                if (t > t_max) t_max = t;
            }

            // Pre-extend each trajectory's coverage
            for (auto& interp : interpolators_) {
                if (interp->config().allow_propagation) {
                    double max_prop = interp->config().max_propagation_time;
                    if (t_max > interp->history_end()) {
                        double dt = t_max - interp->history_end();
                        if (dt <= max_prop) {
                            interp->extend_to(t_max);
                        }
                    }
                    if (t_min < interp->history_start()) {
                        double dt = interp->history_start() - t_min;
                        if (dt <= max_prop) {
                            interp->extend_backward_to(t_min);
                        }
                    }
                }
            }
        }

        std::vector<std::vector<BatchInterpolatedState>> results;
        results.resize(interpolators_.size());

        for (size_t i = 0; i < interpolators_.size(); ++i) {
            results[i].reserve(times.size());

            for (size_t j = 0; j < times.size(); ++j) {
                auto state_opt = interpolators_[i]->get_state(times[j]);
                results[i].push_back(make_batch_state(i, state_opt, times[j]));
            }
        }

        return results;
    }

    // =========================================================================
    // Altitude-Based Propagation
    // =========================================================================

    // Propagate all trajectories until altitude condition is met
    [[nodiscard]] std::vector<BatchAltitudeResult> propagate_to_altitude(
        double target_altitude_m,
        AltitudeCondition condition,
        PropagationDirection direction = PropagationDirection::Forward,
        double max_duration_s = 7.0 * 86400.0)
    {
        ensure_interpolators();
        std::vector<BatchAltitudeResult> results;
        results.reserve(interpolators_.size());

        for (size_t i = 0; i < interpolators_.size(); ++i) {
            auto result = interpolators_[i]->propagate_to_altitude(
                target_altitude_m, condition, direction, max_duration_s);

            BatchAltitudeResult batch_result;
            batch_result.trajectory_index = i;
            batch_result.condition_met = result.condition_met;
            batch_result.steps_taken = result.steps_taken;
            batch_result.starting_altitude_m = result.starting_altitude_m;
            batch_result.ending_altitude_m = result.ending_altitude_m;

            results.push_back(batch_result);
        }

        return results;
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    // Get configuration
    [[nodiscard]] const TrajectoryConfig& config() const noexcept {
        return config_;
    }

    // Set output frame for all interpolators
    void set_output_frame(InterpolationFrame frame) {
        interp_config_.output_frame = frame;
        for (auto& interp : interpolators_) {
            if (interp) {
                interp->set_output_frame(frame);
            }
        }
    }

    // Set ITRF realization for all interpolators
    void set_itrf_realization(coords::ITRFRealization itrf) {
        interp_config_.itrf = itrf;
        for (auto& interp : interpolators_) {
            if (interp) {
                interp->set_itrf_realization(itrf);
            }
        }
    }

    // Set allow propagation for all interpolators
    void set_allow_propagation(bool allow) {
        interp_config_.allow_propagation = allow;
        for (auto& interp : interpolators_) {
            if (interp) {
                interp->set_allow_propagation(allow);
            }
        }
    }

    // Set max propagation time for all interpolators
    void set_max_propagation_time(double max_seconds) {
        interp_config_.max_propagation_time = max_seconds;
        for (auto& interp : interpolators_) {
            if (interp) {
                interp->set_max_propagation_time(max_seconds);
            }
        }
    }

private:
    static void advance(Trajectory& t, const Epoch& target) {
        if (target - t.epoch() >= 0.0)
            (void)t.propagate_until(target, {}, /*stop_at_ground=*/false);
        else
            (void)t.propagate(target);
    }

    TrajectoryConfig config_;
    InterpolatorConfig interp_config_;
    std::vector<std::unique_ptr<Trajectory>> trajectories_;
    std::vector<std::unique_ptr<TrajectoryInterpolator>> interpolators_;

    // Helper to construct BatchInterpolatedState from optional InterpolatedState
    [[nodiscard]] BatchInterpolatedState make_batch_state(
        size_t traj_index,
        const std::optional<InterpolatedState>& state_opt,
        const Epoch& query_time) const
    {
        BatchInterpolatedState result;
        result.trajectory_index = traj_index;

        if (state_opt) {
            result.valid = true;
            result.epoch = state_opt->epoch;
            result.position = state_opt->position;
            result.velocity = state_opt->velocity;
            result.acceleration = state_opt->acceleration;
            result.was_propagated = state_opt->was_propagated;
        } else {
            result.valid = false;
            result.epoch = query_time;
            result.position = Vector3d::Zero();
            result.velocity = Vector3d::Zero();
            result.acceleration = Vector3d::Zero();
            result.was_propagated = false;
        }

        return result;
    }

    // Ensure interpolators are created
    void ensure_interpolators() {
        if (interpolators_.size() != trajectories_.size()) {
            interpolators_.clear();
            interpolators_.reserve(trajectories_.size());
            for (auto& traj : trajectories_) {
                interpolators_.push_back(
                    std::make_unique<TrajectoryInterpolator>(*traj, interp_config_));
            }
        }
    }

    // Invalidate interpolators when history changes
    void invalidate_interpolators() {
        for (auto& interp : interpolators_) {
            if (interp) {
                interp->invalidate();
            }
        }
    }
};

} // namespace python
} // namespace ballistic
