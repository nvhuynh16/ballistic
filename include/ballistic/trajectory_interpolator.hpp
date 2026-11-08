#pragma once

// =============================================================================
// Trajectory Interpolation System
// =============================================================================
//
// Gives the state at ANY time from the discrete states a Trajectory has
// stored, by fitting a quintic (degree-5) Hermite polynomial between each
// pair of neighboring states. "Hermite" means the polynomial matches not
// only the positions at both ends but also the velocities and accelerations,
// which the history stores. The interpolator wraps a Trajectory and provides:
//
// - Sub-millimeter interpolation accuracy for typical step sizes (10-30s)
// - Automatic propagation when a query lies outside the computed history
// - Caching of each segment's polynomial coefficients for repeated queries
// - Output in the Earth-fixed (ECEF/ITRF) or inertial (GCRS) frame
//
// Using position, velocity and acceleration gives a local error that shrinks
// as h⁶ with the step h, much better than cubic Hermite (h⁴, position and
// velocity only) and without needing the extra points a Lagrange polynomial
// of the same order would.
//
// Usage:
//   Trajectory traj(InitialState::from_itrf(t0, pos, vel));
//   while (traj.epoch() - t0 < 3600.0) traj.step();  // Build history
//
//   TrajectoryInterpolator interp(traj);
//   auto state = interp.get_state(t0 + 1800.0);  // Query at any time
//   if (state) {
//       std::cout << "Position: " << state->position.transpose() << std::endl;
//   }
//
// =============================================================================

#include "trajectory.hpp"
#include "math/quintic_hermite.hpp"
#include "coordinates/earth_rotation.hpp"
#include "coordinates/iau_transforms.hpp"
#include "coordinates/geodetic.hpp"
#include "util/frame_transform.hpp"
#include <optional>
#include <unordered_map>
#include <vector>
#include <span>
#include <algorithm>
#include <cmath>

namespace ballistic {

// =============================================================================
// Enums and Configuration
// =============================================================================

/// Frame of the interpolated states.
enum class InterpolationFrame {
    ECEF,      // Earth-Centered Earth-Fixed (ITRF; default, matches Trajectory output)
    GCRS       // Geocentric Celestial Reference System: inertial, rotated from
               // ECEF with the IAU 2006/2000A model
};

/// Which side of the target altitude propagate_to_altitude looks for.
enum class AltitudeCondition {
    Above,     // Propagate until altitude >= target
    Below      // Propagate until altitude <= target
};

/// Time direction for propagate_to_altitude.
enum class PropagationDirection {
    Forward,   // Propagate from last history entry toward future
    Backward   // Propagate from first history entry toward past
};

/// Result of propagate_to_altitude.
struct PropagateToAltitudeResult {
    bool condition_met;           // Did we reach the target altitude within max_duration_s?
    size_t steps_taken;           // Number of entries added to history
    double starting_altitude_m;   // Geodetic altitude at starting point (meters)
    double ending_altitude_m;     // Geodetic altitude at final point (meters)
};

/// The state at one query time.
struct InterpolatedState {
    Epoch epoch;                // Query time
    Vector3d position;          // Position in requested frame (meters)
    Vector3d velocity;          // Velocity in requested frame (m/s)
    Vector3d acceleration;      // Acceleration in requested frame (m/s²)
    bool was_propagated;        // True if history was extended for this query
    size_t segment_index;       // Index of segment used (i = history[i] to history[i+1])
};

/// Settings for a TrajectoryInterpolator.
struct InterpolatorConfig {
    // Output coordinate frame
    InterpolationFrame output_frame = InterpolationFrame::ECEF;

    // ITRF realization for frame conversions. nullopt (default) inherits the
    // wrapped Trajectory's configured realization, so the two can't desync;
    // set explicitly only to override.
    std::optional<coords::ITRFRealization> itrf;

    // Enable coefficient caching (recommended for repeated queries)
    bool cache_coefficients = true;

    // Allow automatic propagation for out-of-range queries
    bool allow_propagation = true;

    // Maximum propagation time beyond current history (seconds)
    // Prevents runaway propagation on unreasonable queries
    double max_propagation_time = 3600.0;
};

// =============================================================================
// TrajectoryInterpolator Class
// =============================================================================

/// Answers "where is the satellite at time t?" from a Trajectory's history,
/// stepping the Trajectory further (forward or backward) when t lies outside
/// it. Holds a reference: the Trajectory must outlive the interpolator.
class TrajectoryInterpolator {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /// Wrap `traj`. If its history is still empty, the first query records
    /// the initial state and takes one step.
    explicit TrajectoryInterpolator(
        Trajectory& traj,
        const InterpolatorConfig& config = {})
        : trajectory_(traj)
        , config_(config)
        , cached_history_size_(0)
    {}

    // =========================================================================
    // Single-Point Queries
    // =========================================================================

    /// The state at time `t`, in the configured output frame.
    /// - Inside the history: interpolated.
    /// - Outside it, with allow_propagation: the Trajectory is stepped to
    ///   cover t (at most max_propagation_time seconds past the history),
    ///   then interpolated.
    /// - Otherwise: std::nullopt.
    /// Errors from the Trajectory while extending (e.g. DataCoverageError at
    /// the edge of the space weather) propagate to the caller.
    [[nodiscard]] std::optional<InterpolatedState> get_state(const Epoch& t) {
        ensure_history_initialized();

        // Check if within current range
        if (covers(t)) {
            return interpolate_within_range(t, false);
        }

        // Out of range - try to extend if allowed
        if (!config_.allow_propagation) {
            return std::nullopt;
        }

        // Check propagation limits
        double dt_forward = t - history_end();
        double dt_backward = history_start() - t;

        if (dt_forward > 0 && dt_forward > config_.max_propagation_time) {
            return std::nullopt;  // Too far in future
        }
        if (dt_backward > 0 && dt_backward > config_.max_propagation_time) {
            return std::nullopt;  // Too far in past
        }

        // Extend forward if needed
        if (dt_forward > 0) {
            if (!extend_to(t)) {
                return std::nullopt;
            }
            return interpolate_within_range(t, true);
        }

        // Extend backward if needed
        if (dt_backward > 0) {
            if (!extend_backward_to(t)) {
                return std::nullopt;
            }
            return interpolate_within_range(t, true);
        }

        return std::nullopt;
    }

    // =========================================================================
    // Convenience Single-Value Queries
    // =========================================================================

    /// Position only (see get_state).
    [[nodiscard]] std::optional<Vector3d> position(const Epoch& t) {
        auto state = get_state(t);
        if (!state) return std::nullopt;
        return state->position;
    }

    /// Velocity only (see get_state).
    [[nodiscard]] std::optional<Vector3d> velocity(const Epoch& t) {
        auto state = get_state(t);
        if (!state) return std::nullopt;
        return state->velocity;
    }

    /// Acceleration only (see get_state).
    [[nodiscard]] std::optional<Vector3d> acceleration(const Epoch& t) {
        auto state = get_state(t);
        if (!state) return std::nullopt;
        return state->acceleration;
    }

    // =========================================================================
    // Batch Queries
    // =========================================================================

    /// States at many times (in any order). Faster than single queries: the
    /// history is extended once to cover the whole range, and the times are
    /// processed in sorted order so neighboring queries reuse cached
    /// segments. Returns one result per input time, in the input order; an
    /// entry is nullopt where get_state would return nullopt.
    [[nodiscard]] std::vector<std::optional<InterpolatedState>> get_states(
        std::span<const Epoch> times)
    {
        if (times.empty()) {
            return {};
        }

        ensure_history_initialized();

        // Find time range
        Epoch t_min = times[0];
        Epoch t_max = times[0];
        for (const auto& t : times) {
            if (t < t_min) t_min = t;
            if (t > t_max) t_max = t;
        }

        // Extend history to cover range (if allowed and needed)
        if (config_.allow_propagation) {
            if (t_max > history_end()) {
                double dt = t_max - history_end();
                if (dt <= config_.max_propagation_time) {
                    extend_to(t_max);
                }
            }
            if (t_min < history_start()) {
                double dt = history_start() - t_min;
                if (dt <= config_.max_propagation_time) {
                    extend_backward_to(t_min);
                }
            }
        }

        // Create index array for sorting while preserving original order
        std::vector<size_t> indices(times.size());
        for (size_t i = 0; i < times.size(); ++i) {
            indices[i] = i;
        }

        // Sort indices by time for cache locality
        std::sort(indices.begin(), indices.end(),
            [&times](size_t a, size_t b) { return times[a] < times[b]; });

        // Query in sorted order
        std::vector<std::optional<InterpolatedState>> results(times.size());
        for (size_t idx : indices) {
            results[idx] = get_state(times[idx]);
        }

        return results;
    }

    // =========================================================================
    // Range Information
    // =========================================================================

    /// Time of the earliest stored state.
    [[nodiscard]] Epoch history_start() const {
        ensure_history_initialized();
        return trajectory_.history().front().epoch;
    }

    /// Time of the latest stored state.
    [[nodiscard]] Epoch history_end() const {
        ensure_history_initialized();
        return trajectory_.history().back().epoch;
    }

    /// True if `t` lies within the stored history (needs at least two states).
    [[nodiscard]] bool covers(const Epoch& t) const {
        ensure_history_initialized();
        const auto& history = trajectory_.history();
        if (history.size() < 2) return false;

        return t >= history.front().epoch && t <= history.back().epoch;
    }

    /// Number of stored states.
    [[nodiscard]] size_t history_size() const {
        return trajectory_.history_size();
    }

    // =========================================================================
    // History Extension
    // =========================================================================

    /// Step the Trajectory forward until its history reaches `target`.
    /// Returns true (errors from the Trajectory are thrown, not returned).
    bool extend_to(const Epoch& target) {
        ensure_history_initialized();

        // Already covered?
        if (target <= history_end()) {
            return true;
        }

        // Propagate forward
        while (trajectory_.epoch() < target) {
            (void)trajectory_.step();
        }

        // Invalidate cache (history changed)
        clear_cache();

        return true;
    }

    /// Step the Trajectory backward until its history starts at or before
    /// `target`. Returns true (errors from the Trajectory are thrown).
    bool extend_backward_to(const Epoch& target) {
        ensure_history_initialized();

        // Already covered?
        if (target >= history_start()) {
            return true;
        }

        // Propagate backward
        while (trajectory_.history().front().epoch > target) {
            (void)trajectory_.step_backward();
        }

        // Invalidate cache (history changed)
        clear_cache();

        return true;
    }

    // =========================================================================
    // Altitude-Based Propagation
    // =========================================================================

    /// Propagate until the geodetic altitude (height above the WGS84
    /// ellipsoid) crosses `target_altitude_m` [m].
    ///
    ///   condition: Above = until altitude >= target (a rising crossing),
    ///              Below = until altitude <= target (a falling crossing).
    ///   max_duration_s: how far to search [s]; condition_met = false if the
    ///              altitude is not reached within it (default 7 days).
    ///
    /// Forward: runs Trajectory::propagate_until with a terminal
    /// events::altitude_crossing, so the history ENDS AT the crossing (located
    /// to 1e-4 s by re-integrating the crossing step).
    /// Ground impact stops a forward search too (condition_met = false unless
    /// the target is the ground).
    /// Backward: steps back from the history front and stops on the first
    /// step past the crossing (not located inside the step).
    /// If the starting point already satisfies the condition, returns at once.
    [[nodiscard]] PropagateToAltitudeResult propagate_to_altitude(
        double target_altitude_m,
        AltitudeCondition condition,
        PropagationDirection direction = PropagationDirection::Forward,
        double max_duration_s = 7.0 * 86400.0)
    {
        ensure_history_initialized();
        const bool forward = direction == PropagationDirection::Forward;
        const auto& ref = forward ? trajectory_.history().back() : trajectory_.history().front();
        const double start_alt = coords::ecef_to_geodetic(ref.position).altitude;
        if (check_altitude_condition(start_alt, target_altitude_m, condition))
            return PropagateToAltitudeResult{true, 0, start_alt, start_alt};

        const size_t size0 = trajectory_.history_size();
        PropagateToAltitudeResult out{false, 0, start_alt, start_alt};
        if (forward) {
            const auto dir = condition == AltitudeCondition::Above ? events::Direction::Rising
                                                                   : events::Direction::Falling;
            const auto r = trajectory_.propagate_until(
                trajectory_.epoch() + max_duration_s,
                {events::altitude_crossing(target_altitude_m, dir, true)});
            out.ending_altitude_m = coords::ecef_to_geodetic(r.state_ecef.position).altitude;
            out.condition_met = r.stopped_by.has_value() &&
                (r.stopped_by->name != "ground impact" || target_altitude_m == 0.0);
        } else {
            const Epoch front0 = trajectory_.history().front().epoch;
            while (front0 - trajectory_.history().front().epoch < max_duration_s) {
                (void)trajectory_.step_backward();
                out.ending_altitude_m =
                    coords::ecef_to_geodetic(trajectory_.history().front().position).altitude;
                if (check_altitude_condition(out.ending_altitude_m, target_altitude_m, condition)) {
                    out.condition_met = true;
                    break;
                }
            }
        }
        out.steps_taken = trajectory_.history_size() - size0;
        clear_cache();
        return out;
    }

    // =========================================================================
    // Cache Management
    // =========================================================================

    /// Drop the cached segment coefficients.
    void clear_cache() {
        coefficient_cache_.clear();
        cached_history_size_ = 0;
    }

    /// Same as clear_cache(). The cache is also dropped automatically whenever
    /// the history's size changes.
    void invalidate() {
        clear_cache();
    }

    // =========================================================================
    // Configuration Access
    // =========================================================================

    /// The current settings.
    [[nodiscard]] const InterpolatorConfig& config() const {
        return config_;
    }

    /// Choose the frame of the returned states.
    void set_output_frame(InterpolationFrame frame) {
        if (config_.output_frame != frame) {
            config_.output_frame = frame;
            clear_cache();  // Frame affects cached results
        }
    }

    /// Override the ITRF realization used for the GCRS conversion (default:
    /// the Trajectory's).
    void set_itrf_realization(coords::ITRFRealization itrf) {
        config_.itrf = itrf;
    }

    /// Allow or forbid stepping the Trajectory for out-of-range queries.
    void set_allow_propagation(bool allow) {
        config_.allow_propagation = allow;
    }

    /// Furthest a query may lie outside the history [s] before get_state
    /// returns nullopt instead of propagating.
    void set_max_propagation_time(double max_seconds) {
        config_.max_propagation_time = max_seconds;
    }

private:
    Trajectory& trajectory_;
    InterpolatorConfig config_;

    // Cache for segment coefficients (key = segment index)
    mutable std::unordered_map<size_t, math::QuinticSegmentCoefficients> coefficient_cache_;
    mutable size_t cached_history_size_;

    // =========================================================================
    // Internal Methods
    // =========================================================================

    // An empty history: record the initial state and take one step, so there
    // is a segment to interpolate on.
    void ensure_history_initialized() const {
        if (trajectory_.history_size() == 0) {
            // Force initial state to be recorded
            (void)const_cast<Trajectory&>(trajectory_).step();
        }
    }

    // Check if cache needs invalidation due to history change
    void check_cache_validity() const {
        if (trajectory_.history_size() != cached_history_size_) {
            coefficient_cache_.clear();
            cached_history_size_ = trajectory_.history_size();
        }
    }

    // Check if altitude satisfies the given condition
    [[nodiscard]] static bool check_altitude_condition(
        double current_altitude_m,
        double target_altitude_m,
        AltitudeCondition condition) noexcept
    {
        if (condition == AltitudeCondition::Above) {
            return current_altitude_m >= target_altitude_m;
        } else {
            return current_altitude_m <= target_altitude_m;
        }
    }

    // Segment duration from the exact elapsed-seconds time axis (t_rel_s),
    // falling back to the epoch difference for entries without t_rel_s (both
    // zero — possible only for hand-built histories; Trajectory always fills it).
    [[nodiscard]] static double segment_duration(const HistoryEntry& h0,
                                                 const HistoryEntry& h1) {
        const double dt_rel = h1.t_rel_s - h0.t_rel_s;
        return dt_rel > 0.0 ? dt_rel : h1.epoch - h0.epoch;
    }

    // Find segment index for a given epoch
    // Returns index i such that history[i].epoch <= t < history[i+1].epoch
    [[nodiscard]] std::optional<size_t> find_segment(const Epoch& t) const {
        const auto& history = trajectory_.history();

        if (history.size() < 2) return std::nullopt;
        if (t < history.front().epoch) return std::nullopt;
        if (t > history.back().epoch) return std::nullopt;

        // Binary search for segment
        size_t lo = 0;
        size_t hi = history.size() - 1;

        while (lo < hi - 1) {
            size_t mid = (lo + hi) / 2;
            if (t < history[mid].epoch) {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        return lo;
    }

    // Get or compute segment coefficients
    [[nodiscard]] const math::QuinticSegmentCoefficients& get_segment_coefficients(
        size_t segment_idx) const
    {
        check_cache_validity();

        if (config_.cache_coefficients) {
            auto it = coefficient_cache_.find(segment_idx);
            if (it != coefficient_cache_.end()) {
                return it->second;
            }
        }

        // Compute coefficients
        const auto& history = trajectory_.history();
        const auto& h0 = history[segment_idx];
        const auto& h1 = history[segment_idx + 1];

        double dt = segment_duration(h0, h1);

        auto coeffs = math::compute_segment_coefficients(
            dt,
            h0.position, h0.velocity, h0.acceleration,
            h1.position, h1.velocity, h1.acceleration);

        if (config_.cache_coefficients) {
            auto [it, inserted] = coefficient_cache_.emplace(segment_idx, coeffs);
            return it->second;
        }

        // Not caching - store temporarily and return
        // (This is a bit inefficient, but caching is recommended)
        static thread_local math::QuinticSegmentCoefficients temp_coeffs;
        temp_coeffs = coeffs;
        return temp_coeffs;
    }

    // Interpolate at a time known to be within history range
    [[nodiscard]] InterpolatedState interpolate_within_range(
        const Epoch& t,
        bool was_propagated) const
    {
        auto seg_opt = find_segment(t);
        if (!seg_opt) {
            // This shouldn't happen if called correctly
            throw std::logic_error("interpolate_within_range called with out-of-range time");
        }

        size_t seg_idx = *seg_opt;
        const auto& history = trajectory_.history();
        const auto& h0 = history[seg_idx];
        const auto& h1 = history[seg_idx + 1];

        // Interpolate on the exact elapsed-seconds time axis (t_rel_s), not
        // on the nodes' Epoch labels: any rounding in the labels, independent
        // from node to node, would add a step-rate sawtooth to the
        // interpolated positions, which estimators that differentiate
        // positions once or twice amplify.
        //
        // A query that EQUALS a node's epoch (e.g. get_state(history_at(i).epoch))
        // is pinned to s = 0 or 1, so it returns the stored node state to
        // rounding error (the position at a segment's start bit for bit).
        double dt = segment_duration(h0, h1);
        double s;
        if (t == h0.epoch) {
            s = 0.0;
        } else if (t == h1.epoch) {
            s = 1.0;
        } else if (dt > 0.0 && h1.t_rel_s > h0.t_rel_s) {
            s = ((t - trajectory_.label_ref()) - h0.t_rel_s) / dt;
        } else {
            s = (t - h0.epoch) / (h1.epoch - h0.epoch);
        }

        // Clamp to [0, 1] so rounding cannot extrapolate past a node: the
        // segment is found from the epoch labels, s from the exact time axis,
        // and the two may disagree by a rounding error at a node. The Hermite
        // polynomial matches the node exactly at s = 0 and 1.
        s = (std::max)(s, 0.0);
        s = (std::min)(s, 1.0);

        // Get coefficients and interpolate
        const auto& coeffs = get_segment_coefficients(seg_idx);

        double inv_dt = 1.0 / dt;
        double inv_dt2 = inv_dt * inv_dt;

        Vector3d pos_ecef = coeffs.position(s);
        Vector3d vel_ecef = coeffs.velocity(s, inv_dt);
        Vector3d acc_ecef = coeffs.acceleration(s, inv_dt2);

        // Transform to output frame if needed
        Vector3d pos_out, vel_out, acc_out;

        if (config_.output_frame == InterpolationFrame::ECEF) {
            pos_out = pos_ecef;
            vel_out = vel_ecef;
            acc_out = acc_ecef;
        } else {
            // GCRS: full IAU 2006/2000A transformation
            StateVector state_ecef(pos_ecef, vel_ecef);
            StateVector state_gcrs = coords::itrf_to_gcrs_full(
                state_ecef, t, config_.itrf.value_or(trajectory_.config().itrf));
            pos_out = state_gcrs.position;
            vel_out = state_gcrs.velocity;

            // The stored acceleration is the second derivative of the Earth-fixed
            // position, which includes the Coriolis and centrifugal terms of the
            // rotating frame. Remove them before rotating so the output is the
            // inertial (GCRS) acceleration.
            acc_out = util::acceleration_to_gcrs(acc_ecef, state_ecef, t);
        }

        return InterpolatedState{
            t,
            pos_out,
            vel_out,
            acc_out,
            was_propagated,
            seg_idx
        };
    }
};

// =============================================================================
// Factory Function
// =============================================================================

/// Same as std::make_unique<TrajectoryInterpolator>(traj, config).
[[nodiscard]] inline std::unique_ptr<TrajectoryInterpolator> create_interpolator(
    Trajectory& traj,
    const InterpolatorConfig& config = {})
{
    return std::make_unique<TrajectoryInterpolator>(traj, config);
}

} // namespace ballistic
