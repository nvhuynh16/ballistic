#pragma once

// =============================================================================
// BatchInterpolatorFunction - Vectorized batch trajectory queries
// =============================================================================
//
// Returns Eigen matrices (→ numpy arrays) instead of per-trajectory Python
// objects. Designed for 100K+ Monte Carlo trajectories.
//
// Usage (from Python):
//   f = batch()                    # BatchInterpolatorFunction
//   r = f(1800.0)                  # BatchArrayResult with (N,3) numpy arrays
//   impact = f.at_alt(0)           # Altitude-crossing search
//   f[42].get_state(epoch + 1800)  # Individual trajectory access
//
// =============================================================================

#include "trajectory_batch.hpp"
#include <ballistic/coordinates/geodetic.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

namespace ballistic {
namespace python {

// Vectorized result from batch queries.
// All arrays have N rows (one per trajectory).
struct BatchArrayResult {
    Eigen::MatrixXd positions;      // (N, 3) in output frame, meters
    Eigen::MatrixXd velocities;     // (N, 3) in output frame, m/s
    Eigen::MatrixXd accelerations;  // (N, 3) in output frame, m/s^2
    Eigen::VectorXd epochs_s;       // (N,)   seconds from reference epoch
    Eigen::VectorXi valid;          // (N,)   1=valid, 0=invalid

    // Number of trajectories (rows)
    [[nodiscard]] Eigen::Index size() const noexcept {
        return valid.size();
    }
};

// Vectorized interpolation function for batch trajectories.
// Created via Trajectory().__call__() with no arguments in batch mode.
class BatchInterpolatorFunction {
public:
    BatchInterpolatorFunction(TrajectoryBatch& batch, const Epoch& reference_epoch)
        : batch_(batch)
        , reference_epoch_(reference_epoch)
        , frame_(InterpolationFrame::ECEF)
    {}

    // Non-copyable (references TrajectoryBatch)
    BatchInterpolatorFunction(const BatchInterpolatorFunction&) = delete;
    BatchInterpolatorFunction& operator=(const BatchInterpolatorFunction&) = delete;

    // Movable
    BatchInterpolatorFunction(BatchInterpolatorFunction&&) = default;
    BatchInterpolatorFunction& operator=(BatchInterpolatorFunction&&) = default;

private:
    // Write one row of a BatchArrayResult, either from an interpolated state
    // (valid=1) or as a zero-filled invalid row (valid=0).
    static void write_state_row(BatchArrayResult& r, Eigen::Index i,
                                const InterpolatedState& s, const Epoch& ref) noexcept {
        r.positions.row(i)     = s.position.transpose();
        r.velocities.row(i)    = s.velocity.transpose();
        r.accelerations.row(i) = s.acceleration.transpose();
        r.epochs_s(i)          = s.epoch - ref;
        r.valid(i)             = 1;
    }
    static void write_zero_row(BatchArrayResult& r, Eigen::Index i, double epoch_s = 0.0) noexcept {
        r.positions.row(i).setZero();
        r.velocities.row(i).setZero();
        r.accelerations.row(i).setZero();
        r.epochs_s(i) = epoch_s;
        r.valid(i)    = 0;
    }

public:

    // =========================================================================
    // Query Methods
    // =========================================================================

    // Query all trajectories at a single epoch.
    // Returns BatchArrayResult with (N,3) position/velocity/acceleration matrices.
    [[nodiscard]] BatchArrayResult query(const Epoch& t) {
        // Enforce frame consistency — the batch's interpolators may have been
        // changed externally (e.g., by at_altitude's ECEF bisection pass, or
        // by direct access to the TrajectoryBatch).
        ensure_frame();

        const auto N = static_cast<Eigen::Index>(batch_.size());

        BatchArrayResult result;
        result.positions.resize(N, 3);
        result.velocities.resize(N, 3);
        result.accelerations.resize(N, 3);
        result.epochs_s.resize(N);
        result.valid.resize(N);

        for (Eigen::Index i = 0; i < N; ++i) {
            auto& interp = batch_.interpolator(static_cast<size_t>(i));
            auto state_opt = interp.get_state(t);
            if (state_opt)  write_state_row(result, i, *state_opt, reference_epoch_);
            else            write_zero_row(result, i, t - reference_epoch_);
        }

        return result;
    }

    // Query all trajectories at multiple epochs.
    // Returns one BatchArrayResult per time point.
    [[nodiscard]] std::vector<BatchArrayResult> query_times(
        const std::vector<Epoch>& times)
    {
        std::vector<BatchArrayResult> results;
        results.reserve(times.size());
        for (const auto& t : times) {
            results.push_back(query(t));
        }
        return results;
    }

    // =========================================================================
    // Altitude Crossing Search
    // =========================================================================

    // Search existing history for altitude crossing (no propagation side effect).
    //
    // descending=true:  find the LAST segment where altitude goes from >= target
    //                   to < target (reentry/impact)
    // descending=false: find the LAST segment where altitude goes from < target
    //                   to >= target (ascending crossing)
    //
    // max_time_s: maximum time window to search (seconds from reference epoch).
    //             0 = search entire history (default). Limits which history
    //             segments are considered for crossing detection.
    //
    // Uses bisection within the crossing segment for ~1mm altitude precision.
    // Returns states in the current output frame.
    [[nodiscard]] BatchArrayResult at_altitude(
        double altitude_m,
        bool descending = true,
        double max_time_s = 0.0)
    {
        const auto N = static_cast<Eigen::Index>(batch_.size());

        // Step 1: Scan history for crossing segments (ECEF, no interpolator needed)
        std::vector<int64_t> crossing_segment(static_cast<size_t>(N), -1);

        // Compute time limit epoch (0 = no limit)
        const bool has_time_limit = max_time_s > 0.0;
        const Epoch time_limit = has_time_limit
            ? (reference_epoch_ + max_time_s) : Epoch{};

        for (Eigen::Index i = 0; i < N; ++i) {
            const auto& hist = batch_.trajectory(static_cast<size_t>(i)).history();
            if (hist.size() < 2) continue;

            // Scan backward to find the LAST crossing (within time limit)
            for (int64_t j = static_cast<int64_t>(hist.size()) - 2; j >= 0; --j) {
                const auto ju = static_cast<size_t>(j);

                // Skip segments beyond time limit
                if (has_time_limit && hist[ju + 1].epoch > time_limit) continue;

                const double alt_j  = coords::ecef_to_altitude(hist[ju].position);
                const double alt_j1 = coords::ecef_to_altitude(hist[ju + 1].position);

                bool is_crossing = descending
                    ? (alt_j >= altitude_m && alt_j1 < altitude_m)
                    : (alt_j < altitude_m && alt_j1 >= altitude_m);

                if (is_crossing) {
                    crossing_segment[static_cast<size_t>(i)] = j;
                    break;
                }
            }
        }

        // Step 2: Bisect within crossing segments to find precise crossing times.
        // Bisection needs ECEF positions for altitude computation.
        batch_.set_output_frame(InterpolationFrame::ECEF);

        std::vector<Epoch> crossing_epochs(static_cast<size_t>(N));
        std::vector<bool> found(static_cast<size_t>(N), false);

        constexpr double ALT_BISECTION_TOL_M = 0.001;  // 1mm
        constexpr int MAX_BISECTION_ITER = 50;

        for (Eigen::Index i = 0; i < N; ++i) {
            const auto idx = static_cast<size_t>(i);
            if (crossing_segment[idx] < 0) continue;

            const auto& hist = batch_.trajectory(idx).history();
            const auto seg = static_cast<size_t>(crossing_segment[idx]);

            Epoch t_lo = hist[seg].epoch;
            Epoch t_hi = hist[seg + 1].epoch;
            Epoch t_mid = t_lo;

            auto& interp = batch_.interpolator(idx);

            for (int iter = 0; iter < MAX_BISECTION_ITER; ++iter) {
                double dt = t_hi - t_lo;
                t_mid = t_lo + dt * 0.5;

                auto pos_opt = interp.position(t_mid);
                if (!pos_opt) break;

                double alt_mid = coords::ecef_to_altitude(*pos_opt);
                if (std::abs(alt_mid - altitude_m) < ALT_BISECTION_TOL_M) break;

                bool above = alt_mid >= altitude_m;
                if (descending ? above : !above) {
                    t_lo = t_mid;
                } else {
                    t_hi = t_mid;
                }
            }

            crossing_epochs[idx] = t_mid;
            found[idx] = true;
        }

        // Step 3: Restore user's frame, then query crossing epochs.
        // ensure_frame() is authoritative — restores frame_ regardless of
        // what at_altitude or external code did to the batch.
        ensure_frame();
        BatchArrayResult result;
        result.positions.resize(N, 3);
        result.velocities.resize(N, 3);
        result.accelerations.resize(N, 3);
        result.epochs_s.resize(N);
        result.valid.resize(N);

        for (Eigen::Index i = 0; i < N; ++i) {
            const auto idx = static_cast<size_t>(i);
            if (!found[idx]) { write_zero_row(result, i); continue; }

            auto& interp = batch_.interpolator(idx);
            auto state_opt = interp.get_state(crossing_epochs[idx]);
            if (state_opt) write_state_row(result, i, *state_opt, reference_epoch_);
            else           write_zero_row(result, i);
        }

        return result;
    }

    // =========================================================================
    // Individual Access
    // =========================================================================

    // Access individual trajectory's interpolator (for debugging).
    [[nodiscard]] TrajectoryInterpolator& at(size_t index) {
        return batch_.interpolator(index);
    }

    // Number of trajectories.
    [[nodiscard]] size_t size() const noexcept {
        return batch_.size();
    }

    // Reference epoch for float→Epoch conversion.
    [[nodiscard]] const Epoch& reference_epoch() const noexcept {
        return reference_epoch_;
    }

    // =========================================================================
    // Frame Configuration
    // =========================================================================

    // Set output coordinate frame for all queries.
    void set_output_frame(InterpolationFrame frame) {
        frame_ = frame;
        batch_.set_output_frame(frame);
    }

    // Get current output frame.
    [[nodiscard]] InterpolationFrame output_frame() const noexcept {
        return frame_;
    }

private:
    TrajectoryBatch& batch_;
    Epoch reference_epoch_;
    InterpolationFrame frame_;

    // Ensure batch interpolators match our frame state.
    // Cheap (just sets a field per interpolator, no recomputation).
    void ensure_frame() {
        batch_.set_output_frame(frame_);
    }
};

} // namespace python
} // namespace ballistic
