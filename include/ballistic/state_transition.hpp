#pragma once

// =============================================================================
// State transition matrix and covariance along a propagated Trajectory
// =============================================================================
//
// The state transition matrix (STM) Phi(t, t0) maps a small change of the state
// at t0 into the change at t:
//
//   dx(t) = Phi(t, t0) dx(t0),      x = [r (m); v (m/s)] in the GCRS frame.
//
// It obeys the variational equation dPhi/dt = A(t) Phi with Phi(t0, t0) = I,
// where A(t) = d[v; a]/d[r; v] is evaluated along the propagated trajectory
// (propagators/variational.hpp). A covariance of the start state is carried to
// t as P(t) = Phi P_0 Phi^T (no process noise, no consider parameters).
//
// StateTransition is a companion to a finished Trajectory: it reads the
// trajectory's history (Trajectory::step or propagate_until record it),
// converts each recorded state to GCRS at its own epoch, interpolates between
// them in GCRS (quintic Hermite on the exact time axis), integrates Phi with
// classical 4th-order Runge-Kutta over each step in substeps of at most
// StmOptions::max_substep_s, and stores Phi at every recorded step. A query
// between steps integrates the remainder from the step before it. The
// trajectory itself is not changed, so propagation results are identical with
// or without it.
//
// Accuracy caveats:
//   * radiation pressure uses a hard cylindrical Earth shadow, so a is not
//     differentiable at the shadow boundary; an arc that crosses it gets a Phi
//     that misses the jump in sunlight;
//   * drag's position columns of da/dr come from finite differences (offset
//     fd_dx), good to roughly 1e-4 relative; everything else is analytic;
//   * the empirical acceleration depends only on time and adds nothing to A.
//
//   Trajectory traj(InitialState::from_itrf(t0, r, v), cfg);
//   traj.propagate_until(t0 + 5400.0);
//   StateTransition st(traj);
//   Eigen::Matrix<double, 6, 6> phi = st.stm(t0 + 3600.0);
//   Eigen::Matrix<double, 6, 6> P   = st.covariance(t0 + 3600.0, P_0);
// =============================================================================

#include "core/defaults.hpp"
#include "core/errors.hpp"
#include "core/types.hpp"
#include "propagators/variational.hpp"
#include "time/epoch.hpp"
#include "trajectory.hpp"
#include "math/quintic_hermite.hpp"
#include "util/frame_transform.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace ballistic {

/// Settings for StateTransition.
struct StmOptions {
    double fd_dx = 1.0;          // [m] finite-difference offset, position (drag columns only)
    double fd_dv = 1e-3;         // [m/s] finite-difference offset, velocity (velocity-dependent beta only)
    double max_substep_s = 10.0; // [s] largest Runge-Kutta substep for Phi
};

/// The state transition matrix and covariance along a propagated Trajectory
/// (see the file comment). Copies the recorded states it needs, but holds a
/// reference to the Trajectory's force model: the Trajectory must outlive this
/// object. Steps the Trajectory takes later are not covered; build a new one.
class StateTransition {
public:
    using Matrix6 = Eigen::Matrix<double, 6, 6>;

    /// Integrate Phi over the whole recorded history of `traj`. Throws
    /// ConfigError when the history has fewer than two entries or the options
    /// are not positive.
    explicit StateTransition(const Trajectory& traj, StmOptions options = {})
        : traj_(&traj), options_(options)
    {
        if (!(options_.max_substep_s > 0.0) || !(options_.fd_dx > 0.0) ||
            !(options_.fd_dv > 0.0)) {
            throw ConfigError(
                "StateTransition: fd_dx, fd_dv and max_substep_s must be positive (got " +
                std::to_string(options_.fd_dx) + " m, " + std::to_string(options_.fd_dv) +
                " m/s, " + std::to_string(options_.max_substep_s) + " s)");
        }
        const auto& history = traj.history();
        if (history.size() < 2) {
            throw ConfigError(
                "StateTransition: the trajectory has " + std::to_string(history.size()) +
                " recorded state(s); it needs at least two, because the transition matrix is "
                "integrated along the recorded steps. Propagate it first with step() or "
                "propagate_until() (propagate() records no history).");
        }
        epoch0_ = history.front().epoch;
        epoch_end_ = history.back().epoch;
        const auto itrf = traj.config().itrf;

        // GCRS nodes, each converted at its own epoch. Interpolating the
        // Earth-fixed history and converting at the query time instead would
        // add the Earth rotation angle's rounding at every query (it is
        // computed from a one-double UT1 Julian date, ~40 us steps = ~1 cm at
        // LEO) and make A(t) jagged between nodes.
        std::vector<Vector3d> p, v, a;
        p.reserve(history.size()); v.reserve(history.size()); a.reserve(history.size());
        t_.reserve(history.size());
        for (const auto& h : history) {
            const StateVector e{h.position, h.velocity};
            const StateVector g = util::itrf_to_gcrs(e, h.epoch, itrf);
            p.push_back(g.position);
            v.push_back(g.velocity);
            a.push_back(util::acceleration_to_gcrs(h.acceleration, e, h.epoch));
            t_.push_back(h.t_rel_s - history.front().t_rel_s);
        }
        seg_.reserve(history.size() - 1);
        for (size_t k = 0; k + 1 < history.size(); ++k) {
            const double dt = t_[k + 1] - t_[k];
            if (!(dt > 0.0)) {
                throw ConfigError(
                    "StateTransition: two recorded states of the trajectory share the time " +
                    utc(history[k].epoch) + " (UTC); the history must increase in time.");
            }
            seg_.push_back(math::compute_segment_coefficients(dt, p[k], v[k], a[k],
                                                              p[k + 1], v[k + 1], a[k + 1]));
        }

        phi_.reserve(history.size());
        Matrix6 phi = Matrix6::Identity();
        phi_.push_back(phi);
        for (size_t k = 0; k + 1 < t_.size(); ++k) {
            phi = integrate(k, phi, t_[k], t_[k + 1]);
            phi_.push_back(phi);
        }
    }

    /// Phi(t, epoch0()): 6x6, GCRS, state order [r (m); v (m/s)]. Throws
    /// DataCoverageError when `t` lies outside [epoch0(), epoch_end()].
    [[nodiscard]] Matrix6 stm(const Epoch& t) const {
        double dt = t - epoch0_;
        const double t_end = t_.back();
        constexpr double tol = defaults::EPOCH_EQUALITY_SECONDS;
        if (dt < -tol || dt > t_end + tol) {
            throw DataCoverageError(
                "StateTransition: " + utc(t) + " is outside the propagated span " + utc(epoch0_) +
                " to " + utc(epoch_end_) + " (UTC); propagate the trajectory to cover it and "
                "build a new StateTransition.");
        }
        dt = std::clamp(dt, 0.0, t_end);
        const auto it = std::upper_bound(t_.begin(), t_.end(), dt);
        const size_t k = static_cast<size_t>(std::max<std::ptrdiff_t>(it - t_.begin() - 1, 0));
        if (dt - t_[k] <= tol) return phi_[k];
        return integrate(k, phi_[k], t_[k], dt);
    }

    /// P(t) = Phi P_0 Phi^T: the start-state covariance `P_0` (6x6, symmetric,
    /// GCRS, [r (m); v (m/s)], so units m^2, m^2/s, m^2/s^2) carried to `t`.
    /// Throws InputPlausibilityError for a non-finite or non-symmetric P_0 and
    /// DataCoverageError as stm().
    [[nodiscard]] Matrix6 covariance(const Epoch& t, const Matrix6& P_0) const {
        if (!P_0.allFinite()) {
            throw InputPlausibilityError(
                "StateTransition::covariance: P_0 has a non-finite entry (NaN or infinity).");
        }
        const double asym = (P_0 - P_0.transpose()).cwiseAbs().maxCoeff();
        const double scale = P_0.cwiseAbs().maxCoeff();
        if (asym > 1e-12 * scale) {
            throw InputPlausibilityError(
                "StateTransition::covariance: P_0 is not symmetric (largest |P_0 - P_0^T| = " +
                std::to_string(asym) + ", largest |P_0| = " + std::to_string(scale) +
                "); a covariance matrix must be. Pass (P_0 + P_0^T) / 2 if the asymmetry is "
                "rounding.");
        }
        const Matrix6 phi = stm(t);
        Matrix6 P = phi * P_0 * phi.transpose();
        return 0.5 * (P + P.transpose());  // symmetric to the last bit
    }

    /// First recorded epoch of the trajectory: Phi is relative to this time.
    [[nodiscard]] const Epoch& epoch0() const noexcept { return epoch0_; }
    /// Last recorded epoch covered.
    [[nodiscard]] const Epoch& epoch_end() const noexcept { return epoch_end_; }
    /// Number of recorded steps Phi is stored at.
    [[nodiscard]] size_t node_count() const noexcept { return t_.size(); }
    /// The options used.
    [[nodiscard]] const StmOptions& options() const noexcept { return options_; }

private:
    const Trajectory* traj_;
    StmOptions options_;
    Epoch epoch0_;
    Epoch epoch_end_;
    std::vector<double> t_;      // node offsets from epoch0_ [s] (exact time axis)
    std::vector<math::QuinticSegmentCoefficients> seg_;  // GCRS interpolant per step
    std::vector<Matrix6> phi_;   // Phi(t_k, t_0)

    // A at offset t [s] inside segment k (t_k <= t <= t_k+1).
    [[nodiscard]] Matrix6 A_at(size_t k, double t) const {
        const double dt = t_[k + 1] - t_[k];
        const double s = std::clamp((t - t_[k]) / dt, 0.0, 1.0);
        const StateVector x{seg_[k].position(s), seg_[k].velocity(s, 1.0 / dt)};
        return propagators::detail::variational_matrix(traj_->force_model(), epoch0_ + t, x,
                                          options_.fd_dx, options_.fd_dv);
    }

    // Classical RK4 for dPhi/dt = A Phi from t_a to t_b (both inside segment
    // k) in equal substeps of at most max_substep_s.
    [[nodiscard]] Matrix6 integrate(size_t k, Matrix6 phi, double t_a, double t_b) const {
        const double span = t_b - t_a;
        if (span <= 0.0) return phi;
        const int n = std::max(1, static_cast<int>(std::ceil(span / options_.max_substep_s - 1e-9)));
        const double h = span / n;
        Matrix6 A0 = A_at(k, t_a);
        for (int i = 0; i < n; ++i) {
            const double t = t_a + i * h;
            const double t_next = (i + 1 == n) ? t_b : t_a + (i + 1) * h;
            const Matrix6 Am = A_at(k, 0.5 * (t + t_next));
            const Matrix6 A1 = A_at(k, t_next);
            const Matrix6 k1 = A0 * phi;
            const Matrix6 k2 = Am * (phi + 0.5 * h * k1);
            const Matrix6 k3 = Am * (phi + 0.5 * h * k2);
            const Matrix6 k4 = A1 * (phi + h * k3);
            phi += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            A0 = A1;
        }
        return phi;
    }

    [[nodiscard]] static std::string utc(const Epoch& e) {
        const auto [y, mo, d, h, mi, s] = e.to_calendar(TimeScale::UTC);
        char buf[48];
        std::snprintf(buf, sizeof buf, "%04d-%02d-%02d %02d:%02d:%06.3f", y, mo, d, h, mi, s);
        return buf;
    }
};

}  // namespace ballistic
