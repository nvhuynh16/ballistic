#pragma once

#include "../core/errors.hpp"
#include "../core/types.hpp"
#include "../core/defaults.hpp"
#include "../time/epoch.hpp"
#include "../forces/force_model.hpp"
#include "../coordinates/iau_transforms.hpp"
#include "../util/frame_transform.hpp"
#include "propagator.hpp"
#include <functional>
#include <cmath>
#include <algorithm>
#include <array>
#include <memory>
#include <string>

namespace ballistic {
namespace propagators {

// Symplectic integrators for orbit propagation.
//
// A symplectic integrator is a fixed-step method built for systems whose total energy is
// conserved (Hamiltonian systems, such as an orbit under gravity alone). Instead of letting the
// energy error grow step after step, as a general-purpose method like RK45 does, it keeps the
// energy error bounded and oscillating. That makes it the better choice for propagations of
// many hours or days. This file implements:
//   - leapfrog (Stormer-Verlet, 2nd order) and Yoshida's compositions of it (6th and 8th
//     order), which alternate "drift" updates of the position with "kick" updates of the
//     velocity;
//   - two ways to add forces that do not conserve energy ("non-conservative" forces: drag,
//     radiation pressure, empirical accelerations): a simple splitting correction and the
//     "slimplectic" variational method of Tsang et al. (2015).

/// Conservative force callback: acceleration [m/s²] at (epoch, position [m]). It must not depend
/// on velocity (gravity, tides, third bodies).
using ConservativeForce = std::function<Vector3d(const Epoch&, const Vector3d&)>;
/// Non-conservative force callback: acceleration [m/s²] at (epoch, position and velocity). Used
/// for forces that remove or add energy (drag, radiation pressure, empirical accelerations).
using NonConservativeForce = std::function<Vector3d(const Epoch&, const StateVector&)>;

/// Order of accuracy of the symplectic method (the local error per step scales as step^(order+1)).
/// For new code, prefer `IntegratorType` in trajectory.hpp, which selects RK45 or a symplectic
/// method through one interface.
enum class SymplecticOrder {
    LEAPFROG = 2,   ///< Stormer-Verlet leapfrog (2nd order); needs small steps
    YOSHIDA6 = 6,   ///< Yoshida 6th order, 7 force evaluations per step (default)
    YOSHIDA8 = 8    ///< Yoshida 8th order, 15 force evaluations per step
};

/// How forces that do not conserve energy (drag, radiation pressure, ...) are added.
enum class NonConservativeStrategy {
    NONE,           ///< Ignore them: pure symplectic step with conservative forces only
    SPLITTING,      ///< Symplectic step for conservative forces, then one velocity correction
                    ///< with the non-conservative acceleration at the step midpoint
    SLIMPLECTIC     ///< Variational ("slimplectic") step that includes them inside each stage
};

/// Settings for `SymplecticIntegrator` and `SymplecticPropagator`.
struct SymplecticConfig {
    double step_size = 30.0;  ///< Fixed step [s] (reference when adaptive); see default_step_size()
    SymplecticOrder order = SymplecticOrder::YOSHIDA6;
    NonConservativeStrategy nc_strategy = NonConservativeStrategy::SLIMPLECTIC;

    // Optional time-reversible adaptive stepping (Hairer, Lubich & Wanner 2006, §VIII.3): the
    // step shrinks where the gravitational acceleration is large (near perigee) and grows where
    // it is small, chosen symmetrically so the method stays time-reversible.
    // adaptive: true = variable step; false = fixed `step_size`.
    bool adaptive = false;
    double min_step_size = 1.0;       ///< Smallest adaptive step [s]
    double max_step_size = 120.0;     ///< Largest adaptive step [s]
    double adaptation_alpha = 0.0;    ///< Exponent of the step rule; 0 = automatic (1/order)
    double symmetry_tolerance = 0.1;  ///< Relative start/end step mismatch accepted
    int max_symmetry_iters = 3;       ///< Maximum retries to make the step choice symmetric

    /// Recommended fixed step size [s] for an order. Leapfrog's 2nd-order error needs much
    /// smaller steps than the 6th/8th-order methods.
    static constexpr double default_step_size(SymplecticOrder ord) {
        switch (ord) {
            case SymplecticOrder::LEAPFROG: return 0.5;   // 2nd order needs small steps
            case SymplecticOrder::YOSHIDA6: return 30.0;  // 6th order
            case SymplecticOrder::YOSHIDA8: return 30.0;  // 8th order
            default: return 30.0;
        }
    }
};

/// Stage counts and leapfrog coefficients, for reference. The integrator computes its own
/// coefficients at construction (see `SymplecticIntegrator`); nothing reads these constants.
/// `c` weights the position (drift) updates, `d` the velocity (kick) updates.
namespace yoshida {
    // 2nd order (Leapfrog/Störmer-Verlet)
    constexpr int stages_2 = 1;
    constexpr double c_2[] = {0.5, 0.5};
    constexpr double d_2[] = {1.0};

    // 6th order Yoshida coefficients (solution A from Yoshida 1990)
    constexpr int stages_6 = 7;

    // 8th order (15 stages)
    constexpr int stages_8 = 15;
}

/// Fixed-step (or optionally reversible-adaptive) symplectic integrator for a second-order
/// system: position q [m] and velocity p [m/s] driven by user-supplied force callbacks. It knows
/// nothing about frames; the caller decides which frame the state and forces are in
/// (`SymplecticPropagator` uses GCRS). Exceptions thrown by the force callbacks pass through.
class SymplecticIntegrator {
public:
    /// Integrator with both conservative and non-conservative forces.
    /// @param conservative_force      acceleration from energy-conserving forces (gravity, tides)
    /// @param non_conservative_force  acceleration from the rest (drag, radiation); may be empty
    /// @param config                  order, step size and non-conservative strategy
    SymplecticIntegrator(
        ConservativeForce conservative_force,
        NonConservativeForce non_conservative_force,
        SymplecticConfig config = {})
        : conservative_force_(std::move(conservative_force))
        , non_conservative_force_(std::move(non_conservative_force))
        , config_(config)
    {
        compute_coefficients();
    }

    /// Integrator with conservative forces only; the non-conservative strategy is forced to NONE.
    explicit SymplecticIntegrator(
        ConservativeForce conservative_force,
        SymplecticConfig config = {})
        : conservative_force_(std::move(conservative_force))
        , non_conservative_force_(nullptr)
        , config_(config)
    {
        config_.nc_strategy = NonConservativeStrategy::NONE;
        compute_coefficients();
    }

    /// Advance the state by one step of `dt` seconds (negative = backward in time) using the
    /// configured non-conservative strategy. Returns the new state; does not update `stats()`.
    [[nodiscard]] StateVector step(
        const Epoch& epoch,
        const StateVector& state,
        double dt)
    {
        switch (config_.nc_strategy) {
            case NonConservativeStrategy::NONE:
                return symplectic_step(epoch, state, dt);

            case NonConservativeStrategy::SPLITTING:
                return splitting_step(epoch, state, dt);

            case NonConservativeStrategy::SLIMPLECTIC:
                return slimplectic_step(epoch, state, dt);

            default:
                return symplectic_step(epoch, state, dt);
        }
    }

    /// Integrate from `start_state` at `start_epoch` to `target_epoch` (either direction) and
    /// return the state there. The last step is shortened to land exactly on the target.
    /// Updates `stats()`.
    [[nodiscard]] StateVector integrate(
        const Epoch& start_epoch,
        const StateVector& start_state,
        const Epoch& target_epoch)
    {
        double remaining = target_epoch - start_epoch;
        double direction = (remaining >= 0) ? 1.0 : -1.0;
        remaining = std::abs(remaining);

        Epoch current_epoch = start_epoch;
        StateVector current_state = start_state;
        double elapsed = 0.0;  // exact accumulator: labels = start_epoch + elapsed
                               // (chaining current_epoch + dt would add the rounding of
                               // every Epoch addition to the time labels)

        if (config_.adaptive) {
            // Adaptive stepping: variable step sizes with time-reversible selection
            while (remaining > defaults::INTEGRATION_EPSILON_SECONDS) {
                double h = compute_step_size(current_epoch, current_state);
                h = std::min(h, remaining);  // Don't overshoot target
                auto [new_state, h_used] = adaptive_step(current_epoch, current_state, h * direction, remaining);
                current_state = new_state;
                double abs_h = std::abs(h_used);
                elapsed += h_used;
                current_epoch = start_epoch + elapsed;
                remaining -= abs_h;
                update_step_stats(abs_h);
            }
        } else {
            // Fixed stepping (original behavior)
            while (remaining > defaults::INTEGRATION_EPSILON_SECONDS) {
                double dt = std::min(config_.step_size, remaining) * direction;
                current_state = step(current_epoch, current_state, dt);
                elapsed += dt;
                current_epoch = start_epoch + elapsed;
                remaining -= std::abs(dt);
                stats_.steps_taken++;
            }
        }

        return current_state;
    }

    /// Approximate orbital energy per unit mass [J/kg = m²/s²] for diagnostics: kinetic energy
    /// v²/2 plus the point-mass Earth potential -GM/r (the Hamiltonian, i.e. the total energy, of
    /// a pure two-body orbit). It ignores the Earth's oblateness and all other forces, so it is
    /// only a rough drift indicator. `epoch` is unused.
    [[nodiscard]] double hamiltonian(
        const Epoch& epoch,
        const StateVector& state) const
    {
        // Kinetic energy: T = 0.5 * v²
        double T = 0.5 * state.velocity.squaredNorm();

        // Potential energy of a point-mass Earth: V = -GM/r
        double r = state.position.norm();
        double V = -constants::earth::GM / r;

        return T + V;
    }

    /// Counters accumulated by `integrate()`.
    struct Stats {
        size_t steps_taken = 0;
        // Adaptive stepping statistics (filled only when `adaptive` is on): smallest and largest
        // step [s], their sum [s] (for the average), and steps re-taken to make the step
        // choice symmetric.
        double min_step_used = 0.0;
        double max_step_used = 0.0;
        double total_step_time = 0.0;  ///< Sum of all step sizes [s] (for the average)
        size_t symmetry_retakes = 0;
        /// Mean adaptive step [s]; 0 before any step.
        [[nodiscard]] double avg_step_used() const {
            return steps_taken > 0 ? total_step_time / static_cast<double>(steps_taken) : 0.0;
        }
    };

    /// Counters since construction or the last `reset_stats()`.
    [[nodiscard]] const Stats& stats() const { return stats_; }
    /// Zero the counters.
    void reset_stats() { stats_ = Stats{}; }

    /// The integrator's current settings.
    [[nodiscard]] const SymplecticConfig& config() const { return config_; }

    /// Set the fixed step [s] (the reference step when adaptive) used from the next
    /// `integrate()` on.
    /// @throws ConfigError if dt is not a positive finite number.
    void set_step_size(double dt) {
        if (!(std::isfinite(dt) && dt > 0.0)) {
            throw ConfigError("SymplecticIntegrator::set_step_size: the step must be a positive "
                              "number of seconds (got " + std::to_string(dt) + ")");
        }
        config_.step_size = dt;
    }

private:
    ConservativeForce conservative_force_;
    NonConservativeForce non_conservative_force_;
    SymplecticConfig config_;
    Stats stats_;

    // Yoshida coefficients (computed at construction)
    std::vector<double> c_;  // Position (drift) update weights, num_stages_ + 1 of them
    std::vector<double> d_;  // Velocity (kick) update weights, num_stages_ of them
    int num_stages_ = 0;

    // Exponent of the adaptive step rule: the configured value, or 1/order when it is 0.
    [[nodiscard]] double effective_alpha() const {
        if (config_.adaptation_alpha > 0.0) return config_.adaptation_alpha;
        return 1.0 / static_cast<double>(static_cast<int>(config_.order));
    }

    // Adaptive step size from the size of the conservative acceleration:
    //   h(y) = step_size * (a_ref / |a(y)|)^alpha, clamped to [min_step_size, max_step_size],
    // where a_ref = GM/r_ref^2 is the gravitational acceleration at r_ref = 7000 km.
    [[nodiscard]] double compute_step_size(
        const Epoch& epoch,
        const StateVector& state) const
    {
        // Get conservative acceleration magnitude at current state
        Vector3d acc = conservative_force_(epoch, state.position);
        double a_mag = acc.norm();
        if (a_mag < defaults::ACCELERATION_EPSILON) return config_.max_step_size;

        // Reference acceleration: GM/r^2 at a typical LEO altitude (~7000 km)
        const double a_ref = constants::earth::GM
            / (defaults::EARTH_REFERENCE_RADIUS_M * defaults::EARTH_REFERENCE_RADIUS_M);

        double alpha = effective_alpha();
        double h = config_.step_size * std::pow(a_ref / a_mag, alpha);

        // Clamp to [min, max]
        return std::clamp(h, config_.min_step_size, config_.max_step_size);
    }

    // One time-reversible adaptive step. A trial step of h_pred is taken; the step size the rule
    // would choose at its END is computed, and the step is re-taken with the geometric mean of
    // the start and end choices until the two agree within symmetry_tolerance (a step chosen
    // the same way forward and backward keeps the method time-reversible).
    // Returns {new_state, h_used}; max_abs_h caps the step so the target is not overshot.
    [[nodiscard]] std::pair<StateVector, double> adaptive_step(
        const Epoch& epoch,
        const StateVector& state,
        double h_pred,
        double max_abs_h)
    {
        double abs_h = std::abs(h_pred);
        double sign = (h_pred >= 0) ? 1.0 : -1.0;

        for (int iter = 0; iter < config_.max_symmetry_iters; ++iter) {
            // Take trial step with current h
            StateVector trial = step(epoch, state, h_pred);
            Epoch trial_epoch = epoch + h_pred;

            // Compute step size indicator at trial state
            double h_end = compute_step_size(trial_epoch, trial);

            // Geometric mean for time-symmetry
            double h_corr = std::sqrt(abs_h * h_end);
            h_corr = std::clamp(h_corr, config_.min_step_size,
                                std::min(config_.max_step_size, max_abs_h));

            // Check convergence
            double rel_change = std::abs(h_corr - abs_h) / abs_h;
            if (rel_change < config_.symmetry_tolerance) {
                // Converged — accept this step
                return {trial, h_pred};
            }

            // Not converged — retake step with corrected h
            abs_h = h_corr;
            h_pred = h_corr * sign;
            stats_.symmetry_retakes++;
        }

        // Max iterations reached — take step with last h_corr
        StateVector result = step(epoch, state, h_pred);
        return {result, h_pred};
    }

    void update_step_stats(double abs_h) {
        stats_.steps_taken++;
        stats_.total_step_time += abs_h;
        if (stats_.steps_taken == 1) {
            stats_.min_step_used = abs_h;
            stats_.max_step_used = abs_h;
        } else {
            stats_.min_step_used = std::min(stats_.min_step_used, abs_h);
            stats_.max_step_used = std::max(stats_.max_step_used, abs_h);
        }
    }

    void compute_coefficients() {
        switch (config_.order) {
            case SymplecticOrder::LEAPFROG:
                compute_leapfrog_coefficients();
                break;
            case SymplecticOrder::YOSHIDA6:
                compute_yoshida6_coefficients();
                break;
            case SymplecticOrder::YOSHIDA8:
                compute_yoshida8_coefficients();
                break;
        }
    }

    void compute_leapfrog_coefficients() {
        num_stages_ = 1;
        c_ = {0.5, 0.5};
        d_ = {1.0};
    }

    void compute_yoshida6_coefficients() {
        // Yoshida 6th order: 7 stages
        // Reference: Yoshida, H. (1990). "Construction of higher order symplectic integrators."
        // Physics Letters A, 150(5-7), 262-268. DOI: 10.1016/0375-9601(90)90092-3
        // Table 1, Solution A
        num_stages_ = 7;

        // Coefficients from Yoshida (1990), Solution A
        // These satisfy the 6th-order conditions for symplectic integration.
        // Each w is the length (as a fraction of dt) of one leapfrog sub-step; they sum to 1.
        double w1 = -1.17767998417887100694500;
        double w2 =  0.23557321335935813368479;
        double w3 =  0.78451361047755726381950;
        double w0 = 1.0 - 2.0 * (w1 + w2 + w3);

        // Palindromic structure: starts with w3, goes to w0, then mirrors back
        // Order: {w3, w2, w1, w0, w1, w2, w3}
        std::array<double, 7> w = {w3, w2, w1, w0, w1, w2, w3};

        // Chaining leapfrog sub-steps merges adjacent half-drifts: drift weight c[i] is the
        // average of the neighboring sub-step lengths, kick weight d[i] is the sub-step length.
        c_.resize(8);
        d_.resize(7);

        c_[0] = w[0] / 2.0;
        for (int i = 1; i < 7; ++i) {
            c_[i] = (w[i-1] + w[i]) / 2.0;
        }
        c_[7] = w[6] / 2.0;

        for (int i = 0; i < 7; ++i) {
            d_[i] = w[i];
        }
    }

    void compute_yoshida8_coefficients() {
        // Yoshida 8th order: 15 stages
        // Reference: Yoshida, H. (1990). "Construction of higher order symplectic integrators."
        // Physics Letters A, 150(5-7), 262-268. DOI: 10.1016/0375-9601(90)90092-3
        // Table 2, Solution A (8th order)
        num_stages_ = 15;

        // 8th-order coefficients from Yoshida (1990), Solution A
        // These satisfy the 8th-order conditions for symplectic integration
        double W1 = -1.61582374150097;
        double W2 = -2.44699182370524;
        double W3 = -0.00716989419708120;
        double W4 =  2.44002732616735;
        double W5 =  0.157739928123617;
        double W6 =  1.82020630970714;
        double W7 =  1.04242620869991;
        double W0 = 1.0 - 2.0 * (W1 + W2 + W3 + W4 + W5 + W6 + W7);

        // Palindromic structure: {W7, W6, W5, W4, W3, W2, W1, W0, W1, W2, W3, W4, W5, W6, W7}
        std::array<double, 15> w = {W7, W6, W5, W4, W3, W2, W1, W0, W1, W2, W3, W4, W5, W6, W7};

        c_.resize(16);
        d_.resize(15);

        c_[0] = w[0] / 2.0;
        for (int i = 1; i < 15; ++i) {
            c_[i] = (w[i-1] + w[i]) / 2.0;
        }
        c_[15] = w[14] / 2.0;

        for (int i = 0; i < 15; ++i) {
            d_[i] = w[i];
        }
    }

    // Pure symplectic step with conservative forces only.
    [[nodiscard]] StateVector symplectic_step(
        const Epoch& epoch,
        const StateVector& state,
        double dt) const
    {
        Vector3d q = state.position;
        Vector3d p = state.velocity;
        double t_offset = 0.0;  // Cumulative time offset from epoch

        // Alternate position drifts and velocity kicks:
        // drift c[0], kick d[0], drift c[1], ..., kick d[n-1], drift c[n].
        for (int i = 0; i <= num_stages_; ++i) {
            // Position drift
            if (i < static_cast<int>(c_.size())) {
                q += c_[i] * dt * p;
                t_offset += c_[i] * dt;  // Track cumulative time
            }

            // Velocity kick (force evaluation at current time)
            if (i < num_stages_) {
                Epoch t = epoch + t_offset;
                Vector3d acc = conservative_force_(t, q);
                p += d_[i] * dt * acc;
            }
        }

        return StateVector(q, p);
    }

    // Splitting: a pure symplectic step for the conservative forces, then one velocity
    // correction dt * a_nc evaluated at the midpoint of the start and end states.
    [[nodiscard]] StateVector splitting_step(
        const Epoch& epoch,
        const StateVector& state,
        double dt) const
    {
        // First: symplectic step for conservative forces
        StateVector result = symplectic_step(epoch, state, dt);

        // Second: apply non-conservative force correction
        if (non_conservative_force_) {
            // Use midpoint method for drag correction
            Epoch mid_epoch = epoch + dt / 2.0;
            StateVector mid_state(
                (state.position + result.position) / 2.0,
                (state.velocity + result.velocity) / 2.0);

            Vector3d acc_nc = non_conservative_force_(mid_epoch, mid_state);
            result.velocity += dt * acc_nc;
        }

        return result;
    }

    // Slimplectic step (variational integrator for non-conservative systems)
    // Tsang, D., Galley, C.R., Stein, L.C. & Turner, A. (2015).
    // "Slimplectic Integrators: Variational Integrators for General Nonconservative Systems."
    // ApJ Letters, 809(1), L9. DOI: 10.1088/2041-8205/809/1/L9
    [[nodiscard]] StateVector slimplectic_step(
        const Epoch& epoch,
        const StateVector& state,
        double dt) const
    {
        // The slimplectic method extends symplectic integration to forces that change the
        // energy by making a modified action stationary (the action is the time integral of
        // kinetic minus potential energy, whose stationarity gives the equations of motion):
        // S_nc = integral(L dt) + integral(F_nc · dr)

        // For basic 2nd-order (LEAPFROG), take one basic step
        if (config_.order == SymplecticOrder::LEAPFROG) {
            auto [q_new, p_new] = basic_slimplectic_step(epoch, state.position, state.velocity, dt);
            return StateVector(q_new, p_new);
        }

        // For higher orders, compose multiple basic steps with the d_ coefficients
        // Each d_[i] is a weight for a basic 2nd-order step
        // Total: d_[0]*dt + d_[1]*dt + ... + d_[n-1]*dt = dt (coefficients sum to 1)
        Vector3d q = state.position;
        Vector3d p = state.velocity;
        Epoch t = epoch;

        for (int i = 0; i < num_stages_; ++i) {
            double sub_dt = d_[i] * dt;
            auto [qi, pi] = basic_slimplectic_step(t, q, p, sub_dt);
            q = qi;
            p = pi;
            t = t + sub_dt;
        }

        return StateVector(q, p);
    }

    // Basic 2nd-order step: half drift, kick with conservative + non-conservative acceleration
    // at the midpoint (the non-conservative force sees a half-kicked velocity estimate), half
    // drift. Returns {position, velocity}.
    [[nodiscard]] std::pair<Vector3d, Vector3d> basic_slimplectic_step(
        const Epoch& epoch,
        const Vector3d& q,
        const Vector3d& p,
        double dt) const
    {
        // Half-step position
        Vector3d q_half = q + (dt / 2.0) * p;
        Epoch t_half = epoch + dt / 2.0;

        // Forces at midpoint
        Vector3d F_cons = conservative_force_(t_half, q_half);
        Vector3d F_nc = Vector3d::Zero();

        if (non_conservative_force_) {
            Vector3d p_pred = p + (dt / 2.0) * F_cons;
            StateVector state_half(q_half, p_pred);
            F_nc = non_conservative_force_(t_half, state_half);
        }

        // Full velocity update
        Vector3d p_new = p + dt * (F_cons + F_nc);

        // Complete position update
        Vector3d q_new = q_half + (dt / 2.0) * p_new;

        return {q_new, p_new};
    }
};

/// Orbit propagator that runs a `SymplecticIntegrator` on the library's force model.
///
/// It takes and returns states in the Earth-fixed frame ITRF (International Terrestrial
/// Reference Frame; "ECEF" = Earth-centered, Earth-fixed) and integrates internally in GCRS
/// (Geocentric Celestial Reference System), the inertial, non-rotating frame. The force model's
/// conservative part (gravity, tides, third bodies, conservative relativity) drives the
/// symplectic stages; drag, radiation pressure, empirical accelerations and non-conservative
/// relativity go through `config.nc_strategy`. Drag is applied only when a drag environment is
/// set on `force_model()`. Frame conversions need Earth orientation parameters (EOP); with no
/// EOP table loaded they throw `DataCoverageError` unless the EOP policy is `AssumeZero`.
class SymplecticPropagator : public PropagatorBase {
public:
    /// @param initial_epoch      epoch of the start state
    /// @param position_ecef      start position, Earth-fixed (ITRF) [m]
    /// @param velocity_ecef      start velocity, Earth-fixed (ITRF) [m/s], i.e. relative to the
    ///                           rotating Earth
    /// @param ballistic          unused (older callers); drag needs a drag environment
    /// @param weather            unused (kept for older callers)
    /// @param enable_third_body  include the Sun and Moon (Jupiter is always off here)
    /// @param config             symplectic settings
    /// @param frame              propagation frame; GCRS is the only value
    /// @param itrf               ITRF realization the Earth-fixed states are expressed in
    /// @throws DataCoverageError when the start epoch needs EOP that is not available
    SymplecticPropagator(
        const Epoch& initial_epoch,
        const Vector3d& position_ecef,
        const Vector3d& velocity_ecef,
        const BallisticModel& ballistic,
        const WeatherConditions& weather,
        bool enable_third_body = true,
        SymplecticConfig config = {},
        PropagationFrame frame = PropagationFrame::GCRS,
        coords::ITRFRealization itrf = coords::ITRFRealization::ITRF2020)
        : force_model_(std::make_shared<forces::ForceModel>())
        , current_epoch_(initial_epoch)
        , config_(config)
        , frame_(frame)
        , itrf_(itrf)
        , initial_energy_(0.0)
        , ballistic_(ballistic)
        , weather_(weather)
        , enable_third_body_(enable_third_body)
    {
        // Configure force model using shared helper
        detail::configure_force_model(*force_model_, ballistic, weather, enable_third_body, itrf);

        // Convert the initial state to the inertial GCRS frame
        StateVector state_ecef(position_ecef, velocity_ecef);
        current_state_gcrs_ = util::itrf_to_gcrs(state_ecef, initial_epoch, itrf_);

        // Create integrator
        create_integrator();

        // Record initial energy
        initial_energy_ = integrator_->hamiltonian(initial_epoch, current_state_gcrs_);
    }

    /// Propagate from the current epoch to `target_epoch` (either direction); returns the GCRS
    /// state there and makes it the current state.
    [[nodiscard]] StateVector propagate_to(const Epoch& target_epoch) override {
        current_state_gcrs_ = integrator_->integrate(
            current_epoch_, current_state_gcrs_, target_epoch);
        current_epoch_ = target_epoch;
        return current_state_gcrs_;
    }

    /// Propagate to `target_epoch` and return position [m], velocity [m/s] and total
    /// acceleration [m/s²] in the Earth-fixed ITRF frame.
    /// @throws DataCoverageError when the frame conversion needs EOP that is not available
    [[nodiscard]] PropagationResult operator()(const Epoch& target_epoch) override {
        (void)propagate_to(target_epoch);  // Side effect: updates current_state_gcrs_ and current_epoch_

        // Convert to the Earth-fixed frame
        StateVector state_ecef = util::gcrs_to_itrf(current_state_gcrs_, target_epoch, itrf_);

        // Compute acceleration
        Vector3d acc_eci = force_model_->acceleration(target_epoch, current_state_gcrs_);
        Vector3d acc_ecef = util::acceleration_to_itrf(acc_eci, state_ecef, target_epoch);

        return PropagationResult(state_ecef.position, state_ecef.velocity, acc_ecef);
    }

    /// Restart from a new Earth-fixed (ITRF) state at `epoch`: rebuilds the integrator (its
    /// statistics start over) and the reference energy for `energy_drift()`.
    void reset(const Epoch& epoch, const StateVector& state_ecef) override {
        current_epoch_ = epoch;
        current_state_gcrs_ = util::itrf_to_gcrs(state_ecef, epoch, itrf_);
        // Recreate integrator (fresh state)
        create_integrator();
        initial_energy_ = integrator_->hamiltonian(epoch, current_state_gcrs_);
    }

    /// Overwrite the current epoch and GCRS state without any conversion (used to step back).
    void set_state_internal(const Epoch& epoch, const StateVector& state_gcrs) override {
        current_epoch_ = epoch;
        current_state_gcrs_ = state_gcrs;
    }

    /// Two-body energy per unit mass [m²/s²] at construction or the last `reset()`
    /// (see `SymplecticIntegrator::hamiltonian`).
    [[nodiscard]] double initial_energy() const { return initial_energy_; }

    /// Two-body energy per unit mass [m²/s²] of the current state.
    [[nodiscard]] double current_energy() const {
        return integrator_->hamiltonian(current_epoch_, current_state_gcrs_);
    }

    /// Relative energy change since the start: (E_now - E_start) / |E_start|.
    [[nodiscard]] double energy_drift() const override {
        return (current_energy() - initial_energy_) / std::abs(initial_energy_);
    }

    // Accessors - PropagatorBase interface
    [[nodiscard]] StateVector current_state_gcrs() const override { return current_state_gcrs_; }
    [[nodiscard]] StateVector current_state_ecef() const override {
        return util::gcrs_to_itrf(current_state_gcrs_, current_epoch_, itrf_);
    }
    [[nodiscard]] Epoch current_epoch() const override { return current_epoch_; }
    [[nodiscard]] forces::ForceModel& force_model() override { return *force_model_; }
    [[nodiscard]] const forces::ForceModel& force_model() const override { return *force_model_; }
    [[nodiscard]] PropagationFrame propagation_frame() const override { return frame_; }
    [[nodiscard]] coords::ITRFRealization itrf_realization() const override { return itrf_; }

    /// The configured fixed step [s].
    [[nodiscard]] double current_step_size() const override { return config_.step_size; }
    /// Change the fixed step [s]; the next `propagate_to()` already uses it (and `reset()`
    /// keeps it).
    /// @throws ConfigError if dt is not a positive finite number.
    void set_step_size(double dt) override {
        integrator_->set_step_size(dt);
        config_.step_size = dt;
    }

    /// The propagator's copy of the symplectic settings.
    [[nodiscard]] const SymplecticConfig& config() const { return config_; }

private:
    std::shared_ptr<forces::ForceModel> force_model_;
    std::unique_ptr<SymplecticIntegrator> integrator_;
    Epoch current_epoch_;
    StateVector current_state_gcrs_;
    SymplecticConfig config_;
    PropagationFrame frame_;
    coords::ITRFRealization itrf_;
    double initial_energy_;
    // Store construction parameters for reset()
    BallisticModel ballistic_;
    WeatherConditions weather_;
    bool enable_third_body_;

    // (Re)build the integrator. The conservative callback evaluates the force model at zero
    // velocity (those forces do not depend on it); the non-conservative one at the full state.
    void create_integrator() {
        ConservativeForce cons_force = [this](const Epoch& epoch, const Vector3d& pos) {
            auto comps = force_model_->split_acceleration(epoch, StateVector(pos, Vector3d::Zero()));
            return comps.conservative;
        };

        NonConservativeForce nc_force = [this](const Epoch& epoch, const StateVector& state) {
            auto comps = force_model_->split_acceleration(epoch, state);
            return comps.non_conservative;
        };

        integrator_ = std::make_unique<SymplecticIntegrator>(
            std::move(cons_force), std::move(nc_force), config_);
    }
};

/// Build a `SymplecticPropagator` on the heap; same parameters as its constructor.
[[nodiscard]] inline std::unique_ptr<SymplecticPropagator> create_symplectic_propagator(
    const Epoch& initial_epoch,
    const Vector3d& position_ecef,
    const Vector3d& velocity_ecef,
    const BallisticModel& ballistic = BallisticModel(),
    const WeatherConditions& weather = WeatherConditions::quiet(),
    bool enable_third_body = true,
    SymplecticConfig config = {},
    PropagationFrame frame = PropagationFrame::GCRS,
    coords::ITRFRealization itrf = coords::ITRFRealization::ITRF2020)
{
    return std::make_unique<SymplecticPropagator>(
        initial_epoch, position_ecef, velocity_ecef,
        ballistic, weather, enable_third_body, config, frame, itrf);
}

} // namespace propagators
} // namespace ballistic
