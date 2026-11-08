#pragma once

#include "../core/types.hpp"
#include "../core/defaults.hpp"
#include "../core/diagnostics.hpp"
#include "../core/errors.hpp"
#include "../time/epoch.hpp"
#include <functional>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace ballistic {
namespace propagators {

// Adaptive Runge-Kutta integrator for orbit propagation.
//
// RK45 here is the Dormand-Prince 4(5) method: each step computes a 5th-order and an embedded
// 4th-order solution from the same 7 force evaluations; their difference estimates the error,
// and the step size grows or shrinks to keep that estimate within the tolerances. It gives the
// best accuracy for predictions of about an hour; for many hours or days a symplectic method
// (symplectic.hpp) keeps the energy error bounded.

/// Right-hand side of the equations of motion: given (epoch, state = {position [m],
/// velocity [m/s]}) return the time derivative {velocity [m/s], acceleration [m/s²]}.
using DerivativeFunction = std::function<StateVector(const Epoch&, const StateVector&)>;

/// Step-size and tolerance settings for `RK45Integrator` (defaults in core/defaults.hpp).
/// A step is accepted when its error estimate is below
/// absolute_tolerance + relative_tolerance * |value|, checked separately for position [m]
/// and velocity [m/s].
struct RK45Config {
    /// First trial step [s] (default 60), smallest step [s] (0.001), largest step [s] (3600).
    double initial_step_size = defaults::RK45_INITIAL_STEP_SIZE;
    double min_step_size = defaults::RK45_MIN_STEP_SIZE;
    double max_step_size = defaults::RK45_MAX_STEP_SIZE;
    double relative_tolerance = defaults::RK45_RELATIVE_TOLERANCE;
    double absolute_tolerance = defaults::RK45_ABSOLUTE_TOLERANCE;
    /// Evaluation budget: `integrate()` throws once the force evaluations exceed 7x this.
    size_t max_iterations = defaults::RK45_MAX_ITERATIONS;

    // Each new step is the old one times safety_factor * (1/error)^(1/5), and that factor
    // is clamped to [min_scale, max_scale].
    double safety_factor = defaults::RK45_SAFETY_FACTOR;
    double min_scale = defaults::RK45_MIN_SCALE;
    double max_scale = defaults::RK45_MAX_SCALE;
};

/// Counters accumulated by `RK45Integrator`: calls of the derivative function (7 per step
/// attempt), accepted steps (including forced ones), attempts that failed the error test, and
/// the smallest and largest accepted step [s].
struct IntegrationStats {
    size_t function_evaluations = 0;
    size_t steps_taken = 0;
    size_t rejected_steps = 0;
    double min_step_size = std::numeric_limits<double>::max();
    double max_step_size = 0.0;
};

/// Dormand-Prince coefficients (the "Butcher tableau": stage times c, stage weights a,
/// solution weights b). Dormand, J.R. & Prince, P.J. (1980). "A family of embedded Runge-Kutta
/// formulae." J. Comp. Appl. Math., 6(1), 19-26. DOI: 10.1016/0771-050X(80)90013-3
/// The RK5(4)7M pair. b_i = a_{7,i}: the 7th stage is evaluated at the new solution, so it
/// could be reused as the next step's first stage ("first same as last", FSAL).
namespace dopri5 {
    // Time coefficients (c)
    constexpr double c2 = 1.0 / 5.0;
    constexpr double c3 = 3.0 / 10.0;
    constexpr double c4 = 4.0 / 5.0;
    constexpr double c5 = 8.0 / 9.0;
    constexpr double c6 = 1.0;
    constexpr double c7 = 1.0;

    // Stage coefficients (a)
    constexpr double a21 = 1.0 / 5.0;

    constexpr double a31 = 3.0 / 40.0;
    constexpr double a32 = 9.0 / 40.0;

    constexpr double a41 = 44.0 / 45.0;
    constexpr double a42 = -56.0 / 15.0;
    constexpr double a43 = 32.0 / 9.0;

    constexpr double a51 = 19372.0 / 6561.0;
    constexpr double a52 = -25360.0 / 2187.0;
    constexpr double a53 = 64448.0 / 6561.0;
    constexpr double a54 = -212.0 / 729.0;

    constexpr double a61 = 9017.0 / 3168.0;
    constexpr double a62 = -355.0 / 33.0;
    constexpr double a63 = 46732.0 / 5247.0;
    constexpr double a64 = 49.0 / 176.0;
    constexpr double a65 = -5103.0 / 18656.0;

    constexpr double a71 = 35.0 / 384.0;
    constexpr double a72 = 0.0;
    constexpr double a73 = 500.0 / 1113.0;
    constexpr double a74 = 125.0 / 192.0;
    constexpr double a75 = -2187.0 / 6784.0;
    constexpr double a76 = 11.0 / 84.0;

    // 5th order solution weights (b)
    constexpr double b1 = 35.0 / 384.0;
    constexpr double b2 = 0.0;
    constexpr double b3 = 500.0 / 1113.0;
    constexpr double b4 = 125.0 / 192.0;
    constexpr double b5 = -2187.0 / 6784.0;
    constexpr double b6 = 11.0 / 84.0;
    constexpr double b7 = 0.0;

    // 4th order solution weights for error estimation (b*)
    constexpr double bs1 = 5179.0 / 57600.0;
    constexpr double bs2 = 0.0;
    constexpr double bs3 = 7571.0 / 16695.0;
    constexpr double bs4 = 393.0 / 640.0;
    constexpr double bs5 = -92097.0 / 339200.0;
    constexpr double bs6 = 187.0 / 2100.0;
    constexpr double bs7 = 1.0 / 40.0;

    // Error coefficients (b - b*)
    constexpr double e1 = b1 - bs1;
    constexpr double e2 = b2 - bs2;
    constexpr double e3 = b3 - bs3;
    constexpr double e4 = b4 - bs4;
    constexpr double e5 = b5 - bs5;
    constexpr double e6 = b6 - bs6;
    constexpr double e7 = b7 - bs7;
}

/// Adaptive Dormand-Prince RK4(5) integrator for a state {position, velocity}. Frame-agnostic:
/// the derivative function decides the frame. Exceptions thrown by it pass through.
class RK45Integrator {
public:
    /// @param derivative  equations of motion (see `DerivativeFunction`)
    /// @param config      step-size and tolerance settings
    explicit RK45Integrator(DerivativeFunction derivative, RK45Config config = {})
        : derivative_(std::move(derivative))
        , config_(config)
        , step_size_(config.initial_step_size)
    {}

    /// Outcome of one step attempt: the 5th-order solution and its epoch (epoch + dt), the error
    /// relative to the tolerance (<= 1 means accepted), whether the error test passed, and the
    /// suggested magnitude of the next step [s], within [min_step_size, max_step_size].
    struct StepResult {
        StateVector state;
        Epoch epoch;
        double error_estimate;
        bool accepted;
        double next_step_size;
    };

    /// Attempt one step of `dt` seconds (negative = backward) from `state` at `epoch`. The
    /// result is returned whether or not the error test passed; the caller decides what to do.
    /// Updates the evaluation counter only, not the stored step size.
    [[nodiscard]] StepResult step(const Epoch& epoch, const StateVector& state, double dt) {
        using namespace dopri5;

        // Compute the 7 stages
        StateVector k1 = derivative_(epoch, state);
        stats_.function_evaluations++;

        StateVector y2 = state + dt * (a21 * k1);
        StateVector k2 = derivative_(epoch + c2 * dt, y2);
        stats_.function_evaluations++;

        StateVector y3 = state + dt * (a31 * k1 + a32 * k2);
        StateVector k3 = derivative_(epoch + c3 * dt, y3);
        stats_.function_evaluations++;

        StateVector y4 = state + dt * (a41 * k1 + a42 * k2 + a43 * k3);
        StateVector k4 = derivative_(epoch + c4 * dt, y4);
        stats_.function_evaluations++;

        StateVector y5 = state + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4);
        StateVector k5 = derivative_(epoch + c5 * dt, y5);
        stats_.function_evaluations++;

        StateVector y6 = state + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5);
        StateVector k6 = derivative_(epoch + c6 * dt, y6);
        stats_.function_evaluations++;

        // 5th order solution
        StateVector y_new = state + dt * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6);

        StateVector k7 = derivative_(epoch + dt, y_new);
        stats_.function_evaluations++;

        // Error estimate (difference between 4th and 5th order)
        StateVector error = dt * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7);

        // Error relative to the mixed absolute/relative tolerance, worst of position and velocity
        double err_pos = error.position.norm();
        double err_vel = error.velocity.norm();

        double scale_pos = config_.absolute_tolerance +
            config_.relative_tolerance * std::max(state.position.norm(), y_new.position.norm());
        double scale_vel = config_.absolute_tolerance +
            config_.relative_tolerance * std::max(state.velocity.norm(), y_new.velocity.norm());

        double error_norm = std::max(err_pos / scale_pos, err_vel / scale_vel);

        // Determine if step is accepted
        bool accepted = (error_norm <= 1.0);

        // Compute new step size
        double scale;
        if (error_norm < defaults::RK45_ERROR_FLOOR) {
            scale = config_.max_scale;
        } else {
            // I-controller for step size: Hairer, Norsett & Wanner (1993),
            // "Solving ODEs I", Springer, Eq. II.4.13. Exponent = 1/(p+1), p=4.
            scale = config_.safety_factor * std::pow(1.0 / error_norm, defaults::RK45_STEP_CONTROLLER_EXPONENT);
        }

        scale = std::clamp(scale, config_.min_scale, config_.max_scale);
        // Use magnitude of dt for step size adaptation (direction handled in integrate())
        double new_dt = std::abs(dt) * scale;
        new_dt = std::clamp(new_dt, config_.min_step_size, config_.max_step_size);

        return {y_new, epoch + dt, error_norm, accepted, new_dt};
    }

    /// Integrate from `start_state` at `start_epoch` to `target_epoch` (either direction) and
    /// return the state there; the last step is shortened to land on the target. Starts from the
    /// stored step size and leaves the last suggested step there for the next call.
    /// A step that fails the error test at the minimum step size is accepted anyway and
    /// reported as a ForcedStep diagnostic (which raises in strict mode).
    /// @throws ballistic::Error when the force evaluations exceed 7 x `max_iterations`
    [[nodiscard]] StateVector integrate(
        const Epoch& start_epoch,
        const StateVector& start_state,
        const Epoch& target_epoch)
    {
        Epoch current_epoch = start_epoch;
        StateVector current_state = start_state;
        double remaining = target_epoch - start_epoch;  // seconds
        double elapsed = 0.0;  // exact accumulator: labels = start_epoch + elapsed
                               // (chaining current_epoch + dt would add the rounding of
                               // every Epoch addition to the time labels)

        // Handle integration direction
        double direction = (remaining >= 0) ? 1.0 : -1.0;
        remaining = std::abs(remaining);

        while (remaining > defaults::INTEGRATION_EPSILON_SECONDS) {
            // Limit step size to remaining time
            double dt = std::min(step_size_, remaining) * direction;

            auto result = step(current_epoch, current_state, dt);

            if (result.accepted) {
                current_state = result.state;
                elapsed += dt;
                current_epoch = start_epoch + elapsed;
                remaining -= std::abs(dt);
                stats_.steps_taken++;
                stats_.min_step_size = std::min(stats_.min_step_size, std::abs(dt));
                stats_.max_step_size = std::max(stats_.max_step_size, std::abs(dt));
            } else {
                stats_.rejected_steps++;
                // Force-accept at min step size — integrator can't do better.
                // Reported (once per process; strict mode raises): the state
                // is no longer within the tolerance.
                if (result.next_step_size <= config_.min_step_size) {
                    diagnostics::report(
                        diagnostics::Category::ForcedStep,
                        "RK45: accepted a step that failed its error test (error norm " +
                            std::to_string(result.error_estimate) + ") at the minimum step size " +
                            std::to_string(config_.min_step_size) + " s; the state from there on is "
                            "outside the integrator's tolerance (a discontinuous or singular force?)",
                        "rk45-forced-step");
                    current_state = result.state;
                    elapsed += dt;
                    current_epoch = start_epoch + elapsed;
                    remaining -= std::abs(dt);
                    stats_.steps_taken++;
                }
            }

            step_size_ = result.next_step_size;

            // Guard against a step size that collapses and never finishes
            if (stats_.function_evaluations > config_.max_iterations * defaults::RK45_EVAL_MULTIPLIER) {
                throw Error("RK45: gave up after " + std::to_string(stats_.function_evaluations) +
                            " force evaluations (7 x RK45Config::max_iterations = " +
                            std::to_string(config_.max_iterations) + ") with " +
                            std::to_string(remaining) + " s still to go: the step size "
                            "collapsed, which usually means a discontinuous or singular force. "
                            "Check the force model near this epoch, or raise max_iterations "
                            "for a very long span.");
            }
        }

        return current_state;
    }

    /// Magnitude of the next trial step [s].
    [[nodiscard]] double current_step_size() const noexcept {
        return step_size_;
    }

    /// Set the next trial step [s], clamped to [min_step_size, max_step_size].
    void set_step_size(double dt) noexcept {
        step_size_ = std::clamp(dt, config_.min_step_size, config_.max_step_size);
    }

    /// Counters since construction or the last `reset_stats()`.
    [[nodiscard]] const IntegrationStats& stats() const noexcept {
        return stats_;
    }

    /// Zero the counters.
    void reset_stats() noexcept {
        stats_ = IntegrationStats{};
    }

    /// The settings this integrator was built with.
    [[nodiscard]] const RK45Config& config() const noexcept {
        return config_;
    }

private:
    DerivativeFunction derivative_;
    RK45Config config_;
    double step_size_;
    IntegrationStats stats_;
};

/// One-shot integration with a fresh `RK45Integrator`; see `RK45Integrator::integrate`.
/// @throws ballistic::Error when the evaluation budget is exceeded
[[nodiscard]] inline StateVector integrate_rk45(
    const Epoch& start_epoch,
    const StateVector& start_state,
    const Epoch& target_epoch,
    DerivativeFunction derivative,
    RK45Config config = {})
{
    RK45Integrator integrator(std::move(derivative), config);
    return integrator.integrate(start_epoch, start_state, target_epoch);
}

} // namespace propagators
} // namespace ballistic
