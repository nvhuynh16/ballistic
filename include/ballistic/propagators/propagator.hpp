#pragma once

#include "../core/types.hpp"
#include "../core/defaults.hpp"
#include "../time/epoch.hpp"
#include "../coordinates/earth_rotation.hpp"
#include "../coordinates/iau_transforms.hpp"
#include "../ephemeris/jpl_ephemeris.hpp"
#include "../forces/force_model.hpp"
#include "../util/frame_transform.hpp"
#include "rk45.hpp"
#include <memory>

namespace ballistic {
namespace propagators {

// =============================================================================
// Detail: Shared propagator initialization helpers
// =============================================================================

namespace detail {

// Force-model settings shared by RK45Propagator and SymplecticPropagator: Sun and Moon on or
// off together (Jupiter always off), and the ITRF realization (the version of the Earth-fixed
// frame) used for the frame conversions. The BallisticModel and WeatherConditions parameters
// are unused and kept only for older callers: drag is configured solely through a
// DragEnvironment, which the caller sets with fm.set_drag_environment(...).
inline void configure_force_model(
    forces::ForceModel& fm,
    const BallisticModel& /*ballistic*/,
    const WeatherConditions& /*weather*/,
    bool enable_third_body,
    coords::ITRFRealization itrf)
{
    fm.third_body().enable_sun(enable_third_body);
    fm.third_body().enable_moon(enable_third_body);
    fm.third_body().enable_jupiter(false);  // Jupiter effect is very small
    fm.set_itrf_realization(itrf);
}

} // namespace detail

// =============================================================================
/// Common interface of the low-level propagators (`RK45Propagator`, `SymplecticPropagator`);
/// `Trajectory` drives them through it.
///
/// Frames: states are integrated in GCRS (Geocentric Celestial Reference System), the inertial,
/// non-rotating frame centered on the Earth, and exchanged with callers in ITRF (International
/// Terrestrial Reference Frame), the Earth-fixed frame that rotates with the Earth ("ECEF",
/// Earth-centered Earth-fixed). Converting between them needs Earth orientation parameters (EOP);
/// with no EOP table loaded the conversions throw `DataCoverageError` unless the EOP policy is
/// `AssumeZero`. Positions are in meters, velocities in m/s, accelerations in m/s².
class PropagatorBase {
public:
    virtual ~PropagatorBase() = default;

    /// Propagate from the current epoch to `target_epoch`; returns the GCRS state there, which
    /// becomes the current state.
    [[nodiscard]] virtual StateVector propagate_to(const Epoch& target_epoch) = 0;

    /// Propagate to `target_epoch` and return position, velocity and total acceleration in the
    /// Earth-fixed ITRF frame.
    [[nodiscard]] virtual PropagationResult operator()(const Epoch& target_epoch) = 0;

    /// Current state in GCRS.
    [[nodiscard]] virtual StateVector current_state_gcrs() const = 0;
    /// Epoch of the current state.
    [[nodiscard]] virtual Epoch current_epoch() const = 0;

    /// Current state converted to the Earth-fixed ITRF frame.
    [[nodiscard]] virtual StateVector current_state_ecef() const = 0;

    /// Restart from a new Earth-fixed (ITRF) state at `epoch`.
    virtual void reset(const Epoch& epoch, const StateVector& state_ecef) = 0;

    /// The force model, to inspect or change (e.g. `force_model().set_drag_environment(env)`).
    [[nodiscard]] virtual forces::ForceModel& force_model() = 0;
    [[nodiscard]] virtual const forces::ForceModel& force_model() const = 0;

    /// Frame the state is integrated in (always GCRS).
    [[nodiscard]] virtual PropagationFrame propagation_frame() const = 0;

    /// ITRF realization (version of the Earth-fixed frame, e.g. ITRF2014, ITRF2020) that the
    /// Earth-fixed states are expressed in.
    [[nodiscard]] virtual coords::ITRFRealization itrf_realization() const = 0;

    /// Overwrite the current epoch and GCRS state without any conversion (used to step back).
    virtual void set_state_internal(const Epoch& epoch, const StateVector& state_gcrs) = 0;

    /// Step size [s] for the next step.
    /// Adaptive integrators (RK45): the current adaptive step size.
    /// Fixed-step integrators (symplectic): the configured step size.
    [[nodiscard]] virtual double current_step_size() const = 0;

    /// Set the step size [s] for the next step.
    /// Adaptive integrators: the first trial step (the error control may still shrink it).
    /// Fixed-step integrators: the configured step size.
    virtual void set_step_size(double dt) = 0;

    /// Relative energy drift, for diagnostics.
    /// Symplectic integrators: (E_current - E_initial) / |E_initial|.
    /// Non-symplectic integrators (RK45): 0.0 (energy is not tracked).
    [[nodiscard]] virtual double energy_drift() const { return 0.0; }
};

/// Propagator using the adaptive Dormand-Prince Runge-Kutta 4(5) integrator (RK45, see rk45.hpp)
/// on the library's force model. Best accuracy for predictions of about an hour.
class RK45Propagator : public PropagatorBase {
public:
    /// @param initial_epoch      epoch of the start state
    /// @param position_ecef      start position, Earth-fixed (ITRF) [m]
    /// @param velocity_ecef      start velocity, Earth-fixed (ITRF) [m/s], i.e. relative to the
    ///                           rotating Earth
    /// @param ballistic          unused (kept for older callers); drag needs a DragEnvironment
    ///                           set on `force_model()`
    /// @param weather            unused (kept for older callers)
    /// @param enable_third_body  include the Sun and Moon (Jupiter is always off here)
    /// @param config             RK45 step-size and tolerance settings
    /// @param ephemeris          source of Sun/Moon positions; nullptr = the global default (a
    ///                           loaded SPK/JPL file, otherwise the embedded DE440 ephemeris)
    /// @param frame              propagation frame; GCRS (IERS 2010, CIO-based) is the only value
    /// @param itrf               ITRF realization of the Earth-fixed states: ITRF2020 (default);
    ///                           use ITRF2014 for ESA Sentinel data before March 4, 2025
    /// @throws DataCoverageError when the start epoch needs EOP that is not available
    RK45Propagator(
        const Epoch& initial_epoch,
        const Vector3d& position_ecef,
        const Vector3d& velocity_ecef,
        const BallisticModel& ballistic,
        const WeatherConditions& weather,
        bool enable_third_body = true,
        RK45Config config = {},
        std::shared_ptr<ephemeris::EphemerisProvider> ephemeris = nullptr,
        PropagationFrame frame = PropagationFrame::GCRS,
        coords::ITRFRealization itrf = coords::ITRFRealization::ITRF2020)
        : force_model_(ephemeris ?
              std::make_shared<forces::ForceModel>(std::move(ephemeris)) :
              std::make_shared<forces::ForceModel>())
        , current_epoch_(initial_epoch)
        , config_(config)
        , prop_frame_(frame)
        , itrf_realization_(itrf)
    {
        // Configure force model using shared helper
        detail::configure_force_model(*force_model_, ballistic, weather, enable_third_body, itrf);

        // Convert initial state to internal GCRS frame
        StateVector state_ecef(position_ecef, velocity_ecef);
        current_state_gcrs_ = util::itrf_to_gcrs(state_ecef, initial_epoch, itrf);

        // Create integrator
        auto derivative = forces::make_derivative_function(force_model_);
        integrator_ = std::make_unique<RK45Integrator>(std::move(derivative), config_);
    }

    /// See `PropagatorBase::propagate_to`. A target within 1 ns of the current epoch returns the
    /// current state without integrating.
    /// @throws ballistic::Error when RK45 exceeds its evaluation budget (see `RK45Integrator`)
    [[nodiscard]] StateVector propagate_to(const Epoch& target_epoch) override {
        if (std::abs(target_epoch - current_epoch_) < defaults::EPOCH_EQUALITY_SECONDS) {
            return current_state_gcrs_;
        }

        current_state_gcrs_ = integrator_->integrate(
            current_epoch_, current_state_gcrs_, target_epoch);
        current_epoch_ = target_epoch;

        return current_state_gcrs_;
    }

    [[nodiscard]] PropagationResult operator()(const Epoch& target_epoch) override {
        propagate_to(target_epoch);

        StateVector state_ecef = util::gcrs_to_itrf(current_state_gcrs_, target_epoch, itrf_realization_);

        // Acceleration in GCRS, rotated to the Earth-fixed frame
        Vector3d acc_eci = force_model_->acceleration(target_epoch, current_state_gcrs_);
        Vector3d acc_ecef = util::acceleration_to_itrf(acc_eci, state_ecef, target_epoch);

        return PropagationResult(state_ecef.position, state_ecef.velocity, acc_ecef);
    }

    [[nodiscard]] StateVector current_state_gcrs() const override {
        return current_state_gcrs_;
    }

    [[nodiscard]] Epoch current_epoch() const override {
        return current_epoch_;
    }

    [[nodiscard]] StateVector current_state_ecef() const override {
        return util::gcrs_to_itrf(current_state_gcrs_, current_epoch_, itrf_realization_);
    }

    void reset(const Epoch& epoch, const StateVector& state_ecef) override {
        current_epoch_ = epoch;
        current_state_gcrs_ = util::itrf_to_gcrs(state_ecef, epoch, itrf_realization_);

        // Fresh integrator: the step size and statistics start over
        auto derivative = forces::make_derivative_function(force_model_);
        integrator_ = std::make_unique<RK45Integrator>(std::move(derivative), config_);
    }

    [[nodiscard]] forces::ForceModel& force_model() override {
        return *force_model_;
    }

    [[nodiscard]] const forces::ForceModel& force_model() const override {
        return *force_model_;
    }

    /// RK45 counters (force evaluations, accepted/rejected steps, step-size range).
    [[nodiscard]] const IntegrationStats& stats() const {
        return integrator_->stats();
    }

    /// Zero the RK45 counters.
    void reset_stats() {
        integrator_->reset_stats();
    }

    /// The underlying integrator, for direct step control.
    [[nodiscard]] RK45Integrator& integrator() {
        return *integrator_;
    }

    [[nodiscard]] const RK45Integrator& integrator() const {
        return *integrator_;
    }

    void set_state_internal(const Epoch& epoch, const StateVector& state_gcrs) override {
        current_epoch_ = epoch;
        current_state_gcrs_ = state_gcrs;
    }

    [[nodiscard]] PropagationFrame propagation_frame() const override {
        return prop_frame_;
    }

    [[nodiscard]] coords::ITRFRealization itrf_realization() const override {
        return itrf_realization_;
    }

    /// The integrator's next step [s] (see `PropagatorBase::current_step_size`).
    [[nodiscard]] double current_step_size() const override {
        return integrator_->current_step_size();
    }

    /// Next trial step [s], clamped to the configured [min_step_size, max_step_size].
    void set_step_size(double dt) override {
        integrator_->set_step_size(dt);
    }

private:
    std::shared_ptr<forces::ForceModel> force_model_;
    std::unique_ptr<RK45Integrator> integrator_;
    Epoch current_epoch_;
    StateVector current_state_gcrs_;
    RK45Config config_;
    PropagationFrame prop_frame_ = PropagationFrame::GCRS;
    coords::ITRFRealization itrf_realization_ = coords::ITRFRealization::ITRF2020;
};

/// Build an `RK45Propagator` with default RK45 settings; parameters as in its constructor
/// (no satellite or weather is needed: both are unused). nullptr ephemeris = the global default.
/// @throws DataCoverageError when the start epoch needs EOP that is not available
[[nodiscard]] inline std::unique_ptr<RK45Propagator> create_rk45_propagator(
    const Epoch& initial_epoch,
    const Vector3d& position_ecef,
    const Vector3d& velocity_ecef,
    const BallisticModel& ballistic = BallisticModel(),
    const WeatherConditions& weather = WeatherConditions::quiet(),
    bool enable_third_body = true,
    std::shared_ptr<ephemeris::EphemerisProvider> ephemeris = nullptr,
    PropagationFrame frame = PropagationFrame::GCRS,
    coords::ITRFRealization itrf = coords::ITRFRealization::ITRF2020)
{
    return std::make_unique<RK45Propagator>(
        initial_epoch, position_ecef, velocity_ecef,
        ballistic, weather, enable_third_body, RK45Config{}, ephemeris, frame, itrf);
}

/// Same as `create_rk45_propagator` with the frame fixed to GCRS (the only frame), taking the
/// start state in ITRF, e.g. from a Sentinel orbit file. itrf: ITRF2020 (default); use ITRF2014
/// for ESA Sentinel data before March 4, 2025.
/// @throws DataCoverageError when the start epoch needs EOP that is not available
[[nodiscard]] inline std::unique_ptr<RK45Propagator> create_gcrs_propagator(
    const Epoch& initial_epoch,
    const Vector3d& position_itrf,
    const Vector3d& velocity_itrf,
    const BallisticModel& ballistic = BallisticModel(),
    const WeatherConditions& weather = WeatherConditions::quiet(),
    bool enable_third_body = true,
    std::shared_ptr<ephemeris::EphemerisProvider> ephemeris = nullptr,
    coords::ITRFRealization itrf = coords::ITRFRealization::ITRF2020)
{
    return std::make_unique<RK45Propagator>(
        initial_epoch, position_itrf, velocity_itrf,
        ballistic, weather, enable_third_body, RK45Config{}, ephemeris,
        PropagationFrame::GCRS, itrf);
}

/// One-shot propagation: start from an Earth-fixed (ITRF2020) state at `initial_epoch` and
/// return the Earth-fixed position, velocity and acceleration at `target_epoch`, with the
/// default force model (no drag, no radiation pressure). nullptr ephemeris = the global default.
/// @throws DataCoverageError when an epoch needs EOP that is not available
[[nodiscard]] inline PropagationResult propagate(
    const Epoch& initial_epoch,
    const Vector3d& position_ecef,
    const Vector3d& velocity_ecef,
    const Epoch& target_epoch,
    const BallisticModel& ballistic = BallisticModel(),
    const WeatherConditions& weather = WeatherConditions::quiet(),
    bool enable_third_body = true,
    std::shared_ptr<ephemeris::EphemerisProvider> ephemeris = nullptr)
{
    RK45Propagator prop(
        initial_epoch, position_ecef, velocity_ecef,
        ballistic, weather, enable_third_body, RK45Config{}, ephemeris);
    return prop(target_epoch);
}

} // namespace propagators
} // namespace ballistic
