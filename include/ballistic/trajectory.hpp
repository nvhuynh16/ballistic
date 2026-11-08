#pragma once

// =============================================================================
// Trajectory: the main entry point for predicting where a satellite will be
// =============================================================================
//
// A Trajectory takes a starting position and velocity at a given time and
// integrates the equations of motion forward (or backward) in time under the
// forces you configure: Earth's gravity field, the Sun, Moon and Jupiter,
// tides, relativity and, optionally, solar radiation pressure and
// atmospheric drag. It keeps a history of the states it has computed so you
// can interpolate between them later (trajectory_interpolator.hpp).
//
// Terms used in this file:
//   - ITRF (International Terrestrial Reference Frame): the Earth-fixed,
//     rotating frame. Positions and velocities "in ITRF" or "ECEF" (Earth-
//     centered, Earth-fixed) move with the ground; a velocity there is
//     relative to the rotating Earth.
//   - GCRS (Geocentric Celestial Reference System): the inertial,
//     non-rotating frame centered on the Earth. The equations of motion are
//     integrated in GCRS; inputs and outputs are converted to ITRF.
//   - LEO: low Earth orbit (roughly 200-2000 km altitude).
//   - RK45: the Dormand-Prince Runge-Kutta 4(5) integrator, which picks its
//     own step size from an error estimate. Yoshida 6/8: fixed-step
//     "symplectic" integrators, which keep the orbit's energy from drifting
//     over long runs.
//   - EOP (Earth orientation parameters): measured corrections to the
//     Earth's rotation (UT1-UTC, polar motion) from the IERS tables.
//   - SRP: solar radiation pressure, the push of sunlight on the satellite.
//
// The defaults are tuned for LEO satellites; their measured accuracy against
// ESA Sentinel precise orbits and other references is recorded in
// docs/validation_claims.md.
//
// Usage:
//   Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
//   Vector3d pos(6778137.0, 0.0, 0.0);   // meters, Earth-fixed
//   Vector3d vel(0.0, 7175.0, 0.0);      // m/s, Earth-fixed: circular 7669 m/s
//                                        // minus Earth rotation 494 m/s
//
//   Trajectory traj(InitialState::from_itrf(t0, pos, vel));
//   auto result = traj.propagate(t0 + 3600.0);  // 1 hour
//
// =============================================================================

#include "propagators/propagator.hpp"
#include "propagators/symplectic.hpp"
#include "core/types.hpp"
#include "core/defaults.hpp"
#include "core/diagnostics.hpp"
#include "core/errors.hpp"
#include "util/frame_transform.hpp"
#include "time/epoch.hpp"
#include "time/eop_data.hpp"
#include "atmosphere/drag_environment.hpp"
#include "atmosphere/space_weather.hpp"
#include "forces/empirical_acceleration.hpp"
#include "events.hpp"
#include "initial_state.hpp"
#include "presets/tle_grade.hpp"
#include "coordinates/geodetic.hpp"
#include <limits>
#include <memory>
#include <optional>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ballistic {

// =============================================================================
// History and Step Result Structures
// =============================================================================

/// One stored state of a Trajectory: the time, the Earth-fixed position,
/// velocity and acceleration, and the step that led to it.
struct HistoryEntry {
    Epoch epoch;              // Time of the state
    double t_rel_s = 0.0;     // Seconds since Trajectory::label_ref(), accumulated exactly.
                              // The interpolator uses this as its time axis: a smooth,
                              // exact time base matters to estimators that differentiate
                              // positions, where even microsecond-level jitter between
                              // nodes would be amplified.
    Vector3d position;        // Position in ECEF (meters)
    Vector3d velocity;        // Velocity in ECEF (m/s)
    Vector3d acceleration;    // Acceleration in ECEF (m/s^2)
    double step_size_used;    // Actual step size taken (seconds)

    HistoryEntry() = default;

    HistoryEntry(const Epoch& t, double t_rel, const Vector3d& pos,
                 const Vector3d& vel, const Vector3d& acc, double dt)
        : epoch(t), t_rel_s(t_rel), position(pos), velocity(vel),
          acceleration(acc), step_size_used(dt) {}
};

/// What Trajectory::step() and step_backward() return: the new state (ECEF)
/// and the step sizes used and suggested.
struct StepResult {
    Epoch epoch;              // New epoch after stepping
    Vector3d position;        // New position (ECEF)
    Vector3d velocity;        // New velocity (ECEF)
    Vector3d acceleration;    // Acceleration at new state (ECEF)
    double step_size_taken;   // Actual step size used (seconds)
    double next_step_size;    // Suggested step size for next step
    size_t history_index;     // Index in history
    std::optional<ForceBreakdown> forces;  // Force breakdown (when report_forces=true)
};

/// Result of Trajectory::propagate_until: where it stopped, and why.
struct PropagateUntilResult {
    Epoch epoch;                                   // where the propagation ended
    StateVector state_ecef;                        // the state there (ITRF)
    std::optional<events::Occurrence> stopped_by;  // the terminal event; nullopt = reached the target
    std::vector<events::Occurrence> log;           // every located event, in time order
    /// True when the propagation reached its target time (no terminal event fired).
    [[nodiscard]] bool reached_target() const noexcept { return !stopped_by.has_value(); }
};

// =============================================================================
// Integrator Type Selection
// =============================================================================

/// Which numerical integrator steps the equations of motion.
enum class IntegratorType {
    RK45,       // Dormand-Prince adaptive (default) - best for short-term accuracy
    YOSHIDA6,   // Yoshida 6th order symplectic (recommended for long-term)
    YOSHIDA8    // Yoshida 8th order symplectic
};

/// Maps an IntegratorType to the symplectic propagator's order enum
/// (RK45, which is not symplectic, maps to YOSHIDA6).
[[nodiscard]] inline propagators::SymplecticOrder to_symplectic_order(IntegratorType type) noexcept {
    switch (type) {
        case IntegratorType::YOSHIDA6: return propagators::SymplecticOrder::YOSHIDA6;
        case IntegratorType::YOSHIDA8: return propagators::SymplecticOrder::YOSHIDA8;
        default: return propagators::SymplecticOrder::YOSHIDA6;  // Default for RK45
    }
}

/// Options for the Yoshida symplectic integrators (ignored by RK45).
struct SymplecticOptions {
    double step_size = 30.0;  // Fixed step size in seconds (reference step for adaptive)
    // How forces that do not conserve energy (drag, radiation pressure) are
    // folded into the energy-conserving scheme.
    propagators::NonConservativeStrategy nc_strategy =
        propagators::NonConservativeStrategy::SLIMPLECTIC;

    // Time-reversible adaptive stepping (Hairer, Lubich & Wanner 2006, §VIII.3):
    // the step varies along the orbit but stays symmetric in time, which keeps
    // the long-run energy behavior of a fixed-step symplectic method.
    bool adaptive = false;
    double min_step_size = 1.0;       // Minimum step (seconds)
    double max_step_size = 120.0;     // Maximum step (seconds)
    std::optional<double> adaptation_alpha;  // nullopt = auto (1/order)
    double symmetry_tolerance = 0.1;  // Relative tolerance for predictor-corrector
};

/// Options for the RK45 (Dormand-Prince) integrator. The step adapts between
/// these bounds; a Trajectory further caps the maximum at 1/20 of the orbital
/// period of the starting orbit.
struct Rk45Options {
    double initial_step_size = defaults::RK45_INITIAL_STEP_SIZE;  // [s]
    double max_step_size = defaults::RK45_MAX_STEP_SIZE;          // [s]
};

/// Solar radiation pressure (SRP) settings. Assigning one to
/// TrajectoryConfig::srp turns the radiation forces on (sunlight, plus the
/// sunlight reflected by the Earth, "albedo", and the Earth's own infrared
/// glow); std::nullopt = radiation off. A Trajectory throws ConfigError if
/// mass_kg is not > 0 or area_m2 is not >= 0 (both are NaN until you set them).
struct SrpConfig {
    // No defaults: a generic area-to-mass ratio (A/m) is 20x wrong for some
    // satellites, so the caller must supply both.
    double mass_kg = std::numeric_limits<double>::quiet_NaN();  // Satellite mass; used by SRP/albedo/IR
                              // only (drag mass enters via the DragEnvironment's beta)
    double area_m2 = std::numeric_limits<double>::quiet_NaN();  // Sun-facing cross-section
    double cr = 1.3;          // Radiation pressure coefficient Cr (1.0 = perfect
                              // absorber, ~1.3 typical; a = P·(A/m)·Cr·ŝ)
    bool albedo = true;       // Earth-albedo component
    bool earth_ir = true;     // Earth-infrared component
};

// =============================================================================
// Trajectory Configuration
// =============================================================================

/// Everything that controls a Trajectory: which forces act, how precise the
/// gravity field is, and which integrator is used.
///
/// The defaults are the best general settings we measured: integration in
/// GCRS, the EGM2008 gravity model to degree 70 (degrees 100-160 were no
/// better on Sentinel data; docs/configuration.md), Sun/Moon/Jupiter gravity,
/// solid tides, relativity and RK45. Validated numbers live in docs/validation_claims.md.
///
/// Radiation pressure and drag are OFF by default only because they need
/// satellite-specific inputs (mass, areas, radiation coefficient Cr,
/// ballistic coefficient β = m/(Cd·A), with Cd the drag coefficient and A
/// the frontal area) and, for drag, space-weather data (solar flux and
/// geomagnetic indices). Turn both on for real satellites, including when
/// the start state comes from a precise orbit product (POE, ESA's precise
/// orbit ephemeris, or an SP3 file, the standard precise-orbit text format):
/// such a product fixes only the START state, and the forces after it must
/// still be modeled, so nothing is counted twice. Measured over all of 2025
/// on 7 Sentinels (SENTINEL-YEAR-2025): radiation pressure and drag both
/// lower the 1-hour error.
///
/// Optional forces follow one idiom — presence is the switch:
//   config.srp      = SrpConfig{...};      // radiation (SRP + albedo + IR) on
//   config.drag_env = make_drag_environment(...);  // atmospheric drag on
struct TrajectoryConfig {
    // Maximum degree of the EGM2008 spherical-harmonic gravity field: higher =
    // finer detail of the Earth's lumpy gravity, slower. 70 is enough for LEO,
    // 4-8 for higher orbits (MEO: medium Earth orbit, e.g. GPS; GEO:
    // geostationary; HEO: highly elliptical orbit). ConfigError above the
    // loaded model's degree (360 embedded).
    int gravity_degree = 70;
    bool enable_third_body = true;        // Sun/Moon/Jupiter perturbations (critical for accuracy)
    // General-relativity corrections to Earth's gravity (Schwarzschild term
    // and Lense-Thirring frame dragging).
    bool enable_relativistic = true;
    // Solid-Earth tides and ocean tides: the Sun and Moon deform the Earth,
    // which changes its gravity field slightly (IERS 2010 Conventions,
    // Steps 1 and 2).
    bool enable_solid_tides = true;
    // Degree of the FES2004 ocean-tide model (0 = off; otherwise >= 2, else
    // ConfigError). It acts only once gravity::OceanTideModel is loaded
    // (Python: ballistic.load()); a Trajectory warns when it is not.
    int ocean_tide_degree = 50;
    // Which realization (version) of the Earth-fixed frame the ITRF inputs
    // and outputs are in. Match your data: ITRF2020 for current products,
    // ITRF2014 for IGS data before Nov 2022.
    coords::ITRFRealization itrf = coords::ITRFRealization::ITRF2020;

    // Integrator selection
    IntegratorType integrator = IntegratorType::RK45;  // RK45 for short-term accuracy, YOSHIDA6 for long-term
    Rk45Options rk45 = {};              // Options for RK45 (ignored for symplectic integrators)
    SymplecticOptions symplectic = {};  // Options for symplectic integrators (ignored if integrator is RK45)

    // Solar radiation pressure: assign an SrpConfig to enable (nullopt = off;
    // it needs the satellite's mass and area — see the note above).
    std::optional<SrpConfig> srp;

    // Atmospheric drag: air density, wind and the satellite's ballistic
    // coefficient, bundled as a DragEnvironment. Set it (usually with
    // make_drag_environment) to turn drag on; nullptr = drag off (it needs
    // the satellite's β and the space weather — see the note above).
    std::shared_ptr<const atmosphere::DragEnvironment> drag_env;

    // Empirical acceleration (EA): a small correction acceleration, usually
    // fitted to observations, added to the modeled forces. Its time keys
    // are seconds since ea_reference_epoch (default: the start epoch).
    std::shared_ptr<forces::EmpiricalAcceleration> empirical_accel;
    // Epoch that the EA's time keys count from. Setting it without a non-zero
    // empirical_accel throws std::invalid_argument at construction.
    std::optional<Epoch> ea_reference_epoch;

    // Force reporting (diagnostic)
    bool report_forces = false;           // When true, compute and return all force components in results
    bool include_disabled_forces = false; // When true, also compute forces that are disabled (for analysis)
};

// =============================================================================
// Trajectory - Primary Propagator
// =============================================================================
//
// Uses:
//   - EGM2008 gravity field, degree 70 by default, summed with the Clenshaw
//     recurrence on column-major coefficient storage
//   - Double precision (64-bit)
//   - RK4(5) Dormand-Prince adaptive integrator (default) or Yoshida symplectic
//   - IAU 2006/2000A precession-nutation model for the GCRS <-> ITRF rotation
//   - Sun/Moon/Jupiter gravity ("third-body" perturbations), positions from
//     the embedded JPL DE440 ephemeris
//   - Schwarzschild + Lense-Thirring relativistic corrections
//
// Validated accuracy (1-hour root-mean-square error per orbit regime):
// docs/validation_claims.md.
//
// Integrator Selection:
//   - RK45 (default): Best for short-term accuracy (1-hour predictions)
//   - YOSHIDA6: Recommended for long-term propagation (energy-preserving)
//   - YOSHIDA8: Higher-order symplectic (minimal improvement over Y6)
//
// Restarting from a recorded state: a new Trajectory built from a history
// entry (InitialState::from_itrf(entry.epoch, entry.position, entry.velocity))
// starts from the same inertial state to ~1e-9 m (the GCRS <-> ITRF conversion
// depends on the epoch alone). It follows the same path only if it takes the
// same steps: give it the step size the original was about to take
// (config.rk45.initial_step_size = the StepResult::next_step_size returned
// with that entry), and the two agree to ~1e-6 m (the step sizes still differ
// by ~1e-7 s, as the step control responds to rounding). With the default
// first step the steps differ and so does the RK45 truncation error: measured
// ~1 mm after an hour in LEO with the default tolerances.
// =============================================================================

class Trajectory {
public:
    /// Start from a checked state whose frame is named (initial_state.hpp:
    /// InitialState::from_itrf / from_gcrs / from_j2000 / from_keplerian).
    /// The recommended constructor. Throws ConfigError for an invalid config
    /// (e.g. gravity_degree above the loaded model, an incomplete SrpConfig)
    /// and DataCoverageError when the state's epoch lacks required data.
    explicit Trajectory(const InitialState& state, const TrajectoryConfig& config = {})
        : Trajectory(state.epoch(), state.itrf_state(config.itrf).position,
                     state.itrf_state(config.itrf).velocity, config) {}

    /// Start from an Earth-fixed state at `initial_epoch`: position [m] and
    /// velocity [m/s] in ITRF (realization config.itrf), the velocity being
    /// relative to the rotating Earth. Only units are checked here
    /// (non-finite values, kilometers passed as meters; InputPlausibilityError);
    /// InitialState also checks the orbit and catches an inertial velocity
    /// passed as Earth-fixed.
    Trajectory(
        const Epoch& initial_epoch,
        const Vector3d& pos_itrf,
        const Vector3d& vel_itrf,
        const TrajectoryConfig& config = {})
        : Trajectory(checked_units(pos_itrf, vel_itrf), initial_epoch, pos_itrf, vel_itrf, config) {}

private:
    struct UnitsChecked {};
    static UnitsChecked checked_units(const Vector3d& r, const Vector3d& v) {
        state_checks::check_units(r, v);
        // BALLISTIC_AUDIT_START_STATES=1: run the full InitialState checks
        // here too and print what they would say (finds callers that pass an
        // inertial velocity as an Earth-fixed one).
        static const bool audit = [] {
            const char* e = std::getenv("BALLISTIC_AUDIT_START_STATES");
            return e && e[0] && e[0] != '0';
        }();
        if (audit) {
            try {
                state_checks::check_plausibility(r, v, StateFrame::ITRF, {});
            } catch (const std::exception& ex) {
                std::fprintf(stderr, "[start-state audit] %s\n", ex.what());
            }
        }
        return {};
    }
    Trajectory(UnitsChecked,
        const Epoch& initial_epoch,
        const Vector3d& position_ecef,
        const Vector3d& velocity_ecef,
        const TrajectoryConfig& config)
        : config_(config)
        , integrator_type_(config.integrator)
        , label_ref_(initial_epoch)
    {
        if (config.ea_reference_epoch &&
            (!config.empirical_accel || config.empirical_accel->is_zero())) {
            throw std::invalid_argument(
                "TrajectoryConfig::ea_reference_epoch is set but empirical_accel "
                "is null/zero — the reference epoch only keys EA lookups");
        }

        // The BallisticModel values are placeholders and unused: drag reads
        // β from the DragEnvironment, and the radiation-pressure mass is set
        // in configure_radiation().
        BallisticModel model(defaults::PLACEHOLDER_BETA, defaults::PLACEHOLDER_MASS, 0.0);

        // Placeholder weather required by the propagator constructors; drag
        // takes its space weather from the DragEnvironment instead.
        WeatherConditions weather = WeatherConditions::quiet();

        if (is_symplectic()) {
            // Create symplectic propagator
            propagators::SymplecticConfig symp_config;
            symp_config.step_size = config.symplectic.step_size;
            symp_config.nc_strategy = config.symplectic.nc_strategy;
            symp_config.order = map_integrator_to_order(config.integrator);
            symp_config.adaptive = config.symplectic.adaptive;
            symp_config.min_step_size = config.symplectic.min_step_size;
            symp_config.max_step_size = config.symplectic.max_step_size;
            symp_config.adaptation_alpha =
                config.symplectic.adaptation_alpha.value_or(0.0);  // propagator: 0 = auto
            symp_config.symmetry_tolerance = config.symplectic.symmetry_tolerance;

            propagator_ = std::make_unique<propagators::SymplecticPropagator>(
                initial_epoch, position_ecef, velocity_ecef, model, weather,
                config.enable_third_body, symp_config, PropagationFrame::GCRS, config.itrf
            );
        } else {
            // Create the RK45 propagator with GCRS frame
            propagators::RK45Config rk45_cfg;
            rk45_cfg.initial_step_size = config.rk45.initial_step_size;
            // At most a twentieth of the orbital period: a step much longer
            // than that can step over a perigee, where the embedded error
            // estimate is not to be trusted (bound orbits only).
            rk45_cfg.max_step_size = std::min(
                config.rk45.max_step_size,
                period_step_cap(initial_epoch, position_ecef, velocity_ecef, config.itrf));
            rk45_cfg.initial_step_size = std::min(rk45_cfg.initial_step_size, rk45_cfg.max_step_size);
            propagator_ = std::make_unique<propagators::RK45Propagator>(
                initial_epoch, position_ecef, velocity_ecef, model, weather,
                config.enable_third_body, rk45_cfg,
                nullptr,  // Use default ephemeris
                PropagationFrame::GCRS, config.itrf
            );
        }
        configure_force_model(propagator_->force_model());
        configure_atmosphere_and_drag();
        note_data_coverage(initial_epoch);
        configure_radiation();
        configure_empirical_accel(initial_epoch);
    }

public:

    /// Propagate to `target` (forward or backward) and return the state there
    /// in ECEF. Records no history (use step() or propagate_until for that).
    /// Throws DataCoverageError if the run would pass the end of the drag
    /// model's space weather or the EOP hold limit; the propagation stops at
    /// that edge and the state up to it is kept.
    [[nodiscard]] PropagationResult propagate(const Epoch& target) {
        // Go in legs that end at the edge of the space weather or of the EOP
        // table if the target lies beyond it (the leg-end helpers throw there).
        PropagationResult result;
        for (;;) {
            const Epoch leg_end = eop_leg_end(drag_leg_end(target));
            result = (*propagator_)(leg_end);
            if (std::abs(target - leg_end) < defaults::EPOCH_EQUALITY_SECONDS) break;
        }

        // Resync the step-label accumulator with the propagator's new epoch
        // (reads back current_epoch() rather than assuming `target`, because
        // propagate_to() early-returns on sub-nanosecond moves).
        elapsed_s_ = propagator_->current_epoch() - label_ref_;
        note_data_coverage(propagator_->current_epoch());
        note_bare_config();

        // Compute force breakdown if requested
        if (config_.report_forces) {
            result.forces = propagator_->force_model().compute_force_breakdown(
                propagator_->current_epoch(),
                propagator_->current_state_gcrs(),
                config_.include_disabled_forces);
        }

        return result;
    }

    /// Same as propagate(target).
    [[nodiscard]] PropagationResult operator()(const Epoch& target) {
        return propagate(target);
    }

    /// Step forward to `target`, watching for `evs` (events.hpp: altitude
    /// crossings, perigee, ground impact...). Each event is located inside the
    /// step where it happens, to about 1e-4 s. The run ends at the first
    /// TERMINAL event or at the target, and every located event is logged in
    /// the result. With stop_at_ground (the default) events::ground_impact()
    /// is added, so a reentering object stops on the Earth's surface (the
    /// WGS84 ellipsoid) instead of integrating into the Earth. History is
    /// recorded as with step(); a terminal event replaces the overshooting
    /// step's entry with the event state.
    /// Throws ConfigError if `target` is before the current epoch (forward only).
    [[nodiscard]] PropagateUntilResult propagate_until(const Epoch& target,
                                                       std::vector<events::Event> evs = {},
                                                       bool stop_at_ground = true) {
        if (stop_at_ground) evs.push_back(events::ground_impact());
        if (history_.empty()) store_current_state(0.0);
        if (target - propagator_->current_epoch() < -defaults::EPOCH_EQUALITY_SECONDS) {
            throw ConfigError("Trajectory::propagate_until runs forward in time; the target " +
                              utc_string(target) + " is before the current epoch " +
                              utc_string(propagator_->current_epoch()));
        }
        auto g_of = [](const events::Event& e, const Epoch& t, const StateVector& s_ecef) {
            return e.g(t, s_ecef.position, s_ecef.velocity);
        };
        PropagateUntilResult out;
        std::vector<double> g_prev;
        {
            const StateVector s0 = propagator_->current_state_ecef();
            for (const auto& e : evs) g_prev.push_back(g_of(e, propagator_->current_epoch(), s0));
        }
        while (target - propagator_->current_epoch() > defaults::EPOCH_EQUALITY_SECONDS) {
            const double remaining = target - propagator_->current_epoch();
            const Epoch t0 = propagator_->current_epoch();
            const StateVector s0_gcrs = propagator_->current_state_gcrs();
            const double e0 = elapsed_s_;
            if (is_symplectic()) {
                if (config_.symplectic.step_size > remaining) advance_exactly(remaining);
                else (void)step();
            } else {
                if (std::abs(propagator_->current_step_size()) > remaining)
                    propagator_->set_step_size(remaining);
                limit_step_near_ground();
                (void)step();
            }
            const double h = elapsed_s_ - e0;
            const Epoch t1 = propagator_->current_epoch();
            const StateVector s1 = propagator_->current_state_ecef();

            struct Hit { double tau; std::size_t i; };
            std::vector<Hit> hits;
            std::vector<double> g_new(evs.size());
            for (std::size_t i = 0; i < evs.size(); ++i) {
                g_new[i] = g_of(evs[i], t1, s1);
                if (!events::detail::crossed(g_prev[i], g_new[i], evs[i].direction)) continue;
                const double tau = events::detail::locate(
                    [&](double tr) { return g_of(evs[i], t0 + tr, state_within_step(t0, s0_gcrs, tr)); },
                    h, g_prev[i], g_new[i]);
                hits.push_back({tau, i});
            }
            std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) { return a.tau < b.tau; });
            const auto stop = std::find_if(hits.begin(), hits.end(),
                                           [&](const Hit& x) { return evs[x.i].terminal; });
            for (auto it = hits.begin(); it != hits.end(); ++it) {
                if (stop != hits.end() && it->tau > stop->tau) break;
                const StateVector se = state_within_step(t0, s0_gcrs, it->tau);
                out.log.push_back({evs[it->i].name, t0 + it->tau, se.position, se.velocity,
                                   evs[it->i].terminal});
            }
            if (stop != hits.end()) {
                // Replace the overshooting step with the step to the event.
                history_.pop_back();
                restore_state(t0, s0_gcrs, e0);
                advance_exactly(stop->tau);
                for (const auto& o : out.log)
                    if (o.terminal) { out.stopped_by = o; break; }
                break;
            }
            g_prev = std::move(g_new);
        }
        elapsed_s_ = propagator_->current_epoch() - label_ref_;
        out.epoch = propagator_->current_epoch();
        out.state_ecef = propagator_->current_state_ecef();
        return out;
    }

    /// Current state (position [m], velocity [m/s]) in ECEF/ITRF.
    [[nodiscard]] StateVector state_ecef() const {
        return propagator_->current_state_ecef();
    }

    /// Current state in the inertial GCRS frame.
    [[nodiscard]] StateVector state_gcrs() const {
        return propagator_->current_state_gcrs();
    }

    /// Current time of the propagator.
    [[nodiscard]] Epoch epoch() const {
        return propagator_->current_epoch();
    }

    /// Reference epoch of the history's time axis: every HistoryEntry::t_rel_s
    /// is seconds since this epoch. Set at construction and on reset().
    [[nodiscard]] Epoch label_ref() const noexcept {
        return label_ref_;
    }

    /// Start over from a new Earth-fixed state (position [m], velocity [m/s]
    /// in ITRF at `epoch`). Clears the history. Unlike the constructors, the
    /// state is not checked.
    void reset(const Epoch& epoch, const Vector3d& position_ecef, const Vector3d& velocity_ecef) {
        propagator_->reset(epoch, StateVector(position_ecef, velocity_ecef));
        history_.clear();
        label_ref_ = epoch;
        elapsed_s_ = 0.0;
        front_elapsed_s_ = 0.0;
    }

    /// The underlying force model, for fine-grained control (e.g.
    /// force_model().third_body().enable_sun(false)).
    [[nodiscard]] forces::ForceModel& force_model() {
        return propagator_->force_model();
    }

    [[nodiscard]] const forces::ForceModel& force_model() const {
        return propagator_->force_model();
    }

    /// True when the integrator is one of the Yoshida symplectic schemes.
    [[nodiscard]] bool is_symplectic() const noexcept {
        return integrator_type_ != IntegratorType::RK45;
    }

    /// Relative change of the orbital energy since the start,
    /// (E_current - E_initial) / |E_initial|. Tracked by the symplectic
    /// integrators only; RK45 returns 0.0.
    [[nodiscard]] double energy_drift() const {
        return propagator_->energy_drift();
    }

    /// The configuration this trajectory was built with.
    [[nodiscard]] const TrajectoryConfig& config() const {
        return config_;
    }

    /// Non-fatal notes from propagation, each recorded once: drag running on
    /// forecast (not measured) space weather, running past a tabulated drag
    /// environment (atmosphere frozen), past the empirical acceleration's arc,
    /// outside the loaded EOP table, or with missing EOP / ocean-tide data.
    /// The same notes also go to the diagnostics channel (core/diagnostics.hpp).
    [[nodiscard]] const std::vector<std::string>& warnings() const noexcept {
        return warnings_;
    }

    /// For trajectories the library builds for itself (estimators, drag
    /// environment generators): notes about extrapolating this trajectory's
    /// own empirical acceleration stay in warnings() but are not reported,
    /// since the estimator propagates past its arc on purpose. Notes about
    /// data (EOP, tides, weather) are still reported.
    void mark_internal() noexcept { internal_ = true; }

    // =========================================================================
    // Step-based propagation with history
    // =========================================================================

    /// Take one step forward in time and append the new state to the history.
    /// RK45 chooses the step from its error estimate; the symplectic
    /// integrators use the fixed step from the config. Throws DataCoverageError
    /// at the edge of the space weather or of the EOP hold limit.
    [[nodiscard]] StepResult step() {
        // Store initial state on first step
        if (history_.empty()) {
            store_current_state(0.0);
        }
        limit_step_in_atmosphere();
        cover_next_step(true);
        cover_next_step_eop(true);
        // The step's START: one step overshooting a target (the interpolator
        // steps past the query time) is not running past the data.
        note_data_coverage(propagator_->current_epoch());
        note_bare_config();

        return is_symplectic() ? step_symplectic() : step_rk45();
    }

    /// Go back `step_back_count` entries in the history (no integration) and
    /// drop the later entries. Returns false if already at the first state;
    /// throws std::out_of_range if the count exceeds the history. Use
    /// step_backward() to integrate to a time before the first state.
    bool rewind(size_t step_back_count = 1) {
        // Need at least initial state + step_back_count entries
        if (history_.size() <= step_back_count) {
            if (history_.size() <= 1) {
                return false;  // Already at initial state
            }
            throw std::out_of_range("rewind count exceeds available history");
        }

        // Calculate target index
        size_t target_index = history_.size() - 1 - step_back_count;
        const HistoryEntry& target = history_.at(target_index);

        // Reset the propagator directly — Trajectory::reset() would clear
        // the history whose earlier entries this restore must preserve.
        propagator_->reset(target.epoch, StateVector(target.position, target.velocity));
        elapsed_s_ = target.t_rel_s;

        // Truncate history to target point (entries 0..target_index survive)
        history_.resize(target_index + 1);

        return true;
    }

    /// Integrate one step backward from the EARLIEST history entry and insert
    /// the new state at the front of the history. The current (latest) state
    /// is unchanged. Step size as in step().
    [[nodiscard]] StepResult step_backward() {
        // Ensure history is initialized
        if (history_.empty()) {
            store_current_state(0.0);
        }

        // Save current propagator state (we'll restore it after)
        Epoch current_epoch = propagator_->current_epoch();
        StateVector current_state_ecef = propagator_->current_state_ecef();

        // Reset propagator to earliest history entry
        const HistoryEntry& earliest = history_.front();
        const Epoch earliest_epoch = earliest.epoch;
        propagator_->reset(earliest.epoch, StateVector(earliest.position, earliest.velocity));
        cover_next_step(false);
        cover_next_step_eop(false);

        // Take one backward step
        StepResult result;
        if (is_symplectic()) {
            result = step_backward_symplectic();
        } else {
            result = step_backward_rk45();
        }

        // Prepend the new state to history
        history_.insert(history_.begin(), HistoryEntry(
            result.epoch,
            front_elapsed_s_,
            result.position,
            result.velocity,
            result.acceleration,
            result.step_size_taken
        ));

        // Update history_index in result (now at index 0)
        result.history_index = 0;
        note_data_coverage(earliest_epoch);

        // Restore propagator to original state (end of history)
        propagator_->reset(current_epoch, current_state_ecef);

        return result;
    }

    // =========================================================================
    // History access methods
    // =========================================================================

    /// Number of stored states, including the initial one.
    [[nodiscard]] size_t history_size() const noexcept {
        return history_.size();
    }

    /// History entry by index (0 = earliest). Throws std::out_of_range.
    [[nodiscard]] const HistoryEntry& history_at(size_t index) const {
        return history_.at(index);
    }

    /// The whole history, earliest first.
    [[nodiscard]] const std::vector<HistoryEntry>& history() const noexcept {
        return history_;
    }

    /// Drop the history, keeping only the latest state as the new first entry.
    void clear_history() {
        if (history_.empty()) {
            store_current_state(0.0);
        } else {
            HistoryEntry current = std::move(history_.back());
            history_.clear();
            history_.push_back(std::move(current));
            front_elapsed_s_ = history_.front().t_rel_s;
        }
    }

    // =========================================================================
    // Step size control
    // =========================================================================

    /// Step size [s] the next step will try: RK45's adaptive proposal, or the
    /// symplectic integrator's fixed step.
    [[nodiscard]] double current_step_size() const {
        return propagator_->current_step_size();
    }

    /// Suggest the next step size [s]. RK45 may still shrink it to meet its
    /// error tolerance; for the symplectic integrators this sets the fixed step
    /// of every following step (it must then be positive: ConfigError otherwise).
    void set_step_size_hint(double dt) {
        propagator_->set_step_size(dt);
        if (is_symplectic()) config_.symplectic.step_size = dt;
    }

private:
    std::unique_ptr<propagators::PropagatorBase> propagator_;
    TrajectoryConfig config_;
    std::vector<HistoryEntry> history_;
    IntegratorType integrator_type_;

    // Step times are built as label_ref_ + elapsed seconds rather than by
    // adding each step to the previous time, so rounding never accumulates
    // step over step (a chained sum of rounded epochs drifts, and the drift
    // shows up as along-track error). t_rel_s in the history is the same
    // accumulator.
    Epoch label_ref_;
    double elapsed_s_ = 0.0;        // propagator's current state vs label_ref_
    double front_elapsed_s_ = 0.0;  // history front vs label_ref_ (backward stepping)

    std::vector<std::string> warnings_;
    bool warned_frozen_ = false;

    bool internal_ = false;

    // Each note is kept on this trajectory (warnings()) and reported to the
    // process-wide diagnostics sink (stderr in C++, a Python warning in
    // Python, an exception in strict mode). The warned_* flags make each
    // note fire once per trajectory. Notes about global data (EOP, ocean
    // tides, space weather) carry a key, so the sink delivers them once per
    // process however many trajectories hit them. A library-internal
    // trajectory (mark_internal) keeps its Extrapolation notes to itself:
    // estimators propagate past their own empirical acceleration on purpose.
    void record(diagnostics::Category category, std::string message, std::string key = {}) {
        warnings_.push_back(message);
        if (internal_ && category == diagnostics::Category::Extrapolation) return;
        diagnostics::report(category, std::move(message), std::move(key));
    }
    // A bare configuration (no radiation pressure, no drag) is legitimate --
    // they need the satellite's properties -- but it costs accuracy that a
    // user may not know about. Say so once per process, with the measured
    // cost at this altitude (presets/tle_grade_table.hpp), at the first
    // propagation rather than at construction, so that estimators can mark
    // their trajectories internal first (they leave forces off on purpose).
    void note_bare_config() {
        if (warned_bare_ || internal_ || config_.srp || config_.drag_env) return;
        warned_bare_ = true;
        const double alt_km = coords::ecef_to_altitude(propagator_->current_state_ecef().position) / 1e3;
        std::string msg = "radiation pressure and drag are off (a TrajectoryConfig without srp or "
                          "drag_env): they need the satellite's properties";
        if (const auto c = presets::bare_config_cost(alt_km)) {
            char b[320];
            std::snprintf(b, sizeof b,
                          ". Measured at %.0f km for a light object (A/m 0.02 m^2/kg, m/(Cd A) "
                          "50 kg/m^2), they move it by %.0f m (radiation) and %.0f m (drag) in "
                          "1 h, %.0f m and %.0f m in 1 day; scale with your object's A/m",
                          c->altitude_km, c->radiation_1h_m, c->drag_1h_m, c->radiation_1d_m,
                          c->drag_1d_m);
            msg += b;
        }
        msg += ". Python: recommended_config(state, satellite | mass_kg=..., srp_area_m2=..., "
               "drag_area_m2=...) (or accuracy=\"tle\" for a TLE start); C++: "
               "TrajectoryConfig::srp and make_drag_environment (model-based drag needs the "
               "NRL models plugin, Python ballistic.install_nrl_models).";
        diagnostics::report(diagnostics::Category::ConfigHint, std::move(msg), "bare-config");
    }
    bool warned_bare_ = false;
    bool warned_predicted_ = false;
    bool warned_ea_ = false;
    bool warned_eop_ = false;
    bool warned_tides_ = false;

    // ---- Drag environment coverage ------------------------------------------
    // make_drag_environment's live env is valid wherever its space weather
    // is: a Trajectory stops at that edge with an error rather than evaluate
    // past it. A tabulated env freezes past its domain (recorded once).

    // Where the next propagate() leg toward `target` must stop: `target`, or
    // the edge of the live env's weather. At the edge, throws.
    Epoch drag_leg_end(const Epoch& target) {
        const auto& env = config_.drag_env;
        if (!env) return target;
        const Epoch now = propagator_->current_epoch();
        const bool forward = !(target < now);
        if (const auto* src = env->live_source(); src && src->valid) {
            const Epoch edge = forward ? src->valid->end : src->valid->start;
            const double beyond = forward ? target - edge : edge - target;
            if (beyond <= defaults::EPOCH_EQUALITY_SECONDS) return target;
            const double to_edge = forward ? edge - now : now - edge;
            if (to_edge > defaults::EPOCH_EQUALITY_SECONDS) return edge;
            throw_weather_edge(*src, forward);
        }
        if (const auto w = env->domain()) {
            const Epoch edge = forward ? w->end : w->start;
            if ((forward ? target - edge : edge - target) > defaults::EPOCH_EQUALITY_SECONDS)
                note_frozen(edge);
        }
        return target;
    }

    // Before one step from the current state: land an RK45 step on the live
    // env's weather edge rather than cross it (throw if already there); note
    // a tabulated env's domain end.
    void cover_next_step(bool forward) {
        const auto& env = config_.drag_env;
        if (!env) return;
        const Epoch now = propagator_->current_epoch();
        const double h = std::abs(is_symplectic() ? config_.symplectic.step_size
                                                  : propagator_->current_step_size());
        if (const auto* src = env->live_source(); src && src->valid) {
            const Epoch edge = forward ? src->valid->end : src->valid->start;
            const double room = forward ? edge - now : now - edge;
            if (room >= h) return;
            if (room <= defaults::EPOCH_EQUALITY_SECONDS || is_symplectic())
                throw_weather_edge(*src, forward);
            propagator_->set_step_size(room);
            return;
        }
        if (const auto w = env->domain()) {
            const Epoch edge = forward ? w->end : w->start;
            if ((forward ? edge - now : now - edge) < h) note_frozen(edge);
        }
    }

    // ---- Earth orientation coverage -------------------------------------------
    // The frame transforms hold the EOP table's edge value for up to
    // time::kEopHoldDays and refuse beyond; a Trajectory stops at that limit
    // (state and history kept), as at the weather edge. Nothing loaded: no
    // limit here (the transforms then raise DataCoverageError, unless the
    // EopPolicy::AssumeZero opt-in makes them use zeros, noted once).
    std::optional<Epoch> eop_limit(bool forward) const {
        const auto& eop = time::EOPData::instance();
        if (!eop.is_loaded()) return std::nullopt;
        const auto [m0, m1] = eop.mjd_range();
        const double mjd = forward ? m1 + time::kEopHoldDays : m0 - time::kEopHoldDays;
        // The UTC day `mjd` (modified Julian date), placed in TT (terrestrial
        // time) with the latest TAI-UTC offset: the same instant as a UTC
        // conversion, but without asking the leap-second table about a date it
        // may not cover (the EOP predictions + 30 d usually end past the
        // table's expiry; converting there would warn on EVERY propagation,
        // whatever its date).
        const double tt_minus_utc =
            time::LeapSecondTable::instance().current_tai_utc() + constants::TT_TAI_OFFSET;
        return Epoch::from_mjd(mjd, time::TimeScale::TT) + tt_minus_utc;
    }

    Epoch eop_leg_end(const Epoch& target) {
        const Epoch now = propagator_->current_epoch();
        const bool forward = !(target < now);
        const auto edge = eop_limit(forward);
        if (!edge) return target;
        const double beyond = forward ? target - *edge : *edge - target;
        if (beyond <= defaults::EPOCH_EQUALITY_SECONDS) return target;
        const double to_edge = forward ? *edge - now : now - *edge;
        if (to_edge > defaults::EPOCH_EQUALITY_SECONDS) return *edge;
        throw_eop_edge(forward);
    }

    void cover_next_step_eop(bool forward) {
        const auto edge = eop_limit(forward);
        if (!edge) return;
        const Epoch now = propagator_->current_epoch();
        const double h = std::abs(is_symplectic() ? config_.symplectic.step_size
                                                  : propagator_->current_step_size());
        const double room = forward ? *edge - now : now - *edge;
        if (room >= h) return;
        if (room <= defaults::EPOCH_EQUALITY_SECONDS || is_symplectic()) throw_eop_edge(forward);
        propagator_->set_step_size(room);
    }

    [[noreturn]] void throw_eop_edge(bool forward) {
        const auto [m0, m1] = time::EOPData::instance().mjd_range();
        throw DataCoverageError(
            "Trajectory: no Earth orientation data " + std::string(forward ? "past " : "before ") +
            utc_string(propagator_->current_epoch()) + ": the loaded table covers MJD " +
            std::to_string(static_cast<long>(m0)) + " to " + std::to_string(static_cast<long>(m1)) +
            " and its edge value is held for at most " +
            std::to_string(static_cast<int>(time::kEopHoldDays)) + " days (UT1-UTC drifts ~1 ms "
            "per few days). Propagation stopped there; the state and history up to it are kept. "
            "Load a newer finals2000A file (Python: ballistic.load()).");
    }

    [[noreturn]] void throw_weather_edge(const atmosphere::LiveModelSource& src, bool forward) {
        const Epoch now = propagator_->current_epoch();
        std::string why;
        if (!src.loaded_weather)
            why = "the space weather passed to make_drag_environment ends here";
        else if (!forward)
            why = "the loaded space weather starts here";
        else if (src.allow_predicted)
            why = "the loaded space weather, including CelesTrak's ~45-day forecast, ends here";
        else
            why = "measured space weather ends here and forecasts were refused "
                  "(allow_predicted_weather=false)";
        throw DataCoverageError(
            "Trajectory: drag has no space weather " + std::string(forward ? "past " : "before ") +
            utc_string(now) + ": " + why + ". Propagation stopped there; the state "
            "and history up to it are kept. Load a newer SW-All.csv.");
    }

    // Measured space weather ran out and drag is on CelesTrak's forecast (the
    // default; allow_predicted_weather = false refuses). Recorded once.
    void note_predicted(const Epoch& from) {
        if (warned_predicted_) return;
        warned_predicted_ = true;
        record(diagnostics::Category::DataCoverage,
            "measured space weather runs out at " + utc_string(from) + "; drag uses "
            "CelesTrak's forecast from there on (less accurate, especially in "
            "storms). Load a newer SW-All.csv for measurements",
            "weather-forecast:" + utc_string(from));
    }

    // "YYYY-MM-DD HH:MM:SS UTC" for messages.
    static std::string utc_string(const Epoch& e) {
        const double jd = e.utc_jd() + 0.5;
        long z = static_cast<long>(std::floor(jd));
        double sec = (jd - z) * 86400.0;
        long l = z + 68569L;
        const long n = 4 * l / 146097L;
        l = l - (146097L * n + 3) / 4;
        const long i = 4000 * (l + 1) / 1461001L;
        l = l - 1461 * i / 4 + 31;
        const long j = 80 * l / 2447;
        const long day = l - 2447 * j / 80;
        l = j / 11;
        const long month = j + 2 - 12 * l;
        const long year = 100 * (n - 49) + i + l;
        long s_int = std::lround(sec);
        if (s_int >= 86400) s_int = 86399;
        char buf[40];
        std::snprintf(buf, sizeof buf, "%04ld-%02ld-%02ld %02ld:%02ld:%02ld UTC",
                      year, month, day, s_int / 3600, (s_int / 60) % 60, s_int % 60);
        return buf;
    }

    // Inputs that change character past a point, or are missing: drag
    // weather turns to forecast, empirical acceleration keeps its last value
    // past its arc, the frame transforms outside the loaded EOP table use
    // the table's edge value for up to 30 days (then the propagation stops,
    // eop_leg_end), and no EOP / no ocean tides loaded at
    // all. Recorded once each.
    void note_data_coverage(const Epoch& t) {
        if (config_.drag_env) {
            if (const auto* src = config_.drag_env->live_source();
                src && src->measured_until && t > *src->measured_until)
                note_predicted(*src->measured_until);
        }
        if (!warned_ea_ && config_.empirical_accel && !config_.empirical_accel->is_zero()) {
            const auto [k0, k1] = config_.empirical_accel->domain();
            const Epoch ref = config_.ea_reference_epoch.value_or(label_ref_);
            const double tr = t - ref;
            // Slack of one sample spacing: tabulated EAs often end a sample
            // short of the arc they were estimated on.
            const auto& keys = config_.empirical_accel->keys();
            double slack = 1.0;
            if (keys.size() >= 2) {
                slack = std::max({slack, keys[1] - keys[0],
                                  keys[keys.size() - 1] - keys[keys.size() - 2]});
            }
            slack += 1e-3;  // small margin for time rounding at the boundary
            if (k1 > k0 && (tr < k0 - slack || tr > k1 + slack)) {
                warned_ea_ = true;
                record(diagnostics::Category::Extrapolation,
                    "propagated outside the empirical acceleration's arc (" +
                    utc_string(ref + k0) + " to " + utc_string(ref + k1) +
                    "): it holds its edge value from there on, which is an "
                    "extrapolation, not an estimate");
            }
        }
        const auto& eop = time::EOPData::instance();
        if (!warned_eop_ && !eop.is_loaded()) {
            warned_eop_ = true;
            record(diagnostics::Category::DataCoverage,
                "no Earth orientation data loaded: UT1-UTC and polar motion are "
                "taken as zero, which costs 3-10 m at LEO. Load finals2000A "
                "(Python: ballistic.load(); C++: InitConfig::eop_path or the "
                "data directory)", "eop-not-loaded");
        } else if (!warned_eop_ && eop.is_loaded()) {
            const auto [m0, m1] = eop.mjd_range();
            const double mjd = t.mjd(time::TimeScale::UTC);
            if (mjd < m0 || mjd > m1) {
                warned_eop_ = true;
                record(diagnostics::Category::DataCoverage,
                    "propagated outside the loaded Earth orientation table (MJD " +
                    std::to_string(static_cast<long>(m0)) + " to " +
                    std::to_string(static_cast<long>(m1)) + "): the table's edge value is "
                    "held (UT1-UTC drifts ~1 ms per few days: meters at LEO within weeks); "
                    "beyond " + std::to_string(static_cast<int>(time::kEopHoldDays)) +
                    " days the propagation stops. Load a newer finals2000A file",
                    "eop-outside-table");
            }
        }
        if (!warned_tides_ && config_.enable_solid_tides && config_.ocean_tide_degree > 0 &&
            !gravity::OceanTideModel::instance().is_loaded()) {
            warned_tides_ = true;
            record(diagnostics::Category::ModelSubstitution,
                "ocean tides are on (ocean_tide_degree > 0) but no ocean-tide "
                "model is loaded, so they are skipped (with the Step-2 solid "
                "tides they cut 1-hour LEO error by 16-28%). Load "
                "fes2004_Cnm-Snm.dat (Python: ballistic.load(); C++: "
                "OceanTideModel::instance().load), or set ocean_tide_degree = 0",
                "ocean-tides-not-loaded");
        }
    }

    void note_frozen(const Epoch& edge) {
        if (warned_frozen_) return;
        warned_frozen_ = true;
        record(diagnostics::Category::DataCoverage,
            "propagated past the drag environment's time domain (" + utc_string(edge) +
            "): the atmosphere is frozen at its last slice from there on. This "
            "is a tabulated environment (measured density, from_models_*, a "
            "saved file); build it over the whole arc, or use "
            "make_drag_environment, which has no time limit but the weather's.");
    }

    // Map IntegratorType enum to SymplecticOrder enum
    [[nodiscard]] static propagators::SymplecticOrder map_integrator_to_order(IntegratorType type) noexcept {
        return to_symplectic_order(type);
    }

    // ---- Events and step limits ----------------------------------------------

    // The state `tau` seconds after (t0, s0_gcrs), re-integrated from the
    // step's start and returned in ITRF. RK45: one Dormand-Prince step of
    // length tau — no longer than the accepted step, so no less accurate.
    // Symplectic: the propagator is moved there and back.
    [[nodiscard]] StateVector state_within_step(const Epoch& t0, const StateVector& s0_gcrs, double tau) {
        if (tau == 0.0) return util::gcrs_to_itrf(s0_gcrs, t0, config_.itrf);
        if (!is_symplectic()) {
            auto& rk45 = *static_cast<propagators::RK45Propagator*>(propagator_.get());
            const auto r = rk45.integrator().step(t0, s0_gcrs, tau);
            return util::gcrs_to_itrf(r.state, t0 + tau, config_.itrf);
        }
        const Epoch t_now = propagator_->current_epoch();
        const StateVector s_now = propagator_->current_state_ecef();
        propagator_->reset(t0, util::gcrs_to_itrf(s0_gcrs, t0, config_.itrf));
        (void)propagator_->propagate_to(t0 + tau);
        const StateVector out = propagator_->current_state_ecef();
        propagator_->reset(t_now, s_now);
        return out;
    }

    void restore_state(const Epoch& t0, const StateVector& s0_gcrs, double elapsed) {
        if (!is_symplectic()) {
            static_cast<propagators::RK45Propagator*>(propagator_.get())->set_state_internal(t0, s0_gcrs);
        } else {
            propagator_->reset(t0, util::gcrs_to_itrf(s0_gcrs, t0, config_.itrf));
        }
        elapsed_s_ = elapsed;
    }

    // Advance exactly dt seconds from the current state (adaptive inside),
    // recording one history entry.
    void advance_exactly(double dt) {
        const double next = std::abs(propagator_->current_step_size());
        elapsed_s_ += dt;
        const Epoch label = label_ref_ + elapsed_s_;
        (void)propagator_->propagate_to(label);
        (void)finalize_step(label, elapsed_s_, dt, next);
    }

    // A twentieth of the orbital period implied by the current state (the
    // "osculating" orbit: the Kepler ellipse the state would follow with
    // point-mass gravity only); no cap (infinity) for an unbound orbit.
    [[nodiscard]] static double period_step_cap(const Epoch& t, const Vector3d& pos_ecef,
                                                const Vector3d& vel_ecef, coords::ITRFRealization itrf) {
        const StateVector si = util::itrf_to_gcrs(StateVector(pos_ecef, vel_ecef), t, itrf);
        const double r = si.position.norm();
        const double energy = 0.5 * si.velocity.squaredNorm() - constants::earth::GM / r;
        if (!(energy < 0.0)) return std::numeric_limits<double>::infinity();
        const double a = -constants::earth::GM / (2.0 * energy);
        const double period = 2.0 * constants::PI * std::sqrt(a * a * a / constants::earth::GM);
        return period / 20.0;
    }

    // With drag on and below 150 km, keep each step's altitude change to
    // about 1 km (density changes by ~e per 6-8 km there), so a step cannot
    // jump across the steep lower thermosphere.
    void limit_step_in_atmosphere() {
        if (is_symplectic() || !config_.drag_env) return;
        const StateVector s = propagator_->current_state_ecef();
        if (coords::ecef_to_geodetic(s.position).altitude > 150e3) return;
        const double vr = std::abs(s.position.dot(s.velocity)) / s.position.norm();
        if (vr <= 0.0) return;
        const double cap = 1000.0 / vr;
        if (std::abs(propagator_->current_step_size()) > cap) propagator_->set_step_size(cap);
    }

    // Descending toward the ground: keep one step from ending more than ~5 km
    // below the surface, so the ground-impact event is bracketed by a step
    // whose stages stay near the surface (the atmosphere refuses -1 km and
    // below). Used by propagate_until only.
    void limit_step_near_ground() {
        const StateVector s = propagator_->current_state_ecef();
        const double vr = s.position.dot(s.velocity) / s.position.norm();
        if (vr >= 0.0) return;
        const double alt = coords::ecef_to_geodetic(s.position).altitude;
        if (alt > 200e3) return;
        const double cap = std::max((alt + 5e3) / -vr, 1e-3);
        if (std::abs(propagator_->current_step_size()) > cap) propagator_->set_step_size(cap);
    }

    void note_forced_step(const Epoch& at, double error_norm, double dt) {
        if (warned_forced_) return;
        warned_forced_ = true;
        char buf[160];
        std::snprintf(buf, sizeof buf, "error norm %.3g (tolerance 1) on a %.3g s step", error_norm, dt);
        record(diagnostics::Category::ForcedStep,
               "Trajectory: the RK45 integrator accepted a step at " + utc_string(at) +
               " that failed its error test (" + buf + ") because it had reached its minimum "
               "step size: the state from here on is outside the integrator's tolerance. "
               "Usually the force is discontinuous or singular there (a position inside the "
               "Earth, a table edge); reported once per trajectory.");
    }
    bool warned_forced_ = false;

    // Configure force model settings from TrajectoryConfig
    void configure_force_model(forces::ForceModel& fm) const {
        const int available = gravity::EGM2008Gravity::instance().available_degree();
        if (config_.gravity_degree < 0 || config_.gravity_degree > available) {
            throw ConfigError("gravity_degree " + std::to_string(config_.gravity_degree) +
                " is outside 0.." + std::to_string(available) + " (the loaded EGM2008 "
                "model; 0-1 = point mass). For degrees above 360 load the NGA "
                "EGM2008_to2190 file (InitConfig.egm2008_path).");
        }
        if (config_.ocean_tide_degree < 0 || config_.ocean_tide_degree == 1) {
            throw ConfigError("ocean_tide_degree " + std::to_string(config_.ocean_tide_degree) +
                " is invalid: use 0 (off) or a degree of at least 2 (default 50).");
        }
        fm.set_gravity_max_degree(config_.gravity_degree);
        fm.set_ocean_tide_degree(config_.ocean_tide_degree);
        forces::ForceComponentFlags flags = fm.force_flags();
        flags.relativistic = config_.enable_relativistic;
        flags.solid_earth_tides = config_.enable_solid_tides;
        // Drag flag is auto-derived by set_drag_environment(); radiation is
        // switched by SrpConfig presence.
        flags.radiation = config_.srp.has_value();
        fm.set_force_flags(flags);
    }

    // Wire the DragEnvironment into the propagator's force model. Drag is
    // exclusively env-driven: setting `config_.drag_env`
    // (a shared_ptr<const DragEnvironment>) turns drag on; leaving it null
    // turns drag off (force_model returns zero drag without an env).
    void configure_atmosphere_and_drag() {
        auto& fm = propagator_->force_model();
        if (config_.drag_env) {
            fm.set_drag_environment(config_.drag_env);
        }
    }

    // Wire SRP-related satellite properties when radiation is enabled.
    void configure_radiation() {
        if (!config_.srp) return;
        const SrpConfig& srp = *config_.srp;
        if (!(srp.mass_kg > 0.0) || !(srp.area_m2 >= 0.0) || !std::isfinite(srp.mass_kg) ||
            !std::isfinite(srp.area_m2) || !std::isfinite(srp.cr))
            throw ConfigError("SrpConfig needs the satellite's mass_kg (> 0) and area_m2 (>= 0); "
                              "got mass " + std::to_string(srp.mass_kg) + " kg, area " +
                              std::to_string(srp.area_m2) + " m^2");
        // SatelliteProperties stores reflectivity = Cr − 1 (its constructor
        // also uses that value for the Earth-albedo reflectivity).
        SatelliteProperties props(srp.area_m2, srp.cr - 1.0);
        auto& fm = propagator_->force_model();
        fm.set_satellite_properties(props, srp.mass_kg);
        fm.radiation().enable_albedo(srp.albedo);
        fm.radiation().enable_ir(srp.earth_ir);
    }

    // Attach an EmpiricalAcceleration source and flip its flag if provided.
    void configure_empirical_accel(const Epoch& initial_epoch) {
        if (!config_.empirical_accel || config_.empirical_accel->is_zero()) return;
        auto& fm = propagator_->force_model();
        fm.set_empirical_acceleration(config_.empirical_accel);
        fm.set_propagation_epoch(config_.ea_reference_epoch.value_or(initial_epoch));
        auto flags = fm.force_flags();
        flags.empirical = true;
        fm.set_force_flags(flags);
    }

    // RK45 step implementation, parameterized by direction.
    // forward=true:  dt = +|current_step_size|; record_history = true
    // forward=false: dt = -|current_step_size|; record_history = false
    // The integrator's stored step size is kept positive in both directions.
    [[nodiscard]] StepResult step_rk45_directional(bool forward) {
        auto& rk45 = *static_cast<propagators::RK45Propagator*>(propagator_.get());

        Epoch old_epoch = rk45.current_epoch();
        StateVector old_state_eci = rk45.current_state_gcrs();
        double min_dt = rk45.integrator().config().min_step_size;

        const double abs_step = std::abs(rk45.integrator().current_step_size());
        double dt = forward ? abs_step : -abs_step;

        auto result = rk45.integrator().step(old_epoch, old_state_eci, dt);
        double dt_used = dt;  // the dt that produced `result` (they diverge on force-accept)

        while (!result.accepted) {
            const double abs_next = std::abs(result.next_step_size);
            dt = forward ? abs_next : -abs_next;
            if (std::abs(dt) <= min_dt) {
                // At minimum step size — can't improve further, force-accept
                // the previous result. dt_used still holds the step that
                // produced it; dt (the rejected proposal) must not be used
                // for time bookkeeping. Reported: the state is no longer
                // within the integrator's tolerance (strict mode raises).
                note_forced_step(old_epoch, result.error_estimate, std::abs(dt_used));
                result.accepted = true;
                break;
            }
            result = rk45.integrator().step(old_epoch, old_state_eci, dt);
            dt_used = dt;
        }

        // Build the label from the exact elapsed-seconds accumulator instead
        // of result.epoch (= old_epoch + dt_used), which chains Epoch rounding
        // step-over-step. dt_used is signed; backward steps run from the
        // history front, so they advance the front accumulator.
        double& elapsed = forward ? elapsed_s_ : front_elapsed_s_;
        elapsed += dt_used;
        Epoch label = label_ref_ + elapsed;

        rk45.set_state_internal(label, result.state);
        rk45.integrator().set_step_size(std::abs(result.next_step_size));

        return finalize_step(label, elapsed, std::abs(dt_used),
                             std::abs(result.next_step_size), forward);
    }

    [[nodiscard]] StepResult step_rk45() { return step_rk45_directional(true); }
    [[nodiscard]] StepResult step_backward_rk45() { return step_rk45_directional(false); }

    // Symplectic step implementation
    [[nodiscard]] StepResult step_symplectic() {
        double dt = config_.symplectic.step_size;

        // Label from the exact accumulator (not old_epoch + dt, which chains
        // Epoch rounding step-over-step)
        elapsed_s_ += dt;
        Epoch new_epoch = label_ref_ + elapsed_s_;
        (void)propagator_->propagate_to(new_epoch);  // Side effect: updates internal state

        return finalize_step(new_epoch, elapsed_s_, dt, dt);  // Symplectic uses fixed step size
    }

    // Symplectic backward step implementation (runs from the history front —
    // see step_backward(), which resets the propagator there first)
    [[nodiscard]] StepResult step_backward_symplectic() {
        double dt = config_.symplectic.step_size;

        front_elapsed_s_ -= dt;
        Epoch new_epoch = label_ref_ + front_elapsed_s_;
        (void)propagator_->propagate_to(new_epoch);

        return finalize_step(new_epoch, front_elapsed_s_, dt, dt, false);
    }

    // Common step finalization: convert to ECEF, compute acceleration, optionally record history
    [[nodiscard]] StepResult finalize_step(const Epoch& new_epoch, double t_rel, double step_taken, double next_step, bool record_history = true) {
        StateVector state_ecef = propagator_->current_state_ecef();
        StateVector state_gcrs = propagator_->current_state_gcrs();

        // Compute acceleration and convert to ECEF
        Vector3d acc_eci = propagator_->force_model().acceleration(new_epoch, state_gcrs);
        Vector3d acc_ecef = util::acceleration_to_itrf(
            acc_eci, state_ecef, new_epoch);

        // Add to history if requested
        if (record_history) {
            history_.push_back(HistoryEntry(
                new_epoch,
                t_rel,
                state_ecef.position,
                state_ecef.velocity,
                acc_ecef,
                step_taken
            ));
        }

        // Compute force breakdown if requested
        std::optional<ForceBreakdown> forces_opt;
        if (config_.report_forces) {
            forces_opt = propagator_->force_model().compute_force_breakdown(
                new_epoch, state_gcrs, config_.include_disabled_forces);
        }

        return StepResult{
            new_epoch,
            state_ecef.position,
            state_ecef.velocity,
            acc_ecef,
            step_taken,
            next_step,
            record_history ? history_.size() - 1 : 0,
            forces_opt
        };
    }

    // Helper to store current state in history (only ever called with an
    // empty history — the new entry becomes the front)
    void store_current_state(double step_size) {
        front_elapsed_s_ = elapsed_s_;
        StateVector state_ecef = propagator_->current_state_ecef();
        StateVector state_gcrs = propagator_->current_state_gcrs();
        Epoch t = propagator_->current_epoch();

        Vector3d acc_eci = propagator_->force_model().acceleration(t, state_gcrs);
        Vector3d acc_ecef = util::acceleration_to_itrf(
            acc_eci, state_ecef, t);

        history_.push_back(HistoryEntry(t, elapsed_s_, state_ecef.position, state_ecef.velocity, acc_ecef, step_size));
    }
};

/// Same as `std::make_unique<Trajectory>(t0, pos_ecef, vel_ecef, config)`:
/// Earth-fixed position [m] and velocity [m/s], units-checked only.
[[nodiscard]] inline std::unique_ptr<Trajectory> create_trajectory(
    const Epoch& t0,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    const TrajectoryConfig& config = {})
{
    return std::make_unique<Trajectory>(t0, pos_ecef, vel_ecef, config);
}

} // namespace ballistic
