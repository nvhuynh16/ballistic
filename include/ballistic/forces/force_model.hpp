#pragma once

// =============================================================================
// Force Model Aggregator
// =============================================================================
//
// Adds up every force acting on an Earth satellite and returns the total
// acceleration. Positions and velocities are in the GCRS (Geocentric Celestial
// Reference System: an Earth-centered frame that does not rotate with the Earth);
// gravity and tides are evaluated in the ITRF (International Terrestrial
// Reference Frame: Earth-fixed, rotating with the Earth) and rotated back.
//
// References:
// -----------
// [1] Montenbruck, O. & Gill, E. (2000). "Satellite Orbits: Models, Methods,
//     and Applications". Springer, Table 3.1 "Perturbative accelerations
//     acting on a low-altitude satellite".
//
// [2] Vallado, D.A. (2013). "Fundamentals of Astrodynamics and Applications",
//     4th ed., Microcosm Press, Chapter 8 "Numerical Techniques".
//
// Typical size of each force at 400 km altitude (orders of magnitude, after
// Montenbruck [1], Table 3.1; the J2 and relativity rows from the formulas used here):
//   Central gravity:    ~8.4 m/s²
//   J2 oblateness:      ~1.2e-2 to 2.5e-2 m/s² (equator to pole; the Earth's equatorial bulge)
//   Higher harmonics:   ~1e-4 m/s² (J3 and above)
//   Moon third-body:    ~1.2e-6 m/s²
//   Sun third-body:     ~5.3e-7 m/s²
//   SRP:                ~9e-8 m/s² (solar radiation pressure; varies with area/mass)
//   Drag:               ~3e-8 to 3e-5 m/s² (varies with solar activity)
//   Relativistic:       ~1.7e-8 m/s² (Schwarzschild, relativistic.hpp)
//   Solid Earth tides:  ~5e-8 m/s²
//
// The ForceModel's own default gravity degree is 360 (all of the embedded
// EGM2008 model); Trajectory sets 70, which is enough for low Earth orbit.
// Frame transforms use the IAU 2006/2000A precession-nutation model.
//
// =============================================================================

#include "../core/types.hpp"
#include "../time/epoch.hpp"
#include "../gravity/point_mass.hpp"
#include "../core/errors.hpp"
#include "../gravity/egm_gravity.hpp"
#include "../gravity/third_body.hpp"
#include "../gravity/solid_earth_tides.hpp"
#include "../gravity/ocean_tides.hpp"
#include "../gravity/clenshaw_engine.hpp"
#include "../coordinates/earth_rotation.hpp"
#include "../coordinates/iau_transforms.hpp"
#include "../ephemeris/jpl_ephemeris.hpp"
#include "drag.hpp"
#include "radiation_pressure.hpp"
#include "relativistic.hpp"
#include "empirical_acceleration.hpp"
#include <algorithm>
#include <limits>
#include <memory>
#include <functional>

namespace ballistic {
namespace forces {

/// On/off switches for each force component. Set them all at once with
/// ForceModel::set_force_flags().
///
/// `radiation` is the master switch for radiation pressure; its three parts
/// (direct solar radiation pressure, Earth albedo = sunlight reflected by the
/// Earth, Earth infrared emission) have their own switches on the
/// RadiationPressure component (`fm.radiation().enable_srp()` etc.), not here.
/// Wind is part of the DragEnvironment, so it has no flag.
struct ForceComponentFlags {
    bool earth_gravity = true;
    bool third_body = true;
    bool drag = false;           // Auto-derived: set_drag_environment() sets it from env presence
    bool radiation = false;      // Master switch for SRP + albedo + IR (requires satellite properties)
    bool relativistic = true;
    bool solid_earth_tides = true;
    bool empirical = false;      // Empirical acceleration corrections

    /// Every force switched on.
    [[nodiscard]] static constexpr ForceComponentFlags all_enabled() noexcept {
        return {true, true, true, true, true, true, true};
    }

    /// The default switches (gravity, third body, relativity and solid tides on;
    /// drag, radiation and empirical acceleration off).
    [[nodiscard]] static constexpr ForceComponentFlags defaults() noexcept {
        return {};
    }

};

/// Sums the enabled forces: Earth gravity, Sun/Moon/planet gravity ("third
/// body"), solid Earth and ocean tides, atmospheric drag, radiation pressure,
/// relativistic corrections and an optional empirical acceleration (EA: a small
/// fitted correction acceleration added to the physical model). All inputs and
/// outputs are in the GCRS frame, SI units (m, m/s, m/s²).
class ForceModel {
public:
    /// Uses the process-wide default ephemeris (the source of Sun/Moon positions):
    /// the shared SPK ephemeris when loaded (a .bsp file given to initialize(),
    /// or the embedded DE440), else a legacy binary DE file loaded by
    /// initialize(); see get_default_ephemeris_provider().
    ForceModel()
        : ForceModel(ephemeris::get_default_ephemeris_provider())
    {}

    /// Uses the given ephemeris provider for Sun/Moon/planet positions.
    explicit ForceModel(std::shared_ptr<ephemeris::EphemerisProvider> ephemeris)
        : third_body_(std::make_unique<gravity::ThirdBodyGravity>(std::move(ephemeris)))
        , drag_(std::make_unique<AtmosphericDrag>())
        , radiation_(std::make_unique<RadiationPressure>())
        , relativistic_(std::make_unique<RelativisticCorrections>())
        , flags_()  // Use default ForceComponentFlags
        , gravity_max_degree_(360)  // Default degree for spherical harmonics
    {}

    /// Replaces all on/off switches at once.
    void set_force_flags(const ForceComponentFlags& flags) noexcept {
        flags_ = flags;
    }
    [[nodiscard]] const ForceComponentFlags& force_flags() const noexcept { return flags_; }

    /// Highest spherical-harmonic degree of the EGM2008 gravity field to evaluate
    /// (1 or less = point mass only; 70 is enough at LEO). Not checked here: a
    /// degree above the loaded model is rejected by gravity::EGM2008Gravity.
    void set_gravity_max_degree(int degree) noexcept { gravity_max_degree_ = degree; }

    /// Ocean tides: highest spherical-harmonic degree of the FES2004 ocean-tide
    /// gravity corrections (0 = off, default 50). Has an effect only once
    /// gravity::OceanTideModel is loaded and solid tides are switched on. A degree
    /// above the loaded file throws ConfigError at the first force evaluation.
    void set_ocean_tide_degree(int degree) noexcept { ocean_tide_degree_ = degree; }
    [[nodiscard]] int ocean_tide_degree() const noexcept { return ocean_tide_degree_; }
    /// True when ocean tides will actually be applied (degree >= 2 and a file loaded).
    [[nodiscard]] bool ocean_tides_active() const {
        return ocean_tide_degree_ >= 2 && gravity::OceanTideModel::instance().is_loaded();
    }

    /// Which realization (version) of the Earth-fixed ITRF frame positions are
    /// expressed in. Default ITRF2020; use ITRF2014 for ESA Sentinel orbit files
    /// before 4 March 2025.
    void set_itrf_realization(coords::ITRFRealization itrf) noexcept {
        itrf_realization_ = itrf;
        drag_->set_itrf_realization(itrf);
    }
    [[nodiscard]] coords::ITRFRealization itrf_realization() const noexcept { return itrf_realization_; }

    // Third-body switches are set on the component directly:
    //   fm.third_body().enable_sun(true);
    //   fm.third_body().enable_moon(true);

    // -------------------------------------------------------------------------
    // Drag
    // -------------------------------------------------------------------------
    /// Sets the drag environment (air density, wind and ballistic coefficient
    /// along the orbit) that drag is computed from; nullptr turns drag off. The
    /// drag flag follows: on exactly when an environment is set.
    void set_drag_environment(
        std::shared_ptr<const atmosphere::DragEnvironment> env) noexcept
    {
        drag_env_ = std::move(env);
        // The env is the data source AND the switch: drag is zero without one,
        // so keep the flag in lockstep rather than trusting callers to agree.
        flags_.drag = static_cast<bool>(drag_env_);
    }

    /// The current drag environment (nullptr when drag is off).
    [[nodiscard]] std::shared_ptr<const atmosphere::DragEnvironment>
    drag_environment() const noexcept {
        return drag_env_;
    }

    /// Switch the two relativistic corrections individually: Schwarzschild (the
    /// main general-relativity correction to Earth's gravity) and Lense-Thirring
    /// (frame dragging by the Earth's spin). The master switch is
    /// ForceComponentFlags::relativistic.
    void enable_schwarzschild(bool enable) noexcept { relativistic_->enable_schwarzschild(enable); }
    void enable_lense_thirring(bool enable) noexcept { relativistic_->enable_lense_thirring(enable); }
    /// True when the relativistic master switch is on.
    [[nodiscard]] bool relativistic_enabled() const noexcept { return flags_.relativistic; }

    // Radiation-pressure parts are set on the component directly:
    //   fm.radiation().enable_srp(true);
    //   fm.radiation().enable_albedo(true);

    /// Sets the empirical acceleration (nullptr = none). It is added only when
    /// ForceComponentFlags::empirical is on. Its time axis is seconds since the
    /// epoch given to set_propagation_epoch().
    void set_empirical_acceleration(std::shared_ptr<EmpiricalAcceleration> emp) {
        empirical_accel_ = std::move(emp);
    }
    /// The current empirical acceleration (may be nullptr).
    [[nodiscard]] std::shared_ptr<EmpiricalAcceleration> empirical_acceleration() const noexcept {
        return empirical_accel_;
    }

    /// Sets the epoch the empirical acceleration's time axis counts from (t = 0).
    void set_propagation_epoch(const Epoch& epoch0) {
        propagation_epoch_ = epoch0;
    }

    /// Sets the radiation-pressure properties (area, reflectivity, emissivity)
    /// from `props` and the satellite mass in kg.
    void set_satellite_properties(const SatelliteProperties& props, double mass) {
        SatelliteRadiationProperties rad_props;
        rad_props.srp_area = props.srp_area;
        rad_props.srp_reflectivity = props.srp_reflectivity;
        rad_props.albedo_reflectivity = props.albedo_reflectivity;
        rad_props.ir_emissivity = props.ir_emissivity;
        radiation_->set_properties(rad_props);
        radiation_->set_mass(mass);
    }

    /// True when the radiation-pressure master switch is on.
    [[nodiscard]] bool radiation_enabled() const noexcept { return flags_.radiation; }

    /// Throws DataCoverageError if `pos` (meters) is within 1000 km of the
    /// Earth's center: the trajectory has passed through the ground. Without
    /// drag nothing else stops such a propagation, and integrating toward the
    /// center makes the step size collapse until the program hangs. The
    /// threshold is deliberately deep, not the surface: synthetic test arcs and
    /// estimator trial states can run tens to hundreds of km below the surface,
    /// and only the singular center must be refused. A real reentry should stop
    /// at the surface with Trajectory::propagate_until (events::ground_impact()
    /// is armed by default).
    static void require_above_ground(const Vector3d& pos) {
        constexpr double kNearCenter = 1.0e6;
        const double r = pos.norm();
        if (r < kNearCenter) {
            throw DataCoverageError(
                "force model: the position is " + std::to_string(r / 1e3) +
                " km from the Earth's center - the trajectory passed through the ground. "
                "Stop at the surface with Trajectory::propagate_until (events::ground_impact() is "
                "armed by default) or Python Trajectory.propagate_until.");
        }
    }

    /// Total acceleration (m/s², GCRS) at `epoch` for the GCRS state. Throws
    /// DataCoverageError if the position is near the Earth's center (see
    /// require_above_ground), and whatever a component throws (e.g. ConfigError).
    [[nodiscard]] Vector3d acceleration(
        const Epoch& epoch,
        const StateVector& state_gcrs) const
    {
        return compute_all_forces(epoch, state_gcrs).total;
    }

    /// Total acceleration AND its 3×6 Jacobian (matrix of partial derivatives)
    /// with respect to the state [position, velocity], both in GCRS. Returns the
    /// pair (a, J) with
    ///   J.block<3,3>(0,0) = ∂a/∂r   (1/s²)
    ///   J.block<3,3>(0,3) = ∂a/∂v   (1/s)
    /// Used to build the state transition matrix (STM: how the state at a later
    /// time changes with the initial state) for Gauss-Newton fitting.
    ///
    /// Each enabled force adds its own acceleration and Jacobian. A component
    /// uses an analytic Jacobian where one is implemented and finite differences
    /// (FD: re-evaluate with a small offset and divide) otherwise. The empirical
    /// acceleration depends only on time, so its Jacobian is exactly zero.
    ///
    /// `fd_dx` (m) and `fd_dv` (m/s) are the finite-difference offsets for the
    /// components that use FD (drag's position columns, and its velocity
    /// columns when beta depends on velocity).
    [[nodiscard]] std::pair<Vector3d, Eigen::Matrix<double, 3, 6>>
    acceleration_with_jacobian(
        const Epoch& epoch,
        const StateVector& state_gcrs,
        double fd_dx = 1.0,
        double fd_dv = 1e-3) const
    {
        using Mat36 = Eigen::Matrix<double, 3, 6>;
        require_above_ground(state_gcrs.position);

        // Forces that depend only on position have ∂a/∂v = 0, so only the three
        // position offsets are evaluated.
        auto fd_position_only_jacobian = [&](auto&& accel_fn, const Vector3d& a0) -> Mat36 {
            Mat36 J = Mat36::Zero();
            for (int i = 0; i < 3; ++i) {
                StateVector xp = state_gcrs;
                xp.position(i) += fd_dx;
                J.col(i) = (accel_fn(xp) - a0) / fd_dx;
            }
            // J.block<3,3>(0,3) stays zero — no velocity dependence.
            return J;
        };

        Vector3d a_total = Vector3d::Zero();
        Mat36 J_total = Mat36::Zero();

        if (flags_.earth_gravity) {
            // Analytic gravity Hessian (second derivatives of the potential)
            // from the gravity engine in ITRF, rotated to GCRS
            // (docs/gravity_hessian_derivation.md).
            auto [a, J3x3] = earth_gravity_acceleration_with_jacobian(epoch, state_gcrs);
            a_total += a;
            J_total.block<3, 3>(0, 0) += J3x3;
            // J_total.block<3,3>(0,3) zero (gravity has no velocity dependence).
        }
        if (flags_.third_body) {
            // Analytic ∂a/∂r_sat (the tidal tensor of each body, summed).
            // Third-body gravity does not depend on velocity. See
            // `ThirdBodyGravity::acceleration_with_jacobian` for the derivation.
            auto [a, J3x3] = third_body_->acceleration_with_jacobian(epoch, state_gcrs.position);
            a_total += a;
            J_total.block<3, 3>(0, 0) += J3x3;
            // J_total.block<3,3>(0,3) unchanged (zero velocity Jacobian).
        }
        if (flags_.drag) {
            // Analytic ∂a/∂v and finite-difference ∂a/∂r. Drag is the only
            // velocity-dependent force apart from relativity. Without an
            // environment drag is off.
            if (drag_env_) {
                auto [a, J3x6] =
                    drag_->acceleration_with_jacobian(*drag_env_, epoch, state_gcrs, fd_dx, fd_dv);
                a_total += a;
                J_total += J3x6;
            }
        }
        if (flags_.radiation) {
            // Analytic SRP, albedo and IR Jacobians. None of them depends on
            // velocity, so only the 3×3 position block is filled.
            auto [a, J3x3] = radiation_->acceleration_with_jacobian(epoch, state_gcrs.position);
            a_total += a;
            J_total.block<3, 3>(0, 0) += J3x3;
        }
        if (flags_.relativistic) {
            // Analytic ∂a/∂[r,v] for Schwarzschild + Lense-Thirring. Both depend
            // on position AND velocity, so the full 3×6 J is filled.
            auto [a, J3x6] = relativistic_->acceleration_with_jacobian(epoch, state_gcrs);
            a_total += a;
            J_total += J3x6;
        }
        if (flags_.solid_earth_tides) {
            // Analytic solid-tide (and ocean-tide) acceleration + Jacobian. A
            // tide only changes the gravity coefficients, which do not depend on
            // the satellite's position, so the same gravity engine computes it
            // with corrected coefficients. Checked by
            // test_force_model_gravity_hessian.
            auto [a_eci, J3x3] = solid_tides_acceleration_with_jacobian(epoch, state_gcrs);
            a_total += a_eci;
            J_total.block<3, 3>(0, 0) += J3x3;
            // Tides do not depend on velocity.
        }
        if (flags_.empirical) {
            // The empirical acceleration depends on time only; ∂a/∂[r,v] = 0.
            a_total += compute_empirical(epoch, state_gcrs);
        }

        return {a_total, J_total};
    }

private:
    // Frame conversions, always with the full IAU 2006/2000A model.
    [[nodiscard]] Vector3d gcrs_position_to_itrf(const Vector3d& pos_eci, const Epoch& epoch) const {
        return coords::gcrs_to_itrf_full(pos_eci, epoch, itrf_realization_);
    }

    [[nodiscard]] Vector3d itrf_vector_to_gcrs(const Vector3d& vec_itrf, const Epoch& epoch) const {
        return coords::itrf_to_gcrs_vector_full(vec_itrf, epoch);
    }

    [[nodiscard]] StateVector gcrs_state_to_itrf(const StateVector& state_gcrs, const Epoch& epoch) const {
        return coords::gcrs_to_itrf_full(state_gcrs, epoch, itrf_realization_);
    }

    // Returns Vector3d::Zero() if disabled, otherwise calls compute_fn.
    template <typename Fn>
    [[nodiscard]] Vector3d compute_if_enabled(bool enabled, Fn&& compute_fn) const {
        return enabled ? std::forward<Fn>(compute_fn)() : Vector3d::Zero();
    }

    // Earth gravity: point mass for degree <= 1, else EGM2008 evaluated in ITRF.
    [[nodiscard]] Vector3d compute_earth_gravity(
        const Epoch& epoch,
        const StateVector& state_gcrs) const
    {
        if (gravity_max_degree_ <= 1) {
            // Point mass only
            return gravity::point_mass_acceleration(state_gcrs.position);
        }

        // EGM2008 spherical harmonics - compute in ECEF/ITRF
        auto state_itrf = gcrs_state_to_itrf(state_gcrs, epoch);
        Vector3d acc_itrf = gravity::EGM2008Gravity::instance().acceleration(
            state_itrf.position, gravity_max_degree_);
        return itrf_vector_to_gcrs(acc_itrf, epoch);
    }

    // Solid Earth tides: the Moon and Sun deform the Earth, which changes its
    // gravity field and so adds a small acceleration on the satellite.
    [[nodiscard]] Vector3d compute_solid_tides(
        const Epoch& epoch,
        const StateVector& state_gcrs) const
    {
        // Get Moon and Sun positions in GCRS/ECI
        Vector3d moon_gcrs = third_body_->moon_position(epoch);
        Vector3d sun_gcrs = third_body_->sun_position(epoch);

        // Transform satellite, Moon, Sun to ITRF for tidal computation
        Vector3d sat_itrf = gcrs_position_to_itrf(state_gcrs.position, epoch);
        Vector3d moon_itrf = gcrs_position_to_itrf(moon_gcrs, epoch);
        Vector3d sun_itrf = gcrs_position_to_itrf(sun_gcrs, epoch);

        const auto beta = gravity::doodson_arguments(epoch);
        Vector3d acc_itrf = solid_tides_.acceleration(sat_itrf, moon_itrf, sun_itrf, &beta);
        return itrf_vector_to_gcrs(acc_itrf, epoch);
    }

    // Ocean-tide acceleration (ITRF) and, optionally, its ITRF Hessian. The
    // tidal gravity corrections change slowly (shortest period in the FES2004
    // file: M4, ~6.2 h), so they are cached per thread for 30 s of model time
    // (a phase change below 0.01 rad). Throws
    // ConfigError if the requested degree exceeds the loaded file.
    [[nodiscard]] std::pair<Vector3d, Eigen::Matrix3d> ocean_tides_ecef(
        const Epoch& epoch, const Vector3d& sat_itrf, bool with_hessian) const
    {
        const int loaded = gravity::OceanTideModel::instance().max_degree();
        if (ocean_tide_degree_ > loaded) {
            // Never clamp silently to the file's degree.
            throw ConfigError("ocean_tide_degree " + std::to_string(ocean_tide_degree_) +
                " requested, but the loaded ocean-tide file stops at degree " +
                std::to_string(loaded) + ". Lower ocean_tide_degree (50 is the default; "
                "30 costs 1-2 cm).");
        }
        const int N = ocean_tide_degree_;
        thread_local gravity::Clenshaw engine;
        thread_local int engine_degree = -1;
        thread_local double coeff_tt = 0.0;
        if (engine_degree != std::max(N, 5)) {
            engine_degree = std::max(N, 5);          // engine needs degree >= 5
            engine.init(constants::earth::GM, constants::earth::RADIUS_EQUATORIAL, engine_degree);
            engine.set_coefficient(0, 0, 0.0, 0.0);
            coeff_tt = std::numeric_limits<double>::quiet_NaN();
        }
        const double tt = epoch.tt_jd();
        if (!(std::abs(tt - coeff_tt) <= 30.0 / 86400.0)) {
            thread_local std::vector<double> dC, dS;
            gravity::OceanTideModel::instance().stokes(gravity::doodson_arguments(epoch), N, dC, dS);
            for (int n = 2; n <= N; ++n)
                for (int m = 0; m <= n; ++m)
                    engine.set_coefficient(n, m, dC[n * (N + 1) + m], dS[n * (N + 1) + m]);
            coeff_tt = tt;
        }
        if (with_hessian) {
            const auto h = engine.compute_with_hessian(sat_itrf, engine_degree);
            return {h.acceleration, h.hessian_ecef};
        }
        return {engine.compute(sat_itrf, engine_degree).acceleration, Eigen::Matrix3d::Zero()};
    }

    [[nodiscard]] Vector3d compute_ocean_tides(const Epoch& epoch, const StateVector& state_gcrs) const {
        const Vector3d sat_itrf = gcrs_position_to_itrf(state_gcrs.position, epoch);
        return itrf_vector_to_gcrs(ocean_tides_ecef(epoch, sat_itrf, false).first, epoch);
    }

    // One evaluation of every force component, used by acceleration(),
    // split_acceleration() and compute_force_breakdown().
    //
    // `include_disabled` also fills the per-component fields of switched-off
    // forces (for diagnostics). `total` always sums only the enabled forces.
    //
    // This routine and `acceleration_with_jacobian` are the two flag-dispatch
    // loops in this file. They are kept separate because the per-force calls
    // differ: the breakdown needs the radiation parts separately
    // (srp_only/albedo_only/ir_only), the Jacobian path calls
    // radiation_->acceleration_with_jacobian.
    [[nodiscard]] ForceBreakdown compute_all_forces(
        const Epoch& epoch,
        const StateVector& state_gcrs,
        bool include_disabled = false) const
    {
        require_above_ground(state_gcrs.position);
        ForceBreakdown fb;
        fb.total = Vector3d::Zero();

        // Write the per-component field when include_disabled or the flag is
        // on; add to fb.total only when the flag is on.
        auto accumulate = [&](bool flag, Vector3d& field, const Vector3d& value) {
            if (include_disabled || flag) field = value;
            if (flag) fb.total += value;
        };

        if (include_disabled || flags_.earth_gravity)
            accumulate(flags_.earth_gravity, fb.gravity, compute_earth_gravity(epoch, state_gcrs));
        if (include_disabled || flags_.solid_earth_tides)
            accumulate(flags_.solid_earth_tides, fb.solid_tides, compute_solid_tides(epoch, state_gcrs));
        // Ocean tides ride on the tide switch (they are part of the
        // time-variable geopotential) and need loaded coefficients.
        if (flags_.solid_earth_tides && ocean_tides_active())
            accumulate(true, fb.ocean_tides, compute_ocean_tides(epoch, state_gcrs));
        if (include_disabled || flags_.third_body)
            accumulate(flags_.third_body, fb.third_body,
                       third_body_->acceleration(epoch, state_gcrs.position));
        if (include_disabled || flags_.drag) {
            const Vector3d a_drag = drag_env_
                ? drag_->acceleration(*drag_env_, epoch, state_gcrs)
                : Vector3d::Zero();
            accumulate(flags_.drag, fb.drag, a_drag);
        }
        if (include_disabled || flags_.radiation) {
            // Radiation is reported as three fields, each gated by the
            // RadiationPressure component's own switches, so this path agrees
            // with acceleration_with_jacobian.
            Vector3d rad_sum = Vector3d::Zero();
            if (include_disabled || radiation_->srp_enabled())
                fb.srp = radiation_->srp_only(epoch, state_gcrs.position);
            if (include_disabled || radiation_->albedo_enabled())
                fb.albedo = radiation_->albedo_only(epoch, state_gcrs.position);
            if (include_disabled || radiation_->ir_enabled())
                fb.ir = radiation_->ir_only(state_gcrs.position);
            if (radiation_->srp_enabled())    rad_sum += fb.srp;
            if (radiation_->albedo_enabled()) rad_sum += fb.albedo;
            if (radiation_->ir_enabled())     rad_sum += fb.ir;
            if (flags_.radiation) fb.total += rad_sum;
        }
        if (include_disabled || flags_.relativistic)
            accumulate(flags_.relativistic, fb.relativistic,
                       relativistic_->acceleration(epoch, state_gcrs));
        if (include_disabled || flags_.empirical)
            accumulate(flags_.empirical, fb.empirical, compute_empirical(epoch, state_gcrs));

        return fb;
    }

public:

    /// The acceleration split into a conservative part (derivable from a
    /// potential: gravity, tides, conservative relativity) and the rest (drag,
    /// radiation, empirical). Symplectic integrators treat the two differently.
    struct AccelerationComponents {
        Vector3d conservative;      // gravity, third body, tides, conservative relativity
        Vector3d non_conservative;  // drag, radiation, empirical, non-conservative relativity
    };

    /// Splits the total acceleration (m/s², GCRS) into conservative and
    /// non-conservative parts; same inputs and exceptions as acceleration().
    [[nodiscard]] AccelerationComponents split_acceleration(
        const Epoch& epoch,
        const StateVector& state_gcrs) const
    {
        auto fb = compute_all_forces(epoch, state_gcrs);

        AccelerationComponents result;
        result.conservative = Vector3d::Zero();
        result.non_conservative = Vector3d::Zero();

        if (flags_.earth_gravity)    result.conservative += fb.gravity;
        if (flags_.third_body)       result.conservative += fb.third_body;
        if (flags_.solid_earth_tides) result.conservative += fb.solid_tides + fb.ocean_tides;
        if (flags_.relativistic) {
            if (relativistic_->is_conservative())
                result.conservative += fb.relativistic;
            else
                result.non_conservative += fb.relativistic;
        }

        if (flags_.drag)             result.non_conservative += fb.drag;
        if (flags_.radiation)        result.non_conservative += fb.radiation();
        if (flags_.empirical)        result.non_conservative += fb.empirical;

        return result;
    }

    // Individual force contributions, for analysis. Each returns zero (m/s², GCRS)
    // when its force is switched off.
    [[nodiscard]] Vector3d earth_gravity_acceleration(
        const Epoch& epoch,
        const StateVector& state_gcrs) const
    {
        return compute_if_enabled(flags_.earth_gravity,
            [&] { return compute_earth_gravity(epoch, state_gcrs); });
    }

    /// Solid-tide (plus ocean-tide, when active) acceleration and its 3×3
    /// position Jacobian ∂a/∂r, both in GCRS. Zero when tides are off.
    ///
    /// Steps: convert the satellite, Sun and Moon positions to ITRF; get the
    /// acceleration and Hessian in ITRF from the gravity engine with
    /// tide-corrected coefficients; rotate back with R (ITRF→GCRS): the
    /// acceleration as a vector, the Hessian as H_gcrs = R · H_itrf · Rᵀ.
    [[nodiscard]] std::pair<Vector3d, Eigen::Matrix3d>
    solid_tides_acceleration_with_jacobian(
        const Epoch& epoch,
        const StateVector& state_gcrs) const
    {
        if (!flags_.solid_earth_tides) {
            return {Vector3d::Zero(), Eigen::Matrix3d::Zero()};
        }

        // Get Moon and Sun positions in GCRS/ECI, convert to ITRF
        const Vector3d moon_gcrs = third_body_->moon_position(epoch);
        const Vector3d sun_gcrs = third_body_->sun_position(epoch);
        const Vector3d sat_itrf = gcrs_position_to_itrf(state_gcrs.position, epoch);
        const Vector3d moon_itrf = gcrs_position_to_itrf(moon_gcrs, epoch);
        const Vector3d sun_itrf = gcrs_position_to_itrf(sun_gcrs, epoch);

        // Acceleration and analytic Hessian in ITRF from the gravity engine
        const auto beta = gravity::doodson_arguments(epoch);
        auto [a_ecef, H_ecef] = solid_tides_.acceleration_with_hessian(
            sat_itrf, moon_itrf, sun_itrf, &beta);
        if (ocean_tides_active()) {
            auto [a_oc, H_oc] = ocean_tides_ecef(epoch, sat_itrf, true);
            a_ecef += a_oc;
            H_ecef += H_oc;
        }

        // Acceleration: vector transform (existing helper)
        const Vector3d a_eci = itrf_vector_to_gcrs(a_ecef, epoch);

        // Hessian: H_gcrs = R · H_itrf · Rᵀ, with R the ITRF→GCRS rotation
        // (full IAU 2006/2000A, cached per thread).
        const Eigen::Matrix3d R = coords::detail::get_cached_transform()
            .itrf_to_gcrs_matrix(epoch);
        const Eigen::Matrix3d H_eci = R * H_ecef * R.transpose();
        return {a_eci, H_eci};
    }

    /// Earth-gravity acceleration and its 3×3 position Jacobian ∂a/∂r, both in
    /// GCRS. Zero when gravity is off.
    ///
    /// For degree <= 1 (point mass) the closed form is evaluated directly in
    /// GCRS (a sphere looks the same in any frame). Otherwise the gravity
    /// engine works in ITRF and the results are rotated back:
    /// H_gcrs = R · H_itrf · Rᵀ.
    [[nodiscard]] std::pair<Vector3d, Eigen::Matrix3d>
    earth_gravity_acceleration_with_jacobian(
        const Epoch& epoch,
        const StateVector& state_gcrs) const
    {
        if (!flags_.earth_gravity) {
            return {Vector3d::Zero(), Eigen::Matrix3d::Zero()};
        }

        if (gravity_max_degree_ <= 1) {
            // Point mass: ∂a/∂r = -GM/r³ · (I − 3·r̂·r̂ᵀ).
            const Vector3d r = state_gcrs.position;
            const double r_norm = r.norm();
            const double r3 = r_norm * r_norm * r_norm;
            const double r5 = r3 * r_norm * r_norm;
            const Vector3d a = -constants::earth::GM / r3 * r;
            Eigen::Matrix3d J = -constants::earth::GM / r3 * Eigen::Matrix3d::Identity()
                                + 3.0 * constants::earth::GM / r5 * (r * r.transpose());
            return {a, J};
        }

            // EGM2008 path: the engine works in ITRF.
        // 1. Position GCRS → ITRF
        const auto state_itrf = gcrs_state_to_itrf(state_gcrs, epoch);
        // 2. Engine returns acceleration + Hessian in ITRF
        const auto h_ecef = gravity::EGM2008Gravity::instance()
            .compute_with_hessian(state_itrf.position, gravity_max_degree_);
        // 3. Acceleration: ITRF → GCRS (vector rotation)
        const Vector3d a_eci = itrf_vector_to_gcrs(h_ecef.acceleration, epoch);
        // 4. Hessian: H_gcrs = R · H_itrf · Rᵀ, R = ITRF→GCRS rotation
        //    (full IAU 2006/2000A, cached per thread).
        const Eigen::Matrix3d R = coords::detail::get_cached_transform()
            .itrf_to_gcrs_matrix(epoch);
        const Eigen::Matrix3d H_eci = R * h_ecef.hessian_ecef * R.transpose();
        return {a_eci, H_eci};
    }

    [[nodiscard]] Vector3d third_body_acceleration(
        const Epoch& epoch,
        const StateVector& state_gcrs) const
    {
        return compute_if_enabled(flags_.third_body,
            [&] { return third_body_->acceleration(epoch, state_gcrs.position); });
    }

    [[nodiscard]] Vector3d drag_acceleration(
        const Epoch& epoch,
        const StateVector& state_gcrs) const
    {
        return compute_if_enabled(flags_.drag, [&] {
            return drag_env_
                ? drag_->acceleration(*drag_env_, epoch, state_gcrs)
                : Vector3d::Zero();
        });
    }

    [[nodiscard]] Vector3d radiation_acceleration(
        const Epoch& epoch,
        const StateVector& state_gcrs) const
    {
        return compute_if_enabled(flags_.radiation,
            [&] { return radiation_->acceleration(epoch, state_gcrs.position); });
    }

    [[nodiscard]] Vector3d srp_acceleration(
        const Epoch& epoch,
        const StateVector& state_gcrs) const
    {
        return compute_if_enabled(flags_.radiation,
            [&] { return radiation_->srp_only(epoch, state_gcrs.position); });
    }

    [[nodiscard]] Vector3d albedo_acceleration(
        const Epoch& epoch,
        const StateVector& state_gcrs) const
    {
        return compute_if_enabled(flags_.radiation,
            [&] { return radiation_->albedo_only(epoch, state_gcrs.position); });
    }

    [[nodiscard]] Vector3d ir_acceleration(const StateVector& state_gcrs) const
    {
        return compute_if_enabled(flags_.radiation,
            [&] { return radiation_->ir_only(state_gcrs.position); });
    }

    [[nodiscard]] Vector3d relativistic_acceleration(
        const Epoch& epoch,
        const StateVector& state_gcrs) const
    {
        return compute_if_enabled(flags_.relativistic,
            [&] { return relativistic_->acceleration(epoch, state_gcrs); });
    }

    [[nodiscard]] Vector3d solid_tides_acceleration(
        const Epoch& epoch,
        const StateVector& state_gcrs) const
    {
        return compute_if_enabled(flags_.solid_earth_tides,
            [&] { return compute_solid_tides(epoch, state_gcrs); });
    }

    /// Every force component separately plus the total (m/s², GCRS). With
    /// include_disabled = true, switched-off forces are computed too (for
    /// diagnostics) but still left out of the total.
    [[nodiscard]] ForceBreakdown compute_force_breakdown(
        const Epoch& epoch,
        const StateVector& state_gcrs,
        bool include_disabled = false) const
    {
        return compute_all_forces(epoch, state_gcrs, include_disabled);
    }

    /// True when any force that cannot be derived from a potential is on (drag,
    /// radiation, empirical, or non-conservative relativity).
    [[nodiscard]] bool has_non_conservative_forces() const noexcept {
        return flags_.drag || flags_.radiation || flags_.empirical ||
               (flags_.relativistic && !relativistic_->is_conservative());
    }

    /// Direct access to the force components, to set their own switches.
    [[nodiscard]] gravity::ThirdBodyGravity& third_body() noexcept {
        return *third_body_;
    }
    [[nodiscard]] const gravity::ThirdBodyGravity& third_body() const noexcept {
        return *third_body_;
    }

    [[nodiscard]] AtmosphericDrag& drag() noexcept {
        return *drag_;
    }
    [[nodiscard]] const AtmosphericDrag& drag() const noexcept {
        return *drag_;
    }

    [[nodiscard]] RadiationPressure& radiation() noexcept {
        return *radiation_;
    }
    [[nodiscard]] const RadiationPressure& radiation() const noexcept {
        return *radiation_;
    }

    [[nodiscard]] RelativisticCorrections& relativistic() noexcept {
        return *relativistic_;
    }
    [[nodiscard]] const RelativisticCorrections& relativistic() const noexcept {
        return *relativistic_;
    }

    [[nodiscard]] gravity::SolidEarthTides& solid_tides() noexcept {
        return solid_tides_;
    }
    [[nodiscard]] const gravity::SolidEarthTides& solid_tides() const noexcept {
        return solid_tides_;
    }

private:
    // Empirical acceleration at `epoch`, converted to GCRS if stored in ITRF.
    [[nodiscard]] Vector3d compute_empirical(
        const Epoch& epoch,
        const StateVector& /*state_gcrs*/) const
    {
        if (!empirical_accel_ || empirical_accel_->is_zero()) {
            return Vector3d::Zero();
        }

        const double t_seconds = epoch - propagation_epoch_;
        Vector3d accel = empirical_accel_->accel_at(t_seconds);

        // Convert to GCRS if needed.
        if (empirical_accel_->frame() == EmpiricalAccelFrame::ITRF)
            accel = itrf_vector_to_gcrs(accel, epoch);

        return accel;
    }

    std::unique_ptr<gravity::ThirdBodyGravity> third_body_;
    std::unique_ptr<AtmosphericDrag> drag_;
    std::unique_ptr<RadiationPressure> radiation_;
    std::unique_ptr<RelativisticCorrections> relativistic_;
    std::shared_ptr<EmpiricalAcceleration> empirical_accel_;  // shared: lifetime managed externally via TrajectoryConfig.empirical_accel
    std::shared_ptr<const atmosphere::DragEnvironment> drag_env_;  // null = drag off (live model or tables)
    gravity::SolidEarthTides solid_tides_;

    ForceComponentFlags flags_;   // Consolidated force enable/disable flags
    int gravity_max_degree_;
    int ocean_tide_degree_ = 50;
    coords::ITRFRealization itrf_realization_ = coords::ITRFRealization::ITRF2020;
    Epoch propagation_epoch_;     // Start epoch for time-keyed empirical accelerations
};

/// Function computing the state derivative (velocity, acceleration) from (epoch, state).
using DerivativeFunction = std::function<StateVector(const Epoch&, const StateVector&)>;

/// Wraps a ForceModel as a DerivativeFunction for the integrators: returns
/// (velocity, acceleration) for a GCRS state.
inline DerivativeFunction make_derivative_function(
    std::shared_ptr<ForceModel> force_model)
{
    return [force_model](const Epoch& epoch, const StateVector& state) -> StateVector {
        Vector3d acc = force_model->acceleration(epoch, state);
        return StateVector(state.velocity, acc);
    };
}

} // namespace forces
} // namespace ballistic
