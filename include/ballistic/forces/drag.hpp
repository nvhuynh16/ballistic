#pragma once

// =============================================================================
// Atmospheric Drag Model for Spacecraft
// =============================================================================
//
// "Cannonball" drag: the satellite is treated as an object with one effective
// area, so the acceleration is
//
//     a = -0.5 * rho * (1/beta) * |v_rel| * v_rel,   beta = m / (Cd * A)
//
// where rho is the air density (kg/m³), v_rel the velocity relative to the
// (rotating, possibly windy) air (m/s), and beta the ballistic coefficient
// (kg/m²; mass over drag coefficient Cd times area A — higher beta means less
// drag). rho, wind and beta all come from an atmosphere::DragEnvironment.
//
// References:
// -----------
// [1] Vallado, D.A. (2013). "Fundamentals of Astrodynamics and Applications",
//     4th ed., Microcosm Press, Section 8.6.2 "Atmospheric Drag".
//
// [2] Montenbruck, O. & Gill, E. (2000). "Satellite Orbits: Models, Methods
//     and Applications". Springer, Section 3.5 "Atmospheric Drag".
//
// A drag coefficient Cd of about 2.2 is the textbook value for LEO spacecraft
// (Vallado [1], Table 8-4; Montenbruck [2], p.84), typically 2.0-2.5 depending
// on shape and air composition. This file has no default: Cd enters only
// through the beta the DragEnvironment supplies.
//
// =============================================================================

#include "../core/types.hpp"
#include "../core/constants.hpp"
#include "../core/defaults.hpp"
#include "../coordinates/geodetic.hpp"
#include "../coordinates/earth_rotation.hpp"
#include "../coordinates/iau_transforms.hpp"
#include "../time/epoch.hpp"
#include "../atmosphere/drag_environment.hpp"
#include <memory>
#include <cmath>

namespace ballistic {
namespace forces {

/// Computes atmospheric drag. It holds no atmosphere or satellite data of its
/// own: density, wind and ballistic coefficient are read from the
/// DragEnvironment passed to each call (the only way drag is configured).
class AtmosphericDrag {
public:
    AtmosphericDrag() = default;

    /// Which realization (version) of the Earth-fixed ITRF frame the
    /// environment's positions are in; used for the GCRS → ITRF rotation.
    void set_itrf_realization(coords::ITRFRealization itrf) noexcept {
        itrf_realization_ = itrf;
    }

    /// Drag removes energy from the orbit, so it is not conservative.
    [[nodiscard]] static constexpr bool is_conservative() noexcept {
        return false;
    }

    /// Drag acceleration (m/s², GCRS) for a GCRS state at `epoch`, using
    /// density, wind and beta from `env`:
    ///
    ///     a_itrf = -0.5 * rho * (1/beta) * |v_rel| * v_rel
    ///     v_rel  = v_itrf - wind_itrf
    ///
    /// computed in the Earth-fixed ITRF frame (where the air co-rotates with the
    /// Earth) and rotated back to GCRS. Returns zero when the env has no drag,
    /// the density or beta is not positive, or the relative speed is ~0.
    /// Exceptions from `env.evaluate` (e.g. DataCoverageError past the loaded
    /// space weather) propagate.
    [[nodiscard]] Vector3d acceleration(
        const atmosphere::DragEnvironment& env,
        const Epoch& epoch,
        const StateVector& state_inertial) const
    {
        if (!env.has_drag()) {
            return Vector3d::Zero();
        }

        // Work in ITRF, where the atmosphere is (apart from wind) at rest.
        const StateVector state_itrf = coords::gcrs_to_itrf_full(
            state_inertial, epoch, itrf_realization_);

        const atmosphere::DragSample s =
            env.evaluate(state_itrf.position, epoch, state_itrf.velocity);
        if (s.rho_kg_m3 <= 0.0 || s.beta_kg_m2 <= 0.0) {
            return Vector3d::Zero();
        }

        const Vector3d v_rel = state_itrf.velocity - s.wind_ecef_m_s;
        const double v_rel_mag = v_rel.norm();
        if (v_rel_mag < defaults::VELOCITY_EPSILON) {
            return Vector3d::Zero();
        }

        const double inv_beta = 1.0 / s.beta_kg_m2;
        const Vector3d acc_itrf =
            (-0.5 * s.rho_kg_m3 * inv_beta * v_rel_mag) * v_rel;

        // Rotate the acceleration back to GCRS.
        return coords::itrf_to_gcrs_vector_full(acc_itrf, epoch);
    }

    /// Drag acceleration and its 3×6 Jacobian ∂a/∂[r, v] (GCRS). The position
    /// columns are forward finite differences (re-evaluate with each position
    /// coordinate offset by `fd_dx` meters). The velocity columns are analytic
    /// (matrix M_v below), except when beta depends on velocity (a FlowRegimeCd,
    /// whose Cd varies with Mach number and speed); then they are central
    /// differences with offset `fd_dv` (m/s).
    [[nodiscard]] std::pair<Vector3d, Eigen::Matrix<double, 3, 6>>
    acceleration_with_jacobian(
        const atmosphere::DragEnvironment& env,
        const Epoch& epoch,
        const StateVector& state_inertial,
        double fd_dx = 1.0,
        double fd_dv = 1e-2) const
    {
        using Mat36 = Eigen::Matrix<double, 3, 6>;
        const Vector3d a0 = acceleration(env, epoch, state_inertial);
        Mat36 J = Mat36::Zero();

        if (a0.squaredNorm() < 1e-30) {
            return {a0, J};
        }

        // Position columns: finite differences through acceleration().
        for (int i = 0; i < 3; ++i) {
            StateVector xp = state_inertial;
            xp.position(i) += fd_dx;
            J.col(i) = (acceleration(env, epoch, xp) - a0) / fd_dx;
        }

        if (env.beta_depends_on_velocity()) {
            for (int i = 0; i < 3; ++i) {
                StateVector xp = state_inertial, xm = state_inertial;
                xp.velocity(i) += fd_dv;
                xm.velocity(i) -= fd_dv;
                J.col(3 + i) = (acceleration(env, epoch, xp) - acceleration(env, epoch, xm)) / (2.0 * fd_dv);
            }
            return {a0, J};
        }

        // Velocity columns, analytic: ∂a/∂v_rel = k·(|v|·I + v·vᵀ/|v|) in ITRF,
        // rotated to GCRS with R (GCRS → ITRF).
        const StateVector state_itrf = coords::gcrs_to_itrf_full(
            state_inertial, epoch, itrf_realization_);
        const atmosphere::DragSample s =
            env.evaluate(state_itrf.position, epoch, state_itrf.velocity);

        if (s.rho_kg_m3 > 0.0 && s.beta_kg_m2 > 0.0) {
            const Vector3d v_rel = state_itrf.velocity - s.wind_ecef_m_s;
            const double v_rel_mag = v_rel.norm();
            if (v_rel_mag >= defaults::VELOCITY_EPSILON) {
                const double inv_beta = 1.0 / s.beta_kg_m2;
                const double k = -0.5 * s.rho_kg_m3 * inv_beta;
                const Eigen::Matrix3d M_v = k * (
                    v_rel_mag * Eigen::Matrix3d::Identity()
                  + (v_rel * v_rel.transpose()) / v_rel_mag
                );
                const Eigen::Matrix3d R =
                    coords::detail::get_cached_transform()
                        .gcrs_to_itrf_matrix(epoch);
                J.block<3, 3>(0, 3) = R.transpose() * M_v * R;
            }
        }

        return {a0, J};
    }

private:
    coords::ITRFRealization itrf_realization_ = coords::ITRFRealization::ITRF2020;
};

} // namespace forces
} // namespace ballistic
