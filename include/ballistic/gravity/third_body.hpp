#pragma once

// =============================================================================
// Third-Body Gravitational Perturbations
// =============================================================================
//
// References:
// -----------
// [1] Vallado, D.A. (2013). "Fundamentals of Astrodynamics and Applications",
//     4th Edition. Microcosm Press.
//     - Third-body perturbation: Section 8.6, Equation 8-36
//
// [2] Montenbruck, O. & Gill, E. (2000). "Satellite Orbits: Models, Methods,
//     and Applications". Springer-Verlag.
//     - Third-body acceleration: Chapter 3.3, Equation 3.34
//
// Mathematical Background:
// ------------------------
// For a satellite at position r relative to Earth, and a third body (Sun, Moon)
// at position r_body relative to Earth, the perturbing acceleration is:
//
//   a = GM_body * [ (r_body - r) / |r_body - r|^3 - r_body / |r_body|^3 ]
//        ^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^
//        body's GM   direct attraction on sat      indirect (Earth ref)
//
// The indirect term accounts for the acceleration of the Earth-centered frame
// due to the third body's attraction on Earth itself: what perturbs the orbit
// is only the DIFFERENCE between the pull on the satellite and the pull on the
// Earth.
//
// Positions are in the GCRS (Geocentric Celestial Reference System: Earth-
// centered, non-rotating axes fixed relative to distant quasars), in meters.
// The code calls this frame "ECI" (Earth-centered inertial).
//
// Typical magnitudes at LEO (400 km altitude):
// - Sun:     ~5e-7 m/s² (periodic daily/yearly variations)
// - Moon:    ~1e-6 m/s² (periodic monthly variations)
// - Jupiter: ~5e-12 m/s² (negligible for most applications)
//
// GM (gravitational constant times the body's mass) values from constants.hpp
// (JPL DE440, gm_de440.tpc):
// - GM_Sun     = 1.32712440041279419e+20 m³/s²
// - GM_Moon    = 4.9028001184575496e+12 m³/s²
// - GM_Jupiter = 1.26712764e+17 m³/s² (system barycenter)
//
// =============================================================================

#include "../core/types.hpp"
#include "../core/constants.hpp"
#include "../time/epoch.hpp"
#include "../ephemeris/jpl_ephemeris.hpp"
#include <memory>
#include <cmath>

namespace ballistic {
namespace gravity {

/// Perturbing acceleration from the Sun, Moon and (optionally) Jupiter.
///
/// Sun and Moon are on by default, Jupiter off. Body positions come from an
/// ephemeris provider (by default the embedded JPL DE440 ephemeris, see
/// ephemeris/jpl_ephemeris.hpp). Reference: Vallado [1], Section 8.6.
/// Every method that takes an epoch can throw what the ephemeris throws
/// (DataCoverageError when the epoch is outside the loaded ephemeris).
class ThirdBodyGravity {
public:
    /// Whether one body is included, and its GM.
    struct BodyConfig {
        bool enabled = true;
        double gm;  ///< gravitational parameter [m^3/s^2]
    };

    /// Use the library's default ephemeris (loads the embedded DE440 if
    /// nothing is loaded yet).
    ThirdBodyGravity()
        : ephemeris_(ephemeris::get_default_ephemeris_provider())
        , sun_config_{true, constants::sun::GM}
        , moon_config_{true, constants::moon::GM}
        , jupiter_config_{false, constants::jupiter::GM}
    {}

    /// Use the given ephemeris for the body positions.
    explicit ThirdBodyGravity(std::shared_ptr<ephemeris::EphemerisProvider> eph)
        : ephemeris_(std::move(eph))
        , sun_config_{true, constants::sun::GM}
        , moon_config_{true, constants::moon::GM}
        , jupiter_config_{false, constants::jupiter::GM}
    {}

    /// Include or exclude a body.
    void enable_sun(bool enable) noexcept { sun_config_.enabled = enable; }
    void enable_moon(bool enable) noexcept { moon_config_.enabled = enable; }
    void enable_jupiter(bool enable) noexcept { jupiter_config_.enabled = enable; }

    /// Override a body's GM [m^3/s^2].
    void set_sun_gm(double gm) noexcept { sun_config_.gm = gm; }
    void set_moon_gm(double gm) noexcept { moon_config_.gm = gm; }
    void set_jupiter_gm(double gm) noexcept { jupiter_config_.gm = gm; }

    /// Sum of the enabled bodies' perturbing accelerations [m/s^2], GCRS.
    /// @param epoch         time of the evaluation
    /// @param position_eci  satellite position, GCRS [m]
    [[nodiscard]] Vector3d acceleration(
        const Epoch& epoch,
        const Vector3d& position_eci) const
    {
        Vector3d acc = Vector3d::Zero();
        for_each_enabled_body(epoch, [&](double gm, const Vector3d& r_body) {
            acc += third_body_acceleration(position_eci, r_body, gm);
        });
        return acc;
    }

    /// One body's perturbing acceleration [m/s^2], GCRS, whether or not that
    /// body is enabled (for analysis).
    [[nodiscard]] Vector3d sun_acceleration(
        const Epoch& epoch,
        const Vector3d& position_eci) const
    {
        Vector3d r_sun = ephemeris_->sun_position(epoch);
        return third_body_acceleration(position_eci, r_sun, sun_config_.gm);
    }

    [[nodiscard]] Vector3d moon_acceleration(
        const Epoch& epoch,
        const Vector3d& position_eci) const
    {
        Vector3d r_moon = ephemeris_->moon_position(epoch);
        return third_body_acceleration(position_eci, r_moon, moon_config_.gm);
    }

    [[nodiscard]] Vector3d jupiter_acceleration(
        const Epoch& epoch,
        const Vector3d& position_eci) const
    {
        Vector3d r_jupiter = ephemeris_->jupiter_position(epoch);
        return third_body_acceleration(position_eci, r_jupiter, jupiter_config_.gm);
    }

    /// Body position [m] from the ephemeris, GCRS axes (for analysis).
    [[nodiscard]] Vector3d sun_position(const Epoch& epoch) const {
        return ephemeris_->sun_position(epoch);
    }

    [[nodiscard]] Vector3d moon_position(const Epoch& epoch) const {
        return ephemeris_->moon_position(epoch);
    }

    /// Whether a body is included.
    [[nodiscard]] bool sun_enabled() const noexcept { return sun_config_.enabled; }
    [[nodiscard]] bool moon_enabled() const noexcept { return moon_config_.enabled; }
    [[nodiscard]] bool jupiter_enabled() const noexcept { return jupiter_config_.enabled; }

    /// Third-body gravity derives from a potential (true).
    [[nodiscard]] static constexpr bool is_conservative() noexcept {
        return true;
    }

    /// Acceleration [m/s^2] AND its 3x3 Jacobian ∂a/∂r_sat [1/s^2] (how the
    /// acceleration changes per meter of satellite position), both GCRS.
    /// The derivative with respect to velocity is zero (gravity does not
    /// depend on velocity).
    ///
    /// Per body (derivation at third_body_accel_and_jac below):
    ///
    ///   ∂a/∂r_sat = Σ_body GM·[−I/d³ + 3·Δ·Δᵀ/d⁵]      (Δ = r_body − r_sat,
    ///                                                    d = |Δ|)
    ///
    /// Properties (checked by tests/unit/test_force_jacobians.cpp):
    ///   - symmetric
    ///   - traceless  (tr = −3/d³ + 3·d²/d⁵ = 0)
    ///   - eigenvalues +2GM/d³ along Δ and −GM/d³ in the two perpendicular directions
    [[nodiscard]] std::pair<Vector3d, Eigen::Matrix3d>
    acceleration_with_jacobian(
        const Epoch& epoch,
        const Vector3d& position_eci) const
    {
        Vector3d a_total = Vector3d::Zero();
        Eigen::Matrix3d J_total = Eigen::Matrix3d::Zero();
        for_each_enabled_body(epoch, [&](double gm, const Vector3d& r_body) {
            auto [a, J] = third_body_accel_and_jac(position_eci, r_body, gm);
            a_total += a;
            J_total += J;
        });
        return {a_total, J_total};
    }

private:
    // Call fn(gm, r_body_eci) for each enabled body.
    template<typename F>
    void for_each_enabled_body(const Epoch& epoch, F&& fn) const {
        if (sun_config_.enabled)     fn(sun_config_.gm,     ephemeris_->sun_position(epoch));
        if (moon_config_.enabled)    fn(moon_config_.gm,    ephemeris_->moon_position(epoch));
        if (jupiter_config_.enabled) fn(jupiter_config_.gm, ephemeris_->jupiter_position(epoch));
    }

    std::shared_ptr<ephemeris::EphemerisProvider> ephemeris_;
    BodyConfig sun_config_;
    BodyConfig moon_config_;
    BodyConfig jupiter_config_;

    // Third-body acceleration formula (Vallado [1], Equation 8-36):
    // a = GM * [ (r_body - r_sat) / |r_body - r_sat|^3 - r_body / |r_body|^3 ]
    //
    // First term: direct gravitational attraction of body on satellite
    // Second term: indirect effect (body's attraction on Earth-centered frame)
    [[nodiscard]] static Vector3d third_body_acceleration(
        const Vector3d& r_sat,      // Satellite position (ECI, from Earth center)
        const Vector3d& r_body,     // Body position (ECI, from Earth center)
        double gm) noexcept
    {
        // Vector from satellite to body
        Vector3d r_rel = r_body - r_sat;

        double r_rel_norm = r_rel.norm();
        double r_body_norm = r_body.norm();

        // Degenerate geometry (satellite at the body, or body at the center): zero
        if (r_rel_norm < 1.0 || r_body_norm < 1.0) {
            return Vector3d::Zero();
        }

        double r_rel_3 = r_rel_norm * r_rel_norm * r_rel_norm;
        double r_body_3 = r_body_norm * r_body_norm * r_body_norm;

        // Direct + indirect terms
        return gm * (r_rel / r_rel_3 - r_body / r_body_3);
    }

    // Per-body (acceleration, Jacobian) computed jointly so they share
    // the d² and d³ work. Returns Zero/Zero for degenerate inputs (matches
    // third_body_acceleration's r_rel_norm < 1.0 guard).
    //
    // Derivation:
    //   a = GM·[Δ/d³ − r_body/D³],  Δ = r_body − r_sat
    //   ∂Δ/∂r_sat = −I
    //   ∂(d⁻³)/∂r_sat = 3·Δ/d⁵                (from ∂d/∂r = −Δ/d via d² = Δ·Δ)
    //   ∂(Δᵢ d⁻³)/∂r_sat_j = (−δᵢⱼ)/d³ + 3·Δᵢ·Δⱼ/d⁵
    //   The −r_body/D³ term has no r_sat dependence → contributes 0 to J.
    [[nodiscard]] static std::pair<Vector3d, Eigen::Matrix3d>
    third_body_accel_and_jac(
        const Vector3d& r_sat,
        const Vector3d& r_body,
        double gm) noexcept
    {
        const Vector3d r_rel = r_body - r_sat;
        const double r_rel_norm = r_rel.norm();
        const double r_body_norm = r_body.norm();

        if (r_rel_norm < 1.0 || r_body_norm < 1.0) {
            return {Vector3d::Zero(), Eigen::Matrix3d::Zero()};
        }

        const double d2 = r_rel_norm * r_rel_norm;
        const double d3 = d2 * r_rel_norm;
        const double d5 = d3 * d2;
        const double D3 = r_body_norm * r_body_norm * r_body_norm;

        const Vector3d a = gm * (r_rel / d3 - r_body / D3);
        const Eigen::Matrix3d J =
              (-gm / d3) * Eigen::Matrix3d::Identity()
            + (3.0 * gm / d5) * (r_rel * r_rel.transpose());
        return {a, J};
    }
};

/// One-off third-body acceleration [m/s^2] at a GCRS position [m] using the
/// default ephemeris. Builds a ThirdBodyGravity per call; keep one object
/// instead when calling repeatedly.
[[nodiscard]] inline Vector3d third_body_perturbation(
    const Epoch& epoch,
    const Vector3d& position_eci,
    bool include_sun = true,
    bool include_moon = true,
    bool include_jupiter = false)
{
    ThirdBodyGravity tb;
    tb.enable_sun(include_sun);
    tb.enable_moon(include_moon);
    tb.enable_jupiter(include_jupiter);
    return tb.acceleration(epoch, position_eci);
}

} // namespace gravity
} // namespace ballistic
