#pragma once

// =============================================================================
// Radiation Pressure Models for Spacecraft
// =============================================================================
//
// Light carries momentum, so light hitting a satellite pushes it. Three sources
// are modeled, each with the "cannonball" approximation (the satellite is a
// sphere-like object with one effective area facing every direction):
// - Solar radiation pressure (SRP): direct sunlight
// - Earth albedo: sunlight reflected by the Earth
// - Earth infrared (IR): heat radiated by the Earth
//
// All positions are Earth-centered inertial (GCRS) in meters; accelerations are
// m/s² in the same frame. "eci" in parameter names means GCRS here.
//
// References:
// -----------
// [1] Montenbruck, O. & Gill, E. (2000). "Satellite Orbits: Models, Methods
//     and Applications". Springer, Section 3.4 "Solar Radiation Pressure".
//
// [2] Knocke, P.C., Ries, J.C., & Tapley, B.D. (1988). "Earth radiation
//     pressure effects on satellites". Proceedings of the AIAA/AAS
//     Astrodynamics Conference, AIAA 88-4292.
//
// [3] Vallado, D.A. (2013). "Fundamentals of Astrodynamics and Applications",
//     4th ed., Microcosm Press, Section 8.6.
//
// =============================================================================

#include "../core/types.hpp"
#include "../core/constants.hpp"
#include "../core/defaults.hpp"
#include "../time/epoch.hpp"
#include "../ephemeris/jpl_ephemeris.hpp"
#include <cmath>
#include <memory>

namespace ballistic {
namespace forces {

// =============================================================================
// Physical Constants for Radiation Pressure
// =============================================================================

namespace radiation {
    // Solar radiation pressure at 1 AU (astronomical unit, the mean Earth-Sun distance)
    // P = S / c, where S = 1361 W/m² (solar constant), c = speed of light
    // Reference: Kopp, G. & Lean, J.L. (2011). "A new, lower value of total
    //   solar irradiance: Evidence and climate significance". GRL 38, L01706.
    //   SORCE/TIM measurement: 1360.8 ± 0.5 W/m², rounded to 1361 W/m².
    // Note: Montenbruck & Gill (2000) Eq. 3.75 used the older value of 1367 W/m²
    //   from pre-SORCE measurements. The ~0.4% difference is negligible for
    //   cannonball SRP but the 1361 value is the accepted modern standard.
    constexpr double SOLAR_CONSTANT = 1361.0;  // W/m² at 1 AU
    constexpr double SOLAR_PRESSURE_1AU = SOLAR_CONSTANT / constants::SPEED_OF_LIGHT;  // ~4.54e-6 N/m²

    // Earth albedo (average reflectivity)
    // Reference: Knocke et al. (1988)
    constexpr double EARTH_ALBEDO = 0.3;  // Average Earth albedo coefficient

    // Earth infrared emission
    // Average outgoing longwave radiation: ~240 W/m²
    // Reference: Montenbruck & Gill (2000), Section 3.4.3
    constexpr double EARTH_IR_FLUX = 240.0;  // W/m² average
    constexpr double EARTH_IR_PRESSURE = EARTH_IR_FLUX / constants::SPEED_OF_LIGHT;  // ~8.0e-7 N/m²
}

// The optical/thermal properties (area, reflectivity, emissivity) are the
// SatelliteProperties of core/types.hpp.
using SatelliteRadiationProperties = SatelliteProperties;

// =============================================================================
// Shadow Function
// =============================================================================

/// Earth's shadow, modeled as a cylinder of Earth radius behind the Earth
/// (no penumbra, the partly lit zone). Returns 1.0 in sunlight, 0.0 in shadow.
/// `sat_pos_eci` and `sun_pos_eci` are GCRS positions in meters.
/// Reference: Montenbruck & Gill (2000), Section 3.4.2.
[[nodiscard]] inline double compute_shadow_factor(
    const Vector3d& sat_pos_eci,
    const Vector3d& sun_pos_eci) noexcept
{
    // Vector from Earth center to satellite
    const double r_sat = sat_pos_eci.norm();

    // Vector from Earth center to Sun
    const double r_sun = sun_pos_eci.norm();

    // Unit vector from Earth to Sun
    const Vector3d e_sun = sun_pos_eci / r_sun;

    // Projection of satellite position onto Sun direction
    const double s = sat_pos_eci.dot(e_sun);

    // If satellite is on sun-side of Earth, no shadow possible
    if (s > 0) {
        return 1.0;
    }

    // Distance from satellite to Earth-Sun axis
    const Vector3d d_vec = sat_pos_eci - s * e_sun;
    const double d = d_vec.norm();

    // Earth radius (use equatorial for simplicity)
    const double R_earth = constants::earth::RADIUS_EQUATORIAL;

    // Check if satellite is in cylindrical shadow
    if (d < R_earth) {
        return 0.0;  // In shadow (umbra)
    }

    return 1.0;  // In sunlight
}

// =============================================================================
// Solar Radiation Pressure (Cannonball Model)
// =============================================================================

/// Solar radiation pressure acceleration (m/s², GCRS), cannonball model.
/// Reference: Montenbruck & Gill (2000), Eq. 3.75.
///
///   a_srp = shadow * P_1AU * (AU/r)² * (A/m) * C_R * s_hat
///
/// where:
///   P_1AU = S / c, the pressure of sunlight at 1 AU (S ~1361 W/m², c = speed
///           of light; ~4.54e-6 N/m²)
///   r     = distance from Sun to satellite (m)
///   A/m   = area-to-mass ratio (m²/kg): props.srp_area / mass
///   C_R   = radiation pressure coefficient = 1 + props.srp_reflectivity
///           (1 for a perfect absorber, 2 for a perfect mirror)
///   s_hat = unit vector from Sun to satellite (the push is away from the Sun)
///
/// `mass` is in kg; the Sun position comes from `ephemeris`. Returns zero in
/// shadow or when mass or area is not positive.
[[nodiscard]] inline Vector3d srp_acceleration(
    const Epoch& epoch,
    const Vector3d& sat_pos_eci,
    double mass,
    const SatelliteRadiationProperties& props,
    const ephemeris::EphemerisProvider& ephemeris)
{
    if (mass <= 0.0 || props.srp_area <= 0.0) {
        return Vector3d::Zero();
    }

    // Get Sun position in ECI
    const Vector3d sun_pos = ephemeris.sun_position(epoch);

    // Check shadow
    const double shadow = compute_shadow_factor(sat_pos_eci, sun_pos);
    if (shadow < defaults::VISIBILITY_EPSILON) {
        return Vector3d::Zero();  // In complete shadow
    }

    // Vector from Sun to satellite
    const Vector3d r_sun_sat = sat_pos_eci - sun_pos;
    const double r = r_sun_sat.norm();

    // Unit vector from Sun to satellite
    const Vector3d s_hat = r_sun_sat / r;

    // Radiation pressure coefficient: C_R = 1 + reflectivity
    // 1.0 for perfect absorber, 2.0 for perfect mirror
    const double C_R = 1.0 + props.srp_reflectivity;

    // Solar pressure scaled by distance from Sun
    // P = P_1AU * (AU / r)²
    const double AU = constants::sun::AU;
    const double distance_factor = (AU * AU) / (r * r);
    const double P = radiation::SOLAR_PRESSURE_1AU * distance_factor;

    // Area-to-mass ratio
    const double A_over_m = props.srp_area / mass;

    // SRP acceleration: a = P * (A/m) * C_R * s_hat
    // SRP pushes satellite away from Sun, in direction of s_hat
    // (s_hat = unit vector from Sun to satellite)
    const Vector3d acc = shadow * P * A_over_m * C_R * s_hat;

    return acc;
}

/// SRP acceleration AND its analytic 3×3 position Jacobian ∂a/∂r_sat (1/s²).
/// SRP does not depend on velocity. Same inputs as srp_acceleration().
///
/// Math:
///   Let s = sat - sun, r = |s|, K = SOLAR_PRESSURE_1AU · AU² · (A/m) · C_R.
///   Then a_srp = shadow · K · (s/r³).
///
///   ∂s/∂r_sat = +I  (opposite sign to third-body's ∂Δ/∂r_sat = -I,
///                    because s = r_sat - r_sun has r_sat with a plus sign).
///   ∂(r²)/∂r_sat = 2s   →  ∂r/∂r_sat = s/r.
///   ∂(r⁻³)/∂r_sat = -3·s/r⁵.
///   ∂(sᵢ/r³)/∂r_sat_j = δᵢⱼ/r³ + sᵢ·(-3·sⱼ/r⁵) = δᵢⱼ/r³ - 3·sᵢ·sⱼ/r⁵.
///
/// Matrix form (away from the shadow boundary):
///   ∂a_srp/∂r_sat = shadow · K · [I/r³ - 3·ssᵀ/r⁵]
///
/// The bracket is symmetric with zero trace; its eigenvalues are -2/r³ along ŝ
/// and +1/r³ in the two perpendicular directions. It has the opposite sign to
/// a gravitational tidal tensor because radiation pushes away from the Sun
/// while gravity pulls toward it.
///
/// Shadow: the hard cylindrical cutoff of compute_shadow_factor makes ∂a/∂r
/// discontinuous at the shadow boundary. Inside the shadow a = 0 and J = 0.
/// A trajectory that crosses the boundary gets a Jacobian error there; one
/// that stays in sunlight (or in shadow) is unaffected.
[[nodiscard]] inline std::pair<Vector3d, Eigen::Matrix3d>
srp_acceleration_with_jacobian(
    const Epoch& epoch,
    const Vector3d& sat_pos_eci,
    double mass,
    const SatelliteRadiationProperties& props,
    const ephemeris::EphemerisProvider& ephemeris)
{
    if (mass <= 0.0 || props.srp_area <= 0.0) {
        return {Vector3d::Zero(), Eigen::Matrix3d::Zero()};
    }

    const Vector3d sun_pos = ephemeris.sun_position(epoch);
    const double shadow = compute_shadow_factor(sat_pos_eci, sun_pos);
    if (shadow < defaults::VISIBILITY_EPSILON) {
        return {Vector3d::Zero(), Eigen::Matrix3d::Zero()};
    }

    const Vector3d s = sat_pos_eci - sun_pos;          // sat - sun
    const double r = s.norm();
    const double r2 = r * r;
    const double r3 = r2 * r;
    const double r5 = r3 * r2;

    const double C_R = 1.0 + props.srp_reflectivity;
    const double AU = constants::sun::AU;
    const double K = radiation::SOLAR_PRESSURE_1AU * (AU * AU)
                   * (props.srp_area / mass) * C_R;

    const Vector3d a = shadow * K * s / r3;
    const Eigen::Matrix3d J =
          (shadow * K / r3) * Eigen::Matrix3d::Identity()
        - (3.0 * shadow * K / r5) * (s * s.transpose());

    return {a, J};
}

// =============================================================================
// Earth Albedo Radiation Pressure
// =============================================================================

/// Earth albedo radiation pressure acceleration (m/s², GCRS): sunlight
/// reflected by the Earth (~30% of what it receives) pushes the satellite
/// away from the Earth. Simplified model: uniform reflectivity, force radially
/// outward, strength proportional to how sunlit the point below the satellite
/// is (cos of the Earth-centered angle between satellite and Sun; zero on the
/// night side).
/// Reference: Knocke et al. (1988); Montenbruck & Gill (2000), Section 3.4.3.
///
/// Inputs as for srp_acceleration(); uses props.albedo_reflectivity
/// (C_R = 1 + albedo_reflectivity) and props.srp_area. Returns zero on the night
/// side, at or below the Earth's surface, or when mass or area is not positive.
[[nodiscard]] inline Vector3d albedo_acceleration(
    const Epoch& epoch,
    const Vector3d& sat_pos_eci,
    double mass,
    const SatelliteRadiationProperties& props,
    const ephemeris::EphemerisProvider& ephemeris)
{
    if (mass <= 0.0 || props.srp_area <= 0.0) {
        return Vector3d::Zero();
    }

    // Get Sun and satellite positions
    const Vector3d sun_pos = ephemeris.sun_position(epoch);
    const double r_sat = sat_pos_eci.norm();

    // Earth radius
    const double R_earth = constants::earth::RADIUS_EQUATORIAL;

    // Satellite must be above Earth surface
    if (r_sat <= R_earth) {
        return Vector3d::Zero();
    }

    // Unit vectors
    const Vector3d e_sat = sat_pos_eci / r_sat;  // Earth to satellite
    const Vector3d e_sun = sun_pos.normalized();  // Earth to Sun

    // Cosine of angle between satellite and Sun (as seen from Earth center)
    const double cos_phi = e_sat.dot(e_sun);

    // Only the sunlit hemisphere contributes; the strength falls to zero
    // smoothly at the day/night boundary.
    double visibility = 0.0;
    if (cos_phi > 0.0) {
        visibility = cos_phi;  // Proportional to how "sunlit" the subsatellite point is
    }

    if (visibility < defaults::VISIBILITY_EPSILON) {
        return Vector3d::Zero();
    }

    // Simplified albedo model: the Earth reflects fraction a of the solar
    // flux S. Reflected flux at the satellite:
    //   F_albedo = a * S * (R/r)² * cos_phi / pi
    // with R = Earth radius, r = satellite distance from the Earth's center.

    const double altitude_factor = (R_earth * R_earth) / (r_sat * r_sat);
    const double F_albedo = radiation::EARTH_ALBEDO * radiation::SOLAR_CONSTANT * altitude_factor * visibility / constants::PI;

    // Pressure from albedo
    const double P_albedo = F_albedo / constants::SPEED_OF_LIGHT;

    // Radiation pressure coefficient for albedo (same form as SRP)
    const double C_R = 1.0 + props.albedo_reflectivity;

    // Area-to-mass ratio
    const double A_over_m = props.srp_area / mass;

    // Albedo force is directed radially outward from Earth
    // (simplified assumption - actual direction depends on Earth surface geometry)
    const Vector3d acc = P_albedo * A_over_m * C_R * e_sat;

    return acc;
}

/// Albedo acceleration AND its analytic 3×3 Jacobian ∂a/∂r_sat (1/s²).
/// Albedo does not depend on velocity. Same inputs as albedo_acceleration().
///
/// Math: with s = r_sat, r = |s|, e_sun = Sun direction (treated as constant),
/// and m(s) = s · e_sun:
///
///   visibility = max(cos_phi, 0)        where cos_phi = (s/r) · e_sun = m/r
///   a_albedo = K · R_E² · (m/r) · (s/r) · (1/r²) · {coefs}
///            = K_full · m · s / r⁴
///
/// where K_full = EARTH_ALBEDO·SOLAR_CONSTANT·R_E²·(A/m)·C_R / (PI·c).
///
/// Component-wise: a_i = K_full · m · s_i / r⁴.
/// Product rule on (m, s_i, r⁻⁴) with ∂m/∂s_j = e_sun_j:
///
///   ∂a_i/∂s_j = K_full · [e_sun_j · s_i / r⁴ + m · δᵢⱼ / r⁴ - 4m · sᵢ·sⱼ / r⁶]
///
/// Matrix form:
///
///   ∂a_albedo/∂s = K_full · [s · e_sunᵀ / r⁴ + (m/r⁴) · I - (4m/r⁶) · s · sᵀ]
///
/// Not symmetric in general: the s·e_sunᵀ term means the force is not the
/// gradient of a potential (the cos_phi factor makes albedo non-conservative).
/// On the night side (visibility ≤ ε) it returns (Zero, Zero), consistent with
/// albedo_acceleration().
[[nodiscard]] inline std::pair<Vector3d, Eigen::Matrix3d>
albedo_acceleration_with_jacobian(
    const Epoch& epoch,
    const Vector3d& sat_pos_eci,
    double mass,
    const SatelliteRadiationProperties& props,
    const ephemeris::EphemerisProvider& ephemeris)
{
    if (mass <= 0.0 || props.srp_area <= 0.0) {
        return {Vector3d::Zero(), Eigen::Matrix3d::Zero()};
    }

    const Vector3d sun_pos = ephemeris.sun_position(epoch);
    const double R_earth = constants::earth::RADIUS_EQUATORIAL;
    const double r_sat = sat_pos_eci.norm();

    if (r_sat <= R_earth) {
        return {Vector3d::Zero(), Eigen::Matrix3d::Zero()};
    }

    const Vector3d e_sun = sun_pos.normalized();
    const Vector3d e_sat = sat_pos_eci / r_sat;
    const double cos_phi = e_sat.dot(e_sun);

    if (cos_phi <= 0.0 || cos_phi < defaults::VISIBILITY_EPSILON) {
        return {Vector3d::Zero(), Eigen::Matrix3d::Zero()};
    }

    // Constants (matches albedo_acceleration above).
    const double C_R = 1.0 + props.albedo_reflectivity;
    const double A_over_m = props.srp_area / mass;
    const double K_full = radiation::EARTH_ALBEDO * radiation::SOLAR_CONSTANT
                        * R_earth * R_earth * A_over_m * C_R
                        / (constants::PI * constants::SPEED_OF_LIGHT);

    const double m_dot = sat_pos_eci.dot(e_sun);  // m = s · e_sun
    const double r2 = r_sat * r_sat;
    const double r3 = r2 * r_sat;
    const double r4 = r2 * r2;
    const double r6 = r3 * r3;

    // Acceleration: a = K_full · m · s / r⁴
    const Vector3d a = (K_full * m_dot / r4) * sat_pos_eci;

    // Jacobian:
    //   J = K_full · [s · e_sunᵀ / r⁴ + (m/r⁴) · I - (4m/r⁶) · s · sᵀ]
    const Eigen::Matrix3d J =
          (K_full / r4) * (sat_pos_eci * e_sun.transpose())
        + (K_full * m_dot / r4) * Eigen::Matrix3d::Identity()
        - (4.0 * K_full * m_dot / r6) * (sat_pos_eci * sat_pos_eci.transpose());

    return {a, J};
}

// =============================================================================
// Earth Infrared Radiation Pressure
// =============================================================================

/// Earth infrared radiation pressure acceleration (m/s², GCRS): the Earth's
/// heat emission (~240 W/m² average outgoing longwave radiation) pushes the
/// satellite radially outward, most strongly at low altitude.
/// Reference: Montenbruck & Gill (2000), Section 3.4.3.
///
/// `sat_pos_eci` in meters, `mass` in kg; uses props.srp_area and
/// props.ir_emissivity. Returns zero at or below the surface or when mass or
/// area is not positive.
[[nodiscard]] inline Vector3d ir_acceleration(
    const Vector3d& sat_pos_eci,
    double mass,
    const SatelliteRadiationProperties& props)
{
    if (mass <= 0.0 || props.srp_area <= 0.0) {
        return Vector3d::Zero();
    }

    // Earth radius
    const double R_earth = constants::earth::RADIUS_EQUATORIAL;

    // Satellite distance from Earth center
    const double r_sat = sat_pos_eci.norm();

    // Must be above Earth surface
    if (r_sat <= R_earth) {
        return Vector3d::Zero();
    }

    // Unit vector from Earth to satellite
    const Vector3d e_sat = sat_pos_eci / r_sat;

    // Earth IR flux scales with the solid angle of the Earth seen from the
    // satellite: Omega ≈ pi * (R/r)² for r >> R
    const double altitude_factor = (R_earth * R_earth) / (r_sat * r_sat);

    // IR pressure at satellite location
    const double P_ir = radiation::EARTH_IR_PRESSURE * altitude_factor;

    // IR absorption/emission coefficient
    // For thermal radiation, use emissivity (not optical reflectivity)
    const double C_IR = props.ir_emissivity;

    // Area-to-mass ratio
    const double A_over_m = props.srp_area / mass;

    // IR force is radially outward from Earth
    const Vector3d acc = P_ir * A_over_m * C_IR * e_sat;

    return acc;
}

/// IR acceleration AND its analytic 3×3 Jacobian ∂a/∂r_sat (1/s²).
/// IR does not depend on velocity. Same inputs as ir_acceleration().
///
/// Math: a_ir = K_ir · R_E² · s / r³  where s = r_sat, r = |s|, and
///   K_ir = EARTH_IR_PRESSURE · (A/m) · C_IR  (constant w.r.t. s).
///
/// Same form as SRP: ∂(s/r³)/∂s = I/r³ − 3·s·sᵀ/r⁵, so
///
///   ∂a_ir/∂s = K_ir · R_E² · [I/r³ − 3·s·sᵀ/r⁵]
///
/// Symmetric with zero trace; eigenvalues -2/r³ along ŝ (radial) and +1/r³ in
/// the two perpendicular directions, as for SRP (IR also weakens as 1/r²).
[[nodiscard]] inline std::pair<Vector3d, Eigen::Matrix3d>
ir_acceleration_with_jacobian(
    const Vector3d& sat_pos_eci,
    double mass,
    const SatelliteRadiationProperties& props)
{
    if (mass <= 0.0 || props.srp_area <= 0.0) {
        return {Vector3d::Zero(), Eigen::Matrix3d::Zero()};
    }

    const double R_earth = constants::earth::RADIUS_EQUATORIAL;
    const double r_sat = sat_pos_eci.norm();
    if (r_sat <= R_earth) {
        return {Vector3d::Zero(), Eigen::Matrix3d::Zero()};
    }

    const double r2 = r_sat * r_sat;
    const double r3 = r2 * r_sat;
    const double r5 = r3 * r2;

    const double C_IR = props.ir_emissivity;
    const double A_over_m = props.srp_area / mass;
    const double K = radiation::EARTH_IR_PRESSURE * R_earth * R_earth * A_over_m * C_IR;

    const Vector3d a = K * sat_pos_eci / r3;
    const Eigen::Matrix3d J =
          (K / r3) * Eigen::Matrix3d::Identity()
        - (3.0 * K / r5) * (sat_pos_eci * sat_pos_eci.transpose());

    return {a, J};
}

// =============================================================================
// Combined Radiation Pressure Model
// =============================================================================

/// Sum of the three radiation-pressure parts (SRP, albedo, IR), each with its
/// own on/off switch (all on by default). Needs the satellite properties and
/// mass set first (the mass defaults to 1 kg).
class RadiationPressure {
public:
    /// Uses the process-wide default ephemeris for the Sun position.
    RadiationPressure()
        : ephemeris_(ephemeris::get_default_ephemeris_provider())
        , srp_enabled_(true)
        , albedo_enabled_(true)
        , ir_enabled_(true)
    {}

    /// Uses the given ephemeris for the Sun position.
    explicit RadiationPressure(std::shared_ptr<ephemeris::EphemerisProvider> eph)
        : ephemeris_(std::move(eph))
        , srp_enabled_(true)
        , albedo_enabled_(true)
        , ir_enabled_(true)
    {}

    /// Switch each part on or off.
    void enable_srp(bool enable) noexcept { srp_enabled_ = enable; }
    void enable_albedo(bool enable) noexcept { albedo_enabled_ = enable; }
    void enable_ir(bool enable) noexcept { ir_enabled_ = enable; }

    /// Sets area (m²), reflectivities and IR emissivity.
    void set_properties(const SatelliteRadiationProperties& props) noexcept {
        properties_ = props;
    }

    /// Sets the satellite mass (kg).
    void set_mass(double mass) noexcept { mass_ = mass; }

    [[nodiscard]] bool srp_enabled() const noexcept { return srp_enabled_; }
    [[nodiscard]] bool albedo_enabled() const noexcept { return albedo_enabled_; }
    [[nodiscard]] bool ir_enabled() const noexcept { return ir_enabled_; }

    /// Total radiation-pressure acceleration (m/s², GCRS) of the enabled parts
    /// at the GCRS position `sat_pos_eci` (m).
    [[nodiscard]] Vector3d acceleration(
        const Epoch& epoch,
        const Vector3d& sat_pos_eci) const
    {
        Vector3d acc = Vector3d::Zero();

        if (srp_enabled_) {
            acc += srp_acceleration(epoch, sat_pos_eci, mass_, properties_, *ephemeris_);
        }

        if (albedo_enabled_) {
            acc += albedo_acceleration(epoch, sat_pos_eci, mass_, properties_, *ephemeris_);
        }

        if (ir_enabled_) {
            acc += ir_acceleration(sat_pos_eci, mass_, properties_);
        }

        return acc;
    }

    /// Each part alone, ignoring its switch (for analysis).
    [[nodiscard]] Vector3d srp_only(const Epoch& epoch, const Vector3d& pos) const {
        return srp_acceleration(epoch, pos, mass_, properties_, *ephemeris_);
    }

    [[nodiscard]] Vector3d albedo_only(const Epoch& epoch, const Vector3d& pos) const {
        return albedo_acceleration(epoch, pos, mass_, properties_, *ephemeris_);
    }

    [[nodiscard]] Vector3d ir_only(const Vector3d& pos) const {
        return ir_acceleration(pos, mass_, properties_);
    }

    /// Total acceleration of the enabled parts AND its 3×3 Jacobian ∂a/∂r_sat,
    /// all analytic. Radiation pressure does not depend on velocity.
    [[nodiscard]] std::pair<Vector3d, Eigen::Matrix3d>
    acceleration_with_jacobian(
        const Epoch& epoch,
        const Vector3d& sat_pos_eci) const
    {
        Vector3d a_total = Vector3d::Zero();
        Eigen::Matrix3d J_total = Eigen::Matrix3d::Zero();

        if (srp_enabled_) {
            auto [a, J] = srp_acceleration_with_jacobian(
                epoch, sat_pos_eci, mass_, properties_, *ephemeris_);
            a_total += a;
            J_total += J;
        }

        if (albedo_enabled_) {
            auto [a, J] = albedo_acceleration_with_jacobian(
                epoch, sat_pos_eci, mass_, properties_, *ephemeris_);
            a_total += a;
            J_total += J;
        }

        if (ir_enabled_) {
            auto [a, J] = ir_acceleration_with_jacobian(
                sat_pos_eci, mass_, properties_);
            a_total += a;
            J_total += J;
        }

        return {a_total, J_total};
    }

private:
    std::shared_ptr<ephemeris::EphemerisProvider> ephemeris_;
    SatelliteRadiationProperties properties_;
    double mass_ = 1.0;  // Default 1 kg
    bool srp_enabled_;
    bool albedo_enabled_;
    bool ir_enabled_;
};

} // namespace forces
} // namespace ballistic
