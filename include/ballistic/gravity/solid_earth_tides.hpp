#pragma once

// =============================================================================
// Solid Earth Tide Corrections (IERS 2010 Conventions, Chapter 6.2)
// =============================================================================
//
// Computes the gravitational acceleration perturbation due to solid Earth
// tidal deformation caused by the Moon and Sun.
//
// The Moon and Sun deform the Earth, changing its gravity field. This is the
// INDIRECT effect (Earth deforms -> geopotential changes -> acceleration on
// satellite). The DIRECT gravitational pull is handled by ThirdBodyGravity.
//
// The deformation is described as changes dC_nm, dS_nm to the gravity-field
// coefficients ("Stokes coefficients" of degree n and order m; see
// egm_gravity.hpp). How much the Earth responds is set by its Love numbers
// k_nm (the ratio of the deformation's potential to the tide-raising potential).
//
// Implements IERS 2010 Step 1 for degrees 2 and 3 (IERS = International Earth
// Rotation and Reference Systems Service; one Love number per (n, m), anelastic,
// Table 6.3; the small degree-4 changes of eq. 6.7 are not modeled) and — when
// the caller passes the Doodson arguments — Step 2, the frequency-dependent
// corrections of Tables 6.5a-c (tide_arguments.hpp). Together with ocean
// tides they lowered the 1-hour prediction error over the 2025 Sentinel orbits.
//
// Algorithm (IERS 2010 Eq. 6.6):
//   For each body j (Moon, Sun):
//     1. Geocentric spherical coords of body
//     2. Normalized Legendre functions for degree 2,3
//     3. Ratio: (GM_j / GM_E) * (R_E / r_j)^(n+1)
//     4. Accumulate delta C_nm, delta S_nm via Love numbers
//   Then compute degree 2/3 gravitational acceleration from delta C/S
//   at satellite position via gradient of disturbing potential.
//
// Typical magnitude at LEO (~800 km): ~5e-8 m/s² (decimeter-level effect
// over 1 hour for LEO).
//
// References:
//   [1] IERS Conventions 2010, IERS Technical Note No. 36, Chapter 6.2
//   [2] Petit, G. & Luzum, B. (2010), "IERS Conventions (2010)"
// =============================================================================

#include "../core/types.hpp"
#include "../core/constants.hpp"
#include "tide_arguments.hpp"
#include <array>
#include <cmath>

namespace ballistic {
namespace gravity {

/// Acceleration on a satellite from the solid Earth tide (degrees 2 and 3).
/// Stateless; the methods are const.
class SolidEarthTides {
public:
    /// Tidal acceleration [m/s^2] at the satellite, in ECEF.
    ///
    /// All positions are ECEF (Earth-centered, Earth-fixed; the ITRF, the
    /// International Terrestrial Reference Frame) in meters, from the Earth's center.
    /// @param sat_pos_ecef   satellite position
    /// @param moon_pos_ecef  Moon position
    /// @param sun_pos_ecef   Sun position
    /// @param beta  doodson_arguments(epoch), which enables the Step-2
    ///              corrections; nullptr = Step 1 only
    /// Returns zero within 1 m of the Earth's center.
    [[nodiscard]] Vector3d acceleration(
        const Vector3d& sat_pos_ecef,
        const Vector3d& moon_pos_ecef,
        const Vector3d& sun_pos_ecef,
        const std::array<double, 6>* beta = nullptr) const
    {
        // Changes to the normalized coefficients, degrees 2-3, orders 0..n
        double dC[4][4] = {};  // dC[n][m]
        double dS[4][4] = {};  // dS[n][m]

        accumulate_body(moon_pos_ecef, constants::moon::GM, dC, dS);
        accumulate_body(sun_pos_ecef, constants::sun::GM, dC, dS);
        if (beta) add_step2_corrections(*beta, dC, dS);

        return acceleration_from_stokes(sat_pos_ecef, dC, dS);
    }

    /// Tidal acceleration [m/s^2] and its Hessian (the 3x3 matrix of second
    /// derivatives of the tidal potential, [1/s^2]) at the satellite, both in
    /// ECEF. Arguments as for acceleration().
    ///
    /// The tide only changes the coefficients, and those depend on where the
    /// Moon and Sun are, not on where the satellite is. So the Hessian is that
    /// of an ordinary gravity field holding just the coefficient changes, and
    /// the general gravity engine computes it (docs/gravity_hessian_derivation.md
    /// §10-11). The caller (the force model) rotates the Hessian into the
    /// inertial frame to build the satellite-state Jacobian.
    ///
    /// Implementation: a per-thread ClenshawTriangular engine of degree 5 (the
    /// smallest the engine supports: its final step reads sqrt(15) from a table
    /// sized 2*degree + 6) holding only degrees 2-3 = (dC, dS); C00, degree 1
    /// and degrees 4-5 are zero.
    [[nodiscard]] std::pair<Vector3d, Eigen::Matrix3d>
    acceleration_with_hessian(
        const Vector3d& sat_pos_ecef,
        const Vector3d& moon_pos_ecef,
        const Vector3d& sun_pos_ecef,
        const std::array<double, 6>* beta = nullptr) const
    {
        double dC[4][4] = {};
        double dS[4][4] = {};
        accumulate_body(moon_pos_ecef, constants::moon::GM, dC, dS);
        accumulate_body(sun_pos_ecef, constants::sun::GM, dC, dS);
        if (beta) add_step2_corrections(*beta, dC, dS);

        // One engine per thread (its coefficients are overwritten on every call).
        // Degree 5 because the engine's m = 0 step reads root_[15] = sqrt(15),
        // and the table holds 2*degree + 6 entries.
        thread_local ClenshawTriangular tides_engine;
        thread_local bool initialized = false;
        if (!initialized) {
            tides_engine.init(constants::earth::GM,
                              constants::earth::RADIUS_EQUATORIAL, 5);
            initialized = true;
        }

        // Coefficients = the tidal change only:
        //   - C00 = 0 (remove the default point-mass term)
        //   - degree 1 = 0 (the Earth's center of mass is the origin)
        //   - degrees 2 and 3: dC, dS
        //   - degrees 4 and 5: 0 (the degree-4 terms of IERS eq. 6.7 are not modeled)
        tides_engine.set_coefficient(0, 0, 0.0, 0.0);
        for (int m = 0; m <= 1; ++m) {
            tides_engine.set_coefficient(1, m, 0.0, 0.0);
        }
        for (int n = 2; n <= 3; ++n) {
            for (int m = 0; m <= n; ++m) {
                tides_engine.set_coefficient(n, m, dC[n][m], dS[n][m]);
            }
        }
        for (int n = 4; n <= 5; ++n) {
            for (int m = 0; m <= n; ++m) {
                tides_engine.set_coefficient(n, m, 0.0, 0.0);
            }
        }

        // Analytic Hessian in one pass
        const auto h_result = tides_engine.compute_with_hessian(sat_pos_ecef, 5);
        return {h_result.acceleration, h_result.hessian_ecef};
    }

    /// The tidal force derives from a potential (true).
    [[nodiscard]] static constexpr bool is_conservative() noexcept { return true; }

private:
    // Love numbers (IERS 2010 Table 6.3, anelastic, nominal): real and imaginary
    // parts (the imaginary part is a small phase lag of the Earth's response)
    static constexpr double k20_real = 0.30190;
    static constexpr double k21_real = 0.29830;
    static constexpr double k21_imag = -0.00144;
    static constexpr double k22_real = 0.30102;
    static constexpr double k22_imag = -0.00130;
    static constexpr double k30_real = 0.093;
    static constexpr double k31_real = 0.093;
    static constexpr double k32_real = 0.093;
    static constexpr double k33_real = 0.094;

    // Accumulate tidal coefficient changes from one body
    void accumulate_body(
        const Vector3d& body_ecef,
        double gm_body,
        double dC[4][4],
        double dS[4][4]) const
    {
        const double r = body_ecef.norm();
        if (r < 1.0) return;  // body at the center: no contribution

        const double GM_E = constants::earth::GM;
        const double R_E = constants::earth::RADIUS_EQUATORIAL;
        const double gm_ratio = gm_body / GM_E;

        // Spherical coordinates of the body
        const double sin_phi = body_ecef.z() / r;
        const double cos_phi = std::sqrt(1.0 - sin_phi * sin_phi);
        const double lambda = std::atan2(body_ecef.y(), body_ecef.x());

        // Powers of R_E/r
        const double re_r = R_E / r;
        const double re_r2 = re_r * re_r;
        const double re_r3 = re_r2 * re_r;   // (R_E/r)^3 for n=2
        const double re_r4 = re_r3 * re_r;   // (R_E/r)^4 for n=3

        // Normalized Legendre functions for the body position
        // Degree 2
        const double P20 = std::sqrt(5.0) / 2.0 * (3.0 * sin_phi * sin_phi - 1.0);
        const double P21 = std::sqrt(5.0 / 3.0) * 3.0 * sin_phi * cos_phi;
        const double P22 = std::sqrt(5.0 / 12.0) * 3.0 * cos_phi * cos_phi;

        // Degree 3
        const double P30 = std::sqrt(7.0) / 2.0 * (5.0 * sin_phi * sin_phi * sin_phi - 3.0 * sin_phi);
        const double P31 = std::sqrt(7.0 / 6.0) / 2.0 * (15.0 * sin_phi * sin_phi - 3.0) * cos_phi;
        const double P32 = std::sqrt(7.0 / 60.0) * 15.0 * sin_phi * cos_phi * cos_phi;
        const double P33 = std::sqrt(7.0 / 360.0) * 15.0 * cos_phi * cos_phi * cos_phi;

        // IERS 2010 Eq. 6.6: ΔC_nm - i·ΔS_nm = (k_nm / (2n+1)) * (GM_j/GM_E) * (R_E/r_j)^(n+1) * P̄_nm(sin φ_j) * exp(-i·m·λ_j)
        // Real part: ΔC_nm = (k_real/(2n+1)) * ratio * P_nm * cos(mλ) + (k_imag/(2n+1)) * ratio * P_nm * sin(mλ)
        // Imaginary: ΔS_nm = (k_real/(2n+1)) * ratio * P_nm * sin(mλ) - (k_imag/(2n+1)) * ratio * P_nm * cos(mλ)

        // Degree 2 contributions (2n+1 = 5)
        {
            const double ratio2 = gm_ratio * re_r3;  // (GM_j/GM_E) * (R_E/r_j)^3

            // (n,m) = (2,0) — k_imag = 0, sin(0) = 0, cos(0) = 1
            dC[2][0] += (k20_real / 5.0) * ratio2 * P20;
            // dS[2][0] is always 0 for m=0

            // (n,m) = (2,1)
            const double cos_l = std::cos(lambda);
            const double sin_l = std::sin(lambda);
            const double f21 = ratio2 * P21 / 5.0;
            dC[2][1] += k21_real * f21 * cos_l + k21_imag * f21 * sin_l;
            dS[2][1] += k21_real * f21 * sin_l - k21_imag * f21 * cos_l;

            // (n,m) = (2,2)
            const double cos_2l = std::cos(2.0 * lambda);
            const double sin_2l = std::sin(2.0 * lambda);
            const double f22 = ratio2 * P22 / 5.0;
            dC[2][2] += k22_real * f22 * cos_2l + k22_imag * f22 * sin_2l;
            dS[2][2] += k22_real * f22 * sin_2l - k22_imag * f22 * cos_2l;
        }

        // Degree 3 contributions (2n+1 = 7)
        // k_imag = 0 for all degree 3
        {
            const double ratio3 = gm_ratio * re_r4;  // (GM_j/GM_E) * (R_E/r_j)^4

            // (n,m) = (3,0)
            dC[3][0] += (k30_real / 7.0) * ratio3 * P30;

            // (n,m) = (3,1)
            const double cos_l = std::cos(lambda);
            const double sin_l = std::sin(lambda);
            dC[3][1] += (k31_real / 7.0) * ratio3 * P31 * cos_l;
            dS[3][1] += (k31_real / 7.0) * ratio3 * P31 * sin_l;

            // (n,m) = (3,2)
            const double cos_2l = std::cos(2.0 * lambda);
            const double sin_2l = std::sin(2.0 * lambda);
            dC[3][2] += (k32_real / 7.0) * ratio3 * P32 * cos_2l;
            dS[3][2] += (k32_real / 7.0) * ratio3 * P32 * sin_2l;

            // (n,m) = (3,3)
            const double cos_3l = std::cos(3.0 * lambda);
            const double sin_3l = std::sin(3.0 * lambda);
            dC[3][3] += (k33_real / 7.0) * ratio3 * P33 * cos_3l;
            dS[3][3] += (k33_real / 7.0) * ratio3 * P33 * sin_3l;
        }
    }

    // Compute gravitational acceleration from Stokes coefficient perturbations
    // Uses analytical gradient of the disturbing potential in spherical coordinates,
    // then converts to Cartesian ECEF.
    [[nodiscard]] Vector3d acceleration_from_stokes(
        const Vector3d& pos_ecef,
        const double dC[4][4],
        const double dS[4][4]) const
    {
        const double GM = constants::earth::GM;
        const double R_E = constants::earth::RADIUS_EQUATORIAL;

        const double r = pos_ecef.norm();
        if (r < 1.0) return Vector3d::Zero();

        const double sin_phi = pos_ecef.z() / r;
        const double cos_phi = std::sqrt(1.0 - sin_phi * sin_phi);
        const double lambda = std::atan2(pos_ecef.y(), pos_ecef.x());

        // Avoid dividing by zero on the polar axis
        const double cos_phi_safe = std::max(cos_phi, 1e-14);

        const double re_r = R_E / r;

        // Potential gradient in spherical coordinates:
        //   V = (GM/r) * sum_{n,m} (R_E/r)^n * P_nm(sin_phi) * [dC_nm*cos(m*lam) + dS_nm*sin(m*lam)]
        //
        // dV/dr   = -(GM/r^2) * sum (n+1) * (R_E/r)^n * P_nm * [dC*cos + dS*sin]
        // dV/dphi = (GM/r) * sum (R_E/r)^n * dP_nm/dphi * [dC*cos + dS*sin]
        // dV/dlam = (GM/r) * sum (R_E/r)^n * P_nm * m * [-dC*sin + dS*cos]

        double dV_dr = 0.0;
        double dV_dphi = 0.0;
        double dV_dlam = 0.0;

        // Process degrees 2 and 3
        for (int n = 2; n <= 3; ++n) {
            // (R_E/r)^n
            double re_r_n = 1.0;
            for (int i = 0; i < n; ++i) re_r_n *= re_r;

            for (int m = 0; m <= n; ++m) {
                const double cos_ml = std::cos(m * lambda);
                const double sin_ml = std::sin(m * lambda);

                const double Pnm = normalized_legendre(n, m, sin_phi, cos_phi);
                const double dPnm = normalized_legendre_dphi(n, m, sin_phi, cos_phi);

                const double cs = dC[n][m] * cos_ml + dS[n][m] * sin_ml;
                const double sc = -dC[n][m] * sin_ml + dS[n][m] * cos_ml;

                dV_dr   += -(static_cast<double>(n) + 1.0) * re_r_n * Pnm * cs;
                dV_dphi += re_r_n * dPnm * cs;
                dV_dlam += re_r_n * Pnm * static_cast<double>(m) * sc;
            }
        }

        dV_dr   *= GM / (r * r);
        dV_dphi *= GM / r;
        dV_dlam *= GM / r;

        // Convert spherical gradient to Cartesian ECEF
        // a_r     = dV/dr           (radial outward)
        // a_phi   = (1/r) * dV/dphi (north)
        // a_lam   = (1/(r*cos_phi)) * dV/dlam (east)
        const double a_r   = dV_dr;
        const double a_phi = dV_dphi / r;
        const double a_lam = dV_dlam / (r * cos_phi_safe);

        // Spherical to Cartesian:
        // x = r*cos_phi*cos_lam, y = r*cos_phi*sin_lam, z = r*sin_phi
        const double cos_l = std::cos(lambda);
        const double sin_l = std::sin(lambda);

        // Unit vectors in spherical:
        // e_r     = (cos_phi*cos_l, cos_phi*sin_l, sin_phi)
        // e_phi   = (-sin_phi*cos_l, -sin_phi*sin_l, cos_phi)  [north]
        // e_lam   = (-sin_l, cos_l, 0)                         [east]
        const double ax = a_r * cos_phi * cos_l + a_phi * (-sin_phi * cos_l) + a_lam * (-sin_l);
        const double ay = a_r * cos_phi * sin_l + a_phi * (-sin_phi * sin_l) + a_lam * cos_l;
        const double az = a_r * sin_phi          + a_phi * cos_phi;

        return Vector3d(ax, ay, az);
    }

    // Normalized associated Legendre functions for degrees 2 and 3
    [[nodiscard]] static double normalized_legendre(int n, int m, double sin_phi, double cos_phi) noexcept {
        if (n == 2) {
            switch (m) {
                case 0: return std::sqrt(5.0) / 2.0 * (3.0 * sin_phi * sin_phi - 1.0);
                case 1: return std::sqrt(5.0 / 3.0) * 3.0 * sin_phi * cos_phi;
                case 2: return std::sqrt(5.0 / 12.0) * 3.0 * cos_phi * cos_phi;
            }
        } else if (n == 3) {
            const double s2 = sin_phi * sin_phi;
            const double c2 = cos_phi * cos_phi;
            switch (m) {
                case 0: return std::sqrt(7.0) / 2.0 * (5.0 * s2 * sin_phi - 3.0 * sin_phi);
                case 1: return std::sqrt(7.0 / 6.0) / 2.0 * (15.0 * s2 - 3.0) * cos_phi;
                case 2: return std::sqrt(7.0 / 60.0) * 15.0 * sin_phi * c2;
                case 3: return std::sqrt(7.0 / 360.0) * 15.0 * c2 * cos_phi;
            }
        }
        return 0.0;
    }

    // Derivative of normalized Legendre function w.r.t. geocentric latitude (phi)
    // d/dphi P_nm(sin_phi) where phi is latitude
    // Using: d(sin_phi)/d(phi) = cos_phi, d(cos_phi)/d(phi) = -sin_phi
    [[nodiscard]] static double normalized_legendre_dphi(int n, int m, double sin_phi, double cos_phi) noexcept {
        if (n == 2) {
            switch (m) {
                case 0: // d/dphi [sqrt(5)/2 * (3s^2 - 1)] = sqrt(5)/2 * 6*s*c
                    return std::sqrt(5.0) / 2.0 * 6.0 * sin_phi * cos_phi;
                case 1: // d/dphi [sqrt(5/3) * 3*s*c] = sqrt(5/3) * 3*(c^2 - s^2)
                    return std::sqrt(5.0 / 3.0) * 3.0 * (cos_phi * cos_phi - sin_phi * sin_phi);
                case 2: // d/dphi [sqrt(5/12) * 3*c^2] = sqrt(5/12) * 3*(-2*s*c) = -sqrt(5/12)*6*s*c
                    return -std::sqrt(5.0 / 12.0) * 6.0 * sin_phi * cos_phi;
            }
        } else if (n == 3) {
            const double s2 = sin_phi * sin_phi;
            const double c2 = cos_phi * cos_phi;
            switch (m) {
                case 0: // d/dphi [sqrt(7)/2 * (5s^3 - 3s)] = sqrt(7)/2 * (15s^2*c - 3c) = sqrt(7)/2 * c*(15s^2 - 3)
                    return std::sqrt(7.0) / 2.0 * cos_phi * (15.0 * s2 - 3.0);
                case 1: // d/dphi [sqrt(7/6)/2 * (15s^2 - 3)*c]
                    // = sqrt(7/6)/2 * [30*s*c*c + (15s^2-3)*(-s)]
                    // = sqrt(7/6)/2 * [30*s*c^2 - 15*s^3 + 3*s]
                    // = sqrt(7/6)/2 * s*(30*c^2 - 15*s^2 + 3)
                    return std::sqrt(7.0 / 6.0) / 2.0 * sin_phi * (30.0 * c2 - 15.0 * s2 + 3.0);
                case 2: // d/dphi [sqrt(7/60) * 15*s*c^2]
                    // = sqrt(7/60)*15*(c^2*c + s*(-2*s*c)) = sqrt(7/60)*15*(c^3 - 2*s^2*c)
                    // = sqrt(7/60)*15*c*(c^2 - 2*s^2)
                    return std::sqrt(7.0 / 60.0) * 15.0 * cos_phi * (c2 - 2.0 * s2);
                case 3: // d/dphi [sqrt(7/360) * 15*c^3] = sqrt(7/360)*15*(-3*c^2*s)
                    return -std::sqrt(7.0 / 360.0) * 45.0 * c2 * sin_phi;
            }
        }
        return 0.0;
    }
};

} // namespace gravity
} // namespace ballistic
