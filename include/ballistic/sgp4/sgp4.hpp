#pragma once

// =============================================================================
// SGP4 / SDP4: the analytic propagator that defines TLE mean elements
// =============================================================================
//
// A two-line element set (TLE, see tle.hpp) describes an orbit by MEAN
// orbital elements: averaged values that only make sense inside one specific
// analytic model, SGP4 ("Simplified General Perturbations 4"), and its
// deep-space extension SDP4 for orbits with periods of 225 minutes or more.
// Given a TLE and a time, the model returns a position and velocity in
// closed form (no step-by-step integration).
//
// Clean-room implementation from the published mathematics:
//   * F.R. Hoots & R.L. Roehrich, "Models for Propagation of NORAD Element
//     Sets", Spacetrack Report No. 3 (1980; US Government work), SGP4 and
//     SDP4 (deep space: lunar-solar secular and long-period terms, 12-hour
//     and 24-hour geopotential resonances);
//   * D.A. Vallado, P. Crawford, R. Hujsak & T.S. Kelso, "Revisiting
//     Spacetrack Report #3", AIAA 2006-6753, whose documented corrections
//     are applied: the deep-space resonance integrator restarts at epoch
//     (here it always starts at epoch, which gives the same steps), the
//     Lyddane form of the long-period lunar-solar terms at inclinations below
//     0.2 rad (it avoids a division by sin(inclination)), negative
//     inclinations allowed after the periodic terms, the 1e-6 lower limit on
//     eccentricity, the singly averaged mean elements, and the "AFSPC" versus
//     "improved" operation modes (how the Earth's rotation angle at epoch is
//     computed).
// No third-party source was used. The MIT-licensed `sgp4` Python package
// is a TEST ORACLE only (tests/python/test_sgp4.py, downloaded at test time,
// together with the SGP4-VER verification vectors it carries).
//
// Output is in the TEME frame (True Equator, Mean Equinox of date: an
// inertial-like frame specific to SGP4; teme.hpp converts it to ITRF, the
// Earth-fixed frame), in km and km/s. The model is stateless after
// construction: propagate_minutes() and teme() are const and thread-safe.
//
// What SGP4 is good for: a TLE's error is ~1 km at its epoch in low Earth
// orbit (LEO) and grows by a few km per day (Flohrer et al. 2008). SGP4 is
// the ONLY propagator that should read TLE elements: they are means in its
// own theory, and starting a numerical propagator from them directly is
// wrong by more than 10 km (tests/python/test_sgp4.py). State.from_tle
// evaluates SGP4 at the TLE epoch and hands the numerical propagator the
// resulting osculating state (the instantaneous position and velocity).
// =============================================================================

#include "../core/errors.hpp"
#include "../core/types.hpp"
#include "tle.hpp"

#include <cmath>
#include <numbers>
#include <string>

namespace ballistic::sgp4 {

/// The Earth constants (gravitational parameter, radius, zonal harmonics J2-J4) SGP4 is
/// evaluated with. Element sets are produced with WGS-72 (the 1972 World Geodetic System);
/// using another set changes the result by meters to kilometers. WGS72Old is the older
/// WGS-72 variant with its xke constant given directly; WGS84 is SGP4's own WGS-84 set.
enum class GravityConstants { WGS72, WGS72Old, WGS84 };

/// How the Earth's rotation angle at the TLE epoch (Greenwich sidereal angle) is computed.
/// Afspc: the original 1970-based expression (what the operational US Space Force code
/// does); Improved: the IAU 1982 Greenwich mean sidereal time (GMST) of the epoch. Near-Earth
/// orbits give identical results; the angle (and the mode) enters only the deep-space terms.
enum class OpsMode { Improved, Afspc };

/// Model options for Sgp4; the defaults match the published verification vectors.
struct Sgp4Options {
    GravityConstants gravity = GravityConstants::WGS72;
    OpsMode mode = OpsMode::Improved;
};

/// The published SGP4 error codes (Sgp4Output::status).
enum class Sgp4Status {
    Ok = 0,
    MeanEccentricity = 1,      // mean eccentricity outside [-0.001, 1)
    MeanMotion = 2,            // mean motion <= 0
    PerturbedEccentricity = 3, // eccentricity outside [0, 1] after the lunar-solar periodics
    SemiLatusRectum = 4,       // semi-latus rectum < 0
    Decayed = 6,               // radius below one Earth radius: the object has reentered
};

/// A short English description of a status code.
[[nodiscard]] inline const char* sgp4_status_text(Sgp4Status s) noexcept {
    switch (s) {
        case Sgp4Status::Ok: return "ok";
        case Sgp4Status::MeanEccentricity: return "mean eccentricity out of range";
        case Sgp4Status::MeanMotion: return "mean motion is not positive";
        case Sgp4Status::PerturbedEccentricity: return "perturbed eccentricity out of range";
        case Sgp4Status::SemiLatusRectum: return "semi-latus rectum is negative";
        case Sgp4Status::Decayed: return "the orbit is below the Earth's surface (decayed)";
    }
    return "unknown";
}

/// One evaluation of the model: TEME position [km] and velocity [km/s], and the status.
struct Sgp4Output {
    Vector3d r_km = Vector3d::Constant(std::nan(""));    // TEME
    Vector3d v_km_s = Vector3d::Constant(std::nan(""));  // TEME
    Sgp4Status status = Sgp4Status::Ok;                  // r, v are NaN unless Ok or Decayed
};

namespace detail_sgp4 {

inline constexpr double kTwoPi = 2.0 * std::numbers::pi;
inline constexpr double kPi = std::numbers::pi;
inline constexpr double kX2o3 = 2.0 / 3.0;

struct Constants {
    double mu, re_km, xke, tumin, j2, j3, j4, j3oj2;
};

[[nodiscard]] inline Constants constants(GravityConstants g) noexcept {
    Constants c{};
    switch (g) {
        case GravityConstants::WGS72Old:
            c.mu = 398600.79964; c.re_km = 6378.135; c.xke = 0.0743669161;
            c.j2 = 0.001082616; c.j3 = -0.00000253881; c.j4 = -0.00000165597;
            break;
        case GravityConstants::WGS72:
            c.mu = 398600.8; c.re_km = 6378.135;
            c.xke = 60.0 / std::sqrt(c.re_km * c.re_km * c.re_km / c.mu);
            c.j2 = 0.001082616; c.j3 = -0.00000253881; c.j4 = -0.00000165597;
            break;
        case GravityConstants::WGS84:
            c.mu = 398600.5; c.re_km = 6378.137;  // guard-exempt(CONST-LITERAL): SGP4's own WGS-84 set, part of the model definition
            c.xke = 60.0 / std::sqrt(c.re_km * c.re_km * c.re_km / c.mu);
            c.j2 = 0.00108262998905; c.j3 = -0.00000253215306; c.j4 = -0.00000161098761;
            break;
    }
    c.tumin = 1.0 / c.xke;
    c.j3oj2 = c.j3 / c.j2;
    return c;
}

// IAU 1982 Greenwich mean sidereal time (GMST, the Earth's mean rotation
// angle) [rad] of a UT1 Julian date.
[[nodiscard]] inline double gmst1982(double jd_ut1) noexcept {
    const double t = (jd_ut1 - 2451545.0) / 36525.0;
    double s = -6.2e-6 * t * t * t + 0.093104 * t * t +
               (876600.0 * 3600.0 + 8640184.812866) * t + 67310.54841;  // seconds of time
    s = std::fmod(s * (kPi / 180.0) / 240.0, kTwoPi);
    return s < 0.0 ? s + kTwoPi : s;
}

// Lunar-solar terms shared by the deep-space initialization and periodics.
struct LunarSolar {
    // long-period periodic coefficients (solar s*, lunar x*/e*)
    double se2, se3, si2, si3, sl2, sl3, sl4, sgh2, sgh3, sgh4, sh2, sh3;
    double ee2, e3, xi2, xi3, xl2, xl3, xl4, xgh2, xgh3, xgh4, xh2, xh3;
    double zmol, zmos;
    // inputs to the secular rates (deep_space_init)
    double s1, s2, s3, s4, s5, ss1, ss2, ss3, ss4, ss5;
    double z1, z3, z11, z13, z21, z23, z31, z33;
    double sz1, sz3, sz11, sz13, sz21, sz23, sz31, sz33;
    double sinim, cosim, emsq;
};

// Spacetrack Report #3, DEEP: the lunar and solar perturbation coefficients
// at epoch (Vallado et al. 2006 "dscom"). `epoch` is days since 1949-12-31
// 00:00 UTC (JD 2433281.5); angles in rad, `np` the mean motion in rad/min.
[[nodiscard]] inline LunarSolar lunar_solar(double epoch, double ep, double argpp, double inclp,
                                            double nodep, double np) noexcept {
    constexpr double zes = 0.01675, zel = 0.05490;
    constexpr double c1ss = 2.9864797e-6, c1l = 4.7968065e-7;
    constexpr double zsinis = 0.39785416, zcosis = 0.91744867;
    constexpr double zcosgs = 0.1945905, zsings = -0.98088458;

    LunarSolar o{};
    const double nm = np, em = ep;
    const double snodm = std::sin(nodep), cnodm = std::cos(nodep);
    const double sinomm = std::sin(argpp), cosomm = std::cos(argpp);
    o.sinim = std::sin(inclp);
    o.cosim = std::cos(inclp);
    o.emsq = em * em;
    const double betasq = 1.0 - o.emsq;
    const double rtemsq = std::sqrt(betasq);

    const double day = epoch + 18261.5;  // days since 1900 Jan 0.5
    const double xnodce = std::fmod(4.5236020 - 9.2422029e-4 * day, kTwoPi);
    const double stem = std::sin(xnodce), ctem = std::cos(xnodce);
    const double zcosil = 0.91375164 - 0.03568096 * ctem;
    const double zsinil = std::sqrt(1.0 - zcosil * zcosil);
    const double zsinhl = 0.089683511 * stem / zsinil;
    const double zcoshl = std::sqrt(1.0 - zsinhl * zsinhl);
    const double gam = 5.8351514 + 0.0019443680 * day;
    double zx = 0.39785416 * stem / zsinil;
    const double zy = zcoshl * ctem + 0.91744867 * zsinhl * stem;
    zx = std::atan2(zx, zy);
    zx = gam + zx - xnodce;
    const double zcosgl = std::cos(zx), zsingl = std::sin(zx);

    // Pass 1 the Sun, pass 2 the Moon.
    double zcosg = zcosgs, zsing = zsings, zcosi = zcosis, zsini = zsinis;
    double zcosh = cnodm, zsinh = snodm, cc = c1ss;
    const double xnoi = 1.0 / nm;
    double s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    double z1 = 0, z2 = 0, z3 = 0, z11 = 0, z12 = 0, z13 = 0, z21 = 0, z22 = 0, z23 = 0;
    double z31 = 0, z32 = 0, z33 = 0;
    double ss6 = 0, ss7 = 0, sz2 = 0, sz12 = 0, sz22 = 0, sz32 = 0;
    for (int body = 1; body <= 2; ++body) {
        const double a1 = zcosg * zcosh + zsing * zcosi * zsinh;
        const double a3 = -zsing * zcosh + zcosg * zcosi * zsinh;
        const double a7 = -zcosg * zsinh + zsing * zcosi * zcosh;
        const double a8 = zsing * zsini;
        const double a9 = zsing * zsinh + zcosg * zcosi * zcosh;
        const double a10 = zcosg * zsini;
        const double a2 = o.cosim * a7 + o.sinim * a8;
        const double a4 = o.cosim * a9 + o.sinim * a10;
        const double a5 = -o.sinim * a7 + o.cosim * a8;
        const double a6 = -o.sinim * a9 + o.cosim * a10;

        const double x1 = a1 * cosomm + a2 * sinomm;
        const double x2 = a3 * cosomm + a4 * sinomm;
        const double x3 = -a1 * sinomm + a2 * cosomm;
        const double x4 = -a3 * sinomm + a4 * cosomm;
        const double x5 = a5 * sinomm;
        const double x6 = a6 * sinomm;
        const double x7 = a5 * cosomm;
        const double x8 = a6 * cosomm;

        z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3;
        z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4;
        z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4;
        z1 = 3.0 * (a1 * a1 + a2 * a2) + z31 * o.emsq;
        z2 = 6.0 * (a1 * a3 + a2 * a4) + z32 * o.emsq;
        z3 = 3.0 * (a3 * a3 + a4 * a4) + z33 * o.emsq;
        z11 = -6.0 * a1 * a5 + o.emsq * (-24.0 * x1 * x7 - 6.0 * x3 * x5);
        z12 = -6.0 * (a1 * a6 + a3 * a5) +
              o.emsq * (-24.0 * (x2 * x7 + x1 * x8) - 6.0 * (x3 * x6 + x4 * x5));
        z13 = -6.0 * a3 * a6 + o.emsq * (-24.0 * x2 * x8 - 6.0 * x4 * x6);
        z21 = 6.0 * a2 * a5 + o.emsq * (24.0 * x1 * x5 - 6.0 * x3 * x7);
        z22 = 6.0 * (a4 * a5 + a2 * a6) +
              o.emsq * (24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8));
        z23 = 6.0 * a4 * a6 + o.emsq * (24.0 * x2 * x6 - 6.0 * x4 * x8);
        z1 = z1 + z1 + betasq * z31;
        z2 = z2 + z2 + betasq * z32;
        z3 = z3 + z3 + betasq * z33;
        s3 = cc * xnoi;
        s2 = -0.5 * s3 / rtemsq;
        s4 = s3 * rtemsq;
        s1 = -15.0 * em * s4;
        s5 = x1 * x3 + x2 * x4;
        s6 = x2 * x3 + x1 * x4;
        s7 = x2 * x4 - x1 * x3;

        if (body == 1) {  // keep the solar values, set up the lunar pass
            o.ss1 = s1; o.ss2 = s2; o.ss3 = s3; o.ss4 = s4; o.ss5 = s5; ss6 = s6; ss7 = s7;
            o.sz1 = z1; sz2 = z2; o.sz3 = z3;
            o.sz11 = z11; sz12 = z12; o.sz13 = z13;
            o.sz21 = z21; sz22 = z22; o.sz23 = z23;
            o.sz31 = z31; sz32 = z32; o.sz33 = z33;
            zcosg = zcosgl; zsing = zsingl; zcosi = zcosil; zsini = zsinil;
            zcosh = zcoshl * cnodm + zsinhl * snodm;
            zsinh = snodm * zcoshl - cnodm * zsinhl;
            cc = c1l;
        }
    }
    o.s1 = s1; o.s2 = s2; o.s3 = s3; o.s4 = s4; o.s5 = s5;
    o.z1 = z1; o.z3 = z3; o.z11 = z11; o.z13 = z13; o.z21 = z21; o.z23 = z23; o.z31 = z31; o.z33 = z33;

    o.zmol = std::fmod(4.7199672 + 0.22997150 * day - gam, kTwoPi);
    o.zmos = std::fmod(6.2565837 + 0.017201977 * day, kTwoPi);

    o.se2 = 2.0 * o.ss1 * ss6;
    o.se3 = 2.0 * o.ss1 * ss7;
    o.si2 = 2.0 * o.ss2 * sz12;
    o.si3 = 2.0 * o.ss2 * (o.sz13 - o.sz11);
    o.sl2 = -2.0 * o.ss3 * sz2;
    o.sl3 = -2.0 * o.ss3 * (o.sz3 - o.sz1);
    o.sl4 = -2.0 * o.ss3 * (-21.0 - 9.0 * o.emsq) * zes;
    o.sgh2 = 2.0 * o.ss4 * sz32;
    o.sgh3 = 2.0 * o.ss4 * (o.sz33 - o.sz31);
    o.sgh4 = -18.0 * o.ss4 * zes;
    o.sh2 = -2.0 * o.ss2 * sz22;
    o.sh3 = -2.0 * o.ss2 * (o.sz23 - o.sz21);

    o.ee2 = 2.0 * s1 * s6;
    o.e3 = 2.0 * s1 * s7;
    o.xi2 = 2.0 * s2 * z12;
    o.xi3 = 2.0 * s2 * (z13 - z11);
    o.xl2 = -2.0 * s3 * z2;
    o.xl3 = -2.0 * s3 * (z3 - z1);
    o.xl4 = -2.0 * s3 * (-21.0 - 9.0 * o.emsq) * zel;
    o.xgh2 = 2.0 * s4 * z32;
    o.xgh3 = 2.0 * s4 * (z33 - z31);
    o.xgh4 = -18.0 * s4 * zel;
    o.xh2 = -2.0 * s2 * z22;
    o.xh3 = -2.0 * s2 * (z23 - z21);
    return o;
}

// Deep-space resonance and secular-rate state fixed at epoch.
struct DeepSpace {
    LunarSolar ls{};
    int irez = 0;  // 0 none, 1 one-day (synchronous), 2 half-day resonance
    double d2201 = 0, d2211 = 0, d3210 = 0, d3222 = 0, d4410 = 0, d4422 = 0;
    double d5220 = 0, d5232 = 0, d5421 = 0, d5433 = 0;
    double dedt = 0, didt = 0, dmdt = 0, dnodt = 0, domdt = 0;
    double del1 = 0, del2 = 0, del3 = 0;
    double xfact = 0, xlamo = 0;
};

}  // namespace detail_sgp4

/// The SGP4/SDP4 model for one TLE. Construction does the epoch-only work (recovering the
/// model's internal mean motion, the drag and deep-space coefficients); evaluation is then
/// cheap and thread-safe.
class Sgp4 {
public:
    /// Initialize the model for `tle` with options `opt`. Does not throw; an element set the
    /// model cannot handle shows up as a non-Ok status when evaluated.
    explicit Sgp4(const Tle& tle, Sgp4Options opt = {}) : tle_(tle), opt_(opt) {
        constexpr double deg = std::numbers::pi / 180.0;
        constexpr double xpdotp = 1440.0 / (2.0 * std::numbers::pi);  // rev/day -> rad/min
        // The reference implementation forms the epoch as a Julian date
        // (day at 0h + fraction, a double near 2.4e6) before subtracting
        // 2433281.5, which rounds it by up to ~2e-5 s. The deep-space terms
        // see that (4 mm on SGP4-VER object 23333, e = 0.97), so the same
        // arithmetic is used here to reproduce the defining model.
        const double doy = std::floor(tle.epoch_day);
        const double jd0 = static_cast<double>(time::civil_to_jdn(tle.epoch_year, 1, 1)) - 0.5 + (doy - 1.0);
        const double jd = jd0 + (tle.epoch_day - doy);
        init(jd - 2433281.5, tle.bstar_per_er, tle.eccentricity,
             tle.argp_deg * deg, tle.inclination_deg * deg, tle.mean_anomaly_deg * deg,
             tle.mean_motion_rev_day / xpdotp, tle.raan_deg * deg);
    }

    /// The element set and options this model was built from.
    [[nodiscard]] const Tle& tle() const noexcept { return tle_; }
    [[nodiscard]] const Sgp4Options& options() const noexcept { return opt_; }
    /// True if the deep-space branch (SDP4, period >= 225 minutes) is used.
    [[nodiscard]] bool deep_space() const noexcept { return deep_; }
    /// The Earth radius of the chosen GravityConstants [km].
    [[nodiscard]] double radius_km() const noexcept { return k_.re_km; }

    /// The model at `tsince_min` minutes from the TLE epoch (negative = before it): TEME
    /// position [km] and velocity [km/s]. Never throws; check Sgp4Output::status.
    [[nodiscard]] Sgp4Output propagate_minutes(double tsince_min) const noexcept;

    /// TEME position [m] and velocity [m/s] at epoch `t`. Throws InputPlausibilityError
    /// when the model fails there (any status other than Ok, e.g. the orbit has decayed).
    [[nodiscard]] StateVector teme(const Epoch& t) const {
        const double dt_min = (t - tle_.epoch()) / 60.0;
        const Sgp4Output o = propagate_minutes(dt_min);
        if (o.status != Sgp4Status::Ok)
            throw InputPlausibilityError(
                "SGP4 for catalog object " + std::to_string(tle_.catalog_number) + " at " +
                std::to_string(dt_min / 1440.0) + " days from its TLE epoch: " +
                sgp4_status_text(o.status) + "; this element set does not describe the orbit there");
        return StateVector(o.r_km * 1e3, o.v_km_s * 1e3);
    }

private:
    void init(double epoch, double bstar, double ecco, double argpo, double inclo, double mo,
              double no_kozai, double nodeo);

    Tle tle_;
    Sgp4Options opt_;
    detail_sgp4::Constants k_{};
    bool deep_ = false, isimp_ = false;
    double bstar_ = 0, ecco_ = 0, argpo_ = 0, inclo_ = 0, mo_ = 0, nodeo_ = 0, no_ = 0;
    double epoch_ = 0, gsto_ = 0;
    double aycof_ = 0, con41_ = 0, cc1_ = 0, cc4_ = 0, cc5_ = 0, d2_ = 0, d3_ = 0, d4_ = 0;
    double delmo_ = 0, eta_ = 0, argpdot_ = 0, omgcof_ = 0, sinmao_ = 0, t2cof_ = 0, t3cof_ = 0;
    double t4cof_ = 0, t5cof_ = 0, x1mth2_ = 0, x7thm1_ = 0, mdot_ = 0, nodedot_ = 0, xlcof_ = 0;
    double xmcof_ = 0, nodecf_ = 0;
    detail_sgp4::DeepSpace ds_{};
};

// -----------------------------------------------------------------------------

namespace detail_sgp4 {

// Lunar-solar long-period periodic terms applied to the mean elements at time
// t (minutes since epoch). Spacetrack Report #3 DPPER with the Lyddane form
// below 0.2 rad inclination (Vallado et al. 2006 "dpper").
inline void lunar_solar_periodics(const LunarSolar& s, double t, OpsMode mode, double& ep,
                                  double& inclp, double& nodep, double& argpp, double& mp) noexcept {
    constexpr double zns = 1.19459e-5, zes = 0.01675, znl = 1.5835218e-4, zel = 0.05490;
    auto terms = [](double zm, double ze, double& f2, double& f3, double& sinzf) {
        const double zf = zm + 2.0 * ze * std::sin(zm);
        sinzf = std::sin(zf);
        f2 = 0.5 * sinzf * sinzf - 0.25;
        f3 = -0.5 * sinzf * std::cos(zf);
    };
    double f2, f3, sinzf;
    terms(s.zmos + zns * t, zes, f2, f3, sinzf);
    const double ses = s.se2 * f2 + s.se3 * f3;
    const double sis = s.si2 * f2 + s.si3 * f3;
    const double sls = s.sl2 * f2 + s.sl3 * f3 + s.sl4 * sinzf;
    const double sghs = s.sgh2 * f2 + s.sgh3 * f3 + s.sgh4 * sinzf;
    const double shs = s.sh2 * f2 + s.sh3 * f3;
    terms(s.zmol + znl * t, zel, f2, f3, sinzf);
    const double sel = s.ee2 * f2 + s.e3 * f3;
    const double sil = s.xi2 * f2 + s.xi3 * f3;
    const double sll = s.xl2 * f2 + s.xl3 * f3 + s.xl4 * sinzf;
    const double sghl = s.xgh2 * f2 + s.xgh3 * f3 + s.xgh4 * sinzf;
    const double shll = s.xh2 * f2 + s.xh3 * f3;

    // The values at epoch are not subtracted (they are zero in the 2006
    // formulation), as in the verification vectors.
    const double pe = ses + sel;
    const double pinc = sis + sil;
    const double pl = sls + sll;
    double pgh = sghs + sghl;
    double ph = shs + shll;

    inclp += pinc;
    ep += pe;
    const double sinip = std::sin(inclp), cosip = std::cos(inclp);
    if (inclp >= 0.2) {
        ph /= sinip;
        pgh -= cosip * ph;
        argpp += pgh;
        nodep += ph;
        mp += pl;
    } else {
        // Lyddane: perturb the node through its direction cosines.
        const double sinop = std::sin(nodep), cosop = std::cos(nodep);
        double alfdp = sinip * sinop;
        double betdp = sinip * cosop;
        alfdp += ph * cosop + pinc * cosip * sinop;
        betdp += -ph * sinop + pinc * cosip * cosop;
        nodep = std::fmod(nodep, kTwoPi);
        if (nodep < 0.0 && mode == OpsMode::Afspc) nodep += kTwoPi;
        double xls = mp + argpp + cosip * nodep;
        xls += pl + pgh - pinc * nodep * sinip;
        const double xnoh = nodep;
        nodep = std::atan2(alfdp, betdp);
        if (nodep < 0.0 && mode == OpsMode::Afspc) nodep += kTwoPi;
        if (std::fabs(xnoh - nodep) > kPi) nodep += (nodep < xnoh) ? kTwoPi : -kTwoPi;
        mp += pl;
        argpp = xls - mp - cosip * nodep;
    }
}

}  // namespace detail_sgp4

inline void Sgp4::init(double epoch, double bstar, double ecco, double argpo, double inclo,
                       double mo, double no_kozai, double nodeo) {
    using namespace detail_sgp4;
    k_ = constants(opt_.gravity);
    bstar_ = bstar; ecco_ = ecco; argpo_ = argpo; inclo_ = inclo; mo_ = mo; nodeo_ = nodeo;
    epoch_ = epoch;

    const double ss = 78.0 / k_.re_km + 1.0;
    const double qzms2t = std::pow((120.0 - 78.0) / k_.re_km, 4);
    constexpr double temp4 = 1.5e-12;

    // --- Recover the Brouwer mean motion from the Kozai one ------------------
    // (TLEs carry the mean motion in Kozai's averaging; SGP4 works with
    // Brouwer's. The two differ by a J2 correction.)
    const double eccsq = ecco * ecco;
    const double omeosq = 1.0 - eccsq;
    const double rteosq = std::sqrt(omeosq);
    const double cosio = std::cos(inclo);
    const double cosio2 = cosio * cosio;
    const double ak = std::pow(k_.xke / no_kozai, kX2o3);
    const double d1 = 0.75 * k_.j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq);
    double del = d1 / (ak * ak);
    const double adel = ak * (1.0 - del * del - del * (1.0 / 3.0 + 134.0 * del * del / 81.0));
    del = d1 / (adel * adel);
    no_ = no_kozai / (1.0 + del);

    const double ao = std::pow(k_.xke / no_, kX2o3);
    const double sinio = std::sin(inclo);
    const double po = ao * omeosq;
    const double con42 = 1.0 - 5.0 * cosio2;
    con41_ = -con42 - cosio2 - cosio2;
    const double posq = po * po;
    const double rp = ao * (1.0 - ecco);

    // Greenwich sidereal angle at epoch.
    if (opt_.mode == OpsMode::Afspc) {
        const double ts70 = epoch - 7305.0;
        const double ds70 = std::floor(ts70 + 1.0e-8);
        const double tfrac = ts70 - ds70;
        constexpr double c1 = 1.72027916940703639e-2, thgr70 = 1.7321343856509374;
        constexpr double fk5r = 5.07551419432269442e-15;
        const double c1p2p = c1 + kTwoPi;
        gsto_ = std::fmod(thgr70 + c1 * ds70 + c1p2p * tfrac + ts70 * ts70 * fk5r, kTwoPi);
        if (gsto_ < 0.0) gsto_ += kTwoPi;
    } else {
        gsto_ = gmst1982(epoch + 2433281.5);
    }

    // --- Secular drag and gravity coefficients -------------------------------
    isimp_ = rp < (220.0 / k_.re_km + 1.0);  // perigee below 220 km: truncated drag terms
    double sfour = ss, qzms24 = qzms2t;
    const double perige = (rp - 1.0) * k_.re_km;
    if (perige < 156.0) {
        sfour = perige - 78.0;
        if (perige < 98.0) sfour = 20.0;
        qzms24 = std::pow((120.0 - sfour) / k_.re_km, 4);
        sfour = sfour / k_.re_km + 1.0;
    }
    const double pinvsq = 1.0 / posq;
    const double tsi = 1.0 / (ao - sfour);
    eta_ = ao * ecco * tsi;
    const double etasq = eta_ * eta_;
    const double eeta = ecco * eta_;
    const double psisq = std::fabs(1.0 - etasq);
    const double coef = qzms24 * std::pow(tsi, 4);
    const double coef1 = coef / std::pow(psisq, 3.5);
    const double cc2 = coef1 * no_ *
                       (ao * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq)) +
                        0.375 * k_.j2 * tsi / psisq * con41_ * (8.0 + 3.0 * etasq * (8.0 + etasq)));
    cc1_ = bstar * cc2;
    double cc3 = 0.0;
    if (ecco > 1.0e-4) cc3 = -2.0 * coef * tsi * k_.j3oj2 * no_ * sinio / ecco;
    x1mth2_ = 1.0 - cosio2;
    cc4_ = 2.0 * no_ * coef1 * ao * omeosq *
           (eta_ * (2.0 + 0.5 * etasq) + ecco * (0.5 + 2.0 * etasq) -
            k_.j2 * tsi / (ao * psisq) *
                (-3.0 * con41_ * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta)) +
                 0.75 * x1mth2_ * (2.0 * etasq - eeta * (1.0 + etasq)) * std::cos(2.0 * argpo)));
    cc5_ = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq);
    const double cosio4 = cosio2 * cosio2;
    const double temp1 = 1.5 * k_.j2 * pinvsq * no_;
    const double temp2 = 0.5 * temp1 * k_.j2 * pinvsq;
    const double temp3 = -0.46875 * k_.j4 * pinvsq * pinvsq * no_;
    mdot_ = no_ + 0.5 * temp1 * rteosq * con41_ +
            0.0625 * temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4);
    argpdot_ = -0.5 * temp1 * con42 + 0.0625 * temp2 * (7.0 - 114.0 * cosio2 + 395.0 * cosio4) +
               temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4);
    const double xhdot1 = -temp1 * cosio;
    nodedot_ = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * cosio2) + 2.0 * temp3 * (3.0 - 7.0 * cosio2)) * cosio;
    const double xpidot = argpdot_ + nodedot_;
    omgcof_ = bstar * cc3 * std::cos(argpo);
    xmcof_ = 0.0;
    if (ecco > 1.0e-4) xmcof_ = -kX2o3 * coef * bstar / eeta;
    nodecf_ = 3.5 * omeosq * xhdot1 * cc1_;
    t2cof_ = 1.5 * cc1_;
    const double den = std::fabs(cosio + 1.0) > 1.5e-12 ? 1.0 + cosio : temp4;
    xlcof_ = -0.25 * k_.j3oj2 * sinio * (3.0 + 5.0 * cosio) / den;
    aycof_ = -0.5 * k_.j3oj2 * sinio;
    delmo_ = std::pow(1.0 + eta_ * std::cos(mo), 3);
    sinmao_ = std::sin(mo);
    x7thm1_ = 7.0 * cosio2 - 1.0;

    // --- Deep space: period of 225 minutes or more ---------------------------
    if (2.0 * kPi / no_ >= 225.0) {
        deep_ = true;
        isimp_ = true;
        DeepSpace& d = ds_;
        d.ls = lunar_solar(epoch, ecco, argpo, inclo, nodeo, no_);
        const LunarSolar& s = d.ls;

        constexpr double q22 = 1.7891679e-6, q31 = 2.1460748e-6, q33 = 2.2123015e-7;
        constexpr double root22 = 1.7891679e-6, root44 = 7.3636953e-9, root54 = 2.1765803e-9;
        constexpr double rptim = 4.37526908801129966e-3;  // Earth rotation, rad/min
        constexpr double root32 = 3.7393792e-7, root52 = 1.1428639e-7;
        constexpr double znl = 1.5835218e-4, zns = 1.19459e-5;

        const double nm = no_;
        const double em = ecco;
        const double sinim = s.sinim, cosim = s.cosim, emsq = s.emsq;
        if (nm < 0.0052359877 && nm > 0.0034906585) d.irez = 1;
        if (nm >= 8.26e-3 && nm <= 9.24e-3 && em >= 0.5) d.irez = 2;

        // Secular rates from the Sun ...
        const double ses = s.ss1 * zns * s.ss5;
        const double sis = s.ss2 * zns * (s.sz11 + s.sz13);
        const double sls = -zns * s.ss3 * (s.sz1 + s.sz3 - 14.0 - 6.0 * emsq);
        const double sghs = s.ss4 * zns * (s.sz31 + s.sz33 - 6.0);
        double shs = -zns * s.ss2 * (s.sz21 + s.sz23);
        const bool near_equatorial = inclo < 5.2359877e-2 || inclo > kPi - 5.2359877e-2;
        if (near_equatorial) shs = 0.0;
        if (sinim != 0.0) shs /= sinim;
        const double sgs = sghs - cosim * shs;
        // ... and the Moon.
        d.dedt = ses + s.s1 * znl * s.s5;
        d.didt = sis + s.s2 * znl * (s.z11 + s.z13);
        d.dmdt = sls - znl * s.s3 * (s.z1 + s.z3 - 14.0 - 6.0 * emsq);
        const double sghl = s.s4 * znl * (s.z31 + s.z33 - 6.0);
        double shll = -znl * s.s2 * (s.z21 + s.z23);
        if (near_equatorial) shll = 0.0;
        d.domdt = sgs + sghl;
        d.dnodt = shs;
        if (sinim != 0.0) {
            d.domdt -= cosim / sinim * shll;
            d.dnodt += shll / sinim;
        }

        const double theta = std::fmod(gsto_, kTwoPi);
        if (d.irez != 0) {
            const double aonv = std::pow(nm / k_.xke, kX2o3);
            if (d.irez == 2) {
                // Half-day (12-hour, e.g. Molniya, GNSS transfer) resonance.
                const double cosisq = cosim * cosim;
                const double e = ecco, e2 = eccsq, e3 = e * e2;
                const double g201 = -0.306 - (e - 0.64) * 0.440;
                double g211, g310, g322, g410, g422, g520, g521, g532, g533;
                if (e <= 0.65) {
                    g211 = 3.616 - 13.2470 * e + 16.2900 * e2;
                    g310 = -19.302 + 117.3900 * e - 228.4190 * e2 + 156.5910 * e3;
                    g322 = -18.9068 + 109.7927 * e - 214.6334 * e2 + 146.5816 * e3;
                    g410 = -41.122 + 242.6940 * e - 471.0940 * e2 + 313.9530 * e3;
                    g422 = -146.407 + 841.8800 * e - 1629.014 * e2 + 1083.4350 * e3;
                    g520 = -532.114 + 3017.977 * e - 5740.032 * e2 + 3708.2760 * e3;
                } else {
                    g211 = -72.099 + 331.819 * e - 508.738 * e2 + 266.724 * e3;
                    g310 = -346.844 + 1582.851 * e - 2415.925 * e2 + 1246.113 * e3;
                    g322 = -342.585 + 1554.908 * e - 2366.899 * e2 + 1215.972 * e3;
                    g410 = -1052.797 + 4758.686 * e - 7193.992 * e2 + 3651.957 * e3;
                    g422 = -3581.690 + 16178.110 * e - 24462.770 * e2 + 12422.520 * e3;
                    g520 = e > 0.715 ? -5149.66 + 29936.92 * e - 54087.36 * e2 + 31324.56 * e3
                                     : 1464.74 - 4664.75 * e + 3763.64 * e2;
                }
                if (e < 0.7) {
                    g533 = -919.22770 + 4988.6100 * e - 9064.7700 * e2 + 5542.21 * e3;
                    g521 = -822.71072 + 4568.6173 * e - 8491.4146 * e2 + 5337.524 * e3;
                    g532 = -853.66600 + 4690.2500 * e - 8624.7700 * e2 + 5341.4 * e3;
                } else {
                    g533 = -37995.780 + 161616.52 * e - 229838.20 * e2 + 109377.94 * e3;
                    g521 = -51752.104 + 218913.95 * e - 309468.16 * e2 + 146349.42 * e3;
                    g532 = -40023.880 + 170470.89 * e - 242699.48 * e2 + 115605.82 * e3;
                }
                const double sini2 = sinim * sinim;
                const double f220 = 0.75 * (1.0 + 2.0 * cosim + cosisq);
                const double f221 = 1.5 * sini2;
                const double f321 = 1.875 * sinim * (1.0 - 2.0 * cosim - 3.0 * cosisq);
                const double f322 = -1.875 * sinim * (1.0 + 2.0 * cosim - 3.0 * cosisq);
                const double f441 = 35.0 * sini2 * f220;
                const double f442 = 39.3750 * sini2 * sini2;
                const double f522 = 9.84375 * sinim *
                                    (sini2 * (1.0 - 2.0 * cosim - 5.0 * cosisq) +
                                     0.33333333 * (-2.0 + 4.0 * cosim + 6.0 * cosisq));
                const double f523 = sinim * (4.92187512 * sini2 * (-2.0 - 4.0 * cosim + 10.0 * cosisq) +
                                             6.56250012 * (1.0 + 2.0 * cosim - 3.0 * cosisq));
                const double f542 = 29.53125 * sinim *
                                    (2.0 - 8.0 * cosim + cosisq * (-12.0 + 8.0 * cosim + 10.0 * cosisq));
                const double f543 = 29.53125 * sinim *
                                    (-2.0 - 8.0 * cosim + cosisq * (12.0 + 8.0 * cosim - 10.0 * cosisq));
                const double xno2 = nm * nm;
                const double ainv2 = aonv * aonv;
                double t1 = 3.0 * xno2 * ainv2;
                double tt = t1 * root22;
                d.d2201 = tt * f220 * g201;
                d.d2211 = tt * f221 * g211;
                t1 *= aonv;
                tt = t1 * root32;
                d.d3210 = tt * f321 * g310;
                d.d3222 = tt * f322 * g322;
                t1 *= aonv;
                tt = 2.0 * t1 * root44;
                d.d4410 = tt * f441 * g410;
                d.d4422 = tt * f442 * g422;
                t1 *= aonv;
                tt = t1 * root52;
                d.d5220 = tt * f522 * g520;
                d.d5232 = tt * f523 * g532;
                tt = 2.0 * t1 * root54;
                d.d5421 = tt * f542 * g521;
                d.d5433 = tt * f543 * g533;
                d.xlamo = std::fmod(mo + nodeo + nodeo - theta - theta, kTwoPi);
                d.xfact = mdot_ + d.dmdt + 2.0 * (nodedot_ + d.dnodt - rptim) - no_;
            } else {
                // One-day (geosynchronous) resonance.
                const double g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq);
                const double g310 = 1.0 + 2.0 * emsq;
                const double g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq);
                const double f220 = 0.75 * (1.0 + cosim) * (1.0 + cosim);
                const double f311 = 0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) - 0.75 * (1.0 + cosim);
                double f330 = 1.0 + cosim;
                f330 = 1.875 * f330 * f330 * f330;
                d.del1 = 3.0 * nm * nm * aonv * aonv;
                d.del2 = 2.0 * d.del1 * f220 * g200 * q22;
                d.del3 = 3.0 * d.del1 * f330 * g300 * q33 * aonv;
                d.del1 = d.del1 * f311 * g310 * q31 * aonv;
                d.xlamo = std::fmod(mo + nodeo + argpo - theta, kTwoPi);
                d.xfact = mdot_ + xpidot - rptim + d.dmdt + d.domdt + d.dnodt - no_;
            }
        }
    }

    if (!isimp_) {
        const double cc1sq = cc1_ * cc1_;
        d2_ = 4.0 * ao * tsi * cc1sq;
        const double temp = d2_ * tsi * cc1_ / 3.0;
        d3_ = (17.0 * ao + sfour) * temp;
        d4_ = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) * cc1_;
        t3cof_ = d2_ + 2.0 * cc1sq;
        t4cof_ = 0.25 * (3.0 * d3_ + cc1_ * (12.0 * d2_ + 10.0 * cc1sq));
        t5cof_ = 0.2 * (3.0 * d4_ + 12.0 * cc1_ * d3_ + 6.0 * d2_ * d2_ + 15.0 * cc1sq * (2.0 * d2_ + cc1sq));
    }
}

inline Sgp4Output Sgp4::propagate_minutes(double t) const noexcept {
    using namespace detail_sgp4;
    Sgp4Output out;
    constexpr double temp4 = 1.5e-12;
    const double vkmpersec = k_.re_km * k_.xke / 60.0;

    // --- Secular gravity and drag ---------------------------------------------
    const double xmdf = mo_ + mdot_ * t;
    const double argpdf = argpo_ + argpdot_ * t;
    const double nodedf = nodeo_ + nodedot_ * t;
    double argpm = argpdf, mm = xmdf;
    const double t2 = t * t;
    double nodem = nodedf + nodecf_ * t2;
    double tempa = 1.0 - cc1_ * t;
    double tempe = bstar_ * cc4_ * t;
    double templ = t2cof_ * t2;
    if (!isimp_) {
        const double delomg = omgcof_ * t;
        const double delmtemp = 1.0 + eta_ * std::cos(xmdf);
        const double delm = xmcof_ * (delmtemp * delmtemp * delmtemp - delmo_);
        const double temp = delomg + delm;
        mm = xmdf + temp;
        argpm = argpdf - temp;
        const double t3 = t2 * t, t4 = t3 * t;
        tempa = tempa - d2_ * t2 - d3_ * t3 - d4_ * t4;
        tempe = tempe + bstar_ * cc5_ * (std::sin(mm) - sinmao_);
        templ = templ + t3cof_ * t3 + t4 * (t4cof_ + t * t5cof_);
    }

    double nm = no_, em = ecco_, inclm = inclo_;
    if (deep_) {
        // Lunar-solar secular rates and the resonance integrator
        // (Spacetrack Report #3 DSPACE), always started at epoch.
        const DeepSpace& d = ds_;
        constexpr double fasx2 = 0.13130908, fasx4 = 2.8843198, fasx6 = 0.37448087;
        constexpr double g22 = 5.7686396, g32 = 0.95240898, g44 = 1.8014998;
        constexpr double g52 = 1.0508330, g54 = 4.4108898;
        constexpr double rptim = 4.37526908801129966e-3;
        constexpr double stepp = 720.0, stepn = -720.0, step2 = 259200.0;

        const double theta = std::fmod(gsto_ + t * rptim, kTwoPi);
        em += d.dedt * t;
        inclm += d.didt * t;
        argpm += d.domdt * t;
        nodem += d.dnodt * t;
        mm += d.dmdt * t;

        if (d.irez != 0) {
            double atime = 0.0, xni = no_, xli = d.xlamo;
            const double delt = t > 0.0 ? stepp : stepn;
            double xndt = 0, xldot = 0, xnddt = 0, ft = 0;
            for (;;) {
                if (d.irez != 2) {
                    xndt = d.del1 * std::sin(xli - fasx2) + d.del2 * std::sin(2.0 * (xli - fasx4)) +
                           d.del3 * std::sin(3.0 * (xli - fasx6));
                    xldot = xni + d.xfact;
                    xnddt = d.del1 * std::cos(xli - fasx2) + 2.0 * d.del2 * std::cos(2.0 * (xli - fasx4)) +
                            3.0 * d.del3 * std::cos(3.0 * (xli - fasx6));
                    xnddt *= xldot;
                } else {
                    const double xomi = argpo_ + argpdot_ * atime;
                    const double x2omi = xomi + xomi;
                    const double x2li = xli + xli;
                    xndt = d.d2201 * std::sin(x2omi + xli - g22) + d.d2211 * std::sin(xli - g22) +
                           d.d3210 * std::sin(xomi + xli - g32) + d.d3222 * std::sin(-xomi + xli - g32) +
                           d.d4410 * std::sin(x2omi + x2li - g44) + d.d4422 * std::sin(x2li - g44) +
                           d.d5220 * std::sin(xomi + xli - g52) + d.d5232 * std::sin(-xomi + xli - g52) +
                           d.d5421 * std::sin(xomi + x2li - g54) + d.d5433 * std::sin(-xomi + x2li - g54);
                    xldot = xni + d.xfact;
                    xnddt = d.d2201 * std::cos(x2omi + xli - g22) + d.d2211 * std::cos(xli - g22) +
                            d.d3210 * std::cos(xomi + xli - g32) + d.d3222 * std::cos(-xomi + xli - g32) +
                            d.d5220 * std::cos(xomi + xli - g52) + d.d5232 * std::cos(-xomi + xli - g52) +
                            2.0 * (d.d4410 * std::cos(x2omi + x2li - g44) + d.d4422 * std::cos(x2li - g44) +
                                   d.d5421 * std::cos(xomi + x2li - g54) +
                                   d.d5433 * std::cos(-xomi + x2li - g54));
                    xnddt *= xldot;
                }
                if (std::fabs(t - atime) < stepp) {
                    ft = t - atime;
                    break;
                }
                xli = xli + xldot * delt + xndt * step2;
                xni = xni + xndt * delt + xnddt * step2;
                atime += delt;
            }
            nm = xni + xndt * ft + xnddt * ft * ft * 0.5;
            const double xl = xli + xldot * ft + xndt * ft * ft * 0.5;
            mm = d.irez != 1 ? xl - 2.0 * nodem + 2.0 * theta : xl - nodem - argpm + theta;
            nm = no_ + (nm - no_);
        }
    }

    if (nm <= 0.0) {
        out.status = Sgp4Status::MeanMotion;
        return out;
    }
    const double am = std::pow(k_.xke / nm, kX2o3) * tempa * tempa;
    nm = k_.xke / std::pow(am, 1.5);
    em -= tempe;
    if (em >= 1.0 || em < -0.001) {
        out.status = Sgp4Status::MeanEccentricity;
        return out;
    }
    if (em < 1.0e-6) em = 1.0e-6;
    mm += no_ * templ;
    double xlm = mm + argpm + nodem;
    nodem = std::fmod(nodem, kTwoPi);
    argpm = std::fmod(argpm, kTwoPi);
    xlm = std::fmod(xlm, kTwoPi);
    mm = std::fmod(xlm - argpm - nodem, kTwoPi);

    // --- Lunar-solar periodics (deep space) -----------------------------------
    double ep = em, xincp = inclm, argpp = argpm, nodep = nodem, mp = mm;
    double sinip = std::sin(inclm), cosip = std::cos(inclm);
    double aycof = aycof_, xlcof = xlcof_, con41 = con41_, x1mth2 = x1mth2_, x7thm1 = x7thm1_;
    if (deep_) {
        lunar_solar_periodics(ds_.ls, t, opt_.mode, ep, xincp, nodep, argpp, mp);
        if (xincp < 0.0) {
            xincp = -xincp;
            nodep += kPi;
            argpp -= kPi;
        }
        if (ep < 0.0 || ep > 1.0) {
            out.status = Sgp4Status::PerturbedEccentricity;
            return out;
        }
        sinip = std::sin(xincp);
        cosip = std::cos(xincp);
        aycof = -0.5 * k_.j3oj2 * sinip;
        const double den = std::fabs(cosip + 1.0) > 1.5e-12 ? 1.0 + cosip : temp4;
        xlcof = -0.25 * k_.j3oj2 * sinip * (3.0 + 5.0 * cosip) / den;
    }

    // --- Long-period periodics -------------------------------------------------
    const double axnl = ep * std::cos(argpp);
    double temp = 1.0 / (am * (1.0 - ep * ep));
    const double aynl = ep * std::sin(argpp) + temp * aycof;
    const double xl = mp + argpp + nodep + temp * xlcof * axnl;

    // --- Kepler's equation in the (axn, ayn) variables -------------------------
    const double u = std::fmod(xl - nodep, kTwoPi);
    double eo1 = u, tem5 = 9999.9, sineo1 = 0.0, coseo1 = 0.0;
    for (int k = 1; std::fabs(tem5) >= 1.0e-12 && k <= 10; ++k) {
        sineo1 = std::sin(eo1);
        coseo1 = std::cos(eo1);
        tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl;
        tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5;
        if (std::fabs(tem5) >= 0.95) tem5 = tem5 > 0.0 ? 0.95 : -0.95;
        eo1 += tem5;
    }

    // --- Short-period periodics ------------------------------------------------
    const double ecose = axnl * coseo1 + aynl * sineo1;
    const double esine = axnl * sineo1 - aynl * coseo1;
    const double el2 = axnl * axnl + aynl * aynl;
    const double pl = am * (1.0 - el2);
    if (pl < 0.0) {
        out.status = Sgp4Status::SemiLatusRectum;
        return out;
    }
    const double rl = am * (1.0 - ecose);
    const double rdotl = std::sqrt(am) * esine / rl;
    const double rvdotl = std::sqrt(pl) / rl;
    const double betal = std::sqrt(1.0 - el2);
    temp = esine / (1.0 + betal);
    const double sinu = am / rl * (sineo1 - aynl - axnl * temp);
    const double cosu = am / rl * (coseo1 - axnl + aynl * temp);
    double su = std::atan2(sinu, cosu);
    const double sin2u = (cosu + cosu) * sinu;
    const double cos2u = 1.0 - 2.0 * sinu * sinu;
    temp = 1.0 / pl;
    const double temp1 = 0.5 * k_.j2 * temp;
    const double temp2 = temp1 * temp;
    if (deep_) {
        const double cosisq = cosip * cosip;
        con41 = 3.0 * cosisq - 1.0;
        x1mth2 = 1.0 - cosisq;
        x7thm1 = 7.0 * cosisq - 1.0;
    }
    const double mrt = rl * (1.0 - 1.5 * temp2 * betal * con41) + 0.5 * temp1 * x1mth2 * cos2u;
    su -= 0.25 * temp2 * x7thm1 * sin2u;
    const double xnode = nodep + 1.5 * temp2 * cosip * sin2u;
    const double xinc = xincp + 1.5 * temp2 * cosip * sinip * cos2u;
    const double mvt = rdotl - nm * temp1 * x1mth2 * sin2u / k_.xke;
    const double rvdot = rvdotl + nm * temp1 * (x1mth2 * cos2u + 1.5 * con41) / k_.xke;

    // --- Orientation -------------------------------------------------------------
    const double sinsu = std::sin(su), cossu = std::cos(su);
    const double snod = std::sin(xnode), cnod = std::cos(xnode);
    const double sini = std::sin(xinc), cosi = std::cos(xinc);
    const double xmx = -snod * cosi, xmy = cnod * cosi;
    const Vector3d uvec(xmx * sinsu + cnod * cossu, xmy * sinsu + snod * cossu, sini * sinsu);
    const Vector3d vvec(xmx * cossu - cnod * sinsu, xmy * cossu - snod * sinsu, sini * cossu);
    out.r_km = mrt * uvec * k_.re_km;
    out.v_km_s = (mvt * uvec + rvdot * vvec) * vkmpersec;
    if (mrt < 1.0) out.status = Sgp4Status::Decayed;
    return out;
}

}  // namespace ballistic::sgp4
