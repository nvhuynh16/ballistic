#pragma once

// =============================================================================
// InitialState: where a trajectory starts, with its frame named
// =============================================================================
//
// A bare (epoch, position, velocity) triple does not say which frame it is
// in, and the two plausible readings differ by the Earth's rotation: an
// Earth-fixed velocity (ITRF, the International Terrestrial Reference Frame,
// which rotates with the Earth) and an inertial one (GCRS, the Geocentric
// Celestial Reference System, which does not rotate) differ by omega x r,
// the rotation rate times the position, up to ~500 m/s in low Earth orbit
// (LEO; 494 m/s over the equator at 400 km altitude).
// Given the wrong one the propagator flies a different, usually eccentric,
// orbit without complaint. InitialState makes the caller name the frame
// (from_itrf / from_gcrs / from_j2000 / from_keplerian / from_teme; two-line
// element sets: from_tle; SP3 precise-orbit files: sp3::initial_state) and
// checks the state before anything runs:
//
//   * units: |r| < 100 km means the position is in km; |r| > 1e6 km is not
//     an Earth orbit;
//   * the ground: geodetic altitude below -1 km;
//   * the orbit: a perigee (lowest point of the orbit) below 100 km is a
//     reentry or a ballistic arc, and
//     needs allow_suborbital = true (otherwise it is usually a unit or frame
//     mistake);
//   * the frame: the same numbers read in the other frame (v -/+ omega x r).
//     If the declared reading is suborbital or hyperbolic while the swapped
//     one is a bound orbit with perigee above 150 km and an eccentricity at
//     least ten times smaller, FrameError; if both are plausible but the
//     swapped one is far more circular, a FrameAssumption diagnostic.
//
// The checks are rules of thumb applied to the "osculating" two-body orbit:
// the Kepler ellipse (or hyperbola) the state would follow under point-mass
// gravity alone. They need no Earth orientation parameters (EOP). A failing
// check raises InputPlausibilityError or FrameError with the fix; a
// FrameAssumption note goes through core/diagnostics.hpp.
// =============================================================================

#include "core/constants.hpp"
#include "core/diagnostics.hpp"
#include "core/errors.hpp"
#include "core/types.hpp"
#include "coordinates/geodetic.hpp"
#include "time/epoch.hpp"
#include "util/frame_transform.hpp"
#include "sgp4/sgp4.hpp"
#include "sgp4/teme.hpp"
#include "sgp4/tle.hpp"

#include <cmath>
#include <cstdio>
#include <optional>
#include <string>

namespace ballistic {

/// The frame a start state is expressed in: Earth-fixed (ITRF) or inertial (GCRS).
enum class StateFrame { ITRF, GCRS };

/// "ITRF" or "GCRS", for messages.
[[nodiscard]] inline const char* state_frame_name(StateFrame f) noexcept {
    return f == StateFrame::ITRF ? "ITRF" : "GCRS";
}

/// The name of an ITRF realization ("ITRF2020", ...), for messages.
[[nodiscard]] inline const char* itrf_realization_name(coords::ITRFRealization r) noexcept {
    switch (r) {
        case coords::ITRFRealization::ITRF2000: return "ITRF2000";
        case coords::ITRFRealization::ITRF2005: return "ITRF2005";
        case coords::ITRFRealization::ITRF2008: return "ITRF2008";
        case coords::ITRFRealization::ITRF2014: return "ITRF2014";
        case coords::ITRFRealization::ITRF2020: return "ITRF2020";
    }
    return "ITRF?";
}

/// Switches for the start-state checks.
struct PlausibilityOptions {
    bool allow_suborbital = false;  // reentry, ballistic or launch state: perigee below 100 km is fine
    bool check_frame = true;        // the Earth-fixed vs inertial mix-up test
};

/// Classical (Keplerian) orbital elements of a two-body ellipse in the GCRS,
/// with the Earth's gravitational parameter GM = constants::earth::GM.
/// Angles in radians.
struct KeplerianElements {
    double a_m = 0.0;                // semi-major axis [m] (half the ellipse's long axis)
    double e = 0.0;                  // eccentricity, [0, 1) (0 = circle)
    double i_rad = 0.0;              // inclination: tilt of the orbit plane to the equator
    double raan_rad = 0.0;           // right ascension of the ascending node: where the
                                     // orbit crosses the equator northward, measured from
                                     // the vernal equinox direction
    double argp_rad = 0.0;           // argument of perigee: angle from that node to perigee
    double true_anomaly_rad = 0.0;   // angle from perigee to the satellite
};

/// The plausibility checks behind InitialState (also usable on their own).
namespace state_checks {

/// Summary of the two-body orbit a state implies.
struct TwoBody {
    bool bound = false;
    double e = 0.0;
    double perigee_alt_m = 0.0;      // above the equatorial radius
};

/// Two-body orbit of an inertial position r [m] and velocity v [m/s].
[[nodiscard]] inline TwoBody two_body(const Vector3d& r, const Vector3d& v) {
    const double mu = constants::earth::GM;
    const double rn = r.norm(), v2 = v.squaredNorm();
    const double energy = 0.5 * v2 - mu / rn;
    const Vector3d e_vec = ((v2 - mu / rn) * r - r.dot(v) * v) / mu;
    TwoBody t;
    t.e = e_vec.norm();
    t.bound = energy < 0.0;
    const double h2 = r.cross(v).squaredNorm();
    const double rp = h2 / mu / (1.0 + t.e);   // valid for every conic
    t.perigee_alt_m = rp - constants::earth::RADIUS_EQUATORIAL;
    return t;
}

/// Meters formatted as "123 km", for messages.
[[nodiscard]] inline std::string km(double m) {
    char b[32];
    std::snprintf(b, sizeof b, "%.0f km", m / 1e3);
    return b;
}

/// One-line description of a TwoBody orbit, for messages.
[[nodiscard]] inline std::string orbit_text(const TwoBody& t) {
    char b[96];
    if (t.bound)
        std::snprintf(b, sizeof b, "bound, e %.3g, perigee %.0f km", t.e, t.perigee_alt_m / 1e3);
    else
        std::snprintf(b, sizeof b, "hyperbolic, e %.3g", t.e);
    return b;
}

/// Unit checks only: throws InputPlausibilityError if r [m] or v [m/s] is not
/// finite, or if |r| < 100 km (kilometers passed as meters) or > 1e6 km.
/// Cheap and never wrong for a real state.
inline void check_units(const Vector3d& r, const Vector3d& v) {
    if (!r.allFinite() || !v.allFinite())
        throw InputPlausibilityError("initial state: position or velocity is not finite");
    const double rn = r.norm();
    if (rn < 1e5)
        throw InputPlausibilityError(
            "initial state: |r| = " + std::to_string(rn) + " m is inside the Earth. Positions are "
            "in METERS; if these are km, multiply position and velocity by 1000");
    if (rn > 1e9)
        throw InputPlausibilityError(
            "initial state: |r| = " + km(rn) + " is beyond any Earth orbit (2.6 lunar distances); "
            "positions are in meters");
}

/// All start-state checks (see the file comment) for a state declared in
/// `frame`. Throws InputPlausibilityError (units, below ground, suborbital
/// without opt.allow_suborbital) or FrameError (the numbers make far more
/// sense in the other frame); may report a FrameAssumption diagnostic.
inline void check_plausibility(const Vector3d& r, const Vector3d& v, StateFrame frame,
                               const PlausibilityOptions& opt) {
    check_units(r, v);
    const double alt = coords::ecef_to_altitude(r);
    if (alt < -1e3)
        throw InputPlausibilityError(
            "initial state: geodetic altitude " + km(alt) + " is below the ground");

    const Vector3d wxr = coords::earth_rotation_vector().cross(r);
    const bool itrf = frame == StateFrame::ITRF;
    const TwoBody decl = two_body(r, itrf ? Vector3d(v + wxr) : v);
    const TwoBody swap = two_body(r, itrf ? v : Vector3d(v + wxr));
    const char* other = itrf ? "GCRS (inertial)" : "ITRF (Earth-fixed)";
    const bool swap_good = swap.bound && swap.perigee_alt_m > 150e3;
    const std::string mixup =
        std::string("initial state declared ") + state_frame_name(frame) + " reads as " +
        orbit_text(decl) + "; the same numbers read as " + other + " give " + orbit_text(swap) +
        ". The two differ by omega x r = " + std::to_string(static_cast<int>(wxr.norm())) +
        " m/s. ";
    const std::string fix = itrf
        ? "If the velocity is inertial, use from_gcrs (or from_j2000)."
        : "If the velocity is Earth-fixed, use from_itrf.";

    const bool suborbital = decl.perigee_alt_m < 100e3;
    if (opt.allow_suborbital) {
        if (opt.check_frame && suborbital && swap_good)
            diagnostics::report(diagnostics::Category::FrameAssumption,
                                mixup + "You allowed a suborbital state; " + fix);
        return;
    }
    if (opt.check_frame && (suborbital || !decl.bound) && swap_good && swap.e <= decl.e / 10.0)
        throw FrameError(mixup + fix + " (check_frame = false skips this test)");
    if (suborbital)
        throw InputPlausibilityError(
            "initial state: the orbit is suborbital (" + orbit_text(decl) + "). For a reentry, "
            "ballistic or launch state pass allow_suborbital = true." +
            (v.norm() < 50.0 ? " |v| < 50 m/s: if the velocity is in km/s, multiply it by 1000." : ""));
    if (opt.check_frame && decl.e > 0.01 && swap_good && swap.e < 0.01 && swap.e < decl.e / 10.0)
        diagnostics::report(diagnostics::Category::FrameAssumption,
                            mixup + "The swapped reading is much more circular. " + fix +
                            " (check_frame = false silences this)");
}

/// Rotation matrix from the GCRS to the J2000 frame (EME2000: the mean
/// equator and equinox of 1 Jan 2000, 12:00 TT). The two differ by a small
/// fixed "frame bias" of ~23 milliarcseconds (~0.7 m at LEO); the angles are
/// the IAU 2000 frame-bias values of ERFA's eraBi00, assembled as eraBp00 does.
[[nodiscard]] inline Eigen::Matrix3d frame_bias_gcrs_to_j2000() {
    constexpr double as = 3.14159265358979323846 / (180.0 * 3600.0);
    const double dpsibi = -0.041775 * as, depsbi = -0.0068192 * as, dra0 = -0.0146 * as;
    const double eps0 = 84381.448 * as;
    auto rx = [](double a) { Eigen::Matrix3d m; const double c = std::cos(a), s = std::sin(a);
                             m << 1, 0, 0, 0, c, s, 0, -s, c; return m; };
    auto ry = [](double a) { Eigen::Matrix3d m; const double c = std::cos(a), s = std::sin(a);
                             m << c, 0, -s, 0, 1, 0, s, 0, c; return m; };
    auto rz = [](double a) { Eigen::Matrix3d m; const double c = std::cos(a), s = std::sin(a);
                             m << c, s, 0, -s, c, 0, 0, 0, 1; return m; };
    return rx(-depsbi) * ry(dpsibi * std::sin(eps0)) * rz(dra0);
}

}  // namespace state_checks

/// A checked start state for a Trajectory: epoch, position, velocity and the
/// frame they are in. Build one with the static from_* functions; each runs
/// the checks above and throws InputPlausibilityError or FrameError on a
/// state that looks wrong.
class InitialState {
public:
    /// Earth-fixed position [m] and velocity [m/s] (velocity relative to the
    /// rotating Earth, as satellite-navigation receivers and SP3/EOF orbit
    /// products give it). `realization` is the ITRF version the coordinates
    /// are in; empty = whatever the trajectory's config says.
    [[nodiscard]] static InitialState from_itrf(
        const Epoch& epoch, const Vector3d& r_m, const Vector3d& v_m_s,
        std::optional<coords::ITRFRealization> realization = std::nullopt,
        const PlausibilityOptions& opt = {}) {
        state_checks::check_plausibility(r_m, v_m_s, StateFrame::ITRF, opt);
        return InitialState(epoch, r_m, v_m_s, StateFrame::ITRF, realization);
    }

    /// Inertial position [m] and velocity [m/s] in the GCRS (axes of the
    /// International Celestial Reference System, origin at the Earth's center).
    [[nodiscard]] static InitialState from_gcrs(
        const Epoch& epoch, const Vector3d& r_m, const Vector3d& v_m_s,
        const PlausibilityOptions& opt = {}) {
        state_checks::check_plausibility(r_m, v_m_s, StateFrame::GCRS, opt);
        return InitialState(epoch, r_m, v_m_s, StateFrame::GCRS, std::nullopt);
    }

    /// Inertial position [m] and velocity [m/s] in J2000 (EME2000). Rotated
    /// to the GCRS by the frame bias (~23 mas, ~0.7 m at LEO).
    [[nodiscard]] static InitialState from_j2000(
        const Epoch& epoch, const Vector3d& r_m, const Vector3d& v_m_s,
        const PlausibilityOptions& opt = {}) {
        const Eigen::Matrix3d b = state_checks::frame_bias_gcrs_to_j2000().transpose();
        return from_gcrs(epoch, b * r_m, b * v_m_s, opt);
    }

    /// From Keplerian elements in the GCRS. Throws ConfigError unless a > 0
    /// and 0 <= e < 1, and InputPlausibilityError if a < 100 km (kilometers
    /// passed as meters).
    [[nodiscard]] static InitialState from_keplerian(
        const Epoch& epoch, const KeplerianElements& k, const PlausibilityOptions& opt = {}) {
        if (!(k.a_m > 0.0) || !(k.e >= 0.0 && k.e < 1.0))
            throw ConfigError("from_keplerian: needs a > 0 and 0 <= e < 1 (got a = " +
                              std::to_string(k.a_m) + " m, e = " + std::to_string(k.e) + ")");
        if (k.a_m < 1e5)
            throw InputPlausibilityError("from_keplerian: a = " + std::to_string(k.a_m) +
                                         " m; the semi-major axis is in METERS");
        const double mu = constants::earth::GM;
        const double p = k.a_m * (1.0 - k.e * k.e);
        const double cn = std::cos(k.true_anomaly_rad), sn = std::sin(k.true_anomaly_rad);
        const double rn = p / (1.0 + k.e * cn);
        const Vector3d r_pf(rn * cn, rn * sn, 0.0);
        const Vector3d v_pf(-std::sqrt(mu / p) * sn, std::sqrt(mu / p) * (k.e + cn), 0.0);
        const Eigen::Matrix3d q =
            (Eigen::AngleAxisd(k.raan_rad, Vector3d::UnitZ()) *
             Eigen::AngleAxisd(k.i_rad, Vector3d::UnitX()) *
             Eigen::AngleAxisd(k.argp_rad, Vector3d::UnitZ())).toRotationMatrix();
        return from_gcrs(epoch, q * r_pf, q * v_pf, opt);
    }

    /// Position [m] and velocity [m/s] in TEME (true equator, mean equinox:
    /// the frame SGP4, the standard propagator for TLEs, outputs; see
    /// sgp4/teme.hpp), converted to the ITRF through the 1982 Greenwich mean
    /// sidereal time (GMST) and polar motion. The ITRF realization is that of
    /// the EOP series (a millimeter-level difference; left unstated).
    [[nodiscard]] static InitialState from_teme(
        const Epoch& epoch, const Vector3d& r_m, const Vector3d& v_m_s,
        const PlausibilityOptions& opt = {}) {
        const StateVector e = sgp4::teme_to_itrf(StateVector(r_m, v_m_s), epoch);
        return from_itrf(epoch, e.position, e.velocity, std::nullopt, opt);
    }

    /// The state SGP4 gives for a two-line element set (TLE), at the TLE's
    /// own epoch or at `at` (SGP4 run with the WGS-72 constants that element
    /// sets are fitted with). A TLE is ~0.5 km wrong at LEO (Flohrer et al.
    /// 2009). Measured caveat (docs/validation_claims.md TLE-GRACEFO): this
    /// state's semi-major axis is tens of meters off, so a NUMERICAL
    /// propagation from it drifts by kilometers per day (12-24 km at 490 km,
    /// 3.6-7 days before the TLE epoch, where SGP4 itself stayed within 1-3 km;
    /// forward horizons are not measured yet). For horizons
    /// beyond hours, Python's ballistic.tle.fit_state (a state fitted to SGP4
    /// over one day: 0.3-1.8 km there) is the better start;
    /// recommended_config(..., accuracy="tle") picks the forces that still
    /// matter at TLE accuracy.
    [[nodiscard]] static InitialState from_tle(
        const sgp4::Tle& tle, std::optional<Epoch> at = std::nullopt,
        const PlausibilityOptions& opt = {}) {
        const sgp4::Sgp4 model(tle);
        const Epoch t = at.value_or(tle.epoch());
        const StateVector x = model.teme(t);
        return from_teme(t, x.position, x.velocity, opt);
    }

    /// Epoch of the state.
    [[nodiscard]] const Epoch& epoch() const noexcept { return epoch_; }
    /// Frame the stored position and velocity are in.
    [[nodiscard]] StateFrame frame() const noexcept { return frame_; }
    /// Stored position [m], in frame().
    [[nodiscard]] const Vector3d& position() const noexcept { return r_; }
    /// Stored velocity [m/s], in frame().
    [[nodiscard]] const Vector3d& velocity() const noexcept { return v_; }
    /// ITRF realization of an Earth-fixed state, if one was given.
    [[nodiscard]] std::optional<coords::ITRFRealization> realization() const noexcept {
        return realization_;
    }

    /// The state in the Earth-fixed frame, realization `itrf` (what
    /// Trajectory's propagators take). Throws ConfigError if the state was
    /// given in a different ITRF realization: set TrajectoryConfig::itrf to
    /// the state's own. A GCRS state is rotated (needs EOP data; the
    /// transform throws DataCoverageError without it, unless
    /// EopPolicy::AssumeZero is set).
    [[nodiscard]] StateVector itrf_state(coords::ITRFRealization itrf) const {
        if (frame_ == StateFrame::GCRS) return util::gcrs_to_itrf(StateVector(r_, v_), epoch_, itrf);
        if (realization_ && *realization_ != itrf)
            throw ConfigError(std::string("initial state is in ") +
                              itrf_realization_name(*realization_) + " but the trajectory is "
                              "configured for " + itrf_realization_name(itrf) +
                              "; set TrajectoryConfig::itrf to the state's realization");
        return StateVector(r_, v_);
    }

    /// The state in the inertial GCRS frame. An ITRF state is rotated using
    /// its own realization, or `itrf` if it stated none.
    [[nodiscard]] StateVector gcrs_state(coords::ITRFRealization itrf) const {
        if (frame_ == StateFrame::GCRS) return StateVector(r_, v_);
        return util::itrf_to_gcrs(StateVector(r_, v_), epoch_, realization_.value_or(itrf));
    }

private:
    InitialState(const Epoch& t, const Vector3d& r, const Vector3d& v, StateFrame f,
                 std::optional<coords::ITRFRealization> real)
        : epoch_(t), r_(r), v_(v), frame_(f), realization_(real) {}

    Epoch epoch_;
    Vector3d r_, v_;
    StateFrame frame_ = StateFrame::ITRF;
    std::optional<coords::ITRFRealization> realization_;
};

}  // namespace ballistic
