#pragma once

// =============================================================================
// Events: stop (or log) a propagation where a condition on the state changes
// sign — an altitude crossing, ground impact, perigee, apogee, an epoch.
// =============================================================================
//
// Used by Trajectory::propagate_until. Each event is a function g(time,
// state) whose sign change marks the event (for example g = altitude - 100 km
// changes sign when the satellite crosses 100 km). After every integrator
// step each g is evaluated at the step's end; when it has changed sign in the
// requested direction, the crossing time is found inside the step by
// re-integrating from the step's start to trial times. The search is the
// "Illinois" variant of regula falsi (false position: a straight-line
// estimate of the root, with a fix that stops one end of the bracket from
// getting stuck), run to 1e-4 s. So the event state is as accurate as the
// integrator itself, not an interpolation. A terminal event ends the
// propagation exactly there.
//
// Ground impact is armed by default, so a propagation cannot fall through
// the Earth unnoticed (docs/trap_census.md, GROUND-PASS-THROUGH).
// =============================================================================

#include "core/types.hpp"
#include "coordinates/geodetic.hpp"
#include "time/epoch.hpp"
#include <algorithm>
#include <functional>
#include <string>
#include <utility>

namespace ballistic {
namespace events {

/// Which sign changes of g count as the event.
enum class Direction {
    Rising,    // g goes from negative to non-negative
    Falling,   // g goes from positive to non-positive
    Either,    // both
};

/// An event function: takes the time and the Earth-fixed (ITRF) position [m]
/// and velocity [m/s], returns a number whose sign change marks the event.
using EventFunction =
    std::function<double(const Epoch& epoch, const Vector3d& pos_ecef, const Vector3d& vel_ecef)>;

/// A condition to watch for during Trajectory::propagate_until.
struct Event {
    std::string name;       // label used in the log
    EventFunction g;
    Direction direction = Direction::Either;
    bool terminal = true;   // stop the propagation here (false = log and continue)
};

/// One located event: which event, when, and the Earth-fixed state there.
struct Occurrence {
    std::string name;
    Epoch epoch;
    Vector3d position_ecef = Vector3d::Zero();   // [m]
    Vector3d velocity_ecef = Vector3d::Zero();   // [m/s]
    bool terminal = false;                       // did it stop the propagation?
};

/// Fires when the geodetic altitude (height above the WGS84 reference
/// ellipsoid, the standard model of the Earth's shape) crosses `altitude_m`
/// [m] in `direction`.
[[nodiscard]] inline Event altitude_crossing(double altitude_m, Direction direction = Direction::Either,
                                             bool terminal = true) {
    return Event{
        "altitude " + std::to_string(altitude_m / 1000.0) + " km",
        [altitude_m](const Epoch&, const Vector3d& p, const Vector3d&) {
            return coords::ecef_to_geodetic(p).altitude - altitude_m;
        },
        direction, terminal};
}

/// Terminal event at the surface of the WGS84 ellipsoid, descending: where a
/// reentering object lands. Terrain and the geoid (the true sea-level surface)
/// are not modeled: the geoid lies within about 110 m of the ellipsoid, and
/// land rises up to about 9 km above it.
[[nodiscard]] inline Event ground_impact() {
    Event e = altitude_crossing(0.0, Direction::Falling, true);
    e.name = "ground impact";
    return e;
}

/// Perigee (lowest point of the orbit): r.v, which is proportional to the
/// radial velocity, goes from negative to positive. r.v is the same in the
/// Earth-fixed and inertial frames, since r.(w x r) = 0. Logged, not
/// terminal, by default.
[[nodiscard]] inline Event perigee(bool terminal = false) {
    return Event{"perigee",
                 [](const Epoch&, const Vector3d& p, const Vector3d& v) { return p.dot(v); },
                 Direction::Rising, terminal};
}

/// Apogee (highest point of the orbit): r.v goes from positive to negative.
[[nodiscard]] inline Event apogee(bool terminal = false) {
    return Event{"apogee",
                 [](const Epoch&, const Vector3d& p, const Vector3d& v) { return p.dot(v); },
                 Direction::Falling, terminal};
}

/// Fires at time `t` (for logging the state at a time inside a propagation;
/// a terminal epoch event is what propagate_until's target already is).
[[nodiscard]] inline Event at_epoch(const Epoch& t, bool terminal = false) {
    return Event{"epoch",
                 [t](const Epoch& e, const Vector3d&, const Vector3d&) { return e - t; },
                 Direction::Rising, terminal};
}

namespace detail {

/// Did g go from g0 to g1 across zero in direction d?
[[nodiscard]] inline bool crossed(double g0, double g1, Direction d) noexcept {
    const bool rising = g0 < 0.0 && g1 >= 0.0;
    const bool falling = g0 > 0.0 && g1 <= 0.0;
    switch (d) {
        case Direction::Rising: return rising;
        case Direction::Falling: return falling;
        case Direction::Either: return rising || falling;
    }
    return false;
}

// Root of f on [0, h] with f(0) = f0, f(h) = fh of opposite signs (fh may be
// 0): Illinois regula falsi to a bracket narrower than tol_s. Returns the
// right end of the final bracket, so the located state is ON the far side of
// the event (a terminal ground impact ends at altitude <= 0, never above).
template <typename F>
[[nodiscard]] double locate(F&& f, double h, double f0, double fh, double tol_s = 1e-4) {
    if (fh == 0.0) return h;
    double a = 0.0, b = h, fa = f0, fb = fh;
    int side = 0;
    for (int it = 0; it < 100 && (b - a) > tol_s; ++it) {
        double c = (a * fb - b * fa) / (fb - fa);
        // keep the trial strictly inside, at least a tenth of tol from the ends
        c = std::clamp(c, a + 0.1 * tol_s, b - 0.1 * tol_s);
        const double fc = f(c);
        if (fc == 0.0) return c;
        if ((fc < 0.0) == (fb < 0.0)) {
            b = c; fb = fc;
            if (side == -1) fa *= 0.5;
            side = -1;
        } else {
            a = c; fa = fc;
            if (side == +1) fb *= 0.5;
            side = +1;
        }
        if (b - a <= tol_s) break;
        // bisect when regula falsi stalls on one side
        if (it % 8 == 7) {
            const double m = 0.5 * (a + b);
            const double fm = f(m);
            if (fm == 0.0) return m;
            if ((fm < 0.0) == (fb < 0.0)) { b = m; fb = fm; } else { a = m; fa = fm; }
            side = 0;
        }
    }
    return b;
}

}  // namespace detail
}  // namespace events
}  // namespace ballistic
