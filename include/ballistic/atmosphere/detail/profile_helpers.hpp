#pragma once

// =============================================================================
// Interpolation helpers shared by the density, wind and beta tables.
// =============================================================================

#include "../../core/types.hpp"
#include "../../time/epoch.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace ballistic {
namespace atmosphere {
namespace detail {

// Where `t` falls in a sorted list of epochs, for linear interpolation.
// Returns (lo, frac) such that
//   value = series[lo] + frac * (series[lo+1] - series[lo])
// Epochs before the first return (0, 0) and after the last (n-2, 1), i.e.
// the edge value is held. Fewer than two epochs return (0, 0).
inline std::pair<std::size_t, double>
linear_time_lookup(const std::vector<Epoch>& times, const Epoch& t) {
    if (times.size() < 2) return {0, 0.0};
    if (t <= times.front()) return {0, 0.0};
    if (t >= times.back()) return {times.size() - 2, 1.0};
    auto it = std::upper_bound(times.begin(), times.end(), t);
    const std::size_t hi = static_cast<std::size_t>(std::distance(times.begin(), it));
    const std::size_t lo = hi - 1;
    const double dt = times[hi] - times[lo];
    const double frac = (dt > 0.0) ? (t - times[lo]) / dt : 0.0;
    return {lo, frac};
}

// The same for a sorted numeric axis (latitude, longitude, ...): returns
// (lo, frac), holding the edge value outside the axis.
inline std::pair<std::size_t, double>
find_axis_index(const std::vector<double>& axis, double value) {
    const std::size_t n = axis.size();
    if (n < 2) return {0, 0.0};
    if (value <= axis.front()) return {0, 0.0};
    if (value >= axis.back()) return {n - 2, 1.0};
    auto it = std::upper_bound(axis.begin(), axis.end(), value);
    const std::size_t hi = static_cast<std::size_t>(std::distance(axis.begin(), it));
    const std::size_t lo = hi - 1;
    const double d = axis[hi] - axis[lo];
    const double frac = (d > 0.0) ? (value - axis[lo]) / d : 0.0;
    return {lo, frac};
}

// Wrap a longitude [deg] into [-180, 180), the range the 4D grids use.
inline double wrap_longitude_deg(double lon_deg) noexcept {
    double l = std::fmod(lon_deg + 180.0, 360.0);
    if (l < 0.0) l += 360.0;
    return l - 180.0;
}

}  // namespace detail
}  // namespace atmosphere
}  // namespace ballistic
