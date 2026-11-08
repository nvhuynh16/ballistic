#pragma once

#include <vector>
#include <cmath>

namespace ballistic {
namespace calibration {
namespace detail {

/// Times at which to sample a B-spline: the arc start, every interior knot (the points where
/// the spline's polynomial pieces join) that lies strictly inside the arc, and the arc end.
/// `knots` is the full clamped knot vector of a spline of the given `degree`; times in seconds.
/// Returns {t_start, interior_knot_1, ..., interior_knot_N, t_end}.
[[nodiscard]] inline std::vector<double> knot_breakpoint_times(
    const std::vector<double>& knots, int degree,
    double t_start, double t_end)
{
    std::vector<double> times;
    times.push_back(t_start);
    for (size_t k = static_cast<size_t>(degree) + 1;
         k < knots.size() - static_cast<size_t>(degree) - 1; ++k) {
        double tk = knots[k];
        if (tk > t_start + 1e-10 && tk < t_end - 1e-10) {
            times.push_back(tk);
        }
    }
    times.push_back(t_end);
    return times;
}

} // namespace detail
} // namespace calibration
} // namespace ballistic
