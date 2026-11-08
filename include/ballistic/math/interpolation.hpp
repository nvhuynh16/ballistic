#pragma once

// =============================================================================
// Lagrange Polynomial Interpolation Utilities
// =============================================================================
//
// Lagrange polynomial interpolation: the single polynomial of degree n-1 that passes through
// n neighboring samples, evaluated (or differentiated) at a time between them. Used to read
// precise reference orbits between their tabulated epochs: EOFParser uses it for ESA orbit
// files (EOF); SP3 files (the standard precise-orbit text format) have their own reader in
// validation/sp3.hpp.
//
// References:
// -----------
// [1] Press, W.H., et al. "Numerical Recipes", 3rd Edition, Section 3.1.
//     - Polynomial Interpolation and Extrapolation (Lagrange basis)
// [2] Montenbruck, O. & Gill, E. "Satellite Orbits", Springer, 2000, Section 3.3.2.
//     - Lagrange interpolation for precise orbit data (SP3)
//
// =============================================================================

#include "../core/defaults.hpp"
#include "../core/errors.hpp"
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <optional>
#include <cmath>

namespace ballistic {
namespace math {

// =============================================================================
// Interpolation Window Helper (binary search)
// =============================================================================

/// Start index of an `order`-point interpolation window centered on the sample nearest to `t`,
/// found by binary search. `get_time(i)` returns the i-th sample time; times must increase.
/// The window is shifted, not shrunk, near the ends so that [start, start + order) stays inside
/// [0, data_size). Requires data_size >= order (not checked here).
template<typename TimeGetter>
[[nodiscard]] int find_interpolation_window(
    const TimeGetter& get_time, size_t data_size, double t, int order) noexcept
{
    // Binary search for first index with time >= t
    size_t lo = 0, hi = data_size;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (get_time(mid) < t) lo = mid + 1;
        else hi = mid;
    }

    // Nearest of lo and lo-1
    size_t center = lo;
    if (lo > 0 && lo < data_size) {
        if (std::abs(get_time(lo - 1) - t) < std::abs(get_time(lo) - t))
            center = lo - 1;
    } else if (lo >= data_size) {
        center = data_size - 1;
    }

    // Clamp window
    int start = static_cast<int>(center) - order / 2;
    if (start < 0) start = 0;
    if (start + order > static_cast<int>(data_size))
        start = static_cast<int>(data_size) - order;
    return start;
}

namespace detail {

// RAW Lagrange templates: no range check, so a time outside the data is
// EXTRAPOLATED (a degree-8 polynomial: wildly wrong a few nodes out). Use the
// range-checked math::lagrange_interpolate / lagrange_derivative, or
// validation::sp3::Arc::interpolate for SP3 data (enforced by the static rule LAGRANGE-RAW).
// Data are passed through accessor callbacks: get_time(i) -> double, get_value(i) -> Vector3d.

// =============================================================================
// Lagrange Interpolation for Vector Data
// =============================================================================

// Interpolate a 3D vector at time t using Lagrange polynomial
//
// Parameters:
//   get_time   - i -> time of sample i (must increase with i)
//   get_value  - i -> Vector3d value of sample i
//   data_size  - number of samples
//   t          - Target time for interpolation (same unit as get_time)
//   order      - Number of points to use (default 9 = 8th degree polynomial)
//
// Returns:
//   Interpolated Vector3d at time t, or nullopt if there are fewer than `order` samples
//
// Note: any time unit works; EOFParser passes seconds from its first record.
template<typename TimeGetter, typename ValueGetter>
[[nodiscard]] std::optional<Eigen::Vector3d> lagrange_interpolate_vector(
    const TimeGetter& get_time,
    const ValueGetter& get_value,
    size_t data_size,
    double t,
    int order = defaults::LAGRANGE_DEFAULT_ORDER)
{
    if (data_size < static_cast<size_t>(order)) {
        return std::nullopt;
    }

    int start = find_interpolation_window(get_time, data_size, t, order);

    // Lagrange interpolation
    Eigen::Vector3d result = Eigen::Vector3d::Zero();

    for (int i = 0; i < order; ++i) {
        double ti = get_time(start + i);
        double Li = 1.0;

        for (int j = 0; j < order; ++j) {
            if (i != j) {
                double tj = get_time(start + j);
                Li *= (t - tj) / (ti - tj);
            }
        }

        result += Li * get_value(start + i);
    }

    return result;
}

// =============================================================================
// Lagrange Derivative for Velocity Computation
// =============================================================================

// Compute derivative of Lagrange polynomial at time t
// This is used to derive velocity from position-only data
//
// Parameters:
//   get_time   - i -> time of sample i (must increase with i)
//   get_value  - i -> Vector3d position of sample i
//   data_size  - number of samples
//   t          - Target time for derivative computation
//   order      - Number of points to use (default 9 = 8th degree polynomial)
//
// Returns:
//   Derivative of interpolated vector (e.g., velocity = dPosition/dt), or nullopt if there
//   are fewer than `order` samples. Units are (value units) / (time units), e.g. meters/day
//   if times are Julian days
//
// Note: Caller must convert units if needed (e.g., divide by 86400 for m/s)
template<typename TimeGetter, typename ValueGetter>
[[nodiscard]] std::optional<Eigen::Vector3d> lagrange_derivative_vector(
    const TimeGetter& get_time,
    const ValueGetter& get_value,
    size_t data_size,
    double t,
    int order = defaults::LAGRANGE_DEFAULT_ORDER)
{
    if (data_size < static_cast<size_t>(order)) {
        return std::nullopt;
    }

    int start = find_interpolation_window(get_time, data_size, t, order);

    // Lagrange interpolation derivative
    // d/dt L_i(t) = sum_{j!=i} [ 1/(t_i - t_j) * prod_{k!=i,j} (t - t_k)/(t_i - t_k) ]
    Eigen::Vector3d result = Eigen::Vector3d::Zero();

    for (int i = 0; i < order; ++i) {
        double ti = get_time(start + i);

        // Compute derivative of Lagrange basis polynomial L_i(t)
        double dLi = 0.0;
        for (int j = 0; j < order; ++j) {
            if (j == i) continue;
            double tj = get_time(start + j);

            // Product term for this j
            double prod = 1.0 / (ti - tj);
            for (int k = 0; k < order; ++k) {
                if (k == i || k == j) continue;
                double tk = get_time(start + k);
                prod *= (t - tk) / (ti - tk);
            }
            dLi += prod;
        }

        result += dLi * get_value(start + i);
    }

    return result;
}

}  // namespace detail

namespace detail {
// Times outside [front, back] would be extrapolated: refuse them.
inline void require_inside(const std::vector<double>& times, double t) {
    if (times.empty() || !(t >= times.front() && t <= times.back())) {
        throw DataCoverageError("Lagrange interpolation at t = " + std::to_string(t) +
            " is outside the data (" +
            (times.empty() ? std::string("none")
                           : std::to_string(times.front()) + " to " + std::to_string(times.back())) +
            "); it would extrapolate");
    }
}
}  // namespace detail

// =============================================================================
// Convenience Overloads for std::vector (range-checked: DataCoverageError
// outside [times.front(), times.back()]; nullopt for too few points)
// =============================================================================

/// Interpolate a 3D vector (e.g. position [m]) at time `t` from samples `positions` at `times`
/// (increasing; any unit), using `order` neighboring points (default 9).
/// Returns nullopt when the two vectors differ in length or hold fewer than `order` points.
/// @throws DataCoverageError when `t` lies outside [times.front(), times.back()]
[[nodiscard]] inline std::optional<Eigen::Vector3d> lagrange_interpolate(
    const std::vector<double>& times,
    const std::vector<Eigen::Vector3d>& positions,
    double t,
    int order = defaults::LAGRANGE_DEFAULT_ORDER)
{
    if (times.size() != positions.size()) {
        return std::nullopt;
    }
    detail::require_inside(times, t);

    auto get_time = [&](size_t i) { return times[i]; };
    auto get_value = [&](size_t i) { return positions[i]; };

    return detail::lagrange_interpolate_vector(get_time, get_value, times.size(), t, order);
}

/// Time derivative (e.g. velocity) of the interpolating polynomial at `t`, in value units per
/// time unit of `times`. Same arguments, nullopt cases and DataCoverageError as
/// `lagrange_interpolate`.
[[nodiscard]] inline std::optional<Eigen::Vector3d> lagrange_derivative(
    const std::vector<double>& times,
    const std::vector<Eigen::Vector3d>& positions,
    double t,
    int order = defaults::LAGRANGE_DEFAULT_ORDER)
{
    if (times.size() != positions.size()) {
        return std::nullopt;
    }
    detail::require_inside(times, t);

    auto get_time = [&](size_t i) { return times[i]; };
    auto get_value = [&](size_t i) { return positions[i]; };

    return detail::lagrange_derivative_vector(get_time, get_value, times.size(), t, order);
}

} // namespace math
} // namespace ballistic
