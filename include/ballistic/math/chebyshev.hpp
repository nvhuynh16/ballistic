#pragma once

// =============================================================================
// Chebyshev Polynomial Evaluation
// =============================================================================
//
// Evaluation of Chebyshev series: f(x) = sum c_i * T_i(x) on x in [-1, 1], where T_i are the
// Chebyshev polynomials (T_0 = 1, T_1 = x, T_{k+1} = 2x*T_k - T_{k-1}). Planetary ephemerides
// such as JPL's DE440 store each body's coordinates as Chebyshev coefficients over short time
// intervals; the caller maps a time t in an interval to x = (t - mid) / half_interval.
//
// References:
// -----------
// [1] Press, W.H., et al. "Numerical Recipes", 3rd Edition.
//     Section 5.8: Chebyshev Approximation - Clenshaw's recurrence
//
// [2] Clenshaw, C.W. (1955). "A note on the summation of Chebyshev series".
//     Mathematics of Computation, 9(51), 118-120.
//
// Chebyshev Polynomial Evaluation:
// --------------------------------
// Uses Clenshaw's algorithm for numerical stability:
//   f(x) = sum(c_i * T_i(x))
//   Computed via recurrence: b_k = c_k + 2*x*b_{k+1} - b_{k+2}
//   Result: c_0 + x*b_1 - b_2
//
// This avoids computing individual Chebyshev polynomials which can
// overflow/underflow for high degrees.
//
// =============================================================================

#include "../core/types.hpp"
#include <cstddef>
#include <utility>
#include <vector>

namespace ballistic {
namespace math {

/// Up to this many coefficients the derivative's coefficients live in a stack array; longer
/// series use a heap buffer (same arithmetic). DE440 uses n_coeffs <= 13.
constexpr int MAX_CHEBYSHEV_COEFFS = 64;

/// Evaluate a Chebyshev series with Clenshaw's algorithm.
/// @param coeffs    n_coeffs coefficients c_0 .. c_{n-1}
/// @param x         normalized argument, expected in [-1, 1] (not checked)
/// @return sum(coeffs[i] * T_i(x)); 0 when n_coeffs <= 0
[[nodiscard]] inline double chebyshev_eval(
    const double* coeffs, int n_coeffs, double x) noexcept
{
    if (n_coeffs <= 0) return 0.0;
    if (n_coeffs == 1) return coeffs[0];

    // Clenshaw recurrence
    double b_kp2 = 0.0;  // b_{k+2}
    double b_kp1 = 0.0;  // b_{k+1}
    double two_x = 2.0 * x;

    for (int k = n_coeffs - 1; k >= 1; --k) {
        double b_k = coeffs[k] + two_x * b_kp1 - b_kp2;
        b_kp2 = b_kp1;
        b_kp1 = b_k;
    }

    // Final step: f(x) = c_0 + x*b_1 - b_2
    return coeffs[0] + x * b_kp1 - b_kp2;
}

/// Evaluate a Chebyshev series and its derivative with respect to the un-normalized variable t.
/// @param x              normalized argument in [-1, 1]
/// @param half_interval  half-width of the t interval (dx/dt = 1/half_interval), in t's unit
/// @return {value, derivative}; the derivative is in coefficient units per t unit.
[[nodiscard]] inline std::pair<double, double> chebyshev_eval_with_derivative(
    const double* coeffs, int n_coeffs, double x, double half_interval)
{
    if (n_coeffs <= 0) return {0.0, 0.0};
    if (n_coeffs == 1) return {coeffs[0], 0.0};

    // Derivative coefficients c'_k = 2*(k+1)*c_{k+1} + c'_{k+2}: on the stack for the usual
    // short series, on the heap for longer ones
    double stack_buf[MAX_CHEBYSHEV_COEFFS] = {};
    std::vector<double> heap_buf;
    if (n_coeffs > MAX_CHEBYSHEV_COEFFS) heap_buf.assign(static_cast<std::size_t>(n_coeffs), 0.0);
    double* deriv_coeffs = heap_buf.empty() ? stack_buf : heap_buf.data();
    for (int k = n_coeffs - 2; k >= 0; --k) {
        deriv_coeffs[k] = 2.0 * (k + 1) * coeffs[k + 1];
        if (k + 2 < n_coeffs) {
            deriv_coeffs[k] += deriv_coeffs[k + 2];
        }
    }
    deriv_coeffs[0] *= 0.5;  // c'_0 halved per primed-sum convention (NR 5.8, eq. 5.8.11)

    // Evaluate both using Clenshaw
    double value = chebyshev_eval(coeffs, n_coeffs, x);
    double deriv = chebyshev_eval(deriv_coeffs, n_coeffs - 1, x);

    // Scale derivative by 1/half_interval (chain rule)
    deriv /= half_interval;

    return {value, deriv};
}

/// Derivative only: d/dt[sum(coeffs[i] * T_i(x))], t the un-normalized variable; see
/// `chebyshev_eval_with_derivative`.
[[nodiscard]] inline double chebyshev_deriv(
    const double* coeffs, int n_coeffs, double x, double half_interval)
{
    return chebyshev_eval_with_derivative(coeffs, n_coeffs, x, half_interval).second;
}

/// Position and velocity of a body from its Chebyshev coefficients, stored one component after
/// another: [x0, x1, ..., xN, y0, y1, ..., yN, z0, z1, ..., zN] (n_coeffs_per_component each).
/// Returns {position, velocity}: position in the coefficients' unit, velocity in that unit per
/// unit of `half_interval` (e.g. km/day for a JPL file with intervals in days).
[[nodiscard]] inline std::pair<Vector3d, Vector3d> chebyshev_eval_3d(
    const double* coeffs, int n_coeffs_per_component,
    double x, double half_interval)
{
    Vector3d pos, vel;
    for (int i = 0; i < 3; ++i) {
        const double* comp_coeffs = coeffs + i * n_coeffs_per_component;
        auto [p, v] = chebyshev_eval_with_derivative(
            comp_coeffs, n_coeffs_per_component, x, half_interval);
        pos[i] = p;
        vel[i] = v;
    }
    return {pos, vel};
}

} // namespace math
} // namespace ballistic
