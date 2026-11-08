#pragma once

// =============================================================================
// B-Spline Basis Functions and Derivatives
// =============================================================================
//
// A B-spline of degree p is a smooth piecewise polynomial written as a weighted sum of basis
// functions N_i(u). The pieces join at "knots" (the non-decreasing knot vector V); between two
// neighboring knots (a "knot span") only p+1 basis functions are non-zero. This file
// implements the standard algorithms of Piegl & Tiller, "The NURBS Book" (2nd ed., 1997):
//   - FindSpan (A2.1): knot span binary search
//   - BasisFuns (A2.2): all p+1 non-zero basis functions at a point
//   - DersBasisFuns (A2.3): basis function derivatives up to order d
//   - Clamped uniform knot vector construction
//
// No argument is validated except in make_clamped_knot_vector; out-of-range indices are
// undefined behavior.
//
// =============================================================================

#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

#include "../core/errors.hpp"

namespace ballistic {
namespace math {

/// Knot span containing u: the largest i with V[i] <= u < V[i+1] (NURBS Book A2.1).
/// @param n  number of basis functions - 1 (index of the last basis function)
/// @param p  degree
/// @param u  parameter; values at or beyond either end are clamped to the first/last span
/// @param V  knot vector of n + p + 2 entries
/// @return span index in [p, n]
inline int bspline_findspan(int n, int p, double u, const double* V) {
    // Clamp to right boundary
    if (u >= V[n + 1]) return n;
    if (u <= V[p]) return p;

    int low = p;
    int high = n + 1;
    int mid = (low + high) / 2;

    while (u < V[mid] || u >= V[mid + 1]) {
        if (u < V[mid])
            high = mid;
        else
            low = mid;
        mid = (low + high) / 2;
    }
    return mid;
}

/// All p+1 non-zero basis functions at u (NURBS Book A2.2).
/// @param span  result of bspline_findspan
/// @param N     output array of size >= p+1; receives N_{span-p,p}(u) .. N_{span,p}(u)
inline void bspline_basis_funs(int p, int span, double u, const double* V,
                               double* N) {
    // Workspace for left/right differences
    std::vector<double> left(p + 1);
    std::vector<double> right(p + 1);

    N[0] = 1.0;
    for (int j = 1; j <= p; ++j) {
        left[j]  = u - V[span + 1 - j];
        right[j] = V[span + j] - u;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            double temp = N[r] / (right[r + 1] + left[j - r]);
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N[j] = saved;
    }
}

/// Non-zero basis functions and their derivatives with respect to u up to order d (NURBS Book
/// A2.3).
/// @param ders  output, row-major: ders[k * (p+1) + j] = N^(k)_{span-p+j,p}(u), the k-th
///              derivative, for k = 0..d and j = 0..p; size >= (d+1) * (p+1)
inline void bspline_ders(int p, int span, double u, const double* V,
                         int d, double* ders) {
    const int stride = p + 1;

    // ndu[j * stride + r] stores basis function values and knot differences
    std::vector<double> ndu((p + 1) * (p + 1));
    std::vector<double> a(2 * (p + 1));  // two rows for alternating
    std::vector<double> left(p + 1);
    std::vector<double> right(p + 1);

    ndu[0] = 1.0;
    for (int j = 1; j <= p; ++j) {
        left[j]  = u - V[span + 1 - j];
        right[j] = V[span + j] - u;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            // Lower triangle: knot differences
            ndu[j * stride + r] = right[r + 1] + left[j - r];
            double temp = ndu[r * stride + j - 1] / ndu[j * stride + r];
            // Upper triangle: basis functions
            ndu[r * stride + j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j * stride + j] = saved;
    }

    // Load basis function values (derivative order 0)
    for (int j = 0; j <= p; ++j) {
        ders[0 * stride + j] = ndu[j * stride + p];
    }

    // Compute derivatives (NURBS Book Eq. 2.9)
    for (int r = 0; r <= p; ++r) {
        int s1 = 0, s2 = stride;  // alternating rows in a[]
        a[0] = 1.0;

        for (int k = 1; k <= d; ++k) {
            double dd = 0.0;
            int rk = r - k;
            int pk = p - k;

            if (r >= k) {
                a[s2 + 0] = a[s1 + 0] / ndu[(pk + 1) * stride + rk];
                dd = a[s2 + 0] * ndu[rk * stride + pk];
            }

            int j1 = (rk >= -1) ? 1 : -rk;
            int j2 = (r - 1 <= pk) ? k - 1 : p - r;

            for (int j = j1; j <= j2; ++j) {
                a[s2 + j] = (a[s1 + j] - a[s1 + j - 1]) /
                             ndu[(pk + 1) * stride + rk + j];
                dd += a[s2 + j] * ndu[(rk + j) * stride + pk];
            }

            if (r <= pk) {
                a[s2 + k] = -a[s1 + k - 1] / ndu[(pk + 1) * stride + r];
                dd += a[s2 + k] * ndu[r * stride + pk];
            }

            ders[k * stride + r] = dd;
            std::swap(s1, s2);
        }
    }

    // Multiply by correct factors: N^(k) *= p! / (p-k)!
    double factor = static_cast<double>(p);
    for (int k = 1; k <= d; ++k) {
        for (int j = 0; j <= p; ++j) {
            ders[k * stride + j] *= factor;
        }
        factor *= static_cast<double>(p - k);
    }
}

/// Clamped uniform knot vector on [t_start, t_end]: degree+1 knots repeated at each end (so
/// the spline starts and ends exactly at its first and last coefficient) and n_interior knots
/// evenly spaced in between.
/// @param degree      polynomial degree (e.g., 3 for cubic)
/// @param n_interior  number of interior knots
/// @return knot vector of size (n_interior + 2*degree + 2); the spline then has
///         n_interior + degree + 1 basis functions
/// @throws ConfigError when t_end <= t_start or n_interior < 0
inline std::vector<double> make_clamped_knot_vector(
    double t_start, double t_end, int n_interior, int degree)
{
    if (t_end <= t_start) {
        throw ConfigError("make_clamped_knot_vector: t_end (" + std::to_string(t_end) +
                          ") must be after t_start (" + std::to_string(t_start) + ")");
    }
    if (n_interior < 0) {
        throw ConfigError("make_clamped_knot_vector: n_interior is " + std::to_string(n_interior) +
                          "; the number of interior knots cannot be negative");
    }

    int n_knots = n_interior + 2 * (degree + 1);
    std::vector<double> knots(n_knots);

    // Clamped: first (degree+1) knots at t_start
    for (int i = 0; i <= degree; ++i) {
        knots[i] = t_start;
    }
    // Clamped: last (degree+1) knots at t_end
    for (int i = 0; i <= degree; ++i) {
        knots[n_knots - 1 - i] = t_end;
    }
    // Interior knots uniformly spaced
    double dt = (t_end - t_start) / (n_interior + 1);
    for (int i = 0; i < n_interior; ++i) {
        knots[degree + 1 + i] = t_start + (i + 1) * dt;
    }

    return knots;
}

} // namespace math
} // namespace ballistic
