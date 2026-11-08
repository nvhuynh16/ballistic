#pragma once

// =============================================================================
// Brent's method — 1D scalar minimization with parabolic interpolation
// =============================================================================
//
// Standard Brent method (Numerical Recipes 3rd ed. §10.3): finds the minimum of a function of
// one variable without derivatives. It combines golden-section search (shrinking the bracket
// by a fixed ratio, always safe) with parabolic interpolation (fitting a parabola through
// three points and jumping to its vertex, fast when the function is smooth).
//
// Caller supplies a U-shape bracket (ax, bx, cx) with ax < bx < cx and
// f(bx) < f(ax), f(bx) < f(cx). Method then narrows the bracket to within
// tol of the minimum. Typical convergence: 5–10 evals.
//
// Used by the line search of the B-spline empirical-acceleration estimator
// (calibration/detail/line_search.hpp) to choose the step length. Finding the bracket is the
// caller's job; this file is the inner Brent loop only.
//
// =============================================================================

#include <cmath>
#include <cstddef>
#include <limits>

namespace ballistic {
namespace math {

/// Result of `brent_minimize`.
struct BrentResult {
    double x_min;      // minimizer
    double f_min;      // f(x_min)
    int n_evals;       // function evaluations performed
    bool converged;    // true if tolerance reached before max_iter
};

/// Minimize f(x) (f: double -> double) inside a bracket using Brent's method. Never throws
/// (exceptions from f pass through).
///
/// Pre: ax < bx < cx, f(bx) <= f(ax), f(bx) <= f(cx). (U-shape bracket; ax and cx may be given
/// in either order.)
/// Post: result.x_min ∈ [ax, cx]; result.f_min is the lowest f seen.
///
/// tol: absolute tolerance in x (default 1e-3). A tiny relative term is added:
/// t1 = 1e-3·sqrt(machine epsilon)·|x| + tol/2 (≈ 1.5e-11·|x| + tol/2), and the search stops
/// when |x - mid| <= 2·t1 - (b - a)/2, i.e. when the bracket around x is within about 2·t1.
/// max_iter: iteration cap (default 100); `converged` is false when it was reached.
template <typename F>
[[nodiscard]] inline BrentResult brent_minimize(
    F&& f,
    double ax, double bx, double cx,
    double tol = 1e-3,
    int max_iter = 100)
{
    constexpr double GOLD = 0.3819660112501051;  // (3 - sqrt(5)) / 2
    constexpr double TINY = 1.0e-20;
    const double ZEPS = std::sqrt(std::numeric_limits<double>::epsilon()) * 1.0e-3;

    // a, b: current bracket; ensure a < b
    double a = (ax < cx) ? ax : cx;
    double b = (ax < cx) ? cx : ax;

    double x = bx;       // best so far
    double w = bx;       // second-best
    double v = bx;       // third-best
    double fx = f(x);
    double fw = fx;
    double fv = fx;
    int n_evals = 1;

    double d = 0.0;      // step taken last iter
    double e = 0.0;      // step taken iter before last (for parabolic guard)

    for (int iter = 0; iter < max_iter; ++iter) {
        const double xm = 0.5 * (a + b);
        const double tol1 = ZEPS * std::abs(x) + 0.5 * tol;
        const double tol2 = 2.0 * tol1;

        // Convergence test
        if (std::abs(x - xm) <= (tol2 - 0.5 * (b - a))) {
            return {x, fx, n_evals, true};
        }

        bool use_golden = true;

        if (std::abs(e) > tol1) {
            // Try parabolic interpolation through (v, fv), (w, fw), (x, fx)
            const double r = (x - w) * (fx - fv);
            const double q_raw = (x - v) * (fx - fw);
            double p = (x - v) * q_raw - (x - w) * r;
            double q = 2.0 * (q_raw - r);
            if (q > 0.0) p = -p;
            q = std::abs(q);
            const double e_prev = e;
            e = d;

            // Accept parabolic step if (a) it lies inside (a, b) and
            // (b) it's less than half the step before last.
            if (std::abs(p) < std::abs(0.5 * q * e_prev)
                && p > q * (a - x)
                && p < q * (b - x))
            {
                d = p / q;
                double u_new = x + d;
                // Don't evaluate too close to bracket endpoints
                if ((u_new - a) < tol2 || (b - u_new) < tol2) {
                    d = (xm - x >= 0.0 ? +1.0 : -1.0) * tol1;
                }
                use_golden = false;
            }
        }

        if (use_golden) {
            e = (x >= xm) ? (a - x) : (b - x);
            d = GOLD * e;
        }

        // Don't step less than tol1 from current best
        const double u = (std::abs(d) >= tol1)
            ? (x + d)
            : (x + ((d >= 0.0) ? +tol1 : -tol1));

        const double fu = f(u);
        ++n_evals;

        if (fu <= fx) {
            // u is the new best
            if (u >= x) a = x; else b = x;
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u; fv = fu;
            }
        }

        (void)TINY;  // reserved for parabolic-step guard — currently unused
    }

    return {x, fx, n_evals, false};
}

} // namespace math
} // namespace ballistic
