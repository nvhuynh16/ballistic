#pragma once

// =============================================================================
// Line-search step grids and dispatchers for IRGN / GN
// =============================================================================
//
// Step-length choices for the two least-squares loops of the empirical-acceleration (EA)
// estimators: the IRGN loop of bspline_accel_estimator.hpp (iteratively regularized
// Gauss-Newton: a Gauss-Newton fit whose regularization weight shrinks each iteration) and
// the Levenberg-Marquardt (LM, damped Gauss-Newton) backtracking of gn_refine.hpp.
//
// A line search tries several multiples α of the computed update step and keeps the best.
// The IRGN searches take a callback `try_alpha(a) -> sse` that applies the step scaled by
// `a`, scores it (SSE = sum of squared errors) and remembers the best candidate itself
// (see `estimate_bspline_impl`). Nothing is returned because EA candidates are expensive to
// build and the callback already holds the winner.
// =============================================================================

#include "../../math/brent_minimize.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace ballistic {
namespace calibration {
namespace detail {

/// Step multiples tried by the brute-force IRGN line search (24 values, largest first). Chosen
/// empirically; large steps come first because they usually win.
inline constexpr std::array<double, 24> BRUTE_LS_ALPHAS = {
    3.0,  2.5,  2.0,   1.5,  1.25, 1.0,  0.85, 0.75,
    0.6,  0.5,  0.4,   0.35, 0.25, 0.2,  0.15, 0.1,
    0.07, 0.05, 0.03,  0.02, 0.01, 0.005, 0.002, 0.001,
};

/// Step multiples first tried by the Brent IRGN line search (5 values, ascending). If the best
/// of them has a worse neighbor on each side, Brent's method (a bracketing 1-D minimizer)
/// then searches between those two neighbors.
inline constexpr std::array<double, 5> BRENT_LS_ANCHORS = {
    0.001, 0.01, 0.1, 1.0, 3.0,
};

/// Step multiples tried, largest first, by the LM backtracking in GN refinement.
inline constexpr std::array<double, 9> GN_BACKTRACK_STEPS = {
    1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.002, 0.001,
};

/// Brute-force IRGN line search: calls `try_alpha(a)` for every value in `BRUTE_LS_ALPHAS`.
/// The callback keeps the lowest-SSE candidate.
template<class F>
inline void brute_alpha_search(F&& try_alpha) {
    for (double a : BRUTE_LS_ALPHAS) (void)try_alpha(a);
}

/// Brent IRGN line search: calls `try_alpha` at the five `BRENT_LS_ANCHORS`; if the best one is
/// interior and lower than both neighbors, refines it with Brent's method (tolerance 1e-3,
/// at most 50 evaluations). Otherwise the best anchor stands. The callback keeps the winner.
template<class F>
inline void brent_alpha_search(F&& try_alpha) {
    std::array<double, BRENT_LS_ANCHORS.size()> sse{};
    for (std::size_t i = 0; i < BRENT_LS_ANCHORS.size(); ++i) {
        sse[i] = try_alpha(BRENT_LS_ANCHORS[i]);
    }

    std::size_t imin = 0;
    for (std::size_t i = 1; i < BRENT_LS_ANCHORS.size(); ++i) {
        if (sse[i] < sse[imin]) imin = i;
    }

    if (imin > 0 && imin + 1 < BRENT_LS_ANCHORS.size()
        && sse[imin] < sse[imin - 1]
        && sse[imin] < sse[imin + 1])
    {
        // Only try_alpha's side effect (recording the best trial) matters; the
        // returned minimum is unused.
        (void)math::brent_minimize(try_alpha,
            BRENT_LS_ANCHORS[imin - 1], BRENT_LS_ANCHORS[imin],
            BRENT_LS_ANCHORS[imin + 1], 1e-3, 50);
    }
}

}  // namespace detail
}  // namespace calibration
}  // namespace ballistic
