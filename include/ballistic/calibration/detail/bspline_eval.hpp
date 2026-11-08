#pragma once

#include "../../math/bspline_basis.hpp"
#include <Eigen/Core>
#include <vector>
#include <algorithm>
#include <optional>
#include <utility>
#include <cassert>

namespace ballistic {
namespace calibration {
namespace detail {

constexpr int BSPLINE_DEFAULT_DEGREE = 3;  // default of `BSplineFitResult::degree`; the estimator itself defaults to 5 (EABSplineConfig::degree)
constexpr int BSPLINE_MAX_DEGREE = 7;  // highest supported degree (sizes the fixed stack buffers)

/// A fitted 3-D B-spline (one coefficient vector per x/y/z component): a smooth curve made of
/// polynomial pieces of `degree` joined at the knots. Times are seconds relative to the
/// estimator's start epoch.
struct BSplineFitResult {
    Eigen::VectorXd cx, cy, cz;          // Spline coefficients per component
    std::vector<double> knots;            // Knot vector
    int n_basis;                          // Number of basis functions
    double t_start;                       // Domain start (relative time)
    double t_end;                         // Domain end (relative time)
    int degree = BSPLINE_DEFAULT_DEGREE;          // B-spline degree (3=cubic, 5=quintic)
    std::optional<double> lambda_used;            // smoothing weight λ actually used; set by fit_bspline (useful when chosen automatically by GCV)
};

/// Value of a 3-D B-spline at time `t` [s] from its raw coefficients, knot vector, number of
/// basis functions and degree. `t` outside the knot range is clamped to the nearest end.
[[nodiscard]] inline Vector3d eval_bspline_value(
    const Eigen::VectorXd& cx, const Eigen::VectorXd& cy, const Eigen::VectorXd& cz,
    const std::vector<double>& knots, int n_basis, int degree, double t)
{
    assert(degree <= BSPLINE_MAX_DEGREE);
    int n_idx = n_basis - 1;
    double tc = std::clamp(t, knots.front(), knots.back());
    int span = math::bspline_findspan(n_idx, degree, tc, knots.data());
    double Ni[BSPLINE_MAX_DEGREE + 1];
    math::bspline_basis_funs(degree, span, tc, knots.data(), Ni);
    int col0 = span - degree;

    Vector3d val = Vector3d::Zero();
    for (int j = 0; j <= degree; ++j) {
        int c = col0 + j;
        val.x() += cx(c) * Ni[j];
        val.y() += cy(c) * Ni[j];
        val.z() += cz(c) * Ni[j];
    }
    return val;
}

/// Second and third time derivatives of the fitted spline at `t` [s] (clamped to the fit's
/// domain). Requires degree ≥ 3. If the spline models position, these are acceleration and
/// jerk.
[[nodiscard]] inline std::pair<Vector3d, Vector3d> eval_d2d3(const BSplineFitResult& fit, double t) {
    const int deg = fit.degree;
    assert(deg >= 3);  // Need at least cubic for 3rd derivative
    double tc = std::clamp(t, fit.t_start, fit.t_end);
    int n_idx = fit.n_basis - 1;
    int span = math::bspline_findspan(n_idx, deg, tc, fit.knots.data());
    double ders[(BSPLINE_MAX_DEGREE + 1) * (BSPLINE_MAX_DEGREE + 1)];
    math::bspline_ders(deg, span, tc, fit.knots.data(), 3, ders);

    const int stride = deg + 1;
    const double* d2row = &ders[2 * stride];
    const double* d3row = &ders[3 * stride];
    int col0 = span - deg;

    Vector3d d2 = Vector3d::Zero(), d3 = Vector3d::Zero();
    for (int j = 0; j <= deg; ++j) {
        int c = col0 + j;
        d2.x() += fit.cx(c) * d2row[j];
        d2.y() += fit.cy(c) * d2row[j];
        d2.z() += fit.cz(c) * d2row[j];
        d3.x() += fit.cx(c) * d3row[j];
        d3.y() += fit.cy(c) * d3row[j];
        d3.z() += fit.cz(c) * d3row[j];
    }
    return {d2, d3};
}

/// Second time derivative of the fitted spline at `t` [s] (clamped to the domain).
[[nodiscard]] inline Vector3d eval_d2(const BSplineFitResult& fit, double t) {
    return eval_d2d3(fit, t).first;
}

/// First time derivative of the fitted spline at `t` [s] (clamped to the domain).
[[nodiscard]] inline Vector3d eval_d1(const BSplineFitResult& fit, double t) {
    const int deg = fit.degree;
    double tc = std::clamp(t, fit.t_start, fit.t_end);
    int n_idx = fit.n_basis - 1;
    int span = math::bspline_findspan(n_idx, deg, tc, fit.knots.data());
    double ders[(BSPLINE_MAX_DEGREE + 1) * 2];
    math::bspline_ders(deg, span, tc, fit.knots.data(), 1, ders);
    const double* d1row = &ders[1 * (deg + 1)];
    int col0 = span - deg;
    Vector3d d1 = Vector3d::Zero();
    for (int j = 0; j <= deg; ++j) {
        int c = col0 + j;
        d1.x() += fit.cx(c) * d1row[j];
        d1.y() += fit.cy(c) * d1row[j];
        d1.z() += fit.cz(c) * d1row[j];
    }
    return d1;
}

/// Value of the fitted spline at `t` [s] (clamped to the knot range).
[[nodiscard]] inline Vector3d eval_d0(const BSplineFitResult& fit, double t) {
    return eval_bspline_value(fit.cx, fit.cy, fit.cz, fit.knots, fit.n_basis, fit.degree, t);
}

} // namespace detail
} // namespace calibration
} // namespace ballistic
