#pragma once

// =============================================================================
// Natural Cubic Spline Interpolation
// =============================================================================
//
// Template cubic spline for double or Eigen::Vector3d values: a piecewise cubic through every
// sample whose value, slope and curvature are continuous (C^2). "Natural" boundary conditions:
// second derivative = 0 at both ends.
//
// For n=1: constant.  For n=2: linear.  For n>=3: cubic spline.
// Queries outside the domain clamp to boundary values (no error, no warning; check domain()
// first if that matters). Sample positions x must be strictly increasing (not checked).
//
// For Vector3d: three independent scalar splines on components.
//
// =============================================================================

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <cassert>

#include "../core/errors.hpp"

namespace ballistic {
namespace math {

// ============================================================================
// Scalar cubic spline (primary implementation)
// ============================================================================

/// Natural cubic spline through scalar samples (x_i, y_i).
class CubicSplineScalar {
public:
    /// Empty spline (evaluates to 0).
    CubicSplineScalar() = default;

    /// Spline through (x[i], y[i]); x strictly increasing.
    /// @throws InputPlausibilityError when x and y differ in length
    CubicSplineScalar(std::vector<double> x, std::vector<double> y)
        : x_(std::move(x)), y_(std::move(y))
    {
        if (x_.size() != y_.size()) {
            throw InputPlausibilityError("CubicSpline: " + std::to_string(x_.size()) + " x values but " +
                                         std::to_string(y_.size()) + " y values; pass one y per x");
        }
        if (x_.size() >= 3) {
            compute_moments();
        }
    }

    /// Same as `evaluate(x)`.
    [[nodiscard]] double operator()(double x) const {
        return evaluate(x);
    }

    /// Spline value at x; outside [x_front, x_back] the end value; 0 when empty.
    [[nodiscard]] double evaluate(double x) const {
        const size_t n = x_.size();
        if (n == 0) return 0.0;
        if (n == 1) return y_[0];

        // Clamp to domain
        if (x <= x_[0]) return y_[0];
        if (x >= x_[n - 1]) return y_[n - 1];

        if (n == 2) {
            // Linear interpolation
            double t = (x - x_[0]) / (x_[1] - x_[0]);
            return y_[0] + t * (y_[1] - y_[0]);
        }

        // Find interval [x_[i], x_[i+1]] containing x
        size_t i = find_interval(x);
        return eval_segment(i, x);
    }

    /// Slope dy/dx at x; outside the domain, the slope at the nearer end; 0 with fewer than
    /// two samples.
    [[nodiscard]] double derivative(double x) const {
        const size_t n = x_.size();
        if (n <= 1) return 0.0;

        // Clamp to domain for derivative
        if (x <= x_[0]) x = x_[0];
        if (x >= x_[n - 1]) x = x_[n - 1];

        if (n == 2) {
            return (y_[1] - y_[0]) / (x_[1] - x_[0]);
        }

        size_t i = find_interval(x);
        return eval_derivative(i, x);
    }

    /// True when there are no samples.
    [[nodiscard]] bool empty() const noexcept { return x_.empty(); }
    /// Number of samples.
    [[nodiscard]] size_t size() const noexcept { return x_.size(); }

    /// {first x, last x}; {0, 0} when empty.
    [[nodiscard]] std::pair<double, double> domain() const noexcept {
        if (x_.empty()) return {0.0, 0.0};
        return {x_.front(), x_.back()};
    }

private:
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<double> m_;  // Second derivatives (moments)

    // Solve for the second derivatives ("moments") at the samples. Continuity of the slope
    // gives a tridiagonal linear system, solved with the Thomas algorithm (Gaussian elimination
    // specialized to three diagonals). Natural boundary conditions: m[0] = m[n-1] = 0.
    void compute_moments() {
        const size_t n = x_.size();
        assert(n >= 3);

        // Interior points: n-2 unknowns (m[1]..m[n-2])
        // Natural BC: m[0] = 0, m[n-1] = 0
        const size_t nm = n - 2;

        // Precompute h[i] = x[i+1] - x[i] and divided differences
        std::vector<double> h(n - 1);
        std::vector<double> d(n - 1);  // divided differences
        for (size_t i = 0; i < n - 1; ++i) {
            h[i] = x_[i + 1] - x_[i];
            d[i] = (y_[i + 1] - y_[i]) / h[i];
        }

        // Build tridiagonal system for interior moments
        // a[i] m[i-1] + b[i] m[i] + c[i] m[i+1] = rhs[i]
        // where indices are shifted: system index j corresponds to m[j+1]
        std::vector<double> diag(nm);    // main diagonal
        std::vector<double> upper(nm);   // upper diagonal
        std::vector<double> rhs(nm);

        for (size_t j = 0; j < nm; ++j) {
            size_t i = j + 1;  // actual knot index
            diag[j] = 2.0 * (h[i - 1] + h[i]);
            upper[j] = h[i];
            rhs[j] = 6.0 * (d[i] - d[i - 1]);
        }

        // Forward elimination (Thomas algorithm)
        for (size_t j = 1; j < nm; ++j) {
            double lower = h[j];  // sub-diagonal element
            double factor = lower / diag[j - 1];
            diag[j] -= factor * upper[j - 1];
            rhs[j] -= factor * rhs[j - 1];
        }

        // Back substitution
        m_.assign(n, 0.0);  // m[0] = m[n-1] = 0 (natural BC)
        m_[nm] = rhs[nm - 1] / diag[nm - 1];
        for (size_t j = nm - 1; j > 0; --j) {
            m_[j] = (rhs[j - 1] - upper[j - 1] * m_[j + 1]) / diag[j - 1];
        }
    }

    // Binary search for interval containing x
    [[nodiscard]] size_t find_interval(double x) const {
        auto it = std::upper_bound(x_.begin(), x_.end(), x);
        size_t i = static_cast<size_t>(it - x_.begin());
        if (i == 0) return 0;
        if (i >= x_.size()) return x_.size() - 2;
        return i - 1;
    }

    // Evaluate spline on segment [x_[i], x_[i+1]]
    [[nodiscard]] double eval_segment(size_t i, double x) const {
        double h = x_[i + 1] - x_[i];
        double t = x - x_[i];
        double u = x_[i + 1] - x;  // h - t

        // Standard cubic spline formula using moments m_
        return (m_[i] * u * u * u + m_[i + 1] * t * t * t) / (6.0 * h)
             + (y_[i] / h - m_[i] * h / 6.0) * u
             + (y_[i + 1] / h - m_[i + 1] * h / 6.0) * t;
    }

    // Evaluate derivative on segment
    [[nodiscard]] double eval_derivative(size_t i, double x) const {
        double h = x_[i + 1] - x_[i];
        double t = x - x_[i];
        double u = x_[i + 1] - x;

        return (-m_[i] * u * u + m_[i + 1] * t * t) / (2.0 * h)
             + (y_[i + 1] - y_[i]) / h
             - (m_[i + 1] - m_[i]) * h / 6.0;
    }
};

// ============================================================================
// CubicSpline<T> template: double delegates to CubicSplineScalar,
// Vector3d uses 3 independent scalar splines
// ============================================================================

template<typename T>
class CubicSpline;

/// Scalar spline: identical to `CubicSplineScalar`.
template<>
class CubicSpline<double> : public CubicSplineScalar {
public:
    using CubicSplineScalar::CubicSplineScalar;
};

/// 3D spline: three independent scalar splines, one per component. Same clamping as the
/// scalar spline.
template<>
class CubicSpline<Eigen::Vector3d> {
public:
    CubicSpline() = default;

    /// Spline through (x[i], y[i]); x strictly increasing.
    /// @throws InputPlausibilityError when x and y differ in length
    CubicSpline(std::vector<double> x, std::vector<Eigen::Vector3d> y)
    {
        if (x.size() != y.size()) {
            throw InputPlausibilityError("CubicSpline<Vector3d>: " + std::to_string(x.size()) +
                                         " x values but " + std::to_string(y.size()) +
                                         " y values; pass one y per x");
        }
        // Decompose into 3 scalar splines
        const size_t n = y.size();
        std::vector<double> yx(n), yy(n), yz(n);
        for (size_t i = 0; i < n; ++i) {
            yx[i] = y[i].x();
            yy[i] = y[i].y();
            yz[i] = y[i].z();
        }
        spline_x_ = CubicSplineScalar(x, std::move(yx));
        spline_y_ = CubicSplineScalar(x, std::move(yy));
        spline_z_ = CubicSplineScalar(std::move(x), std::move(yz));
    }

    [[nodiscard]] Eigen::Vector3d operator()(double x) const {
        return evaluate(x);
    }

    /// Value at x (clamped outside the domain).
    [[nodiscard]] Eigen::Vector3d evaluate(double x) const {
        return Eigen::Vector3d(
            spline_x_.evaluate(x),
            spline_y_.evaluate(x),
            spline_z_.evaluate(x)
        );
    }

    /// Componentwise slope at x (end slope outside the domain).
    [[nodiscard]] Eigen::Vector3d derivative(double x) const {
        return Eigen::Vector3d(
            spline_x_.derivative(x),
            spline_y_.derivative(x),
            spline_z_.derivative(x)
        );
    }

    [[nodiscard]] bool empty() const noexcept { return spline_x_.empty(); }
    [[nodiscard]] size_t size() const noexcept { return spline_x_.size(); }

    [[nodiscard]] std::pair<double, double> domain() const noexcept {
        return spline_x_.domain();
    }

private:
    CubicSplineScalar spline_x_, spline_y_, spline_z_;
};

} // namespace math
} // namespace ballistic
