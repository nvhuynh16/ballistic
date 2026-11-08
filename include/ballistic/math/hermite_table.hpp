#pragma once

// =============================================================================
// Hermite Table Interpolation
// =============================================================================
//
// General-purpose table interpolator:
//   - With derivatives: cubic Hermite (matches value and slope at every sample; value and
//     slope are continuous, "C^1")
//   - Without derivatives: delegates to CubicSpline (value, slope and curvature continuous, C^2)
//   - Vector3d only: piecewise linear (`make_linear`; only the value is continuous, C^0)
//
// Template on T: double or Eigen::Vector3d. Sample positions must be strictly increasing (not
// checked). Queries outside the table return the end value (clamped, no error or warning).
//
// =============================================================================

#include "cubic_spline.hpp"
#include "../core/errors.hpp"
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <variant>
#include <string>
#include <optional>

namespace ballistic {
namespace math {

// ============================================================================
// Scalar Hermite interpolation (with derivatives)
// ============================================================================

/// Piecewise cubic Hermite interpolation of scalar samples y(x) with given slopes dy/dx.
class HermiteScalar {
public:
    /// Empty table (evaluates to 0).
    HermiteScalar() = default;

    /// Table of values y and slopes dy at positions x (strictly increasing).
    /// @throws InputPlausibilityError when x, y and dy differ in length
    HermiteScalar(std::vector<double> x, std::vector<double> y, std::vector<double> dy)
        : x_(std::move(x)), y_(std::move(y)), dy_(std::move(dy))
    {
        if (x_.size() != y_.size() || x_.size() != dy_.size()) {
            throw InputPlausibilityError("HermiteScalar: x, y and dy differ in length (" +
                std::to_string(x_.size()) + ", " + std::to_string(y_.size()) + ", " +
                std::to_string(dy_.size()) + "); pass one value and one derivative per x");
        }
    }

    /// Interpolated value at x; outside the table the end value; 0 when empty.
    [[nodiscard]] double evaluate(double x) const {
        const size_t n = x_.size();
        if (n == 0) return 0.0;
        if (n == 1) return y_[0];

        // Clamp to domain
        if (x <= x_[0]) return y_[0];
        if (x >= x_[n - 1]) return y_[n - 1];

        size_t i = find_interval(x);
        return eval_hermite(i, x);
    }

    [[nodiscard]] double operator()(double x) const { return evaluate(x); }

    [[nodiscard]] bool empty() const noexcept { return x_.empty(); }
    [[nodiscard]] size_t size() const noexcept { return x_.size(); }

private:
    std::vector<double> x_, y_, dy_;

    [[nodiscard]] size_t find_interval(double x) const {
        auto it = std::upper_bound(x_.begin(), x_.end(), x);
        size_t i = static_cast<size_t>(it - x_.begin());
        if (i == 0) return 0;
        if (i >= x_.size()) return x_.size() - 2;
        return i - 1;
    }

    // Cubic Hermite on [x_[i], x_[i+1]]
    [[nodiscard]] double eval_hermite(size_t i, double x) const {
        double h = x_[i + 1] - x_[i];
        double t = (x - x_[i]) / h;
        double t2 = t * t;
        double t3 = t2 * t;

        // Hermite basis functions
        double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        double h10 = t3 - 2.0 * t2 + t;
        double h01 = -2.0 * t3 + 3.0 * t2;
        double h11 = t3 - t2;

        return h00 * y_[i] + h10 * h * dy_[i]
             + h01 * y_[i + 1] + h11 * h * dy_[i + 1];
    }
};

// ============================================================================
// HermiteTable<T> template
// ============================================================================

template<typename T>
class HermiteTable;

/// Scalar table: cubic Hermite when slopes are given, otherwise a natural cubic spline.
template<>
class HermiteTable<double> {
public:
    HermiteTable() = default;

    // Without derivatives: delegates to CubicSpline
    HermiteTable(std::vector<double> x, std::vector<double> y)
        : impl_(CubicSpline<double>(std::move(x), std::move(y))), has_derivs_(false) {}

    // With derivatives: cubic Hermite
    HermiteTable(std::vector<double> x, std::vector<double> y, std::vector<double> dy)
        : impl_(HermiteScalar(std::move(x), std::move(y), std::move(dy))), has_derivs_(true) {}

    [[nodiscard]] double operator()(double x) const { return evaluate(x); }

    [[nodiscard]] double evaluate(double x) const {
        if (has_derivs_) {
            return std::get<HermiteScalar>(impl_).evaluate(x);
        }
        return std::get<CubicSpline<double>>(impl_).evaluate(x);
    }

    [[nodiscard]] bool has_derivatives() const noexcept { return has_derivs_; }

    [[nodiscard]] bool empty() const noexcept {
        if (has_derivs_) return std::get<HermiteScalar>(impl_).empty();
        return std::get<CubicSpline<double>>(impl_).empty();
    }

    [[nodiscard]] size_t size() const noexcept {
        if (has_derivs_) return std::get<HermiteScalar>(impl_).size();
        return std::get<CubicSpline<double>>(impl_).size();
    }

private:
    std::variant<CubicSpline<double>, HermiteScalar> impl_;
    bool has_derivs_ = false;
};

// ============================================================================
// Vector3d Hermite interpolation (with derivatives)
// ============================================================================

/// Cubic Hermite interpolation of 3D samples with given derivatives (one scalar table per
/// component).
class HermiteVector3d {
public:
    HermiteVector3d() = default;

    /// @throws InputPlausibilityError when x, y and dy differ in length
    HermiteVector3d(std::vector<double> x, std::vector<Eigen::Vector3d> y,
                    std::vector<Eigen::Vector3d> dy)
    {
        if (x.size() != y.size() || x.size() != dy.size()) {
            throw InputPlausibilityError("HermiteVector3d: x, y and dy differ in length (" +
                std::to_string(x.size()) + ", " + std::to_string(y.size()) + ", " +
                std::to_string(dy.size()) + "); pass one value and one derivative per x");
        }
        const size_t n = y.size();
        std::vector<double> yx(n), yy(n), yz(n), dyx(n), dyy(n), dyz(n);
        for (size_t i = 0; i < n; ++i) {
            yx[i] = y[i].x(); yy[i] = y[i].y(); yz[i] = y[i].z();
            dyx[i] = dy[i].x(); dyy[i] = dy[i].y(); dyz[i] = dy[i].z();
        }
        hx_ = HermiteScalar(x, std::move(yx), std::move(dyx));
        hy_ = HermiteScalar(x, std::move(yy), std::move(dyy));
        hz_ = HermiteScalar(std::move(x), std::move(yz), std::move(dyz));
    }

    [[nodiscard]] Eigen::Vector3d evaluate(double x) const {
        return Eigen::Vector3d(hx_.evaluate(x), hy_.evaluate(x), hz_.evaluate(x));
    }

    [[nodiscard]] Eigen::Vector3d operator()(double x) const { return evaluate(x); }
    [[nodiscard]] bool empty() const noexcept { return hx_.empty(); }
    [[nodiscard]] size_t size() const noexcept { return hx_.size(); }

private:
    HermiteScalar hx_, hy_, hz_;
};

// ============================================================================
// Linear interpolation for Vector3d (piecewise linear, C^0 continuous)
// ============================================================================

/// Piecewise linear interpolation of 3D samples; clamped to the end values outside the table.
class LinearVector3d {
public:
    LinearVector3d() = default;

    /// @throws InputPlausibilityError when x and y differ in length
    LinearVector3d(std::vector<double> x, std::vector<Eigen::Vector3d> y)
        : x_(std::move(x)), y_(std::move(y))
    {
        if (x_.size() != y_.size()) {
            throw InputPlausibilityError("LinearVector3d: " + std::to_string(x_.size()) +
                " x values but " + std::to_string(y_.size()) + " y values; pass one y per x");
        }
    }

    [[nodiscard]] Eigen::Vector3d evaluate(double x) const {
        const size_t n = x_.size();
        if (n == 0) return Eigen::Vector3d::Zero();
        if (n == 1) return y_[0];
        if (x <= x_[0]) return y_[0];
        if (x >= x_[n - 1]) return y_[n - 1];

        auto it = std::upper_bound(x_.begin(), x_.end(), x);
        size_t i = static_cast<size_t>(it - x_.begin());
        if (i == 0) i = 1;
        if (i >= n) i = n - 1;

        double t = (x - x_[i - 1]) / (x_[i] - x_[i - 1]);
        return (1.0 - t) * y_[i - 1] + t * y_[i];
    }

    [[nodiscard]] Eigen::Vector3d operator()(double x) const { return evaluate(x); }
    [[nodiscard]] bool empty() const noexcept { return x_.empty(); }
    [[nodiscard]] size_t size() const noexcept { return x_.size(); }

private:
    std::vector<double> x_;
    std::vector<Eigen::Vector3d> y_;
};

/// 3D table: cubic Hermite when derivatives are given, a natural cubic spline when not, or
/// piecewise linear via `make_linear`.
template<>
class HermiteTable<Eigen::Vector3d> {
public:
    HermiteTable() = default;

    // Without derivatives: delegates to CubicSpline<Vector3d>
    HermiteTable(std::vector<double> x, std::vector<Eigen::Vector3d> y)
        : impl_(CubicSpline<Eigen::Vector3d>(std::move(x), std::move(y))), mode_(CUBIC_SPLINE) {}

    // With derivatives: cubic Hermite
    HermiteTable(std::vector<double> x, std::vector<Eigen::Vector3d> y,
                 std::vector<Eigen::Vector3d> dy)
        : impl_(HermiteVector3d(std::move(x), std::move(y), std::move(dy))), mode_(HERMITE) {}

    // Piecewise-linear table. Used for empirical accelerations (EA) taken from the second
    // derivative of a B-spline, which are tabulated at points and joined linearly.
    static HermiteTable make_linear(std::vector<double> x, std::vector<Eigen::Vector3d> y) {
        HermiteTable t;
        t.impl_ = LinearVector3d(std::move(x), std::move(y));
        t.mode_ = LINEAR;
        return t;
    }

    [[nodiscard]] Eigen::Vector3d operator()(double x) const { return evaluate(x); }

    [[nodiscard]] Eigen::Vector3d evaluate(double x) const {
        switch (mode_) {
            case HERMITE: return std::get<HermiteVector3d>(impl_).evaluate(x);
            case LINEAR:  return std::get<LinearVector3d>(impl_).evaluate(x);
            default:      return std::get<CubicSpline<Eigen::Vector3d>>(impl_).evaluate(x);
        }
    }

    [[nodiscard]] bool has_derivatives() const noexcept { return mode_ == HERMITE; }

    [[nodiscard]] bool empty() const noexcept {
        switch (mode_) {
            case HERMITE: return std::get<HermiteVector3d>(impl_).empty();
            case LINEAR:  return std::get<LinearVector3d>(impl_).empty();
            default:      return std::get<CubicSpline<Eigen::Vector3d>>(impl_).empty();
        }
    }

    [[nodiscard]] size_t size() const noexcept {
        switch (mode_) {
            case HERMITE: return std::get<HermiteVector3d>(impl_).size();
            case LINEAR:  return std::get<LinearVector3d>(impl_).size();
            default:      return std::get<CubicSpline<Eigen::Vector3d>>(impl_).size();
        }
    }

private:
    enum Mode { CUBIC_SPLINE, HERMITE, LINEAR };
    std::variant<CubicSpline<Eigen::Vector3d>, HermiteVector3d, LinearVector3d> impl_;
    Mode mode_ = CUBIC_SPLINE;
};

} // namespace math
} // namespace ballistic
