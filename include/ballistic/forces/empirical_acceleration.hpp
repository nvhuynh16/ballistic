#pragma once

// =============================================================================
// Empirical Acceleration
// =============================================================================
//
// An empirical acceleration (EA) is a small correction acceleration, a function
// of time only, that is fitted to observations and added to the physical force
// model during propagation (it absorbs whatever the models get wrong). This
// class is a value type with three internal storage forms ("backends"):
//
//   - Zero       : no correction (default state)
//   - Table      : samples at given times, interpolated (cubic spline, cubic
//                  Hermite = cubic through values and slopes, or linear)
//   - BSpline    : B-spline coefficients (a smooth piecewise polynomial); the
//                  spline itself, or its 1st or 2nd time derivative, is the
//                  acceleration, as selected by EARepresents
//
// Construction goes through named factories (there are no setters):
//
//   ea = EmpiricalAcceleration::zero();
//   ea = EmpiricalAcceleration::from_entries(entries);
//   ea = EmpiricalAcceleration::from_entries(entries, frame, EAInterp::Linear);
//   ea = EmpiricalAcceleration::from_entries_hermite(entries, derivs);
//   ea = EmpiricalAcceleration::from_bspline(BSplineCoefficients{...},
//                                            EARepresents::Acceleration);
//
// The value is read with `accel_at(t_seconds)`. The caller (ForceModel) does
// any frame conversion, using the EA's stored frame().
//
// =============================================================================

#include "../core/types.hpp"
#include "../core/errors.hpp"
#include "../math/hermite_table.hpp"
#include "../math/bspline_basis.hpp"
#include <Eigen/Dense>
#include <vector>
#include <variant>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <utility>

namespace ballistic {
namespace forces {

/// Frame the EA vectors are stored in: GCRS (Earth-centered, non-rotating) or
/// ITRF (Earth-fixed, rotating with the Earth).
enum class EmpiricalAccelFrame {
    GCRS,   // No conversion needed in GCRS propagation
    ITRF    // ForceModel converts to propagation frame
};

/// Interpolation between table samples: cubic spline or straight lines.
enum class EAInterp { Cubic, Linear };

/// What a B-spline backend's spline is a fit of. The acceleration is then:
///   Acceleration → the spline value             (derivative order 0)
///   Velocity     → the spline's 1st derivative  (derivative order 1)
///   Position     → the spline's 2nd derivative  (derivative order 2)
enum class EARepresents { Acceleration, Velocity, Position };

/// One sample: `key` is seconds from the propagation epoch, `accel` the
/// acceleration (m/s²) at that time.
struct EmpiricalAccelEntry {
    double key;
    Vector3d accel;

    EmpiricalAccelEntry()
        : key(0.0), accel(Vector3d::Zero()) {}

    EmpiricalAccelEntry(double k, const Vector3d& a)
        : key(k), accel(a) {}
};

/// The three coordinate splines (x, y, z) sharing one knot vector and degree.
/// The constructor (and EmpiricalAcceleration::from_bspline, since the fields
/// are public) checks 1 <= degree <= kMaxDegree (ConfigError above it) and
/// `cx.size() == cy.size() == cz.size() == knots.size() - degree - 1`
/// (std::invalid_argument otherwise), so later code can rely on it.
/// Knot times are in seconds from the propagation epoch.
struct BSplineCoefficients {
    /// Highest degree the evaluator supports: it works in fixed stack buffers
    /// of degree + 1 basis values (no heap allocation per force evaluation).
    static constexpr int kMaxDegree = 7;

    Eigen::VectorXd cx;
    Eigen::VectorXd cy;
    Eigen::VectorXd cz;
    std::vector<double> knots;
    int degree;

    BSplineCoefficients() : degree(3) {}

    BSplineCoefficients(Eigen::VectorXd cx_,
                        Eigen::VectorXd cy_,
                        Eigen::VectorXd cz_,
                        std::vector<double> knots_,
                        int degree_)
        : cx(std::move(cx_)),
          cy(std::move(cy_)),
          cz(std::move(cz_)),
          knots(std::move(knots_)),
          degree(degree_)
    {
        validate();
    }

    /// The checks the constructor makes (see the struct comment).
    void validate() const {
        if (degree < 1) {
            throw std::invalid_argument(
                "BSplineCoefficients: degree must be >= 1");
        }
        if (degree > kMaxDegree) {
            throw ConfigError("BSplineCoefficients: degree " + std::to_string(degree) +
                " requested, but the empirical-acceleration evaluator supports degrees "
                "1 to " + std::to_string(kMaxDegree) + ". Use a degree of at most " +
                std::to_string(kMaxDegree) + " (the estimators use 3 or 5).");
        }
        if (cx.size() != cy.size() || cx.size() != cz.size()) {
            throw std::invalid_argument(
                "BSplineCoefficients: cx/cy/cz must have equal size");
        }
        const auto expected =
            static_cast<Eigen::Index>(knots.size()) - degree - 1;
        if (cx.size() != expected) {
            throw std::invalid_argument(
                "BSplineCoefficients: cx.size() must equal knots.size() - degree - 1");
        }
    }

    /// Number of B-spline basis functions (= coefficients per axis).
    [[nodiscard]] int n_basis() const noexcept {
        return static_cast<int>(cx.size());
    }
};

/// A time-only correction acceleration (see the file header). Cheap to copy
/// apart from its sample/coefficient arrays.
class EmpiricalAcceleration {
public:
    /// A default-constructed EA is the zero EA: accel_at returns zero at any
    /// time. Same as `EmpiricalAcceleration::zero()`.
    EmpiricalAcceleration() = default;

    // ===== Factories =====

    /// The zero EA (no correction).
    [[nodiscard]] static EmpiricalAcceleration zero() {
        return EmpiricalAcceleration{};
    }

    /// Samples interpolated by a cubic spline (default) or linearly. The
    /// entries may be unsorted (they are sorted by key). An empty vector gives
    /// the zero EA.
    [[nodiscard]] static EmpiricalAcceleration from_entries(
        std::vector<EmpiricalAccelEntry> entries,
        EmpiricalAccelFrame frame = EmpiricalAccelFrame::GCRS,
        EAInterp interp = EAInterp::Cubic);

    /// Samples interpolated by cubic Hermite polynomials using the given time
    /// derivatives (m/s³), which must be as many as the entries (else
    /// std::invalid_argument) and are reordered with them. An empty vector
    /// gives the zero EA.
    [[nodiscard]] static EmpiricalAcceleration from_entries_hermite(
        std::vector<EmpiricalAccelEntry> entries,
        std::vector<Vector3d> derivatives,
        EmpiricalAccelFrame frame = EmpiricalAccelFrame::GCRS);

    /// Keeps the B-spline as is (no conversion to a table); the acceleration is
    /// the spline derivative implied by `represents`.
    [[nodiscard]] static EmpiricalAcceleration from_bspline(
        BSplineCoefficients coeffs,
        EARepresents represents,
        EmpiricalAccelFrame frame = EmpiricalAccelFrame::GCRS);

    // ===== Query =====

    /// Acceleration (m/s²) at `t_seconds` since the propagation epoch, in the
    /// stored frame(). Zero for a zero EA. Outside the EA's time span the value
    /// at the nearest end is held.
    [[nodiscard]] Vector3d accel_at(double t_seconds) const;

    /// The stored frame (GCRS for the zero EA).
    [[nodiscard]] EmpiricalAccelFrame frame() const noexcept;

    /// True for the zero EA.
    [[nodiscard]] bool is_zero() const noexcept {
        return std::holds_alternative<ZeroBackend>(backend_);
    }

    /// Time grid in seconds from the epoch: the sample times for a table; the
    /// start, the distinct interior knots and the end for a B-spline. The
    /// Gauss-Newton refinement uses it to choose where to evaluate. Empty for
    /// the zero EA.
    [[nodiscard]] const std::vector<double>& keys() const noexcept;

    // ===== Read-only view of the B-spline =====

    /// Knot vector and degree of a B-spline EA.
    struct BSplineView {
        const std::vector<double>& knots;
        int degree;
    };

    /// The knots and degree, or nullopt unless built with from_bspline().
    [[nodiscard]] std::optional<BSplineView> as_bspline() const noexcept;

    /// Time span [t_start, t_end] in seconds from the epoch; {0, 0} for the
    /// zero EA.
    [[nodiscard]] std::pair<double, double> domain() const noexcept;

private:
    struct ZeroBackend {};

    struct TableBackend {
        math::HermiteTable<Vector3d> table;
        std::vector<double> keys;
        EmpiricalAccelFrame frame = EmpiricalAccelFrame::GCRS;
    };

    struct BSplineBackend {
        BSplineCoefficients coeffs;
        EARepresents represents = EARepresents::Acceleration;
        std::vector<double> keys;          // start, interior knots, end
        EmpiricalAccelFrame frame = EmpiricalAccelFrame::GCRS;

        [[nodiscard]] Vector3d eval(double t) const;
    };

    std::variant<ZeroBackend, TableBackend, BSplineBackend> backend_;
};

// ============================================================================
// Inline definitions
// ============================================================================

inline EmpiricalAcceleration EmpiricalAcceleration::from_entries(
    std::vector<EmpiricalAccelEntry> entries,
    EmpiricalAccelFrame frame,
    EAInterp interp)
{
    if (entries.empty()) {
        return EmpiricalAcceleration{};
    }

    std::sort(entries.begin(), entries.end(),
              [](const EmpiricalAccelEntry& a, const EmpiricalAccelEntry& b) {
                  return a.key < b.key;
              });

    std::vector<double> keys(entries.size());
    std::vector<Vector3d> accels(entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        keys[i] = entries[i].key;
        accels[i] = entries[i].accel;
    }

    TableBackend backend;
    backend.frame = frame;
    backend.keys = keys;
    if (interp == EAInterp::Linear) {
        backend.table = math::HermiteTable<Vector3d>::make_linear(
            std::move(keys), std::move(accels));
    } else {
        backend.table = math::HermiteTable<Vector3d>(
            std::move(keys), std::move(accels));
    }

    EmpiricalAcceleration ea;
    ea.backend_ = std::move(backend);
    return ea;
}

inline EmpiricalAcceleration EmpiricalAcceleration::from_entries_hermite(
    std::vector<EmpiricalAccelEntry> entries,
    std::vector<Vector3d> derivatives,
    EmpiricalAccelFrame frame)
{
    if (entries.empty()) {
        return EmpiricalAcceleration{};
    }
    if (entries.size() != derivatives.size()) {
        throw std::invalid_argument(
            "from_entries_hermite: entries.size() != derivatives.size()");
    }

    // Sort entries (and derivatives in matching order) by key
    std::vector<std::size_t> idx(entries.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](std::size_t a, std::size_t b) {
                  return entries[a].key < entries[b].key;
              });

    std::vector<double> keys(entries.size());
    std::vector<Vector3d> accels(entries.size());
    std::vector<Vector3d> derivs(entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        keys[i] = entries[idx[i]].key;
        accels[i] = entries[idx[i]].accel;
        derivs[i] = derivatives[idx[i]];
    }

    TableBackend backend;
    backend.frame = frame;
    backend.keys = keys;
    backend.table = math::HermiteTable<Vector3d>(
        std::move(keys), std::move(accels), std::move(derivs));

    EmpiricalAcceleration ea;
    ea.backend_ = std::move(backend);
    return ea;
}

inline EmpiricalAcceleration EmpiricalAcceleration::from_bspline(
    BSplineCoefficients coeffs,
    EARepresents represents,
    EmpiricalAccelFrame frame)
{
    coeffs.validate();  // the fields are public: re-check what eval() relies on
    BSplineBackend backend;
    backend.coeffs = std::move(coeffs);
    backend.represents = represents;
    backend.frame = frame;

    // Build the breakpoint grid: t_start, every distinct interior knot, t_end.
    const auto& knots = backend.coeffs.knots;
    const int degree = backend.coeffs.degree;
    const double t_start = knots[degree];
    const double t_end = knots[knots.size() - degree - 1];
    backend.keys.push_back(t_start);
    for (std::size_t k = static_cast<std::size_t>(degree) + 1;
         k + degree + 1 < knots.size(); ++k)
    {
        const double tk = knots[k];
        if (tk > t_start + 1e-10 && tk < t_end - 1e-10) {
            backend.keys.push_back(tk);
        }
    }
    backend.keys.push_back(t_end);

    EmpiricalAcceleration ea;
    ea.backend_ = std::move(backend);
    return ea;
}

inline Vector3d EmpiricalAcceleration::BSplineBackend::eval(double t) const {
    const int deg = coeffs.degree;
    const int n_idx = coeffs.n_basis() - 1;
    const double tc = std::clamp(t, coeffs.knots.front(), coeffs.knots.back());
    const int span = math::bspline_findspan(n_idx, deg, tc, coeffs.knots.data());
    const int col0 = span - deg;

    int deriv_order = 0;
    switch (represents) {
        case EARepresents::Acceleration: deriv_order = 0; break;
        case EARepresents::Velocity:     deriv_order = 1; break;
        case EARepresents::Position:     deriv_order = 2; break;
    }

    if (deriv_order == 0) {
        // Fixed buffer avoids heap allocation on this frequently called path
        // (degree <= kMaxDegree, checked when the coefficients were accepted)
        double N[BSplineCoefficients::kMaxDegree + 1];
        math::bspline_basis_funs(deg, span, tc, coeffs.knots.data(), N);
        Vector3d val = Vector3d::Zero();
        for (int j = 0; j <= deg; ++j) {
            val.x() += coeffs.cx(col0 + j) * N[j];
            val.y() += coeffs.cy(col0 + j) * N[j];
            val.z() += coeffs.cz(col0 + j) * N[j];
        }
        return val;
    } else {
        // (degree+1) * (deriv_order+1) entries, deriv_order <= 2
        double ders[(BSplineCoefficients::kMaxDegree + 1) * 3];
        math::bspline_ders(deg, span, tc, coeffs.knots.data(),
                           deriv_order, ders);
        const double* row = &ders[deriv_order * (deg + 1)];
        Vector3d val = Vector3d::Zero();
        for (int j = 0; j <= deg; ++j) {
            val.x() += coeffs.cx(col0 + j) * row[j];
            val.y() += coeffs.cy(col0 + j) * row[j];
            val.z() += coeffs.cz(col0 + j) * row[j];
        }
        return val;
    }
}

inline Vector3d EmpiricalAcceleration::accel_at(double t_seconds) const {
    return std::visit([t_seconds](const auto& b) -> Vector3d {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, ZeroBackend>)         return Vector3d::Zero();
        else if constexpr (std::is_same_v<T, TableBackend>)   return b.table(t_seconds);
        else if constexpr (std::is_same_v<T, BSplineBackend>) return b.eval(t_seconds);
    }, backend_);
}

inline EmpiricalAccelFrame EmpiricalAcceleration::frame() const noexcept {
    return std::visit([](const auto& b) -> EmpiricalAccelFrame {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, ZeroBackend>) return EmpiricalAccelFrame::GCRS;
        else                                          return b.frame;
    }, backend_);
}

inline const std::vector<double>& EmpiricalAcceleration::keys() const noexcept {
    static const std::vector<double> empty_keys;
    return std::visit([&](const auto& b) -> const std::vector<double>& {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, ZeroBackend>) return empty_keys;
        else                                          return b.keys;
    }, backend_);
}

inline std::optional<EmpiricalAcceleration::BSplineView>
EmpiricalAcceleration::as_bspline() const noexcept {
    if (std::holds_alternative<BSplineBackend>(backend_)) {
        const auto& b = std::get<BSplineBackend>(backend_);
        return BSplineView{b.coeffs.knots, b.coeffs.degree};
    }
    return std::nullopt;
}

inline std::pair<double, double>
EmpiricalAcceleration::domain() const noexcept {
    const auto& k = keys();
    if (k.empty()) return {0.0, 0.0};
    return {k.front(), k.back()};
}

}  // namespace forces
}  // namespace ballistic
