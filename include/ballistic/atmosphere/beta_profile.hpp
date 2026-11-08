#pragma once

// =============================================================================
// Ballistic-coefficient tables for DragEnvironment (the "beta column")
// =============================================================================
//
// The ballistic coefficient β = m / (Cd · A) [kg/m²] is the satellite's mass
// m over its drag coefficient Cd (dimensionless, typically 2-4 in orbit) times
// its reference area A. The drag deceleration is inversely proportional to β:
// a higher β means less drag.
//
// Two shapes:
//   BetaProfile1D  -- β against altitude (usually a constant: the satellite's
//                      mass, shape and attitude do not change over the arc)
//   BetaProfile2D  -- β against altitude × time, for an arc over which mass
//                      or attitude (and so Cd·A) changes
//
// β describes the satellite, not the air, so it does not depend on latitude
// or longitude and there is no 4D table. For a Cd computed from the flow
// (reentry), use aero::AeroBeta instead (aero/flow_regime.hpp).
//
// Interpolation: cubic spline in altitude; linear in time (2D only).
// Outside the table: the edge value is held on every axis.
//
// =============================================================================

#include "../core/types.hpp"
#include "../math/cubic_spline.hpp"
#include "../coordinates/geodetic.hpp"
#include "../time/epoch.hpp"
#include "detail/profile_helpers.hpp"
#include <vector>
#include <stdexcept>

namespace ballistic {
namespace atmosphere {

// -----------------------------------------------------------------------------
// BetaProfile1D — altitude-only.
// -----------------------------------------------------------------------------
/// β [kg/m²] at each geodetic altitude [km]. Default-constructed = empty (no
/// drag). Usually built with constant() or from_mass_cd_area().
class BetaProfile1D {
public:
    BetaProfile1D() = default;

    /// Throws std::invalid_argument if a β value is not strictly positive.
    BetaProfile1D(std::vector<double> altitude_km,
                  std::vector<double> beta_kg_m2)
        : altitude_km_(std::move(altitude_km))
        , beta_kg_m2_(std::move(beta_kg_m2))
        , spline_(altitude_km_, beta_kg_m2_)
    {
        for (double b : beta_kg_m2_) {
            if (!(b > 0.0)) {
                throw std::invalid_argument(
                    "BetaProfile1D: beta values must be strictly positive");
            }
        }
    }

    /// Constant β [kg/m²], the usual case. The altitude span [km] only sets
    /// the two knots; the value is held outside it too. Throws
    /// std::invalid_argument unless beta_kg_m2 > 0.
    [[nodiscard]] static BetaProfile1D constant(double beta_kg_m2,
                                                 double altitude_min_km = 0.0,
                                                 double altitude_max_km = 1500.0) {
        if (!(beta_kg_m2 > 0.0)) {
            throw std::invalid_argument(
                "BetaProfile1D::constant: beta must be > 0");
        }
        return BetaProfile1D({altitude_min_km, altitude_max_km},
                              {beta_kg_m2, beta_kg_m2});
    }

    /// Constant β = mass_kg / (cd · area_m2) [kg/m²] from the satellite's mass
    /// [kg], drag coefficient and reference area [m²] (the mean area facing
    /// the flow, not the total surface). Throws std::invalid_argument unless
    /// all three are > 0.
    [[nodiscard]] static BetaProfile1D from_mass_cd_area(double mass_kg,
                                                          double cd,
                                                          double area_m2,
                                                          double altitude_min_km = 0.0,
                                                          double altitude_max_km = 1500.0) {
        if (!(mass_kg > 0.0) || !(cd > 0.0) || !(area_m2 > 0.0)) {
            throw std::invalid_argument(
                "BetaProfile1D::from_mass_cd_area: mass, Cd, and area must be > 0");
        }
        return constant(mass_kg / (cd * area_m2), altitude_min_km, altitude_max_km);
    }

    [[nodiscard]] double evaluate(const Vector3d& position_ecef,
                                   const Epoch& /*epoch*/) const {
        if (spline_.empty()) return 0.0;
        auto geo = coords::ecef_to_geodetic(position_ecef);
        return spline_(geo.altitude / 1000.0);
    }

    [[nodiscard]] bool empty() const noexcept { return spline_.empty(); }
    [[nodiscard]] const std::vector<double>& altitude_km() const noexcept { return altitude_km_; }
    [[nodiscard]] const std::vector<double>& beta_kg_m2() const noexcept { return beta_kg_m2_; }

private:
    std::vector<double> altitude_km_;
    std::vector<double> beta_kg_m2_;
    math::CubicSpline<double> spline_;
};

// -----------------------------------------------------------------------------
// BetaProfile2D — (altitude × epoch). For mass loss or attitude changes.
// -----------------------------------------------------------------------------
/// β over altitude [km] × time.
class BetaProfile2D {
public:
    BetaProfile2D() = default;

    /// `beta_grid[t_idx * n_alt + h_idx]` is β [kg/m²] at altitude h_idx and
    /// epoch t_idx. Throws std::invalid_argument if the size is not
    /// altitudes × epochs or a value is not strictly positive.
    BetaProfile2D(std::vector<double> altitude_km,
                  std::vector<Epoch>  epochs,
                  std::vector<double> beta_grid)
        : altitude_km_(std::move(altitude_km))
        , epochs_(std::move(epochs))
        , beta_grid_(std::move(beta_grid))
    {
        if (beta_grid_.size() != altitude_km_.size() * epochs_.size()) {
            throw std::invalid_argument(
                "BetaProfile2D: beta_grid size must equal n_alt * n_epoch");
        }
        for (double b : beta_grid_) {
            if (!(b > 0.0)) {
                throw std::invalid_argument(
                    "BetaProfile2D: beta values must be strictly positive");
            }
        }
        slices_.reserve(epochs_.size());
        const std::size_t n_alt = altitude_km_.size();
        for (std::size_t ti = 0; ti < epochs_.size(); ++ti) {
            std::vector<double> slice(beta_grid_.begin() + ti * n_alt,
                                       beta_grid_.begin() + (ti + 1) * n_alt);
            slices_.emplace_back(altitude_km_, std::move(slice));
        }
    }

    [[nodiscard]] double evaluate(const Vector3d& position_ecef,
                                   const Epoch& epoch) const {
        if (slices_.empty()) return 0.0;
        auto geo = coords::ecef_to_geodetic(position_ecef);
        const double alt_km = geo.altitude / 1000.0;
        if (slices_.size() == 1) return slices_[0](alt_km);
        auto [lo, frac] = detail::linear_time_lookup(epochs_, epoch);
        const double b_lo = slices_[lo](alt_km);
        const double b_hi = slices_[lo + 1](alt_km);
        return b_lo + frac * (b_hi - b_lo);
    }

    [[nodiscard]] bool empty() const noexcept { return slices_.empty(); }
    // Time axis. DragEnvironment::domain() includes it, so a query past the
    // last slice (β held at its edge value) sets DragSample::time_clamped
    // and a Trajectory warns, as for tabulated density and wind.
    [[nodiscard]] const std::vector<Epoch>& epochs() const noexcept { return epochs_; }

private:
    std::vector<double> altitude_km_;
    std::vector<Epoch>  epochs_;
    std::vector<double> beta_grid_;
    std::vector<math::CubicSpline<double>> slices_;
};

}  // namespace atmosphere
}  // namespace ballistic
