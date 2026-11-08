#pragma once

// =============================================================================
// Air-density tables for DragEnvironment (the "density column")
// =============================================================================
//
// Density ρ [kg/m³] tabulated against geodetic altitude [km] (height above
// the WGS84 ellipsoid), and optionally time and place. Three shapes:
//   DensityProfile1D  -- altitude only (short arcs near one lat/lon)
//   DensityProfile2D  -- altitude × time along a ground track
//                        (from_models_along, from_accelerometer_density)
//   DensityField4D    -- altitude × latitude × longitude × time, for many
//                        satellites sharing one atmosphere. Stores log(ρ) in
//                        16-bit floats (see log_density_storage.hpp).
// The default drag setup (make_drag_environment) uses none of these: it
// evaluates the model live (live_model.hpp).
//
// Interpolation:
//   - Altitude: cubic spline in log(ρ) (density falls roughly exponentially
//     with height, so log(ρ) is smooth and nearly linear)
//   - Time:     linear
//   - Lat/Lon:  bilinear (4D only)
//
// Outside the table:
//   - Above the top altitude: held at the top value (the air is so thin there
//     that drag is negligible)
//   - Below the lowest altitude: throws std::out_of_range — extend the grid,
//     or add the standard atmosphere below a floor with extend_density_below.
//   - Time, lat, lon: held at the first/last value.
//
// Every evaluate(position_ecef, epoch) takes the same arguments (ECEF
// position [m], epoch), like the wind tables, so DragEnvironment can call any
// of them the same way.
//
// =============================================================================

#include "../core/errors.hpp"
#include "../core/types.hpp"
#include "../math/cubic_spline.hpp"
#include "../coordinates/geodetic.hpp"
#include "../time/epoch.hpp"
#include "log_density_storage.hpp"
#include "detail/profile_helpers.hpp"
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <string>

namespace ballistic {
namespace atmosphere {

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

namespace detail {

// Cubic spline of ln(ρ) against altitude; evaluate() returns ρ =
// exp(spline(altitude_km)). Throws std::invalid_argument for mismatched sizes
// or a non-positive density, and evaluate() throws std::out_of_range below
// the lowest knot.
class LogRhoSpline {
public:
    LogRhoSpline() = default;
    explicit LogRhoSpline(const std::vector<double>& altitude_km,
                          const std::vector<double>& density_kg_m3) {
        if (altitude_km.size() != density_kg_m3.size()) {
            throw std::invalid_argument(
                "LogRhoSpline: altitude / density size mismatch");
        }
        std::vector<double> log_rho;
        log_rho.reserve(density_kg_m3.size());
        for (double rho : density_kg_m3) {
            if (!(rho > 0.0)) {
                throw std::invalid_argument(
                    "LogRhoSpline: density values must be strictly positive");
            }
            log_rho.push_back(std::log(rho));
        }
        spline_ = math::CubicSpline<double>(altitude_km, std::move(log_rho));
        if (!altitude_km.empty()) {
            alt_lo_ = altitude_km.front();
            alt_hi_ = altitude_km.back();
        }
    }

    [[nodiscard]] double evaluate(double altitude_km) const {
        if (spline_.empty()) return 0.0;
        // 1 mm of slack: a position built exactly on the lowest knot comes
        // back from ecef_to_geodetic ~1e-9 m below it (round-off).
        if (altitude_km < alt_lo_ - 1e-6) {
            throw std::out_of_range(
                "Density evaluation below the table's lowest altitude (" +
                std::to_string(alt_lo_) + " km). Build the table down to the "
                "altitudes you fly, or add the standard atmosphere below a floor "
                "with extend_density_below.");
        }
        altitude_km = std::max(altitude_km, alt_lo_);
        // Above-table clamping is handled by the spline itself.
        return std::exp(spline_(altitude_km));
    }

    [[nodiscard]] bool empty() const noexcept { return spline_.empty(); }
    [[nodiscard]] double altitude_lo_km() const noexcept { return alt_lo_; }
    [[nodiscard]] double altitude_hi_km() const noexcept { return alt_hi_; }

private:
    math::CubicSpline<double> spline_;
    double alt_lo_ = 0.0;
    double alt_hi_ = 0.0;
};

// Geodetic altitude (km) from an ECEF position.
inline double altitude_km_from_ecef(const Vector3d& pos_ecef) {
    auto geo = coords::ecef_to_geodetic(pos_ecef);
    return geo.altitude / 1000.0;
}

}  // namespace detail

// -----------------------------------------------------------------------------
// DensityProfile1D — snapshot at one (lat, lon, epoch); altitude only.
// -----------------------------------------------------------------------------
/// Density [kg/m³] at each altitude [km], independent of time and place.
/// Densities must be strictly positive (std::invalid_argument otherwise);
/// evaluate() throws std::out_of_range below the lowest altitude.
class DensityProfile1D {
public:
    DensityProfile1D() = default;

    DensityProfile1D(std::vector<double> altitude_km,
                     std::vector<double> density_kg_m3)
        : altitude_km_(std::move(altitude_km))
        , density_kg_m3_(std::move(density_kg_m3))
        , spline_(altitude_km_, density_kg_m3_)
    {}

    [[nodiscard]] double evaluate(const Vector3d& position_ecef,
                                   const Epoch& /*epoch*/) const {
        return spline_.evaluate(detail::altitude_km_from_ecef(position_ecef));
    }

    [[nodiscard]] bool empty() const noexcept { return spline_.empty(); }
    [[nodiscard]] const std::vector<double>& altitude_km() const noexcept { return altitude_km_; }
    [[nodiscard]] const std::vector<double>& density_kg_m3() const noexcept { return density_kg_m3_; }

private:
    std::vector<double> altitude_km_;
    std::vector<double> density_kg_m3_;
    detail::LogRhoSpline spline_;
};

// -----------------------------------------------------------------------------
// DensityProfile2D — (altitude × epoch), for a table that follows a track.
// -----------------------------------------------------------------------------
/// Density over altitude × time. Built along a ground track
/// (from_models_along), so the place at each time is implied by the track.
class DensityProfile2D {
public:
    DensityProfile2D() = default;

    /// `density_grid[t_idx * n_alt + h_idx]` is ρ [kg/m³] at altitude h_idx
    /// and epoch t_idx. Throws std::invalid_argument if the size is not
    /// altitudes × epochs or a density is not strictly positive.
    DensityProfile2D(std::vector<double> altitude_km,
                     std::vector<Epoch> epochs,
                     std::vector<double> density_grid)
        : altitude_km_(std::move(altitude_km))
        , epochs_(std::move(epochs))
        , density_grid_(std::move(density_grid))
    {
        validate();
        // Build one spline per epoch.
        slices_.reserve(epochs_.size());
        const std::size_t n_alt = altitude_km_.size();
        for (std::size_t ti = 0; ti < epochs_.size(); ++ti) {
            std::vector<double> rho_slice(density_grid_.begin() + ti * n_alt,
                                           density_grid_.begin() + (ti + 1) * n_alt);
            slices_.emplace_back(altitude_km_, rho_slice);
        }
    }

    [[nodiscard]] double evaluate(const Vector3d& position_ecef,
                                   const Epoch& epoch) const {
        if (slices_.empty()) return 0.0;
        const double h_km = detail::altitude_km_from_ecef(position_ecef);
        if (slices_.size() == 1) return slices_[0].evaluate(h_km);
        auto [lo, frac] = detail::linear_time_lookup(epochs_, epoch);
        const double rho_lo = slices_[lo].evaluate(h_km);
        const double rho_hi = slices_[lo + 1].evaluate(h_km);
        // Linear in time (not in log(ρ)): the change between neighboring
        // time samples is small enough that the difference does not matter.
        return rho_lo + frac * (rho_hi - rho_lo);
    }

    [[nodiscard]] bool empty() const noexcept { return slices_.empty(); }
    [[nodiscard]] const std::vector<double>& altitude_km() const noexcept { return altitude_km_; }
    [[nodiscard]] const std::vector<Epoch>& epochs() const noexcept { return epochs_; }
    [[nodiscard]] const std::vector<double>& density_grid() const noexcept { return density_grid_; }

private:
    void validate() const {
        if (density_grid_.size() != altitude_km_.size() * epochs_.size()) {
            throw std::invalid_argument(
                "DensityProfile2D: density_grid size must equal "
                "altitude_km.size() * epochs.size()");
        }
    }

    std::vector<double> altitude_km_;
    std::vector<Epoch>  epochs_;
    std::vector<double> density_grid_;  // row-major: time × altitude
    std::vector<detail::LogRhoSpline> slices_;
};

// -----------------------------------------------------------------------------
// DensityField4D — (altitude × lat × lon × epoch), log(rho) in binary16 (`half`).
// -----------------------------------------------------------------------------
/// Density over altitude [km] × latitude [deg] × longitude [deg, -180..180]
/// × time, stored as log(ρ) in 16-bit floats. Storage layout:
///   log_rho_[((t * n_lon + j) * n_lat + i) * n_alt + k]
///     = encoded log(ρ) at (alt[k], lat[i], lon[j], epoch[t])
/// so altitude varies fastest and time slowest; one altitude column is
/// contiguous, which is what each query reads. The altitude axis needs at
/// least two values (a spline through one point has no shape; ConfigError
/// otherwise). A latitude, longitude or time axis with a single value is
/// held at that value. Throws std::invalid_argument if the storage size does
/// not match the axes.
// -----------------------------------------------------------------------------
class DensityField4D {
public:
    DensityField4D() = default;

    DensityField4D(std::vector<double> altitude_km,
                   std::vector<double> latitude_deg,
                   std::vector<double> longitude_deg,
                   std::vector<Epoch>  epochs,
                   std::vector<half>   log_rho_storage)
        : altitude_km_(std::move(altitude_km))
        , latitude_deg_(std::move(latitude_deg))
        , longitude_deg_(std::move(longitude_deg))
        , epochs_(std::move(epochs))
        , log_rho_(std::move(log_rho_storage))
    {
        validate();
    }

    [[nodiscard]] double evaluate(const Vector3d& position_ecef,
                                   const Epoch& epoch) const {
        if (empty()) return 0.0;
        auto geo = coords::ecef_to_geodetic(position_ecef);
        constexpr double RAD_TO_DEG = 57.29577951308232;
        const double lat_deg = geo.latitude * RAD_TO_DEG;
        const double lon_deg = wrap_lon_to_grid(geo.longitude * RAD_TO_DEG);
        const double alt_km  = geo.altitude / 1000.0;

        auto [ti, t_frac] = detail::linear_time_lookup(epochs_, epoch);
        const auto [ii, i_frac] = detail::find_axis_index(latitude_deg_, lat_deg);
        const auto [ji, j_frac] = detail::find_axis_index(longitude_deg_, lon_deg);

        // Cubic spline in altitude (log-rho) at each of the 8 surrounding cells.
        auto sample_cell = [&](std::size_t t_idx, std::size_t i_idx, std::size_t j_idx) {
            return cell_density_at_altitude(t_idx, i_idx, j_idx, alt_km);
        };

        // Upper neighbor on each axis; a single-value axis uses its only
        // cell for both (the lookups return frac 0 there, so it is held).
        const std::size_t ti1 = std::min(ti + 1, epochs_.size() - 1);
        const std::size_t ii1 = std::min(ii + 1, latitude_deg_.size() - 1);
        const std::size_t ji1 = std::min(ji + 1, longitude_deg_.size() - 1);

        const double r000 = sample_cell(ti,  ii,  ji );
        const double r001 = sample_cell(ti,  ii,  ji1);
        const double r010 = sample_cell(ti,  ii1, ji );
        const double r011 = sample_cell(ti,  ii1, ji1);
        const double r100 = sample_cell(ti1, ii,  ji );
        const double r101 = sample_cell(ti1, ii,  ji1);
        const double r110 = sample_cell(ti1, ii1, ji );
        const double r111 = sample_cell(ti1, ii1, ji1);

        // Bilinear in lat/lon at each time slice, then linear in time.
        const double r_t0 = bilinear(r000, r001, r010, r011, i_frac, j_frac);
        const double r_t1 = bilinear(r100, r101, r110, r111, i_frac, j_frac);
        return r_t0 + t_frac * (r_t1 - r_t0);
    }

    [[nodiscard]] bool empty() const noexcept { return log_rho_.empty(); }
    [[nodiscard]] std::size_t n_altitude() const noexcept { return altitude_km_.size(); }
    [[nodiscard]] std::size_t n_latitude() const noexcept { return latitude_deg_.size(); }
    [[nodiscard]] std::size_t n_longitude() const noexcept { return longitude_deg_.size(); }
    [[nodiscard]] std::size_t n_epoch() const noexcept { return epochs_.size(); }

    // Axes and raw storage, read by the Python save_drag_env / load_drag_env
    // (NPZ files). Storage order as described above the class.
    [[nodiscard]] const std::vector<double>& altitude_km() const noexcept { return altitude_km_; }
    [[nodiscard]] const std::vector<double>& latitude_deg() const noexcept { return latitude_deg_; }
    [[nodiscard]] const std::vector<double>& longitude_deg() const noexcept { return longitude_deg_; }
    [[nodiscard]] const std::vector<Epoch>& epochs() const noexcept { return epochs_; }
    [[nodiscard]] const std::vector<half>& encoded_log_rho() const noexcept { return log_rho_; }

private:
    void validate() const {
        const std::size_t expected = altitude_km_.size() * latitude_deg_.size()
                                     * longitude_deg_.size() * epochs_.size();
        if (log_rho_.size() != expected) {
            throw std::invalid_argument(
                "DensityField4D: log_rho_storage size must equal "
                "n_alt * n_lat * n_lon * n_epoch");
        }
        if (!log_rho_.empty() && altitude_km_.size() < 2) {
            throw ConfigError(
                "DensityField4D: the altitude axis has " +
                std::to_string(altitude_km_.size()) +
                " value; it needs at least two (the density is a spline in "
                "altitude, and one point gives no profile). Tabulate at least "
                "two altitudes spanning the heights you fly.");
        }
    }

    static double wrap_lon_to_grid(double lon_deg) noexcept {
        return detail::wrap_longitude_deg(lon_deg);
    }

    static double bilinear(double v00, double v01, double v10, double v11,
                            double fi, double fj) noexcept {
        const double v_lo = v00 + fj * (v01 - v00);
        const double v_hi = v10 + fj * (v11 - v10);
        return v_lo + fi * (v_hi - v_lo);
    }

    // Decode the altitude column of one (time, lat, lon) cell and evaluate its
    // log(ρ) spline at alt_km.
    double cell_density_at_altitude(std::size_t ti, std::size_t ii,
                                     std::size_t ji, double alt_km) const {
        const std::size_t n_alt = altitude_km_.size();
        const std::size_t n_lat = latitude_deg_.size();
        const std::size_t n_lon = longitude_deg_.size();
        const std::size_t base = ((ti * n_lon + ji) * n_lat + ii) * n_alt;

        // Decode the altitude column from half-precision log-rho.
        std::vector<double> rho_column(n_alt);
        for (std::size_t k = 0; k < n_alt; ++k) {
            rho_column[k] = decode_log_rho(log_rho_[base + k]);
        }
        detail::LogRhoSpline spline(altitude_km_, rho_column);
        return spline.evaluate(alt_km);
    }

    std::vector<double> altitude_km_;
    std::vector<double> latitude_deg_;
    std::vector<double> longitude_deg_;
    std::vector<Epoch>  epochs_;
    std::vector<half>   log_rho_;
};

}  // namespace atmosphere
}  // namespace ballistic
