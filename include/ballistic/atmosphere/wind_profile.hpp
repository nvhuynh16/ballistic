#pragma once

// =============================================================================
// Wind tables for DragEnvironment (the "wind column")
// =============================================================================
//
// Wind is the air's velocity relative to the rotating Earth [m/s]. Two sets
// of axes appear here:
//   - ENU (east, north, up): local axes at the point below the satellite.
//     HWM14 gives the wind as a meridional (northward) and a zonal (eastward)
//     component in these axes.
//   - ECEF (Earth-centered, Earth-fixed): global axes rotating with the Earth.
//
// Three shapes:
//   WindProfile1D_ENU   -- altitude only, stored as ENU components and turned
//                          into ECEF at the query point when evaluated.
//   WindProfile2D_ECEF  -- altitude × time along a ground track, in ECEF.
//   WindField4D_ECEF    -- altitude × latitude × longitude × time, in ECEF.
//
// ENU suits only the 1D profile, which stores no latitude/longitude. Tables
// that follow a track or a grid store ECEF vectors directly, so no rotation
// is needed per query.
//
// Interpolation, as in density_profile.hpp:
//   - Altitude: cubic spline per component
//   - Lat/Lon:  bilinear
//   - Time:     linear
// Outside the table every axis is clamped to its first/last value (the wind
// is bounded and smooth, so holding the edge value is harmless).
//
// =============================================================================

#include "../core/errors.hpp"
#include "../core/types.hpp"
#include "../math/cubic_spline.hpp"
#include "../coordinates/geodetic.hpp"
#include "../time/epoch.hpp"
#include "detail/profile_helpers.hpp"
#include "log_density_storage.hpp"  // atmosphere::half, float_to_half, half_to_float
#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>

namespace ballistic {
namespace atmosphere {

// -----------------------------------------------------------------------------
// WindProfile1D_ENU — northward + eastward wind vs altitude.
// -----------------------------------------------------------------------------
/// Horizontal wind as a function of altitude only: northward and eastward
/// components [m/s] at each altitude [km]. When evaluated, the components are
/// placed in the local east/north axes of the query point and rotated to
/// ECEF; the vertical wind is zero. Default-constructed = no wind.
class WindProfile1D_ENU {
public:
    WindProfile1D_ENU() = default;

    /// Throws InputPlausibilityError naming the three lengths if they differ.
    WindProfile1D_ENU(std::vector<double> altitude_km,
                      std::vector<double> meridional_ms,  // north +
                      std::vector<double> zonal_ms)       // east +
        : altitude_km_(checked_altitudes(std::move(altitude_km),
                                         meridional_ms.size(), zonal_ms.size()))
        , meridional_ms_(std::move(meridional_ms))
        , zonal_ms_(std::move(zonal_ms))
        , spline_meridional_(altitude_km_, meridional_ms_)
        , spline_zonal_(altitude_km_, zonal_ms_)
    {}

    [[nodiscard]] Vector3d evaluate(const Vector3d& position_ecef,
                                     const Epoch& /*epoch*/) const {
        if (empty()) return Vector3d::Zero();
        auto geo = coords::ecef_to_geodetic(position_ecef);
        const double alt_km = geo.altitude / 1000.0;
        const double meridional = spline_meridional_(alt_km);
        const double zonal      = spline_zonal_(alt_km);
        // ENU at sub-satellite (lat, lon), transformed to ECEF.
        const Matrix3d R = coords::ecef_to_enu_rotation(geo.latitude, geo.longitude);
        const Vector3d wind_enu(zonal, meridional, 0.0);
        return R.transpose() * wind_enu;
    }

    [[nodiscard]] bool empty() const noexcept { return spline_meridional_.empty(); }
    [[nodiscard]] const std::vector<double>& altitude_km() const noexcept { return altitude_km_; }
    [[nodiscard]] const std::vector<double>& meridional_ms() const noexcept { return meridional_ms_; }
    [[nodiscard]] const std::vector<double>& zonal_ms() const noexcept { return zonal_ms_; }

private:
    // Runs before the splines are built, so a length mismatch is reported with
    // all three lengths rather than as one spline's x/y mismatch.
    static std::vector<double> checked_altitudes(std::vector<double> altitude_km,
                                                 std::size_t n_meridional,
                                                 std::size_t n_zonal) {
        if (n_meridional != altitude_km.size() || n_zonal != altitude_km.size()) {
            throw InputPlausibilityError(
                "WindProfile1D_ENU: " + std::to_string(altitude_km.size()) +
                " altitudes but " + std::to_string(n_meridional) + " meridional and " +
                std::to_string(n_zonal) + " zonal wind values; give one wind pair per altitude.");
        }
        return altitude_km;
    }

    std::vector<double> altitude_km_;
    std::vector<double> meridional_ms_;
    std::vector<double> zonal_ms_;
    math::CubicSpline<double> spline_meridional_;
    math::CubicSpline<double> spline_zonal_;
};

// -----------------------------------------------------------------------------
// WindProfile2D_ECEF — (altitude × epoch), wind stored in ECEF directly.
// -----------------------------------------------------------------------------
/// Wind over altitude × time, evaluated beforehand along a ground track
/// (from_models_along): the place at each time is implied by the track, and
/// the vectors are already in ECEF axes [m/s].
///
/// Storage: wind_grid[t_idx * n_alt + h_idx] is the ECEF wind at altitude
/// h_idx and epoch t_idx.
// -----------------------------------------------------------------------------
class WindProfile2D_ECEF {
public:
    WindProfile2D_ECEF() = default;

    /// Throws std::invalid_argument if wind_grid_ecef.size() is not
    /// altitudes × epochs.
    WindProfile2D_ECEF(std::vector<double> altitude_km,
                       std::vector<Epoch>  epochs,
                       std::vector<Vector3d> wind_grid_ecef)
        : altitude_km_(std::move(altitude_km))
        , epochs_(std::move(epochs))
        , wind_grid_(std::move(wind_grid_ecef))
    {
        if (wind_grid_.size() != altitude_km_.size() * epochs_.size()) {
            throw std::invalid_argument(
                "WindProfile2D_ECEF: wind_grid size must equal n_alt * n_epoch");
        }
        // Build per-component splines per epoch slice.
        slices_x_.reserve(epochs_.size());
        slices_y_.reserve(epochs_.size());
        slices_z_.reserve(epochs_.size());
        const std::size_t n_alt = altitude_km_.size();
        for (std::size_t ti = 0; ti < epochs_.size(); ++ti) {
            std::vector<double> wx(n_alt), wy(n_alt), wz(n_alt);
            for (std::size_t k = 0; k < n_alt; ++k) {
                const Vector3d& w = wind_grid_[ti * n_alt + k];
                wx[k] = w.x(); wy[k] = w.y(); wz[k] = w.z();
            }
            slices_x_.emplace_back(altitude_km_, std::move(wx));
            slices_y_.emplace_back(altitude_km_, std::move(wy));
            slices_z_.emplace_back(altitude_km_, std::move(wz));
        }
    }

    [[nodiscard]] Vector3d evaluate(const Vector3d& position_ecef,
                                     const Epoch& epoch) const {
        if (empty()) return Vector3d::Zero();
        auto geo = coords::ecef_to_geodetic(position_ecef);
        const double alt_km = geo.altitude / 1000.0;
        auto [lo, frac] = detail::linear_time_lookup(epochs_, epoch);
        if (slices_x_.size() == 1) {
            return Vector3d(slices_x_[0](alt_km),
                            slices_y_[0](alt_km),
                            slices_z_[0](alt_km));
        }
        const Vector3d w_lo(slices_x_[lo](alt_km),
                            slices_y_[lo](alt_km),
                            slices_z_[lo](alt_km));
        const Vector3d w_hi(slices_x_[lo + 1](alt_km),
                            slices_y_[lo + 1](alt_km),
                            slices_z_[lo + 1](alt_km));
        return w_lo + frac * (w_hi - w_lo);
    }

    [[nodiscard]] bool empty() const noexcept { return slices_x_.empty(); }
    [[nodiscard]] const std::vector<double>& altitude_km() const noexcept { return altitude_km_; }
    [[nodiscard]] const std::vector<Epoch>& epochs() const noexcept { return epochs_; }
    [[nodiscard]] const std::vector<Vector3d>& wind_grid() const noexcept { return wind_grid_; }

private:
    std::vector<double>   altitude_km_;
    std::vector<Epoch>    epochs_;
    std::vector<Vector3d> wind_grid_;
    std::vector<math::CubicSpline<double>> slices_x_, slices_y_, slices_z_;
};

// -----------------------------------------------------------------------------
// WindField4D_ECEF — (altitude × lat × lon × epoch), wind in ECEF.
// -----------------------------------------------------------------------------
/// Wind over altitude [km] × latitude [deg] × longitude [deg, -180..180] ×
/// time, in ECEF axes [m/s], stored compactly as three 16-bit floats per cell
/// (IEEE binary16, `half`). The rounding of a 16-bit float, at most ~0.05% of
/// the value, is far below the model's own uncertainty. Memory: 6 bytes per
/// cell, ≈ 72 MB for a 12-million-cell grid.
///
/// Storage: cell index ((t * n_lon + j) * n_lat + i) * n_alt + k (altitude
/// varies fastest, time slowest), components at wind_packed_[3 * idx + 0..2].
// -----------------------------------------------------------------------------
class WindField4D_ECEF {
public:
    WindField4D_ECEF() = default;

    /// Takes full-precision vectors in the storage order above and rounds
    /// them to 16 bits. Throws std::invalid_argument on a size mismatch and
    /// ConfigError for fewer than two altitudes. A latitude, longitude or
    /// time axis with a single value is held at that value.
    WindField4D_ECEF(std::vector<double> altitude_km,
                     std::vector<double> latitude_deg,
                     std::vector<double> longitude_deg,
                     std::vector<Epoch>  epochs,
                     const std::vector<Vector3d>& wind_storage_ecef)
        : altitude_km_(std::move(altitude_km))
        , latitude_deg_(std::move(latitude_deg))
        , longitude_deg_(std::move(longitude_deg))
        , epochs_(std::move(epochs))
    {
        const std::size_t expected = altitude_km_.size() * latitude_deg_.size()
                                     * longitude_deg_.size() * epochs_.size();
        if (wind_storage_ecef.size() != expected) {
            throw std::invalid_argument(
                "WindField4D_ECEF: wind_storage size must equal "
                "n_alt * n_lat * n_lon * n_epoch");
        }
        if (expected > 0 && altitude_km_.size() < 2) {
            throw ConfigError(
                "WindField4D_ECEF: the altitude axis has " +
                std::to_string(altitude_km_.size()) +
                " value; it needs at least two (the wind is a spline in "
                "altitude, and through one point it would read as zero). "
                "Tabulate at least two altitudes spanning the heights you fly.");
        }
        // Pack each Vector3d into 3 halves.
        wind_packed_.reserve(3 * expected);
        for (const auto& v : wind_storage_ecef) {
            wind_packed_.push_back(float_to_half(static_cast<float>(v.x())));
            wind_packed_.push_back(float_to_half(static_cast<float>(v.y())));
            wind_packed_.push_back(float_to_half(static_cast<float>(v.z())));
        }
    }

    [[nodiscard]] Vector3d evaluate(const Vector3d& position_ecef,
                                     const Epoch& epoch) const {
        if (empty()) return Vector3d::Zero();
        auto geo = coords::ecef_to_geodetic(position_ecef);
        constexpr double RAD_TO_DEG = 57.29577951308232;
        const double lat_deg = geo.latitude * RAD_TO_DEG;
        const double lon_deg = wrap_lon_to_grid(geo.longitude * RAD_TO_DEG);
        const double alt_km  = geo.altitude / 1000.0;

        auto [ti, t_frac] = detail::linear_time_lookup(epochs_, epoch);
        const auto [ii, i_frac] = detail::find_axis_index(latitude_deg_, lat_deg);
        const auto [ji, j_frac] = detail::find_axis_index(longitude_deg_, lon_deg);

        auto sample_cell = [&](std::size_t t_idx, std::size_t i_idx, std::size_t j_idx) {
            return cell_wind_at_altitude(t_idx, i_idx, j_idx, alt_km);
        };

        // Upper neighbor on each axis; a single-value axis uses its only
        // cell for both (the lookups return frac 0 there, so it is held).
        const std::size_t ti1 = std::min(ti + 1, epochs_.size() - 1);
        const std::size_t ii1 = std::min(ii + 1, latitude_deg_.size() - 1);
        const std::size_t ji1 = std::min(ji + 1, longitude_deg_.size() - 1);

        const Vector3d v000 = sample_cell(ti,  ii,  ji );
        const Vector3d v001 = sample_cell(ti,  ii,  ji1);
        const Vector3d v010 = sample_cell(ti,  ii1, ji );
        const Vector3d v011 = sample_cell(ti,  ii1, ji1);
        const Vector3d v100 = sample_cell(ti1, ii,  ji );
        const Vector3d v101 = sample_cell(ti1, ii,  ji1);
        const Vector3d v110 = sample_cell(ti1, ii1, ji );
        const Vector3d v111 = sample_cell(ti1, ii1, ji1);

        const Vector3d v_t0 = bilinear(v000, v001, v010, v011, i_frac, j_frac);
        const Vector3d v_t1 = bilinear(v100, v101, v110, v111, i_frac, j_frac);
        return v_t0 + t_frac * (v_t1 - v_t0);
    }

    [[nodiscard]] bool empty() const noexcept { return wind_packed_.empty(); }

    // Axes and raw storage, read by the Python save_drag_env / load_drag_env
    // (NPZ files). Storage order as described above the class.
    [[nodiscard]] const std::vector<double>& altitude_km() const noexcept { return altitude_km_; }
    [[nodiscard]] const std::vector<double>& latitude_deg() const noexcept { return latitude_deg_; }
    [[nodiscard]] const std::vector<double>& longitude_deg() const noexcept { return longitude_deg_; }
    [[nodiscard]] const std::vector<Epoch>& epochs() const noexcept { return epochs_; }
    [[nodiscard]] const std::vector<half>& encoded_packed() const noexcept { return wind_packed_; }

private:
    static double wrap_lon_to_grid(double lon_deg) noexcept {
        return detail::wrap_longitude_deg(lon_deg);
    }

    // Decode the 3 packed halves at the given cell index to a Vector3d.
    [[nodiscard]] Vector3d unpack_cell(std::size_t idx) const noexcept {
        return Vector3d(
            static_cast<double>(half_to_float(wind_packed_[3 * idx + 0])),
            static_cast<double>(half_to_float(wind_packed_[3 * idx + 1])),
            static_cast<double>(half_to_float(wind_packed_[3 * idx + 2])));
    }

    static Vector3d bilinear(const Vector3d& v00, const Vector3d& v01,
                              const Vector3d& v10, const Vector3d& v11,
                              double fi, double fj) noexcept {
        const Vector3d v_lo = v00 + fj * (v01 - v00);
        const Vector3d v_hi = v10 + fj * (v11 - v10);
        return v_lo + fi * (v_hi - v_lo);
    }

    Vector3d cell_wind_at_altitude(std::size_t ti, std::size_t ii,
                                    std::size_t ji, double alt_km) const {
        const std::size_t n_alt = altitude_km_.size();
        const std::size_t n_lat = latitude_deg_.size();
        const std::size_t n_lon = longitude_deg_.size();
        const std::size_t base = ((ti * n_lon + ji) * n_lat + ii) * n_alt;

        // Cubic spline in altitude for each component, in double precision.
        // The stored values carry 16-bit rounding (~5e-4 relative): fine for
        // the wind itself, too coarse to trust derivatives of the spline.
        std::vector<double> wx(n_alt), wy(n_alt), wz(n_alt);
        for (std::size_t k = 0; k < n_alt; ++k) {
            const Vector3d w = unpack_cell(base + k);
            wx[k] = w.x(); wy[k] = w.y(); wz[k] = w.z();
        }
        math::CubicSpline<double> sx(altitude_km_, std::move(wx));
        math::CubicSpline<double> sy(altitude_km_, std::move(wy));
        math::CubicSpline<double> sz(altitude_km_, std::move(wz));
        return Vector3d(sx(alt_km), sy(alt_km), sz(alt_km));
    }

    std::vector<double> altitude_km_;
    std::vector<double> latitude_deg_;
    std::vector<double> longitude_deg_;
    std::vector<Epoch>  epochs_;
    // Half-precision packed storage: wind_packed_[3*idx + c] is the
    // c-th component (0=x,1=y,2=z) of the wind at flat cell `idx`.
    // Encoded in the ctor from Vector3d input.
    std::vector<half>   wind_packed_;
};

}  // namespace atmosphere
}  // namespace ballistic
