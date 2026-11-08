#pragma once

// =============================================================================
// Live model columns — the atmosphere models evaluated at the satellite's
// exact position on every force evaluation
// =============================================================================
//
// NRLMSIS 2.1 (empirical model of the neutral atmosphere's density,
// temperature and composition) and HWM14 (Horizontal Wind Model 2014), driven
// by space weather — daily solar radio flux F10.7 and geomagnetic index Ap —
// loaded once for the whole file. This is what make_drag_environment returns.
//
// Why not a table: the tables (density_profile.hpp / wind_profile.hpp) sample
// the models along a predicted ground track over a fixed window, which ties
// them to one trajectory and one duration and takes several propagations to
// build. Evaluating the models directly costs far less than building an
// iterated table, follows whatever trajectory is flown, and is limited only by
// the span of the space weather.
//
// The models themselves are not part of the library: they run in the
// ballistic_nrl plugin, loaded at run time (nrl_models.hpp). Constructing a
// live column raises ConfigError when the plugin is not installed.
// =============================================================================

#include "../core/errors.hpp"
#include "../core/types.hpp"
#include "../time/epoch.hpp"
#include "nrl_models.hpp"
#include "space_weather_history.hpp"
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace ballistic {
namespace atmosphere {

/// Space weather and settings shared by the density and wind columns of one
/// environment.
struct LiveModelSource {
    SpaceWeatherHistory weather;          // flat (from_constant) or time-varying
    std::optional<TimeWindow> valid;      // where `weather` can be used; nullopt = any epoch
    std::optional<Epoch> measured_until;  // forecast weather after this; nullopt = none
    bool loaded_weather = false;          // weather is the loaded file (vs passed explicitly)
    bool allow_predicted = true;          // how the loaded file was read
    double isa_floor_km = 0.0;            // > 0: USSA76 below this (extend_density_below); 0 = NRLMSIS to the ground
};

/// Make sure the NRLMSIS 2.1 / HWM14 plugin is loaded (nrl_models.hpp), else
/// raise ConfigError naming `caller` and saying how to install it. Called by
/// every builder that evaluates the models.
inline void require_nrl_models(std::string_view caller) { nrl::require(caller); }

/// Density [kg/m³], temperature [K] and mean molar mass [kg/kmol] of the air
/// at one point.
struct AtmosphereState {
    double density_kg_m3 = 0.0;
    double temperature_k = 0.0;
    double molar_mass_kg_kmol = 0.0;
};

/// Density column that runs NRLMSIS 2.1 at the query point (or the 1976
/// standard atmosphere below an extend_density_below floor). Default-
/// constructed = empty (no drag).
class LiveModelDensity {
public:
    LiveModelDensity() = default;
    explicit LiveModelDensity(std::shared_ptr<const LiveModelSource> src) : src_(std::move(src)) {
        require_nrl_models("live NRLMSIS 2.1 density column");
    }

    /// Total mass density [kg/m³] at `position_ecef` [m, ECEF] and `epoch`.
    /// Throws std::out_of_range outside the weather's valid window (a
    /// Trajectory stops at the edge before that) and DataCoverageError more
    /// than 1 km below the ground (the trajectory passed through the Earth).
    [[nodiscard]] double evaluate(const Vector3d& position_ecef, const Epoch& epoch) const {
        return evaluate_state(position_ecef, epoch).density_kg_m3;
    }

    /// Density with the temperature and mean molar mass from the same model
    /// call (NRLMSIS 2.1, or the U.S. Standard Atmosphere 1976 below an
    /// extend_density_below floor) — what a FlowRegimeCd needs. Same throws as
    /// evaluate().
    [[nodiscard]] AtmosphereState evaluate_state(const Vector3d& position_ecef, const Epoch& epoch) const;

    [[nodiscard]] bool empty() const noexcept { return src_ == nullptr; }
    [[nodiscard]] const std::shared_ptr<const LiveModelSource>& source() const noexcept { return src_; }

private:
    std::shared_ptr<const LiveModelSource> src_;
};

/// Wind column that runs HWM14 at the query point. Default-constructed =
/// empty (no wind).
class LiveModelWind {
public:
    LiveModelWind() = default;
    explicit LiveModelWind(std::shared_ptr<const LiveModelSource> src) : src_(std::move(src)) {
        require_nrl_models("live HWM14 wind column");
    }

    /// HWM14 horizontal wind at `position_ecef` [m, ECEF] and `epoch`, in
    /// ECEF axes [m/s]. Between the ground and 1 km below it the 0 km wind is
    /// used; deeper throws DataCoverageError. Throws std::out_of_range
    /// outside the weather's valid window.
    [[nodiscard]] Vector3d evaluate(const Vector3d& position_ecef, const Epoch& epoch) const;

    [[nodiscard]] bool empty() const noexcept { return src_ == nullptr; }
    [[nodiscard]] const std::shared_ptr<const LiveModelSource>& source() const noexcept { return src_; }

private:
    std::shared_ptr<const LiveModelSource> src_;
};

}  // namespace atmosphere
}  // namespace ballistic
