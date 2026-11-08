#pragma once

// =============================================================================
// DragEnvironment — everything the drag force needs at a point: air density ρ,
// wind and the satellite's ballistic coefficient β
// =============================================================================
//
// Terms used in this file:
//   - density ρ: mass of air per volume [kg/m³].
//   - wind: the air's own velocity relative to the rotating Earth [m/s], in
//     ECEF (Earth-centered, Earth-fixed axes that rotate with the Earth).
//   - ballistic coefficient β = m / (Cd · A) [kg/m²]: mass over drag
//     coefficient (Cd, dimensionless) times reference area. Higher β means
//     the satellite is slowed less by the same air.
//
// The environment holds three "columns", one per quantity, each of which can
// be one of several representations (a std::variant): a table over altitude,
// over altitude and time, over a full 4D grid, or the atmosphere models
// themselves evaluated on demand (live_model.hpp, what make_drag_environment
// returns and the default). Builders live in drag_environment_generators.hpp;
// tests may build one by hand, and refine_beta_from_ea replaces the β column
// with a value fitted to tracking data.
//
// evaluate(position_ecef, epoch) returns a DragSample {rho, wind, beta, ...}
// that the drag acceleration formula consumes directly:
//
//     a_drag = -½ · ρ · |v_rel|² · v_rel_hat / β
//     v_rel  = v_sat - Ω × r - wind        (velocity relative to the air)
//
// where Ω is the Earth's rotation vector and v_rel_hat the unit vector of v_rel.
// =============================================================================

#include "../core/types.hpp"
#include "../time/epoch.hpp"
#include "density_profile.hpp"
#include "wind_profile.hpp"
#include "beta_profile.hpp"
#include "space_weather_history.hpp"  // TimeWindow
#include "live_model.hpp"
#include "../aero/flow_regime.hpp"
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <variant>
#include <utility>

namespace ballistic {
namespace atmosphere {

/// The drag environment at one position and time, as returned by
/// DragEnvironment::evaluate(). A zero-filled sample means "no drag".
struct DragSample {
    // Air density [kg/m³], wind velocity in ECEF axes [m/s], and ballistic
    // coefficient m/(Cd·A) [kg/m²].
    double   rho_kg_m3   = 0.0;
    Vector3d wind_ecef_m_s = Vector3d::Zero();
    double   beta_kg_m2  = 0.0;
    bool     time_clamped = false;  // Epoch fell outside the env's time domain;
                                    // a tabulated column returned its first/last
                                    // time slice (the numbers are still returned,
                                    // but the atmosphere is frozen there)
    // The air's state, where the density column provides it (the live model
    // does; tables give NaN). Kelvin and kg per kilomole.
    double   temperature_k = std::numeric_limits<double>::quiet_NaN();
    double   molar_mass_kg_kmol = std::numeric_limits<double>::quiet_NaN();
    // The flow, where the beta column is an AeroBeta (NaN otherwise): the drag
    // coefficient Cd used, and — only for a FlowRegimeCd — the Knudsen number
    // (mean free path of the air molecules over body size) and the Mach number.
    double   cd = std::numeric_limits<double>::quiet_NaN();
    double   knudsen = std::numeric_limits<double>::quiet_NaN();
    double   mach = std::numeric_limits<double>::quiet_NaN();
};

/// Density, wind and β for the drag force, each as an interchangeable column
/// (table or live model). Immutable once built in normal use; a Trajectory
/// holds it through TrajectoryConfig::drag_env (null there = drag off).
class DragEnvironment {
public:
    using DensityVariant = std::variant<DensityProfile1D, DensityProfile2D, DensityField4D,
                                        LiveModelDensity>;
    using WindVariant    = std::variant<WindProfile1D_ENU, WindProfile2D_ECEF, WindField4D_ECEF,
                                        LiveModelWind>;
    // AeroBeta: beta = m / (Cd A) with Cd computed from the flow around the
    // body (aero/flow_regime.hpp; FlowRegimeCd for reentry).
    using BetaVariant    = std::variant<BetaProfile1D, BetaProfile2D, aero::AeroBeta>;

    /// Empty environment: has_drag() is false and evaluate() returns a
    /// zero-filled DragSample — the same as passing no environment at all.
    DragEnvironment() = default;

    /// Build from the three columns. Throws ConfigError if `beta` is an
    /// AeroBeta with invalid settings (see aero::AeroBeta::validate).
    DragEnvironment(DensityVariant density, WindVariant wind, BetaVariant beta)
        : density_(std::move(density))
        , wind_(std::move(wind))
        , beta_(std::move(beta))
    {
        if (const auto* a = std::get_if<aero::AeroBeta>(&beta_)) a->validate();
    }

    /// Density, wind and β at `position_ecef` [m, ECEF] and `epoch`.
    /// `velocity_ecef` [m/s, Earth-fixed] is needed only when the β column is
    /// an AeroBeta with a FlowRegimeCd (its Cd depends on the speed through
    /// the air); without it that case throws ConfigError. Column errors pass
    /// through: below a table's lowest altitude, outside a live model's
    /// weather, or below the ground (see the column types).
    [[nodiscard]] DragSample evaluate(const Vector3d& position_ecef,
                                       const Epoch& epoch,
                                       const std::optional<Vector3d>& velocity_ecef = std::nullopt) const {
        DragSample s;
        std::visit(
            [&](const auto& p) {
                if constexpr (requires { p.evaluate_state(position_ecef, epoch); }) {
                    const auto a = p.evaluate_state(position_ecef, epoch);
                    s.rho_kg_m3 = a.density_kg_m3;
                    s.temperature_k = a.temperature_k;
                    s.molar_mass_kg_kmol = a.molar_mass_kg_kmol;
                } else {
                    s.rho_kg_m3 = p.evaluate(position_ecef, epoch);
                }
            },
            density_);
        s.wind_ecef_m_s = std::visit(
            [&](const auto& p) { return p.evaluate(position_ecef, epoch); },
            wind_);
        std::visit(
            [&](const auto& p) {
                if constexpr (std::is_same_v<std::decay_t<decltype(p)>, aero::AeroBeta>) {
                    if (p.empty()) return;
                    aero::FlowState f;
                    f.density_kg_m3 = s.rho_kg_m3;
                    f.temperature_k = s.temperature_k;
                    f.molar_mass_kg_kmol = s.molar_mass_kg_kmol;
                    if (p.depends_on_flow()) {
                        if (!velocity_ecef) {
                            throw ConfigError(
                                "DragEnvironment::evaluate: a FlowRegimeCd beta depends on the speed "
                                "through the air; pass the Earth-fixed velocity");
                        }
                        f.speed_m_s = (*velocity_ecef - s.wind_ecef_m_s).norm();
                        if (!(s.rho_kg_m3 > 0.0)) return;   // no air: no drag, no Cd
                    }
                    const auto [beta, r] = p.evaluate(f);
                    s.beta_kg_m2 = beta;
                    s.cd = r.cd;
                    s.knudsen = r.knudsen;
                    s.mach = r.mach;
                } else {
                    s.beta_kg_m2 = p.evaluate(position_ecef, epoch);
                }
            },
            beta_);
        if (auto w = domain()) {
            s.time_clamped = epoch < w->start || epoch > w->end;
        }
        return s;
    }

    /// Time span over which every TABULATED time-varying column has data: the
    /// overlap of the epoch ranges of the 2D/4D density, wind and beta tables.
    /// nullopt when no column is a time-varying table (1D profiles do not
    /// depend on time; live-model columns are limited by their weather
    /// instead, see live_source()). Outside this span a table returns its
    /// first/last time slice — the atmosphere freezes — and
    /// DragSample::time_clamped is set.
    [[nodiscard]] std::optional<TimeWindow> domain() const {
        std::optional<TimeWindow> w;
        accumulate_epoch_range(density_, w);
        accumulate_epoch_range(wind_, w);
        accumulate_epoch_range(beta_, w);
        return w;
    }

    /// True when both the density and the beta columns are set. Wind may be
    /// empty (the air then co-rotates with the Earth).
    [[nodiscard]] bool has_drag() const noexcept {
        const bool rho_set = std::visit(
            [](const auto& p) { return !p.empty(); }, density_);
        const bool beta_set = std::visit(
            [](const auto& p) { return !p.empty(); }, beta_);
        return rho_set && beta_set;
    }

    /// Read access to the three columns (used by the Python bindings and
    /// refine_beta_from_ea).
    [[nodiscard]] const DensityVariant& density() const noexcept { return density_; }
    [[nodiscard]] const WindVariant&    wind()    const noexcept { return wind_; }
    [[nodiscard]] const BetaVariant&    beta()    const noexcept { return beta_; }

    /// Replace one column (e.g. calibration keeps density and wind and swaps
    /// in a fitted β). set_beta throws ConfigError for an invalid AeroBeta.
    void set_density(DensityVariant d) { density_ = std::move(d); }
    void set_wind(WindVariant w)       { wind_    = std::move(w); }
    void set_beta(BetaVariant b) {
        if (const auto* a = std::get_if<aero::AeroBeta>(&b)) a->validate();
        beta_ = std::move(b);
    }

    /// True when β depends on the velocity (an AeroBeta with a FlowRegimeCd).
    /// The drag force's derivative with respect to velocity (its Jacobian) is
    /// then computed by finite differences instead of a formula.
    [[nodiscard]] bool beta_depends_on_velocity() const noexcept {
        const auto* a = std::get_if<aero::AeroBeta>(&beta_);
        return a && !a->empty() && a->depends_on_flow();
    }

    /// Copy of this environment with no wind (the air co-rotates with the
    /// Earth); density and beta unchanged. Useful to measure what the wind
    /// contributes by comparing runs with and without it.
    [[nodiscard]] DragEnvironment without_wind() const {
        DragEnvironment copy = *this;
        copy.wind_ = WindProfile1D_ENU{};  // default ctor → empty, evaluates to zero
        return copy;
    }

    // ---- Live model (make_drag_environment) --------------------------------

    /// The space-weather source shared by the live columns, or nullptr when no
    /// column is live. Its `valid` window is where the environment can be
    /// evaluated (a Trajectory stops at the edge) and `measured_until` is
    /// where forecast weather begins.
    [[nodiscard]] const LiveModelSource* live_source() const noexcept {
        if (const auto* d = std::get_if<LiveModelDensity>(&density_); d && !d->empty())
            return d->source().get();
        if (const auto* w = std::get_if<LiveModelWind>(&wind_); w && !w->empty())
            return w->source().get();
        return nullptr;
    }

    /// True when the environment can be evaluated past the last day of
    /// measured space weather (it then runs on CelesTrak's forecast).
    [[nodiscard]] bool uses_predicted_weather() const noexcept {
        const auto* src = live_source();
        return src && src->measured_until.has_value();
    }

private:
    // Narrow the window to a time-varying column's epoch range (intersection).
    template <typename Variant>
    static void accumulate_epoch_range(const Variant& v,
                                       std::optional<TimeWindow>& window) {
        std::visit([&](const auto& p) {
            if constexpr (requires { p.epochs(); }) {
                const auto& eps = p.epochs();
                if (eps.size() >= 2) {
                    if (!window) {
                        window = TimeWindow{eps.front(), eps.back()};
                    } else {
                        if (eps.front() > window->start) window->start = eps.front();
                        if (eps.back() < window->end) window->end = eps.back();
                    }
                }
            }
        }, v);
    }

    DensityVariant density_;
    WindVariant    wind_;
    BetaVariant    beta_;
};

}  // namespace atmosphere
}  // namespace ballistic
