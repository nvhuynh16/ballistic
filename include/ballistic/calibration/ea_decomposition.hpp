#pragma once

// =============================================================================
// EA decomposition — split a fitted correction into drag and the rest
// =============================================================================
//
// An empirical acceleration (EA) is a small correction acceleration fitted to
// observations and added to the force model. If the force model's drag is
// wrong, much of the EA points against the velocity. This file splits an EA
// into a drag-like part along −v̂ (opposite the Earth-fixed velocity) and the
// perpendicular remainder, and turns the drag part into a ballistic
// coefficient β = m / (Cd·A) [kg/m²] (mass over drag coefficient times area;
// higher β means less drag) via the drag equation a = ρ v² / (2β).
// The split is done in ECEF (Earth-fixed) axes — it is NOT a split into
// radial/along-track/cross-track or spacecraft body axes.
//
// Public entry points:
//   decompose_ea          — split one EA vector (returns EADecomposition)
//   refine_beta_from_ea   — given a drag environment, an EA and the trajectory
//                           it was fitted along, return a copy of the
//                           environment with a β profile fitted from the EA
//                           (density and wind unchanged)
//
// detail::decompose_to_beta / detail::compose_ea_from_beta are internal helpers
// (also used by tests). Estimator results supply the trajectory through their
// collocation_history.
// =============================================================================

#include "ea_result.hpp"
#include "../core/diagnostics.hpp"
#include "../forces/empirical_acceleration.hpp"
#include "../trajectory.hpp"
#include "../coordinates/geodetic.hpp"
#include "../coordinates/iau_transforms.hpp"
#include "../util/frame_transform.hpp"
#include "../atmosphere/drag_environment.hpp"
#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <span>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace ballistic {
namespace calibration {

// =============================================================================
// Single-vector EA decomposition
// =============================================================================

/// Result of `decompose_ea`.
struct EADecomposition {
    double beta = 0.0;          // ballistic coefficient [kg/m²] (0 if no drag-like part)
    Vector3d ea_drag = Vector3d::Zero();      // part along the velocity axis [m/s²]
    Vector3d ea_residual = Vector3d::Zero();  // perpendicular remainder [m/s²] (other forces)
    double drag_fraction = 0.0; // |ea_drag| / |EA|, between 0 and 1
};

/// Split EA vector `ea` [m/s²] into its component along `velocity` [m/s] and
/// the perpendicular remainder, and compute β [kg/m²] from the drag equation
/// with air density `density_kg_m3` [kg/m³]. Both vectors must be in the same
/// frame; use ECEF, where the velocity is the velocity relative to the
/// (co-rotating) air. β stays 0 when the along-velocity part does not oppose
/// the motion, the density is not positive, |v| < 1 m/s or the EA is zero.
/// Never throws.
[[nodiscard]] inline EADecomposition decompose_ea(
    const Vector3d& ea,
    const Vector3d& velocity,
    double density_kg_m3)
{
    EADecomposition d{};
    const double v_mag = velocity.norm();
    const double ea_mag = ea.norm();
    if (v_mag < 1.0 || ea_mag < 1e-20) return d;

    const Vector3d v_hat = velocity / v_mag;
    const double drag_mag = -ea.dot(v_hat);   // positive if EA opposes velocity
    d.ea_drag = -drag_mag * v_hat;            // along -v_hat
    d.ea_residual = ea - d.ea_drag;           // perpendicular by construction
    d.drag_fraction = std::abs(drag_mag) / ea_mag;

    if (drag_mag > 1e-15 && density_kg_m3 > 0.0) {
        d.beta = density_kg_m3 * v_mag * v_mag / (2.0 * drag_mag);
    }
    return d;
}

// =============================================================================
// Batch decomposition: trajectory + EA → per-point beta + binned summary
// =============================================================================

/// β estimated at one trajectory point.
struct BetaStepEstimate {
    double time_s = 0.0;        // seconds after epoch0
    double altitude_km = 0.0;   // geodetic altitude [km]
    double beta = 0.0;          // ballistic coefficient [kg/m²]
    double w = 0.0;             // 1 / beta [m²/kg]
    double weight = 1.0;        // confidence weight (currently always 1.0)
};

/// β fitted from an EA along a trajectory. Use refine_beta_from_ea(...) to get
/// a DragEnvironment with this β applied.
struct BetaResult {
    std::vector<BetaStepEstimate> step_estimates;   // one per accepted point
    double constant_beta = 0.0;                     // median of the point estimates [kg/m²]
    std::vector<double> node_altitudes_km;          // altitude grid, if one was requested
    std::vector<double> node_betas;                 // median β per grid altitude [kg/m²]
    int steps_solved = 0;                           // number of accepted points
    int steps_skipped = 0;                          // (not filled at present)
    std::string error_message;                      // non-empty on failure
};

namespace detail {
// Air density callback: (altitude km, latitude deg, longitude deg, time) → kg/m³.
// Internal; refine_beta_from_ea builds one from the drag environment.
using DensityFn = std::function<double(double, double, double, const Epoch&)>;
}  // namespace detail

/// Settings for fitting β from an EA.
struct BetaDecompositionConfig {
    double min_altitude_m = 50000.0;        // skip points below this geodetic altitude [m]
    double min_beta = 1.0;                  // discard point estimates below this [kg/m²]
    double max_beta = 1e5;                  // discard point estimates above this [kg/m²]
    std::vector<double> altitude_nodes_km;  // optional altitude grid [km]: each point goes
                                            // to the nearest node; β per node = median
};

namespace detail {

// =============================================================================
// decompose_to_beta — internal helper for refine_beta_from_ea
// =============================================================================
//
// Estimate β at each point of `trajectory` (ECEF states) from the EA. The
// trajectory must be the one the EA was applied to — propagated WITH the EA in
// the force model — so that its velocities are the ones the drag acted on.
// The split is done in ECEF; an EA stored in GCRS (inertial) is rotated to
// ECEF at each point's time. Failures are returned in error_message.
//
// Public callers use refine_beta_from_ea, which takes a DragEnvironment and
// builds the density callback itself.
[[nodiscard]] inline BetaResult decompose_to_beta(
    const forces::EmpiricalAcceleration& ea,
    std::span<const HistoryEntry> trajectory,
    const Epoch& epoch0,
    DensityFn density_fn,
    const BetaDecompositionConfig& config = {})
{
    BetaResult result;

    if (ea.is_zero()) {
        result.error_message = "EA is zero (no correction to decompose)";
        return result;
    }
    if (trajectory.empty()) {
        result.error_message = "Trajectory is empty";
        return result;
    }

    for (const auto& h : trajectory) {
        const double t_k = h.epoch - epoch0;

        const auto geo = coords::ecef_to_geodetic(h.position);
        if (geo.altitude < config.min_altitude_m) continue;

        const double alt_km = geo.altitude / 1000.0;
        const double lat_deg = geo.latitude * 180.0 / constants::PI;
        const double lon_deg = geo.longitude * 180.0 / constants::PI;

        const double rho = density_fn(alt_km, lat_deg, lon_deg, h.epoch);
        Vector3d ea_vec = ea.accel_at(t_k);
        if (ea.frame() == forces::EmpiricalAccelFrame::GCRS) {
            ea_vec = coords::gcrs_to_itrf_vector_full(ea_vec, h.epoch);
        }
        const auto d = decompose_ea(ea_vec, h.velocity, rho);

        if (d.beta < config.min_beta || d.beta > config.max_beta) continue;

        BetaStepEstimate est;
        est.time_s = t_k;
        est.altitude_km = alt_km;
        est.beta = d.beta;
        est.w = 1.0 / d.beta;
        est.weight = 1.0;
        result.step_estimates.push_back(est);
    }

    result.steps_solved = static_cast<int>(result.step_estimates.size());

    if (result.steps_solved == 0) {
        result.error_message = "No valid beta estimates";
        return result;
    }

    // Constant β = median of the point estimates
    std::vector<double> betas;
    betas.reserve(result.steps_solved);
    for (const auto& est : result.step_estimates)
        betas.push_back(est.beta);
    std::sort(betas.begin(), betas.end());
    result.constant_beta = betas[betas.size() / 2];

    // β per altitude node (if nodes were given); an empty node gets the median
    if (!config.altitude_nodes_km.empty()) {
        const auto& nodes = config.altitude_nodes_km;
        const std::size_t n_nodes = nodes.size();

        std::vector<std::vector<double>> bins(n_nodes);
        for (const auto& est : result.step_estimates) {
            std::size_t best = 0;
            double best_dist = std::abs(est.altitude_km - nodes[0]);
            for (std::size_t j = 1; j < n_nodes; ++j) {
                const double dd = std::abs(est.altitude_km - nodes[j]);
                if (dd < best_dist) { best_dist = dd; best = j; }
            }
            bins[best].push_back(est.beta);
        }

        result.node_altitudes_km = nodes;
        result.node_betas.resize(n_nodes);
        for (std::size_t j = 0; j < n_nodes; ++j) {
            if (bins[j].empty()) {
                result.node_betas[j] = result.constant_beta;
            } else {
                std::sort(bins[j].begin(), bins[j].end());
                result.node_betas[j] = bins[j][bins[j].size() / 2];
            }
        }

    }

    return result;
}

// =============================================================================
// decompose_to_beta — convenience overload
// =============================================================================
//
// Same, taking an estimator result: uses its EA and its collocation_history
// (the trajectory that goes with the EA; see ea_estimator.hpp). Returns an
// error result if the result has no non-zero EA or no history (then call the
// overload above with your own trajectory).
[[nodiscard]] inline BetaResult decompose_to_beta(
    const EAResult& ea_result,
    const Epoch& epoch0,
    DensityFn density_fn,
    const BetaDecompositionConfig& config = {})
{
    BetaResult result;
    if (!ea_result.has_correction()) {
        result.error_message =
            "estimator result not ok: "
            + (ea_result.error_message.empty()
                 ? std::string("EA missing or zero")
                 : ea_result.error_message);
        return result;
    }
    if (ea_result.collocation_history.empty()) {
        // The public estimators fill collocation_history; an empty one means
        // the result was built by hand.
        result.error_message =
            "estimator result has no collocation_history; "
            "call the primary decompose_to_beta(ea, trajectory, ...) overload "
            "with an explicit trajectory";
        return result;
    }
    return decompose_to_beta(*ea_result.empirical_accel,
                             ea_result.collocation_history,
                             epoch0, std::move(density_fn), config);
}

// =============================================================================
// compose_ea_from_beta — inverse of decompose_to_beta
// =============================================================================
//
// Given β as a function of altitude and the air density along a known
// trajectory, build the drag acceleration (along −v, ECEF velocity) at each
// trajectory point as a tabulated EA, stored in `frame`. Points with |v| < 1
// m/s, β ≤ 0 or ρ ≤ 0 get zero. `BetaFn` maps altitude [km] to β [kg/m²]: a
// constant (`[=](double){ return beta; }`) or a table lookup
// (`[&](double alt_km){ return table->get(alt_km); }`).
using BetaFn = std::function<double(double)>;

[[nodiscard]] inline std::shared_ptr<forces::EmpiricalAcceleration> compose_ea_from_beta(
    std::span<const HistoryEntry> trajectory,
    const Epoch& epoch0,
    BetaFn beta_fn,
    DensityFn density_fn,
    forces::EmpiricalAccelFrame frame = forces::EmpiricalAccelFrame::GCRS)
{
    std::vector<forces::EmpiricalAccelEntry> entries;
    entries.reserve(trajectory.size());

    for (const auto& h : trajectory) {
        const double t_k = h.epoch - epoch0;

        const auto geo = coords::ecef_to_geodetic(h.position);
        const double alt_km = geo.altitude / 1000.0;
        const double lat_deg = geo.latitude * 180.0 / constants::PI;
        const double lon_deg = geo.longitude * 180.0 / constants::PI;

        const double beta = beta_fn(alt_km);
        const double rho = density_fn(alt_km, lat_deg, lon_deg, h.epoch);
        const double v_mag = h.velocity.norm();

        if (v_mag < 1.0 || beta <= 0.0 || rho <= 0.0) {
            entries.emplace_back(t_k, Vector3d::Zero());
            continue;
        }

        const double drag_mag = rho * v_mag * v_mag / (2.0 * beta);
        const Vector3d v_hat = h.velocity / v_mag;
        const Vector3d ea_ecef = -drag_mag * v_hat;

        if (frame == forces::EmpiricalAccelFrame::GCRS) {
            entries.emplace_back(t_k, coords::itrf_to_gcrs_vector_full(ea_ecef, h.epoch));
        } else {
            entries.emplace_back(t_k, ea_ecef);
        }
    }

    return std::make_shared<forces::EmpiricalAcceleration>(
        forces::EmpiricalAcceleration::from_entries(std::move(entries), frame));
}

}  // namespace detail

// =============================================================================
// refine_beta_from_ea — DragEnvironment-based calibration loop closure
// =============================================================================
//
/// Fit β from an EA and return a copy of `prior` with its β replaced.
///
/// `prior` supplies the air density (its wind is kept unchanged); `ea` is the
/// fitted EA; `trajectory` (ECEF states) must be the orbit propagated WITH the
/// EA, e.g. an estimator result's collocation_history; `epoch0` is the time EA
/// time 0 refers to. The EA is split along the velocity at each point and the
/// drag part gives β = ρ v² / (2·|drag part|).
///
/// The new β is an altitude-only BetaProfile1D (median β per altitude node).
/// The nodes are `base_config.altitude_nodes_km` if given, else the altitude
/// grid of a tabulated 1D/2D density; for a 4D density field or a live
/// density model, a fixed 200–1000 km grid in 100 km steps. If no node can be
/// filled, the median over all points is used at every node. For β that varies
/// in time (mass loss, attitude changes), call this once per time segment.
///
/// If no valid β can be extracted at all (zero EA, empty trajectory, or no
/// point where the EA opposes the motion with β inside [min_beta, max_beta]),
/// `prior` is returned unchanged and a ModelSubstitution diagnostic says why
/// (an error in strict mode). `fit_report`, when given, receives the fit
/// itself: the point estimates, `steps_solved` and `error_message` (non-empty
/// exactly when the prior was kept). Exceptions from the density evaluation
/// (e.g. a time outside the loaded space weather) propagate.
[[nodiscard]] inline atmosphere::DragEnvironment refine_beta_from_ea(
    const atmosphere::DragEnvironment& prior,
    const forces::EmpiricalAcceleration& ea,
    std::span<const HistoryEntry> trajectory,
    const Epoch& epoch0,
    const BetaDecompositionConfig& base_config = {},
    BetaResult* fit_report = nullptr)
{
    // Altitude grid of the prior's density column, so the new β has the same
    // altitude resolution.
    auto altitude_grid_from = [](const atmosphere::DragEnvironment::DensityVariant& v) {
        return std::visit([](const auto& p) -> std::vector<double> {
            using T = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<T, atmosphere::DensityProfile1D>) {
                return p.altitude_km();
            } else if constexpr (std::is_same_v<T, atmosphere::DensityProfile2D>) {
                return p.altitude_km();
            } else {
                // 4D field or live model — no altitude grid to borrow; use a
                // fixed 200–1000 km grid.
                std::vector<double> grid;
                for (double h = 200.0; h <= 1000.0; h += 100.0) grid.push_back(h);
                return grid;
            }
        }, v);
    };

    // Density callback for decompose_to_beta, evaluating the prior.
    auto density_callback = [&prior](double alt_km, double lat_deg,
                                       double lon_deg, const Epoch& t) -> double {
        constexpr double DEG_TO_RAD = 0.017453292519943295;
        const Vector3d pos_ecef = coords::geodetic_to_ecef(
            lat_deg * DEG_TO_RAD, lon_deg * DEG_TO_RAD, alt_km * 1000.0);
        return prior.evaluate(pos_ecef, t).rho_kg_m3;
    };

    BetaDecompositionConfig cfg = base_config;
    if (cfg.altitude_nodes_km.empty()) {
        cfg.altitude_nodes_km = altitude_grid_from(prior.density());
    }

    BetaResult result = detail::decompose_to_beta(
        ea, trajectory, epoch0, density_callback, cfg);

    // Build the new BetaProfile1D from the per-node medians.
    std::vector<double> beta_kg_m2;
    std::vector<double> altitude_km;
    if (!result.node_betas.empty() && result.node_betas.size() == result.node_altitudes_km.size()) {
        altitude_km = result.node_altitudes_km;
        beta_kg_m2 = result.node_betas;
    } else if (result.constant_beta > 0.0) {
        // No per-node β: use the overall median at every node.
        altitude_km = altitude_grid_from(prior.density());
        beta_kg_m2.assign(altitude_km.size(), result.constant_beta);
    } else {
        // No valid β at all: keep the prior, and say so (see the doc comment).
        if (result.error_message.empty())
            result.error_message = "No valid beta estimates";
        std::ostringstream msg;
        msg << "refine_beta_from_ea: no ballistic coefficient could be fitted ("
            << result.error_message << ", " << trajectory.size()
            << " trajectory points), so the prior beta is returned unchanged. A "
               "drag part is found only where the EA opposes the Earth-fixed "
               "velocity and gives beta in [" << base_config.min_beta << ", "
            << base_config.max_beta << "] kg/m^2 above "
            << base_config.min_altitude_m / 1000.0 << " km: pass the trajectory "
               "propagated with this EA (the estimator's collocation_history) and "
               "check that the EA is non-zero.";
        diagnostics::report(diagnostics::Category::ModelSubstitution, msg.str());
        if (fit_report) *fit_report = std::move(result);
        return prior;
    }
    if (fit_report) *fit_report = result;

    atmosphere::DragEnvironment refined = prior;
    refined.set_beta(atmosphere::BetaProfile1D(std::move(altitude_km),
                                                 std::move(beta_kg_m2)));
    return refined;
}

}  // namespace calibration
}  // namespace ballistic
