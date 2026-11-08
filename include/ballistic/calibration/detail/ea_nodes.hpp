#pragma once

// =============================================================================
// Empirical acceleration (EA) as values at evenly spaced nodes — internal detail
// =============================================================================
//
// An EA (a small correction acceleration fitted to observations and added to the force
// model) described by its value in the inertial GCRS frame at uniformly spaced node times,
// interpolated linearly between nodes. The parameter vector θ holds all x values, then all
// y, then all z (`param_idx(comp, node, n_nodes)`, calibration_constants.hpp). Used by the
// information bound (ea_information_bound.hpp) together with the state-transition-matrix
// Jacobian of gn_stm.hpp.
// =============================================================================

#include "calibration_constants.hpp"
#include "../../core/types.hpp"
#include "../../forces/empirical_acceleration.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace ballistic {
namespace calibration {
namespace detail {

/// Node times [s from the arc start] 0, h, 2h, ... not beyond `duration_s − h`, with
/// h = `knot_spacing_s`. Always at least one node (t = 0).
[[nodiscard]] inline std::vector<double> build_ea_keys(
    double duration_s, double knot_spacing_s)
{
    std::vector<double> keys;
    double t_last = std::max(0.0, duration_s - knot_spacing_s);
    int n = static_cast<int>(std::floor(t_last / knot_spacing_s)) + 1;
    keys.reserve(n);
    for (int k = 0; k < n; ++k) {
        keys.push_back(k * knot_spacing_s);
    }
    return keys;
}

/// Build the EA described by θ (3·n_nodes values, x block then y then z, m/s²) at node times
/// `t_nodes` [s]: a GCRS table, linearly interpolated between nodes.
[[nodiscard]] inline std::shared_ptr<forces::EmpiricalAcceleration>
ea_from_theta(const Eigen::VectorXd& theta, const std::vector<double>& t_nodes)
{
    using ballistic::calibration::detail::param_idx;
    const int n_nodes = static_cast<int>(t_nodes.size());
    std::vector<forces::EmpiricalAccelEntry> entries;
    entries.reserve(n_nodes);
    for (int k = 0; k < n_nodes; ++k) {
        Vector3d a;
        a << theta(param_idx(0, k, n_nodes)),
             theta(param_idx(1, k, n_nodes)),
             theta(param_idx(2, k, n_nodes));
        entries.emplace_back(t_nodes[k], a);
    }
    return std::make_shared<forces::EmpiricalAcceleration>(
        forces::EmpiricalAcceleration::from_entries(
            std::move(entries),
            forces::EmpiricalAccelFrame::GCRS,
            forces::EAInterp::Linear));
}

/// Smoothness penalty matrix LᵀL for θ, where L takes first differences between neighboring
/// nodes divided by the node spacing `dt` [s], per component, in θ's layout. θᵀ·LᵀL·θ is then
/// the discrete ‖dEA/dt‖², which penalizes rapid changes of the EA. Dense (3·n_nodes square).

[[nodiscard]] inline Eigen::MatrixXd
build_first_difference_LtL(int n_nodes, double dt)
{
    const int n_total = 3 * n_nodes;
    Eigen::MatrixXd LtL = Eigen::MatrixXd::Zero(n_total, n_total);
    const double s2 = 1.0 / (dt * dt);
    for (int c = 0; c < 3; ++c) {
        const int base = c * n_nodes;
        // For the [-1 +1] difference: tridiagonal, diagonal 2/dt² (1/dt² at the two ends),
        // off-diagonal -1/dt².
        for (int k = 0; k < n_nodes; ++k) {
            double diag = 2.0;
            if (k == 0 || k == n_nodes - 1) diag = 1.0;
            LtL(base + k, base + k) = diag * s2;
            if (k + 1 < n_nodes) {
                LtL(base + k, base + k + 1) = -1.0 * s2;
                LtL(base + k + 1, base + k) = -1.0 * s2;
            }
        }
    }
    return LtL;
}

}  // namespace detail
}  // namespace calibration
}  // namespace ballistic
