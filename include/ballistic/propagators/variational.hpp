#pragma once

// =============================================================================
// Variational matrix A(t) = d(xdot)/dx of the equations of motion
// =============================================================================
//
// For the state x = [r; v] (position [m], velocity [m/s], both in the inertial
// GCRS frame) the equations of motion are xdot = f(x, t) = [v; a(r, v, t)].
// Their linearization about a reference state is the 6x6 matrix
//
//   A = df/dx = [ 0        I      ]
//               [ da/dr    da/dv  ]
//
// with da/dr in 1/s^2 and da/dv in 1/s. A drives the state transition matrix
// (dPhi/dt = A Phi) and every sensitivity (variational-equation) integration in
// the library. The acceleration Jacobian comes from
// ForceModel::acceleration_with_jacobian: analytic for gravity, third bodies,
// tides, relativity and radiation pressure, and by finite differences where no
// analytic form exists (drag's position columns, with offsets fd_dx [m] and
// fd_dv [m/s]). The empirical acceleration depends only on time and adds zero.
// =============================================================================

#include "../core/types.hpp"
#include "../forces/force_model.hpp"
#include "../time/epoch.hpp"

#include <Eigen/Dense>

namespace ballistic {
namespace propagators {
namespace detail {

/// A = d[v; a]/d[r; v] at `state_gcrs` (GCRS, m and m/s) and `epoch`; see the
/// file comment. `fd_dx` [m] and `fd_dv` [m/s] are the finite-difference offsets
/// for force terms without an analytic Jacobian.
[[nodiscard]] inline Eigen::Matrix<double, 6, 6> variational_matrix(
    const forces::ForceModel& force_model,
    const Epoch& epoch,
    const StateVector& state_gcrs,
    double fd_dx,
    double fd_dv)
{
    const auto jac = force_model.acceleration_with_jacobian(epoch, state_gcrs, fd_dx, fd_dv);
    Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
    A.block<3, 3>(0, 3).setIdentity();  // d(dr/dt)/dv = I
    A.block<3, 6>(3, 0) = jac.second;   // d(dv/dt)/d[r, v]
    return A;
}

}  // namespace detail
}  // namespace propagators
}  // namespace ballistic
