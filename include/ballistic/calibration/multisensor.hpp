#pragma once

// =============================================================================
// Multi-sensor observations — sensor types and the scenario that holds them
// =============================================================================
//
// The observation types used by `estimate_ea_smoother` (ea_smoother.hpp), which
// fits an empirical acceleration (EA) — a small correction acceleration added
// to the force model — to several kinds of sensor data at once, and by
// `estimate_ea` / `information_bound`. Four sensors are supported: a ground
// radar, a ground camera ("optics", angles only), a position fix (e.g. GPS) and
// an onboard accelerometer. Each carries its noise level, an optional declared
// error model (a known bias, a per-sample scaling of the noise, how its clock
// is treated) and its observation model: `predict` (what the sensor would read
// for a given satellite state) and `residual_and_jacobian`.
// `MultiSensorScenario` bundles the start state, arc length and sensors;
// `SensorMask` selects which sensors an estimate uses.
//
// Frames (ECEF = Earth-centered, Earth-fixed; here the ITRF, the international
// terrestrial reference frame, ITRF2020 by default; GCRS = the Earth-centered
// inertial frame the propagator integrates in):
//   - Sensor positions and the start state (MultiSensorScenario::pos0_ecef /
//     vel0_ecef, m and m/s): ECEF.
//   - Sample times are `Epoch` values (an absolute instant; build them with
//     Epoch::from_utc or another named time scale).
//   - AccelObs samples are specific force (the non-gravitational acceleration
//     an accelerometer measures) in ECEF, m/s². If your sensor reports in the
//     spacecraft body frame, rotate each sample to ECEF with the attitude at
//     that time before storing it.
//   - The estimated EA (EASmootherResult::empirical_accel) is in GCRS.
// =============================================================================

#include <ballistic/core/types.hpp>
#include <ballistic/core/constants.hpp>
#include <ballistic/time/epoch.hpp>
#include <ballistic/forces/empirical_acceleration.hpp>
#include <ballistic/calibration/ea_result_core.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ballistic {
namespace calibration {

/// Difference a − b of two angles in radians, wrapped to (−π, π], so that
/// e.g. azimuths 359° and 1° differ by 2°, not 358°.
[[nodiscard]] inline double angle_diff(double a, double b) {
    double d = a - b;
    return std::atan2(std::sin(d), std::cos(d));
}

// =============================================================================
// Sensor types — settings, samples and observation models.
//
// Conventions:
//   `predict(x)` returns the predicted observation h(x) for satellite state x.
//   The residual is r = observed − h(x).
//   J_state = ∂h/∂x, the derivative of the PREDICTION (not of the residual), so
//   a state correction Δx solves J·Δx ≈ r. Its 6 columns are the ECEF position
//   (m) and velocity (m/s) components, in that order.
//   `residual_and_jacobian(i, ...)` computes both for sample i (no bounds check).
// =============================================================================

/// How the smoother treats a sensor's time tags (its clock). A clock offset
/// makes every sample look as if taken slightly earlier or later; the smoother
/// can estimate such an offset when EASmootherConfig::estimate_clock_bias is on.
/// Each sensor then declares:
///   None   — its time tags are accurate; no clock offset is estimated for it.
///   Shared — one offset shared by every sensor marked Shared (the radar
///            default: all radars on one clock).
///   Own    — this sensor gets its own offset.
enum class SensorClock { None, Shared, Own };

/// Ground radar: measures range (m), azimuth and elevation (rad) and range
/// rate (m/s, the rate at which the distance changes) to the satellite.
/// Each sample is [range, azimuth, elevation, range rate]; azimuth is
/// measured from north toward east.
struct RadarObs {
    Vector3d position_ecef = Vector3d::Zero();     // radar location, ECEF [m]
    double sigma_range = 2.0;                      // range noise std dev [m]
    double sigma_angle = 0.02 * constants::DEG_TO_RAD;  // angle noise std dev [rad]
    double sigma_rate  = 0.02;   // range-rate noise std dev [m/s] — moderate-precision radar
    double lat_rad = 0.0;        // radar latitude [rad] (informational; not used by the model)
    double lon_rad = 0.0;        // radar longitude [rad] (informational; not used by the model)
    /// Rotation from ECEF to the radar's local east-north-up (ENU) axes. Must
    /// be set to match the location, e.g. coords::ecef_to_enu_rotation(lat, lon);
    /// the identity default is only a placeholder.
    Matrix3d ecef_to_enu = Matrix3d::Identity();
    /// Known measurement bias [range m, az rad, el rad, rate m/s] —
    /// subtracted from every observation before estimation (a known
    /// bias is data, not a state). Default zero.
    Eigen::Vector4d known_bias = Eigen::Vector4d::Zero();
    /// Optional per-sample multiplier of the noise std devs above (one entry
    /// per sample; empty = the std devs apply unchanged to every sample).
    std::vector<double> sigma_scale;
    SensorClock clock = SensorClock::Shared;
    /// Observations: (time, [range m, az rad, el rad, range rate m/s]).
    std::vector<std::pair<Epoch, Eigen::Vector4d>> samples;

    /// Predicted [range, azimuth, elevation, range rate] for a satellite state
    /// in ECEF (m, m/s). Returns zeros if the satellite is within 1 m of the radar.
    [[nodiscard]] Eigen::Vector4d predict(const StateVector& sat_ecef) const {
        Vector3d d = sat_ecef.position - position_ecef;
        double range = d.norm();
        if (range < 1.0) return Eigen::Vector4d::Zero();
        Vector3d d_hat = d / range;
        Vector3d d_enu = ecef_to_enu * d;
        double E = d_enu.x(), N = d_enu.y(), U = d_enu.z();
        double rho_h = std::max(std::sqrt(E * E + N * N), 1e-12);
        double az = std::atan2(E, N);
        double el = std::atan2(U, rho_h);
        double rate = d_hat.dot(sat_ecef.velocity);
        Eigen::Vector4d h;
        h << range, az, el, rate;
        return h;
    }

    /// Residual r (observed − predicted; angles wrapped) and J_state = ∂h/∂x
    /// for sample i at the ECEF satellite state `sat_ecef`.
    void residual_and_jacobian(
        size_t i,
        const StateVector& sat_ecef,
        Eigen::Vector4d& r,
        Eigen::Matrix<double, 4, 6>& J_state) const
    {
        Vector3d d = sat_ecef.position - position_ecef;
        double range = d.norm();
        Vector3d d_hat = d / std::max(range, 1e-12);
        Vector3d d_enu = ecef_to_enu * d;
        double E = d_enu.x(), N = d_enu.y(), U = d_enu.z();
        double rho_h2 = E * E + N * N;
        double rho_h = std::max(std::sqrt(rho_h2), 1e-12);
        double az = std::atan2(E, N);
        double el = std::atan2(U, rho_h);
        double rate = d_hat.dot(sat_ecef.velocity);

        const Eigen::Vector4d& obs = samples[i].second;
        r << obs(0) - range,
             angle_diff(obs(1), az),
             angle_diff(obs(2), el),
             obs(3) - rate;

        J_state.setZero();
        // Range row: ∂range/∂r = d̂ᵀ ; ∂range/∂v = 0.
        J_state.block<1, 3>(0, 0) = d_hat.transpose();
        // Azimuth row: ∂az/∂d_enu = [N/ρ_h², -E/ρ_h², 0]; chain via ECEF→ENU.
        Eigen::RowVector3d dAz_de;
        dAz_de << N / rho_h2, -E / rho_h2, 0.0;
        J_state.block<1, 3>(1, 0) = dAz_de * ecef_to_enu;
        // Elevation row: ∂el/∂d_enu = [-UE/(ρ_h r²), -UN/(ρ_h r²), ρ_h/r²].
        double r2 = std::max(range * range, 1e-24);
        Eigen::RowVector3d dEl_de;
        dEl_de << -U * E / (rho_h * r2),
                  -U * N / (rho_h * r2),
                  rho_h / r2;
        J_state.block<1, 3>(2, 0) = dEl_de * ecef_to_enu;
        // Range-rate row: ∂ρ̇/∂r = (v - ρ̇·d̂)ᵀ / range ; ∂ρ̇/∂v = d̂ᵀ.
        J_state.block<1, 3>(3, 0) = (sat_ecef.velocity - rate * d_hat).transpose() / range;
        J_state.block<1, 3>(3, 3) = d_hat.transpose();
    }
};

/// Ground camera ("optics"): measures azimuth and elevation (rad) only.
/// Each sample is [azimuth, elevation]. Fields as for RadarObs.
struct OpticsObs {
    Vector3d position_ecef = Vector3d::Zero();     // camera location, ECEF [m]
    double sigma_angle = 50e-6;                    // angle noise std dev [rad]
    double lat_rad = 0.0;        // latitude [rad] (informational; not used by the model)
    double lon_rad = 0.0;        // longitude [rad] (informational; not used by the model)
    Matrix3d ecef_to_enu = Matrix3d::Identity();   // ECEF → local east-north-up; must be set
    /// Known bias [az rad, el rad], subtracted from every sample before
    /// estimation. Default zero.
    Eigen::Vector2d known_bias = Eigen::Vector2d::Zero();
    std::vector<double> sigma_scale;   // per-sample noise multiplier (optional)
    SensorClock clock = SensorClock::None;   // camera time tags assumed accurate
    std::vector<std::pair<Epoch, Eigen::Vector2d>> samples;   // (time, [az, el])

    /// Predicted [azimuth, elevation] for an ECEF satellite state; zeros if
    /// the satellite is within 1 m of the camera.
    [[nodiscard]] Eigen::Vector2d predict(const StateVector& sat_ecef) const {
        Vector3d d = sat_ecef.position - position_ecef;
        double range = d.norm();
        if (range < 1.0) return Eigen::Vector2d::Zero();
        Vector3d d_enu = ecef_to_enu * d;
        double E = d_enu.x(), N = d_enu.y(), U = d_enu.z();
        double rho_h = std::max(std::sqrt(E * E + N * N), 1e-12);
        Eigen::Vector2d h;
        h << std::atan2(E, N), std::atan2(U, rho_h);
        return h;
    }

    /// Residual (wrapped angles) and J_state for sample i.
    void residual_and_jacobian(
        size_t i,
        const StateVector& sat_ecef,
        Eigen::Vector2d& r,
        Eigen::Matrix<double, 2, 6>& J_state) const
    {
        Vector3d d = sat_ecef.position - position_ecef;
        double range = std::max(d.norm(), 1e-12);
        Vector3d d_enu = ecef_to_enu * d;
        double E = d_enu.x(), N = d_enu.y(), U = d_enu.z();
        double rho_h2 = E * E + N * N;
        double rho_h = std::max(std::sqrt(rho_h2), 1e-12);
        double az = std::atan2(E, N);
        double el = std::atan2(U, rho_h);

        const Eigen::Vector2d& obs = samples[i].second;
        r << angle_diff(obs(0), az), angle_diff(obs(1), el);

        J_state.setZero();
        Eigen::RowVector3d dAz_de;
        dAz_de << N / rho_h2, -E / rho_h2, 0.0;
        J_state.block<1, 3>(0, 0) = dAz_de * ecef_to_enu;

        double r2 = std::max(range * range, 1e-24);
        Eigen::RowVector3d dEl_de;
        dEl_de << -U * E / (rho_h * r2),
                  -U * N / (rho_h * r2),
                  rho_h / r2;
        J_state.block<1, 3>(1, 0) = dEl_de * ecef_to_enu;
    }
};

/// Position fix (e.g. from GPS): the satellite's ECEF position [m] directly.
struct PositionFixObs {
    double sigma_pos = 3.0;                  // noise std dev per axis [m]
    Vector3d known_bias = Vector3d::Zero();  // known bias [m], subtracted before estimation
    std::vector<double> sigma_scale;         // per-sample noise multiplier (optional)
    std::vector<std::pair<Epoch, Vector3d>> samples;   // (time, ECEF position m)

    /// Predicted observation: the satellite's ECEF position.
    [[nodiscard]] Vector3d predict(const StateVector& sat_ecef) const {
        return sat_ecef.position;
    }

    /// Residual (observed − predicted position) and J_state = [I | 0].
    void residual_and_jacobian(
        size_t i,
        const StateVector& sat_ecef,
        Vector3d& r,
        Eigen::Matrix<double, 3, 6>& J_state) const
    {
        r = samples[i].second - sat_ecef.position;
        J_state.setZero();
        J_state.block<3, 3>(0, 0).setIdentity();
    }
};

/// Onboard accelerometer. Model: reading = EA (in ECEF) + sensor bias. The
/// model assumes the force model holds gravity only, so all the non-gravitational
/// acceleration the accelerometer feels (its specific force) is attributed to
/// the EA; the reading then does not depend on position or velocity
/// (∂h/∂state = 0). The bias is estimated together with the EA.
///
/// Samples are ECEF specific force [m/s²]. Real accelerometers report in the
/// spacecraft body frame; rotate each sample to ECEF with the attitude at that
/// time before storing it.
struct AccelObs {
    double sigma_accel = 0.01;        // noise std dev per axis [m/s²]
    double sigma_bias_prior = 1e-3;   // expected size (std dev) of the unknown bias [m/s²]
    /// Known part of the bias [m/s²], subtracted from every sample; the
    /// remaining unknown bias is still estimated, with sigma_bias_prior as its
    /// expected size.
    Vector3d known_bias = Vector3d::Zero();
    /// Optional outlier model. Declares that the noise is mostly Gaussian with
    /// std dev σ but a fraction p of samples are outliers with std dev k·σ:
    /// (1−p)·N(0,σ²) + p·N(0,(k·σ)²). outlier_prob = p (0 = off, the default);
    /// outlier_scale = k. With p > 0 the smoother down-weights each sample
    /// component according to how far it falls from the prediction (the
    /// Masreliez robust update for this mixture). Measured with 5% of samples
    /// at 10σ (test_ea_smoother): error 21.8 m without it, 11.2 m with p and k
    /// wrong by 2.5× and 2× — having the model matters more than exact values,
    /// so declare your best estimate. It works here, in a filter that compares
    /// each sample with a prediction made before seeing it, which separates
    /// outliers from signal cleanly.

    double outlier_prob = 0.0;
    double outlier_scale = 10.0;
    std::vector<double> sigma_scale;         // per-sample noise multiplier (optional)
    std::vector<std::pair<Epoch, Vector3d>> samples;   // (time, ECEF specific force m/s²)

    /// Predicted reading for EA `ea_ecef` and bias `bias` (both ECEF, m/s²).
    [[nodiscard]] Vector3d predict(const Vector3d& ea_ecef, const Vector3d& bias) const {
        return ea_ecef + bias;
    }

    /// Residual for sample i and its derivatives: J_state = 0, and identity
    /// with respect to the EA (J_ea) and the bias (J_bias).
    void residual_and_jacobian(
        size_t i,
        const StateVector& /*sat_ecef*/,
        const Vector3d& ea_ecef,
        const Vector3d& bias,
        Vector3d& r,
        Eigen::Matrix<double, 3, 6>& J_state,
        Eigen::Matrix<double, 3, 3>& J_ea,
        Eigen::Matrix<double, 3, 3>& J_bias) const
    {
        r = samples[i].second - (ea_ecef + bias);
        J_state.setZero();
        J_ea.setIdentity();
        J_bias.setIdentity();
    }
};

// =============================================================================
// Multi-sensor scenario container + selection mask
// =============================================================================

/// Everything an EA fit needs besides settings: the start time and state, the
/// arc length and the sensors with their samples. The `add_*` methods append
/// a sensor and return the scenario so calls can be chained.
struct MultiSensorScenario {
    Epoch epoch0;                             // start time
    Vector3d pos0_ecef = Vector3d::Zero();    // start position, ECEF (Earth-fixed) [m]
    Vector3d vel0_ecef = Vector3d::Zero();    // start velocity, ECEF (Earth-fixed) [m/s]
    double arc_duration_s = 0.0;              // length of the arc to fit [s]
    std::vector<RadarObs> radars;
    std::vector<OpticsObs> optics;
    std::vector<PositionFixObs> fixes;
    std::vector<AccelObs> accels;

    MultiSensorScenario& add_radar(RadarObs r) {
        radars.push_back(std::move(r));
        return *this;
    }
    MultiSensorScenario& add_optics(OpticsObs c) {
        optics.push_back(std::move(c));
        return *this;
    }
    MultiSensorScenario& add_position_fix(PositionFixObs f) {
        fixes.push_back(std::move(f));
        return *this;
    }
    MultiSensorScenario& add_accelerometer(AccelObs a) {
        accels.push_back(std::move(a));
        return *this;
    }
};

/// Which sensors of a scenario an estimate uses: one flag per sensor, in the
/// scenario's order. Start from `all_on(scenario)` and switch groups off with
/// the chaining `with_*` methods, e.g. `SensorMask::all_on(s).with_accels(false)`.
struct SensorMask {
    std::vector<bool> radars;
    std::vector<bool> optics;
    std::vector<bool> fixes;
    std::vector<bool> accels;

    static SensorMask all_on(const MultiSensorScenario& s) {
        SensorMask m;
        m.radars.assign(s.radars.size(), true);
        m.optics.assign(s.optics.size(), true);
        m.fixes.assign(s.fixes.size(), true);
        m.accels.assign(s.accels.size(), true);
        return m;
    }
    SensorMask& with_radars(bool on) { std::fill(radars.begin(), radars.end(), on); return *this; }
    SensorMask& with_optics(bool on) { std::fill(optics.begin(), optics.end(), on); return *this; }
    SensorMask& with_fixes(bool on) { std::fill(fixes.begin(), fixes.end(), on); return *this; }
    SensorMask& with_accels(bool on) { std::fill(accels.begin(), accels.end(), on); return *this; }
};

}  // namespace calibration
}  // namespace ballistic
