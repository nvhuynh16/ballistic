#pragma once

// =============================================================================
// Core of estimate_ea_smoother: a Kalman filter run forward, then smoothed backward (detail)
// =============================================================================
//
// The smoother estimates an empirical acceleration (EA: an unknown extra acceleration,
// beyond the force model, that the observations reveal) together with the trajectory. The
// state is position r, velocity v and the EA a, all in the inertial GCRS frame. The EA is
// modeled as a first-order Gauss-Markov (GM) process: a random signal that decays towards
// zero with correlation time τ [s] and has steady-state standard deviation σ_a [m/s²].
//
// An extended Kalman filter (EKF: a Kalman filter linearized about the current estimate)
// runs forward through all observations in time order; a Rauch-Tung-Striebel (RTS) backward
// pass then combines each filtered estimate with the information from later observations.
// The filter's consistency is checked with the normalized innovation squared (NIS): each
// observation's residual squared, weighted by its predicted covariance, divided by the
// number of measured values. NIS ≈ 1 means the stated noise levels match the data.
//
// Pieces:
//   StateLayout    — the one place that defines where each quantity sits in the state
//                    vector and how long it is
//   build_timeline — all active observations of all sensors, merged and sorted by time
//   EpochContext   — the frame conversions for one observation time, computed once and
//                    shared by every sensor. Uses the full IAU precession-nutation chain;
//                    the Earth-rotation-angle-only shortcut leaves out precession and
//                    nutation, tens of kilometers at the Earth's surface.
//   radar/fix/optics/accel_update — one function per sensor kind returning the residual r,
//                    its Jacobian H with respect to the state, and the noise covariance R
//   GmProcessModel — moves the state and its covariance P forward in time (Runge-Kutta 4
//                    for the state; transition matrix Φ and process noise Q for P)
//   ekf_forward    — the forward filter loop
//   rts_backward   — the backward smoothing pass
//   run_gm_rts     — runs both with the settings in RunOptions
//
// Tests: tests/unit/test_gm_rts_core.cpp (on a linear system the smoother equals the batch
// least-squares solution) and the smoother integration tests.
// =============================================================================

#include "../multisensor.hpp"
#include "../../coordinates/earth_rotation.hpp"
#include "../../coordinates/iau_transforms.hpp"
#include "../../forces/empirical_acceleration.hpp"
#include "../../forces/force_model.hpp"
#include "../../util/frame_transform.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ballistic {
namespace calibration {
namespace detail {

// ---------------------------------------------------------------------------
// State layout: [r(0..2), v(3..5), a(6..8)] always, then optional constant unknowns: the
// accelerometer bias (3 values, m/s²) and sensor clock offsets (1 value each, s).
// ---------------------------------------------------------------------------
/// Indices of every quantity in the smoother's state vector, and its length `nx`. Sensor
/// updates must build their Jacobians through these indices only.
struct StateLayout {
    static constexpr int idx_r = 0;
    static constexpr int idx_v = 3;
    static constexpr int idx_a = 6;
    int idx_bias = -1;              // accelerometer bias block, or -1 if none
    int idx_clock_shared = -1;      // clock offset shared by all SensorClock::Shared sensors, or -1
    std::vector<int> radar_clock;   // per scenario.radars: clock index or -1
    std::vector<int> optics_clock;  // per scenario.optics: clock index or -1
    std::vector<std::pair<std::string, int>> clock_blocks;  // label → index
    int nx = 9;
    double sigma_bias_prior = 1e-3; // prior σ of the accelerometer bias [m/s²], from the active AccelObs

    [[nodiscard]] bool has_bias() const noexcept { return idx_bias >= 0; }
    [[nodiscard]] bool has_clock() const noexcept { return !clock_blocks.empty(); }

    /// Build the layout for the active sensors of `scenario` (`mask` switches sensors off).
    /// An accelerometer bias block is added if any accelerometer is active (its prior σ is
    /// taken from the last active one). With `est_clock`, each radar and camera adds a clock
    /// offset according to its `clock` setting (None / Shared / Own): first one shared offset
    /// if any sensor uses it, then one per Own sensor.
    [[nodiscard]] static StateLayout make(const MultiSensorScenario& scenario,
                                          const SensorMask& mask,
                                          bool est_clock)
    {
        StateLayout L;
        bool has_accel = false;
        for (size_t s = 0; s < scenario.accels.size(); ++s) {
            if (s < mask.accels.size() && !mask.accels[s]) continue;
            has_accel = true;
            L.sigma_bias_prior = scenario.accels[s].sigma_bias_prior;
        }
        int n = 9;
        if (has_accel) { L.idx_bias = n; n += 3; }

        L.radar_clock.assign(scenario.radars.size(), -1);
        L.optics_clock.assign(scenario.optics.size(), -1);
        if (est_clock) {
            auto active = [](const auto& maskv, size_t s) {
                return s >= maskv.size() || maskv[s];
            };
            bool any_shared = false;
            for (size_t s = 0; s < scenario.radars.size(); ++s)
                if (active(mask.radars, s) &&
                    scenario.radars[s].clock == SensorClock::Shared)
                    any_shared = true;
            for (size_t s = 0; s < scenario.optics.size(); ++s)
                if (active(mask.optics, s) &&
                    scenario.optics[s].clock == SensorClock::Shared)
                    any_shared = true;
            if (any_shared) {
                L.idx_clock_shared = n;
                L.clock_blocks.emplace_back("shared", n);
                n += 1;
            }
            for (size_t s = 0; s < scenario.radars.size(); ++s) {
                if (!active(mask.radars, s)) continue;
                if (scenario.radars[s].clock == SensorClock::Shared)
                    L.radar_clock[s] = L.idx_clock_shared;
                else if (scenario.radars[s].clock == SensorClock::Own) {
                    L.radar_clock[s] = n;
                    L.clock_blocks.emplace_back("radar" + std::to_string(s), n);
                    n += 1;
                }
            }
            for (size_t s = 0; s < scenario.optics.size(); ++s) {
                if (!active(mask.optics, s)) continue;
                if (scenario.optics[s].clock == SensorClock::Shared)
                    L.optics_clock[s] = L.idx_clock_shared;
                else if (scenario.optics[s].clock == SensorClock::Own) {
                    L.optics_clock[s] = n;
                    L.clock_blocks.emplace_back("optics" + std::to_string(s), n);
                    n += 1;
                }
            }
        }
        L.nx = n;
        return L;
    }
};

// ---------------------------------------------------------------------------
// Merged observation timeline.
// ---------------------------------------------------------------------------
/// One observation in the merged timeline: when it is, and which sensor sample it is.
struct ObsRef {
    double t;        // seconds from scenario.epoch0
    int kind;        // 0 radar, 1 position fix, 2 optics (camera), 3 accelerometer, 4 custom
    size_t sensor;   // index into the scenario's per-kind vector
    size_t idx;      // sample index within that sensor
};

struct EpochContext;   // defined below

/// What one observation contributes to the filter: residual r (observed − predicted), its
/// Jacobian H with respect to the full state (rows × nx), and noise covariance R.
struct SensorUpdate {
    Eigen::VectorXd r;
    Eigen::MatrixXd H;
    Eigen::MatrixXd R;
};

/// A user-defined sensor (C++ only; not offered in Python because a Python callback per
/// observation would have to take the interpreter lock on every update). It gives its
/// observation times [s from scenario.epoch0] and a callback returning {r, H, R} for sample i
/// given the frame context, the current state z and the layout. Use the context's frame
/// conversions rather than your own, and build H through StateLayout indices. If the noise is
/// not additive (e.g. multiplicative), transform the measurement inside the callback (e.g.
/// take a logarithm); the NIS check then tells whether the result is consistent. Returning
/// mismatched sizes throws std::invalid_argument.
struct CustomSensor {
    std::vector<double> times_s;   // seconds from scenario.epoch0
    std::function<SensorUpdate(const EpochContext&, const Eigen::VectorXd& z,
                               const StateLayout&, size_t i)> update;
};

/// All samples of the active sensors (and any custom sensors), sorted by time.
[[nodiscard]] inline std::vector<ObsRef> build_timeline(
    const MultiSensorScenario& scenario, const SensorMask& mask,
    const std::vector<CustomSensor>* custom = nullptr)
{
    const Epoch& e0 = scenario.epoch0;
    std::vector<ObsRef> obs;
    for (size_t s = 0; s < scenario.radars.size(); ++s) {
        if (s < mask.radars.size() && !mask.radars[s]) continue;
        for (size_t i = 0; i < scenario.radars[s].samples.size(); ++i)
            obs.push_back({scenario.radars[s].samples[i].first - e0, 0, s, i});
    }
    for (size_t s = 0; s < scenario.fixes.size(); ++s) {
        if (s < mask.fixes.size() && !mask.fixes[s]) continue;
        for (size_t i = 0; i < scenario.fixes[s].samples.size(); ++i)
            obs.push_back({scenario.fixes[s].samples[i].first - e0, 1, s, i});
    }
    for (size_t s = 0; s < scenario.optics.size(); ++s) {
        if (s < mask.optics.size() && !mask.optics[s]) continue;
        for (size_t i = 0; i < scenario.optics[s].samples.size(); ++i)
            obs.push_back({scenario.optics[s].samples[i].first - e0, 2, s, i});
    }
    for (size_t s = 0; s < scenario.accels.size(); ++s) {
        if (s < mask.accels.size() && !mask.accels[s]) continue;
        for (size_t i = 0; i < scenario.accels[s].samples.size(); ++i)
            obs.push_back({scenario.accels[s].samples[i].first - e0, 3, s, i});
    }
    if (custom)
        for (size_t s = 0; s < custom->size(); ++s)
            for (size_t i = 0; i < (*custom)[s].times_s.size(); ++i)
                obs.push_back({(*custom)[s].times_s[i], 4, s, i});
    std::sort(obs.begin(), obs.end(),
              [](const ObsRef& a, const ObsRef& b) { return a.t < b.t; });
    return obs;
}

// ---------------------------------------------------------------------------
// Per-epoch frame context, computed once and shared by every sensor update.
// ---------------------------------------------------------------------------

/// Jacobian of the Earth-fixed (ECEF) state with respect to the GCRS state, from the
/// GCRS→ECEF rotation: r_e = R r_g, v_e = R v_g − ω×r_e (ω = Earth's rotation rate). The
/// tiny terms from the slowly moving pole (precession-nutation rate) are left out; they do
/// not matter in a Jacobian, and residuals use the exact transform.
[[nodiscard]] inline Eigen::Matrix<double, 6, 6>
smoother_ecef_wrt_gcrs(const Matrix3d& R_g2e)
{
    Eigen::Matrix<double, 6, 6> T = Eigen::Matrix<double, 6, 6>::Zero();
    const double w = constants::earth::ANGULAR_VELOCITY;
    Matrix3d Om;
    Om << 0, -w, 0, w, 0, 0, 0, 0, 0;
    T.block<3, 3>(0, 0) = R_g2e;
    T.block<3, 3>(3, 3) = R_g2e;
    T.block<3, 3>(3, 0) = -Om * R_g2e;
    return T;
}

/// Frame conversions for one observation time, shared by all sensor updates.
struct EpochContext {
    Epoch epoch;
    StateVector st_ecef;                 // current state converted exactly to ECEF
    // GCRS → ECEF rotation.
    Matrix3d R_g2e;
    Eigen::Matrix<double, 6, 6> T;       // d(ecef state)/d(gcrs state)
    bool has_xdot = false;
    Eigen::Matrix<double, 6, 1> xdot_e;  // ECEF velocity and acceleration, for clock-offset H columns
};

// ---------------------------------------------------------------------------
// Sensor updates — one function per built-in sensor kind.
// ---------------------------------------------------------------------------
/// Radar sample i: residuals of range [m], two angles [rad] and, if `sigma_rate_eff` > 0,
/// range rate [m/s] (σ = `sigma_rate_eff`; ≤ 0 drops the rate row). A declared known bias is
/// subtracted, and `sigma_scale` scales the noise. With a clock offset state (`clock_idx` ≥ 0)
/// its Jacobian column is the rate of change of the measurement.
[[nodiscard]] inline SensorUpdate radar_update(
    const EpochContext& ctx, const RadarObs& radar, size_t i,
    const StateLayout& L, double sigma_rate_eff, int clock_idx)
{
    SensorUpdate u;
    Eigen::Vector4d r4;
    Eigen::Matrix<double, 4, 6> J6;
    radar.residual_and_jacobian(i, ctx.st_ecef, r4, J6);
    r4 -= radar.known_bias;   // a known bias is data, not a state
    const bool use_rate = sigma_rate_eff > 0.0;
    const int rows = use_rate ? 4 : 3;
    u.r = r4.head(rows);
    u.H = Eigen::MatrixXd::Zero(rows, L.nx);
    u.H.leftCols<6>() = (J6 * ctx.T).topRows(rows);
    u.R = Eigen::MatrixXd::Zero(rows, rows);
    u.R(0, 0) = radar.sigma_range * radar.sigma_range;
    u.R(1, 1) = u.R(2, 2) = radar.sigma_angle * radar.sigma_angle;
    if (use_rate) u.R(3, 3) = sigma_rate_eff * sigma_rate_eff;
    if (!radar.sigma_scale.empty() && i < radar.sigma_scale.size()) {
        const double sc = radar.sigma_scale[i];
        u.R *= sc * sc;
    }
    if (clock_idx >= 0 && ctx.has_xdot) {
        // ∂h/∂(clock offset) = J6 · d(x_ecef)/dt. The velocity part is exact; the acceleration
        // is approximated by R·(f + a), leaving out the small Coriolis/centrifugal terms.
        // CAVEAT: the estimated clock offset is not calibrated: in a test with a +5 ms offset
        // the estimate did not match it, yet estimating it still made the EA about 6× more
        // accurate (test_ea_smoother: 316 m without the state, 52 m with it). Treat the
        // reported clock offset as a health indicator, not a measurement.
        u.H.col(clock_idx) = (J6 * ctx.xdot_e).topRows(rows);
    }
    return u;
}

/// Position-fix sample i: ECEF position residual [m], minus any known bias.
[[nodiscard]] inline SensorUpdate fix_update(
    const EpochContext& ctx, const PositionFixObs& fix, size_t i,
    const StateLayout& L)
{
    SensorUpdate u;
    Vector3d r3;
    Eigen::Matrix<double, 3, 6> J6;
    fix.residual_and_jacobian(i, ctx.st_ecef, r3, J6);
    u.r = r3 - fix.known_bias;
    u.H = Eigen::MatrixXd::Zero(3, L.nx);
    u.H.leftCols<6>() = J6 * ctx.T;
    u.R = fix.sigma_pos * fix.sigma_pos * Eigen::MatrixXd::Identity(3, 3);
    if (!fix.sigma_scale.empty() && i < fix.sigma_scale.size())
        u.R *= fix.sigma_scale[i] * fix.sigma_scale[i];
    return u;
}

/// Camera (optics) sample i: two angle residuals [rad], minus any known bias; optional clock
/// offset column as for radar.
[[nodiscard]] inline SensorUpdate optics_update(
    const EpochContext& ctx, const OpticsObs& cam, size_t i,
    const StateLayout& L, int clock_idx)
{
    SensorUpdate u;
    Eigen::Vector2d r2;
    Eigen::Matrix<double, 2, 6> J6;
    cam.residual_and_jacobian(i, ctx.st_ecef, r2, J6);
    u.r = r2 - cam.known_bias;
    u.H = Eigen::MatrixXd::Zero(2, L.nx);
    u.H.leftCols<6>() = J6 * ctx.T;
    u.R = cam.sigma_angle * cam.sigma_angle * Eigen::MatrixXd::Identity(2, 2);
    if (!cam.sigma_scale.empty() && i < cam.sigma_scale.size())
        u.R *= cam.sigma_scale[i] * cam.sigma_scale[i];
    if (clock_idx >= 0 && ctx.has_xdot)
        u.H.col(clock_idx) = J6 * ctx.xdot_e;
    return u;
}

/// Accelerometer sample i: specific-force residual [m/s²], with the bias state; optional
/// outlier-tolerant weighting when `acc.outlier_prob` > 0.
[[nodiscard]] inline SensorUpdate accel_update(
    const EpochContext& ctx, const AccelObs& acc, size_t i,
    const StateLayout& L, const Eigen::VectorXd& z)
{
    // Predicted reading h = R_g2e·a_gcrs + b: the EA rotated to ECEF plus the accelerometer
    // bias. An accelerometer measures specific force (non-gravitational acceleration); in
    // this model everything not in the force model is the EA, so specific force = EA.
    SensorUpdate u;
    Vector3d pred = ctx.R_g2e * z.segment<3>(StateLayout::idx_a)
                  + z.segment<3>(L.idx_bias);
    u.r = acc.samples[i].second - acc.known_bias - pred;
    u.H = Eigen::MatrixXd::Zero(3, L.nx);
    u.H.block<3, 3>(0, StateLayout::idx_a) = ctx.R_g2e;
    u.H.block<3, 3>(0, L.idx_bias).setIdentity();
    u.R = acc.sigma_accel * acc.sigma_accel * Eigen::MatrixXd::Identity(3, 3);
    if (!acc.sigma_scale.empty() && i < acc.sigma_scale.size())
        u.R *= acc.sigma_scale[i] * acc.sigma_scale[i];
    if (acc.outlier_prob > 0.0) {
        // Outlier-tolerant update (Masreliez): the noise is modeled as a mixture, with
        // probability p of an outlier whose σ is k times larger. For each component the
        // effective variance comes from how likely the current residual is to be an outlier,
        // so likely outliers get little weight. Measured better than plain least squares
        // even when p and k were stated wrongly (test_ea_smoother).
        const double p = acc.outlier_prob, k = acc.outlier_scale;
        for (int c = 0; c < 3; ++c) {
            const double s2 = u.R(c, c);
            const double s = std::sqrt(s2);
            const double nu = u.r(c) / s;
            const double pin = (1.0 - p) * std::exp(-0.5 * nu * nu) / s;
            const double pout =
                p * std::exp(-0.5 * nu * nu / (k * k)) / (k * s);
            const double sum = pin + pout;
            const double info =
                (pin / sum) / s2 + (pout / sum) / (k * k * s2);
            u.R(c, c) = 1.0 / info;
        }
    }
    return u;
}

// ---------------------------------------------------------------------------
// Process model: moves the state forward with Runge-Kutta 4 and the covariance with the
// transition matrix Φ and process noise Q. It refreshes the force-model Jacobian every
// `jac_refresh` substeps and holds the process-noise multiplier the filter sets after a
// suspected maneuver (set_q_boost).
// ---------------------------------------------------------------------------
/// Dynamics of the smoother state: r' = v, v' = f(r, v, t) + a (+ a_ref), a' = −a/τ, constant
/// bias and clock blocks.
struct GmProcessModel {
    const forces::ForceModel& fm;
    Epoch e0;
    double sigma_a;
    double tau;
    const forces::EmpiricalAcceleration* a_ref = nullptr;   // known EA from a previous pass, added to the dynamics (iterated smoothing); may be null
    int jac_refresh = 16;

    double qc = 0.0;                 // strength (power spectral density) of the noise driving a, 2σ_a²/τ
    double q_boost_cur = 1.0;        // current process-noise multiplier (raised after a suspected maneuver)
    Eigen::Matrix<double, 3, 6> Jf = Eigen::Matrix<double, 3, 6>::Zero();
    int jac_age = 0;
    double t_cur = 0.0;

    GmProcessModel(const forces::ForceModel& fm_, const Epoch& e0_,
                   double sa, double tu,
                   const forces::EmpiricalAcceleration* ref, int refresh)
        : fm(fm_), e0(e0_), sigma_a(sa), tau(tu), a_ref(ref),
          jac_refresh(refresh), qc(2.0 * sa * sa / tu), jac_age(refresh) {}

    /// Multiply the process noise by `b` from now on (1 = normal).
    void set_q_boost(double b) noexcept { q_boost_cur = b; }

    /// Time derivative of the state `zz` (length `nx`) at epoch `ep`.
    [[nodiscard]] Eigen::VectorXd zdot(const Eigen::VectorXd& zz,
                                       const Epoch& ep, int nx) const
    {
        Eigen::VectorXd d = Eigen::VectorXd::Zero(nx);
        StateVector s{zz.segment<3>(0), zz.segment<3>(3)};
        Vector3d f = fm.acceleration(ep, s);
        d.segment<3>(0) = zz.segment<3>(3);
        d.segment<3>(3) = f + zz.segment<3>(6);
        if (a_ref) d.segment<3>(3) += a_ref->accel_at(ep - e0);
        d.segment<3>(6) = -zz.segment<3>(6) / tau;
        // Bias and clock blocks are unknown constants: derivative 0.
        return d;
    }

    /// Advance state z and covariance P from the current time to `t_target` [s from e0] in
    /// substeps of at most `max_dt` [s], multiplying each substep's Φ into `Phi_total`. Φ is a
    /// second-order Taylor expansion of exp(F·dt); Q is the integral for white noise driving a
    /// through the r–v–a chain with the decay of a (1/τ) left out within a substep, which is
    /// accurate for max_dt ≪ τ.

    void propagate_to(Eigen::VectorXd& z, Eigen::MatrixXd& P,
                      Eigen::MatrixXd& Phi_total, double t_target,
                      double max_dt, const StateLayout& L)
    {
        const int nx = L.nx;
        double remaining = t_target - t_cur;
        while (remaining > 1e-12) {
            double dt = std::min(remaining, max_dt);
            Epoch ep = e0 + t_cur;
            Eigen::VectorXd k1 = zdot(z, ep, nx);
            Eigen::VectorXd k2 = zdot(z + 0.5 * dt * k1, e0 + (t_cur + 0.5 * dt), nx);
            Eigen::VectorXd k3 = zdot(z + 0.5 * dt * k2, e0 + (t_cur + 0.5 * dt), nx);
            Eigen::VectorXd k4 = zdot(z + dt * k3, e0 + (t_cur + dt), nx);
            z += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
            if (++jac_age >= jac_refresh) {
                StateVector s{z.segment<3>(0), z.segment<3>(3)};
                auto [aa, JJ] = fm.acceleration_with_jacobian(e0 + (t_cur + dt), s);
                Jf = JJ;
                jac_age = 0;
            }
            Eigen::MatrixXd F = Eigen::MatrixXd::Zero(nx, nx);
            F.block<3, 3>(0, 3).setIdentity();
            F.block<3, 3>(3, 0) = Jf.block<3, 3>(0, 0);
            F.block<3, 3>(3, 3) = Jf.block<3, 3>(0, 3);
            F.block<3, 3>(3, 6).setIdentity();
            F.block<3, 3>(6, 6) = -Matrix3d::Identity() / tau;
            // Bias/clock blocks: F = 0, so Φ = I there and no process noise.
            Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(nx, nx)
                                + F * dt + 0.5 * (F * F) * (dt * dt);
            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nx, nx);
            const double qc_eff = qc * q_boost_cur;
            const double d1 = dt, d2 = dt * dt, d3 = d2 * dt, d4 = d3 * dt,
                         d5 = d4 * dt;
            for (int c = 0; c < 3; ++c) {
                Q(6 + c, 6 + c) = qc_eff * d1;
                Q(3 + c, 6 + c) = Q(6 + c, 3 + c) = qc_eff * d2 / 2.0;
                Q(3 + c, 3 + c) = qc_eff * d3 / 3.0;
                Q(0 + c, 6 + c) = Q(6 + c, 0 + c) = qc_eff * d3 / 6.0;
                Q(0 + c, 3 + c) = Q(3 + c, 0 + c) = qc_eff * d4 / 8.0;
                Q(0 + c, 0 + c) = qc_eff * d5 / 20.0;
            }
            P = Phi * P * Phi.transpose() + Q;
            Phi_total = Phi * Phi_total;
            t_cur += dt;
            remaining = t_target - t_cur;
        }
    }
};

// ---------------------------------------------------------------------------
// Stored filter results and the two passes.
// ---------------------------------------------------------------------------
/// Filter results at one observation: predicted and updated state and covariance, and the
/// transition matrix Φ from the previous observation.
struct FilterNode {
    Eigen::VectorXd z_pred, z_filt;
    Eigen::MatrixXd P_pred, P_filt, Phi;
};

/// Settings of one smoother run.
struct RunOptions {
    // EA steady-state σ [m/s²] and correlation time [s].
    double sigma_a = 3.0;
    double tau = 35.0;
    double sigma_rate_eff = 0.0;     // σ used for radar range rate [m/s]; ≤ 0 = range rate not used
    // max_dt: largest propagation substep [s]. jac_refresh: substeps between force-Jacobian
    // refreshes. a_ref: known EA added to the dynamics, or null. est_clock: estimate sensor
    // clock offsets, each with prior σ clock_prior_s [s]. adaptive_q: after an observation
    // whose NIS per value exceeds nis_gate, multiply the process noise by q_boost (until the
    // next observation), so the filter can follow a sudden maneuver.
    double max_dt = 0.05;
    int jac_refresh = 16;
    const forces::EmpiricalAcceleration* a_ref = nullptr;
    bool est_clock = false;
    double clock_prior_s = 0.02;
    bool adaptive_q = false;
    double nis_gate = 9.0;
    double q_boost = 25.0;
    const std::vector<CustomSensor>* custom = nullptr;   // user-defined sensors, or null
};

/// Forward EKF over the timeline `obs`, starting from state z and covariance P (both updated
/// in place). Fills `nodes` (one per observation) and returns the NIS sum and the number of
/// measured values in `nis_sum` / `nis_dof`. Throws std::invalid_argument if a custom sensor
/// returns inconsistent sizes.
inline void ekf_forward(
    const MultiSensorScenario& scenario,
    const std::vector<ObsRef>& obs,
    const StateLayout& L,
    coords::ITRFRealization itrf,
    GmProcessModel& proc,
    const RunOptions& opts,
    Eigen::VectorXd& z, Eigen::MatrixXd& P,
    std::vector<FilterNode>& nodes,
    double& nis_sum, long& nis_dof)
{
    const Epoch& e0 = scenario.epoch0;
    const int nx = L.nx;
    const size_t N = obs.size();
    nodes.resize(N);
    nis_sum = 0.0;
    nis_dof = 0;

    for (size_t k = 0; k < N; ++k) {
        Eigen::MatrixXd Phi_total = Eigen::MatrixXd::Identity(nx, nx);
        proc.propagate_to(z, P, Phi_total, obs[k].t, opts.max_dt, L);
        nodes[k].z_pred = z;
        nodes[k].P_pred = P;
        nodes[k].Phi = Phi_total;

        // Frame conversions for this observation time, shared by every sensor.
        EpochContext ctx;
        ctx.epoch = e0 + obs[k].t;
        StateVector sg_cur{z.segment<3>(0), z.segment<3>(3)};
        ctx.st_ecef = util::gcrs_to_itrf(sg_cur, ctx.epoch, itrf);
        ctx.R_g2e =
            coords::detail::get_cached_transform().gcrs_to_itrf_matrix(ctx.epoch);
        ctx.T = smoother_ecef_wrt_gcrs(ctx.R_g2e);
        const int clock_idx =
            (obs[k].kind == 0) ? L.radar_clock[obs[k].sensor]
            : (obs[k].kind == 2) ? L.optics_clock[obs[k].sensor] : -1;
        if (clock_idx >= 0) {
            StateVector s_g{z.segment<3>(0), z.segment<3>(3)};
            Vector3d a_g = proc.fm.acceleration(ctx.epoch, s_g) + z.segment<3>(6);
            if (proc.a_ref) a_g += proc.a_ref->accel_at(obs[k].t);
            ctx.xdot_e << ctx.st_ecef.velocity, ctx.R_g2e * a_g;
            ctx.has_xdot = true;
        }

        SensorUpdate u;
        switch (obs[k].kind) {
            case 0: u = radar_update(ctx, scenario.radars[obs[k].sensor],
                                     obs[k].idx, L, opts.sigma_rate_eff,
                                     clock_idx);
                    break;
            case 1: u = fix_update(ctx, scenario.fixes[obs[k].sensor],
                                   obs[k].idx, L);
                    break;
            case 2: u = optics_update(ctx, scenario.optics[obs[k].sensor],
                                      obs[k].idx, L, clock_idx);
                    break;
            case 4:  u = (*opts.custom)[obs[k].sensor].update(
                             ctx, z, L, obs[k].idx);
                    // Check the sizes: a mismatched Eigen product is undefined behavior in a
                    // release build, and this is user code.
                    if (u.H.cols() != L.nx || u.r.size() != u.H.rows() ||
                        u.R.rows() != u.H.rows() ||
                        u.R.cols() != u.R.rows())
                        throw std::invalid_argument(
                            "CustomSensor update returned inconsistent "
                            "dimensions: r=" + std::to_string(u.r.size()) +
                            " H=" + std::to_string(u.H.rows()) + "x" +
                            std::to_string(u.H.cols()) + " R=" +
                            std::to_string(u.R.rows()) + "x" +
                            std::to_string(u.R.cols()) + " (state nx=" +
                            std::to_string(L.nx) + ")");
                    break;
            default: u = accel_update(ctx, scenario.accels[obs[k].sensor],
                                      obs[k].idx, L, z);
                     break;
        }

        Eigen::MatrixXd S = u.H * P * u.H.transpose() + u.R;
        Eigen::LDLT<Eigen::MatrixXd> Sld(S);
        const double nis_i = u.r.dot(Sld.solve(u.r));
        nis_sum += nis_i;
        nis_dof += u.r.size();
        if (opts.adaptive_q)
            proc.set_q_boost(
                (nis_i / u.r.size() > opts.nis_gate) ? opts.q_boost : 1.0);
        Eigen::MatrixXd K = P * u.H.transpose()
            * Sld.solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
        z += K * u.r;
        Eigen::MatrixXd IKH = Eigen::MatrixXd::Identity(nx, nx) - K * u.H;
        P = IKH * P * IKH.transpose() + K * u.R * K.transpose();

        nodes[k].z_filt = z;
        nodes[k].P_filt = P;
    }
}

/// RTS backward pass over the forward results. Returns the smoothed state at every node in
/// `zs` and the 1σ uncertainty of the EA (from the smoothed covariance) in `a_sig`, the only
/// uncertainty the smoother reports. Requires at least one node.
inline void rts_backward(const std::vector<FilterNode>& nodes, int nx,
                         std::vector<Eigen::VectorXd>& zs,
                         std::vector<Vector3d>& a_sig)
{
    const size_t N = nodes.size();
    zs.resize(N);
    a_sig.resize(N);
    zs[N - 1] = nodes[N - 1].z_filt;
    Eigen::MatrixXd Ps = nodes[N - 1].P_filt;
    a_sig[N - 1] = Ps.diagonal().segment<3>(6).cwiseSqrt();
    for (size_t k = N - 1; k > 0; --k) {
        const auto& nk = nodes[k];
        const auto& nkm = nodes[k - 1];
        Eigen::LDLT<Eigen::MatrixXd> Pp(nk.P_pred);
        Eigen::MatrixXd C = nkm.P_filt * nk.Phi.transpose()
            * Pp.solve(Eigen::MatrixXd::Identity(nx, nx));
        zs[k - 1] = nkm.z_filt + C * (zs[k] - nk.z_pred);
        Ps = nkm.P_filt + C * (Ps - nk.P_pred) * C.transpose();
        a_sig[k - 1] = Ps.diagonal().segment<3>(6).cwiseSqrt();
    }
}

// ---------------------------------------------------------------------------
// One full forward-backward run.
// ---------------------------------------------------------------------------
/// Result of run_gm_rts. `ok` is false (with `error`) if the run could not start.
struct SmootherRun {
    // Observation times [s from epoch0]; accelerometer bias [m/s²].
    std::vector<double> t;
    Vector3d bias = Vector3d::Zero();
    double clock_bias = 0.0;   // clock offset [s]: the shared one, else the first estimated
    std::vector<std::pair<std::string, double>> clocks;   // label → bias [s]
    std::vector<Vector3d> a_smooth;          // smoothed EA at each time, GCRS [m/s²]
    // 1σ of a_smooth [m/s²].
    std::vector<Vector3d> a_sigma;
    std::vector<Eigen::VectorXd> z_smooth;   // full states at obs times
    // Average NIS per measured value (≈ 1 if the declared noise matches the data).
    double nis = 0.0;
    bool ok = false;
    std::string error;
};

/// Run the forward filter and backward smoother over the active observations. The start
/// state is the scenario's initial state (Earth-fixed, converted to GCRS), trusted tightly
/// (prior σ 1 cm, 1 mm/s); the EA starts at zero with prior σ_a. Fewer than 10 active
/// observations returns ok = false. The bias and clock offsets are reported at the first
/// observation (they are constants).
[[nodiscard]] inline SmootherRun run_gm_rts(
    const MultiSensorScenario& scenario,
    const SensorMask& mask,
    const forces::ForceModel& fm,
    coords::ITRFRealization itrf,
    const RunOptions& opts)
{
    SmootherRun out;
    const Epoch& e0 = scenario.epoch0;

    std::vector<ObsRef> obs = build_timeline(scenario, mask, opts.custom);
    if (obs.size() < 10) {
        out.error = "ea_smoother: fewer than 10 active observations";
        return out;
    }
    const StateLayout L = StateLayout::make(scenario, mask, opts.est_clock);
    const size_t N = obs.size();

    // Initial state: the scenario's start state, trusted tightly as the other estimators do.
    StateVector se{scenario.pos0_ecef, scenario.vel0_ecef};
    StateVector sg = util::itrf_to_gcrs(se, e0, itrf);
    Eigen::VectorXd z = Eigen::VectorXd::Zero(L.nx);
    z.segment<3>(0) = sg.position;
    z.segment<3>(3) = sg.velocity;
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(L.nx, L.nx);
    P.diagonal().head<9>() << 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6,
        opts.sigma_a * opts.sigma_a, opts.sigma_a * opts.sigma_a,
        opts.sigma_a * opts.sigma_a;
    if (L.has_bias())
        P.diagonal().segment<3>(L.idx_bias)
            .setConstant(L.sigma_bias_prior * L.sigma_bias_prior);
    for (const auto& [label, ci] : L.clock_blocks)
        P(ci, ci) = opts.clock_prior_s * opts.clock_prior_s;

    GmProcessModel proc(fm, e0, opts.sigma_a, opts.tau, opts.a_ref,
                        opts.jac_refresh);

    std::vector<FilterNode> nodes;
    double nis_sum = 0.0;
    long nis_dof = 0;
    ekf_forward(scenario, obs, L, itrf, proc, opts, z, P, nodes,
                nis_sum, nis_dof);

    std::vector<Eigen::VectorXd> zs;
    std::vector<Vector3d> a_sig;
    rts_backward(nodes, L.nx, zs, a_sig);

    out.t.reserve(N);
    out.a_smooth.reserve(N);
    for (size_t k = 0; k < N; ++k) {
        out.t.push_back(obs[k].t);
        out.a_smooth.push_back(zs[k].segment<3>(6));
    }
    out.a_sigma = std::move(a_sig);
    if (L.has_bias()) out.bias = zs.front().segment<3>(L.idx_bias);
    for (const auto& [label, ci] : L.clock_blocks) {
        out.clocks.emplace_back(label, zs.front()(ci));
        if (label == "shared") out.clock_bias = zs.front()(ci);
    }
    if (out.clock_bias == 0.0 && !out.clocks.empty())
        out.clock_bias = out.clocks.front().second;
    out.z_smooth = std::move(zs);
    out.nis = nis_sum / static_cast<double>(std::max<long>(nis_dof, 1));
    out.ok = true;
    return out;
}

/// RMS residual of every measurement channel (radar range/angle/rate, fix position, camera
/// angle) at the smoothed solution, labeled e.g. "radar0_range". Compare each with the noise
/// σ the user declared: a channel whose residual is far above its declared σ could not be
/// fitted as declared (wrong noise level, an unmodeled bias or timing error, or a limit of
/// the model).
[[nodiscard]] inline std::vector<std::pair<std::string, double>>
measure_honored_sigmas(const MultiSensorScenario& scenario,
                       const SensorMask& mask,
                       coords::ITRFRealization itrf,
                       const SmootherRun& run)
{
    const Epoch& e0 = scenario.epoch0;
    std::vector<std::pair<std::string, double>> out;
    auto state_at = [&](double t, size_t& cursor) -> const Eigen::VectorXd* {
        while (cursor + 1 < run.t.size() && run.t[cursor] < t - 1e-9) ++cursor;
        if (std::abs(run.t[cursor] - t) > 1e-6) return nullptr;
        return &run.z_smooth[cursor];
    };
    for (size_t s = 0; s < scenario.radars.size(); ++s) {
        if (s < mask.radars.size() && !mask.radars[s]) continue;
        const auto& radar = scenario.radars[s];
        Eigen::Vector4d sse = Eigen::Vector4d::Zero();
        long n = 0;
        size_t cursor = 0;
        for (size_t i = 0; i < radar.samples.size(); ++i) {
            const double t = radar.samples[i].first - e0;
            const auto* zz = state_at(t, cursor);
            if (!zz) continue;
            StateVector sg{zz->segment<3>(0), zz->segment<3>(3)};
            StateVector st_e = util::gcrs_to_itrf(sg, e0 + t, itrf);
            Eigen::Vector4d r4;
            Eigen::Matrix<double, 4, 6> J6;
            radar.residual_and_jacobian(i, st_e, r4, J6);
            r4 -= radar.known_bias;
            sse += r4.cwiseProduct(r4);
            n++;
        }
        if (n > 0) {
            const std::string p = "radar" + std::to_string(s) + "_";
            out.emplace_back(p + "range", std::sqrt(sse(0) / n));
            out.emplace_back(p + "angle", std::sqrt((sse(1) + sse(2)) / (2 * n)));
            out.emplace_back(p + "rate", std::sqrt(sse(3) / n));
        }
    }
    for (size_t s = 0; s < scenario.fixes.size(); ++s) {
        if (s < mask.fixes.size() && !mask.fixes[s]) continue;
        const auto& fix = scenario.fixes[s];
        double sse = 0.0;
        long n = 0;
        size_t cursor = 0;
        for (size_t i = 0; i < fix.samples.size(); ++i) {
            const double t = fix.samples[i].first - e0;
            const auto* zz = state_at(t, cursor);
            if (!zz) continue;
            StateVector sg{zz->segment<3>(0), zz->segment<3>(3)};
            StateVector st_e = util::gcrs_to_itrf(sg, e0 + t, itrf);
            Vector3d r3;
            Eigen::Matrix<double, 3, 6> J6;
            fix.residual_and_jacobian(i, st_e, r3, J6);
            sse += (r3 - fix.known_bias).squaredNorm();
            n += 3;
        }
        if (n > 0)
            out.emplace_back("fix" + std::to_string(s) + "_pos",
                             std::sqrt(sse / n));
    }
    for (size_t s = 0; s < scenario.optics.size(); ++s) {
        if (s < mask.optics.size() && !mask.optics[s]) continue;
        const auto& cam = scenario.optics[s];
        double sse = 0.0;
        long n = 0;
        size_t cursor = 0;
        for (size_t i = 0; i < cam.samples.size(); ++i) {
            const double t = cam.samples[i].first - e0;
            const auto* zz = state_at(t, cursor);
            if (!zz) continue;
            StateVector sg{zz->segment<3>(0), zz->segment<3>(3)};
            StateVector st_e = util::gcrs_to_itrf(sg, e0 + t, itrf);
            Eigen::Vector2d r2;
            Eigen::Matrix<double, 2, 6> J6;
            cam.residual_and_jacobian(i, st_e, r2, J6);
            sse += (r2 - cam.known_bias).squaredNorm();
            n += 2;
        }
        if (n > 0)
            out.emplace_back("optics" + std::to_string(s) + "_angle",
                             std::sqrt(sse / n));
    }
    return out;
}

/// RMS of the radar range-rate residuals [m/s] at the smoothed solution: the range-rate
/// accuracy the model can actually achieve, used to set the σ for range rate. 0 if there is no
/// matching sample.
[[nodiscard]] inline double smoother_rate_residual_rms(
    const MultiSensorScenario& scenario,
    const SensorMask& mask,
    coords::ITRFRealization itrf,
    const SmootherRun& run)
{
    const Epoch& e0 = scenario.epoch0;
    double sse = 0.0;
    long n = 0;
    for (size_t s = 0; s < scenario.radars.size(); ++s) {
        if (s < mask.radars.size() && !mask.radars[s]) continue;
        const auto& radar = scenario.radars[s];
        // Each radar's samples are time-ordered on their own, but radars interleave in time:
        // the search into run.t restarts for every radar, or a later radar's samples that
        // precede the end of an earlier one would be skipped.
        size_t cursor = 0;
        for (size_t i = 0; i < radar.samples.size(); ++i) {
            const double t = radar.samples[i].first - e0;
            while (cursor + 1 < run.t.size() && run.t[cursor] < t - 1e-9)
                ++cursor;
            if (std::abs(run.t[cursor] - t) > 1e-6) continue;
            const Eigen::VectorXd& zz = run.z_smooth[cursor];
            StateVector sg{zz.segment<3>(0), zz.segment<3>(3)};
            StateVector st_e = util::gcrs_to_itrf(
                sg, e0 + t, itrf);
            Eigen::Vector4d r4;
            Eigen::Matrix<double, 4, 6> J6;
            radar.residual_and_jacobian(i, st_e, r4, J6);
            r4 -= radar.known_bias;   // a declared bias is data, as in the filter update
            sse += r4(3) * r4(3);
            n++;
        }
    }
    return n ? std::sqrt(sse / n) : 0.0;
}

}  // namespace detail
}  // namespace calibration
}  // namespace ballistic
