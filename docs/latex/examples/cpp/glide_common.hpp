// =============================================================================
// Shared glide helper for the C++ EA tutorial examples (ex04/05/06/07).
// =============================================================================
//
// Reproduces the canonical "glide" reentry scenario using ONLY the public
// API + the committed glide_fixture_data.hpp constants (which gen_glide_
// fixture.cpp derived bit-for-bit from tests/test_helpers.hpp). The tutorial
// cannot include the test-only helper, so this is the public-API equivalent.
//
//   - make_truth_ea()  : the injected truth EmpiricalAcceleration (GCRS)
//   - base_config()    : gravity-only deg-20 config (NO EA) — the "ballistic"
//   - GlideTruth       : truth trajectory + interpolator + clean obs sampler
//   - reprop_rmse()    : re-propagate with a recovered EA, RMSE vs truth
//
#pragma once

#include <ballistic/ballistic.hpp>
#include <ballistic/calibration/ea_estimator.hpp>

#include "glide_fixture_data.hpp"

#include <cmath>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace tut {

using namespace ballistic;
namespace gf = tut::glidefix;

inline Epoch glide_epoch() {
    return Epoch::from_utc(gf::EPOCH_UTC[0], gf::EPOCH_UTC[1], gf::EPOCH_UTC[2],
                           gf::EPOCH_UTC[3], gf::EPOCH_UTC[4], gf::EPOCH_UTC[5]);
}

// Gravity-only deg-20 config; everything else off. With no empirical_accel
// this IS the "ballistic" trajectory.
inline TrajectoryConfig base_config() {
    TrajectoryConfig c;
    c.gravity_degree = 20;
    c.enable_third_body = false;
    c.enable_relativistic = false;
    c.enable_solid_tides = false;
    return c;
}

// The injected truth EA: two Gaussian bumps + recoil, in GCRS, exactly the
// make_forced_config() formula, using the fixture's derived recoil dir.
inline std::shared_ptr<forces::EmpiricalAcceleration> make_truth_ea() {
    std::vector<forces::EmpiricalAccelEntry> e;
    for (int i = 0; i <= 700; ++i) {
        double t = i;
        double g  = std::exp(-0.5 * std::pow((t - gf::FC) / gf::FW, 2));
        double rg = std::exp(-0.5 * std::pow((t - gf::RC) / gf::RW, 2));
        e.emplace_back(t, Vector3d(0,
                                   gf::LIFT * g,
                                   gf::LAT * g * std::sin(2 * constants::PI * t / gf::LP))
                              + gf::RET * rg * gf::RECOIL_DIR);
    }
    return std::make_shared<forces::EmpiricalAcceleration>(
        forces::EmpiricalAcceleration::from_entries(
            std::move(e), forces::EmpiricalAccelFrame::GCRS));
}

using Obs = std::vector<std::pair<Epoch, Vector3d>>;

// Truth trajectory (base_config + EA) with clean ECEF position/velocity/
// acceleration samplers. `with_ea = false` gives the pure ballistic truth.
struct GlideTruth {
    Epoch t0 = glide_epoch();
    Vector3d pos = gf::POS_ECEF, vel = gf::VEL_ECEF;
    double eval_end = gf::EVAL_END_S;
    std::unique_ptr<Trajectory> traj;
    std::unique_ptr<TrajectoryInterpolator> ecef_i;

    explicit GlideTruth(bool with_ea = true) {
        TrajectoryConfig c = base_config();
        if (with_ea) c.empirical_accel = make_truth_ea();
        traj = std::make_unique<Trajectory>(t0, pos, vel, c);
        while (traj->epoch() - t0 < eval_end + 1.0) traj->step();
        ecef_i = std::make_unique<TrajectoryInterpolator>(*traj);
    }

    // dt in seconds (0.01 = 100 Hz, 0.005 = 200 Hz). pos/vel/accel are the
    // ECEF coordinate quantities from the interpolated history (the same
    // path Python uses, so C++/Python obs are identical).
    void sample(double dt, Obs& po, Obs& vo, Obs& ao) const {
        // t = i*dt, not t += dt: Epoch is exact, so the summed form's
        // ~1e-13 s drift would make these times differ from Python's arange.
        for (int i = 0; i * dt <= eval_end; ++i) {
            const double t = i * dt;
            auto se = ecef_i->get_state(t0 + t);
            if (!se) continue;
            po.emplace_back(t0 + t, se->position);
            vo.emplace_back(t0 + t, se->velocity);
            ao.emplace_back(t0 + t, se->acceleration);
        }
    }
};

inline Obs add_noise(const Obs& clean, double sigma, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> n(0.0, sigma);
    Obs out;
    out.reserve(clean.size());
    for (auto& p : clean)
        out.emplace_back(p.first, p.second + Vector3d(n(rng), n(rng), n(rng)));
    return out;
}

// Re-propagate from the glide IC with a recovered EA; position RMSE vs the
// supplied truth at 1 Hz over [10 s, eval_end]. (1e9 sentinel on failure.)
inline double reprop_rmse(const GlideTruth& truth,
                          const std::shared_ptr<forces::EmpiricalAcceleration>& ea,
                          double max_step_s = 0.0) {
    if (!ea) return 1e9;
    TrajectoryConfig c = base_config();
    c.empirical_accel = ea;
    if (max_step_s > 0.0) c.rk45.max_step_size = max_step_s;
    Trajectory rt(truth.t0, truth.pos, truth.vel, c);
    while (rt.epoch() - truth.t0 < truth.eval_end) rt.step();
    TrajectoryInterpolator ri(rt);
    double sse = 0.0;
    int n = 0;
    for (double t = 10.0; t <= truth.eval_end; t += 1.0) {
        auto a = ri.get_state(truth.t0 + t);
        auto b = truth.ecef_i->get_state(truth.t0 + t);
        if (a && b) { sse += (a->position - b->position).squaredNorm(); ++n; }
    }
    return n > 0 ? std::sqrt(sse / n) : 1e9;
}

// EA=0 baseline: propagate the glide IC with NO empirical acceleration and
// RMSE against the FORCED-glide truth, on the same [10 s, eval_end] @1 Hz
// grid reprop_rmse uses. This is the "do nothing" control — the reference
// scale every recovered-EA RMSE must be read against.
inline double ea0_baseline_rmse(const GlideTruth& forced) {
    GlideTruth ballistic(/*with_ea=*/false);   // same IC, EA=0
    double sse = 0.0;
    int n = 0;
    for (double t = 10.0; t <= forced.eval_end; t += 1.0) {
        auto a = ballistic.ecef_i->get_state(ballistic.t0 + t);
        auto b = forced.ecef_i->get_state(forced.t0 + t);
        if (a && b) { sse += (a->position - b->position).squaredNorm(); ++n; }
    }
    return n > 0 ? std::sqrt(sse / n) : 1e9;
}

}  // namespace tut
