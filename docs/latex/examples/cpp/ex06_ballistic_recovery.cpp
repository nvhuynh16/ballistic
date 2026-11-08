// =============================================================================
// Tutorial example 06 — The ballistic sanity check (C++)
// =============================================================================
//
// "Ballistic" = the glide initial state propagated with NO empirical
// acceleration (gravity only, base_config). We sample its positions, run the
// SAME EA estimator we use on the forced glide, then re-propagate with the
// recovered EA and RMSE against the ballistic truth.
//
// A correct estimator must NOT hallucinate acceleration where there is none:
// the recovered EA should be ~0 and the re-propagation RMSE should be small
// (numerical / fit floor), confirming the pipeline is honest before we trust
// it on forced data.
//
//   ex06_ballistic_recovery [out.csv]

#include "glide_common.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace ballistic;

int main(int argc, char** argv) {
    const std::string out =
        argc > 1 ? argv[1] : "docs/latex/results/ea_ballistic_recovery.csv";
    ballistic::initialize({});

    // Ballistic truth: NO empirical acceleration.
    tut::GlideTruth truth(/*with_ea=*/false);

    tut::Obs pos, vel, acc;
    truth.sample(0.5, pos, vel, acc);   // 2 Hz — plenty for a no-signal check

    calibration::EABSplineConfig cfg;   // defaults: IRGN, knot 2 s, degree 5
    auto r = calibration::estimate_ea_bspline(
        truth.t0, truth.pos, truth.vel, calibration::PositionSamples{pos},
        tut::base_config(), cfg);

    if (!r.has_correction()) {
        std::cerr << "estimator failed: " << r.error_message << "\n";
        return 1;
    }

    // Peak |EA| over the arc — should be ~0 for ballistic data.
    double peak_ea = 0.0;
    for (double t = 0; t <= truth.eval_end; t += 1.0)
        peak_ea = std::max(peak_ea, r.empirical_accel->accel_at(t).norm());

    double rmse = tut::reprop_rmse(truth, r.empirical_accel);

    // Case 2 — the deployment pipeline on noisy null data: does the
    // train-test (hv-block CV) split overfit 1 m of noise into a phantom
    // EA when there is NO real acceleration? POS 100 Hz, sigma=1.0 m,
    // ea::CV{} + GN (the ex05 deployment config).
    tut::Obs P, V, A;
    truth.sample(0.01, P, V, A);                 // 100 Hz
    auto noisy = tut::add_noise(P, 1.0, 42);
    calibration::EABSplineConfig nc;
    nc.regularization = calibration::ea::CV{};
    nc.noise_sigma = 1.0;
    nc.gn = calibration::GNRefine{};
    auto rn = calibration::estimate_ea_bspline(
        truth.t0, truth.pos, truth.vel, calibration::PositionSamples{noisy},
        tut::base_config(), nc);
    double peak_n = 0.0;
    for (double t = 0; t <= truth.eval_end; t += 1.0)
        peak_n = std::max(peak_n, rn.empirical_accel->accel_at(t).norm());
    double rmse_n = tut::reprop_rmse(truth, rn.empirical_accel);

    std::cout << std::fixed << std::setprecision(4)
              << "clean  (2 Hz, IRGN)      : peak|EA|=" << peak_ea
              << "  reprop=" << rmse << " m  (expect ~0)\n"
              << "noisy  (100 Hz, hv-CV+GN): peak|EA|=" << peak_n
              << "  reprop=" << rmse_n << " m  (no phantom EA from noise)\n";

    std::ofstream f(out);
    f << "case,obs_hz,n_obs,sigma,peak_ea_mps2,reprop_rmse_m\n"
      << std::fixed << std::setprecision(6)
      << "clean_default,2," << pos.size() << ",0," << peak_ea << "," << rmse << "\n"
      << "noisy_hvblock_cv,100," << noisy.size() << ",1.0," << peak_n
      << "," << rmse_n << "\n";
    std::cout << "wrote " << out << "\n";

    // Sanity gates: clean null ~0; noisy-CV null must stay well sub-arc
    // (no phantom EA) — generous bound vs the 36 km no-EA-baseline scale.
    return (rmse < 5.0 && rmse_n < 50.0 && rn.has_correction()) ? 0 : 2;
}
