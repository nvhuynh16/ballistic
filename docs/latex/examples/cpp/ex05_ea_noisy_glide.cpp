// =============================================================================
// Tutorial example 05 — Empirical acceleration from NOISY data, hv-block CV
// =============================================================================
//
// Forced-glide truth with Gaussian sensor noise (POS 1.0 m, VEL 0.25 m/s,
// ACC 0.1 m/s^2; deterministic seeds 42/43/44). Hyperparameters are chosen
// by hv-block cross-validation (ea::CV{}) — no truth peeking — matching the
// deployment pipeline. GN (consistent-objective regularized) is default-ON
// and benefits POS and VEL alike; the ACC facade ignores it (circular).
//
// This is the HEAVY example (the CV grid is hundreds of fits). Run one
// {obs,rate} cell per process via regen.sh.
//
//   ex05_ea_noisy_glide <pos|vel|acc> <100|200> [out.csv]

#include "glide_common.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace ballistic;
using namespace ballistic::calibration;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: ex05_ea_noisy_glide <pos|vel|acc> <100|200> [out.csv]\n";
        return 2;
    }
    const std::string obs = argv[1];
    const int hz = std::stoi(argv[2]);
    const std::string out = argc > 3 ? argv[3]
        : "docs/latex/results/ea_noisy_" + obs + "_" + std::to_string(hz) + ".csv";
    ballistic::initialize({});

    const double dt = (hz == 200) ? 0.005 : 0.01;
    tut::GlideTruth truth(/*with_ea=*/true);
    tut::Obs P, V, A;
    truth.sample(dt, P, V, A);
    const auto fm = tut::base_config();

    double sigma = 0.0;
    tut::Obs noisy;
    if (obs == "pos")      { sigma = 1.0;  noisy = tut::add_noise(P, sigma, 42); }
    else if (obs == "vel") { sigma = 0.25; noisy = tut::add_noise(V, sigma, 43); }
    else if (obs == "acc") { sigma = 0.1;  noisy = tut::add_noise(A, sigma, 44); }
    else { std::cerr << "obs must be pos|vel|acc\n"; return 2; }

    EABSplineConfig c;
    c.regularization = ea::CV{};        // hv-block CV (no truth peeking)
    c.noise_sigma = sigma;
    // c.gn defaults to GNRefine{} (regularized GN — on for POS and VEL alike);
    // the ACC facade ignores it (circular; gn_status reports the skip).

    auto t_start = std::chrono::steady_clock::now();
    EAResult r;
    if (obs == "pos")
        r = estimate_ea_bspline(truth.t0, truth.pos, truth.vel, PositionSamples{noisy}, fm, c);
    else if (obs == "vel")
        r = estimate_ea_bspline(truth.t0, truth.pos, truth.vel, VelocitySamples{noisy}, fm, c);
    else
        r = estimate_ea_bspline(truth.t0, truth.pos, truth.vel, AccelSamples{noisy}, fm, c);
    double wall_min =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() / 60.0;

    double rmse = tut::reprop_rmse(truth, r.empirical_accel);

    std::cout << std::fixed << std::setprecision(4)
              << "obs=" << obs << " hz=" << hz << " n=" << noisy.size()
              << " sigma=" << sigma << "\n"
              << "  ok=" << std::boolalpha << r.has_correction()
              << "  lambda_used=" << (r.lambda_used ? *r.lambda_used : -1.0)
              << "  reprop_RMSE=" << rmse << " m"
              << "  wall=" << std::setprecision(1) << wall_min << " min\n";

    std::ofstream f(out);
    f << "obs_type,obs_hz,n_obs,sigma,reg,gn,reprop_rmse_m,lambda_used,wall_min\n"
      << std::fixed << std::setprecision(6)
      << obs << "," << hz << "," << noisy.size() << "," << sigma
      << ",hv_block_cv," << (obs == "acc" ? "n/a" : "on")
      << "," << rmse << ","
      << (r.lambda_used ? *r.lambda_used : -1.0) << ","
      << std::setprecision(2) << wall_min << "\n";
    std::cout << "wrote " << out << "\n";
    return r.has_correction() ? 0 : 1;
}
