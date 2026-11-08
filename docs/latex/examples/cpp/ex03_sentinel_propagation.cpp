// =============================================================================
// Tutorial example 03 — Propagation vs truth: Sentinel-1, RK45 vs YOSHIDA8,
//                        ITRF2014 vs ITRF2020 (C++)
// =============================================================================
//
// We take an early state vector from an ESA Sentinel-1 POE (precise orbit)
// EOF file as the initial condition, propagate it for one hour, and RMSE-
// compare against the remaining EOF truth using compute_propagation_result()
// (RTN-decomposed). We sweep:
//
//   * integrator : RK45 (adaptive, best 1-hour accuracy) vs YOSHIDA8
//                  (symplectic, best long-term energy behavior)
//   * ITRF       : ITRF2014 (CORRECT for Sentinel POE before 2025-03-04)
//                  vs ITRF2020 (the post-2025 realization — wrong here)
//
// SRP/drag are OFF: this example validates the force-model-only baseline.
// (Not because the POE "already contains them" — turning both on improves
// the fit; see docs/validation_claims.md SENTINEL-YEAR-2025.) The ITRF
// mismatch is intentionally small (~mm Helmert) —
// the lesson is that it is real but second-order vs frame/EOP correctness.
//
//   ex03_sentinel_propagation [eof_dir] [out.json]

#include <ballistic/ballistic.hpp>
#include <ballistic/validation/eof_parser.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace ballistic;

namespace {

validation::PropagationTestResult run_case(
    const validation::EOFFile& eof, const validation::EOFStateVector& ic,
    IntegratorType integ, coords::ITRFRealization itrf,
    double duration_s, double sample_s)
{
    TrajectoryConfig cfg;
    cfg.itrf = itrf;
    cfg.integrator = integ;
    if (integ != IntegratorType::RK45) cfg.symplectic.step_size = 30.0;

    Trajectory traj(ic.epoch, ic.position, ic.velocity, cfg);
    while (traj.epoch() - ic.epoch < duration_s + sample_s) traj.step();
    TrajectoryInterpolator interp(traj);

    std::vector<std::pair<Epoch, StateVector>> predicted;
    for (double t = 0.0; t <= duration_s; t += sample_s) {
        auto st = interp.get_state(ic.epoch + t);
        if (st) predicted.emplace_back(ic.epoch + t,
                                       StateVector{st->position, st->velocity});
    }
    return validation::compute_propagation_result(predicted, eof);
}

const char* itrf_name(coords::ITRFRealization r) {
    return r == coords::ITRFRealization::ITRF2014 ? "ITRF2014" : "ITRF2020";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string eof_dir =
        argc > 1 ? argv[1] : "tests/data/eof/sentinel1_2016/2016-1";
    const std::string out =
        argc > 2 ? argv[2] : "docs/latex/results/ex03_sentinel_rmse.json";
    ballistic::initialize({});

    auto eof = validation::find_and_parse_eof(eof_dir);
    if (!eof || eof->state_vectors.size() < 400) {
        std::cerr << "no usable EOF in " << eof_dir << "\n";
        return 1;
    }
    std::cout << "Parsed " << eof->mission << "  ("
              << eof->state_vectors.size() << " state vectors, frame "
              << eof->reference_frame << ")\n";

    const auto& ic = eof->state_vectors[10];   // skip edge; well inside arc
    const double DUR = 3600.0, DT = 60.0;

    struct Case { IntegratorType integ; const char* iname;
                  coords::ITRFRealization itrf; };
    const Case cases[] = {
        {IntegratorType::RK45,     "RK45",     coords::ITRFRealization::ITRF2014},
        {IntegratorType::YOSHIDA8, "YOSHIDA8", coords::ITRFRealization::ITRF2014},
        {IntegratorType::RK45,     "RK45",     coords::ITRFRealization::ITRF2020},
    };

    std::ofstream f(out);
    f << std::fixed << std::setprecision(4) << "{\n  \"mission\": \""
      << eof->mission << "\",\n  \"arc_s\": " << DUR
      << ",\n  \"sample_s\": " << DT << ",\n  \"cases\": [\n";

    std::cout << std::fixed << std::setprecision(3)
              << "\n  integrator  ITRF      pos_RMSE   radial   along    cross   vel_RMSE\n"
              << "  ---------------------------------------------------------------------\n";
    for (size_t i = 0; i < std::size(cases); ++i) {
        const auto& c = cases[i];
        auto r = run_case(*eof, ic, c.integ, c.itrf, DUR, DT);
        std::cout << "  " << std::setw(9) << c.iname << "  "
                  << std::setw(8) << itrf_name(c.itrf) << "  "
                  << std::setw(8) << r.position_rmse << "  "
                  << std::setw(7) << r.radial_rms << "  "
                  << std::setw(7) << r.along_track_rms << "  "
                  << std::setw(7) << r.cross_track_rms << "  "
                  << std::setw(7) << r.velocity_rmse << "  (n="
                  << r.num_comparisons << ")\n";
        f << "    {\"integrator\": \"" << c.iname << "\", \"itrf\": \""
          << itrf_name(c.itrf) << "\", \"pos_rmse_m\": " << r.position_rmse
          << ", \"radial_m\": " << r.radial_rms << ", \"along_m\": "
          << r.along_track_rms << ", \"cross_m\": " << r.cross_track_rms
          << ", \"vel_rmse_mps\": " << r.velocity_rmse
          << ", \"n\": " << r.num_comparisons << "}"
          << (i + 1 < std::size(cases) ? ",\n" : "\n");
    }
    f << "  ]\n}\n";
    std::cout << "\nwrote " << out << "\n";
    return 0;
}
