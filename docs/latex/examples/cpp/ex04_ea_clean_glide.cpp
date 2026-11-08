// =============================================================================
// Tutorial example 04 — Empirical acceleration from CLEAN data (C++)
// =============================================================================
//
// Forced-glide truth, clean ECEF observations at 100 Hz. The lesson here is
// METHOD SELECTION — bare defaults underperform on an aggressive reentry;
// the recommended config per obs type recovers sub-meter EA:
//
//   POS  -> B-spline IRGN, NO GN   (cautionary: GN is load-bearing for POS)
//   POS  -> B-spline IRGN + GN     (recommended for clean POS)
//   VEL  -> B-spline IRGN + GN     (GN helps VEL too, measured)
//   ACC  -> B-spline IRGN          (direct; GN circular for ACC, ignored)
//
//   ex04_ea_clean_glide [out.csv]

#include "glide_common.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace ballistic;
using namespace ballistic::calibration;

int main(int argc, char** argv) {
    const std::string out =
        argc > 1 ? argv[1] : "docs/latex/results/ea_clean_glide.csv";
    ballistic::initialize({});

    tut::GlideTruth truth(/*with_ea=*/true);
    tut::Obs P, V, A;
    truth.sample(0.01, P, V, A);          // 100 Hz
    const auto fm = tut::base_config();

    struct Row { const char* obs; const char* method; double knot;
                 double rmse; bool ok; int it; };
    std::vector<Row> rows;

    // The "do nothing" control: EA=0 reprop vs forced-glide truth. Every
    // row below must be read against this reference scale.
    rows.push_back({"---", "no_ea_baseline", 0.0,
                    tut::ea0_baseline_rmse(truth), true, 0});

    {   EABSplineConfig c;                 // POS B-spline IRGN, NO GN
        c.gn = std::nullopt;               // GN is default-ON — disable to show the baseline
        auto r = estimate_ea_bspline(truth.t0, truth.pos, truth.vel,
                                              PositionSamples{P}, fm, c);
        rows.push_back({"POS", "bspline_no_gn", c.knot_spacing_s,
                        tut::reprop_rmse(truth, r.empirical_accel), r.has_correction(),
                        r.iterations_used});
    }
    {   EABSplineConfig c;                 // POS B-spline IRGN + GN
        c.gn = GNRefine{};                 // load-bearing accuracy step
        auto r = estimate_ea_bspline(truth.t0, truth.pos, truth.vel,
                                              PositionSamples{P}, fm, c);
        rows.push_back({"POS", "bspline_gn", c.knot_spacing_s,
                        tut::reprop_rmse(truth, r.empirical_accel), r.has_correction(),
                        r.iterations_used});
    }
    {   EABSplineConfig c;                 // VEL B-spline IRGN + GN
        c.gn = GNRefine{};                 // GN helps VEL too (measured)
        auto r = estimate_ea_bspline(truth.t0, truth.pos, truth.vel,
                                              VelocitySamples{V}, fm, c);
        rows.push_back({"VEL", "bspline_gn", c.knot_spacing_s,
                        tut::reprop_rmse(truth, r.empirical_accel), r.has_correction(),
                        r.iterations_used});
    }
    {   EABSplineConfig c;                 // ACC B-spline IRGN
        auto r = estimate_ea_bspline(truth.t0, truth.pos,
                                                  truth.vel, AccelSamples{A}, fm, c);
        rows.push_back({"ACC", "bspline_irgn", c.knot_spacing_s,
                        tut::reprop_rmse(truth, r.empirical_accel), r.has_correction(),
                        r.iterations_used});
    }

    std::cout << std::fixed << std::setprecision(4)
              << "obs=100 Hz  n=" << P.size() << "\n"
              << "  obs  method         knot   reprop_RMSE  ok  iters\n"
              << "  ---------------------------------------------------\n";
    std::ofstream f(out);
    f << "obs_type,method,knot_s,obs_hz,n_obs,reprop_rmse_m,ok,iterations\n"
      << std::fixed << std::setprecision(6);
    for (auto& r : rows) {
        std::cout << "  " << std::setw(3) << r.obs << "  " << std::setw(13)
                  << r.method << "  " << std::setw(4) << r.knot << "  "
                  << std::setw(10) << r.rmse << "  " << std::boolalpha
                  << r.ok << "  " << r.it << "\n";
        f << r.obs << "," << r.method << "," << r.knot << ",100," << P.size()
          << "," << r.rmse << "," << (r.ok ? "true" : "false") << ","
          << r.it << "\n";
    }
    std::cout << "wrote " << out << "\n";
    return 0;
}
