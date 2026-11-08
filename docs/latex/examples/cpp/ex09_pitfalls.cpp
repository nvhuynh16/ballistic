// =============================================================================
// Tutorial example 09 — Pitfalls, demonstrated (C++)
// =============================================================================
//
// Three setup mistakes that silently cost accuracy, made concrete:
//
//  1. Wrong ITRF realization — propagate the same LEO arc as ITRF2014 vs
//     ITRF2020 and measure the position divergence (small but real; ~mm).
//  2. Missing EOP — you cannot un-load EOP at runtime, so we report the
//     actionable check (InitResult.eop_loaded) and the documented cost
//     (~3 m at LEO if UT1-UTC defaults to 0).
//  3. Stale leap seconds — report embedded coverage + expiration + whether
//     a file was loaded; querying past expiration without a fresh file is
//     the risk (seconds of TAI/UTC error → km of position).
//
//   ex09_pitfalls [out.json]

#include <ballistic/ballistic.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace ballistic;

int main(int argc, char** argv) {
    const std::string out =
        argc > 1 ? argv[1] : "docs/latex/results/ex09_pitfalls.json";
    auto init = ballistic::initialize({});
    auto leap = ballistic::get_leap_second_info();

    // --- Pitfall 1: wrong ITRF realization (measured) -----------------------
    Epoch t0 = Epoch::from_utc(2024, 6, 1, 0, 0, 0);
    Vector3d p0(6878137.0, 0.0, 0.0);
    // ~500 km circular LEO, Earth-fixed: inertial 7613 m/s minus omega x r
    // (7613 m/s given as Earth-fixed would be an e 0.13 orbit; State checks it).
    Vector3d v0(0.0, 7613.0 - 501.56, 0.0);
    // Sample WHILE the Trajectory is alive — a TrajectoryInterpolator must
    // not outlive its Trajectory (returning one from a scope where the
    // Trajectory dies is a dangling reference → crash).
    auto track = [&](coords::ITRFRealization itrf) {
        TrajectoryConfig c;
        c.itrf = itrf;
        Trajectory tr(t0, p0, v0, c);
        while (tr.epoch() - t0 < 3660.0) tr.step();
        TrajectoryInterpolator interp(tr);
        std::vector<Vector3d> samples;
        for (double t = 0; t <= 3600.0; t += 60.0) {
            auto s = interp.get_state(t0 + t);
            samples.push_back(s ? s->position : Vector3d::Zero());
        }
        return samples;
    };
    auto s14 = track(coords::ITRFRealization::ITRF2014);
    auto s20 = track(coords::ITRFRealization::ITRF2020);
    double sse = 0.0; int n = 0;
    for (size_t i = 0; i < s14.size() && i < s20.size(); ++i) {
        sse += (s14[i] - s20[i]).squaredNorm(); ++n;
    }
    double itrf_rmse = n ? std::sqrt(sse / n) : -1.0;

    // --- Pitfall 2: EOP presence (actionable check) -------------------------
    bool eop_ok = init.eop_loaded;

    // --- Pitfall 3: leap-second freshness -----------------------------------
    bool leap_from_file = leap.loaded_from_file;
    int tai_utc = leap.current_tai_utc;

    std::cout << std::fixed << std::setprecision(6)
        << "1) Wrong ITRF (ITRF2014 vs ITRF2020), 1 h LEO:\n"
        << "     position RMSE = " << itrf_rmse << " m  (small but real;\n"
        << "     second-order vs frame/EOP — but use the right one anyway)\n"
        << "2) EOP loaded            : " << std::boolalpha << eop_ok
        << "   (if false: ~3 m LEO error from UT1-UTC=0 — run Loader/load_eop)\n"
        << "3) leap from file        : " << leap_from_file
        << " ; TAI-UTC = " << tai_utc << " s ; entries = " << leap.entry_count
        << "\n     (embedded covers through 2025; refresh leap-seconds.list\n"
        << "      before querying later dates — staleness = km-level error)\n";

    std::ofstream f(out);
    f << std::fixed << std::setprecision(6) << "{\n"
      << "  \"wrong_itrf_rmse_m\": " << itrf_rmse << ",\n"
      << "  \"eop_loaded\": " << (eop_ok ? "true" : "false") << ",\n"
      << "  \"missing_eop_documented_leo_error_m\": 3.0,\n"
      << "  \"leap_from_file\": " << (leap_from_file ? "true" : "false") << ",\n"
      << "  \"tai_utc_s\": " << tai_utc << ",\n"
      << "  \"leap_entry_count\": " << leap.entry_count << "\n}\n";
    std::cout << "wrote " << out << "\n";
    return 0;
}
