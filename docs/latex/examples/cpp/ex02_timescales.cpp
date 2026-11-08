// =============================================================================
// Tutorial example 02 — Epoch & time scales (C++)
// =============================================================================
//
// One instant, six time scales. They differ by fixed or slowly-varying
// offsets, all expressed against TAI (International Atomic Time):
//
//   TT  = TAI + 32.184 s          (exact, IAU 1991)
//   GPS = TAI - 19 s              (exact, GPS ICD, since 1980)
//   UTC = TAI - (leap seconds)    (currently 37 s; integer, steps on leap days)
//   UT1 = UTC + DUT1              (Earth-rotation; needs EOP, else DUT1=0)
//   TDB ~ TT                      (periodic, sub-ms)
//
// Epoch stores TT internally; every accessor converts on demand. Epoch
// subtraction yields SECONDS.
//
// PITFALL demonstrated here: recovering a small offset by differencing two
// large Julian Dates loses precision. At JD ~ 2.46e6 one double ulp is
// ~4.7e-5 s, so TT-TAI prints as 32.18401.. not exactly 32.18400. The
// offsets ARE exact by definition (IAU/GPS ICD); the µs wobble is float
// cancellation, not a library error. Never derive sub-ms time offsets by
// subtracting raw JDs — keep the constant, or difference Epochs directly.
//
//   ex02_timescales [out.json]

#include <ballistic/ballistic.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace ballistic;

int main(int argc, char** argv) {
    const std::string out =
        argc > 1 ? argv[1] : "docs/latex/results/ex02_timescale_offsets.json";
    ballistic::initialize({});

    Epoch e = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    const double SPD = constants::SECONDS_PER_DAY;

    // Offsets vs TAI, in seconds.
    double tai = e.tai_jd();
    double tt_tai  = (e.tt_jd()  - tai) * SPD;
    double gps_tai = (e.gps_jd() - tai) * SPD;
    double utc_tai = (e.utc_jd() - tai) * SPD;
    double ut1_tai = (e.ut1_jd() - tai) * SPD;
    double tdb_tai = (e.tdb_jd() - tai) * SPD;

    std::cout << std::fixed << std::setprecision(6)
              << "Epoch (UTC 2024-01-15 12:00:00)\n"
              << "  TT  - TAI = " << tt_tai  << " s  (expect 32.184000)\n"
              << "  GPS - TAI = " << gps_tai << " s  (expect -19.000000)\n"
              << "  UTC - TAI = " << utc_tai << " s  (= -(leap seconds))\n"
              << "  UT1 - TAI = " << ut1_tai << " s  (UTC + DUT1; needs EOP)\n"
              << "  TDB - TAI = " << tdb_tai << " s  (~ TT, periodic)\n";

    // Round-trip the calendar in UTC and TAI to expose the offset directly.
    auto [Y, M, D, h, m, s] = e.to_calendar(time::TimeScale::TAI);
    std::cout << "  Calendar in TAI: " << Y << "-" << M << "-" << D << " "
              << h << ":" << m << ":" << std::setprecision(3) << s << "\n";

    std::ofstream f(out);
    f << std::fixed << std::setprecision(9) << "{\n"
      << "  \"epoch_utc\": \"2024-01-15T12:00:00\",\n"
      << "  \"tt_minus_tai_s\": "  << tt_tai  << ",\n"
      << "  \"gps_minus_tai_s\": " << gps_tai << ",\n"
      << "  \"utc_minus_tai_s\": " << utc_tai << ",\n"
      << "  \"ut1_minus_tai_s\": " << ut1_tai << ",\n"
      << "  \"tdb_minus_tai_s\": " << tdb_tai << "\n}\n";
    std::cout << "wrote " << out << "\n";

    // Self-check tolerance is JD-precision-aware (~5e-5 s ulp floor at this
    // epoch; 1e-3 still catches a genuine wrong offset, e.g. a 19↔18 s bug,
    // while tolerating the documented float-cancellation wobble).
    constexpr double JD_DIFF_TOL_S = 1e-3;
    bool ok = std::abs(tt_tai - 32.184) < JD_DIFF_TOL_S &&
              std::abs(gps_tai - (-19.0)) < JD_DIFF_TOL_S;
    if (!ok) { std::cerr << "OFFSET MISMATCH (beyond JD-cancellation noise)\n"; return 1; }
    return 0;
}
