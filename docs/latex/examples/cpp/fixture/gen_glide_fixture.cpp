// =============================================================================
// Glide fixture generator  (tutorial support tool — NOT a tutorial listing)
// =============================================================================
//
// The canonical "glide" reentry scenario lives in tests/test_helpers.hpp
// (test-only) and derives its recoil direction from a probe propagation +
// geodetic conversion that the *public* Python API cannot reproduce. To keep
// both tutorials byte-comparable while using only the public API, this
// generator replays the exact make_glide() construction once and emits a
// committed JSON of constants that ex04/05/06/07 (C++ and Python) consume.
//
// Keep this in lock-step with tests/test_helpers.hpp `namespace glide`.
//
//   gen_glide_fixture [out.json]   (default: docs/latex/examples/fixtures/glide_fixture.json)

#include <ballistic/ballistic.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

using namespace ballistic;

namespace {

// ---- canonical glide constants (mirror tests/test_helpers.hpp) --------------
constexpr double LIFT = 8, LAT = 3, FC = 70, FW = 35, LP = 100,
                 RET = 6, RC = 420, RW = 80;

TrajectoryConfig base_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 20;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    return cfg;
}

TrajectoryConfig make_forced_config(const Vector3d& rd) {
    TrajectoryConfig cfg = base_config();
    std::vector<forces::EmpiricalAccelEntry> entries;
    for (int i = 0; i <= 700; ++i) {
        double t = i;
        double g = std::exp(-0.5 * std::pow((t - FC) / FW, 2));
        double rg = std::exp(-0.5 * std::pow((t - RC) / RW, 2));
        entries.emplace_back(
            t,
            Vector3d(0, LIFT * g, LAT * g * std::sin(2 * constants::PI * t / LP))
                + RET * rg * rd);
    }
    cfg.empirical_accel = std::make_shared<forces::EmpiricalAcceleration>(
        forces::EmpiricalAcceleration::from_entries(
            std::move(entries), forces::EmpiricalAccelFrame::GCRS));
    return cfg;
}

std::string vec3(const Vector3d& v) {
    std::ostringstream o;
    o << std::setprecision(17) << "[" << v.x() << ", " << v.y() << ", "
      << v.z() << "]";
    return o.str();
}

}  // namespace

int main(int argc, char** argv) {
    const std::string out_path =
        argc > 1 ? argv[1] : "docs/latex/examples/fixtures/glide_fixture.json";

    auto init = ballistic::initialize({});
    if (!init.success) {
        std::cerr << "initialize failed: " << init.error_message << "\n";
        return 1;
    }

    // ---- initial conditions (identical to make_glide) -----------------------
    const Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    const double R = constants::earth::RADIUS_EQUATORIAL + 120000.0;
    const double vc = std::sqrt(constants::earth::GM / R);
    const double wr = constants::earth::ANGULAR_VELOCITY * R;
    const Vector3d pos(R, 0, 0);
    const Vector3d vel(-30.0, 0.85 * vc - wr, 0);

    // ---- probe pass → recoil direction --------------------------------------
    Trajectory sp(t0, pos, vel, make_forced_config(Vector3d::Zero()));
    for (int i = 0; i < 50000; ++i) {
        auto sr = sp.step();
        if (sr.epoch - t0 >= RC + 10) break;
    }
    TrajectoryInterpolator si(sp);
    auto pk = si.get_state(t0 + RC);
    const Vector3d rd =
        pk ? (-pk->position.normalized() + 0.4 * Vector3d(0, 0, 1)).normalized()
           : Vector3d::Zero();

    // ---- truth pass → first-crossing fd -------------------------------------
    Trajectory truth(t0, pos, vel, make_forced_config(rd));
    double fd = 0;
    for (int i = 0; i < 100000; ++i) {
        auto sr = truth.step();
        if (coords::ecef_to_geodetic(sr.position).altitude < 1000) {
            fd = sr.epoch - t0;
            break;
        }
        if (sr.epoch - t0 > 700) { fd = sr.epoch - t0; break; }
    }
    TrajectoryInterpolator truth_i(truth);
    double lo = fd - 2, hi = fd;
    for (int i = 0; i < 50; ++i) {
        double mid = 0.5 * (lo + hi);
        auto st = truth_i.get_state(t0 + mid);
        if (st && coords::ecef_to_geodetic(st->position).altitude < 1000) hi = mid;
        else lo = mid;
    }
    const double eval_end = hi - 10.0;

    // ---- ballistic (no-EA) impact -------------------------------------------
    // Same IC, base_config (gravity only, NO empirical accel). The adaptive
    // step overshoots ground, so locate the alt==0 crossing by bisection on
    // the interpolated arc (same technique make_glide uses for eval_end).
    TrajectoryConfig bal_cfg = base_config();
    bal_cfg.rk45.max_step_size = 2.0;  // fine steps so the ground crossing is tight
    Trajectory bal(t0, pos, vel, bal_cfg);
    double bal_prev = 0.0, bal_crossed = 0.0;  // last-above / first-below times
    for (int i = 0; i < 200000; ++i) {
        auto sr = bal.step();
        double tt = sr.epoch - t0;
        if (coords::ecef_to_geodetic(sr.position).altitude <= 0.0) {
            bal_crossed = tt;
            break;
        }
        bal_prev = tt;
        if (tt > 1200.0) { bal_crossed = tt; break; }  // safety cap
    }
    TrajectoryInterpolator bal_i(bal);
    // Bracket the sign change with [last-above, first-below] — guaranteed to
    // contain the alt==0 crossing regardless of the adaptive step size.
    double blo = bal_prev, bhi = bal_crossed;
    for (int i = 0; i < 60; ++i) {
        double mid = 0.5 * (blo + bhi);
        auto st = bal_i.get_state(t0 + mid);
        if (st && coords::ecef_to_geodetic(st->position).altitude <= 0.0) bhi = mid;
        else blo = mid;
    }
    const double bal_t = bhi;
    auto bal_state = bal_i.get_state(t0 + bal_t);
    const Vector3d bal_pos = bal_state ? bal_state->position : pos;
    const Vector3d bal_vel = bal_state ? bal_state->velocity : vel;

    // ---- two ground radar stations (sub-satellite at t=0 and at impact) -----
    auto station_at = [](const Vector3d& sub) {
        auto g = coords::ecef_to_geodetic(sub);
        return coords::geodetic_to_ecef(g.latitude, g.longitude, 0.0);
    };
    const Vector3d st0 = station_at(pos);       // under launch point
    const Vector3d st1 = station_at(bal_pos);   // under ballistic impact
    auto g0 = coords::ecef_to_geodetic(st0);
    auto g1 = coords::ecef_to_geodetic(st1);
    constexpr double R2D = 180.0 / constants::PI;

    // ---- emit JSON ----------------------------------------------------------
    std::ostringstream js;
    js << std::setprecision(17);
    js << "{\n";
    js << "  \"_comment\": \"Generated by gen_glide_fixture.cpp — mirrors "
          "tests/test_helpers.hpp namespace glide. Do not hand-edit.\",\n";
    js << "  \"schema\": 1,\n";
    js << "  \"epoch_utc\": [2024, 1, 15, 12, 0, 0],\n";
    js << "  \"ic\": {\n";
    js << "    \"pos_ecef\": " << vec3(pos) << ",\n";
    js << "    \"vel_ecef\": " << vec3(vel) << ",\n";
    js << "    \"R_m\": " << R << ", \"vc_mps\": " << vc
       << ", \"wr_mps\": " << wr << "\n  },\n";
    js << "  \"ea\": {\n";
    js << "    \"frame\": \"GCRS\", \"t_start_s\": 0, \"t_end_s\": 700, "
          "\"dt_s\": 1,\n";
    js << "    \"LIFT\": " << LIFT << ", \"LAT\": " << LAT << ", \"FC\": " << FC
       << ", \"FW\": " << FW << ", \"LP\": " << LP << ", \"RET\": " << RET
       << ", \"RC\": " << RC << ", \"RW\": " << RW << ",\n";
    js << "    \"recoil_direction\": " << vec3(rd) << ",\n";
    js << "    \"formula\": \"a(t) = [0, LIFT*g, LAT*g*sin(2*pi*t/LP)] + "
          "RET*rg*rd ;  g=exp(-0.5*((t-FC)/FW)^2) ;  "
          "rg=exp(-0.5*((t-RC)/RW)^2)\"\n  },\n";
    js << "  \"arc\": { \"first_crossing_s\": " << fd
       << ", \"eval_end_s\": " << eval_end << " },\n";
    js << "  \"ballistic_impact\": {\n";
    js << "    \"t_s\": " << bal_t << ",\n";
    js << "    \"pos_ecef\": " << vec3(bal_pos) << ",\n";
    js << "    \"vel_ecef\": " << vec3(bal_vel) << "\n  },\n";
    js << "  \"radar_stations\": [\n";
    js << "    { \"id\": \"R0\", \"lat_deg\": " << g0.latitude * R2D
       << ", \"lon_deg\": " << g0.longitude * R2D
       << ", \"alt_m\": 0.0, \"pos_ecef\": " << vec3(st0) << " },\n";
    js << "    { \"id\": \"R1\", \"lat_deg\": " << g1.latitude * R2D
       << ", \"lon_deg\": " << g1.longitude * R2D
       << ", \"alt_m\": 0.0, \"pos_ecef\": " << vec3(st1) << " }\n  ]\n";
    js << "}\n";

    std::ofstream f(out_path);
    if (!f) {
        std::cerr << "cannot write " << out_path << "\n";
        return 1;
    }
    f << js.str();
    f.close();

    // Also emit a C++-includable header of the same constants so the C++
    // tutorial examples avoid JSON parsing. Python uses the .json. Both
    // come from THIS run → guaranteed identical.
    std::string hpp_path = out_path;
    if (auto p = hpp_path.rfind(".json"); p != std::string::npos)
        hpp_path.replace(p, 5, "_data.hpp");
    else
        hpp_path += "_data.hpp";
    std::ofstream h(hpp_path);
    if (h) {
        h << std::setprecision(17)
          << "// AUTO-GENERATED by gen_glide_fixture.cpp — do not edit.\n"
          << "#pragma once\n#include <ballistic/core/types.hpp>\n"
          << "namespace tut::glidefix {\n"
          << "inline constexpr int    EPOCH_UTC[6] = {2024,1,15,12,0,0};\n"
          << "inline const ballistic::Vector3d POS_ECEF{" << pos.x() << ","
          << pos.y() << "," << pos.z() << "};\n"
          << "inline const ballistic::Vector3d VEL_ECEF{" << vel.x() << ","
          << vel.y() << "," << vel.z() << "};\n"
          << "inline const ballistic::Vector3d RECOIL_DIR{" << rd.x() << ","
          << rd.y() << "," << rd.z() << "};\n"
          << "inline constexpr double LIFT=" << LIFT << ", LAT=" << LAT
          << ", FC=" << FC << ", FW=" << FW << ", LP=" << LP << ", RET="
          << RET << ", RC=" << RC << ", RW=" << RW << ";\n"
          << "inline constexpr double EVAL_END_S=" << eval_end << ";\n"
          << "inline constexpr double BALLISTIC_IMPACT_T_S=" << bal_t << ";\n"
          << "inline const ballistic::Vector3d BALLISTIC_IMPACT_ECEF{"
          << bal_pos.x() << "," << bal_pos.y() << "," << bal_pos.z() << "};\n"
          << "inline const ballistic::Vector3d RADAR0_ECEF{" << st0.x() << ","
          << st0.y() << "," << st0.z() << "};\n"
          << "inline const ballistic::Vector3d RADAR1_ECEF{" << st1.x() << ","
          << st1.y() << "," << st1.z() << "};\n"
          << "}  // namespace tut::glidefix\n";
        h.close();
        std::cout << "wrote " << hpp_path << "\n";
    }

    std::cout << "wrote " << out_path << "\n"
              << "  recoil_dir = " << vec3(rd) << "\n"
              << "  eval_end   = " << eval_end << " s\n"
              << "  ballistic impact @ " << bal_t << " s, alt="
              << coords::ecef_to_geodetic(bal_pos).altitude << " m\n";
    return 0;
}
