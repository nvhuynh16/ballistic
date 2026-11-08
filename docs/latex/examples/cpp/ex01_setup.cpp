// =============================================================================
// Tutorial example 01 — Setup & initialization (C++)
// =============================================================================
//
// Every program starts with ballistic::initialize({}). It loads the embedded
// gravity field + DE440 ephemeris and AUTO-DISCOVERS finals2000A.data (EOP)
// and leap-seconds.list from the platform data directory if present.
//
// InitResult tells you exactly what got loaded — always check it. EOP and a
// fresh leap-second file are what separate meter-level from sub-meter work
// (see the Pitfalls chapter).
//
//   ex01_setup [out.json]   (default: docs/latex/results/ex01_init_status.json)

#include <ballistic/ballistic.hpp>

#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    const std::string out =
        argc > 1 ? argv[1] : "docs/latex/results/ex01_init_status.json";

    // Embedded gravity + DE440 + auto-discovered EOP / leap seconds.
    ballistic::InitResult r = ballistic::initialize({});

    auto leap = ballistic::get_leap_second_info();
    auto data_dir = ballistic::get_data_directory();

    std::cout << "initialize success      : " << std::boolalpha << r.success << "\n"
              << "EOP loaded              : " << r.eop_loaded
              << "  (auto-discovered: " << r.eop_auto_discovered << ")\n"
              << "leap seconds loaded     : " << r.leap_seconds_loaded
              << "  (auto-discovered: " << r.leap_seconds_auto_discovered << ")\n"
              << "embedded ephemeris      : " << r.embedded_ephemeris_loaded
              << "  (" << r.jpl_de_version << ")\n"
              << "gravity degree available: " << r.gravity_degree_loaded << "\n"
              << "current TAI-UTC         : " << leap.current_tai_utc << " s\n"
              << "leap from file          : " << leap.loaded_from_file << "\n"
              << "data directory          : " << data_dir.string() << "\n";

    std::ofstream f(out);
    f << "{\n"
      << "  \"success\": " << (r.success ? "true" : "false") << ",\n"
      << "  \"eop_loaded\": " << (r.eop_loaded ? "true" : "false") << ",\n"
      << "  \"eop_auto_discovered\": " << (r.eop_auto_discovered ? "true" : "false") << ",\n"
      << "  \"leap_seconds_loaded\": " << (r.leap_seconds_loaded ? "true" : "false") << ",\n"
      << "  \"leap_seconds_from_file\": " << (leap.loaded_from_file ? "true" : "false") << ",\n"
      << "  \"current_tai_utc_s\": " << leap.current_tai_utc << ",\n"
      << "  \"leap_entry_count\": " << leap.entry_count << ",\n"
      << "  \"gravity_degree_loaded\": " << r.gravity_degree_loaded << ",\n"
      << "  \"jpl_de_version\": \"" << r.jpl_de_version << "\",\n"
      << "  \"data_directory\": \"" << data_dir.string() << "\"\n"
      << "}\n";
    std::cout << "wrote " << out << "\n";
    return r.success ? 0 : 1;
}
