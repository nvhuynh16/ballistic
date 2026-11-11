// The C++ translation of NRLMSIS 2.1 against the official reference output.
//
// tests/data/nrlmsis/msis2.1_test_ref_dp.txt is the double-precision output
// distributed with the NRL Fortran code: 200 input sets, densities in cm^-3
// and mass density in g/cm^3 printed to four significant figures, temperature
// in K to 0.01 K. Every value must match to half its last printed digit (plus
// 1e-6 relative for rounding ties). A looser check once let two constants
// differ from msis_constants.F90 (the surface mixing ratios and the
// anomalous-O scale height) while the port was said to match.

#include "../test_utils.hpp"
#include <ballistic/atmosphere/internal/nrlmsis21_full.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

using namespace ballistic::test;
using ballistic::atmosphere::NRLMSIS21;

int main() {
    TestRunner runner("NRLMSIS 2.1 reference output");

    runner.add_test("official_reference_vectors", []() {
        const std::string path = std::string(BALLISTIC_SOURCE_DIR) + "/tests/data/nrlmsis/msis2.1_test_ref_dp.txt";
        std::ifstream file(path);
        TEST_ASSERT_MSG(file.is_open(), "cannot open " << path);

        // File columns He O N2 O2 Ar rho H N OA NO -> Output::dn index.
        const char* names[10] = {"He", "O", "N2", "O2", "Ar", "rho", "H", "N", "OA", "NO"};
        const int dn_index[10] = {5, 4, 2, 3, 7, 1, 6, 8, 9, 10};
        const double si_per_file[10] = {1e6, 1e6, 1e6, 1e6, 1e6, 1e3, 1e6, 1e6, 1e6, 1e6};

        NRLMSIS21 model;
        std::string line;
        std::getline(file, line);  // header
        int rows = 0;
        double worst[11] = {};     // largest error per column, in printed half-digits
        while (std::getline(file, line)) {
            std::istringstream in(line);
            double iyd, sec, alt, lat, lon, stl, f107a, f107, ap, temperature;
            if (!(in >> iyd >> sec >> alt >> lat >> lon >> stl >> f107a >> f107 >> ap)) continue;
            std::string token[10];
            for (auto& t : token) in >> t;
            in >> temperature;

            NRLMSIS21::Input input;
            input.day = std::fmod(iyd, 1000.0);
            input.utsec = sec;
            input.alt = alt;
            input.lat = lat;
            input.lon = lon;
            input.f107a = f107a;
            input.f107 = f107;
            input.ap.fill(ap);
            const auto out = model.msiscalc(input);

            for (int k = 0; k < 10; ++k) {
                const double ref = std::stod(token[k]);
                if (ref < 1e-30) continue;  // 0.9999E-37: not computed at this altitude
                const int exponent = std::stoi(token[k].substr(token[k].find('E') + 1));
                const double half_digit = 0.5 * std::pow(10.0, exponent - 4);
                const double got = out.dn[dn_index[k]] / si_per_file[k];
                worst[k] = std::max(worst[k], std::abs(got - ref) / half_digit);
                TEST_ASSERT_MSG(std::abs(got - ref) <= half_digit + 1e-6 * ref,
                                "row " << rows << " " << names[k] << ": " << got << " vs " << token[k]);
            }
            worst[10] = std::max(worst[10], std::abs(out.tn - temperature) / 0.005);
            TEST_ASSERT_MSG(std::abs(out.tn - temperature) <= 0.005 + 1e-6 * temperature,
                            "row " << rows << " T: " << out.tn << " vs " << temperature);
            ++rows;
        }
        TEST_ASSERT_EQ(rows, 200);
        std::cout << "  largest error in printed half-digits:";
        for (int k = 0; k < 10; ++k) std::cout << " " << names[k] << " " << worst[k];
        std::cout << " T " << worst[10] << "\n";
    });

    runner.run();
    return print_final_summary();
}
