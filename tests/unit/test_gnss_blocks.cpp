// GNSS block radiation table (include/ballistic/gnss_blocks.hpp).
//
// Every block name the IGS satellite metadata file uses for the GPS, QZSS and
// BeiDou satellites the campaigns cover must resolve to a row (the campaign
// would otherwise stop on that satellite); the prefix rule must pick the
// longest matching name; each fitted scale must name a real block and be
// applied to it.

#include "../test_utils.hpp"
#include <ballistic/gnss_blocks.hpp>

#include <string>
#include <string_view>

using namespace ballistic;
using namespace ballistic::test;

namespace {

// Block strings as they appear in the IGS file (SATELLITE/IDENTIFIER).
constexpr std::string_view kIgsNames[] = {
    "GPS-IIR-A", "GPS-IIR-B", "GPS-IIR-M", "GPS-IIF", "GPS-IIIA",
    "QZS-1", "QZS-2I", "QZS-2A", "QZS-2G", "QZS-3G",
    "BDS-2G", "BDS-2I", "BDS-2M", "BDS-3M-CAST", "BDS-3M-SECM-A", "BDS-3M-SECM-B",
    "BDS-3I", "BDS-3G",
};

}  // namespace

int main() {
    TestRunner runner("GNSS blocks");

    runner.add_test("every_igs_block_name_resolves", []() {
        for (auto n : kIgsNames) {
            const gnss::Block* b = gnss::find_block(n);
            TEST_ASSERT_MSG(b != nullptr, std::string("no row for ") + std::string(n));
            TEST_ASSERT(n.starts_with(b->name));
        }
    });

    runner.add_test("prefix_rule", []() {
        static_assert(gnss::find_block("GPS-IIR-A")->name == "GPS-IIR");
        static_assert(gnss::find_block("GPS-IIR-B")->name == "GPS-IIR");
        // an exact name wins over a shorter prefix
        static_assert(gnss::find_block("GPS-IIR-M")->name == "GPS-IIR-M");
        static_assert(gnss::find_block("BDS-3M-SECM-A")->name == "BDS-3M-SECM");
        // a prefix counts only at a '-' boundary
        TEST_ASSERT(gnss::find_block("GPS-IIRX") == nullptr);
        TEST_ASSERT(gnss::find_block("GPS") == nullptr);
        TEST_ASSERT(gnss::find_block("") == nullptr);
        TEST_ASSERT_THROWS((void)gnss::get_block("GAL-2"), ConfigError);
        TEST_ASSERT_EQ(std::string(gnss::get_block("QZS-2I").name), std::string("QZS-2I"));
    });

    runner.add_test("fit_table_consistent_and_applied", []() {
        static_assert(gnss::srp_fit_consistent());
        for (const auto& f : gnss::kGnssSrpFit) {
            const gnss::Block& b = gnss::get_block(f.block);
            TEST_ASSERT_NEAR(b.srp_k, f.k, 0.0);
            TEST_ASSERT(b.srp_kind == satellites::FitKind::Fitted);
            TEST_ASSERT_NEAR(b.srp_area_m2(), b.srp_base_area_m2 * f.k, 1e-12);
        }
        // unfitted rows keep k = 1 and their own kind
        for (const auto& b : gnss::all()) {
            bool fitted = false;
            for (const auto& f : gnss::kGnssSrpFit) fitted = fitted || f.block == b.name;
            if (!fitted) {
                TEST_ASSERT_NEAR(b.srp_k, 1.0, 0.0);
                TEST_ASSERT(b.srp_kind != satellites::FitKind::Fitted);
            }
        }
    });

    runner.add_test("rows_well_formed", []() {
        for (const auto& b : gnss::all()) {
            TEST_ASSERT(b.srp_base_area_m2 > 0.0 && b.srp_base_area_m2 < 100.0);
            TEST_ASSERT(b.srp_area_m2() > 0.0);
            TEST_ASSERT(b.cr >= 1.0 && b.cr <= 2.0);
            TEST_ASSERT(b.mass_kg_nominal >= 0.0);
            TEST_ASSERT(!b.provenance.empty());
            // a placeholder must say so
            if (b.srp_kind == satellites::FitKind::Nominal)
                TEST_ASSERT_MSG(b.provenance.find("ASSUMED") != std::string_view::npos ||
                                    b.provenance.find("not checked") != std::string_view::npos ||
                                    b.provenance.find("estimate") != std::string_view::npos,
                                std::string(b.name) + ": nominal row without a stated assumption");
        }
    });

    runner.run();
    return print_final_summary();
}
