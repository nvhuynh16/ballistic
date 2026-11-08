// Unit tests for the CV argmin tie-breaker.
//
// Pins the deterministic behavior of `pick_best_grid_idx`:
//   - Strictly-better picks update best_idx
//   - Near-ties (within tie_eps) prefer larger knot_spacing (Occam's razor)
//   - Knot-tied sub-ties keep first-seen index (deterministic fallback)
//   - Differences beyond tie_eps behave like the original strict-less-than

#include <ballistic/calibration/ea_estimator.hpp>  // brings ea_cv.hpp + helpers
#include "../test_utils.hpp"

using ballistic::calibration::detail::CVHyperParams;
using ballistic::calibration::detail::pick_best_grid_idx;
using ballistic::calibration::detail::CV_TIE_EPS_M;
using namespace ballistic::test;

void test_pick_best_grid_idx() {
    TestRunner runner("pick_best_grid_idx");

    runner.add_test("strict better wins by margin", [] {
        std::vector<CVHyperParams> grid = {
            {1.0, 1.0, 0.5, 1e-12},
            {1.0, 1.0, 0.5, 1e-12},
            {1.0, 1.0, 0.5, 1e-12},
        };
        std::vector<double> holdout = {1.000, 0.500, 1.500};
        TEST_ASSERT_EQ(pick_best_grid_idx(holdout, grid), 1);
    });

    runner.add_test("near-tie picks larger knot (Occam)", [] {
        std::vector<CVHyperParams> grid = {
            {1.0, 1.0, 0.5, 1e-12},  // smaller knot at idx 0
            {2.0, 1.0, 0.5, 1e-12},  // larger knot at idx 1
        };
        // 0.5 mm difference — well within 1 mm TIE_EPS — treated as tie
        std::vector<double> holdout = {1.0000, 1.0005};
        TEST_ASSERT_EQ(pick_best_grid_idx(holdout, grid), 1);  // larger knot wins
    });

    runner.add_test("near-tie picks larger knot (reverse order)", [] {
        std::vector<CVHyperParams> grid = {
            {2.0, 1.0, 0.5, 1e-12},  // larger knot at idx 0
            {1.0, 1.0, 0.5, 1e-12},  // smaller knot at idx 1
        };
        std::vector<double> holdout = {1.0005, 1.0000};
        // idx 1 is "strictly better by ULP" but within tie_eps
        TEST_ASSERT_EQ(pick_best_grid_idx(holdout, grid), 0);  // larger knot wins
    });

    runner.add_test("knot-tied near-tie keeps first-seen", [] {
        std::vector<CVHyperParams> grid = {
            {2.0, 1.0,  0.5, 1e-12},  // same knot
            {2.0, 0.1,  0.5, 1e-12},  // same knot
            {2.0, 0.01, 0.5, 1e-12},  // same knot
        };
        std::vector<double> holdout = {1.0000, 1.0003, 0.9998};
        // All within 1mm. idx 2 has lower score by ULP; should NOT
        // promote (within tie_eps), so first-seen idx 0 stays.
        TEST_ASSERT_EQ(pick_best_grid_idx(holdout, grid), 0);
    });

    runner.add_test("difference beyond tie_eps behaves like strict <", [] {
        std::vector<CVHyperParams> grid = {
            {2.0, 1.0, 0.5, 1e-12},  // larger knot
            {1.0, 1.0, 0.5, 1e-12},  // smaller knot
        };
        // 5 mm difference — well beyond 1 mm tie_eps — strict argmin wins
        std::vector<double> holdout = {1.000, 0.995};
        TEST_ASSERT_EQ(pick_best_grid_idx(holdout, grid), 1);  // smaller-knot, strictly better
    });

    runner.add_test("ULP-level difference treated as tie (cross-binary case)", [] {
        // Reproduces the cross-binary variance scenario: same grid points,
        // but two binaries assign slightly different holdout values.
        std::vector<CVHyperParams> grid = {
            {12.0, 1.0, 0.5, 1e-12},  // larger knot
            {2.0,  1.0, 0.5, 1e-12},  // smaller knot
        };
        // Two "binaries": A picks idx 0 (smaller score by 1e-10);
        // B picks idx 1 (smaller score by 1e-10).
        std::vector<double> holdout_A = {1.0000000001, 1.0000000002};
        std::vector<double> holdout_B = {1.0000000002, 1.0000000001};
        // BOTH should resolve to idx 0 (larger knot) since both
        // ULP differences are within tie_eps.
        TEST_ASSERT_EQ(pick_best_grid_idx(holdout_A, grid), 0);
        TEST_ASSERT_EQ(pick_best_grid_idx(holdout_B, grid), 0);
    });

    runner.add_test("degenerate single-element grid", [] {
        std::vector<CVHyperParams> grid = {{1.0, 1.0, 0.5, 1e-12}};
        std::vector<double> holdout = {2.5};
        TEST_ASSERT_EQ(pick_best_grid_idx(holdout, grid), 0);
    });

    runner.add_test("custom tie_eps=0 reverts to strict <", [] {
        std::vector<CVHyperParams> grid = {
            {2.0, 1.0, 0.5, 1e-12},  // larger knot
            {1.0, 1.0, 0.5, 1e-12},  // smaller knot
        };
        std::vector<double> holdout = {1.0000, 1.0000 - 1e-12};  // ULP-level
        // With tie_eps=0, strict less-than wins → idx 1
        TEST_ASSERT_EQ(pick_best_grid_idx(holdout, grid, 0.0), 1);
        // With default tie_eps=1mm, this is a tie → larger knot at idx 0
        TEST_ASSERT_EQ(pick_best_grid_idx(holdout, grid), 0);
    });

    runner.run();
}

int main() {
    test_pick_best_grid_idx();
    return print_final_summary();
}
