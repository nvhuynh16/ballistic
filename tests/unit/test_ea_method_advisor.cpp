// =============================================================================
// Unit test: recommend_ea_method, the advisor that suggests how to estimate an
// empirical acceleration (EA: the extra acceleration the force model is missing,
// fitted from observations). It is a pure function (no estimator runs), so each
// cell below checks one row of its decision table: which estimator, whether to
// add Gauss-Newton refinement (GN) and whether to pick the smoothing weight by
// cross-validation (CV). POS / VEL / ACC = position / velocity / accelerometer
// observations.
// =============================================================================

#include "../test_utils.hpp"
#include <ballistic/calibration/ea_method_advisor.hpp>

using namespace ballistic;
using namespace ballistic::calibration;
using namespace ballistic::test;

namespace {

EAScenarioHint hint(EAObsType o, double sigma = 0.0,
                    bool aggr = false, bool known = false, int nr = 1) {
    EAScenarioHint h;
    h.obs = o; h.noise_sigma = sigma;
    h.aggressive_dynamics = aggr; h.params_known = known; h.n_radars = nr;
    return h;
}

}  // namespace

int main() {
    TestRunner runner("ea_method_advisor");

    // Clean positions: B-spline estimator with GN, no CV (nothing to tune noise for).
    runner.add_test("pos_clean_benign_bspline_gn", []() {
        auto r = recommend_ea_method(hint(EAObsType::Position));
        TEST_ASSERT(r.method == EAMethod::BSplinePosition);
        TEST_ASSERT(r.use_gn);
        TEST_ASSERT(!r.use_cv);
        TEST_ASSERT(!r.rationale.empty());
    });

    // Aggressive dynamics (reentry/glide) do not change the advice for clean positions.
    runner.add_test("pos_clean_aggressive_bspline", []() {
        auto r = recommend_ea_method(
            hint(EAObsType::Position, 0.0, /*aggr=*/true));
        TEST_ASSERT(r.method == EAMethod::BSplinePosition);
        TEST_ASSERT(r.use_gn);
        TEST_ASSERT(!r.use_cv);
    });

    // Noisy positions whose feature time scale is known: the fit settings follow from it,
    // so no CV.
    runner.add_test("pos_noisy_known_bspline_gn_no_cv", []() {
        auto r = recommend_ea_method(
            hint(EAObsType::Position, 3.0, false, /*known=*/true));
        TEST_ASSERT(r.method == EAMethod::BSplinePosition);
        TEST_ASSERT(r.use_gn);
        TEST_ASSERT(!r.use_cv);
    });

    // Noisy positions, feature time scale unknown: CV chooses the fit settings.
    runner.add_test("pos_noisy_unknown_bspline_gn_cv", []() {
        auto r = recommend_ea_method(hint(EAObsType::Position, 3.0));
        TEST_ASSERT(r.method == EAMethod::BSplinePosition);
        TEST_ASSERT(r.use_gn);
        TEST_ASSERT(r.use_cv);
    });

    runner.add_test("vel_clean_bspline_gn", []() {
        auto r = recommend_ea_method(hint(EAObsType::Velocity));
        TEST_ASSERT(r.method == EAMethod::BSplineVelocity);
        // GN helps VEL as well (measured over multiple seeds); the
        // regularized GN does not destabilize velocity fits.
        TEST_ASSERT(r.use_gn);
        TEST_ASSERT(!r.use_cv);
    });

    runner.add_test("vel_noisy_bspline_cv_gn", []() {
        auto r = recommend_ea_method(hint(EAObsType::Velocity, 0.25));
        TEST_ASSERT(r.method == EAMethod::BSplineVelocity);
        TEST_ASSERT(r.use_gn);
        TEST_ASSERT(r.use_cv);
    });

    // Accelerometer data: no GN, because GN would fit the EA against the same
    // accelerations it was estimated from (circular).
    runner.add_test("acc_clean_bspline", []() {
        auto r = recommend_ea_method(hint(EAObsType::Acceleration));
        TEST_ASSERT(r.method == EAMethod::BSplineAcceleration);
        TEST_ASSERT(!r.use_gn);   // circular for accelerometer data
        TEST_ASSERT(!r.use_cv);
    });

    runner.add_test("acc_noisy_bspline_cv", []() {
        auto r = recommend_ea_method(hint(EAObsType::Acceleration, 0.01));
        TEST_ASSERT(r.method == EAMethod::BSplineAcceleration);
        TEST_ASSERT(!r.use_gn);
        TEST_ASSERT(r.use_cv);
    });

    runner.add_test("radar_single_recommends_smoother", []() {
        // Radar alone goes to the smoother (measured 12.7 m on the synthetic glide).
        // With the range-rate channel weighted by its measured honest error budget,
        // no rule about turning range-rate off or choosing a special weight is needed.
        auto r = recommend_ea_method(hint(EAObsType::Radar, 0.0, false,
                                          false, /*nr=*/1));
        TEST_ASSERT(r.method == EAMethod::Smoother);
        TEST_ASSERT(!r.use_gn);
        TEST_ASSERT(!r.use_cv);
        TEST_ASSERT(r.rationale.find("smoother") != std::string::npos);
    });

    // Two radars also go to the smoother; the rationale quotes the value
    // test_ea_smoother::two_radars_beat_one measures (10.6 m) and says where.
    runner.add_test("radar_two_recommends_smoother", []() {
        auto r = recommend_ea_method(hint(EAObsType::Radar, 0.0, false,
                                          false, /*nr=*/2));
        TEST_ASSERT(r.method == EAMethod::Smoother);
        TEST_ASSERT(r.rationale.find("10.6 m") != std::string::npos);
        TEST_ASSERT(r.rationale.find("measured on the test geometry") != std::string::npos);
    });

    // Numbers in the rationales are the ones the test suite measures; prose
    // replaces those it does not (a stale quote once said 0.006 m, "5/6
    // cells", 0.11-0.22 m and 0.53-1.78 -> 0.23-0.44 m).
    runner.add_test("rationales_quote_only_measured_numbers", []() {
        auto clean_pos = recommend_ea_method(hint(EAObsType::Position));
        TEST_ASSERT(clean_pos.rationale.find("millimeter") != std::string::npos);
        auto single = recommend_ea_method(hint(EAObsType::Radar));
        TEST_ASSERT(single.rationale.find("12.7 m measured on the test geometry")
                    != std::string::npos);
        auto fusion = recommend_ea_method(hint(EAObsType::MultiSensor));
        TEST_ASSERT(fusion.rationale.find("12.7 m on combo 1") != std::string::npos);
        for (const auto& h : {hint(EAObsType::Position), hint(EAObsType::Position, 3.0),
                              hint(EAObsType::Velocity, 0.25), hint(EAObsType::Acceleration)}) {
            const auto r = recommend_ea_method(h);
            for (const char* stale : {"0.006", "5/6", "0.11", "1.78", "0.23-0.44"})
                TEST_ASSERT_MSG(r.rationale.find(stale) == std::string::npos,
                                std::string("stale figure ") + stale + " in: " + r.rationale);
        }
    });

    runner.add_test("multisensor_recommends_smoother", []() {
        // The smoother (a Gauss-Markov Rauch-Tung-Striebel smoother: a Kalman filter
        // run forward, then backward) fuses radar, optics, position fixes and
        // accelerometers. Its tuning is picked automatically from the normalized
        // innovation squared (NIS: do the residuals match their declared noise?), so
        // no CV is needed.
        auto r = recommend_ea_method(hint(EAObsType::MultiSensor));
        TEST_ASSERT(r.method == EAMethod::Smoother);
        TEST_ASSERT(!r.use_cv);
        TEST_ASSERT(r.rationale.find("smoother") != std::string::npos);
    });

    runner.run();
    return print_final_summary();
}
