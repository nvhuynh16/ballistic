// Concurrency under realistic configurations. Separate Trajectory objects may
// be propagated on separate threads; what they share (the drag environment,
// the gravity and ephemeris tables, the loaded space weather and Earth
// orientation tables) must not change their results. Every cell compares the
// threaded results with a serial run, bit for bit:
//   (a) eight threads share ONE live drag environment (NRLMSIS 2.1 + HWM14 on
//       the committed weather file) plus radiation pressure;
//   (b) four threads run estimate_ea_bspline on the glide fixture while four
//       others propagate with drag;
//   (c) one thread reads the loaded space weather (get_record, history) in a
//       loop while others propagate with drag.
// Ocean tides stay off: their FES2004 file is downloaded, not committed.
//
// Data: the committed Earth orientation table (tests/data/eop) and weather
// file (tests/data/weather/SW-All.csv), loaded directly so the test never
// reads the user's data directory and issues no warnings.

#include "../test_helpers.hpp"
#include "../test_utils.hpp"
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/atmosphere/space_weather.hpp>
#include <ballistic/calibration/ea_estimator.hpp>
#include <ballistic/satellites.hpp>
#include <ballistic/time/eop_data.hpp>
#include <atomic>
#include <thread>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;
namespace atm = ballistic::atmosphere;
namespace glide_helpers = ballistic::test::helpers::glide;

namespace {

const Epoch& t0() {
    static const Epoch e = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    return e;
}
// Circular 400 km orbit, Earth-fixed velocity, inclined 51.6 deg.
const Vector3d kPos(6778137.0, 0.0, 0.0);
const Vector3d kVel(0.0, 7669.0 * 0.6211 - 494.3, 7669.0 * 0.7837);

// Gravity degree 70 + third body + solid tides (Step 1 and 2) + relativity,
// Sentinel-1A radiation pressure, and the given drag environment.
TrajectoryConfig realistic(std::shared_ptr<const atm::DragEnvironment> env) {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 70;
    cfg.ocean_tide_degree = 0;
    const auto& s1 = satellites::get("S1A");
    cfg.srp = SrpConfig{.mass_kg = s1.mass_kg, .area_m2 = s1.srp_area_m2, .cr = s1.cr()};
    cfg.drag_env = std::move(env);
    return cfg;
}

std::shared_ptr<const atm::DragEnvironment> shared_env() {
    static const auto env = atm::make_drag_environment(
        t0(), kPos, kVel, 0.0, realistic(nullptr),
        {.beta = atm::BetaProfile1D::constant(satellites::get("S1A").beta_kg_m2())});
    return env;
}

// One propagation of `duration_s` with the shared drag environment; the
// k-th thread starts k seconds later so the threads do not run in lockstep.
StateVector propagate(double duration_s, int k = 0,
                      std::shared_ptr<const atm::DragEnvironment> env = shared_env()) {
    const Epoch start = t0() + static_cast<double>(k);
    Trajectory tr(InitialState::from_itrf(start, kPos, kVel), realistic(std::move(env)));
    const PropagationResult p = tr.propagate(start + duration_s);
    return StateVector{p.position, p.velocity};
}

bool same(const StateVector& a, const StateVector& b) {
    return a.position == b.position && a.velocity == b.velocity;
}

// Run f(i) on n threads, collecting exception messages.
template <class F>
std::vector<std::string> run_threads(int n, F f) {
    std::vector<std::string> errors(n);
    std::vector<std::thread> pool;
    for (int i = 0; i < n; ++i) {
        pool.emplace_back([&, i] {
            try { f(i); } catch (const std::exception& e) { errors[i] = e.what(); }
        });
    }
    for (auto& t : pool) t.join();
    return errors;
}

void assert_no_errors(const std::vector<std::string>& errors) {
    for (std::size_t i = 0; i < errors.size(); ++i)
        TEST_ASSERT_MSG(errors[i].empty(), "thread " + std::to_string(i) + " threw: " + errors[i]);
}

}  // namespace

int main() {
    if (!time::EOPData::instance().load(BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all")) {
        std::cerr << "could not load the committed EOP table\n";
        return 1;
    }
    auto& sw = atm::SpaceWeatherData::instance();
    sw.clear();
    if (!sw.load_celestrak_csv(BALLISTIC_PROJECT_ROOT "/tests/data/weather/SW-All.csv")) {
        std::cerr << "could not load the committed weather file\n";
        return 1;
    }

    TestRunner runner("concurrency");

    runner.add_test("eight_threads_share_one_live_drag_env", []() {
        constexpr int n = 8;
        std::vector<StateVector> serial(n), threaded(n);
        for (int k = 0; k < n; ++k) serial[k] = propagate(900.0, k);
        assert_no_errors(run_threads(n, [&](int k) { threaded[k] = propagate(900.0, k); }));
        for (int k = 0; k < n; ++k)
            TEST_ASSERT_MSG(same(threaded[k], serial[k]),
                            "thread " + std::to_string(k) + " differs by " +
                                std::to_string((threaded[k].position - serial[k].position).norm()) +
                                " m");
        // Drag is really on: the same start without it ends elsewhere.
        TrajectoryConfig no_drag = realistic(nullptr);
        Trajectory tr(InitialState::from_itrf(t0(), kPos, kVel), no_drag);
        const PropagationResult p = tr.propagate(t0() + 900.0);
        TEST_ASSERT((p.position - serial[0].position).norm() > 1e-3);
    });

    runner.add_test("estimators_and_propagations_together", []() {
        // A short glide arc: 10 Hz positions over the first 120 s (σ 1 m,
        // the fixture's noisy set), IRGN fit without GN to keep it short.
        const auto glide = glide_helpers::make_glide();
        std::vector<std::pair<Epoch, Vector3d>> obs;
        for (std::size_t i = 0; i < glide.pos_n100.size(); i += 10)
            if (glide.pos_n100[i].first - glide.t0 <= 120.0) obs.push_back(glide.pos_n100[i]);
        calibration::EABSplineConfig cfg;
        cfg.gn = std::nullopt;
        cfg.noise_sigma = 1.0;
        const TrajectoryConfig fm = glide_helpers::base_config();
        auto estimate = [&] {
            return calibration::estimate_ea_bspline(glide.t0, glide.pos, glide.vel,
                                                    calibration::PositionSamples{obs}, fm, cfg);
        };
        auto sample = [](const calibration::EAResult& r) {
            std::vector<Vector3d> out;
            if (!r.empirical_accel) return out;
            for (double t = 0.0; t <= 120.0; t += 7.5) out.push_back(r.empirical_accel->accel_at(t));
            return out;
        };
        const auto ref_ea = sample(estimate());
        TEST_ASSERT(!ref_ea.empty());
        const StateVector ref_prop = propagate(600.0);

        std::vector<std::vector<Vector3d>> eas(4);
        std::vector<StateVector> props(4);
        assert_no_errors(run_threads(8, [&](int i) {
            if (i < 4) eas[i] = sample(estimate());
            else props[i - 4] = propagate(600.0);
        }));
        for (int i = 0; i < 4; ++i) {
            TEST_ASSERT_MSG(eas[i] == ref_ea, "estimator thread " + std::to_string(i) +
                                                  " differs from the serial fit");
            TEST_ASSERT_MSG(same(props[i], ref_prop), "propagation thread " +
                                                          std::to_string(i) + " differs");
        }
    });

    runner.add_test("weather_reader_during_drag_propagation", []() {
        constexpr int n = 4;
        const StateVector ref = propagate(900.0);
        const double mjd = t0().mjd(time::TimeScale::UTC);
        std::atomic<int> running{n};
        std::atomic<long> reads{0};
        std::string reader_error;
        std::thread reader([&] {
            try {
                auto& w = atm::SpaceWeatherData::instance();
                while (running.load() > 0) {
                    const auto r = w.get_record(std::floor(mjd) + 0.5);
                    const auto h = w.history(atm::TimeWindow{t0(), t0() + 3600.0});
                    if (!r || !(r->f107_obs > 0.0) || !(h.at(t0()).f107_daily > 0.0))
                        throw std::runtime_error("weather read returned no data");
                    ++reads;
                }
            } catch (const std::exception& e) { reader_error = e.what(); }
        });
        std::vector<StateVector> out(n);
        const auto errors = run_threads(n, [&](int i) {
            out[i] = propagate(900.0);
            --running;
        });
        reader.join();
        assert_no_errors(errors);
        TEST_ASSERT_MSG(reader_error.empty(), "reader threw: " + reader_error);
        TEST_ASSERT(reads.load() > 0);
        for (int i = 0; i < n; ++i)
            TEST_ASSERT_MSG(same(out[i], ref), "thread " + std::to_string(i) + " differs");
    });

    runner.run();
    return print_final_summary();
}
