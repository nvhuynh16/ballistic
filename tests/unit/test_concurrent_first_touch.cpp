// The embedded gravity model (EGM2008) and ephemeris (DE440) load themselves
// the first time a force model needs them. This test makes that first use
// happen on eight threads at once: before any other library call that could
// load them, eight threads each build a Trajectory (gravity degree 70 + Sun,
// Moon and Jupiter) and propagate it 600 s. Then the same propagation runs on
// the main thread; all eight results must equal it bit for bit. Without the
// locks around the first load, threads could compute with a half-filled
// coefficient table or load the ephemeris twice while another thread reads it.
//
// The Earth orientation table is the committed one, loaded directly (it does
// not touch gravity or the ephemeris), so the propagations issue no warnings.

#include "../test_utils.hpp"
#include <ballistic/time/eop_data.hpp>
#include <ballistic/trajectory.hpp>
#include <ballistic/initial_state.hpp>
#include <array>
#include <thread>

using namespace ballistic;
using namespace ballistic::test;

namespace {

TrajectoryConfig config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 70;
    cfg.enable_third_body = true;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    cfg.ocean_tide_degree = 0;
    return cfg;
}

StateVector run_once() {
    const Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    // Circular 400 km orbit, Earth-fixed velocity (7669 m/s inertial minus
    // the Earth's rotation).
    const double r = 6778137.0;
    Trajectory traj(InitialState::from_itrf(t0, Vector3d(r, 0.0, 0.0),
                                            Vector3d(0.0, 7175.0, 0.0)),
                    config());
    const PropagationResult p = traj.propagate(t0 + 600.0);
    return StateVector{p.position, p.velocity};
}

bool same(const StateVector& a, const StateVector& b) {
    return a.position == b.position && a.velocity == b.velocity;
}

}  // namespace

int main() {
    // Nothing else touches the library before the threads start.
    if (!time::EOPData::instance().load(BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all")) {
        std::cerr << "could not load the committed EOP table\n";
        return 1;
    }

    constexpr int kThreads = 8;
    std::array<StateVector, kThreads> results;
    std::array<std::string, kThreads> errors;
    {
        std::vector<std::thread> pool;
        for (int i = 0; i < kThreads; ++i) {
            pool.emplace_back([&, i] {
                try {
                    results[i] = run_once();
                } catch (const std::exception& e) {
                    errors[i] = e.what();
                }
            });
        }
        for (auto& t : pool) t.join();
    }
    const StateVector serial = run_once();

    TestRunner runner("concurrent first use of gravity and ephemeris");
    runner.add_test("eight_threads_equal_serial", [&]() {
        for (int i = 0; i < kThreads; ++i) {
            TEST_ASSERT_MSG(errors[i].empty(),
                            "thread " + std::to_string(i) + " threw: " + errors[i]);
            TEST_ASSERT_MSG(same(results[i], serial),
                            "thread " + std::to_string(i) + " differs from the serial run by " +
                                std::to_string((results[i].position - serial.position).norm()) +
                                " m");
        }
        // The orbit actually moved (not a trivially equal default state).
        TEST_ASSERT(serial.position.norm() > 6.7e6 && serial.velocity.norm() > 7000.0);
    });
    runner.run();
    return print_final_summary();
}
