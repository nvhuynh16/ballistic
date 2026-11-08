// Diagnostics channel and error taxonomy (core/diagnostics.hpp, core/errors.hpp).
//
// The channel is how the library avoids silent substitutions: every
// approximation it continues with is reported here. These cells pin the
// contract: delivery, dedup by key, strict mode, the error hierarchy, and
// that a Trajectory's notes reach the channel (except a library-internal
// trajectory's notes about its own empirical acceleration).

#include <ballistic/time/eop_data.hpp>
#include "../test_utils.hpp"
#include <ballistic/ballistic.hpp>
#include <ballistic/core/diagnostics.hpp>
#include <ballistic/core/errors.hpp>

#include <string>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;

namespace {

struct Capture {
    std::vector<diagnostics::Diagnostic> got;
    Capture() {
        diagnostics::Sink::instance().reset_dedup();
        diagnostics::set_handler([this](const diagnostics::Diagnostic& d) { got.push_back(d); });
    }
    ~Capture() { diagnostics::set_handler(nullptr); }
};

}  // namespace

int main() {
    // Runs without an Earth orientation table on purpose (synthetic or
    // self-consistent checks; pinned values assume zero UT1-UTC/polar motion).
    ballistic::time::EOPData::instance().set_policy(ballistic::time::EopPolicy::AssumeZero);
    TestRunner runner("Diagnostics channel");

    runner.add_test("delivered_with_category", [] {
        Capture c;
        diagnostics::report(diagnostics::Category::ConfigHint, "hello");
        TEST_ASSERT(c.got.size() == 1);
        TEST_ASSERT(c.got[0].category == diagnostics::Category::ConfigHint);
        TEST_ASSERT(c.got[0].message == "hello");
    });

    runner.add_test("keyed_reports_are_delivered_once", [] {
        Capture c;
        diagnostics::report(diagnostics::Category::DataCoverage, "a", "k");
        diagnostics::report(diagnostics::Category::DataCoverage, "a again", "k");
        diagnostics::report(diagnostics::Category::ModelSubstitution, "same key, other category", "k");
        diagnostics::report(diagnostics::Category::DataCoverage, "unkeyed");
        diagnostics::report(diagnostics::Category::DataCoverage, "unkeyed");
        TEST_ASSERT(c.got.size() == 4);
    });

    runner.add_test("strict_mode_throws_without_a_handler", [] {
        diagnostics::set_handler(nullptr);
        diagnostics::set_strict(true);
        bool threw = false;
        try {
            diagnostics::report(diagnostics::Category::ForcedStep, "step forced");
        } catch (const StrictDiagnosticError& e) {
            threw = std::string(e.what()).find("ForcedStep") != std::string::npos;
        }
        diagnostics::set_strict(false);
        TEST_ASSERT(threw);
    });

    runner.add_test("strict_mode_defers_to_an_installed_handler", [] {
        Capture c;
        diagnostics::set_strict(true);
        diagnostics::report(diagnostics::Category::ConfigHint, "handled");
        diagnostics::set_strict(false);
        TEST_ASSERT(c.got.size() == 1);
    });

    runner.add_test("only_config_hints_are_advisory", [] {
        TEST_ASSERT(diagnostics::is_advisory(diagnostics::Category::ConfigHint));
        TEST_ASSERT(!diagnostics::is_advisory(diagnostics::Category::DataCoverage));
        TEST_ASSERT(!diagnostics::is_advisory(diagnostics::Category::ModelSubstitution));
    });

    runner.add_test("error_hierarchy", [] {
        bool caught_as_base = false, caught_as_runtime = false;
        try { throw DataCoverageError("x"); } catch (const Error&) { caught_as_base = true; }
        try { throw ConfigError("x"); } catch (const std::runtime_error&) { caught_as_runtime = true; }
        TEST_ASSERT(caught_as_base && caught_as_runtime);
    });

    runner.add_test("trajectory_notes_reach_the_channel", [] {
        // No EOP (Earth orientation parameters) is loaded in this process: the
        // trajectory notes it, keyed, so a second trajectory does not repeat it
        // on the channel but still records it in its own warnings().
        Capture c;
        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Trajectory a(t0, Vector3d(6.778e6, 0, 0), Vector3d(0, 7.5e3, 0));
        Trajectory b(t0, Vector3d(6.778e6, 0, 0), Vector3d(0, 7.5e3, 0));
        int eop = 0;
        for (const auto& d : c.got)
            if (d.message.find("Earth orientation") != std::string::npos) ++eop;
        TEST_ASSERT(eop == 1);
        TEST_ASSERT(!a.warnings().empty() && !b.warnings().empty());
    });

    runner.add_test("internal_trajectory_keeps_its_extrapolation_notes", [] {
        // An empirical acceleration (EA) table covering 10 s, propagated 60 s: a
        // user trajectory reports the extrapolation, a library-internal one only
        // records it.
        std::vector<forces::EmpiricalAccelEntry> entries(11);
        for (int i = 0; i <= 10; ++i) { entries[i].key = double(i); entries[i].accel = Vector3d(1e-6, 0, 0); }
        auto ea = std::make_shared<forces::EmpiricalAcceleration>(
            forces::EmpiricalAcceleration::from_entries(entries, forces::EmpiricalAccelFrame::GCRS));
        TrajectoryConfig cfg;
        cfg.empirical_accel = ea;
        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        auto count_extrap = [](const std::vector<diagnostics::Diagnostic>& v) {
            int n = 0;
            for (const auto& d : v) n += d.category == diagnostics::Category::Extrapolation;
            return n;
        };
        {
            Capture c;
            Trajectory user(t0, Vector3d(6.778e6, 0, 0), Vector3d(0, 7.5e3, 0), cfg);
            user.propagate(t0 + 60.0);
            TEST_ASSERT(count_extrap(c.got) == 1);
        }
        {
            Capture c;
            Trajectory internal(t0, Vector3d(6.778e6, 0, 0), Vector3d(0, 7.5e3, 0), cfg);
            internal.mark_internal();
            internal.propagate(t0 + 60.0);
            TEST_ASSERT(count_extrap(c.got) == 0);
            bool recorded = false;
            for (const auto& w : internal.warnings())
                recorded |= w.find("empirical acceleration") != std::string::npos;
            TEST_ASSERT(recorded);
        }
    });

    runner.run();
    return print_final_summary();
}
