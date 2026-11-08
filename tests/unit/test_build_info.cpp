// build_info() reports how the library was compiled. (The NRL atmosphere and
// wind models are a run-time plugin, not a build property: test_nrl_models.)

#include "../test_utils.hpp"
#include <ballistic/ballistic.hpp>
#include <ballistic/core/build_info.hpp>
#include <string>

using namespace ballistic;
using namespace ballistic::test;

int main() {
    InitConfig init;
    init.eop_path = BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all";
    initialize(init);

    TestRunner runner("build info");

    runner.add_test("fields_match_the_macros", []() {
        constexpr BuildInfo b = build_info();   // usable in constant expressions
        static_assert(b.cpp_standard >= 23);
#ifdef _OPENMP
        TEST_ASSERT(b.with_openmp);
#else
        TEST_ASSERT(!b.with_openmp);
#endif
#ifdef NDEBUG
        TEST_ASSERT(b.build_type == "Release");
#else
        TEST_ASSERT(b.build_type == "Debug");
#endif
        TEST_ASSERT(b.cpp_standard >= 23);   // the library requires C++23
        TEST_ASSERT(!b.compiler.empty());
#if defined(__clang__)
        TEST_ASSERT(b.compiler.starts_with("Clang "));
#elif defined(__GNUC__)
        TEST_ASSERT(b.compiler.starts_with("GCC "));
        TEST_ASSERT(b.compiler.find(std::to_string(__GNUC__) + ".") != std::string_view::npos);
#endif
        std::cout << "(" << b.compiler << ", C++" << b.cpp_standard << ", " << b.build_type
                  << ", OpenMP " << (b.with_openmp ? "yes" : "no") << ") ";
    });

    runner.run();
    return print_final_summary();
}
