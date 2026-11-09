// tests/portable_random.hpp must draw exactly what libstdc++'s distributions
// draw from the same std::mt19937, so test noise is the same on every
// platform.

#include "../test_utils.hpp"
#include "../portable_random.hpp"
#include <random>

using namespace ballistic::test;

int main() {
    TestRunner runner("Portable random distributions");

    runner.add_test("fixed_sequence_from_libstdcxx", []() {
        // Printed by libstdc++ (GCC 13): seed 12345, normal(0.5, 2) x4, then
        // uniform(-1, 3) x4 from the same engine.
        std::mt19937 g(12345);
        Normal n(0.5, 2.0);
        UniformReal u(-1.0, 3.0);
        const double normals[] = {1.6901823323131235, 0.5836680582945859, 2.1008934160521564,
                                  0.35615663323921309};
        const double uniforms[] = {0.49568470916608964, -0.38010657704102824, 2.5693749065955309,
                                   -0.89284108955291408};
        // std::log may differ in its last bit between C libraries.
        for (double v : normals) TEST_ASSERT_NEAR(n(g), v, 1e-15 * (1.0 + std::abs(v)));
        for (double v : uniforms) TEST_ASSERT(u(g) == v);
    });

#if defined(__GLIBCXX__)
    runner.add_test("bit_identical_to_libstdcxx", []() {
        for (unsigned seed : {1u, 42u, 2024u}) {
            std::mt19937 a(seed), b(seed);
            std::normal_distribution<double> sn(-3.0, 0.25);
            std::uniform_real_distribution<double> su(0.0, 1.0);
            Normal pn(-3.0, 0.25);
            UniformReal pu(0.0, 1.0);
            for (int i = 0; i < 20000; ++i) {
                TEST_ASSERT(sn(a) == pn(b));
                if (i % 7 == 0) TEST_ASSERT(su(a) == pu(b));
            }
        }
    });
#endif

    runner.run();
    return print_final_summary();
}
