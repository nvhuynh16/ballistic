// Parity check (dev tool, not a tutorial listing): calls the REAL
// tests/test_helpers.hpp glide::make_glide() and prints its IC + eval_end at
// full precision so we can diff against gen_glide_fixture's output. eval_end
// is a deterministic function of the recoil-direction derivation and the
// truth propagation, so equality here implies the generator reproduces the
// canonical bed exactly (including the recoil direction Setup does not expose).

#include "test_helpers.hpp"

#include <cstdio>

int main() {
    // Same data as gen_glide_fixture (the data directory's EOP); without it
    // the frame transforms raise since 2026-11 (they used to assume zeros).
    if (!ballistic::initialize({}).success) return 1;
    auto s = ballistic::test::helpers::glide::make_glide();
    std::printf("PARITY pos_ecef = [%.17g, %.17g, %.17g]\n",
                s.pos.x(), s.pos.y(), s.pos.z());
    std::printf("PARITY vel_ecef = [%.17g, %.17g, %.17g]\n",
                s.vel.x(), s.vel.y(), s.vel.z());
    std::printf("PARITY eval_end = %.17g\n", s.eval_end);
    return 0;
}
