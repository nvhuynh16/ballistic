// Unit tests for the compact density storage used by the 4D density table:
// density ρ is stored as log(ρ) in a 16-bit float (IEEE binary16, "half"),
// over 16 orders of magnitude (sea level 1.225 kg/m³ to ~1e-15 kg/m³ at
// 1000 km). Zero and negative densities have no logarithm and must be
// rejected.
//
// Round-trip bound: half a binary16 step of log(ρ), i.e. a relative error in
// ρ of expm1(2^(floor(log2|log ρ|) - 11)) — about 0.4% / 0.8% / 1.6% for
// |log ρ| in [8,16) / [16,32) / [32,64), as atmosphere/log_density_storage.hpp
// states.
//
// Where the compiler provides C++23 std::float16_t, the hand-written
// conversions are checked bit for bit against it: every one of the 65,536
// half patterns decodes identically, and a dense float sweep (every rounding
// boundary, subnormals, overflow, NaN payloads, ±0) encodes identically.

#include "../test_utils.hpp"
#include <ballistic/atmosphere/log_density_storage.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#if defined(__STDCPP_FLOAT16_T__)
#include <stdfloat>
#endif

using namespace ballistic;
using namespace ballistic::test;
using ballistic::atmosphere::decode_log_rho;
using ballistic::atmosphere::encode_log_rho;
using ballistic::atmosphere::float_to_half;
using ballistic::atmosphere::half;
using ballistic::atmosphere::half_to_float;

namespace {

// Documented round-trip bound for a density: half a binary16 step of log(ρ),
// with 1% slack for the float rounding of log(ρ) itself.
double round_trip_bound(double rho) {
    const double lr = std::abs(std::log(rho));
    const double half_step = std::ldexp(1.0, static_cast<int>(std::floor(std::log2(lr))) - 11);
    return std::expm1(half_step) * 1.01;
}

void check_round_trip(double rho) {
    const double decoded = decode_log_rho(encode_log_rho(rho));
    const double rel_err = std::abs(decoded - rho) / rho;
    const double bound = round_trip_bound(rho);
    if (!(rel_err <= bound)) {
        std::cerr << "rho=" << rho << " decoded=" << decoded << " rel_err=" << rel_err
                  << " bound=" << bound << "\n";
    }
    TEST_ASSERT(rel_err <= bound);
}

std::uint32_t bits_of(float f) { return std::bit_cast<std::uint32_t>(f); }

}  // namespace

void test_round_trip_precision() {
    TestRunner runner("RoundTripPrecision");

    runner.add_test("ground_level_density", []() {
        check_round_trip(1.225);  // kg/m³ at sea level, |log ρ| = 0.2: ~6e-5
    });

    runner.add_test("leo_400km_density", []() {
        check_round_trip(4e-12);  // |log ρ| = 26.2: within 0.8%
    });

    runner.add_test("very_high_altitude_density", []() {
        check_round_trip(1e-15);  // ~1000 km, |log ρ| = 34.5: within 1.6%
    });

    runner.add_test("full_dynamic_range_within_documented_precision", []() {
        for (int e = -15; e <= 0; ++e) {
            for (double mantissa : {1.0, 1.5, 2.0, 3.0, 5.0, 7.5}) {
                check_round_trip(mantissa * std::pow(10.0, e));
            }
        }
    });

    runner.run();
}

void test_edge_cases() {
    TestRunner runner("EdgeCases");

    runner.add_test("zero_density_rejected", []() {
        bool threw = false;
        try {
            (void)encode_log_rho(0.0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.add_test("negative_density_rejected", []() {
        bool threw = false;
        try {
            (void)encode_log_rho(-1e-12);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.add_test("size_of_half_is_2_bytes", []() {
        // The storage saving the 4D table relies on.
        TEST_ASSERT(sizeof(half) == 2);
    });

    runner.run();
}

void test_half_conversions() {
    TestRunner runner("HalfConversions");

    runner.add_test("known_values", []() {
        // IEEE 754 binary16 reference patterns.
        TEST_ASSERT(float_to_half(0.0f).bits == 0x0000);
        TEST_ASSERT(float_to_half(-0.0f).bits == 0x8000);
        TEST_ASSERT(float_to_half(1.0f).bits == 0x3C00);
        TEST_ASSERT(float_to_half(-2.0f).bits == 0xC000);
        TEST_ASSERT(float_to_half(65504.0f).bits == 0x7BFF);        // largest finite
        TEST_ASSERT(float_to_half(65519.996f).bits == 0x7BFF);      // just below the tie
        TEST_ASSERT(float_to_half(65520.0f).bits == 0x7C00);        // tie to even = inf
        TEST_ASSERT(float_to_half(1e10f).bits == 0x7C00);
        TEST_ASSERT(float_to_half(-std::numeric_limits<float>::infinity()).bits == 0xFC00);
        TEST_ASSERT(float_to_half(std::ldexp(1.0f, -14)).bits == 0x0400);  // smallest normal
        TEST_ASSERT(float_to_half(std::ldexp(1.0f, -24)).bits == 0x0001);  // smallest subnormal
        TEST_ASSERT(float_to_half(std::ldexp(1.0f, -25)).bits == 0x0000);  // tie to even = 0
        TEST_ASSERT(float_to_half(std::ldexp(1.5f, -25)).bits == 0x0001);
        TEST_ASSERT(float_to_half(1.0f + std::ldexp(1.0f, -11)).bits == 0x3C00);  // tie, even
        TEST_ASSERT(float_to_half(1.0f + 3 * std::ldexp(1.0f, -11)).bits == 0x3C02);
        const half nan = float_to_half(std::numeric_limits<float>::quiet_NaN());
        TEST_ASSERT((nan.bits & 0x7C00) == 0x7C00 && (nan.bits & 0x03FF) != 0);
        TEST_ASSERT(std::isnan(half_to_float(nan)));
        TEST_ASSERT(half_to_float(half{0x3555}) == 0.333251953125f);
        TEST_ASSERT(half_to_float(half{0x0001}) == std::ldexp(1.0f, -24));
        TEST_ASSERT(std::isinf(half_to_float(half{0xFC00})) && half_to_float(half{0xFC00}) < 0);
    });

    runner.add_test("every_finite_half_round_trips", []() {
        // decode then encode is the identity on every non-NaN pattern.
        int bad = 0;
        for (std::uint32_t b = 0; b < 65536u; ++b) {
            const half h{static_cast<std::uint16_t>(b)};
            const bool is_nan = (b & 0x7C00u) == 0x7C00u && (b & 0x03FFu) != 0u;
            if (is_nan) continue;
            if (float_to_half(half_to_float(h)).bits != h.bits) ++bad;
        }
        TEST_ASSERT_EQ(bad, 0);
    });

    runner.add_test("constexpr_usable", []() {
        static_assert(float_to_half(1.0f).bits == 0x3C00);
        static_assert(half_to_float(half{0xC000}) == -2.0f);
        TEST_ASSERT(true);
    });

#if defined(__STDCPP_FLOAT16_T__)
    runner.add_test("half_bit_exact_vs_std_float16", []() {
        // Decode: all 65,536 patterns.
        int bad_decode = 0;
        for (std::uint32_t b = 0; b < 65536u; ++b) {
            const auto bits = static_cast<std::uint16_t>(b);
            const float ours = half_to_float(half{bits});
            const float ref = static_cast<float>(std::bit_cast<std::float16_t>(bits));
            if (bits_of(ours) != bits_of(ref)) {
                if (bad_decode < 5) std::cerr << "decode mismatch at 0x" << std::hex << b << std::dec << "\n";
                ++bad_decode;
            }
        }
        TEST_ASSERT_EQ(bad_decode, 0);

        // Encode: a float sweep with 2^20 strided bit patterns (every float
        // class, both signs, NaN payloads, float subnormals, overflow) plus,
        // around every half value, the value itself and the rounding
        // boundaries above it (midpoint and its float neighbors).
        long long bad_encode = 0;
        long long checked = 0;
        auto check = [&](float f) {
            ++checked;
            const std::uint16_t ours = float_to_half(f).bits;
            const auto ref = std::bit_cast<std::uint16_t>(static_cast<std::float16_t>(f));
            if (ours != ref) {
                if (bad_encode < 5) {
                    std::cerr << "encode mismatch at float bits 0x" << std::hex << bits_of(f)
                              << ": ours 0x" << ours << " std 0x" << ref << std::dec << "\n";
                }
                ++bad_encode;
            }
        };
        constexpr std::uint32_t kStride = 4096u;  // 2^32 / 4096 = 2^20 patterns
        for (std::uint32_t i = 0; i < (1u << 20); ++i) {
            check(std::bit_cast<float>(i * kStride + (i % kStride)));
        }
        for (std::uint32_t b = 0; b < 0x7C00u; ++b) {
            for (std::uint32_t s : {0u, 0x8000u}) {
                const float lo = half_to_float(half{static_cast<std::uint16_t>(b | s)});
                const float hi = half_to_float(half{static_cast<std::uint16_t>((b + 1) | s)});
                // Midpoint: 12 significant bits, exact in float. Above the
                // largest finite half the boundary is 65520.
                const float mid = std::isinf(hi)
                    ? std::copysign(65520.0f, lo)
                    : static_cast<float>((static_cast<double>(lo) + static_cast<double>(hi)) / 2.0);
                check(lo);
                check(mid);
                check(std::nextafter(mid, 0.0f));
                check(std::nextafter(mid, std::copysign(std::numeric_limits<float>::infinity(), mid)));
            }
        }
        for (float f : {0.0f, -0.0f, std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN(),
                        -std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::signaling_NaN(),
                        std::numeric_limits<float>::denorm_min(),
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::lowest()}) {
            check(f);
        }
        std::cout << "    checked " << checked << " floats against std::float16_t\n";
        TEST_ASSERT_EQ(bad_encode, 0LL);
    });
#else
    std::cout << "  (std::float16_t not provided by this compiler: the bit-exact "
                 "comparison does not apply)\n";
#endif

    runner.run();
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Log-Density Float16 Storage Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_round_trip_precision();
    test_edge_cases();
    test_half_conversions();

    return print_final_summary();
}
