#pragma once

// =============================================================================
// Storing air density compactly as log(ρ) in 16-bit floats
// =============================================================================
//
// Air density between the ground and 1000 km spans ~15 orders of magnitude
// (1.2 kg/m³ at the surface to ~1e-15 kg/m³ at 1000 km). A 16-bit float
// (range ~6e-5 to ~6e4) cannot hold the small values, so we store log(ρ)
// instead: log(ρ) lies in about [-35, +1], well inside the range. The
// rounding error in ρ after decoding is set by the 16-bit step at that
// magnitude: about 0.4% for |log ρ| in [8, 16), 0.8% in [16, 32) (most of
// LEO) and 1.6% above 32 — within the atmosphere model's own uncertainty.
//
// Used by the 4D DensityField4D table only; the 1D/2D tables store double
// precision, since they are small.
//
// Anything that differentiates the density (e.g. a finite-difference
// Jacobian) must work on the decoded double values of a smooth interpolant,
// never difference raw stored values: the 0.4-1.6% rounding is too coarse
// for derivatives.
//
// The 16-bit float is IEEE 754 binary16 held as raw bits (`half`), with the
// conversions written out in integer arithmetic: storage only, no 16-bit
// arithmetic, and the same bits on every compiler (C++23 std::float16_t is
// not available everywhere). The conversions round to nearest, ties to even,
// keep subnormals, send overflow to ±infinity and keep NaN a NaN (quieted, as
// x86 and ARM hardware conversions do) — bit for bit what a static_cast
// to/from std::float16_t gives where that type exists (gated by
// test_log_density_storage). This header needs no Eigen.
//
// =============================================================================

#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ballistic {
namespace atmosphere {

/// IEEE 754 binary16 value stored as its raw bit pattern.
struct half {
    std::uint16_t bits = 0;
};

/// Convert a float to binary16: round to nearest, ties to even; values
/// beyond the largest half (65504) round to ±infinity; NaN stays NaN with
/// the quiet bit set and the top payload bits kept.
[[nodiscard]] constexpr half float_to_half(float f) noexcept {
    const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
    const auto sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
    const std::uint32_t ax = x & 0x7FFFFFFFu;

    if (ax >= 0x7F800000u) {                        // infinity or NaN
        if (ax == 0x7F800000u) return half{static_cast<std::uint16_t>(sign | 0x7C00u)};
        return half{static_cast<std::uint16_t>(
            sign | 0x7E00u | ((ax >> 13) & 0x03FFu))};
    }
    if (ax >= 0x477FF000u) {                        // >= 65520: rounds to infinity
        return half{static_cast<std::uint16_t>(sign | 0x7C00u)};
    }
    const std::uint32_t exp_f = ax >> 23;
    if (exp_f >= 113u) {                            // normal half (>= 2^-14)
        std::uint32_t q = ((exp_f - 112u) << 10) | ((ax & 0x7FFFFFu) >> 13);
        const std::uint32_t r = ax & 0x1FFFu;
        if (r > 0x1000u || (r == 0x1000u && (q & 1u))) ++q;
        return half{static_cast<std::uint16_t>(sign | q)};
    }
    // Subnormal half (or zero): value = q * 2^-24.
    const std::uint32_t shift = 126u - exp_f;       // >= 14
    if (exp_f == 0u || shift > 24u) return half{sign};  // below half of 2^-24
    const std::uint32_t mant = (ax & 0x7FFFFFu) | 0x800000u;
    std::uint32_t q = mant >> shift;
    const std::uint32_t r = mant & ((1u << shift) - 1u);
    const std::uint32_t mid = 1u << (shift - 1u);
    if (r > mid || (r == mid && (q & 1u))) ++q;     // may carry to 2^-14: still correct
    return half{static_cast<std::uint16_t>(sign | q)};
}

/// Convert binary16 to float (exact; a signaling NaN comes back quiet).
[[nodiscard]] constexpr float half_to_float(half h) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(h.bits & 0x8000u) << 16;
    const std::uint32_t e = (h.bits >> 10) & 0x1Fu;
    std::uint32_t m = h.bits & 0x03FFu;
    std::uint32_t out = 0;
    if (e == 0x1Fu) {
        out = sign | 0x7F800000u | (m << 13);
        if (m != 0u) out |= 0x00400000u;            // quiet NaN
    } else if (e == 0u) {
        if (m == 0u) {
            out = sign;
        } else {                                    // subnormal: normalize
            std::uint32_t e_f = 113u;
            while ((m & 0x0400u) == 0u) {
                m <<= 1;
                --e_f;
            }
            out = sign | (e_f << 23) | ((m & 0x03FFu) << 13);
        }
    } else {
        out = sign | ((e + 112u) << 23) | (m << 13);
    }
    return std::bit_cast<float>(out);
}

/// Encode a density [kg/m³] as log(ρ) in a 16-bit float. Throws
/// std::invalid_argument unless rho > 0.
[[nodiscard]] inline half encode_log_rho(double rho) {
    if (!(rho > 0.0)) {
        throw std::invalid_argument(
            "encode_log_rho: density must be strictly positive");
    }
    return float_to_half(static_cast<float>(std::log(rho)));
}

/// Decode a stored 16-bit log(ρ) back to a density [kg/m³].
[[nodiscard]] inline double decode_log_rho(half stored_log_rho) noexcept {
    return std::exp(static_cast<double>(half_to_float(stored_log_rho)));
}

}  // namespace atmosphere
}  // namespace ballistic
