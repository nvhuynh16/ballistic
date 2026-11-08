#pragma once

// =============================================================================
// DE440 Embedded Chebyshev Coefficients (1950-2100 Subset)
// =============================================================================
//
// This file contains embedded DE440 ephemeris coefficients for:
// - Sun (body 10) relative to Solar System Barycenter (body 0)
// - Earth-Moon Barycenter (body 3) relative to SSB (body 0)
// - Moon (body 301) relative to EMB (body 3)
// - Jupiter Barycenter (body 5) relative to SSB (body 0)
//
// Time range: 1950-01-01 to 2100-01-01 (150 years)
// Source: JPL DE440 ephemeris (Park et al. 2021)
//
// Generated: 2026-02-01 17:22:57  // guard-exempt(PUB-DATE): generation date of the data
// Generator: scripts/extract_de440_subset.py
//
// =============================================================================

#include <array>
#include <cstddef>

namespace ballistic {
namespace ephemeris {
namespace de440_embedded {

// DE440 constants
inline constexpr double EMRAT = 81.3005690699;  // Earth/Moon mass ratio
inline constexpr double START_ET = -1577880000.0;  // 1950-01-01 TDB seconds past J2000
inline constexpr double END_ET = 3155716800.0;    // 2100-01-01 TDB seconds past J2000

// Segment metadata
struct SegmentInfo {
    double init_et;        // Initial epoch (TDB seconds past J2000)
    double interval_sec;   // Time interval per record
    int n_coeffs;          // Chebyshev coefficients per component
    int n_records;         // Number of records
    int record_size;       // Size of each record (doubles)
};

// EMB_SSB: body 3 relative to body 0
// Records: 3426, Coeffs per component: 13, Total doubles: 140466
inline constexpr SegmentInfo EMB_SSB_INFO = {
    -1578052800.0,  // init_et
    1382400.0,        // interval_sec
    13,              // n_coeffs
    3426,            // n_records
    41              // record_size
};

extern const std::array<double, 140466> EMB_SSB_DATA;

// JUPITER_SSB: body 5 relative to body 0
// Records: 1714, Coeffs per component: 8, Total doubles: 44564
inline constexpr SegmentInfo JUPITER_SSB_INFO = {
    -1579435200.0,  // init_et
    2764800.0,        // interval_sec
    8,              // n_coeffs
    1714,            // n_records
    26              // record_size
};

extern const std::array<double, 44564> JUPITER_SSB_DATA;

// SUN_SSB: body 10 relative to body 0
// Records: 3426, Coeffs per component: 11, Total doubles: 119910
inline constexpr SegmentInfo SUN_SSB_INFO = {
    -1578052800.0,  // init_et
    1382400.0,        // interval_sec
    11,              // n_coeffs
    3426,            // n_records
    35              // record_size
};

extern const std::array<double, 119910> SUN_SSB_DATA;

// MOON_EMB: body 301 relative to body 3
// Records: 13699, Coeffs per component: 13, Total doubles: 561659
inline constexpr SegmentInfo MOON_EMB_INFO = {
    -1578052800.0,  // init_et
    345600.0,        // interval_sec
    13,              // n_coeffs
    13699,            // n_records
    41              // record_size
};

extern const std::array<double, 561659> MOON_EMB_DATA;

// Total embedded data: 866599 doubles = 6,932,792 bytes (6.61 MB)

} // namespace de440_embedded
} // namespace ephemeris
} // namespace ballistic
