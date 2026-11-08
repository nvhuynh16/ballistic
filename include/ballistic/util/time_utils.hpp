#pragma once

// =============================================================================
// Time Utility Functions
// =============================================================================
//
// Common time-related utility functions used across the codebase.
//
// =============================================================================

#include "../time/epoch.hpp"

namespace ballistic {
namespace util {

/// The UTC date as an approximate decimal year (months as 1/12 year, days as 1/365.25 year),
/// for the time-dependent Helmert transformations (7-parameter shift/rotation/scale, plus
/// their rates) between ITRF realizations, where this precision is ample. Throws
/// DataCoverageError for dates before 1961 (no UTC).
// Example: 2024-06-15 12:00:00 → ~2024.456  // guard-exempt(PUB-DATE): data
[[nodiscard]] inline double epoch_to_decimal_year(const time::Epoch& epoch) {
    auto [year, month, day, hour, minute, second] = epoch.to_calendar(time::TimeScale::UTC);
    // Approximate decimal year (sufficient for Helmert transformations)
    return year + (month - 1) / 12.0 + (day - 1) / 365.25 +
           (hour + minute / 60.0 + second / 3600.0) / (24.0 * 365.25);
}

} // namespace util
} // namespace ballistic
