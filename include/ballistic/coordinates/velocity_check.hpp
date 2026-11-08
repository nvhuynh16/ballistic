#pragma once

// =============================================================================
// Velocity-record check: are an orbit product's ECEF velocities the time
// derivative of its own positions?
// =============================================================================
//
// The library needs the TRUE time derivative of the ECEF (Earth-fixed)
// position as a start velocity, because its ECEF -> GCRS (inertial) transform
// includes the CIP drift: the slow motion of the Earth's spin axis in space
// (the Celestial Intermediate Pole, CIP; see iau_transforms.hpp), worth
// ~2e-5 m/s at low Earth orbit (LEO). Orbit products do not all write the true
// derivative: some leave the CIP drift out, some differ in other ways. An
// along-track velocity error of 4.6e-5 m/s grows to ~0.5 m at LEO in one
// hour, and nothing downstream can tell it from a force-model error:
// "correcting" records that are already true derivatives can hide a drag
// shortfall.
//
// check_velocity_records() differentiates the positions (9-point Lagrange
// polynomial, interior epochs only — one-sided windows at the ends amplify
// the millimeter rounding of the positions) and compares the result with the records,
// in the radial / along-track / cross-track frame. It also reports the mean
// CIP drift so a "records omit the drift" product is recognizable.
// =============================================================================

#include "../core/types.hpp"
#include "../core/errors.hpp"
#include "../time/epoch.hpp"
#include "iau_transforms.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace ballistic {
namespace coords {

/// What check_velocity_records() concluded about a product's velocity records.
enum class VelocityRecordVerdict {
    TrueDerivative,   // records match the derivative of the positions: use as written
    OmitsCipDrift,    // records = derivative + CIP drift: subtract cip_drift_velocity()
    Inconsistent,     // neither: start from velocities derived from the positions
};

/// Result of check_velocity_records().
struct VelocityRecordCheck {
    int n_compared = 0;                     // epochs compared
    // median (derived - record), radial / along-track / cross-track [m/s];
    // medians, because arc joins inside a product give a few large outliers
    Vector3d median_difference_rtn = Vector3d::Zero();
    double rms_difference = 0.0;            // per-epoch |derived - record| RMS [m/s]
    double mean_cip_drift_along = 0.0;      // along-track CIP drift, mean [m/s]
    double standard_error_along = 0.0;      // robust (MAD) error of the along-track median [m/s]
    VelocityRecordVerdict verdict = VelocityRecordVerdict::Inconsistent;
    std::string summary;                    // one-line verdict with the numbers, for printing
};

namespace detail {

// d/dt at t[k] of the Lagrange polynomial through the n points
// (t[i], x[i]), i = lo .. lo+n-1.
inline Vector3d lagrange_derivative(const std::vector<double>& t, const std::vector<Vector3d>& x,
                                    std::size_t lo, std::size_t n, std::size_t k) {
    Vector3d d = Vector3d::Zero();
    for (std::size_t i = lo; i < lo + n; ++i) {
        double coeff = 0.0;
        for (std::size_t j = lo; j < lo + n; ++j) {
            if (j == i) continue;
            double term = 1.0 / (t[i] - t[j]);
            for (std::size_t m = lo; m < lo + n; ++m) {
                if (m == i || m == j) continue;
                term *= (t[k] - t[m]) / (t[i] - t[m]);
            }
            coeff += term;
        }
        d += coeff * x[i];
    }
    return d;
}

}  // namespace detail

/// Checks whether one satellite's velocity records are the time derivative of
/// its positions (see the file comment), and says what to do with them.
///
/// - epoch0 + offsets_s[i] is the time of sample i; offsets (s) strictly
///   increasing, best taken from the product's own time tags. Sample timing
///   errors of tens of microseconds already swamp the test at LEO speeds.
/// - positions, velocities: ECEF (Earth-fixed), m and m/s, one per offset;
///   ideally evenly spaced at <= 60 s for LEO (<= 300 s above medium Earth
///   orbit, MEO).
/// - itrf: the ITRF realization of the positions (used for the CIP drift).
///
/// Compares every interior epoch when there are fewer than 400, otherwise 200
/// to 400 of them spread over the span. Throws InputPlausibilityError for
/// mismatched lengths, fewer than 11 epochs or non-increasing offsets, and
/// DataCoverageError from the Earth orientation (EOP) lookup behind the CIP
/// drift (no EOP table loaded under the default policy, or epochs more than
/// 30 days beyond it).
[[nodiscard]] inline VelocityRecordCheck check_velocity_records(
    const Epoch& epoch0,
    const std::vector<double>& offsets_s,
    const std::vector<Vector3d>& positions,
    const std::vector<Vector3d>& velocities,
    ITRFRealization itrf = ITRFRealization::ITRF2020)
{
    constexpr std::size_t kPoints = 9;
    constexpr std::size_t kHalf = kPoints / 2;
    const std::size_t n = offsets_s.size();
    if (positions.size() != n || velocities.size() != n)
        throw InputPlausibilityError("check_velocity_records: " + std::to_string(n) + " offsets, " +
            std::to_string(positions.size()) + " positions and " + std::to_string(velocities.size()) +
            " velocities: pass one position and one velocity per offset");
    if (n < kPoints + 2)
        throw InputPlausibilityError("check_velocity_records: " + std::to_string(n) + " epochs given; "
            "the 9-point derivative needs at least 11 (pass a longer stretch of the file)");

    const std::vector<double>& t = offsets_s;
    for (std::size_t i = 1; i < n; ++i)
        if (!(t[i] > t[i - 1]))
            throw InputPlausibilityError("check_velocity_records: offset " + std::to_string(i) + " (" +
                std::to_string(t[i]) + " s) is not after the previous one (" + std::to_string(t[i - 1]) +
                " s); sort the records and drop duplicate epochs");

    // Every interior epoch below 400, else 200-400 spread over the span. Along-track (t_hat)
    // and cross-track (n_hat) directions come from the recorded velocity.
    const std::size_t interior = n - 2 * kHalf;
    const std::size_t stride = std::max<std::size_t>(1, interior / 200);

    VelocityRecordCheck out;
    std::vector<double> radial, along, cross;
    double sum_sq = 0.0, drift_sum = 0.0;
    for (std::size_t k = kHalf; k + kHalf < n; k += stride) {
        const Vector3d derived = detail::lagrange_derivative(t, positions, k - kHalf, kPoints, k);
        const Vector3d diff = derived - velocities[k];
        const Vector3d r_hat = positions[k].normalized();
        const Vector3d n_hat = positions[k].cross(velocities[k]).normalized();
        const Vector3d t_hat = n_hat.cross(r_hat);
        radial.push_back(diff.dot(r_hat));
        along.push_back(diff.dot(t_hat));
        cross.push_back(diff.dot(n_hat));
        sum_sq += diff.squaredNorm();
        drift_sum += cip_drift_velocity(positions[k], epoch0 + t[k], itrf).dot(t_hat);
        ++out.n_compared;
    }
    const double m = static_cast<double>(out.n_compared);
    auto median = [](std::vector<double> v) {
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    };
    out.median_difference_rtn = Vector3d(median(radial), median(along), median(cross));
    out.rms_difference = std::sqrt(sum_sq / m);
    out.mean_cip_drift_along = drift_sum / m;
    std::vector<double> dev(along.size());
    for (std::size_t i = 0; i < along.size(); ++i) dev[i] = std::abs(along[i] - out.median_difference_rtn.y());
    // 1.4826 MAD = sigma for Gaussian data; the median's error is ~1.25 sigma/sqrt(m)
    out.standard_error_along = 1.25 * 1.4826 * median(dev) / std::sqrt(m);

    // Tolerance: 3 standard errors, but never tighter than 5e-6 m/s
    // (0.05 m along-track after one hour at LEO).
    const double tol = std::max(5e-6, 3.0 * out.standard_error_along);
    const double d_along = out.median_difference_rtn.y();
    const double drift = out.mean_cip_drift_along;
    char buf[320];
    if (std::abs(d_along) <= tol) {
        out.verdict = VelocityRecordVerdict::TrueDerivative;
        std::snprintf(buf, sizeof buf,
                      "records are the time derivative of the positions (median along-track difference "
                      "%.1e m/s, tolerance %.1e): use them as written", d_along, tol);
    } else if (std::abs(drift) > tol && std::abs(d_along + drift) <= tol) {
        out.verdict = VelocityRecordVerdict::OmitsCipDrift;
        std::snprintf(buf, sizeof buf,
                      "records omit the CIP drift (difference %.1e m/s vs drift %.1e): "
                      "subtract cip_drift_velocity()", d_along, drift);
    } else {
        out.verdict = VelocityRecordVerdict::Inconsistent;
        std::snprintf(buf, sizeof buf,
                      "records differ from the derivative of the positions by %.1e m/s along-track "
                      "(CIP drift %.1e, tolerance %.1e): start from velocities derived from the positions",
                      d_along, drift, tol);
    }
    out.summary = buf;
    return out;
}

}  // namespace coords
}  // namespace ballistic
