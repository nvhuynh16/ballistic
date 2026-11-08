#pragma once

// =============================================================================
// EGM2008 Embedded Spherical Harmonic Coefficients
// =============================================================================
//
// This file contains embedded EGM2008 fully-normalized spherical harmonic
// coefficients up to degree 360.
//
// Source: NGA EGM2008_to2190_TideFree
// Reference: Pavlis, N.K., et al. (2012). "The development and evaluation of
//            the Earth Gravitational Model 2008 (EGM2008)."
//            Journal of Geophysical Research, 117, B04406.
//
// Total coefficients: 65338 (degrees 2-360)
// Generated automatically - do not edit manually.
//
// =============================================================================

#include <array>
#include <cstddef>

namespace ballistic {
namespace gravity {
namespace egm2008_data {

// Maximum degree of embedded coefficients
constexpr int MAX_EMBEDDED_DEGREE = 360;

// Number of coefficient pairs (n, m, Cnm, Snm)
constexpr std::size_t NUM_COEFFICIENTS = 65338;

// Coefficient structure: {degree, order, Cnm, Snm}
struct Coefficient {
    int n;
    int m;
    double C;
    double S;
};

// Embedded EGM2008 coefficients (fully normalized, tide-free)
// Format: {degree, order, Cnm, Snm}
extern const std::array<Coefficient, NUM_COEFFICIENTS> COEFFICIENTS;

} // namespace egm2008_data
} // namespace gravity
} // namespace ballistic
