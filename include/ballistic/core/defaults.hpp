#pragma once

#include <cstddef>

// =============================================================================
// Default Constants and Tolerances
// =============================================================================
//
// Named tolerances and default values used throughout the library. Use these
// instead of inline literals. Units are in each comment (seconds, meters,
// m/s). RK45 is the adaptive Runge-Kutta 4(5) (Dormand-Prince) integrator;
// Yoshida 6/8 are the fixed-step symplectic integrators.
//
// =============================================================================

namespace ballistic {
namespace defaults {

// =============================================================================
// Epoch Comparison Tolerances
// =============================================================================

// Tolerance for epoch equality comparison (seconds): two epochs closer than
// this are the same instant (exact Epoch arithmetic agrees to ~1e-11 s)
constexpr double EPOCH_EQUALITY_SECONDS = 1e-9;

// Tolerance for integration completion (seconds)
// Used in integrator loops to detect when target time is reached
constexpr double INTEGRATION_EPSILON_SECONDS = 1e-12;

// Tolerance for finding nearest epoch in validation data (seconds)
// Records within this threshold are considered "near enough" to match
constexpr double EPOCH_PROXIMITY_SECONDS = 60.0;

// Tolerance for matching velocity (V) and position (P) records in an SP3
// (standard precise-orbit text format) file (seconds). Not used by the
// library: the SP3 reader pairs each V line with the P line of its epoch.
constexpr double SP3_VELOCITY_MATCH_SECONDS = 0.001;

// Maximum time difference for RMSE (root-mean-square error) computation
// (seconds): predicted/reference pairs further apart are excluded
constexpr double RMSE_MAX_TIME_DIFF_SECONDS = 1.0;

// =============================================================================
// Integrator Constants
// =============================================================================

// RK45: when a step's normalized error estimate is below this, the next step
// grows by the maximum factor RK45_MAX_SCALE
constexpr double RK45_ERROR_FLOOR = 1e-10;

// Exponent of the step-size controller (new step = old step *
// (1/error)^exponent, an integral or "I" controller).
// Hairer, Norsett & Wanner (1993), Eq. II.4.13: 1/(p+1) = 1/5 for p=4.
constexpr double RK45_STEP_CONTROLLER_EXPONENT = 0.2;

// RK45 safety limit: integration fails if force evaluations exceed
// max_iterations * this factor
constexpr int RK45_EVAL_MULTIPLIER = 7;

// =============================================================================
// Physical Model Defaults
// =============================================================================

// Default degree of the EGM2008 spherical-harmonic gravity field for low Earth
// orbit (LEO). Degrees up to 360 are embedded, but 70 is enough for LEO
// (degrees 100-160 were no better on Sentinel validation data, and slower;
// docs/configuration.md). The order always equals
// the degree in this library.
constexpr int DEFAULT_GRAVITY_DEGREE = 70;

// Inert BallisticModel starting values for Trajectory construction. Drag reads
// the ballistic coefficient beta = m/(Cd A) from the DragEnvironment and the
// solar radiation pressure (SRP) mass comes through set_satellite_properties(),
// so these never influence a propagation.
constexpr double PLACEHOLDER_BETA = 45.45;   // kg/m² (legacy seed, unused)
constexpr double PLACEHOLDER_MASS = 1000.0;  // kg (legacy seed, unused)

// =============================================================================
// RK45 Step Size Defaults
// =============================================================================

// Default initial step size (seconds)
constexpr double RK45_INITIAL_STEP_SIZE = 60.0;

// Minimum allowed step size (seconds)
constexpr double RK45_MIN_STEP_SIZE = 0.001;

// Maximum allowed step size (seconds) - 1 hour
constexpr double RK45_MAX_STEP_SIZE = 3600.0;

// Default relative tolerance
constexpr double RK45_RELATIVE_TOLERANCE = 1e-10;

// Default absolute tolerance
constexpr double RK45_ABSOLUTE_TOLERANCE = 1e-12;

// Safety factor for step size adjustment
constexpr double RK45_SAFETY_FACTOR = 0.9;

// Minimum step size reduction factor
constexpr double RK45_MIN_SCALE = 0.2;

// Maximum step size increase factor
constexpr double RK45_MAX_SCALE = 10.0;

// RK45 step budget: integration fails when force evaluations exceed this
// times RK45_EVAL_MULTIPLIER (7 evaluations per Dormand-Prince step)
constexpr size_t RK45_MAX_ITERATIONS = 1000000;

// =============================================================================
// Symplectic Integrator Defaults
// =============================================================================

// Default fixed step size for the Yoshida symplectic integrators (seconds)
constexpr double SYMPLECTIC_DEFAULT_STEP_SIZE = 30.0;

// =============================================================================
// Validation Defaults
// =============================================================================

// Default Lagrange polynomial interpolation of tabulated orbits: number of
// points (9 points = 8th-degree polynomial)
constexpr int LAGRANGE_DEFAULT_ORDER = 9;

// Maximum time difference when matching a predicted state to a reference
// orbit record in validation (seconds)
constexpr double VALIDATION_MAX_TIME_DIFF = 5.0;

// Interpolation points for the CHAMP satellite's accelerometer data (5 points
// = 4th-degree polynomial). Fewer than LAGRANGE_DEFAULT_ORDER because the
// 10-second CHAMP cadence is coarser; a 9-point window would span 80 seconds.
// Not used by the library at present.
constexpr int CHAMP_INTERPOLATION_ORDER = 5;

// Maximum time difference for CHAMP accelerometer comparison (seconds)
constexpr double CHAMP_MAX_TIME_DIFF = 5.0;

// =============================================================================
// Force Model Numerical Thresholds
// =============================================================================

// Threshold below which velocity magnitude is considered zero (m/s)
// Used in drag calculations to avoid division by zero
constexpr double VELOCITY_EPSILON = 1e-10;

// Threshold below which an acceleration magnitude is treated as zero
// Used in symplectic integrator inner-step guards
constexpr double ACCELERATION_EPSILON = 1e-15;

// Threshold below which |cos(latitude)| is treated as a pole singularity
// Meant for Earth-fixed (ECEF) <-> geodetic conversions (0/0 in x/cos(lat), y/cos(lat));
// not used by the library at present (the geodetic conversion is closed-form)
constexpr double COS_LATITUDE_EPSILON = 1e-10;

// Reference radius used by the symplectic integrator for relative-distance
// scaling around the Earth (meters; ~mean Earth + 600 km LEO altitude)
constexpr double EARTH_REFERENCE_RADIUS_M = 7.0e6;

// Threshold below which shadow/visibility factors are considered zero
// Used in radiation pressure calculations to determine complete shadow
constexpr double VISIBILITY_EPSILON = 1e-10;

// Altitude above which atmosphere effects are negligible (meters, 1000 km)
// Used to skip atmosphere calculations for high-altitude orbits (NRLMSIS 2.1 validity ceiling)
constexpr double ATMOSPHERE_ALTITUDE_THRESHOLD_M = 1000000.0;

// =============================================================================
// Gravity Model Numerical Thresholds
// =============================================================================

// Threshold for detecting pole singularity in longitude calculation
// When p/r < this value, we're effectively at a pole and cos(lon)=1, sin(lon)=0
// Used in the gravity field evaluation (Clenshaw summation of the spherical
// harmonics) to avoid 0/0 in x/p, y/p (p = distance from the rotation axis)
constexpr double GRAVITY_POLE_RATIO_THRESHOLD = 1e-10;

// Threshold for u (cos of geocentric latitude) near zero at poles
// When |u| < this value, use special pole handling in Clenshaw outer sum
// Used to protect t/u calculation from pole singularity
constexpr double GRAVITY_POLE_U_THRESHOLD = 1e-15;

} // namespace defaults
} // namespace ballistic
