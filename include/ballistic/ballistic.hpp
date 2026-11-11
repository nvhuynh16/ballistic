#pragma once

// Ballistic Trajectory Prediction Library: the umbrella header.
// Predicts where a satellite (or any object in Earth orbit or on a reentry
// path) will be, from a known position and velocity, with high-fidelity
// force models.
//
// Features:
// - State vector propagation: position and velocity in, position and
//   velocity out, in ECEF (Earth-centered, Earth-fixed) coordinates
// - Earth gravity with full spherical harmonics (the EGM2008 model)
// - Optional Sun, Moon, Jupiter gravitational perturbations
// - Atmospheric drag via TrajectoryConfig.drag_env (a DragEnvironment, usually
//   from atmosphere::make_drag_environment; tabulated variants from
//   atmosphere::from_models_snapshot / _along / _4d / _accelerometer_density)
// - Runge-Kutta 4(5) Dormand-Prince adaptive integrator
// - Symplectic (Yoshida) integrators, which keep the orbital energy from
//   drifting, with the "slimplectic" extension for drag
// - Full IAU 2006/2000A transformations between the Earth-fixed and the
//   inertial (non-rotating) frame
// - SP3 precise-orbit file reading for validation (validation/sp3.hpp, not
//   included here)
//
// Example usage:
//
//   #include <ballistic/ballistic.hpp>
//
//   using namespace ballistic;
//
//   // Load embedded data and, if present, Earth orientation data (init.hpp)
//   ballistic::initialize({});
//
//   // Create epoch for initial state
//   Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
//
//   // Initial state (ECEF)
//   Vector3d pos_ecef(6778137.0, 0.0, 0.0);  // meters
//   Vector3d vel_ecef(0.0, 7175.0, 0.0);  // Earth-fixed: circular 7669 m/s minus Earth rotation 494 m/s
//
//   // Create trajectory (default settings: gravity degree 70, inertial
//   // integration frame, Sun/Moon/Jupiter); the state is checked first
//   Trajectory traj(InitialState::from_itrf(t0, pos_ecef, vel_ecef));
//
//   // Propagate 1 hour forward
//   Epoch t1 = t0 + 3600.0;
//   PropagationResult result = traj.propagate(t1);
//
//   // Result contains position, velocity, acceleration in ECEF
//   std::cout << "Position: " << result.position.transpose() << std::endl;
//
// Advanced usage with custom configuration (optional forces switch on by
// presence — assign an SrpConfig / DragEnvironment to enable them):
//
//   TrajectoryConfig config;                       // measured-best defaults
//   config.srp = SrpConfig{.mass_kg = 2200.0, .area_m2 = 38.4, .cr = 1.0};  // S1A
//   // drag: beta = m/(Cd*A) in kg/m^2 (Sentinel-1, from satellites.hpp)
//   atmosphere::SpaceWeatherData::instance().load_celestrak_csv("SW-All.csv");
//   config.drag_env = atmosphere::make_drag_environment(   // measured weather
//       t0, pos_ecef, vel_ecef, 3600.0, config,
//       {.beta = atmosphere::BetaProfile1D::constant(157.5)});
//
//   Trajectory traj(InitialState::from_itrf(t0, pos_ecef, vel_ecef), config);
//

// Core types and constants
#include "core/build_info.hpp"
#include "core/constants.hpp"
#include "core/types.hpp"

// Time handling
#include "time/julian_date.hpp"
#include "time/epoch.hpp"
#include "time/eop_data.hpp"

// Coordinate transformations
#include "coordinates/geodetic.hpp"
#include "coordinates/earth_rotation.hpp"
#include "coordinates/iau_transforms.hpp"
#include "coordinates/velocity_check.hpp"
#include "satellites.hpp"
#include "gnss_blocks.hpp"           // GNSS block radiation areas (IGS block names)

// Gravity models
#include "gravity/point_mass.hpp"
#include "gravity/egm_gravity.hpp"
#include "gravity/third_body.hpp"

// Drag (env-driven via DragEnvironment — see atmosphere/drag_environment.hpp).
// The NRLMSIS21 / HWM14 headers (atmosphere/internal/) are not
// part of the public umbrella; include them explicitly if you need them.
#include "atmosphere/drag_environment.hpp"
#include "atmosphere/drag_environment_generators.hpp"
#include "atmosphere/space_weather_history.hpp"
#include "atmosphere/space_weather.hpp"          // SpaceWeatherData: the measured default drag weather
#include "atmosphere/beta_profile.hpp"
#include "forces/drag.hpp"

// Ephemeris
#include "ephemeris/ephemeris_provider.hpp"
#include "ephemeris/jpl_ephemeris.hpp"
#include "ephemeris/spk_ephemeris.hpp"  // SPK (DE421/DE440/DE441) support

// Force model
#include "forces/empirical_acceleration.hpp"
#include "forces/force_model.hpp"

// Propagators
#include "propagators/rk45.hpp"
#include "propagators/symplectic.hpp"
#include "propagators/propagator.hpp"

// Trajectory API (clean wrapper)
#include "trajectory.hpp"
#include "trajectory_interpolator.hpp"
#include "state_transition.hpp"   // STM and covariance along a propagated Trajectory

// Math utilities
#include "math/quintic_hermite.hpp"
#include "math/cubic_spline.hpp"
#include "math/hermite_table.hpp"

// Calibration
#include "calibration/ea_decomposition.hpp"

// Initialization
#include "init.hpp"

namespace ballistic {

// Library version.
constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 1;
constexpr int VERSION_PATCH = 3;

constexpr const char* VERSION_STRING = "0.1.3";

} // namespace ballistic
