#pragma once

#include <Eigen/Dense>
#include <array>
#include <optional>
#include <tuple>
#include <memory>

namespace ballistic {

// (Empty: reserves the forces namespace for headers that include only this one.)
namespace forces {
}

/// Versions ("realizations") of the ITRF, the International Terrestrial Reference Frame
/// (the Earth-fixed frame). Successive realizations differ by millimeters to centimeters;
/// pick the one a data product is expressed in. ESA moved Sentinel precise orbit
/// determination (POD) products from ITRF2014 to ITRF2020 on March 4, 2025.
namespace coords {
enum class ITRFRealization {
    ITRF2000,
    ITRF2005,
    ITRF2008,
    ITRF2014,   // Use for Sentinel data before March 4, 2025
    ITRF2020    // Default, use for Sentinel data after March 4, 2025
};
} // namespace coords

// Eigen (linear algebra library) vector and matrix types used throughout
using Vector3d = Eigen::Vector3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix3d = Eigen::Matrix3d;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Quaterniond = Eigen::Quaterniond;

/// The frame the equations of motion are integrated in. GCRS (the Geocentric Celestial
/// Reference System: Earth-centered, non-rotating) is the only choice; sub-meter accuracy
/// requires it. The enum has a single value so the configuration field keeps its type.
enum class PropagationFrame {
    GCRS        // Geocentric Celestial Reference System (CIO-based, IERS 2010)
};

/// Position [m] and velocity [m/s] in some frame; the frame is not stored, so the variable
/// name or the function returning it says which (e.g. state_ecef, state_gcrs).
struct StateVector {
    Vector3d position;  // meters
    Vector3d velocity;  // meters/second

    StateVector() : position(Vector3d::Zero()), velocity(Vector3d::Zero()) {}

    StateVector(const Vector3d& pos, const Vector3d& vel)
        : position(pos), velocity(vel) {}

    /// As a 6-element vector [x, y, z, vx, vy, vz].
    [[nodiscard]] Vector6d to_vector() const noexcept {
        Vector6d v;
        v << position, velocity;
        return v;
    }

    /// From a 6-element vector [x, y, z, vx, vy, vz].
    [[nodiscard]] static StateVector from_vector(const Vector6d& v) noexcept {
        return StateVector(v.head<3>(), v.tail<3>());
    }

    // Element-wise arithmetic, for the integrators
    StateVector operator+(const StateVector& other) const noexcept {
        return StateVector(position + other.position, velocity + other.velocity);
    }

    StateVector operator-(const StateVector& other) const noexcept {
        return StateVector(position - other.position, velocity - other.velocity);
    }

    StateVector operator*(double scalar) const noexcept {
        return StateVector(position * scalar, velocity * scalar);
    }

    StateVector& operator+=(const StateVector& other) noexcept {
        position += other.position;
        velocity += other.velocity;
        return *this;
    }

    StateVector& operator*=(double scalar) noexcept {
        position *= scalar;
        velocity *= scalar;
        return *this;
    }
};

inline StateVector operator*(double scalar, const StateVector& state) noexcept {
    return state * scalar;
}

/// The acceleration from each force model separately (all in m/s²), filled when
/// TrajectoryConfig::report_forces is enabled.
struct ForceBreakdown {
    Vector3d gravity;       // Earth gravity (EGM2008/zonal)
    Vector3d third_body;    // Sun + Moon + Jupiter
    Vector3d drag;          // Atmospheric drag
    Vector3d srp;           // Solar radiation pressure
    Vector3d albedo;        // Earth albedo radiation
    Vector3d ir;            // Earth infrared radiation
    Vector3d relativistic;  // Schwarzschild + Lense-Thirring
    Vector3d solid_tides;   // Solid Earth tide corrections
    Vector3d ocean_tides;   // Ocean tide corrections (when coefficients are loaded)
    Vector3d empirical;     // Empirical acceleration corrections
    Vector3d total;         // Sum of all enabled components

    ForceBreakdown()
        : gravity(Vector3d::Zero())
        , third_body(Vector3d::Zero())
        , drag(Vector3d::Zero())
        , srp(Vector3d::Zero())
        , albedo(Vector3d::Zero())
        , ir(Vector3d::Zero())
        , relativistic(Vector3d::Zero())
        , solid_tides(Vector3d::Zero())
        , ocean_tides(Vector3d::Zero())
        , empirical(Vector3d::Zero())
        , total(Vector3d::Zero()) {}

    /// Sum of the radiation components: solar radiation pressure (SRP) + sunlight reflected
    /// by the Earth (albedo) + the Earth's infrared emission (IR).
    [[nodiscard]] Vector3d radiation() const noexcept {
        return srp + albedo + ir;
    }
};

/// Position, velocity and acceleration at a requested time, in the Earth-fixed frame (ECEF,
/// Earth-centered Earth-fixed, i.e. the ITRF).
struct PropagationResult {
    Vector3d position;      // ECEF, meters
    Vector3d velocity;      // ECEF, meters/second
    Vector3d acceleration;  // ECEF, meters/second^2

    // Optional 6x6 position/velocity covariance; not filled by the propagators at present
    std::optional<Matrix6d> covariance;

    // Optional force breakdown (populated when TrajectoryConfig::report_forces is true)
    std::optional<ForceBreakdown> forces;

    PropagationResult()
        : position(Vector3d::Zero())
        , velocity(Vector3d::Zero())
        , acceleration(Vector3d::Zero())
        , covariance(std::nullopt)
        , forces(std::nullopt) {}

    PropagationResult(const Vector3d& pos, const Vector3d& vel, const Vector3d& acc)
        : position(pos), velocity(vel), acceleration(acc), covariance(std::nullopt), forces(std::nullopt) {}
};

/// Drag described by the ballistic coefficient beta = m/(Cd*A) [kg/m^2] (mass m, drag
/// coefficient Cd, area A). Trajectory data constrain only this ratio, not m, Cd and A
/// separately. A Trajectory takes drag from its DragEnvironment, not from this struct.
struct BallisticModel {
    double beta = 45.45;            // kg/m^2, ballistic coefficient (primary drag param)
                                    // Default: 1000/(2.2*10) = 45.45

    // Optional metadata: mass for SRP, drag_area for diagnostic display
    double mass = 0.0;              // kg (also used by SRP via satellite_mass)
    double drag_area = 0.0;         // m^2

    BallisticModel()
        : beta(45.45)
        , mass(0.0)
        , drag_area(0.0) {}

    explicit BallisticModel(double beta_val)
        : beta(beta_val)
        , mass(0.0)
        , drag_area(0.0) {}

    BallisticModel(double beta_val, double mass_val, double area_val)
        : beta(beta_val)
        , mass(mass_val)
        , drag_area(area_val) {}

    /// Ballistic coefficient beta [kg/m^2].
    [[nodiscard]] double ballistic_coefficient() const noexcept {
        return beta;
    }

    /// Inverse ballistic coefficient 1/beta [m^2/kg].
    [[nodiscard]] double inverse_ballistic_coefficient() const noexcept {
        return 1.0 / beta;
    }

    /// Inverse ballistic coefficient [m^2/kg]; the altitude argument is ignored (always the
    /// constant beta). For a beta that varies with altitude use a BetaProfile1D inside a
    /// DragEnvironment.
    [[nodiscard]] double inverse_ballistic_coefficient_at_altitude(double /*altitude_km*/) const noexcept {
        return 1.0 / beta;
    }

};

/// Space weather inputs of the atmosphere model (they drive upper-atmosphere density).
/// F10.7 is the Sun's radio flux at 10.7 cm wavelength, a proxy for its ultraviolet output;
/// Ap is the daily index of geomagnetic activity.
struct WeatherConditions {
    double f107_daily;       // Daily F10.7 solar flux (SFU, 10^-22 W/m^2/Hz)
    double f107_average;     // 81-day centered average F10.7
    double ap_daily;         // Daily Ap geomagnetic index
    // Ap history in the NRLMSIS order: [0] daily Ap, [1] current 3-hour Ap, [2..4] the
    // 3-hour Ap 3, 6 and 9 h before, [5] mean of eight 3-hour values 12-33 h before,
    // [6] mean of eight 3-hour values 36-57 h before.
    std::array<double, 7> ap_history;  // 3-hour Ap values for magnetic activity history

    WeatherConditions()
        : f107_daily(150.0)
        , f107_average(150.0)
        , ap_daily(4.0)
        , ap_history{4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0} {}

    /// Quiet conditions (solar minimum): F10.7 70, Ap 4. The default constructor gives
    /// F10.7 150, Ap 4.
    [[nodiscard]] static WeatherConditions quiet() noexcept {
        WeatherConditions w;
        w.f107_daily = 70.0;
        w.f107_average = 70.0;
        w.ap_daily = 4.0;
        w.ap_history = {4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0};
        return w;
    }

    /// Active conditions (solar maximum): F10.7 200, Ap 15.
    [[nodiscard]] static WeatherConditions active() noexcept {
        WeatherConditions w;
        w.f107_daily = 200.0;
        w.f107_average = 200.0;
        w.ap_daily = 15.0;
        w.ap_history = {15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0};
        return w;
    }

    /// Geomagnetic storm: F10.7 250 (average 200), Ap 100 with a decaying history.
    [[nodiscard]] static WeatherConditions storm() noexcept {
        WeatherConditions w;
        w.f107_daily = 250.0;
        w.f107_average = 200.0;
        w.ap_daily = 100.0;
        w.ap_history = {100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 15.0};
        return w;
    }
};

/// Satellite surface properties for radiation pressure (Montenbruck & Gill (2000),
/// "Satellite Orbits", Section 3.4). The radiation pressure coefficient used is
/// Cr = 1 + reflectivity (1 = fully absorbing, 2 = mirror).
struct SatelliteProperties {
    double srp_area = 0.0;           // Cross-sectional area for SRP [m²]
    double srp_reflectivity = 0.3;   // Optical reflectivity coefficient [0-1]
    double albedo_reflectivity = 0.3; // Albedo reflectivity coefficient [0-1]
    double ir_emissivity = 0.8;      // IR absorptivity/emissivity [0-1]

    SatelliteProperties() = default;

    SatelliteProperties(double area, double refl = 0.3, double ir_emis = 0.8)
        : srp_area(area)
        , srp_reflectivity(refl)
        , albedo_reflectivity(refl)
        , ir_emissivity(ir_emis)
    {}
};

/// Geodetic coordinates on the WGS84 ellipsoid (the standard Earth shape): latitude,
/// longitude and height above the ellipsoid.
struct GeodeticCoord {
    double latitude;    // radians, positive north
    double longitude;   // radians, positive east
    double altitude;    // meters above WGS84 ellipsoid

    GeodeticCoord() : latitude(0.0), longitude(0.0), altitude(0.0) {}

    GeodeticCoord(double lat, double lon, double alt)
        : latitude(lat), longitude(lon), altitude(alt) {}

    /// From latitude and longitude in degrees and altitude in meters.
    [[nodiscard]] static GeodeticCoord from_degrees(double lat_deg, double lon_deg, double alt_m);
};

} // namespace ballistic
