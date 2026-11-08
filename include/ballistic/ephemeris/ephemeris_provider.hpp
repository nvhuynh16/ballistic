#pragma once

#include "../core/types.hpp"
#include "../time/epoch.hpp"

namespace ballistic {
namespace ephemeris {

/// Abstract source of Sun, Moon and Jupiter positions (an "ephemeris" is a
/// table of where solar-system bodies are over time).
///
/// Implementations: SPKEphemeris (the embedded JPL DE440 ephemeris or an
/// external .bsp file; recommended) and JPLEphemeris (legacy binary DE
/// format). Both throw DataCoverageError for an epoch outside
/// the loaded data's time range, or before anything has been loaded — call
/// ballistic::initialize({}) first.
class EphemerisProvider {
public:
    virtual ~EphemerisProvider() = default;

    /// Body position [m] at `epoch`, from (approximately) the Earth's center,
    /// on GCRS axes (Geocentric Celestial Reference System: non-rotating axes
    /// aligned with the International Celestial Reference Frame). See each
    /// implementation for exactly which origin it uses.
    [[nodiscard]] virtual Vector3d sun_position(const Epoch& epoch) const = 0;
    [[nodiscard]] virtual Vector3d moon_position(const Epoch& epoch) const = 0;
    [[nodiscard]] virtual Vector3d jupiter_position(const Epoch& epoch) const = 0;
};

} // namespace ephemeris
} // namespace ballistic
