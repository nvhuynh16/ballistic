#pragma once

// =============================================================================
// JPL Development Ephemeris (DE440/DE441) Reader
// =============================================================================
//
// LEGACY reader for JPL Development Ephemeris (DE) files in JPL's own binary
// format (not the SPK format). Prefer SPKEphemeris (spk_ephemeris.hpp): this
// reader assumes the DE440 record layout instead of reading it from the file
// (see read_header below).
//
// This file also defines get_default_ephemeris_provider(), the function the
// force model uses to pick an ephemeris.
//
// References:
// -----------
// [1] Folkner, W.M., et al. (2014). "The Planetary and Lunar Ephemerides
//     DE430 and DE431". IPN Progress Report 42-196.
//
// [2] Park, R.S., et al. (2021). "The JPL Planetary and Lunar Ephemerides
//     DE440 and DE441". The Astronomical Journal, 161:105.
//     DOI: 10.3847/1538-3881/abd414
//
// [3] Press, W.H., et al. "Numerical Recipes", 3rd Edition.
//     Section 5.8: Chebyshev Approximation - Clenshaw's recurrence
//
// Chebyshev Polynomial Evaluation:
// --------------------------------
// Each coordinate over a time interval is f(x) = sum_i c_i * T_i(x), with T_i
// the Chebyshev polynomials and x the time scaled to [-1, 1]. It is evaluated
// with Clenshaw's algorithm for numerical stability:
//   recurrence: b_k = c_k + 2*x*b_{k+1} - b_{k+2}
//   result:     f(x) = c_0 + x*b_1 - b_2
//
// This avoids computing individual Chebyshev polynomials which can
// overflow/underflow for high degrees.
//
// File Format (JPL DE Binary):
// ----------------------------
// - Record 1: Header with title, start/end times, AU, EMRAT, coefficients info
// - Record 2: Constant names and values
// - Records 3+: Chebyshev coefficients organized by body and time interval
//
// Body indices used here (0-based; JPLBody and the hardcoded table in
// read_header):
//   0: Mercury       5: Saturn        10: Sun
//   1: Venus         6: Uranus        11: Nutations (dPsi, dEps)
//   2: Earth-Moon    7: Neptune       12: Librations
//      barycenter    8: Pluto         13, 14: lunar mantle, TT-TDB (not available)
//   3: Mars          9: Moon (geocentric)
//   4: Jupiter
// In the DE files the planets, the Earth-Moon barycenter and the Sun are
// relative to the solar-system barycenter (SSB); the Moon is geocentric.
//
// =============================================================================

#include "../core/errors.hpp"
#include "../core/types.hpp"
#include "../core/constants.hpp"
#include "../time/epoch.hpp"
#include "../math/chebyshev.hpp"
#include "ephemeris_provider.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <array>
#include <string>
#include <cstring>
#include <cmath>
#include <memory>
#include <filesystem>
#include <atomic>
#include <mutex>

namespace ballistic {
namespace ephemeris {

/// Body indices of a DE file (0-based).
enum class JPLBody : int {
    MERCURY = 0,
    VENUS = 1,
    EARTH_MOON_BARYCENTER = 2,
    MARS = 3,
    JUPITER = 4,
    SATURN = 5,
    URANUS = 6,
    NEPTUNE = 7,
    PLUTO = 8,
    MOON_GEOCENTRIC = 9,
    SUN = 10,
    NUTATIONS = 11,
    LIBRATIONS = 12,
    // Note: indices can differ between DE versions
};

/// Header of a DE binary file (as far as this reader fills it in).
struct JPLHeader {
    std::array<char, 84 * 3> title;  ///< 3 lines of 84 characters
    double start_jd;                  ///< first Julian date covered (TDB)
    double end_jd;                    ///< last Julian date covered (TDB)
    double interval_days;             ///< days per data record
    int n_constants;                  ///< number of constants (not read)
    double au_km;                     ///< astronomical unit [km]
    double emrat;                     ///< Earth/Moon mass ratio

    /// Coefficient layout per body: [offset (1-based), coefficients per
    /// coordinate, sub-intervals per record]. 15 slots; the 13 used in DE440
    /// are Mercury through Pluto, Moon, Sun, nutations, librations.
    std::array<std::array<int, 3>, 15> body_info;

    int de_number;                    ///< DE version (e.g. 440)
    int record_size;                  ///< data record length in doubles
};

/// Legacy reader for JPL binary DE files (prefer SPKEphemeris).
///
/// The record layout is assumed to be DE440's (see read_header); other DE
/// versions give wrong positions without an error.
class JPLEphemeris : public EphemerisProvider {
public:
    JPLEphemeris() = default;

    /// Read a DE binary file into memory.
    /// @return false if the file cannot be opened or holds no data record.
    bool load(const std::filesystem::path& de_file) {
        std::ifstream file(de_file, std::ios::binary);
        if (!file.is_open()) {
            loaded_ = false;
            return false;
        }

        // Header (the first two records)
        if (!read_header(file)) {
            loaded_ = false;
            return false;
        }

        // Read all data records
        if (!read_data(file)) {
            loaded_ = false;
            return false;
        }

        loaded_ = true;
        file_path_ = de_file;
        return true;
    }

    /// True once a file has loaded.
    [[nodiscard]] bool is_loaded() const noexcept { return loaded_; }

    /// First and last Julian dates (TDB) covered.
    [[nodiscard]] std::pair<double, double> time_range() const noexcept {
        return {header_.start_jd, header_.end_jd};
    }

    /// DE version number (always 440 with this reader).
    [[nodiscard]] int de_version() const noexcept { return header_.de_number; }

    /// Sun position [m] from the Earth's center, on ICRF (GCRS) axes.
    ///
    /// The file holds the Sun and the Earth-Moon barycenter (EMB) relative to
    /// the solar-system barycenter and the Moon relative to the Earth, so the
    /// Earth is EMB - Moon / (1 + EMRAT) and the geocentric Sun is
    /// Sun - Earth (EMRAT = Earth/Moon mass ratio).
    /// @throws DataCoverageError if no file is loaded or the epoch is outside it.
    [[nodiscard]] Vector3d sun_position(const Epoch& epoch) const override {
        const double jd = checked_tdb_jd(epoch, "sun_position");
        const auto [sun_ssb, sun_v] = get_body_state(jd, 10);
        const auto [emb_ssb, emb_v] = get_body_state(jd, 2);
        const auto [moon_geo, moon_v] = get_body_state(jd, 9);
        const Vector3d earth_ssb = emb_ssb - moon_geo / (1.0 + header_.emrat);
        return (sun_ssb - earth_ssb) * 1000.0;   // km -> m
    }

    /// Moon position [m] from the Earth's center, on GCRS axes.
    /// @throws DataCoverageError if no file is loaded or the epoch is outside it.
    [[nodiscard]] Vector3d moon_position(const Epoch& epoch) const override {
        const double jd = checked_tdb_jd(epoch, "moon_position");

        // Moon, geocentric (index 9)
        auto [pos, vel] = get_body_state(jd, 9);

        // Convert from km to meters
        return pos * 1000.0;
    }

    /// Position [m] of the Jupiter system barycenter relative to the Earth-Moon
    /// barycenter (EMB; ~4671 km from the Earth's center, negligible at Jupiter's
    /// distance), on GCRS axes.
    /// @throws DataCoverageError if no file is loaded or the epoch is outside it.
    [[nodiscard]] Vector3d jupiter_position(const Epoch& epoch) const override {
        const double jd = checked_tdb_jd(epoch, "jupiter_position");

        // Jupiter (index 4) and the Earth-Moon barycenter (index 2), both
        // relative to the solar-system barycenter
        auto [jup_pos, jup_vel] = get_body_state(jd, 4);
        auto [emb_pos, emb_vel] = get_body_state(jd, 2);

        // Jupiter relative to the Earth-Moon barycenter
        Vector3d pos = jup_pos - emb_pos;

        // Convert from km to meters
        return pos * 1000.0;
    }

    /// Position [km] and velocity [km/day] of body `body_index` (0..12, see
    /// JPLBody) at Julian date `jd_tdb` (TDB), relative to the origin the file
    /// uses for that body.
    /// @throws ballistic::Error if no file is loaded, the body is not in the
    ///         file, or the record is shorter than the layout says.
    /// @throws ConfigError if body_index is not 0..12.
    /// @throws DataCoverageError if the date is outside the loaded records.
    [[nodiscard]] std::pair<Vector3d, Vector3d> get_body_state(
        double jd_tdb, int body_index) const
    {
        // Raise rather than return zero vectors (or evaluate the first/last
        // record far past its interval): a zero Sun or Moon is a force-model
        // error with no symptom.
        if (!loaded_) throw Error("JPLEphemeris: no ephemeris file loaded");
        if (body_index < 0 || body_index >= 13)
            throw ConfigError("JPLEphemeris: body index " +
                              std::to_string(body_index) + " is not 0..12 (see JPLBody)");

        // Data record containing this time
        int record_idx = static_cast<int>(std::floor((jd_tdb - header_.start_jd) / header_.interval_days));
        if (record_idx == static_cast<int>(data_records_.size()) &&
            jd_tdb <= header_.start_jd + data_records_.size() * header_.interval_days) {
            --record_idx;   // exactly the end of the last record
        }
        if (record_idx < 0 || record_idx >= static_cast<int>(data_records_.size())) {
            throw DataCoverageError("JPLEphemeris: JD(TDB) " + std::to_string(jd_tdb) +
                " is outside the loaded ephemeris (JD " + std::to_string(header_.start_jd) +
                " to " + std::to_string(header_.start_jd +
                    data_records_.size() * header_.interval_days) +
                "). Load a file that covers the date (DE441 spans -13200 to 17191).");
        }

        const auto& record = data_records_[record_idx];
        double record_start = header_.start_jd + record_idx * header_.interval_days;

        // This body's layout within a record
        int offset = header_.body_info[body_index][0] - 1;  // Convert to 0-based
        int n_coeffs = header_.body_info[body_index][1];
        int n_subint = header_.body_info[body_index][2];

        if (offset < 0 || n_coeffs <= 0 || n_subint <= 0) {
            throw Error("JPLEphemeris: body " + std::to_string(body_index) +
                        " is not in the loaded file");
        }

        // Sub-interval of the record containing this time
        double subint_days = header_.interval_days / n_subint;
        double time_in_record = jd_tdb - record_start;
        int subint_idx = static_cast<int>(time_in_record / subint_days);
        if (subint_idx >= n_subint) subint_idx = n_subint - 1;
        if (subint_idx < 0) subint_idx = 0;

        // Time scaled to [-1, 1] across the sub-interval (the Chebyshev domain)
        double subint_start = subint_idx * subint_days;
        double subint_mid = subint_start + subint_days / 2.0;
        double x = (time_in_record - subint_mid) / (subint_days / 2.0);

        // Clamp round-off at the sub-interval edges
        x = std::max(-1.0, std::min(1.0, x));

        // Coefficients of this sub-interval (x, y, z blocks of n_coeffs each)
        int coeff_offset = offset + subint_idx * n_coeffs * 3;
        if (coeff_offset + n_coeffs * 3 > static_cast<int>(record.size())) {
            throw Error("JPLEphemeris: record " + std::to_string(record_idx) +
                        " is shorter than its header says; the file is damaged");
        }

        const double* coeffs = record.data() + coeff_offset;

        // Evaluate (Clenshaw); the half-interval in days turns the derivative
        // into km/day
        return math::chebyshev_eval_3d(coeffs, n_coeffs, x, subint_days / 2.0);
    }

private:
    // TDB Julian date of `epoch`, after checking that a file is loaded and covers it.
    [[nodiscard]] double checked_tdb_jd(const Epoch& epoch, const char* what) const {
        if (!loaded_) {
            throw DataCoverageError(std::string("JPLEphemeris::") + what +
                ": no DE file loaded. Call ballistic::initialize() with "
                "InitConfig::jpl_ephemeris_path set, or use the embedded DE440 (SPKEphemeris).");
        }
        const double jd = epoch.tdb_jd();
        if (jd < header_.start_jd || jd > header_.end_jd) {
            throw DataCoverageError(std::string("JPLEphemeris::") + what + ": JD(TDB) " +
                std::to_string(jd) + " is outside the loaded DE file (JD " +
                std::to_string(header_.start_jd) + " to " + std::to_string(header_.end_jd) +
                "). Load a file that covers the date (DE441 spans -13200 to 17191).");
        }
        return jd;
    }

    bool loaded_ = false;
    std::filesystem::path file_path_;
    JPLHeader header_;
    std::vector<std::vector<double>> data_records_;

    bool read_header(std::ifstream& file) {
        // LIMITATION: this does NOT really parse the DE binary header. It tries
        // to read the start/end times from the file and falls back to hardcoded
        // DE440 values if they look wrong. The body_info table (coefficient
        // offsets, counts, sub-intervals) and the record size are ALWAYS the
        // DE440 values, never read from the file, so a file of another DE
        // version gives wrong positions without any error.
        //
        // For reliable reading use SPKEphemeris (spk_ephemeris.hpp), which
        // parses the DAF/SPK format properly.

        // First record (title and metadata)
        std::vector<char> buffer(1024 * 8);  // 8 KB covers the header
        file.read(buffer.data(), buffer.size());

        if (file.gcount() < 256) {
            return false;
        }

        // Parse title (first 252 bytes = 3 lines of 84 chars)
        std::memcpy(header_.title.data(), buffer.data(), 252);

        // Numbers after the title (approximate: the layout varies by DE version)
        const double* dptr = reinterpret_cast<const double*>(buffer.data() + 252);

        header_.start_jd = dptr[0];
        header_.end_jd = dptr[1];
        header_.interval_days = dptr[2];

        // Implausible values: fall back to DE440's span
        if (header_.start_jd < 1000000 || header_.end_jd < header_.start_jd) {
            header_.start_jd = 2287184.5;  // DE440 default start
            header_.end_jd = 2688976.5;    // DE440 default end
            header_.interval_days = 32.0;   // Typical interval
        }

        // DE440 layout, assumed for every file
        // Format: [offset, n_coeffs, n_subintervals]
        header_.body_info = {{
            {{3, 14, 4}},     // Mercury
            {{171, 10, 2}},   // Venus
            {{231, 13, 2}},   // Earth-Moon barycenter
            {{309, 11, 1}},   // Mars
            {{342, 8, 1}},    // Jupiter
            {{366, 7, 1}},    // Saturn
            {{387, 6, 1}},    // Uranus
            {{405, 6, 1}},    // Neptune
            {{423, 6, 1}},    // Pluto
            {{441, 13, 8}},   // Moon (geocentric)
            {{753, 11, 2}},   // Sun
            {{819, 10, 4}},   // Nutations
            {{899, 10, 4}},   // Librations
            {{0, 0, 0}},      // Lunar mantle
            {{0, 0, 0}}       // TT-TDB
        }};

        header_.de_number = 440;
        header_.record_size = 1018;  // Typical for DE440
        header_.au_km = 149597870.7;
        header_.emrat = 81.3005690699;

        return true;
    }

    bool read_data(std::ifstream& file) {
        // Data records start after the two header records
        file.seekg(2 * header_.record_size * 8, std::ios::beg);

        // Read data records
        size_t record_bytes = header_.record_size * sizeof(double);
        std::vector<double> record(header_.record_size);

        while (file.read(reinterpret_cast<char*>(record.data()), record_bytes)) {
            data_records_.push_back(record);
        }

        return !data_records_.empty();
    }
};

/// Process-wide JPLEphemeris (used by get_default_ephemeris_provider() only
/// if no SPK ephemeris is loaded).
///
/// Thread safety: load() is serialized by a lock and is_loaded() is a lock-free
/// read. Loading while another thread evaluates the ephemeris is a data race:
/// load before propagating from several threads.
class JPLEphemerisManager {
public:
    /// The single shared instance.
    static JPLEphemerisManager& instance() {
        static JPLEphemerisManager mgr;
        return mgr;
    }

    /// Load a DE binary file into the shared instance; false on failure.
    bool load(const std::filesystem::path& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        loaded_.store(false, std::memory_order_release);
        const bool ok = ephemeris_.load(path);
        loaded_.store(ephemeris_.is_loaded(), std::memory_order_release);
        return ok;
    }

    /// True once a file has loaded.
    [[nodiscard]] bool is_loaded() const noexcept {
        return loaded_.load(std::memory_order_acquire);
    }

    /// The shared ephemeris object.
    [[nodiscard]] const JPLEphemeris& ephemeris() const noexcept {
        return ephemeris_;
    }

    [[nodiscard]] JPLEphemeris& ephemeris() noexcept {
        return ephemeris_;
    }

private:
    JPLEphemerisManager() = default;
    JPLEphemeris ephemeris_;
    std::atomic<bool> loaded_{false};   // ephemeris_ holds data (lock-free read)
    std::mutex mutex_;                  // serializes load()
};

/// Load a DE binary file into the shared instance; false on failure.
inline bool load_jpl_ephemeris(const std::filesystem::path& path) {
    return JPLEphemerisManager::instance().load(path);
}

/// The shared legacy DE ephemeris.
inline const JPLEphemeris& jpl_ephemeris() {
    return JPLEphemerisManager::instance().ephemeris();
}

// Access to the shared SPK ephemeris, defined in spk_ephemeris.hpp (declared
// here so this header does not have to include it)
namespace detail {
    bool is_spk_loaded();
    bool load_embedded_spk();
    bool ensure_embedded_spk_loaded();
    std::shared_ptr<EphemerisProvider> get_spk_provider();
}

/// The ephemeris the force model should use.
///
/// Order: the shared SPK ephemeris if loaded (embedded DE440, or a DE421/DE441
/// .bsp file), else the legacy binary DE reader if loaded. If neither is
/// loaded, it loads the embedded DE440 first, so force models work even
/// without a ballistic::initialize({}) call. The returned pointer does not own
/// the object (the shared instance lives for the whole program).
/// @throws ballistic::Error if the embedded data fails to load (a build problem).
[[nodiscard]] inline std::shared_ptr<EphemerisProvider> get_default_ephemeris_provider() {
    if (detail::is_spk_loaded()) {
        return detail::get_spk_provider();
    }

    if (JPLEphemerisManager::instance().is_loaded()) {
        return std::shared_ptr<EphemerisProvider>(
            std::shared_ptr<void>{}, &JPLEphemerisManager::instance().ephemeris());
    }

    // Nothing loaded yet: load the embedded DE440 (code that builds a force
    // model before calling ballistic::initialize() lands here). The check and
    // the load are one locked step, so concurrent first callers load it once.
    if (detail::ensure_embedded_spk_loaded()) {
        return detail::get_spk_provider();
    }

    throw Error(
        "get_default_ephemeris_provider: embedded SPK failed to load. "
        "This indicates a library build issue.");
}

} // namespace ephemeris
} // namespace ballistic
