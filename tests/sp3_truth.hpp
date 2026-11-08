#pragma once

// Reference ("truth") orbits for the C++ validation tests, read from SP3 files.
// SP3 is the standard text format for precise satellite orbits (positions,
// optionally velocities, at fixed intervals, per satellite).
//
// The files are read with the library's SP3 reader (validation/sp3.hpp), which
// honors the file's time system (GPS time, UTC, ...), keeps exact time
// offsets, and interpolates only inside the data and never across a gap. This
// adapter adds a flat per-record list (TruthRecord) on top of it, which is the
// shape the validation tests use.

#include <ballistic/core/errors.hpp>
#include <ballistic/validation/sp3.hpp>

#include <cmath>
#include <cstdio>
#include <memory>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifdef BALLISTIC_HAVE_ZLIB
#include <zlib.h>
#endif

namespace ballistic::test {

// One SP3 epoch for one satellite (Earth-fixed or inertial, as the file says).
struct TruthRecord {
    Epoch epoch;
    Vector3d position = Vector3d::Zero();   // m
    Vector3d velocity = Vector3d::Zero();   // m/s (zero when has_velocity is false)
    bool has_velocity = false;
    std::string satellite_id;
};

struct TruthFile {
    validation::sp3::File file;
    std::vector<TruthRecord> records;       // every satellite, file order per arc

    [[nodiscard]] const validation::sp3::Header& header() const { return file.header; }

    [[nodiscard]] std::vector<TruthRecord> get_satellite_records(const std::string& sat) const {
        std::vector<TruthRecord> out;
        for (const auto& r : records)
            if (r.satellite_id == sat) out.push_back(r);
        return out;
    }

    [[nodiscard]] std::optional<TruthRecord> find_nearest(const std::string& sat, const Epoch& t) const {
        std::optional<TruthRecord> best;
        double best_dt = std::numeric_limits<double>::max();
        for (const auto& r : records) {
            if (r.satellite_id != sat) continue;
            const double d = std::abs(r.epoch - t);
            if (d < best_dt) { best_dt = d; best = r; }
        }
        return best;
    }

    // Lagrange polynomial interpolation through `order` points inside one
    // gap-free segment of the satellite's arc; nullopt for an unknown
    // satellite, a time outside the data or a segment shorter than `order`
    // (never extrapolates). The velocity is the derivative of the
    // interpolated positions, not the file's velocity records.
    [[nodiscard]] std::optional<StateVector> interpolate_state(const std::string& sat, const Epoch& t,
                                                               int order = 9) const {
        for (const auto& a : file.arcs) {
            if (a.satellite != sat) continue;
            try {
                const auto s = a.interpolate(t - a.epoch0, order);
                return StateVector(s.position_m, s.velocity_m_s);
            } catch (const DataCoverageError&) {
                return std::nullopt;
            }
        }
        return std::nullopt;
    }
    [[nodiscard]] std::optional<Vector3d> interpolate_position(const std::string& sat, const Epoch& t,
                                                               int order = 9) const {
        auto s = interpolate_state(sat, t, order);
        if (!s) return std::nullopt;
        return s->position;
    }
    [[nodiscard]] std::optional<Vector3d> interpolate_velocity(const std::string& sat, const Epoch& t,
                                                               int order = 9) const {
        auto s = interpolate_state(sat, t, order);
        if (!s) return std::nullopt;
        return s->velocity;
    }
};

// Flattens a parsed SP3 file into TruthRecords; velocities that the file
// leaves blank (NaN after reading) are marked has_velocity = false.
[[nodiscard]] inline TruthFile to_truth(validation::sp3::File f) {
    TruthFile t;
    for (const auto& a : f.arcs) {
        for (std::size_t i = 0; i < a.size(); ++i) {
            TruthRecord r;
            r.epoch = a.epoch(i);
            r.position = a.position_m[i];
            r.satellite_id = a.satellite;
            if (!a.velocity_m_s.empty() && a.velocity_m_s[i].allFinite()) {
                r.velocity = a.velocity_m_s[i];
                r.has_velocity = true;
            }
            t.records.push_back(std::move(r));
        }
    }
    t.file = std::move(f);
    return t;
}

[[nodiscard]] inline TruthFile parse_sp3_truth(const std::string& text) {
    return to_truth(validation::sp3::read_string(text));
}

// The whole decompressed text of a .gz file. With zlib (tests/CMakeLists.txt
// links it when available) this is done in process; otherwise through a
// `gzip -dc` pipe, which needs gzip on the PATH.
[[nodiscard]] inline std::string read_gzip_text(const std::filesystem::path& path) {
    std::string text;
#ifdef BALLISTIC_HAVE_ZLIB
    std::unique_ptr<gzFile_s, int (*)(gzFile)> gz(gzopen(path.string().c_str(), "rb"), gzclose);
    if (!gz) throw Error("cannot open " + path.string() + " for gzip decompression");
    char buf[65536];
    int n = 0;
    while ((n = gzread(gz.get(), buf, static_cast<unsigned>(sizeof buf))) > 0) {
        text.append(buf, static_cast<std::size_t>(n));
    }
    if (n < 0) {
        int errnum = 0;
        const char* msg = gzerror(gz.get(), &errnum);
        throw Error("gzip decompression of " + path.string() + " failed: " +
                    (msg ? msg : "unknown error"));
    }
    if (gzdirect(gz.get()) == 1) {
        throw Error(path.string() + " is not in gzip format");  // as `gzip -dc` refuses it
    }
#else
#  ifdef _WIN32
    const std::string cmd = "gzip -dc \"" + path.string() + "\"";
    std::unique_ptr<FILE, int (*)(FILE*)> pipe(_popen(cmd.c_str(), "rb"), _pclose);
#  else
    const std::string cmd = "gzip -dc '" + path.string() + "'";
    std::unique_ptr<FILE, int (*)(FILE*)> pipe(popen(cmd.c_str(), "r"), pclose);
#  endif
    if (!pipe) throw Error("cannot run gzip for " + path.string());
    char buf[65536];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, pipe.get())) > 0) text.append(buf, n);
#endif
    if (text.empty()) throw Error("gzip produced nothing for " + path.string());
    return text;
}

// nullopt when the file does not exist (tests skip); a malformed file throws.
// A .gz file is decompressed by read_gzip_text (tests only; the library reader
// takes plain text and Python's ballistic.sp3.read handles gzip itself).
[[nodiscard]] inline std::optional<TruthFile> load_sp3_truth(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) return std::nullopt;
    if (path.extension() != ".gz") return to_truth(validation::sp3::read(path));
    auto t = parse_sp3_truth(read_gzip_text(path));
    t.file.source = path.string();
    return t;
}

// Position RMSE between predictions and the truth nearest in time (within
// max_time_diff seconds); NaN when nothing matches.
[[nodiscard]] inline double compute_position_rmse(
    const std::vector<std::pair<Epoch, Vector3d>>& predicted,
    const std::vector<std::pair<Epoch, Vector3d>>& truth,
    double max_time_diff = 1.0)
{
    double sum_sq = 0.0;
    int count = 0;
    for (const auto& [pe, pp] : predicted) {
        double best = std::numeric_limits<double>::max();
        Vector3d closest = Vector3d::Zero();
        for (const auto& [te, tp] : truth) {
            const double d = std::abs(pe - te);
            if (d < best) { best = d; closest = tp; }
        }
        if (best <= max_time_diff) {
            sum_sq += (pp - closest).squaredNorm();
            ++count;
        }
    }
    return count ? std::sqrt(sum_sq / count) : std::numeric_limits<double>::quiet_NaN();
}

}  // namespace ballistic::test
