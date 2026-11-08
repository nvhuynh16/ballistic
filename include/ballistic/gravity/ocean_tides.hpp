#pragma once

// =============================================================================
// Ocean tides (IERS 2010 Conventions, Sec. 6.3)
// =============================================================================
// IERS is the International Earth Rotation and Reference Systems Service; its
// Conventions (2010) define the standard models used here.
//
// The water moved by ocean tides changes the Earth's gravity field slightly and
// periodically. The model expresses this as small time-varying corrections
// dC_nm, dS_nm to the gravity-field coefficients (the "Stokes coefficients"
// C_nm, S_nm of degree n and order m; see egm_gravity.hpp), one sinusoid per
// tidal constituent f (eq. 6.15):
//
//   dC_nm(t) = sum_f [(C+ + C-) cos th_f + (S+ + S-) sin th_f]
//   dS_nm(t) = sum_f [(S+ - S-) cos th_f - (C+ - C-) sin th_f]
//
// with th_f the constituent's phase angle (its Doodson argument, built from six
// slowly varying astronomical angles; tide_arguments.hpp) and C+, S+, C-, S- the
// constituent's tabulated amplitudes. The amplitudes come from the IERS
// Conventions Chapter 6 file `fes2004_Cnm-Snm.dat` (the FES2004 ocean-tide
// model, fully normalized, units 1e-11, to degree 100). The file is NOT
// embedded (third-party model; downloaded on demand — Python `ballistic.load()`,
// C++ `OceanTideModel::instance().load(path)`). Until it is loaded, the force
// model adds no ocean-tide acceleration.
//
// Over the 2025 Sentinel orbits, adding them (together with the
// frequency-dependent solid-tide corrections) lowered the 1-hour prediction
// error. The force model evaluates them to degree 50 by default
// (ForceModel::set_ocean_tide_degree).
// =============================================================================

#include "../core/errors.hpp"
#include "tide_arguments.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace ballistic {
namespace gravity {

/// Process-wide store of the ocean-tide amplitudes. All methods are thread-safe
/// (one mutex).
class OceanTideModel {
public:
    /// The single shared instance.
    static OceanTideModel& instance() {
        static OceanTideModel inst;
        return inst;
    }

    /// Load an IERS Stokes-variation file such as fes2004_Cnm-Snm.dat, replacing
    /// anything loaded before.
    ///
    /// Header lines (first token not a number) are skipped; data rows are
    /// `Doodson Darwin n m DelC+ DelS+ DelC- DelS-` (Darwin = the constituent's
    /// name, e.g. M2), amplitudes in units of 1e-11.
    /// Degree-1 rows and degrees above `max_degree` are dropped.
    /// @return false if the file cannot be opened or holds no usable row.
    /// @throws ballistic::Error naming the line if a data row cannot be parsed.
    bool load(const std::filesystem::path& path, int max_degree = 100) {
        std::ifstream f(path);
        if (!f) return false;
        std::vector<Wave> waves;
        std::string line;
        long line_no = 0;
        int nmax = 0;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::string doodson, name;
            int n = -1, m = -1;
            double cp, sp, cm, sm;
            ++line_no;
            // Header lines are text; a line whose first token is a number is
            // a data row and must parse (a bad row raises, never skipped silently).
            if (!(ss >> doodson)) continue;
            char* end = nullptr;
            std::strtod(doodson.c_str(), &end);
            if (end == doodson.c_str()) continue;            // header text
            const auto mult = parse_doodson(doodson);
            if (!(ss >> name >> n >> m >> cp >> sp >> cm >> sm) || !mult ||
                n < 0 || m < 0 || m > n) {
                throw Error("ocean-tide file " + path.string() + " line " +
                            std::to_string(line_no) + " is not a Stokes row: \"" + line + "\"");
            }
            if (n < 2 || n > max_degree) continue;   // degree 1 (geocenter) rows are zero by convention
            auto it = std::find_if(waves.begin(), waves.end(),
                                   [&](const Wave& w) { return w.n == *mult; });
            if (it == waves.end()) { waves.push_back({*mult, {}}); it = waves.end() - 1; }
            it->terms.push_back({n, m, cp * 1e-11, sp * 1e-11, cm * 1e-11, sm * 1e-11});
            nmax = std::max(nmax, n);
        }
        if (waves.empty()) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        waves_ = std::move(waves);
        max_degree_ = nmax;
        return true;
    }

    /// True if amplitudes are loaded.
    [[nodiscard]] bool is_loaded() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !waves_.empty();
    }
    /// Highest degree loaded (0 if none).
    [[nodiscard]] int max_degree() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return max_degree_;
    }
    /// Unload everything (the force model then adds no ocean tides).
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        waves_.clear();
        max_degree_ = 0;
    }

    /// The coefficient corrections dC, dS (eq. 6.15) for degrees 2..nmax.
    /// @param beta  the six Doodson arguments at the epoch, from
    ///              doodson_arguments() [rad]
    /// @param nmax  highest degree wanted
    /// @param dC,dS outputs, resized to (nmax+1)^2; entry (n, m) is at
    ///              index n*(nmax+1) + m. Entries not covered are zero.
    void stokes(const std::array<double, 6>& beta, int nmax,
                std::vector<double>& dC, std::vector<double>& dS) const {
        const int w = nmax + 1;
        dC.assign(static_cast<size_t>(w) * w, 0.0);
        dS.assign(static_cast<size_t>(w) * w, 0.0);
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& wave : waves_) {
            const double th = doodson_angle(wave.n, beta);
            const double c = std::cos(th), s = std::sin(th);
            for (const auto& t : wave.terms) {
                if (t.n > nmax) continue;
                const size_t i = static_cast<size_t>(t.n) * w + t.m;
                dC[i] += (t.cp + t.cm) * c + (t.sp + t.sm) * s;
                if (t.m > 0) dS[i] += (t.sp - t.sm) * c - (t.cp - t.cm) * s;
            }
        }
    }

private:
    struct Term { int n, m; double cp, sp, cm, sm; };
    struct Wave { std::array<int, 6> n; std::vector<Term> terms; };

    // Doodson number A.BCDEF (leading zeros may be dropped) -> multipliers
    // (A, B-5, C-5, D-5, E-5, F-5):
    // "255.555" -> (2, 0, 0, 0, 0, 0); "55.565" -> (0, 0, 0, 0, 1, 0)
    static std::optional<std::array<int, 6>> parse_doodson(const std::string& s) {
        const auto dot = s.find('.');
        if (dot == std::string::npos || s.size() - dot - 1 != 3 || dot > 3 || dot == 0) return std::nullopt;
        std::string digits = std::string(3 - dot, '0') + s.substr(0, dot) + s.substr(dot + 1);
        std::array<int, 6> n{};
        for (int i = 0; i < 6; ++i) {
            if (digits[i] < '0' || digits[i] > '9') return std::nullopt;
            n[i] = (digits[i] - '0') - (i == 0 ? 0 : 5);
        }
        return n;
    }

    OceanTideModel() = default;
    mutable std::mutex mutex_;
    std::vector<Wave> waves_;
    int max_degree_ = 0;
};

}  // namespace gravity
}  // namespace ballistic
