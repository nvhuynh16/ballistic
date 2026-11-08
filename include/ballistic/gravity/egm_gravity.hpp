#pragma once

// =============================================================================
// EGM2008 Spherical Harmonic Gravity Model
// =============================================================================
//
// The Earth's gravity field is not that of a perfect sphere. Geodesists describe
// it as a "spherical harmonic" series: a sum of terms indexed by degree n and
// order m, each a fixed pattern over the globe weighted by a coefficient pair
// (C_nm, S_nm). Higher degrees describe finer detail (degree n resolves features
// of roughly 20000 km / n). EGM2008, the Earth Gravitational Model 2008, is one
// published set of these coefficients.
//
// This file provides EGM2008Gravity, a process-wide holder of the coefficients
// that evaluates the series with Clenshaw summation (a numerically stable way
// to add up the series without forming each term separately; see
// clenshaw_engine.hpp). Positions and accelerations are in ECEF (Earth-centered,
// Earth-fixed: axes that rotate with the Earth), in meters and m/s^2.
//
// References:
// -----------
// [1] Holmes, S.A. & Featherstone, W.E. (2002). "A unified approach to the
//     Clenshaw summation and the recursive computation of very high degree
//     and order normalized associated Legendre functions."
//     Journal of Geodesy, 76(5), 279-299.
//     DOI: 10.1007/s00190-002-0216-2
//
// [2] Pavlis, N.K., et al. (2012). "The development and evaluation of the
//     Earth Gravitational Model 2008 (EGM2008)."
//     Journal of Geophysical Research, 117, B04406.
//
// [3] Karney, C.F.F. (2011). GeographicLib library documentation.
//     https://geographiclib.sourceforge.io/
//
// =============================================================================

#include "../core/types.hpp"
#include "../core/constants.hpp"
#include "egm2008_coefficients.hpp"
#include "clenshaw_engine.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <atomic>
#include <mutex>

namespace ballistic {
namespace gravity {

// =============================================================================
// EGM2008 Model
// =============================================================================
// EGM2008 provides coefficients up to degree 2190. The library embeds them up
// to degree 360. The default evaluation degree used by the propagator is 70,
// which is enough for low Earth orbit (LEO).
//
// For degrees beyond 360, load the full file published by the US National
// Geospatial-Intelligence Agency (NGA), EGM2008_to2190_TideFree, with
// load_from_file().
// =============================================================================

/// Memory layout of the coefficient table. Both give identical results;
/// COLUMN_MAJOR is the one EGM2008Gravity computes with.
enum class GravityStorage {
    TRIANGULAR,      ///< index n*(n+1)/2 + m (the layout GeographicLib uses)
    COLUMN_MAJOR     ///< grouped by order m, degrees contiguous (faster: cache-friendly)
};

/// The process-wide EGM2008 gravity model.
///
/// Access it through instance(). It initializes itself from the embedded
/// coefficients (degree 360) the first time a computation needs it. Computations
/// use the column-major table.
///
/// Thread safety: the first-use initialization is locked, so several threads may
/// start computing at once. init() and load_from_file() take the same lock, but
/// replacing the coefficients while another thread is computing is still a data
/// race: load a file (or a larger degree) before propagating from several threads.
class EGM2008Gravity {
public:
    /// The single shared instance.
    static EGM2008Gravity& instance() {
        static EGM2008Gravity inst;
        return inst;
    }

    /// Highest degree of the embedded coefficients (360).
    static constexpr int MAX_EMBEDDED_DEGREE = egm2008_data::MAX_EMBEDDED_DEGREE;

    /// Load the embedded coefficients up to `max_degree` (default 360, all of them).
    ///
    /// A model already loaded to at least this degree is kept as it is.
    /// @throws ConfigError if max_degree > 360; use load_from_file() with the
    ///         NGA file for higher degrees.
    void init(int max_degree = 360) {
        std::lock_guard<std::mutex> lock(init_mutex_);
        init_unlocked(max_degree);
    }

    /// Load coefficients from a text file (needed for degrees above 360).
    ///
    /// Each data row is `n m C_nm S_nm [sigma_C sigma_S]` (fully normalized
    /// coefficients; Fortran 'D' exponents such as -0.484D-03 are accepted;
    /// blank lines and lines starting with '#' are skipped). Rows above
    /// `max_degree` are ignored. The coefficients go into the table that
    /// acceleration() reads (column-major), replacing the embedded set.
    /// @param filename    path to the file, e.g. NGA's EGM2008_to2190_TideFree
    /// @param max_degree  highest degree to keep (default 2190)
    /// @return false if the file cannot be opened or holds no degree >= 2.
    /// @throws ballistic::Error naming the line if a data row cannot be parsed.
    bool load_from_file(const std::filesystem::path& filename, int max_degree = 2190) {
        std::lock_guard<std::mutex> lock(init_mutex_);
        ready_.store(false, std::memory_order_release);
        // Whatever happens below (including a parse error thrown by the
        // engine), the flag ends up saying whether the table is usable.
        struct RestoreReady {
            EGM2008Gravity& g;
            ~RestoreReady() { g.ready_.store(g.active_initialized(), std::memory_order_release); }
        } restore{*this};
        const bool result = (storage_ == GravityStorage::COLUMN_MAJOR)
            ? colmajor_model_.load_from_file(filename, max_degree)
            : model_.load_from_file(filename, max_degree);
        if (result) {
            loaded_from_file_ = true;
            loaded_path_ = filename;
            loaded_degree_ = (storage_ == GravityStorage::COLUMN_MAJOR)
                ? colmajor_model_.max_degree() : model_.max_degree();
        }
        return result;
    }

private:
    // init() without taking the lock (the caller holds init_mutex_).
    void init_unlocked(int max_degree) {
        if (max_degree > MAX_EMBEDDED_DEGREE) {
            // Refuse rather than quietly cap the degree at 360.
            throw ConfigError("EGM2008Gravity::init: degree " + std::to_string(max_degree) +
                " exceeds the embedded coefficients (" + std::to_string(MAX_EMBEDDED_DEGREE) +
                "); load the NGA EGM2008_to2190 file with load_from_file for more");
        }
        const int effective_degree = max_degree;

        // A model that is already loaded to at least this degree is kept (a file
        // loaded to degree 2190 is not replaced by the embedded table). A request
        // above the loaded degree re-initializes: keeping the smaller model would
        // make a later acceleration(pos, max_degree) throw for a degree the
        // caller believes is loaded.
        if (storage_ == GravityStorage::COLUMN_MAJOR) {
            if (!colmajor_model_.is_initialized() || loaded_degree_ < effective_degree) {
                ready_.store(false, std::memory_order_release);
                colmajor_model_.init(constants::earth::GM, constants::earth::RADIUS_EQUATORIAL, effective_degree);
                load_embedded_coefficients(effective_degree);
                loaded_degree_ = effective_degree;
                loaded_from_file_ = false;
            }
        } else {
            if (!model_.is_initialized() || loaded_degree_ < effective_degree) {
                ready_.store(false, std::memory_order_release);
                model_.init(constants::earth::GM, constants::earth::RADIUS_EQUATORIAL, effective_degree);
                load_embedded_coefficients(effective_degree);
                loaded_degree_ = effective_degree;
                loaded_from_file_ = false;
            }
        }
        ready_.store(active_initialized(), std::memory_order_release);
    }

    // True if the table computations read has been filled.
    [[nodiscard]] bool active_initialized() const {
        return (storage_ == GravityStorage::COLUMN_MAJOR)
            ? colmajor_model_.is_initialized() : model_.is_initialized();
    }

public:
    /// Highest degree a computation may request. Loads the embedded model
    /// first if nothing is loaded yet.
    [[nodiscard]] int available_degree() {
        ensure_initialized();
        return max_degree();
    }

    /// Same as load_from_file() (kept as a more descriptive name for loading
    /// the full degree-2190 file). Returns true on success.
    bool load_full(const std::filesystem::path& filename, int max_degree = 2190) {
        return load_from_file(filename, max_degree);
    }

    /// True if the current coefficients came from a file rather than the
    /// embedded table.
    [[nodiscard]] bool is_loaded_from_file() const noexcept { return loaded_from_file_; }

    /// Degree of the currently loaded coefficients (0 before anything is loaded).
    [[nodiscard]] int loaded_degree() const noexcept { return loaded_degree_; }

    /// Gravitational acceleration [m/s^2] at an ECEF position [m], in ECEF.
    ///
    /// Includes the central (point-mass) term. `eval_degree` truncates the
    /// series; <= 0 means every loaded degree.
    /// @throws ConfigError if eval_degree exceeds the loaded degree.
    /// @throws ballistic::Error if the result is not finite (e.g. a NaN position).
    [[nodiscard]] Vector3d acceleration(const Vector3d& pos_ecef, int eval_degree = -1) {
        ensure_initialized();
        if (storage_ == GravityStorage::COLUMN_MAJOR) {
            return colmajor_model_.compute(pos_ecef, eval_degree).acceleration;
        } else {
            return model_.acceleration(pos_ecef, eval_degree);
        }
    }

    /// Gravitational potential [m^2/s^2] at an ECEF position [m].
    ///
    /// Geodesy sign convention: V = +GM/r + ..., so acceleration = +gradient(V).
    /// `eval_degree` and the exceptions are as for acceleration().
    [[nodiscard]] double potential(const Vector3d& pos_ecef, int eval_degree = -1) {
        ensure_initialized();
        if (storage_ == GravityStorage::COLUMN_MAJOR) {
            return colmajor_model_.compute(pos_ecef, eval_degree).potential;
        } else {
            return model_.potential(pos_ecef, eval_degree);
        }
    }

    /// Potential, acceleration and Hessian (the 3x3 matrix of second derivatives
    /// of the potential, i.e. how the acceleration changes as the position
    /// moves), all in ECEF. Used to build the force Jacobian.
    ///
    /// `fd_delta` is ignored (the Hessian is computed analytically; see
    /// ClenshawEngine::compute_with_hessian). Exceptions as for acceleration().
    [[nodiscard]] GravityHessianResult compute_with_hessian(
        const Vector3d& pos_ecef,
        int eval_degree = -1,
        double fd_delta = 1.0)
    {
        ensure_initialized();
        if (storage_ == GravityStorage::COLUMN_MAJOR) {
            return colmajor_model_.compute_with_hessian(pos_ecef, eval_degree, fd_delta);
        } else {
            return model_.compute_with_hessian(pos_ecef, eval_degree, fd_delta);
        }
    }

    /// The same coefficients in the triangular layout (the one GeographicLib
    /// uses). Built on first access from the loaded file or the embedded table;
    /// computations through this object do not use it.
    [[nodiscard]] ClenshawGravity& model() {
        ensure_triangular_initialized();
        return model_;
    }
    [[nodiscard]] const ClenshawGravity& model() const {
        const_cast<EGM2008Gravity*>(this)->ensure_triangular_initialized();
        return model_;
    }

    /// Degree of the table computations use (0 if nothing is loaded yet; unlike
    /// available_degree(), this does not trigger loading).
    [[nodiscard]] int max_degree() const {
        if (storage_ == GravityStorage::COLUMN_MAJOR) {
            return colmajor_model_.max_degree();
        }
        return model_.max_degree();
    }

    /// Layout computations use (always COLUMN_MAJOR at present).
    [[nodiscard]] GravityStorage storage() const noexcept { return storage_; }

    /// Direct access to the column-major engine that computations use. It is
    /// empty until something has been loaded (call available_degree() first).
    [[nodiscard]] Clenshaw& colmajor_model() { return colmajor_model_; }
    [[nodiscard]] const Clenshaw& colmajor_model() const { return colmajor_model_; }

private:
    EGM2008Gravity() = default;

    // Load the embedded coefficients into the active table on first use.
    // Double-checked: after the first load this is one acquire load; threads
    // that arrive while the table is being filled wait on the lock.
    void ensure_initialized() {
        if (ready_.load(std::memory_order_acquire)) return;
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (!active_initialized()) init_unlocked(MAX_EMBEDDED_DEGREE);
        ready_.store(true, std::memory_order_release);
    }

    // Fill the triangular table for model(): from the same file if one was
    // loaded, otherwise from the embedded coefficients.
    void ensure_triangular_initialized() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (!model_.is_initialized() && loaded_from_file_) {
            model_.load_from_file(loaded_path_, loaded_degree_);
        }
        if (!model_.is_initialized()) {
            int degree = loaded_degree_ > 0 ? loaded_degree_ : 360;
            model_.init(constants::earth::GM, constants::earth::RADIUS_EQUATORIAL, degree);
            for (const auto& coeff : egm2008_data::COEFFICIENTS) {
                if (coeff.n > degree) break;
                model_.set_coefficient(coeff.n, coeff.m, coeff.C, coeff.S);
            }
        }
    }
    ClenshawGravity model_;               // Triangular layout (for model() only)
    Clenshaw colmajor_model_;             // Column-major layout (used for computation)
    GravityStorage storage_ = GravityStorage::COLUMN_MAJOR;  // Layout used for computation
    bool loaded_from_file_ = false;
    std::filesystem::path loaded_path_;
    int loaded_degree_ = 0;
    std::atomic<bool> ready_{false};      // active table filled (fast-path flag)
    std::mutex init_mutex_;               // serializes every load of the tables

    // Copy the embedded coefficients up to max_degree into the active table
    // (the table is sorted by degree, hence the early break).
    void load_embedded_coefficients(int max_degree) {
        for (const auto& coeff : egm2008_data::COEFFICIENTS) {
            if (coeff.n > max_degree) break;
            if (storage_ == GravityStorage::COLUMN_MAJOR) {
                colmajor_model_.set_coefficient(coeff.n, coeff.m, coeff.C, coeff.S);
            } else {
                model_.set_coefficient(coeff.n, coeff.m, coeff.C, coeff.S);
            }
        }
    }
};

} // namespace gravity
} // namespace ballistic
