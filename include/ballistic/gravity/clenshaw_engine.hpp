#pragma once

// =============================================================================
// Unified Clenshaw Summation Gravity Engine
// =============================================================================
//
// Evaluates a spherical harmonic gravity field (a sum of terms of degree n and
// order m weighted by coefficients C_nm, S_nm; see egm_gravity.hpp) at one
// position. Clenshaw summation adds the series from the highest degree down
// using a three-term recurrence, so the associated Legendre functions (the
// latitude-dependent factor of each term) never have to be formed one by one;
// this stays accurate to degree 2190 where forming them directly would
// overflow or underflow.
//
// The coefficient table layout is a template parameter (StoragePolicy, see
// clenshaw_storage.hpp). Positions are ECEF (Earth-centered, Earth-fixed) in
// meters; outputs are in ECEF too.
//
// Coordinates used inside: r (distance from the Earth's center), phi
// (geocentric latitude), lambda (longitude); t = sin(phi), u = cos(phi),
// q = R/r with R the model's reference radius. The single-letter accumulator
// names (wc, wrc, vtc, ...) follow GeographicLib's SphericalEngine and the
// derivation in docs/gravity_hessian_derivation.md: w = inner sum over degree,
// v = outer sum over order, then r/t/l = derivative with respect to
// r/phi/lambda, c/s = cosine/sine series, and a trailing 2 = the previous
// iterate.
//
// References:
// -----------
// [1] Holmes, S.A. & Featherstone, W.E. (2002). "A unified approach..."
//     Journal of Geodesy, 76(5), 279-299.
//
// [2] Karney, C.F.F. GeographicLib SphericalEngine implementation
//     https://geographiclib.sourceforge.io/
//
// [3] Elodin optimization: precomputed position-independent recursion factors
//     https://www.elodin.systems/post/a-better-approach-to-gravity-how-we-made-egm2008-faster
// =============================================================================

#include "../core/types.hpp"
#include "../core/constants.hpp"
#include "../core/defaults.hpp"
#include "../core/errors.hpp"
#include "clenshaw_storage.hpp"
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <limits>

namespace ballistic {
namespace gravity {

/// Acceleration and potential at one point.
struct GravityResult {
    Vector3d acceleration;  ///< ECEF [m/s^2]
    double potential;       ///< gravitational potential [m^2/s^2], sign convention V = +GM/r
};

/// Acceleration, potential and Hessian at one point.
///
/// All outputs are in ECEF. The force model rotates the Hessian into the
/// inertial propagation frame (R * H * R^T; docs/gravity_hessian_derivation.md §3.5).
///
/// hessian_ecef(i, j) = d a_i / d x_j = d^2 V / (d x_i d x_j): how acceleration
/// component i changes per meter of position component j. With the geodesy
/// convention V = +GM/r, acceleration = +gradient(V), and the matrix is symmetric.
struct GravityHessianResult {
    Vector3d acceleration;        // ECEF [m/s²]
    double potential;             // [m²/s²]
    Eigen::Matrix3d hessian_ecef; // ECEF [1/s²]
};

/// Spherical harmonic gravity evaluator for one coefficient set.
///
/// Typical use: init(GM, R, N), set_coefficient() for each (n, m) (or
/// load_from_file()), then compute() at any number of positions. compute* are
/// const and keep no state, so one loaded engine can be shared by threads.
/// The aliases at the end of the file name the two layouts.
template<typename StoragePolicy>
class ClenshawEngine {
public:
    ClenshawEngine() = default;

    /// Size the model and set its constants; every coefficient is zero except
    /// C_00 = 1 (the central point-mass term).
    /// @param GM          gravitational parameter [m^3/s^2]
    /// @param radius      reference radius of the coefficients [m]
    /// @param max_degree  highest degree to hold
    void init(double GM, double radius, int max_degree) {
        GM_ = GM;
        radius_ = radius;
        max_degree_ = max_degree;

        storage_.allocate(max_degree);

        // Table of sqrt(i), used by the recurrence factors. The m = 0 step reads
        // sqrt(15) whatever the degree, so the table never has fewer than 16 entries.
        int root_size = std::max(2 * max_degree + 6, 16);
        root_.resize(root_size);
        for (int i = 0; i < root_size; ++i) {
            root_[i] = std::sqrt(static_cast<double>(i));
        }

        storage_.precompute_recursion_factors(max_degree, root_);
        initialized_ = true;
    }

    /// Load "n m C S [sigmaC sigmaS]" rows (Fortran D exponents accepted; '#'
    /// comments and blank lines skipped). Rows with n > max_degree_load are
    /// ignored (-1 = keep all). GM and radius stay as they are.
    /// The model is sized once to the highest degree read, so a file sorted by
    /// degree keeps every row (growing the model row by row would re-allocate,
    /// zeroing everything read so far).
    /// @return false if the file cannot be opened or holds no degree >= 2.
    /// @throws ballistic::Error naming the line when a data row cannot be parsed.
    bool load_from_file(const std::filesystem::path& filename, int max_degree_load = -1) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;

        struct Row { int n, m; double C, S; };
        std::vector<Row> rows;
        std::string line;
        int max_n_loaded = 0;
        long line_no = 0;

        while (std::getline(file, line)) {
            ++line_no;
            if (line.find_first_not_of(" \t\r") == std::string::npos || line[0] == '#') continue;

            int n, m;
            double Cnm, Snm;

            std::replace(line.begin(), line.end(), 'D', 'E');
            std::replace(line.begin(), line.end(), 'd', 'e');

            if (std::sscanf(line.c_str(), "%d %d %lf %lf", &n, &m, &Cnm, &Snm) != 4 ||
                n < 0 || m < 0 || m > n) {
                throw Error("gravity file " + filename.string() + " line " +
                            std::to_string(line_no) + " is not 'n m C S': \"" + line +
                            "\". The file is damaged or not an EGM coefficient file.");
            }

            if (max_degree_load >= 0 && n > max_degree_load) continue;
            if (n > max_n_loaded) max_n_loaded = n;
            rows.push_back({n, m, Cnm, Snm});
        }
        if (max_n_loaded < 2) return false;

        init(GM_, radius_, std::max(max_n_loaded, initialized_ ? max_degree_ : 0));
        for (const auto& r : rows) set_coefficient(r.n, r.m, r.C, r.S);
        return true;
    }

    /// Set the fully normalized coefficient pair of degree n, order m.
    /// Invalid indices (negative, or m > n) are ignored. A degree above the
    /// current size re-initializes the model to that degree, which ERASES all
    /// coefficients set so far — set the highest degree first, or call init().
    void set_coefficient(int n, int m, double Cnm, double Snm) {
        if (n < 0 || m < 0 || m > n) return;
        if (!initialized_ || n > max_degree_) {
            init(GM_, radius_, n);
        }
        storage_.set_coefficient(n, m, Cnm, Snm, max_degree_);
    }

    /// The (C_nm, S_nm) pair; (0, 0) for indices outside the model or before init().
    [[nodiscard]] std::pair<double, double> get_coefficient(int n, int m) const {
        if (!initialized_) return {0.0, 0.0};
        return storage_.get_coefficient(n, m, max_degree_);
    }

    /// Acceleration [m/s^2] and potential [m^2/s^2] at an ECEF position [m].
    ///
    /// @param eval_degree  truncate the series at this degree; <= 0 = all loaded.
    /// Before init() (or for a model of degree 0) only the point-mass term is returned
    /// (potential +GM/r, the same convention as the series).
    /// Within 1 m of the center the result is zero.
    /// @throws ConfigError if eval_degree exceeds the loaded degree.
    /// @throws ballistic::Error if the acceleration is not finite.
    [[nodiscard]] GravityResult compute(const Vector3d& pos_ecef, int eval_degree = -1) const {
        GravityResult result;
        result.acceleration = Vector3d::Zero();
        result.potential = 0.0;

        if (!initialized_ || max_degree_ == 0) {
            // No coefficients: point mass only (V = +GM/r, as the series)
            double r = pos_ecef.norm();
            if (r > 1.0) {
                result.acceleration = -GM_ / (r * r * r) * pos_ecef;
                result.potential = GM_ / r;
            }
            return result;
        }

        const double x = pos_ecef.x();
        const double y = pos_ecef.y();
        const double z = pos_ecef.z();

        const double r2 = x*x + y*y + z*z;
        const double r = std::sqrt(r2);
        const double p = std::sqrt(x*x + y*y);

        if (r < 1.0) return result;

        // Spherical coordinates (GeographicLib convention)
        const double t = z / r;      // sin(geocentric_lat)
        const double u = p / r;      // cos(geocentric_lat)
        const double q = radius_ / r;

        // cos/sin of longitude; on the polar axis longitude is undefined, so
        // any fixed value (lambda = 0) gives the same field there
        double cl, sl;
        if (p > defaults::GRAVITY_POLE_RATIO_THRESHOLD * r) {
            cl = x / p;
            sl = y / p;
        } else {
            cl = 1.0;
            sl = 0.0;
        }

        const double q2 = q * q;
        const double tu = (std::abs(u) > defaults::GRAVITY_POLE_U_THRESHOLD) ? t / u : 0.0;
        const double uq = u * q;
        const double uq2 = uq * uq;

        const int N = checked_degree(eval_degree);

        // Outer sum accumulators
        double vc = 0, vc2 = 0;
        double vs = 0, vs2 = 0;
        double vrc = 0, vrc2 = 0;
        double vrs = 0, vrs2 = 0;
        double vtc = 0, vtc2 = 0;
        double vts = 0, vts2 = 0;
        double vlc = 0, vlc2 = 0;
        double vls = 0, vls2 = 0;

        // Outer sum over order m, from N down to 0 (Clenshaw runs backwards)
        for (int m = N; m >= 0; --m) {
            // Inner sum accumulators
            double wc = 0, wc2 = 0;
            double ws = 0, ws2 = 0;
            double wrc = 0, wrc2 = 0;
            double wrs = 0, wrs2 = 0;
            double wtc = 0, wtc2 = 0;
            double wts = 0, wts2 = 0;

            // Inner sum over degree n, from N down to m
            for (int n = N; n >= m; --n) {
                const auto& rf = storage_.get_rf(n, m);
                const double Ax = q * rf.w_sqrt2n3;
                const double A = t * Ax;
                const double B = -q2 * rf.b_factor;

                const double Cv = storage_.get_C(n, m);
                const double Sv = (m > 0) ? storage_.get_S(n, m) : 0.0;

                // Value
                double temp = A * wc + B * wc2 + Cv;
                wc2 = wc;
                wc = temp;

                // Radial derivative
                temp = A * wrc + B * wrc2 + (n + 1) * Cv;
                wrc2 = wrc;
                wrc = temp;

                // Latitude derivative
                temp = A * wtc + B * wtc2 - u * Ax * wc2;
                wtc2 = wtc;
                wtc = temp;

                // Sine terms (for m > 0)
                if (m > 0) {
                    temp = A * ws + B * ws2 + Sv;
                    ws2 = ws;
                    ws = temp;

                    temp = A * wrs + B * wrs2 + (n + 1) * Sv;
                    wrs2 = wrs;
                    wrs = temp;

                    temp = A * wts + B * wts2 - u * Ax * ws2;
                    wts2 = wts;
                    wts = temp;
                }
            }

            // Sectorial (n = m) term of the latitude derivative
            if (m > 0) {
                wtc += m * tu * wc;
                wts += m * tu * ws;
            }

            // Outer recurrence step over order
            if (m > 0) {
                double v = root_[2] * root_[2*m + 3] / root_[m + 1];
                double A = cl * v * uq;
                double B = -v * root_[2*m + 5] / (root_[8] * root_[m + 2]) * uq2;

                double temp;
                temp = A * vc + B * vc2 + wc;   vc2 = vc;   vc = temp;
                temp = A * vs + B * vs2 + ws;   vs2 = vs;   vs = temp;
                temp = A * vrc + B * vrc2 + wrc; vrc2 = vrc; vrc = temp;
                temp = A * vrs + B * vrs2 + wrs; vrs2 = vrs; vrs = temp;
                temp = A * vtc + B * vtc2 + wtc; vtc2 = vtc; vtc = temp;
                temp = A * vts + B * vts2 + wts; vts2 = vts; vts = temp;
                temp = A * vlc + B * vlc2 + m * ws; vlc2 = vlc; vlc = temp;
                temp = A * vls + B * vls2 - m * wc; vls2 = vls; vls = temp;
            } else {
                // m = 0: final combination
                double A = root_[3] * uq;
                double B = -root_[15] / 2.0 * uq2;

                vc = q * (wc + A * (cl * vc + sl * vs) + B * vc2);

                double qs_r = q / r;
                vrc = -qs_r * (wrc + A * (cl * vrc + sl * vrs) + B * vrc2);
                vtc = qs_r * (wtc + A * (cl * vtc + sl * vts) + B * vtc2);
                vlc = (std::abs(u) > defaults::GRAVITY_POLE_U_THRESHOLD) ?
                      qs_r / u * (A * (cl * vlc + sl * vls) + B * vlc2) : 0.0;
            }
        }

        // Scale, and turn the spherical gradient (r, phi, lambda) into ECEF x/y/z
        const double scale_factor = GM_ / radius_;
        result.potential = scale_factor * vc;

        double gradx = cl * (u * vrc + t * vtc) - sl * vlc;
        double grady = sl * (u * vrc + t * vtc) + cl * vlc;
        double gradz = t * vrc - u * vtc;

        result.acceleration = scale_factor * Vector3d(gradx, grady, gradz);

        if (!result.acceleration.allFinite()) throw_not_finite(pos_ecef);

        return result;
    }

    /// compute(...).acceleration
    [[nodiscard]] Vector3d acceleration(const Vector3d& pos_ecef, int eval_degree = -1) const {
        return compute(pos_ecef, eval_degree).acceleration;
    }

    /// compute(...).potential
    [[nodiscard]] double potential(const Vector3d& pos_ecef, int eval_degree = -1) const {
        return compute(pos_ecef, eval_degree).potential;
    }

    /// Acceleration, potential and Hessian (second derivatives of the potential)
    /// at an ECEF position [m]; the default way to get the Hessian.
    ///
    /// Computed analytically in one pass of an extended Clenshaw recurrence
    /// (docs/gravity_hessian_derivation.md; each step checked symbolically with
    /// SymPy and in high precision with mpmath). test_gravity_hessian checks it
    /// against the closed-form J2 Hessian, against fourth-order central finite
    /// differences of acceleration() for single (n, m) terms and for the full
    /// field to degree 70 (1e-9 relative), and against compute_with_hessian_fd().
    ///
    /// The third argument (a finite-difference step) is ignored; it is kept so
    /// both methods share a signature. Exceptions as for compute().
    [[nodiscard]] GravityHessianResult compute_with_hessian(
        const Vector3d& pos_ecef,
        int eval_degree = -1,
        double /*fd_delta_unused*/ = 1.0) const
    {
        return compute_with_hessian_analytic(pos_ecef, eval_degree);
    }

    /// Hessian by central finite differences (FD) of the analytic acceleration,
    /// for cross-checking compute_with_hessian(). Costs 7 evaluations (1
    /// central + 6 displaced by +/- fd_delta along each axis).
    /// @param fd_delta  displacement [m]
    [[nodiscard]] GravityHessianResult compute_with_hessian_fd(
        const Vector3d& pos_ecef,
        int eval_degree = -1,
        double fd_delta = 1.0) const
    {
        // Central evaluation: gradient + potential
        GravityResult central = compute(pos_ecef, eval_degree);

        Eigen::Matrix3d H_raw;
        for (int j = 0; j < 3; ++j) {
            Vector3d dx = Vector3d::Zero();
            dx(j) = fd_delta;
            const Vector3d a_p = compute(pos_ecef + dx, eval_degree).acceleration;
            const Vector3d a_m = compute(pos_ecef - dx, eval_degree).acceleration;
            H_raw.col(j) = (a_p - a_m) / (2.0 * fd_delta);
        }

        // The exact Hessian is symmetric; averaging removes round-off asymmetry
        const Eigen::Matrix3d H_sym = 0.5 * (H_raw + H_raw.transpose());

        return {central.acceleration, central.potential, H_sym};
    }

    /// The analytic Hessian behind compute_with_hessian() (same result).
    ///
    /// It runs the compute() recurrence plus second-derivative accumulators:
    /// in the inner loop wrrc/wrtc/wttc, their sine twins and the auxiliary
    /// wphic/wphis; in the outer loop vrrc/vrtc/vttc/vrlc/vtlc/vllc and their
    /// sine twins. It then turns the spherical second derivatives into the
    /// Cartesian Hessian. Where each formula comes from, in
    /// docs/gravity_hessian_derivation.md:
    ///   §2.6.1-2.6.3  inner-loop second-derivative recurrences (cosine)
    ///   §2.6.4        their sine twins
    ///   §2.6.5        sectorial (n = m) corrections, with the wphic auxiliary
    ///   §2.5, §2.7    outer-loop recurrences, longitude source-term rule
    ///   §3.4          m = 0 finalization
    ///   §3.4.1        conversion to the natural spherical Hessian
    ///   §3.1-3.3      chain rule and Cartesian assembly (the K matrices)
    /// Exceptions as for compute().
    [[nodiscard]] GravityHessianResult compute_with_hessian_analytic(
        const Vector3d& pos_ecef,
        int eval_degree = -1) const
    {
        GravityHessianResult result;
        result.acceleration = Vector3d::Zero();
        result.potential = 0.0;
        result.hessian_ecef = Eigen::Matrix3d::Zero();

        if (!initialized_ || max_degree_ == 0) {
            // No coefficients: point mass only, closed form (V = +GM/r, as the series)
            const double r = pos_ecef.norm();
            if (r > 1.0) {
                const double r3 = r * r * r;
                const double r5 = r3 * r * r;
                result.acceleration = -GM_ / r3 * pos_ecef;
                result.potential = GM_ / r;
                result.hessian_ecef = -GM_ / r3 * Eigen::Matrix3d::Identity()
                                      + 3.0 * GM_ / r5 * (pos_ecef * pos_ecef.transpose());
            }
            return result;
        }

        // ===== Setup (same as compute()) =====
        const double x = pos_ecef.x();
        const double y = pos_ecef.y();
        const double z = pos_ecef.z();
        const double r2 = x*x + y*y + z*z;
        const double r = std::sqrt(r2);
        const double p = std::sqrt(x*x + y*y);
        if (r < 1.0) return result;

        const double t = z / r;
        const double u = p / r;
        const double q = radius_ / r;
        const double q2 = q * q;

        const bool u_safe = (std::abs(u) > defaults::GRAVITY_POLE_U_THRESHOLD);
        const double tu = u_safe ? t / u : 0.0;
        const double inv_u2 = u_safe ? 1.0 / (u * u) : 0.0;
        const double uq = u * q;
        const double uq2 = uq * uq;

        double cl, sl;
        if (p > defaults::GRAVITY_POLE_RATIO_THRESHOLD * r) {
            cl = x / p; sl = y / p;
        } else {
            cl = 1.0; sl = 0.0;
        }

        const int N = checked_degree(eval_degree);

        // ===== Outer-loop accumulators =====
        double vc = 0, vc2 = 0, vs = 0, vs2 = 0;
        double vrc = 0, vrc2 = 0, vrs = 0, vrs2 = 0;
        double vtc = 0, vtc2 = 0, vts = 0, vts2 = 0;
        double vlc = 0, vlc2 = 0, vls = 0, vls2 = 0;
        double vrrc = 0, vrrc2 = 0, vrrs = 0, vrrs2 = 0;
        double vrtc = 0, vrtc2 = 0, vrts = 0, vrts2 = 0;
        double vttc = 0, vttc2 = 0, vtts = 0, vtts2 = 0;
        double vrlc = 0, vrlc2 = 0, vrls = 0, vrls2 = 0;
        double vtlc = 0, vtlc2 = 0, vtls = 0, vtls2 = 0;
        double vllc = 0, vllc2 = 0, vlls = 0, vlls2 = 0;

        for (int m = N; m >= 0; --m) {
            // ===== Inner-loop accumulators =====
            double wc = 0, wc2 = 0, ws = 0, ws2 = 0;
            double wrc = 0, wrc2 = 0, wrs = 0, wrs2 = 0;
            double wtc = 0, wtc2 = 0, wts = 0, wts2 = 0;
            double wrrc = 0, wrrc2 = 0, wrrs = 0, wrrs2 = 0;
            double wrtc = 0, wrtc2 = 0, wrts = 0, wrts2 = 0;
            double wttc = 0, wttc2 = 0, wtts = 0, wtts2 = 0;
            double wphic = 0, wphic2 = 0, wphis = 0, wphis2 = 0;

            for (int n = N; n >= m; --n) {
                const auto& rf = storage_.get_rf(n, m);
                const double Ax = q * rf.w_sqrt2n3;
                const double A = t * Ax;
                const double B = -q2 * rf.b_factor;
                const double Cv = storage_.get_C(n, m);
                const double Sv = (m > 0) ? storage_.get_S(n, m) : 0.0;

                // Second-derivative updates come BEFORE the first-derivative ones:
                // they need the values of wc, wc2, wrc, wrc2, wtc, wtc2 from the previous step.
                double temp;

                // wrrc, wrrs (∂²V/∂r²) — derivation §2.6.1, §2.6.4
                temp = A * wrrc + B * wrrc2 - (A / r) * wrc - (2 * B / r) * wrc2;
                wrrc2 = wrrc; wrrc = temp;
                if (m > 0) {
                    temp = A * wrrs + B * wrrs2 - (A / r) * wrs - (2 * B / r) * wrs2;
                    wrrs2 = wrrs; wrrs = temp;
                }

                // wrtc, wrts (∂²V/∂r∂φ) — §2.6.2, §2.6.4
                temp = A * wrtc + B * wrtc2 + u * Ax * wrc;
                wrtc2 = wrtc; wrtc = temp;
                if (m > 0) {
                    temp = A * wrts + B * wrts2 + u * Ax * wrs;
                    wrts2 = wrts; wrts = temp;
                }

                // wttc, wtts (∂²V/∂φ², engine convention) — §2.6.3, §2.6.4
                // This is the derivative of the wtc update below, whose last term
                // is -u·Ax·wc2 evaluated AFTER wc has been advanced, i.e. with
                // wc2 = wc(n+1), not wc(n+2). Its derivative therefore needs
                // wc(n+1) and wphic(n+1) — and at this point (before this step's
                // first-derivative updates) those are held in `wc` and `wphic`,
                // not in `wc2`/`wphic2`. The same holds for the t·Ax·wc term.
                //
                // Check by hand for J2 only (degree 2) on the equator:
                //   ∂wtc(n=0)/∂φ computed directly = -2·Ax(n=0)·Ax(n=1)·Cv(n=2),
                //   and this recurrence gives the same value.
                temp = A * wttc + B * wttc2 + u * Ax * wtc + t * Ax * wc - u * Ax * wphic;
                wttc2 = wttc; wttc = temp;
                if (m > 0) {
                    temp = A * wtts + B * wtts2 + u * Ax * wts + t * Ax * ws - u * Ax * wphis;
                    wtts2 = wtts; wtts = temp;
                }

                // wphic, wphis (auxiliary: ∂wc/∂φ in the natural convention) — §2.6.3
                // Updated AFTER wttc, which needs the previous step's value.
                temp = A * wphic + B * wphic2 + u * Ax * wc;
                wphic2 = wphic; wphic = temp;
                if (m > 0) {
                    temp = A * wphis + B * wphis2 + u * Ax * ws;
                    wphis2 = wphis; wphis = temp;
                }

                // First-derivative updates (same as compute())
                temp = A * wc + B * wc2 + Cv;
                wc2 = wc; wc = temp;
                temp = A * wrc + B * wrc2 + (n + 1) * Cv;
                wrc2 = wrc; wrc = temp;
                temp = A * wtc + B * wtc2 - u * Ax * wc2;
                wtc2 = wtc; wtc = temp;
                if (m > 0) {
                    temp = A * ws + B * ws2 + Sv;
                    ws2 = ws; ws = temp;
                    temp = A * wrs + B * wrs2 + (n + 1) * Sv;
                    wrs2 = wrs; wrs = temp;
                    temp = A * wts + B * wts2 - u * Ax * ws2;
                    wts2 = wts; wts = temp;
                }
            }

            // Sectorial (n = m) corrections (m > 0) — §2.6.5
            if (m > 0) {
                wttc += m * inv_u2 * wc + m * tu * wphic;
                wtts += m * inv_u2 * ws + m * tu * wphis;
                wtc += m * tu * wc;
                wts += m * tu * ws;
            }

            // Outer recurrence step over order
            if (m > 0) {
                const double v = root_[2] * root_[2*m + 3] / root_[m + 1];
                const double A_outer = cl * v * uq;
                const double B_outer = -v * root_[2*m + 5] / (root_[8] * root_[m + 2]) * uq2;

                double temp;

                // Second-derivative outer updates FIRST: they use the previous vrc/vrc2/vtc/vtc2 (§2.7)
                temp = A_outer * vrrc + B_outer * vrrc2 + wrrc
                       - (A_outer / r) * vrc - (2 * B_outer / r) * vrc2;
                vrrc2 = vrrc; vrrc = temp;
                temp = A_outer * vrrs + B_outer * vrrs2 + wrrs
                       - (A_outer / r) * vrs - (2 * B_outer / r) * vrs2;
                vrrs2 = vrrs; vrrs = temp;

                temp = A_outer * vrtc + B_outer * vrtc2 + wrtc
                       - tu * A_outer * vrc - 2 * tu * B_outer * vrc2;
                vrtc2 = vrtc; vrtc = temp;
                temp = A_outer * vrts + B_outer * vrts2 + wrts
                       - tu * A_outer * vrs - 2 * tu * B_outer * vrs2;
                vrts2 = vrts; vrts = temp;

                temp = A_outer * vttc + B_outer * vttc2 + wttc
                       - tu * A_outer * vtc - 2 * tu * B_outer * vtc2;
                vttc2 = vttc; vttc = temp;
                temp = A_outer * vtts + B_outer * vtts2 + wtts
                       - tu * A_outer * vts - 2 * tu * B_outer * vts2;
                vtts2 = vtts; vtts = temp;

                // Longitude second derivatives (source-term rule, §2.5)
                temp = A_outer * vrlc + B_outer * vrlc2 + m * wrs;
                vrlc2 = vrlc; vrlc = temp;
                temp = A_outer * vrls + B_outer * vrls2 - m * wrc;
                vrls2 = vrls; vrls = temp;

                temp = A_outer * vtlc + B_outer * vtlc2 + m * wts;
                vtlc2 = vtlc; vtlc = temp;
                temp = A_outer * vtls + B_outer * vtls2 - m * wtc;
                vtls2 = vtls; vtls = temp;

                temp = A_outer * vllc + B_outer * vllc2 - m * m * wc;
                vllc2 = vllc; vllc = temp;
                temp = A_outer * vlls + B_outer * vlls2 - m * m * ws;
                vlls2 = vlls; vlls = temp;

                // First-derivative outer updates (same as compute(), run SECOND)
                temp = A_outer * vc + B_outer * vc2 + wc;   vc2 = vc;   vc = temp;
                temp = A_outer * vs + B_outer * vs2 + ws;   vs2 = vs;   vs = temp;
                temp = A_outer * vrc + B_outer * vrc2 + wrc; vrc2 = vrc; vrc = temp;
                temp = A_outer * vrs + B_outer * vrs2 + wrs; vrs2 = vrs; vrs = temp;
                temp = A_outer * vtc + B_outer * vtc2 + wtc; vtc2 = vtc; vtc = temp;
                temp = A_outer * vts + B_outer * vts2 + wts; vts2 = vts; vts = temp;
                temp = A_outer * vlc + B_outer * vlc2 + m * ws; vlc2 = vlc; vlc = temp;
                temp = A_outer * vls + B_outer * vls2 - m * wc; vls2 = vls; vls = temp;
            } else {
                // m = 0 finalization — §3.4
                const double A_final = root_[3] * uq;
                const double B_final = -root_[15] / 2.0 * uq2;
                const double qs_r = q / r;

                // Keep the m = 1 first-derivative values: the second-derivative finalization needs them
                const double vrc_m1 = vrc, vrc2_m1 = vrc2, vrs_m1 = vrs;
                const double vtc_m1 = vtc, vtc2_m1 = vtc2, vts_m1 = vts;
                const double vlc_m1 = vlc, vlc2_m1 = vlc2, vls_m1 = vls;

                // First-derivative finalization (identical to compute())
                vc = q * (wc + A_final * (cl * vc + sl * vs) + B_final * vc2);
                vrc = -qs_r * (wrc + A_final * (cl * vrc_m1 + sl * vrs_m1) + B_final * vrc2_m1);
                vtc = qs_r * (wtc + A_final * (cl * vtc_m1 + sl * vts_m1) + B_final * vtc2_m1);
                vlc = u_safe ?
                      qs_r / u * (A_final * (cl * vlc_m1 + sl * vls_m1) + B_final * vlc2_m1) : 0.0;

                // Second-derivative finalization (uses the finalized vrc, vtc, vlc and the saved m = 1 values)
                {
                    const double new_val = -(2.0 / r) * vrc - qs_r * (
                        wrrc + A_final * (cl * vrrc + sl * vrrs) + B_final * vrrc2
                        - (A_final / r) * (cl * vrc_m1 + sl * vrs_m1)
                        - (2 * B_final / r) * vrc2_m1
                    );
                    vrrc = new_val;
                }
                {
                    const double new_val = -qs_r * (
                        wrtc + A_final * (cl * vrtc + sl * vrts) + B_final * vrtc2
                        - tu * A_final * (cl * vrc_m1 + sl * vrs_m1)
                        - 2 * tu * B_final * vrc2_m1
                    );
                    vrtc = new_val;
                }
                {
                    const double new_val = qs_r * (
                        wttc + A_final * (cl * vttc + sl * vtts) + B_final * vttc2
                        - tu * A_final * (cl * vtc_m1 + sl * vts_m1)
                        - 2 * tu * B_final * vtc2_m1
                    );
                    vttc = new_val;
                }
                // Longitude second-derivative finalization.
                //
                // Differentiating the finalized vlc with respect to λ would
                // treat vllc as ∂vlc(m=1)/∂λ. It is not: by the source-term
                // rule (§2.5), vllc is the output of the same recurrence driven
                // by the source -m²·wc. The two readings differ for sectorial
                // cosine terms (n = m >= 2) and break Laplace's equation (the
                // trace of the Hessian must be zero outside the Earth).
                //
                // So vllc is finalized with the same pattern as vlc, and no
                // (-sl·vlc + cl·vls) correction term is added.
                {
                    const double new_val = -qs_r * (
                        A_final * (cl * vrlc + sl * vrls)
                        + B_final * vrlc2
                    );
                    vrlc = new_val;
                }
                {
                    const double new_val = qs_r * (
                        A_final * (cl * vtlc + sl * vtls)
                        + B_final * vtlc2
                    );
                    vtlc = new_val;
                }
                {
                    const double new_val = u_safe ?
                        qs_r / u * (
                            A_final * (cl * vllc + sl * vlls)
                            + B_final * vllc2
                        ) : 0.0;
                    vllc = new_val;
                }
            }
        }

        // ===== Final assembly =====
        const double scale_factor = GM_ / radius_;
        result.potential = scale_factor * vc;

        // Cartesian gradient (same as compute())
        const double gradx = cl * (u * vrc + t * vtc) - sl * vlc;
        const double grady = sl * (u * vrc + t * vtc) + cl * vlc;
        const double gradz = t * vrc - u * vtc;
        result.acceleration = scale_factor * Vector3d(gradx, grady, gradz);

        // ===== Spherical Hessian in natural coordinates (§3.4.1) =====
        const double H_rr = scale_factor * vrrc;
        const double H_rt = scale_factor * vrtc;
        const double H_tt = scale_factor * (-r) * vttc;
        const double H_rl = scale_factor * vrlc;
        const double H_tl = scale_factor * (-r) * vtlc;
        const double H_ll = scale_factor * (r * u) * vllc;

        // First derivatives with respect to q = (r, φ, λ) (§3.2)
        const double dV_dr = scale_factor * vrc;
        const double dV_dphi = scale_factor * (-r) * vtc;
        const double dV_dlam = scale_factor * (r * u) * vlc;

        // ===== Cartesian Hessian assembly (§3.3) =====
        // H = J^T · H_sph · J + Σ_a (∂V/∂q^a) · K^a, with J^a_i = ∂q^a/∂x_i (§3.1)
        // and K^a_ij = ∂²q^a/∂x_i∂x_j (the curvature of the spherical coordinates).
        const double Jr_x = u * cl, Jr_y = u * sl, Jr_z = t;
        const double Jp_x = -t * cl / r, Jp_y = -t * sl / r, Jp_z = u / r;
        const double Jl_x = u_safe ? -sl / (r * u) : 0.0;
        const double Jl_y = u_safe ? cl / (r * u) : 0.0;
        const double Jl_z = 0.0;

        Eigen::Matrix3d J_inv;
        J_inv << Jr_x, Jr_y, Jr_z,
                 Jp_x, Jp_y, Jp_z,
                 Jl_x, Jl_y, Jl_z;

        Eigen::Matrix3d H_sph_mat;
        H_sph_mat << H_rr, H_rt, H_rl,
                     H_rt, H_tt, H_tl,
                     H_rl, H_tl, H_ll;

        // Term 1: J^T · H_sph · J
        Eigen::Matrix3d H_cart = J_inv.transpose() * H_sph_mat * J_inv;

        // Term 2: Σ_a (∂V/∂q^a) · K^a_{ij}  (§3.3)
        const double r3 = r * r2;
        const double r4 = r2 * r2;
        const double Kr_xx = (y*y + z*z) / r3;
        const double Kr_yy = (x*x + z*z) / r3;
        const double Kr_zz = (x*x + y*y) / r3;
        const double Kr_xy = -x * y / r3;
        const double Kr_xz = -x * z / r3;
        const double Kr_yz = -y * z / r3;

        const double p2_loc = p * p;
        const double p3_loc = p2_loc * p;
        const double p4_loc = p2_loc * p2_loc;
        const bool p_safe = (p > 1.0);
        const double Kl_xx = p_safe ? 2.0 * x * y / p4_loc : 0.0;
        const double Kl_yy = p_safe ? -2.0 * x * y / p4_loc : 0.0;
        const double Kl_xy = p_safe ? (y*y - x*x) / p4_loc : 0.0;

        const double Kp_xx = p_safe ? -z * (2*x*x*z*z - 3*x*x*r2 + p2_loc*r2) / (p3_loc * r4) : 0.0;
        const double Kp_yy = p_safe ? -z * (2*y*y*z*z - 3*y*y*r2 + p2_loc*r2) / (p3_loc * r4) : 0.0;
        const double Kp_zz = -2.0 * z * p / r4;
        const double Kp_xy = p_safe ? x * y * z * (3*x*x + 3*y*y + z*z) / (p3_loc * r4) : 0.0;
        const double Kp_xz = p_safe ? x * (-x*x - y*y + z*z) / (p * r4) : 0.0;
        const double Kp_yz = p_safe ? y * (-x*x - y*y + z*z) / (p * r4) : 0.0;

        Eigen::Matrix3d K_combined;
        K_combined(0, 0) = dV_dr * Kr_xx + dV_dphi * Kp_xx + dV_dlam * Kl_xx;
        K_combined(1, 1) = dV_dr * Kr_yy + dV_dphi * Kp_yy + dV_dlam * Kl_yy;
        K_combined(2, 2) = dV_dr * Kr_zz + dV_dphi * Kp_zz;
        K_combined(0, 1) = dV_dr * Kr_xy + dV_dphi * Kp_xy + dV_dlam * Kl_xy;
        K_combined(0, 2) = dV_dr * Kr_xz + dV_dphi * Kp_xz;
        K_combined(1, 2) = dV_dr * Kr_yz + dV_dphi * Kp_yz;
        K_combined(1, 0) = K_combined(0, 1);
        K_combined(2, 0) = K_combined(0, 2);
        K_combined(2, 1) = K_combined(1, 2);

        H_cart += K_combined;

        // The exact Hessian is symmetric; average away round-off asymmetry
        result.hessian_ecef = 0.5 * (H_cart + H_cart.transpose());

        if (!result.acceleration.allFinite() || !result.hessian_ecef.allFinite())
            throw_not_finite(pos_ecef);

        return result;
    }

    /// True once init() (or a load) has sized the model.
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }
    /// Highest degree the model holds.
    [[nodiscard]] int max_degree() const noexcept { return max_degree_; }
    /// Gravitational parameter [m^3/s^2].
    [[nodiscard]] double GM() const noexcept { return GM_; }
    /// Reference radius [m].
    [[nodiscard]] double radius() const noexcept { return radius_; }

    /// The coefficient table itself (read-only).
    [[nodiscard]] const StoragePolicy& storage() const noexcept { return storage_; }

private:
    // eval_degree <= 0 means the whole loaded model. A degree above it raises
    // rather than quietly running the lower loaded degree.
    [[nodiscard]] int checked_degree(int eval_degree) const {
        if (eval_degree <= 0) return max_degree_;
        if (eval_degree > max_degree_) {
            throw ConfigError("gravity degree " + std::to_string(eval_degree) +
                " requested, but the loaded model stops at degree " +
                std::to_string(max_degree_) + ". Lower the degree, or load the "
                "NGA EGM2008_to2190 file (InitConfig.egm2008_path) for degrees above 360.");
        }
        return eval_degree;
    }

    // Non-finite gravity raises: replacing it by the point-mass term would
    // hide a non-finite state or a broken coefficient set.
    [[noreturn]] static void throw_not_finite(const Vector3d& pos_ecef) {
        throw Error("gravity is not finite at ECEF position (" +
            std::to_string(pos_ecef.x()) + ", " + std::to_string(pos_ecef.y()) + ", " +
            std::to_string(pos_ecef.z()) + ") m. The state has diverged or is not a "
            "number; check the initial state and the integrator step.");
    }

    bool initialized_ = false;
    double GM_ = constants::earth::GM;
    double radius_ = constants::earth::RADIUS_EQUATORIAL;
    int max_degree_ = 0;

    StoragePolicy storage_;
    std::vector<double> root_;  // root_[i] = sqrt(i)
};

/// Names for the two layouts (ClenshawGravity and Clenshaw are the older spellings).
using ClenshawTriangular = ClenshawEngine<TriangularStorage>;
using ClenshawColumnMajor = ClenshawEngine<ColumnMajorStorage>;
using ClenshawGravity = ClenshawEngine<TriangularStorage>;
using Clenshaw = ClenshawEngine<ColumnMajorStorage>;

} // namespace gravity
} // namespace ballistic
