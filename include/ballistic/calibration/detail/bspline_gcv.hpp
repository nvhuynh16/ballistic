#pragma once

// =============================================================================
// Automatic choice of the smoothing weight λ by GCV
// =============================================================================
//
// A smoothing B-spline is fitted to 3-D residuals dp(τ) by minimizing
//   ||B c − dp||² + λ · s · cᵀ R c
// where B holds the basis functions at the sample times, R penalizes the k-th time
// derivative of the spline (roughness), and s = N·h^(2k+1) (N samples, knot spacing h) is a
// scale that makes λ mean the same thing at any sample count and knot spacing. Larger λ gives
// a smoother curve.
//
// Generalized cross-validation (GCV) picks λ from the data alone, without a noise estimate:
//   GCV(λ) = RSS/(3N) / (1 − tr(A_λ)/N)²
// with RSS the residual sum of squares over the three components and A_λ the "hat" matrix
// mapping data to fitted values (tr(A_λ) = effective number of parameters). GCV tends to
// under-smooth. The search is a log-spaced grid followed by golden-section refinement.
// (Only GCV is implemented here; there is no REML criterion.)
// Dense solve: one n_basis × n_basis inverse per λ evaluation.


//
// =============================================================================

#include "bspline_eval.hpp"
#include "../../math/bspline_basis.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>

namespace ballistic {
namespace calibration {
namespace detail {

/// How the B-spline smoothing weight λ is chosen: held fixed, chosen by GCV, or decreased
/// over iterations.
enum class LambdaMethod { FIXED, GCV, DECREASING };

/// The pieces of the smoothing-spline normal equations that do not depend on λ, assembled
/// once so many λ values can be scored cheaply. The system solved is (G + λ·reg_scale·R)c = g.
struct BSplineMatrices {
    Eigen::SparseMatrix<double> G;          // B^T B (Gram matrix)
    Eigen::SparseMatrix<double> R;          // D_k^T D_k (penalty matrix, unscaled by lambda)
    Eigen::VectorXd gx, gy, gz;            // B^T dp
    double dpx_sq = 0, dpy_sq = 0, dpz_sq = 0;  // ||dp||² per component
    double reg_scale = 0;                   // N × h^(2k+1): makes λ independent of N and knot spacing h
    int n_basis = 0;
    int N = 0;
    std::vector<double> knots;
    double t_start = 0, t_end = 0;
    int degree = 3;
};

/// GCV score of one λ value and the quantities behind it.
struct LambdaResult {
    double score = 1e30;        // GCV score (lower = better)
    double rss = 0;             // Residual sum of squares (all 3 components)
    double trace_ratio = 0;     // tr(A_λ) / N
    double lambda = 0;
    double sigma_sq = -1.0;     // Estimated noise variance per component: (rss/3) / (N - tr(A))
};

/// Assemble the λ-independent matrices for residuals `dp` at times `tau` [s, ascending] with
/// interior knots every `knot_spacing_s` [s], a spline of `degree`, and a roughness penalty on
/// the `reg_deriv_order`-th derivative (evaluated at the Greville points, the natural "center"
/// of each basis function, and halfway between them). No penalty is built when the order is
/// outside 1..degree. No λ is applied and nothing is solved.
[[nodiscard]] inline BSplineMatrices assemble_bspline_matrices(
    const std::vector<double>& tau,
    const std::vector<Vector3d>& dp,
    double knot_spacing_s,
    int reg_deriv_order,
    int degree)
{
    BSplineMatrices m;
    m.N = static_cast<int>(tau.size());
    m.t_start = tau.front();
    m.t_end = tau.back();
    m.degree = degree;
    const double arc = m.t_end - m.t_start;

    int n_interior = static_cast<int>(std::floor(arc / knot_spacing_s)) - 1;
    if (n_interior < 0) n_interior = 0;
    m.knots = math::make_clamped_knot_vector(m.t_start, m.t_end, n_interior, degree);
    m.n_basis = n_interior + degree + 1;
    const int n_idx = m.n_basis - 1;

    // Gram matrix G = B^T B and RHS g = B^T dp
    std::vector<Eigen::Triplet<double>> G_trips, R_trips;
    G_trips.reserve(m.N * (degree + 1) * (degree + 1));
    m.gx = Eigen::VectorXd::Zero(m.n_basis);
    m.gy = Eigen::VectorXd::Zero(m.n_basis);
    m.gz = Eigen::VectorXd::Zero(m.n_basis);

    std::vector<double> Ni(degree + 1);
    for (int i = 0; i < m.N; ++i) {
        int span = math::bspline_findspan(n_idx, degree, tau[i], m.knots.data());
        math::bspline_basis_funs(degree, span, tau[i], m.knots.data(), Ni.data());
        int col0 = span - degree;
        m.dpx_sq += dp[i].x() * dp[i].x();
        m.dpy_sq += dp[i].y() * dp[i].y();
        m.dpz_sq += dp[i].z() * dp[i].z();
        for (int a = 0; a <= degree; ++a) {
            int ca = col0 + a;
            m.gx(ca) += Ni[a] * dp[i].x();
            m.gy(ca) += Ni[a] * dp[i].y();
            m.gz(ca) += Ni[a] * dp[i].z();
            for (int b = a; b <= degree; ++b) {
                int cb = col0 + b;
                double v = Ni[a] * Ni[b];
                G_trips.emplace_back(ca, cb, v);
                if (ca != cb) G_trips.emplace_back(cb, ca, v);
            }
        }
    }
    m.G.resize(m.n_basis, m.n_basis);
    m.G.setFromTriplets(G_trips.begin(), G_trips.end());

    // Regularization penalty matrix R (unscaled — no lambda, no reg_scale)
    const double h_pow = std::pow(knot_spacing_s, 2 * reg_deriv_order + 1);
    m.reg_scale = static_cast<double>(m.N) * h_pow;

    if (reg_deriv_order >= 1 && reg_deriv_order <= degree) {
        std::vector<double> greville(m.n_basis);
        for (int j = 0; j < m.n_basis; ++j) {
            double sum = 0.0;
            for (int k = 1; k <= degree; ++k) sum += m.knots[j + k];
            greville[j] = sum / degree;
        }
        std::vector<double> reg_pts;
        reg_pts.reserve(2 * m.n_basis);
        for (int j = 0; j < m.n_basis; ++j) reg_pts.push_back(greville[j]);
        for (int j = 0; j < m.n_basis - 1; ++j)
            reg_pts.push_back(0.5 * (greville[j] + greville[j + 1]));

        std::vector<double> ders((degree + 1) * (reg_deriv_order + 1));
        for (double t : reg_pts) {
            double tc = std::clamp(t, m.t_start, m.t_end);
            int span = math::bspline_findspan(n_idx, degree, tc, m.knots.data());
            math::bspline_ders(degree, span, tc, m.knots.data(), reg_deriv_order, ders.data());
            int col0 = span - degree;
            const double* drow = &ders[reg_deriv_order * (degree + 1)];
            for (int a = 0; a <= degree; ++a) {
                int ca = col0 + a;
                for (int b = a; b <= degree; ++b) {
                    int cb = col0 + b;
                    double v = drow[a] * drow[b];
                    R_trips.emplace_back(ca, cb, v);
                    if (ca != cb) R_trips.emplace_back(cb, ca, v);
                }
            }
        }
        m.R.resize(m.n_basis, m.n_basis);
        m.R.setFromTriplets(R_trips.begin(), R_trips.end());
    }

    return m;
}

// =========================================================================
// Shared internals of the GCV evaluation
// =========================================================================

/// RSS and tr(A_λ) for one λ; `ok` is false if the system matrix could not be factorised.
struct LambdaEvalInternals {
    double rss = 0;
    double trace_A = 0;
    bool ok = false;
};

/// Solve (G + λ·reg_scale·R)c = g for each component and return the residual sum of squares
/// and the trace of the hat matrix, tr(H⁻¹G).
[[nodiscard]] inline LambdaEvalInternals eval_lambda_internals(
    const BSplineMatrices& mats, double lambda)
{
    const int M = mats.n_basis;

    Eigen::MatrixXd G_d = Eigen::MatrixXd(mats.G);
    Eigen::MatrixXd R_d;
    Eigen::MatrixXd H_d = G_d;
    if (lambda > 0.0 && mats.R.nonZeros() > 0) {
        R_d = Eigen::MatrixXd(mats.R);
        H_d += (lambda * mats.reg_scale) * R_d;
    }

    Eigen::LDLT<Eigen::MatrixXd> ldlt(H_d);
    if (ldlt.info() != Eigen::Success) return {};

    Eigen::VectorXd cx = ldlt.solve(mats.gx);
    Eigen::VectorXd cy = ldlt.solve(mats.gy);
    Eigen::VectorXd cz = ldlt.solve(mats.gz);

    // RSS per component: ||dp||² - c^T g - λ·s·c^T R c
    auto rss_comp = [&](const Eigen::VectorXd& c, const Eigen::VectorXd& g, double dp_sq) {
        double cg = c.dot(g);
        double cRc = (R_d.size() > 0) ? (R_d * c).dot(c) * lambda * mats.reg_scale : 0.0;
        return (std::max)(0.0, dp_sq - cg - cRc);
    };
    double rss = rss_comp(cx, mats.gx, mats.dpx_sq)
               + rss_comp(cy, mats.gy, mats.dpy_sq)
               + rss_comp(cz, mats.gz, mats.dpz_sq);

    // tr(A_λ) = tr(H^{-1} G)
    Eigen::MatrixXd H_inv = ldlt.solve(Eigen::MatrixXd::Identity(M, M));
    double trace_A = (H_inv.array() * G_d.array()).sum();

    return {rss, trace_A, true};
}

// =========================================================================
// GCV
// =========================================================================

/// GCV score for one λ (score 1e30 if the solve fails), plus the per-component noise-variance
/// estimate RSS/3 / (N − tr(A_λ)) (−1 when fewer than ~1 degree of freedom remain).
[[nodiscard]] inline LambdaResult compute_gcv(const BSplineMatrices& mats, double lambda) {
    auto e = eval_lambda_internals(mats, lambda);
    if (!e.ok) return {1e30, 0, 0, lambda};

    double trace_ratio = e.trace_A / mats.N;
    double denom = 1.0 - trace_ratio;
    if (denom < 1e-10) denom = 1e-10;
    double gcv = (e.rss / (3.0 * mats.N)) / (denom * denom);

    // σ² estimate: per-component noise variance (Wahba 1990 §4.3)
    double edf = mats.N - e.trace_A;  // effective degrees of freedom
    double sigma_sq = (edf > 1.0) ? (e.rss / 3.0) / edf : -1.0;

    return {gcv, e.rss, trace_ratio, lambda, sigma_sq};
}

/// Choose λ by minimizing GCV over `n_grid` log-spaced values in [lam_min, lam_max], then five
/// golden-section steps around the best grid point. When there are too few basis functions
/// (< 2·(degree+1)) to choose meaningfully, returns 1e-6 kept inside [lam_min, lam_max].
/// Prints the chosen λ to stdout.
[[nodiscard]] inline double select_lambda_gcv(
    const BSplineMatrices& mats,
    double lam_min = 1e-12,
    double lam_max = 1e-1,
    int n_grid = 15)
{
    if (mats.n_basis < 2 * (mats.degree + 1)) return std::clamp(1e-6, lam_min, lam_max);

    double log_min = std::log(lam_min), log_max = std::log(lam_max);
    double log_step = (log_max - log_min) / (n_grid - 1);

    std::vector<LambdaResult> results(n_grid);
    int best_idx = 0;
    for (int i = 0; i < n_grid; ++i) {
        double lam = std::exp(log_min + i * log_step);
        results[i] = compute_gcv(mats, lam);
        if (results[i].score < results[best_idx].score) best_idx = i;
    }

    double lo = (best_idx > 0) ? std::log(results[best_idx - 1].lambda) : log_min;
    double hi = (best_idx < n_grid - 1) ? std::log(results[best_idx + 1].lambda) : log_max;

    const double phi = 0.381966011250105;
    for (int i = 0; i < 5; ++i) {
        double m1 = lo + phi * (hi - lo);
        double m2 = hi - phi * (hi - lo);
        if (compute_gcv(mats, std::exp(m1)).score < compute_gcv(mats, std::exp(m2)).score)
            hi = m2;
        else lo = m1;
    }

    double best_lambda = std::exp(0.5 * (lo + hi));
    auto best = compute_gcv(mats, best_lambda);

    std::cout << "  [bspline-gcv] lambda=" << std::scientific << std::setprecision(2)
              << best_lambda << " gcv=" << best.score
              << " tr/N=" << std::fixed << std::setprecision(4) << best.trace_ratio
              << std::endl;

    return best_lambda;
}

} // namespace detail
} // namespace calibration
} // namespace ballistic
