#pragma once

// =============================================================================
// Storage Policies for Clenshaw Gravity Engine
// =============================================================================
//
// Two memory layouts for the spherical harmonic coefficients C_nm, S_nm
// (degree n, order m, 0 <= m <= n) used by ClenshawEngine (clenshaw_engine.hpp):
// - TriangularStorage: index(n,m) = n*(n+1)/2 + m  (the layout GeographicLib uses)
// - ColumnMajorStorage: offset[m] + (n-m)           (cache-friendly, faster)
// The engine's inner loop runs over n at fixed m, so keeping one order's
// degrees next to each other in memory is what makes the second layout faster.
//
// Each layout also stores the recurrence factors for every (n, m), which depend
// only on n and m and are therefore computed once when the table is sized.
// Both layouts are used only through ClenshawEngine; the members are public so
// the engine can reach them.
//
// =============================================================================

#include <vector>
#include <cstddef>

namespace ballistic {
namespace gravity {

/// Position-independent factors of the Clenshaw recurrence for one (n, m).
struct RecursionFactors {
    double w_sqrt2n3;   // w * sqrt(2n+3) - multiply by q to get Ax
    double b_factor;    // sqrt(2n+5) / (w * sqrt(n-m+2) * sqrt(n+m+2)) - multiply by -q^2 to get B
};

// =============================================================================
// Triangular Storage Policy
// =============================================================================
// index(n,m) = n*(n+1)/2 + m
// Total size: (N+1)*(N+2)/2

/// Coefficients stored degree by degree: (0,0), (1,0), (1,1), (2,0), ...
struct TriangularStorage {
    std::vector<double> Cnm;
    std::vector<double> Snm;
    std::vector<RecursionFactors> rf;

    /// Size for degrees 0..max_degree; all coefficients 0 except C_00 = 1.
    void allocate(int max_degree) {
        size_t size = static_cast<size_t>(max_degree + 1) * (max_degree + 2) / 2;
        Cnm.assign(size, 0.0);
        Snm.assign(size, 0.0);
        rf.resize(size);
        Cnm[0] = 1.0;  // C00 = 1.0
    }

    static size_t index(int n, int m) noexcept {
        return static_cast<size_t>(n) * (n + 1) / 2 + m;
    }

    /// Store (C, S) at (n, m); indices outside 0 <= m <= n <= max_degree are ignored.
    void set_coefficient(int n, int m, double C, double S, int max_degree) {
        if (n <= max_degree && m <= n && m >= 0) {
            size_t idx = index(n, m);
            Cnm[idx] = C;
            Snm[idx] = S;
        }
    }

    /// (C, S) at (n, m), or (0, 0) outside the table.
    std::pair<double, double> get_coefficient(int n, int m, int max_degree) const {
        if (n < 0 || m < 0 || m > n || n > max_degree) return {0.0, 0.0};
        size_t idx = index(n, m);
        return {Cnm[idx], Snm[idx]};
    }

    /// Fill `rf` for every (n, m); `root[i]` must hold sqrt(i) up to i = 2*max_degree + 5.
    void precompute_recursion_factors(int max_degree, const std::vector<double>& root) {
        for (int n = 0; n <= max_degree; ++n) {
            for (int m = 0; m <= n; ++m) {
                size_t idx = index(n, m);
                double w = root[2*n + 1] / (root[n - m + 1] * root[n + m + 1]);
                rf[idx].w_sqrt2n3 = w * root[2*n + 3];
                rf[idx].b_factor = root[2*n + 5] / (w * root[n - m + 2] * root[n + m + 2]);
            }
        }
    }

    // Unchecked accessors for the engine's inner loop (no bounds checks)
    double get_C(int n, int m) const { return Cnm[index(n, m)]; }
    double get_S(int n, int m) const { return Snm[index(n, m)]; }
    const RecursionFactors& get_rf(int n, int m) const { return rf[index(n, m)]; }
};

// =============================================================================
// Column-Major Storage Policy
// =============================================================================
// col_index(n,m) = col_offset[m] + (n-m)
// Coefficients stored by order m, contiguous by degree n — matches the inner loop

/// Coefficients stored order by order: (0,0), (1,0), ..., (N,0), (1,1), (2,1), ...
struct ColumnMajorStorage {
    std::vector<size_t> col_offset;
    std::vector<double> Cnm;
    std::vector<double> Snm;
    std::vector<RecursionFactors> rf;

    /// Size for degrees 0..max_degree; all coefficients 0 except C_00 = 1.
    void allocate(int max_degree) {
        col_offset.resize(max_degree + 2);
        size_t total = 0;
        for (int m = 0; m <= max_degree; ++m) {
            col_offset[m] = total;
            total += (max_degree - m + 1);
        }
        col_offset[max_degree + 1] = total;

        Cnm.assign(total, 0.0);
        Snm.assign(total, 0.0);
        rf.resize(total);
        Cnm[col_offset[0]] = 1.0;  // C00 = 1.0
    }

    size_t index(int n, int m) const noexcept {
        return col_offset[m] + (n - m);
    }

    /// Store (C, S) at (n, m); indices outside 0 <= m <= n <= max_degree are ignored.
    void set_coefficient(int n, int m, double C, double S, int max_degree) {
        if (n <= max_degree && m <= n && m >= 0) {
            size_t idx = index(n, m);
            Cnm[idx] = C;
            Snm[idx] = S;
        }
    }

    /// (C, S) at (n, m), or (0, 0) outside the table.
    std::pair<double, double> get_coefficient(int n, int m, int max_degree) const {
        if (n < 0 || m < 0 || m > n || n > max_degree) return {0.0, 0.0};
        size_t idx = index(n, m);
        return {Cnm[idx], Snm[idx]};
    }

    /// Fill `rf` for every (n, m); `root[i]` must hold sqrt(i) up to i = 2*max_degree + 5.
    void precompute_recursion_factors(int max_degree, const std::vector<double>& root) {
        for (int m = 0; m <= max_degree; ++m) {
            for (int n = m; n <= max_degree; ++n) {
                size_t idx = index(n, m);
                double w = root[2*n + 1] / (root[n - m + 1] * root[n + m + 1]);
                rf[idx].w_sqrt2n3 = w * root[2*n + 3];
                rf[idx].b_factor = root[2*n + 5] / (w * root[n - m + 2] * root[n + m + 2]);
            }
        }
    }

    // Unchecked accessors for the engine's inner loop (no bounds checks)
    double get_C(int n, int m) const { return Cnm[index(n, m)]; }
    double get_S(int n, int m) const { return Snm[index(n, m)]; }
    const RecursionFactors& get_rf(int n, int m) const { return rf[index(n, m)]; }
};

} // namespace gravity
} // namespace ballistic
