#pragma once

// =============================================================================
// Horizontal Wind Model 2014 (HWM14) - C++ Port
// =============================================================================
//
// Complete C++ port of the HWM14 Fortran model (Drob et al., 2015).
// Provides thermospheric and mesospheric horizontal wind velocities (0-450+ km).
//
// Two-component model:
//   1. Quiet-time winds: VSH expansion with B-spline vertical structure
//   2. Disturbance winds (DWM07): Kp-dependent, quasi-dipole coordinates
//
// Original Fortran: Douglas Drob, John Emmert (NRL)
// Fortran source: NRL, https://map.nrl.navy.mil/map/pub/nrl/HWM/HWM14/
// (mirror used for the translation: https://github.com/jacobwilliams/HWM14)
//
// Provenance/terms: HWM14 was written by NRL employees as a work of the U.S.
// Government (not subject to copyright in the United States, 17 U.S.C. 105);
// NRL attached no license to the package. Cite Drob et al. (2015),
// doi:10.1002/2014EA000089. See THIRD_PARTY_NOTICES.md. Change notice: translated from hwm14.f90 to C++ on
// 2026-02-23; model formulation and coefficients unchanged.
//
// References:
// -----------
// [1] Drob, D.P., et al. (2015). "An update to the Horizontal Wind Model
//     (HWM): The quiet time thermosphere". Earth and Space Science, 2, 301-319.
// [2] Emmert, J.T., et al. (2008). "DWM07 global climatological model of
//     upper thermospheric storm-induced disturbance winds". JGR, 113, A11319.
// [3] Richmond, A.D. (1995). "Ionospheric electrodynamics using magnetic
//     apex coordinates". J. Geomag. Geoelec., 47, 191-212.
//
// =============================================================================

#include "../../core/types.hpp"
#include "../../time/epoch.hpp"
#include <array>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

// Embedded coefficient data
#include "hwm14_coefficients.hpp"

namespace ballistic {
namespace atmosphere {

// Wind output: meridional (northward, +N) and zonal (eastward, +E) in m/s
struct WindComponents {
    double meridional = 0.0;  // Northward (m/s)
    double zonal = 0.0;       // Eastward (m/s)
};

// =============================================================================
// Associated Legendre Function (ALF) Basis
// =============================================================================
// Computes vector spherical harmonic basis functions P, V, W
// Port of Fortran alf module

class ALFBasis {
public:
    void init(int nmax_in, int mmax_in) {
        nmax_ = nmax_in;
        mmax_ = mmax_in;

        anm_.assign((nmax_ + 1) * (mmax_ + 1), 0.0);
        bnm_.assign((nmax_ + 1) * (mmax_ + 1), 0.0);
        dnm_.assign((nmax_ + 1) * (mmax_ + 1), 0.0);
        cm_.assign(mmax_ + 1, 0.0);
        en_.assign(nmax_ + 1, 0.0);
        marr_.assign(mmax_ + 1, 0.0);
        narr_.assign(nmax_ + 1, 0.0);

        for (int n = 1; n <= nmax_; ++n) {
            narr_[n] = static_cast<double>(n);
            en_[n] = std::sqrt(static_cast<double>(n * (n + 1)));
            anm_[idx(n, 0)] = std::sqrt(static_cast<double>((2 * n - 1) * (2 * n + 1))) / narr_[n];
            bnm_[idx(n, 0)] = std::sqrt(static_cast<double>((2 * n + 1) * (n - 1) * (n - 1))
                                         / static_cast<double>(2 * n - 3)) / narr_[n];
        }
        for (int m = 1; m <= mmax_; ++m) {
            marr_[m] = static_cast<double>(m);
            cm_[m] = std::sqrt(static_cast<double>(2 * m + 1)
                               / static_cast<double>(2 * m * m * (m + 1)));
            for (int n = m + 1; n <= nmax_; ++n) {
                int64_t n64 = n, m64 = m;
                anm_[idx(n, m)] = std::sqrt(
                    static_cast<double>((2*n64-1)*(2*n64+1)*(n64-1))
                    / static_cast<double>((n64-m64)*(n64+m64)*(n64+1)));
                bnm_[idx(n, m)] = std::sqrt(
                    static_cast<double>((2*n64+1)*(n64+m64-1)*(n64-m64-1)*(n64-2)*(n64-1))
                    / static_cast<double>((n64-m64)*(n64+m64)*(2*n64-3)*n64*(n64+1)));
                dnm_[idx(n, m)] = std::sqrt(
                    static_cast<double>((n64-m64)*(n64+m64)*(2*n64+1)*(n64-1))
                    / static_cast<double>((2*n64-1)*(n64+1)));
            }
        }
    }

    // Compute P, V, W basis functions at colatitude theta (radians)
    void compute(int nmax, int mmax, double theta,
                 std::vector<double>& P, std::vector<double>& V, std::vector<double>& W) const
    {
        static constexpr double p00 = 0.70710678118654746;
        double x = std::cos(theta);
        double y = std::sin(theta);

        P[idx(0, 0)] = p00;

        for (int m = 1; m <= mmax; ++m) {
            W[idx(m, m)] = cm_[m] * P[idx(m - 1, m - 1)];
            P[idx(m, m)] = y * en_[m] * W[idx(m, m)];
            for (int n = m + 1; n <= nmax; ++n) {
                W[idx(n, m)] = anm_[idx(n, m)] * x * W[idx(n - 1, m)] - bnm_[idx(n, m)] * W[idx(n - 2, m)];
                P[idx(n, m)] = y * en_[n] * W[idx(n, m)];
                V[idx(n, m)] = narr_[n] * x * W[idx(n, m)] - dnm_[idx(n, m)] * W[idx(n - 1, m)];
                W[idx(n - 2, m)] = marr_[m] * W[idx(n - 2, m)];
            }
            W[idx(nmax - 1, m)] = marr_[m] * W[idx(nmax - 1, m)];
            W[idx(nmax, m)] = marr_[m] * W[idx(nmax, m)];
            V[idx(m, m)] = x * W[idx(m, m)];
        }
        P[idx(1, 0)] = anm_[idx(1, 0)] * x * P[idx(0, 0)];
        V[idx(1, 0)] = -P[idx(1, 1)];
        for (int n = 2; n <= nmax; ++n) {
            P[idx(n, 0)] = anm_[idx(n, 0)] * x * P[idx(n - 1, 0)] - bnm_[idx(n, 0)] * P[idx(n - 2, 0)];
            V[idx(n, 0)] = -P[idx(n, 1)];
        }
    }

    [[nodiscard]] int idx(int n, int m) const noexcept {
        return n * (mmax_ + 1) + m;
    }

    [[nodiscard]] int nmax() const noexcept { return nmax_; }
    [[nodiscard]] int mmax() const noexcept { return mmax_; }
    [[nodiscard]] int array_size() const noexcept { return (nmax_ + 1) * (mmax_ + 1); }

private:
    int nmax_ = 0, mmax_ = 0;
    std::vector<double> anm_, bnm_, dnm_, cm_, en_, marr_, narr_;
};

// =============================================================================
// Quiet-Time Wind Model (QWM)
// =============================================================================

class QWM {
public:
    void init() {
        if (initialized_) return;

        // Load coefficients from embedded data
        load_coefficients();

        initialized_ = true;
    }

    // Compute quiet-time winds
    // Input: day of year, UT seconds, altitude km, glat deg, glon deg
    // Output: w[0]=meridional, w[1]=zonal
    void compute(int day_of_year, double ut_sec, double alt_km,
                 double glat_deg, double glon_deg,
                 ALFBasis& alf,
                 std::vector<double>& gpbar, std::vector<double>& gvbar, std::vector<double>& gwbar,
                 double& glat_cached,
                 double w[2]) const
    {
        static constexpr double TWO_PI = 2.0 * 3.14159265358979323846;
        static constexpr double DEG2RAD = TWO_PI / 360.0;

        double input[5];
        input[0] = static_cast<double>(day_of_year);
        input[1] = ut_sec;
        input[2] = glon_deg;
        input[3] = glat_deg;
        input[4] = alt_km;

        bool refresh[5] = {false, false, false, false, false};

        // Seasonal variations
        if (input[0] != previous_[0]) {
            double AA = input[0] * TWO_PI / 365.25;
            for (int s = 0; s <= maxs_; ++s) {
                double BB = static_cast<double>(s) * AA;
                fs_[s * 2] = std::cos(BB);
                fs_[s * 2 + 1] = std::sin(BB);
            }
            for (int i = 0; i < 5; ++i) refresh[i] = true;
            previous_[0] = input[0];
        }

        // Tidal variations (depends on UT and longitude)
        if (input[1] != previous_[1] || input[2] != previous_[2]) {
            double AA = std::fmod(input[1] / 3600.0 + input[2] / 15.0 + 48.0, 24.0);
            double BB = AA * TWO_PI / 24.0;
            for (int l = 0; l <= maxl_; ++l) {
                double CC = static_cast<double>(l) * BB;
                fl_[l * 2] = std::cos(CC);
                fl_[l * 2 + 1] = std::sin(CC);
            }
            refresh[2] = true;
            previous_[1] = input[1];
        }

        // Stationary planetary waves (depends on longitude)
        if (input[2] != previous_[2]) {
            double AA = input[2] * DEG2RAD;
            for (int m = 0; m <= maxm_; ++m) {
                double BB = static_cast<double>(m) * AA;
                fm_[m * 2] = std::cos(BB);
                fm_[m * 2 + 1] = std::sin(BB);
            }
            refresh[1] = true;
            previous_[2] = input[2];
        }

        // Latitude (ALF basis)
        double theta = (90.0 - input[3]) * DEG2RAD;
        if (input[3] != glat_cached) {
            double AA = (90.0 - input[3]) * DEG2RAD;
            alf.compute(maxn_, maxo_, AA, gpbar, gvbar, gwbar);
            for (int i = 0; i < 4; ++i) refresh[i] = true;
            glat_cached = input[3];
            previous_[3] = input[3];
        }

        // Altitude (B-spline weights)
        if (input[4] != previous_[4]) {
            vertwght(input[4], zwght_, lev_);
            previous_[4] = input[4];
        }

        // Calculate VSH functions and accumulate winds
        double u = 0.0, v = 0.0;

        for (int b = 0; b <= p_; ++b) {
            if (zwght_[b] == 0.0) continue;

            int d = b + lev_;

            if (priornb_ != nb_[d]) {
                for (int i = 0; i < 5; ++i) refresh[i] = true;
            }
            priornb_ = nb_[d];

            bool any_refresh = false;
            for (int i = 0; i < 5; ++i) if (refresh[i]) any_refresh = true;

            if (!any_refresh) {
                int c = nb_[d];
                u += zwght_[b] * dot_product(bz_.data(), mparm_.data() + d * nbf_, c);
                v += zwght_[b] * dot_product(bz_.data(), tparm_.data() + d * nbf_, c);
                continue;
            }

            int amaxs = order_[d * ncomp_ + 0];
            int amaxn = order_[d * ncomp_ + 1];
            int pmaxm = order_[d * ncomp_ + 2];
            int pmaxs = order_[d * ncomp_ + 3];
            int pmaxn = order_[d * ncomp_ + 4];
            int tmaxl = order_[d * ncomp_ + 5];
            int tmaxs = order_[d * ncomp_ + 6];
            int tmaxn = order_[d * ncomp_ + 7];

            int c = 0;

            // Seasonal - Zonal average (m = 0)
            if (refresh[0]) {
                for (int n = 1; n <= amaxn; ++n) {
                    bz_[c] = -std::sin(n * theta);
                    bz_[c + 1] = std::sin(n * theta);
                    c += 2;
                }
                for (int s = 1; s <= amaxs; ++s) {
                    double cs_val = fs_[s * 2];
                    double ss_val = fs_[s * 2 + 1];
                    for (int n = 1; n <= amaxn; ++n) {
                        double sc = std::sin(n * theta);
                        bz_[c] = -sc * cs_val;
                        bz_[c + 1] = sc * ss_val;
                        bz_[c + 2] = sc * cs_val;
                        bz_[c + 3] = -sc * ss_val;
                        c += 4;
                    }
                }
                cseason_ = c;
            } else {
                c = cseason_;
            }

            // Stationary planetary waves
            if (refresh[1]) {
                for (int m = 1; m <= pmaxm; ++m) {
                    double cm_val = fm_[m * 2];
                    double sm_val = fm_[m * 2 + 1];
                    for (int n = m; n <= pmaxn; ++n) {
                        double vb = gvbar[alf.idx(n, m)];
                        double wb = gwbar[alf.idx(n, m)];
                        bz_[c] = -vb * cm_val;
                        bz_[c + 1] = vb * sm_val;
                        bz_[c + 2] = -wb * sm_val;
                        bz_[c + 3] = -wb * cm_val;
                        c += 4;
                    }
                    for (int s = 1; s <= pmaxs; ++s) {
                        double cs = fs_[s * 2];
                        double ss = fs_[s * 2 + 1];
                        for (int n = m; n <= pmaxn; ++n) {
                            double vb = gvbar[alf.idx(n, m)];
                            double wb = gwbar[alf.idx(n, m)];
                            bz_[c] = -vb * cm_val * cs;
                            bz_[c + 1] = vb * sm_val * cs;
                            bz_[c + 2] = -wb * sm_val * cs;
                            bz_[c + 3] = -wb * cm_val * cs;
                            bz_[c + 4] = -vb * cm_val * ss;
                            bz_[c + 5] = vb * sm_val * ss;
                            bz_[c + 6] = -wb * sm_val * ss;
                            bz_[c + 7] = -wb * cm_val * ss;
                            c += 8;
                        }
                    }
                }
                cwave_ = c;
            } else {
                c = cwave_;
            }

            // Migrating Solar Tides
            if (refresh[2]) {
                for (int l = 1; l <= tmaxl; ++l) {
                    double cl = fl_[l * 2];
                    double sl = fl_[l * 2 + 1];
                    for (int n = l; n <= tmaxn; ++n) {
                        double vb = gvbar[alf.idx(n, l)];
                        double wb = gwbar[alf.idx(n, l)];
                        bz_[c] = -vb * cl;
                        bz_[c + 1] = vb * sl;
                        bz_[c + 2] = -wb * sl;
                        bz_[c + 3] = -wb * cl;
                        c += 4;
                    }
                    for (int s = 1; s <= tmaxs; ++s) {
                        double cs = fs_[s * 2];
                        double ss = fs_[s * 2 + 1];
                        for (int n = l; n <= tmaxn; ++n) {
                            double vb = gvbar[alf.idx(n, l)];
                            double wb = gwbar[alf.idx(n, l)];
                            bz_[c] = -vb * cl * cs;
                            bz_[c + 1] = vb * sl * cs;
                            bz_[c + 2] = -wb * sl * cs;
                            bz_[c + 3] = -wb * cl * cs;
                            bz_[c + 4] = -vb * cl * ss;
                            bz_[c + 5] = vb * sl * ss;
                            bz_[c + 6] = -wb * sl * ss;
                            bz_[c + 7] = -wb * cl * ss;
                            c += 8;
                        }
                    }
                }
                ctide_ = c;
            } else {
                c = ctide_;
            }

            // Calculate wind components
            u += zwght_[b] * dot_product(bz_.data(), mparm_.data() + d * nbf_, c);
            v += zwght_[b] * dot_product(bz_.data(), tparm_.data() + d * nbf_, c);
        }

        w[0] = v;  // meridional
        w[1] = u;  // zonal
    }

    [[nodiscard]] int maxn() const noexcept { return maxn_; }
    [[nodiscard]] int maxo() const noexcept { return maxo_; }

private:
    bool initialized_ = false;

    int nbf_ = 0;
    int ncomp_ = 0;
    int maxn_ = 0, maxs_ = 0, maxm_ = 0, maxl_ = 0, maxo_ = 0;
    int p_ = 0, nlev_ = 0, nnode_ = 0;
    double alttns_ = 0.0, altsym_ = 0.0, altiso_ = 0.0;
    static constexpr double H_ = 60.0;  // Scale height (km)

    std::vector<int> nb_;         // basis function count per level
    std::vector<int> order_;      // spectral content per level (8 ints per level)
    std::vector<double> vnode_;   // vertical altitude nodes
    std::vector<double> mparm_;   // model parameters (nbf x nlev, flattened)
    std::vector<double> tparm_;   // parity-transformed parameters
    std::array<double, 5> e1_{};
    std::array<double, 5> e2_{};

    // Cached state (mutable for const compute)
    mutable std::array<double, 5> previous_ = {-1.0e32, -1.0e32, -1.0e32, -1.0e32, -1.0e32};
    mutable int priornb_ = 0;
    mutable int cseason_ = 0, cwave_ = 0, ctide_ = 0;
    mutable std::vector<double> fs_, fm_, fl_;  // Fourier terms
    mutable std::vector<double> bz_;            // basis function values
    mutable std::array<double, 4> zwght_{};
    mutable int lev_ = 0;

    static double dot_product(const double* a, const double* b, int n) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }

    void load_coefficients() {
        const auto& data = hwm14::qwm_data;

        nbf_ = data.nbf;
        ncomp_ = data.ncomp;
        maxs_ = data.maxs;
        maxm_ = data.maxm;
        maxl_ = data.maxl;
        maxn_ = data.maxn;
        p_ = data.p;
        nlev_ = data.nlev;
        nnode_ = nlev_ + p_;

        // Copy vnode
        vnode_.assign(data.vnode.begin(), data.vnode.begin() + nnode_ + 1);
        vnode_[3] = 0.0;  // Match Fortran: vnode(3) = 0.0

        // Copy model parameters and order arrays
        int nlevels = nlev_ - p_ + 1 - 2 + 1;  // number of levels with data
        nb_.resize(nnode_ + 1, 0);
        order_.resize((nnode_ + 1) * ncomp_, 0);
        mparm_.resize(nbf_ * (nlev_ + 1), 0.0);
        tparm_.resize(nbf_ * (nlev_ + 1), 0.0);

        for (int i = 0; i < nlevels; ++i) {
            for (int j = 0; j < ncomp_; ++j) {
                order_[i * ncomp_ + j] = data.order[i * ncomp_ + j];
            }
            nb_[i] = data.nb[i];
            for (int j = 0; j < nbf_; ++j) {
                mparm_[i * nbf_ + j] = data.mparm[i * nbf_ + j];
            }
        }

        // Compute parity permutations
        for (int i = 0; i < nlevels; ++i) {
            compute_parity(i);
        }

        // Transition levels
        alttns_ = vnode_[nlev_ - 2];
        altsym_ = vnode_[nlev_ - 1];
        altiso_ = vnode_[nlev_];

        // Upper-atmosphere basis vectors
        e1_ = data.e1;
        e2_ = data.e2;

        // Set maxo
        maxo_ = (std::max)({maxs_, maxm_, maxl_});

        // Allocate workspace
        fs_.resize((maxs_ + 1) * 2, 0.0);
        fm_.resize((maxm_ + 1) * 2, 0.0);
        fl_.resize((maxl_ + 1) * 2, 0.0);
        bz_.resize(nbf_, 0.0);
    }

    void compute_parity(int level) {
        const int* ord = &order_[level * ncomp_];
        int amaxs = ord[0], amaxn = ord[1];
        int pmaxm = ord[2], pmaxs = ord[3], pmaxn = ord[4];
        int tmaxl = ord[5], tmaxs = ord[6], tmaxn = ord[7];

        double* mp = mparm_.data() + level * nbf_;
        double* tp = tparm_.data() + level * nbf_;

        int c = 0;

        // Zonal average parity
        for (int n = 1; n <= amaxn; ++n) {
            tp[c] = 0.0;
            tp[c + 1] = -mp[c + 1];
            mp[c + 1] = 0.0;
            c += 2;
        }
        for (int s = 1; s <= amaxs; ++s) {
            for (int n = 1; n <= amaxn; ++n) {
                tp[c] = 0.0;
                tp[c + 1] = 0.0;
                tp[c + 2] = -mp[c + 2];
                tp[c + 3] = -mp[c + 3];
                mp[c + 2] = 0.0;
                mp[c + 3] = 0.0;
                c += 4;
            }
        }

        // Stationary planetary wave parity
        for (int m = 1; m <= pmaxm; ++m) {
            for (int n = m; n <= pmaxn; ++n) {
                tp[c] = mp[c + 2];
                tp[c + 1] = mp[c + 3];
                tp[c + 2] = -mp[c];
                tp[c + 3] = -mp[c + 1];
                c += 4;
            }
            for (int s = 1; s <= pmaxs; ++s) {
                for (int n = m; n <= pmaxn; ++n) {
                    tp[c] = mp[c + 2];
                    tp[c + 1] = mp[c + 3];
                    tp[c + 2] = -mp[c];
                    tp[c + 3] = -mp[c + 1];
                    tp[c + 4] = mp[c + 6];
                    tp[c + 5] = mp[c + 7];
                    tp[c + 6] = -mp[c + 4];
                    tp[c + 7] = -mp[c + 5];
                    c += 8;
                }
            }
        }

        // Migrating solar tide parity
        for (int l = 1; l <= tmaxl; ++l) {
            for (int n = l; n <= tmaxn; ++n) {
                tp[c] = mp[c + 2];
                tp[c + 1] = mp[c + 3];
                tp[c + 2] = -mp[c];
                tp[c + 3] = -mp[c + 1];
                c += 4;
            }
            for (int s = 1; s <= tmaxs; ++s) {
                for (int n = l; n <= tmaxn; ++n) {
                    tp[c] = mp[c + 2];
                    tp[c + 1] = mp[c + 3];
                    tp[c + 2] = -mp[c];
                    tp[c + 3] = -mp[c + 1];
                    tp[c + 4] = mp[c + 6];
                    tp[c + 5] = mp[c + 7];
                    tp[c + 6] = -mp[c + 4];
                    tp[c + 7] = -mp[c + 5];
                    c += 8;
                }
            }
        }
    }

    // B-spline vertical weighting
    void vertwght(double alt, std::array<double, 4>& wght, int& iz) const {
        iz = findspan(nnode_ - p_ - 1, p_, alt, vnode_.data()) - p_;
        iz = (std::min)(iz, 26);

        wght[0] = bspline(p_, nnode_, vnode_.data(), iz, alt);
        wght[1] = bspline(p_, nnode_, vnode_.data(), iz + 1, alt);

        if (iz <= 25) {
            wght[2] = bspline(p_, nnode_, vnode_.data(), iz + 2, alt);
            wght[3] = bspline(p_, nnode_, vnode_.data(), iz + 3, alt);
            return;
        }

        std::array<double, 5> we{};
        if (alt > alttns_) {
            we[3] = std::exp(-(alt - alttns_) / H_);
            we[4] = 1.0;
        } else {
            we[0] = bspline(p_, nnode_, vnode_.data(), iz + 2, alt);
            we[1] = bspline(p_, nnode_, vnode_.data(), iz + 3, alt);
            we[2] = bspline(p_, nnode_, vnode_.data(), iz + 4, alt);
        }
        wght[2] = we[0] * e1_[0] + we[1] * e1_[1] + we[2] * e1_[2] + we[3] * e1_[3] + we[4] * e1_[4];
        wght[3] = we[0] * e2_[0] + we[1] * e2_[1] + we[2] * e2_[2] + we[3] * e2_[3] + we[4] * e2_[4];
    }

    static double bspline(int p, int m, const double* V, int i, double u) {
        if (i == 0 && u == V[0]) return 1.0;
        if (i == (m - p - 1) && u == V[m]) return 1.0;
        if (u < V[i] || u >= V[i + p + 1]) return 0.0;

        std::vector<double> N(p + 2, 0.0);
        for (int j = 0; j <= p; ++j) {
            N[j] = (u >= V[i + j] && u < V[i + j + 1]) ? 1.0 : 0.0;
        }
        for (int k = 1; k <= p; ++k) {
            double saved = 0.0;
            if (N[0] != 0.0) {
                saved = ((u - V[i]) * N[0]) / (V[i + k] - V[i]);
            }
            for (int j = 0; j <= p - k; ++j) {
                double Vleft = V[i + j + 1];
                double Vright = V[i + j + k + 1];
                if (N[j + 1] == 0.0) {
                    N[j] = saved;
                    saved = 0.0;
                } else {
                    double temp = N[j + 1] / (Vright - Vleft);
                    N[j] = saved + (Vright - u) * temp;
                    saved = (u - Vleft) * temp;
                }
            }
        }
        return N[0];
    }

    static int findspan(int n, int p, double u, const double* V) {
        if (u >= V[n + 1]) return n;
        // Below the lowest knot (u < 0 km) the bisection below never ends
        // (it converges on mid = low with u < V[mid]): a propagation that
        // crossed the ground hung here in its first wind evaluation. The
        // lowest span; LiveModelWind also clamps the altitude to 0 km.
        if (u <= V[p]) return p;

        int low = p;
        int high = n + 1;
        int mid = (low + high) / 2;

        while (u < V[mid] || u >= V[mid + 1]) {
            if (u < V[mid])
                high = mid;
            else
                low = mid;
            mid = (low + high) / 2;
        }
        return mid;
    }
};

// Forward declaration
class GD2QD;

// =============================================================================
// Disturbance Wind Model (DWM07)
// =============================================================================

class DWM07 {
public:
    void init(int& nmaxout, int& mmaxout) {
        if (initialized_) {
            nmaxout = nmax_;
            mmaxout = mmax_;
            return;
        }

        const auto& data = hwm14::dwm_data;

        nterm_ = data.nterm;
        mmax_ = data.mmax;
        nmax_ = data.nmax;
        twidth_ = data.twidth;

        termarr_.assign(data.termarr.begin(), data.termarr.begin() + nterm_ * 3);
        coeff_.assign(data.coeff.begin(), data.coeff.begin() + nterm_);

        nvshterm_ = (((nmax_ + 1) * (nmax_ + 2) - (nmax_ - mmax_) * (nmax_ - mmax_ + 1)) / 2 - 1) * 4 - 2 * nmax_;

        termval_.resize(nterm_ * 2, 0.0f);
        dpbar_.resize((nmax_ + 1) * (mmax_ + 1), 0.0);
        dvbar_.resize((nmax_ + 1) * (mmax_ + 1), 0.0);
        dwbar_.resize((nmax_ + 1) * (mmax_ + 1), 0.0);
        mltterms_.resize((mmax_ + 1) * 2, 0.0);
        vshterms_.resize(nvshterm_ * 2, 0.0f);

        nmaxout = nmax_;
        mmaxout = mmax_;
        initialized_ = true;
    }

    // Compute disturbance winds in geographic coordinates
    // Defined after GD2QD class below
    void compute(int iyd, float sec, float alt, float glat, float glon,
                 const float ap[2],
                 ALFBasis& alf,
                 std::vector<double>& gpbar, std::vector<double>& gvbar, std::vector<double>& gwbar,
                 std::vector<double>& spbar, std::vector<double>& svbar, std::vector<double>& swbar,
                 double& glat_cached,
                 const GD2QD& gd2qd,
                 float dw[2]) const;

    [[nodiscard]] int nmax() const noexcept { return nmax_; }
    [[nodiscard]] int mmax() const noexcept { return mmax_; }

private:
    bool initialized_ = false;

    int nterm_ = 0, nmax_ = 0, mmax_ = 0;
    int nvshterm_ = 0;
    float twidth_ = 0.0f;
    std::vector<int> termarr_;      // 3 * nterm
    std::vector<float> coeff_;      // nterm
    mutable std::vector<float> vshterms_;   // 2 * nvshterm
    mutable std::vector<float> termval_;    // 2 * nterm
    mutable std::vector<double> dpbar_, dvbar_, dwbar_;
    mutable std::vector<double> mltterms_;

    mutable float glatlast_ = 1.0e16f, glonlast_ = 1.0e16f;
    mutable float daylast_ = 1.0e16f, utlast_ = 1.0e16f, aplast_ = 1.0e16f;
    mutable float kplast_ = 0.0f;
    mutable float mlatlast_ = 0.0f, mlonlast_ = 0.0f, mltlast_ = 0.0f;
    mutable float f1elast_ = 0.0f, f1nlast_ = 0.0f, f2elast_ = 0.0f, f2nlast_ = 0.0f;

    // Core DWM07 evaluation
    void dwm07b(float mlt, float mlat, float kp, ALFBasis& alf,
                float& mmpwind, float& mzpwind) const
    {
        static constexpr double DTOR = 3.14159265358979323846 / 180.0;

        // Ensure DWM ALF arrays are large enough for the shared ALF basis
        // (ALF uses idx(n,m) = n*(alf.mmax_+1)+m, so we need alf.array_size() elements)
        if (dpbar_.size() < static_cast<size_t>(alf.array_size())) {
            dpbar_.resize(alf.array_size(), 0.0);
            dvbar_.resize(alf.array_size(), 0.0);
            dwbar_.resize(alf.array_size(), 0.0);
        }

        // Compute ALF basis in magnetic coordinates
        if (mlat != mlatlast_dwm_) {
            double theta = (90.0 - static_cast<double>(mlat)) * DTOR;
            alf.compute(nmax_, mmax_, theta, dpbar_, dvbar_, dwbar_);
            mlatlast_dwm_ = mlat;
        }

        // Compute MLT Fourier terms
        if (mlt != mltlast_dwm_) {
            double phi = static_cast<double>(mlt) * DTOR * 15.0;
            for (int m = 0; m <= mmax_; ++m) {
                double mphi = static_cast<double>(m) * phi;
                mltterms_[m * 2] = std::cos(mphi);
                mltterms_[m * 2 + 1] = std::sin(mphi);
            }
            mltlast_dwm_ = mlt;
        }

        // Compute VSH terms
        if (mlat != mlatlast_dwm_prev_ || mlt != mltlast_dwm_prev_) {
            int iv = 0;
            for (int n = 1; n <= nmax_; ++n) {
                float dv = static_cast<float>(dvbar_[alf.idx(n, 0)] * mltterms_[0]);
                float dw = static_cast<float>(dwbar_[alf.idx(n, 0)] * mltterms_[0]);
                vshterms_[iv * 2] = -dv;              // vsh(0,iv) = -dv
                vshterms_[(iv + 1) * 2] = dw;         // vsh(0,iv+1) = +dw
                vshterms_[iv * 2 + 1] = -dw;          // vsh(1,iv) = -vsh(0,iv+1) = -dw
                vshterms_[(iv + 1) * 2 + 1] = -dv;    // vsh(1,iv+1) = vsh(0,iv) = -dv
                iv += 2;
                for (int m = 1; m <= mmax_; ++m) {
                    if (m > n) continue;
                    float dvnm = static_cast<float>(dvbar_[alf.idx(n, m)]);
                    float dwnm = static_cast<float>(dwbar_[alf.idx(n, m)]);
                    float cm0 = static_cast<float>(mltterms_[m * 2]);
                    float sm1 = static_cast<float>(mltterms_[m * 2 + 1]);
                    vshterms_[iv * 2] = -dvnm * cm0;
                    vshterms_[(iv + 1) * 2] = dvnm * sm1;
                    vshterms_[(iv + 2) * 2] = dwnm * sm1;
                    vshterms_[(iv + 3) * 2] = dwnm * cm0;
                    vshterms_[iv * 2 + 1] = -vshterms_[(iv + 2) * 2];
                    vshterms_[(iv + 1) * 2 + 1] = -vshterms_[(iv + 3) * 2];
                    vshterms_[(iv + 2) * 2 + 1] = vshterms_[iv * 2];
                    vshterms_[(iv + 3) * 2 + 1] = vshterms_[(iv + 1) * 2];
                    iv += 4;
                }
            }
            mlatlast_dwm_prev_ = mlat;
            mltlast_dwm_prev_ = mlt;
        }

        // Compute Kp terms
        std::array<float, 3> kpterms{};
        if (kp != kplast_dwm_) {
            kpspl3(kp, kpterms);
            kpterms_cached_ = kpterms;
            kplast_dwm_ = kp;
        } else {
            kpterms = kpterms_cached_;
        }

        // Latitude weighting
        float latwgt = latwgt2(mlat, mlt, kp, twidth_);

        // Generate coupled terms and apply coefficients
        mmpwind = 0.0f;
        mzpwind = 0.0f;
        for (int i = 0; i < nterm_; ++i) {
            float tv0 = 1.0f, tv1 = 1.0f;
            int t0 = termarr_[i * 3];
            int t1 = termarr_[i * 3 + 1];
            int t2 = termarr_[i * 3 + 2];
            if (t0 != 999) { tv0 *= vshterms_[t0 * 2]; tv1 *= vshterms_[t0 * 2 + 1]; }
            if (t1 != 999) { float kpt = kpterms[t1]; tv0 *= kpt; tv1 *= kpt; }
            if (t2 != 999) { tv0 *= latwgt; tv1 *= latwgt; }
            mmpwind += coeff_[i] * tv0;
            mzpwind += coeff_[i] * tv1;
        }
    }

    mutable float mlatlast_dwm_ = 1.0e16f, mltlast_dwm_ = 1.0e16f;
    mutable float mlatlast_dwm_prev_ = 1.0e16f, mltlast_dwm_prev_ = 1.0e16f;
    mutable float kplast_dwm_ = 1.0e16f;
    mutable std::array<float, 3> kpterms_cached_{};

    // Ap to Kp conversion (piecewise linear)
    static float ap2kp(float ap0) {
        static constexpr float apgrid[28] = {
            0.f,2.f,3.f,4.f,5.f,6.f,7.f,9.f,12.f,15.f,18.f,
            22.f,27.f,32.f,39.f,48.f,56.f,67.f,80.f,94.f,
            111.f,132.f,154.f,179.f,207.f,236.f,300.f,400.f
        };
        static constexpr float kpgrid[28] = {
            0.f/3.f,1.f/3.f,2.f/3.f,3.f/3.f,4.f/3.f,5.f/3.f,6.f/3.f,
            7.f/3.f,8.f/3.f,9.f/3.f,10.f/3.f,11.f/3.f,12.f/3.f,
            13.f/3.f,14.f/3.f,15.f/3.f,16.f/3.f,17.f/3.f,18.f/3.f,
            19.f/3.f,20.f/3.f,21.f/3.f,22.f/3.f,23.f/3.f,24.f/3.f,
            25.f/3.f,26.f/3.f,27.f/3.f
        };

        float ap = (std::max)(ap0, 0.0f);
        ap = (std::min)(ap, 400.0f);

        int i = 1;
        while (i < 27 && ap > apgrid[i]) ++i;

        if (ap == apgrid[i]) return kpgrid[i];
        return kpgrid[i - 1] + (ap - apgrid[i - 1]) / (3.0f * (apgrid[i] - apgrid[i - 1]));
    }

    // Cubic B-spline interpolation of Kp
    static void kpspl3(float kp, std::array<float, 3>& kpterms) {
        static constexpr float node[8] = {-10.f, -8.f, 0.f, 2.f, 5.f, 8.f, 18.f, 20.f};

        float x = (std::max)(kp, 0.0f);
        x = (std::min)(x, 8.0f);

        kpterms = {0.0f, 0.0f, 0.0f};
        std::array<float, 7> kpspl{};
        for (int i = 0; i < 7; ++i) {
            kpspl[i] = (x >= node[i] && x < node[i + 1]) ? 1.0f : 0.0f;
        }
        for (int j = 2; j <= 3; ++j) {
            for (int i = 0; i <= 8 - j - 1; ++i) {
                kpspl[i] = kpspl[i] * (x - node[i]) / (node[i + j - 1] - node[i])
                         + kpspl[i + 1] * (node[i + j] - x) / (node[i + j] - node[i + 1]);
            }
        }
        kpterms[0] = kpspl[0] + kpspl[1];
        kpterms[1] = kpspl[2];
        kpterms[2] = kpspl[3] + kpspl[4];
    }

    // Latitude weighting function
    static float latwgt2(float mlat, float mlt, float kp0, float twidth) {
        static constexpr float coeff[6] = {65.7633f, -4.60256f, -3.53915f,
                                            -1.99971f, -0.752193f, 0.972388f};
        static constexpr float DTOR = 3.14159265f / 180.0f;

        float mltrad = mlt * 15.0f * DTOR;
        float sinmlt = std::sin(mltrad);
        float cosmlt = std::cos(mltrad);
        float kp = (std::max)(kp0, 0.0f);
        kp = (std::min)(kp, 8.0f);
        float tlat = coeff[0] + coeff[1] * cosmlt + coeff[2] * sinmlt
                   + kp * (coeff[3] + coeff[4] * cosmlt + coeff[5] * sinmlt);
        return 1.0f / (1.0f + std::exp(-(std::abs(mlat) - tlat) / twidth));
    }

};

// =============================================================================
// Geographic to Quasi-Dipole Coordinate Conversion
// =============================================================================

class GD2QD {
public:
    void init() {
        if (initialized_) return;

        const auto& data = hwm14::gd2qd_data;
        nmax_ = data.nmax;
        mmax_ = data.mmax;
        nterm_ = data.nterm;

        xcoeff_.assign(data.xcoeff.begin(), data.xcoeff.begin() + nterm_);
        ycoeff_.assign(data.ycoeff.begin(), data.ycoeff.begin() + nterm_);
        zcoeff_.assign(data.zcoeff.begin(), data.zcoeff.begin() + nterm_);

        sh_.resize(nterm_, 0.0);
        shgradtheta_.resize(nterm_, 0.0);
        shgradphi_.resize(nterm_, 0.0);
        normadj_.resize(nmax_ + 1, 0.0);

        for (int n = 0; n <= nmax_; ++n) {
            normadj_[n] = std::sqrt(static_cast<double>(n * (n + 1)));
        }

        initialized_ = true;
    }

    void convert(float glatin, float glon,
                 ALFBasis& alf,
                 std::vector<double>& gpbar, std::vector<double>& gvbar, std::vector<double>& gwbar,
                 double& glat_cached,
                 float& qlat, float& qlon,
                 float& f1e, float& f1n, float& f2e, float& f2n) const
    {
        static constexpr double DTOR = 3.14159265358979323846 / 180.0;

        double glat = static_cast<double>(glatin);
        if (glat != glat_cached) {
            double theta = (90.0 - glat) * DTOR;
            alf.compute(nmax_, mmax_, theta, gpbar, gvbar, gwbar);
            glat_cached = glat;
        }
        double phi = static_cast<double>(glon) * DTOR;

        // Build spherical harmonic terms
        int i = 0;
        for (int n = 0; n <= nmax_; ++n) {
            sh_[i] = gpbar[alf.idx(n, 0)];
            shgradtheta_[i] = gvbar[alf.idx(n, 0)] * normadj_[n];
            shgradphi_[i] = 0.0;
            ++i;
        }
        for (int m = 1; m <= mmax_; ++m) {
            double mphi = static_cast<double>(m) * phi;
            double cosmphi = std::cos(mphi);
            double sinmphi = std::sin(mphi);
            for (int n = m; n <= nmax_; ++n) {
                sh_[i] = gpbar[alf.idx(n, m)] * cosmphi;
                sh_[i + 1] = gpbar[alf.idx(n, m)] * sinmphi;
                shgradtheta_[i] = gvbar[alf.idx(n, m)] * normadj_[n] * cosmphi;
                shgradtheta_[i + 1] = gvbar[alf.idx(n, m)] * normadj_[n] * sinmphi;
                shgradphi_[i] = -gwbar[alf.idx(n, m)] * normadj_[n] * sinmphi;
                shgradphi_[i + 1] = gwbar[alf.idx(n, m)] * normadj_[n] * cosmphi;
                i += 2;
            }
        }

        // Compute QD coordinates
        double x = dot(sh_.data(), xcoeff_.data(), nterm_);
        double y = dot(sh_.data(), ycoeff_.data(), nterm_);
        double z = dot(sh_.data(), zcoeff_.data(), nterm_);

        double qlonrad = std::atan2(y, x);
        double cosqlon = std::cos(qlonrad);
        double sinqlon = std::sin(qlonrad);
        double cosqlat = x * cosqlon + y * sinqlon;

        qlat = static_cast<float>(std::atan2(z, cosqlat) / DTOR);
        qlon = static_cast<float>(qlonrad / DTOR);

        // Compute gradients for rotation matrix
        double xgt = dot(shgradtheta_.data(), xcoeff_.data(), nterm_);
        double ygt = dot(shgradtheta_.data(), ycoeff_.data(), nterm_);
        double zgt = dot(shgradtheta_.data(), zcoeff_.data(), nterm_);
        double xgp = dot(shgradphi_.data(), xcoeff_.data(), nterm_);
        double ygp = dot(shgradphi_.data(), ycoeff_.data(), nterm_);
        double zgp = dot(shgradphi_.data(), zcoeff_.data(), nterm_);

        f1e = static_cast<float>(-zgt * cosqlat + (xgt * cosqlon + ygt * sinqlon) * z);
        f1n = static_cast<float>(-zgp * cosqlat + (xgp * cosqlon + ygp * sinqlon) * z);
        f2e = static_cast<float>(ygt * cosqlon - xgt * sinqlon);
        f2n = static_cast<float>(ygp * cosqlon - xgp * sinqlon);
    }

    // Compute Magnetic Local Time
    float mltcalc(float qlat, float qlon, float day, float ut,
                  ALFBasis& alf,
                  std::vector<double>& spbar, std::vector<double>& svbar, std::vector<double>& swbar) const
    {
        static constexpr double DTOR = 3.14159265358979323846 / 180.0;
        static constexpr double SINEPS = 0.39781868;

        // Compute geographic coordinates of anti-sunward direction
        double asunglat = -std::asin(std::sin((static_cast<double>(day)
                          + static_cast<double>(ut) / 24.0 - 80.0) * DTOR) * SINEPS) / DTOR;
        double asunglon = -static_cast<double>(ut) * 15.0;

        // Compute magnetic coordinates of anti-sunward direction
        double theta = (90.0 - asunglat) * DTOR;
        alf.compute(nmax_, mmax_, theta, spbar, svbar, swbar);
        double phi = asunglon * DTOR;

        int i = 0;
        for (int n = 0; n <= nmax_; ++n) {
            sh_[i] = spbar[alf.idx(n, 0)];
            ++i;
        }
        for (int m = 1; m <= mmax_; ++m) {
            double mphi = static_cast<double>(m) * phi;
            double cosmphi = std::cos(mphi);
            double sinmphi = std::sin(mphi);
            for (int n = m; n <= nmax_; ++n) {
                sh_[i] = spbar[alf.idx(n, m)] * cosmphi;
                sh_[i + 1] = spbar[alf.idx(n, m)] * sinmphi;
                i += 2;
            }
        }

        double x = dot(sh_.data(), xcoeff_.data(), nterm_);
        double y = dot(sh_.data(), ycoeff_.data(), nterm_);
        double asunqlon = static_cast<float>(std::atan2(y, x) / DTOR);

        return (qlon - static_cast<float>(asunqlon)) / 15.0f;
    }

    [[nodiscard]] int nmax() const noexcept { return nmax_; }
    [[nodiscard]] int mmax() const noexcept { return mmax_; }

private:
    bool initialized_ = false;
    int nmax_ = 0, mmax_ = 0, nterm_ = 0;

    std::vector<double> xcoeff_, ycoeff_, zcoeff_;
    mutable std::vector<double> sh_, shgradtheta_, shgradphi_;
    std::vector<double> normadj_;

    static double dot(const double* a, const double* b, int n) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }
};

// =============================================================================
// DWM07::compute() - Deferred definition (needs GD2QD)
// =============================================================================

inline void DWM07::compute(int iyd, float sec, float alt, float glat, float glon,
                            const float ap[2],
                            ALFBasis& alf,
                            std::vector<double>& gpbar, std::vector<double>& gvbar, std::vector<double>& gwbar,
                            std::vector<double>& spbar, std::vector<double>& svbar, std::vector<double>& swbar,
                            double& glat_cached,
                            const GD2QD& gd2qd,
                            float dw[2]) const
{
    static constexpr float TALT = 125.0f;

    // Convert Ap to Kp
    float kp = 0.0f;
    if (ap[1] != aplast_) {
        kp = ap2kp(ap[1]);
        kplast_ = kp;
    } else {
        kp = kplast_;
    }

    // Convert geographic to quasi-dipole coordinates
    float mlat = 0.0f, mlon = 0.0f;
    float f1e = 0.0f, f1n = 0.0f, f2e = 0.0f, f2n = 0.0f;
    if (glat != glatlast_ || glon != glonlast_) {
        gd2qd.convert(glat, glon, alf, gpbar, gvbar, gwbar, glat_cached,
                      mlat, mlon, f1e, f1n, f2e, f2n);
        mlatlast_ = mlat;
        mlonlast_ = mlon;
        f1elast_ = f1e; f1nlast_ = f1n;
        f2elast_ = f2e; f2nlast_ = f2n;
    } else {
        mlat = mlatlast_; mlon = mlonlast_;
        f1e = f1elast_; f1n = f1nlast_;
        f2e = f2elast_; f2n = f2nlast_;
    }

    // Compute MLT
    float day = static_cast<float>(iyd % 1000);
    float ut = sec / 3600.0f;
    float mlt = 0.0f;
    if (day != daylast_ || ut != utlast_ || glat != glatlast_ || glon != glonlast_) {
        mlt = gd2qd.mltcalc(mlat, mlon, day, ut, alf, spbar, svbar, swbar);
        mltlast_ = mlt;
    } else {
        mlt = mltlast_;
    }

    // Compute DWM winds in magnetic coordinates
    float mmpwind = 0.0f, mzpwind = 0.0f;
    dwm07b(mlt, mlat, kp, alf, mmpwind, mzpwind);

    // Convert to geographic coordinates
    dw[0] = f2n * mmpwind + f1n * mzpwind;
    dw[1] = f2e * mmpwind + f1e * mzpwind;

    // Apply altitude profile (sigmoid)
    float height_factor = 1.0f / (1.0f + std::exp(-(alt - TALT) / twidth_));
    dw[0] *= height_factor;
    dw[1] *= height_factor;

    glatlast_ = glat;
    glonlast_ = glon;
    daylast_ = day;
    utlast_ = ut;
    aplast_ = ap[1];
}

// =============================================================================
// HWM14 Main Class
// =============================================================================

class HWM14 {
public:
    // Initialize all submodules
    void init() {
        if (initialized_) return;

        qwm_.init();
        gd2qd_.init();

        int nmaxdwm = 0, mmaxdwm = 0;
        dwm_.init(nmaxdwm, mmaxdwm);

        // Compute maximum degrees for shared ALF arrays
        int nmaxgeo = (std::max)(qwm_.maxn(), gd2qd_.nmax());
        int mmaxgeo = (std::max)(qwm_.maxo(), gd2qd_.mmax());
        int nmax0 = (std::max)(nmaxgeo, nmaxdwm);
        int mmax0 = (std::max)(mmaxgeo, mmaxdwm);

        alf_.init(nmax0, mmax0);

        // Allocate shared ALF arrays
        int geo_size = (nmaxgeo + 1) * (mmaxgeo + 1);
        gpbar_.resize(alf_.array_size(), 0.0);
        gvbar_.resize(alf_.array_size(), 0.0);
        gwbar_.resize(alf_.array_size(), 0.0);
        spbar_.resize(alf_.array_size(), 0.0);
        svbar_.resize(alf_.array_size(), 0.0);
        swbar_.resize(alf_.array_size(), 0.0);

        glat_cached_ = -1.0e32;
        initialized_ = true;
    }

    // Main HWM14 entry point
    // Matches Fortran: hwm14(iyd, sec, alt, glat, glon, stl, f107a, f107, ap, w)
    void hwm14_calc(int iyd, float sec, float alt, float glat, float glon,
                    float stl, float f107a, float f107, const float ap[2],
                    float w[2])
    {
        if (!initialized_) init();

        // Quiet-time winds
        double wq[2] = {0.0, 0.0};
        qwm_.compute(iyd % 1000, static_cast<double>(sec), static_cast<double>(alt),
                      static_cast<double>(glat), static_cast<double>(glon),
                      alf_, gpbar_, gvbar_, gwbar_, glat_cached_, wq);
        w[0] = static_cast<float>(wq[0]);
        w[1] = static_cast<float>(wq[1]);

        // Add disturbance winds if Ap >= 0
        if (ap[1] >= 0.0f) {
            float dw[2] = {0.0f, 0.0f};
            dwm_.compute(iyd, sec, alt, glat, glon, ap,
                         alf_, gpbar_, gvbar_, gwbar_, spbar_, svbar_, swbar_,
                         glat_cached_, gd2qd_, dw);
            w[0] += dw[0];
            w[1] += dw[1];
        }
    }

    // Convenience interface matching the library pattern
    // alt_km: altitude in km
    // lat_deg: geodetic latitude in degrees
    // lon_deg: geodetic longitude in degrees
    // epoch: time
    // weather: space weather conditions (uses ap_index)
    WindComponents wind(double alt_km, double lat_deg, double lon_deg,
                        const Epoch& epoch, const WeatherConditions& weather)
    {
        if (!initialized_) init();

        // Convert epoch to year-day and UT seconds
        auto [year, month, day, hour, minute, second] = epoch.to_calendar(TimeScale::UTC);
        int doy = static_cast<int>(epoch.day_of_year());
        int iyd = (year % 100) * 1000 + doy;
        float sec = static_cast<float>(hour * 3600 + minute * 60 + second);

        float ap[2] = {0.0f, static_cast<float>(weather.ap_daily)};
        float w[2] = {0.0f, 0.0f};

        // Compute solar local time from UT and longitude
        double ut_hours = hour + minute / 60.0 + second / 3600.0;
        double stl = ut_hours + lon_deg / 15.0;
        // Wrap to [0, 24)
        stl = std::fmod(stl, 24.0);
        if (stl < 0.0) stl += 24.0;

        hwm14_calc(iyd, sec, static_cast<float>(alt_km), static_cast<float>(lat_deg),
                   static_cast<float>(lon_deg), static_cast<float>(stl), 0.0f, 0.0f, ap, w);

        return {static_cast<double>(w[0]), static_cast<double>(w[1])};
    }

private:
    bool initialized_ = false;
    ALFBasis alf_;
    QWM qwm_;
    DWM07 dwm_;
    GD2QD gd2qd_;

    // Shared ALF arrays
    mutable std::vector<double> gpbar_, gvbar_, gwbar_;
    mutable std::vector<double> spbar_, svbar_, swbar_;
    mutable double glat_cached_ = -1.0e32;
};

} // namespace atmosphere
} // namespace ballistic
