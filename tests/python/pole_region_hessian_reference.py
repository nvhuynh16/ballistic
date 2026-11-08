"""Pole-region gravity Hessian reference generator (mpmath @ 50-digit).

Generates ground-truth Hessian values at a (lat, alt) grid that approaches
both poles. Output: tests/data/pole_hessian_refs.csv.

Used by tests/unit/test_pole_region_hessian.cpp to measure the C++
ClenshawEngine::compute_with_hessian_analytic accuracy near the pole and
decide whether the deferred Pines Cartesian implementation (Stage G) is
worth pursuing.

Reference math:
  V(x, y, z) = +GM/r × Σ_l (R/r)^l × Σ_m P̄_lm(sin φ) × [C̄_lm cos(mλ) + S̄_lm sin(mλ)]
  Hessian H[i][j] = ∂²V/∂x_i∂x_j
  Computed via central-FD on V at step δ=1e-2 m. mpmath @ 50-digit makes
  truncation+roundoff floor ~1e-30 rel (far below the ~1e-12 we care about).

Convention matches GravityHessianResult in clenshaw_engine.hpp:
  V = +GM/r (geodesy convention, NOT the physics -GM/r)
  Acceleration a = +∇V
  Hessian symmetric, traceless in vacuum (Laplace).

Usage:
  .venv/bin/python tests/python/pole_region_hessian_reference.py
"""
from __future__ import annotations
import csv
import time
from pathlib import Path

import mpmath as mp

# ---- Constants (match include/ballistic/core/constants.hpp) ----
GM_EARTH = mp.mpf("3.986004415e14")
R_EARTH = mp.mpf("6378136.3")

EGM2008_DATA = Path("data/EGM2008_Spherical_Harmonics/EGM2008_to2190_TideFree")
OUTPUT_CSV = Path("tests/data/pole_hessian_refs.csv")
MAX_DEGREE = 70  # TrajectoryConfig default gravity degree

FD_DELTA_M = mp.mpf("1e-2")  # 1 cm — well within mpmath 50-digit accuracy budget


def load_egm2008_coefficients(path: Path, max_deg: int):
    """Returns C[l][m], S[l][m] as mpmath; fully-normalized."""
    C = [[mp.mpf(0)] * (l + 1) for l in range(max_deg + 1)]
    S = [[mp.mpf(0)] * (l + 1) for l in range(max_deg + 1)]
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            l = int(parts[0])
            m = int(parts[1])
            if l > max_deg:
                break
            C[l][m] = mp.mpf(parts[2].replace("D", "E"))
            S[l][m] = mp.mpf(parts[3].replace("D", "E"))
    return C, S


def normalized_associated_legendre(max_deg: int, x):
    """Returns P[l][m] = P̄_lm(x), fully-normalized geodesy convention.

    x = sin(geocentric_latitude). For x in [-1, 1].
    Recurrence: Heiskanen & Moritz (1967), §1.13-1.14, fully-normalized form.
    """
    # Allocate
    P = [[mp.mpf(0)] * (max_deg + 1) for _ in range(max_deg + 1)]

    # P̄_00 = 1
    P[0][0] = mp.mpf(1)
    if max_deg == 0:
        return P

    # u = sqrt(1 - x²) = cos(latitude)
    u = mp.sqrt(1 - x * x)

    # Sectorial: P̄_mm = sqrt((2m+1)/(2m)) × u × P̄_(m-1)(m-1)
    for m in range(1, max_deg + 1):
        coef = mp.sqrt(mp.mpf(2 * m + 1) / (2 * m))
        # Note: m=1 special case; standard formula for fully-normalized:
        #   P̄_11 = sqrt(3) × u × P̄_00 — handled below since (2*1+1)/(2*1) = 3/2 → sqrt(3/2)
        # Wait: Heiskanen-Moritz fully-normalized has different factor.
        # Standard "geodesy 4π" normalization:
        #   P̄_lm = N_lm × P_lm  where N_lm = sqrt((2-δ_m0)(2l+1)(l-m)!/(l+m)!)
        # Sectorial recursion: P̄_mm = sqrt((2m+1)(2-δ_m0)/(2m(2-δ_(m-1)0))) × u × P̄_(m-1)(m-1)
        # For m=1: factor = sqrt(3·2/(2·1)) = sqrt(3). For m≥2: sqrt((2m+1)/(2m)).
        if m == 1:
            coef = mp.sqrt(mp.mpf(3))
        P[m][m] = coef * u * P[m - 1][m - 1]

    # Off-diagonal: P̄_(m+1)m = sqrt(2m+3) × x × P̄_mm
    for m in range(0, max_deg):
        if m + 1 > max_deg:
            break
        P[m + 1][m] = mp.sqrt(mp.mpf(2 * m + 3)) * x * P[m][m]

    # Three-term recurrence in l (for fixed m), l ≥ m+2
    # P̄_lm = a_lm × x × P̄_(l-1)m − b_lm × P̄_(l-2)m
    #   a_lm = sqrt((2l-1)(2l+1) / ((l-m)(l+m)))
    #   b_lm = sqrt((2l+1)(l-1+m)(l-1-m) / ((l-m)(l+m)(2l-3)))
    for m in range(0, max_deg + 1):
        for l in range(m + 2, max_deg + 1):
            a = mp.sqrt(
                (mp.mpf(2 * l - 1) * (2 * l + 1))
                / ((l - m) * (l + m))
            )
            b = mp.sqrt(
                (mp.mpf(2 * l + 1) * (l - 1 + m) * (l - 1 - m))
                / ((l - m) * (l + m) * (2 * l - 3))
            )
            P[l][m] = a * x * P[l - 1][m] - b * P[l - 2][m]
    return P


def gravity_potential(x, y, z, C, S, max_deg: int):
    """V(x,y,z) at mpmath precision; geodesy convention V = +GM/r·(1 + ...)."""
    r = mp.sqrt(x * x + y * y + z * z)
    sin_phi = z / r
    rho = mp.sqrt(x * x + y * y)  # cylindrical distance from z-axis
    cos_phi = rho / r

    # Longitude: cos(λ), sin(λ). Special case at pole (rho ≈ 0).
    if cos_phi > mp.mpf("1e-30"):
        cos_lam = x / rho
        sin_lam = y / rho
    else:
        cos_lam = mp.mpf(1)
        sin_lam = mp.mpf(0)

    P = normalized_associated_legendre(max_deg, sin_phi)

    R_over_r = R_EARTH / r
    Rr_l = mp.mpf(1)  # (R/r)^0
    series = mp.mpf(0)
    for l in range(max_deg + 1):
        sum_m = mp.mpf(0)
        cos_ml = mp.mpf(1)  # cos(0·λ) = 1
        sin_ml = mp.mpf(0)
        for m in range(l + 1):
            sum_m += P[l][m] * (C[l][m] * cos_ml + S[l][m] * sin_ml)
            # advance to (m+1)·λ via angle-addition
            new_cos = cos_ml * cos_lam - sin_ml * sin_lam
            new_sin = sin_ml * cos_lam + cos_ml * sin_lam
            cos_ml, sin_ml = new_cos, new_sin
        series += Rr_l * sum_m
        Rr_l *= R_over_r

    return GM_EARTH / r * series


def gravity_hessian_fd(pos, C, S, max_deg: int, delta=FD_DELTA_M):
    """4-point central FD for off-diagonal, 3-point for diagonal."""
    H = [[mp.mpf(0)] * 3 for _ in range(3)]

    def V_at(p):
        return gravity_potential(p[0], p[1], p[2], C, S, max_deg)

    V0 = V_at(pos)
    inv_d2 = 1 / (delta * delta)

    # Diagonal: H_ii = (V(r+δi) − 2V0 + V(r−δi)) / δ²
    for i in range(3):
        pp = list(pos); pp[i] = pos[i] + delta
        pm = list(pos); pm[i] = pos[i] - delta
        Vpp = V_at(pp)
        Vmm = V_at(pm)
        H[i][i] = (Vpp - 2 * V0 + Vmm) * inv_d2

    # Off-diagonal: H_ij = (V(r+δi+δj) − V(r+δi−δj) − V(r−δi+δj) + V(r−δi−δj))/(4δ²)
    for i in range(3):
        for j in range(i + 1, 3):
            ppp = list(pos); ppp[i] = pos[i] + delta; ppp[j] = pos[j] + delta
            ppm = list(pos); ppm[i] = pos[i] + delta; ppm[j] = pos[j] - delta
            pmp = list(pos); pmp[i] = pos[i] - delta; pmp[j] = pos[j] + delta
            pmm = list(pos); pmm[i] = pos[i] - delta; pmm[j] = pos[j] - delta
            Vppp = V_at(ppp); Vppm = V_at(ppm); Vpmp = V_at(pmp); Vpmm = V_at(pmm)
            H[i][j] = (Vppp - Vppm - Vpmp + Vpmm) * inv_d2 / 4
            H[j][i] = H[i][j]

    return H


def lat_alt_to_ecef(lat_deg, lon_deg, alt_m):
    """Spherical (geocentric) latitude/longitude → ECEF Cartesian."""
    lat = mp.radians(mp.mpf(str(lat_deg)))
    lon = mp.radians(mp.mpf(str(lon_deg)))
    r = R_EARTH + mp.mpf(str(alt_m))
    cos_lat = mp.cos(lat)
    sin_lat = mp.sin(lat)
    return [r * cos_lat * mp.cos(lon),
            r * cos_lat * mp.sin(lon),
            r * sin_lat]


def main():
    mp.mp.dps = 50

    print("=" * 72)
    print("  Pole-region Hessian reference generator (mpmath @ 50 digits)")
    print(f"  max_degree = {MAX_DEGREE}, FD δ = {FD_DELTA_M} m")
    print("=" * 72)

    print("Loading EGM2008 coefficients...")
    t0 = time.time()
    C, S = load_egm2008_coefficients(EGM2008_DATA, MAX_DEGREE)
    print(f"  loaded ({time.time()-t0:.1f}s)")

    # Test grid: latitude approaching pole × multiple altitudes × both poles
    latitudes = [
        45.0,                                         # mid-lat baseline
        75.0, 80.0, 85.0,                             # far from pole
        88.0, 89.0,                                   # approaching
        89.5, 89.9,                                   # near
        89.99, 89.999,                                # very near
        90.0,                                         # exact pole
        -89.99, -89.0,                                # negative side: symmetry
    ]
    alts_km = [200.0, 800.0, 20000.0]
    lon_deg = 30.0  # arbitrary non-zero longitude (rotate to expose any ZX symmetry bugs)

    rows = []
    n_total = len(latitudes) * len(alts_km)
    counter = 0
    for lat in latitudes:
        for alt in alts_km:
            counter += 1
            t0 = time.time()
            pos = lat_alt_to_ecef(lat, lon_deg, alt * 1000.0)
            H = gravity_hessian_fd(pos, C, S, MAX_DEGREE)
            elapsed = time.time() - t0
            print(f"  [{counter}/{n_total}] lat={lat:7.3f}° alt={alt:7.0f}km  "
                  f"H_xx={float(H[0][0]):+.6e}  ({elapsed:.1f}s)")
            rows.append({
                "lat_deg": str(lat),
                "lon_deg": str(lon_deg),
                "alt_km": str(alt),
                "max_deg": str(MAX_DEGREE),
                "x_m": mp.nstr(pos[0], 25),
                "y_m": mp.nstr(pos[1], 25),
                "z_m": mp.nstr(pos[2], 25),
                "H_xx": mp.nstr(H[0][0], 25),
                "H_xy": mp.nstr(H[0][1], 25),
                "H_xz": mp.nstr(H[0][2], 25),
                "H_yy": mp.nstr(H[1][1], 25),
                "H_yz": mp.nstr(H[1][2], 25),
                "H_zz": mp.nstr(H[2][2], 25),
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote {len(rows)} reference points to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
