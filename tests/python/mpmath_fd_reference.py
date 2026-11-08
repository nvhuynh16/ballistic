"""
mpmath-FD reference for gravity Hessian (Stage D).

PURPOSE
=======

Generate high-precision (50-digit) reference values for the gravity Hessian
at specific test points using **finite-differences-on-gradient at very
small perturbation**. mpmath's arbitrary precision (default 50 digits here)
makes the truncation+roundoff floor of FD effectively zero (~1e-30 rel),
so these values serve as ground-truth references for the C++ analytic
Hessian implementation (Stage F-G).

WHY mpmath-FD AND NOT JUST FD-ON-DOUBLE
=======================================

Standard double-precision FD on the gradient hits a noise floor around
~1e-7 relative error: the perturbation `delta` must be small enough that
truncation error (∝ delta²) is small, but large enough that round-off
error (∝ machine_epsilon / delta) is small. The optimum is delta ~
machine_epsilon^(1/3) ≈ 1e-5, giving ~1e-7 floor.

This 1e-7 floor MASKS subtle bugs at the ~1e-8 to 1e-12 level — exactly
where sign flips on small terms or off-by-one degree-loop errors live.

With mpmath at 50 digits, machine_epsilon ≈ 1e-50, so the optimum delta
can be ~1e-17 with truncation error ~1e-34 and round-off ~1e-33. The FD
floor drops to ~1e-30 rel — far below the 1e-12 we care about.

Therefore: mpmath-FD reference values, exported as test data, give the
C++ implementation a reliable target to hit.

WHAT THIS SCRIPT PRODUCES
=========================

For now, this script computes the Hessian of a J2-only potential (deg=2,
zonal) at multiple test points, both via:
  (a) The analytical closed form (matching `j2_hessian_closed_form.hpp`).
  (b) FD on the gradient (also analytical for J2).
And cross-checks them at 30+ digit agreement.

For higher-degree EGM2008, this script would be extended to:
  - Read the EGM2008 coefficient file with mpmath-precision rationals.
  - Implement the Clenshaw gradient recurrence in mpmath.
  - FD on the gradient to produce the Hessian reference.

The current script demonstrates the methodology on the J2 case as proof-of-
concept; full EGM2008 reference is deferred until the C++ Hessian
implementation needs it.

USAGE
=====

    .venv/bin/python tests/python/mpmath_fd_reference.py

Outputs reference values + agreement analysis. Reference CSV writing
(for C++ test consumption) is to be added in a follow-up.
"""

from __future__ import annotations
import mpmath as mp
from typing import List, Tuple


# Earth constants (CODATA 2018 / IERS 2010)
GM_EARTH = mp.mpf("3.986004415e14")  # m³/s²
R_EARTH = mp.mpf("6378136.3")        # m (mean equatorial)
J2 = mp.mpf("1.0826267e-3")          # un-normalized


def j2_potential(x: mp.mpf, y: mp.mpf, z: mp.mpf) -> mp.mpf:
    """V_J2(x, y, z) = -GM·J2·R²·(2z²−x²−y²) / (2·r⁵), per geodesy convention."""
    r2 = x*x + y*y + z*z
    r5 = r2 ** mp.mpf("2.5")
    N = 2*z*z - x*x - y*y
    K = -GM_EARTH * J2 * R_EARTH * R_EARTH / 2
    return K * N / r5


def j2_acceleration(x: mp.mpf, y: mp.mpf, z: mp.mpf) -> Tuple[mp.mpf, mp.mpf, mp.mpf]:
    """∇V_J2 in closed form."""
    r2 = x*x + y*y + z*z
    r5 = r2 ** mp.mpf("2.5")
    r7 = r5 * r2
    N = 2*z*z - x*x - y*y
    K = -GM_EARTH * J2 * R_EARTH * R_EARTH / 2
    Nx, Ny, Nz = -2*x, -2*y, 4*z

    ax = K * (Nx / r5 - 5 * N * x / r7)
    ay = K * (Ny / r5 - 5 * N * y / r7)
    az = K * (Nz / r5 - 5 * N * z / r7)
    return ax, ay, az


def j2_hessian_closed(x: mp.mpf, y: mp.mpf, z: mp.mpf) -> List[List[mp.mpf]]:
    """Closed-form J2 Hessian, 3×3."""
    r2 = x*x + y*y + z*z
    r5 = r2 ** mp.mpf("2.5")
    r7 = r5 * r2
    r9 = r7 * r2
    N = 2*z*z - x*x - y*y
    K = -GM_EARTH * J2 * R_EARTH * R_EARTH / 2
    Nx, Ny, Nz = -2*x, -2*y, 4*z
    Nxx, Nyy, Nzz = mp.mpf(-2), mp.mpf(-2), mp.mpf(4)

    def Hij(Nij, Ni, Nj, xi, xj, dij):
        return K * (Nij / r5
                    - 5 * (Ni * xj + Nj * xi) / r7
                    - 5 * dij * N / r7
                    + 35 * N * xi * xj / r9)

    H = [
        [Hij(Nxx, Nx, Nx, x, x, mp.mpf(1)),
         Hij(mp.mpf(0), Nx, Ny, x, y, mp.mpf(0)),
         Hij(mp.mpf(0), Nx, Nz, x, z, mp.mpf(0))],
        [None,
         Hij(Nyy, Ny, Ny, y, y, mp.mpf(1)),
         Hij(mp.mpf(0), Ny, Nz, y, z, mp.mpf(0))],
        [None, None,
         Hij(Nzz, Nz, Nz, z, z, mp.mpf(1))],
    ]
    H[1][0] = H[0][1]
    H[2][0] = H[0][2]
    H[2][1] = H[1][2]
    return H


def j2_hessian_fd(
    x: mp.mpf, y: mp.mpf, z: mp.mpf,
    delta: mp.mpf,
) -> List[List[mp.mpf]]:
    """Hessian via central FD on the gradient. mpmath precision makes this
    a reliable reference at delta ~ 1e-17 (50-digit floor ~ 1e-30 rel).
    """
    H = [[mp.mpf(0)] * 3 for _ in range(3)]
    pos = [x, y, z]
    for j in range(3):
        pos_p = pos[:]
        pos_m = pos[:]
        pos_p[j] = pos[j] + delta
        pos_m[j] = pos[j] - delta
        ap = j2_acceleration(*pos_p)
        am = j2_acceleration(*pos_m)
        for i in range(3):
            H[i][j] = (ap[i] - am[i]) / (2 * delta)
    return H


def main() -> int:
    mp.mp.dps = 50  # 50 digits

    print("=" * 72)
    print("  mpmath-FD reference for gravity Hessian (Stage D)")
    print(f"  precision = {mp.mp.dps} digits")
    print("=" * 72)

    test_points = [
        ("LEO equatorial", mp.mpf("6.778e6"), mp.mpf(0), mp.mpf(0)),
        ("LEO mid-lat", mp.mpf("5.6e6"), mp.mpf(0), mp.mpf("3.85e6")),
        ("LEO near-pole", mp.mpf("0.115e6"), mp.mpf(0), mp.mpf("6.777e6")),
        ("MEO equatorial", mp.mpf("2.65e7"), mp.mpf(0), mp.mpf(0)),
        ("GEO equatorial", mp.mpf("4.2164e7"), mp.mpf(0), mp.mpf(0)),
        ("Off-axis", mp.mpf("5e6"), mp.mpf("3e6"), mp.mpf("4e6")),
    ]

    delta = mp.mpf("1e-15")  # well below mpmath noise floor

    overall_pass = True
    for label, x, y, z in test_points:
        H_closed = j2_hessian_closed(x, y, z)
        H_fd = j2_hessian_fd(x, y, z, delta)

        # Frobenius norms
        scale_closed = mp.sqrt(sum(H_closed[i][j]**2 for i in range(3) for j in range(3)))
        diff_norm = mp.sqrt(sum(
            (H_closed[i][j] - H_fd[i][j])**2 for i in range(3) for j in range(3)
        ))
        rel = diff_norm / (scale_closed + mp.mpf("1e-100"))

        # Laplace check on closed form (must be exact)
        laplace_closed = H_closed[0][0] + H_closed[1][1] + H_closed[2][2]
        laplace_rel = abs(laplace_closed) / (scale_closed + mp.mpf("1e-100"))

        # Symmetry check on FD (will have tiny noise)
        sym_max_fd = max(abs(H_fd[i][j] - H_fd[j][i])
                         for i in range(3) for j in range(i+1, 3))

        match = rel < mp.mpf("1e-25")
        if not match:
            overall_pass = False

        print()
        print(f"  {label}")
        print(f"    ‖H_closed‖_F        = {mp.nstr(scale_closed, 6)}")
        print(f"    ‖H_closed − H_fd‖_F = {mp.nstr(diff_norm, 6)}")
        print(f"    relative diff        = {mp.nstr(rel, 6)}  "
              f"({'PASS' if match else 'FAIL'})")
        print(f"    Laplace on closed    = {mp.nstr(laplace_rel, 6)}")
        print(f"    FD symmetry residual = {mp.nstr(sym_max_fd, 6)}")

    print()
    print("=" * 72)
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")
    print()
    print("  Interpretation:")
    print("  - 'relative diff' compares closed-form Hessian to FD-on-gradient")
    print("    at delta=1e-15 with mpmath at 50 digits.")
    print("  - PASS at < 1e-25 confirms FD-on-gradient is reliable down to")
    print("    ~25 digits — far below the 1e-12 we care about for C++ tests.")
    print("  - Laplace residual ≈ 0 (closed-form is exact harmonic).")
    print("  - This methodology generalizes to higher-degree EGM2008 once a")
    print("    mpmath gradient implementation is added.")
    print("=" * 72)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
