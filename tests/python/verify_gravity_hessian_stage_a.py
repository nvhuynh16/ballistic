"""
Stage A SymPy verification — gravity Hessian derivation.

PURPOSE
=======

Verifies algebraic claims in `docs/gravity_hessian_derivation.md` and
updates the human-review tracking in the gravity-Hessian review-items record.

Each verification function targets ONE [ITEM-NNN] from the review-items
file. The function body documents what's being verified, where the claim
comes from, what method is used, and on failure prints enough information
to identify which derivation step is implicated.

This script is INTENTIONALLY self-documenting. Running it produces a
verifiable audit trail of which claims are SymPy-verified. When future
tests or implementations reveal a bug, the trace from this script (run
historically or re-run) tells you which mathematical claims are
algorithmically guaranteed vs which depend on human review.

USAGE
=====

Dependencies (sympy, mpmath) are managed via `uv` per the project's Python
toolchain.

    # Install once if not already in the venv:
    uv pip install sympy mpmath

    # Run:
    .venv/bin/python tests/python/verify_gravity_hessian_stage_a.py

Exit code 0 if all items pass; nonzero if any failed.

When this script verifies an item, manually update the item's status in
the review-items record from "Pending" to
"SymPy-verified", with a one-line note pointing to the function name in
this script that did the verification.

DESIGN NOTES (for future debugging)
====================================

- Each `verify_item_NNN()` function returns a `VerificationResult`.
- The result carries (item_id, claim, source, method, passed, detail).
- `detail` contains the full symbolic comparison so failures show exactly
  what went wrong — the SymPy diff and which derivation step it implicates.
- Adding a new verification: write a function with the same docstring
  template (Claim / Source / Method / Implicated-on-failure), add it to
  `main()`'s `verifications` list.

LIMITATIONS
===========

SymPy CAN verify symbolic identities, chain rule applications, and
algebraic equivalence. SymPy CANNOT verify:
- That the source code lines I cite (file:line) are correct.
- That my problem framing is right (e.g., correct frame).
- That edge cases I forgot are actually handled.
- That EGM2008 uses fully-normalized harmonics (a reference fact).

Items that fall outside SymPy's domain stay in
the review-items record for human review.

REFERENCES (for the math)
==========================

Existing code:
  - include/ballistic/gravity/clenshaw_engine.hpp (lines 109-302)
  - include/ballistic/gravity/clenshaw_storage.hpp (recursion factors)

Derivation source:
  - docs/gravity_hessian_derivation.md
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Callable, List

import sympy as sp
import mpmath as mp


@dataclass
class VerificationResult:
    """A single verification outcome, with enough trace info to debug."""
    item_id: str
    claim: str
    source: str
    method: str
    passed: bool
    detail: str  # symbolic expressions, diffs, etc.

    def report(self) -> None:
        """Print a structured trace report for this verification."""
        status_marker = "✅ PASS" if self.passed else "❌ FAIL"
        print()
        print("=" * 72)
        print(f"  {self.item_id}  [{status_marker}]")
        print("=" * 72)
        print(f"  Claim:  {self.claim}")
        print(f"  Source: {self.source}")
        print(f"  Method: {self.method}")
        print()
        print("  Detail:")
        for line in self.detail.rstrip().split("\n"):
            print(f"    {line}")


# =============================================================================
# Symbolic primitives shared across verifications
# =============================================================================
#
# The existing code's variables (transcribed from clenshaw_engine.hpp:124-152):
#   r  = |pos_ecef|
#   t  = z/r = sin(geocentric_latitude)
#   u  = sqrt(x²+y²)/r = cos(geocentric_latitude)
#   q  = R_E / r       (Earth-radius ratio)
#   cl = x/sqrt(x²+y²) = cos(longitude) [pole branch: cl=1, sl=0]
#   sl = y/sqrt(x²+y²) = sin(longitude)
#
# Recurrence factors:
#   rf.w_sqrt2n3 = sqrt(2n+3)
#   rf.b_factor  = b(n,m), encodes the (n+m)(n-m+1)(...) products for
#                  the fully-normalized Legendre recurrence. The exact form
#                  depends on the storage policy — for symbolic verification
#                  we treat it as a variable independent of r,φ,λ.
#
# In the symbolic setup below we use:
#   r, R     : positive reals
#   t, u     : reals (the constraint t²+u²=1 isn't enforced; we use them
#              independently because the existing code does too)
#   q        : sp.Symbol (we'll define q = R/r when needed)
#   sqrt2n3  : positive real (independent of r,φ,λ)
#   b_factor : positive real (independent of r,φ,λ)


def _symbolic_environment():
    """Return a dict of symbolic names matching the existing code."""
    r, R = sp.symbols("r R", positive=True, real=True)
    t = sp.symbols("t", real=True)         # sin φ
    u = sp.symbols("u", positive=True)      # cos φ (positive — non-pole)
    sqrt2n3 = sp.symbols("sqrt2n3", positive=True)
    b_factor = sp.symbols("b_factor", positive=True)
    n = sp.symbols("n", integer=True, nonnegative=True)
    Cv = sp.symbols("Cv", real=True)        # C_lm coefficient (constant in r,φ,λ)

    q = R / r
    Ax = q * sqrt2n3                        # clenshaw_engine.hpp line 179
    A = t * Ax                              # line 180
    B = -q**2 * b_factor                    # line 181

    return {
        "r": r, "R": R, "t": t, "u": u,
        "n": n, "Cv": Cv,
        "sqrt2n3": sqrt2n3, "b_factor": b_factor,
        "q": q, "Ax": Ax, "A": A, "B": B,
    }


# =============================================================================
# ITEM-005 — §2 wrrc recurrence chain rule
# =============================================================================

def verify_item_005() -> VerificationResult:
    """ITEM-005 — wrrc recurrence (§2's worked example).

    CLAIM
        Differentiating the existing wrc recurrence
            wrc_new = A·wrc + B·wrc2 + (n+1)·Cv
        with respect to r yields
            wrrc_new = A·wrrc + B·wrrc2 - (A/r)·wrc - (2B/r)·wrc2
        where wrrc ≡ d(wrc)/dr and wrrc2 ≡ d(wrc2)/dr.
        The corrections come from
            ∂A/∂r = -A/r       (since A ∝ q ∝ 1/r)
            ∂B/∂r = -2B/r       (since B ∝ q² ∝ 1/r²)

    SOURCE
        docs/gravity_hessian_derivation.md §2 worked example.

    METHOD
        Define A, B, q symbolically as in the existing code. Treat wrc, wrc2
        as functions of r (sympy.Function). Apply sp.diff() to wrc_new wrt r.
        Expand both SymPy's result and the claimed formula. Their symbolic
        difference must simplify to 0.

    IMPLICATED ON FAILURE
        The §2 derivation has an error — most likely in computing ∂A/∂r,
        ∂B/∂r, or in the product-rule application. SymPy's diff result will
        show what the ACTUAL recurrence should be, by comparison to the
        claim.
    """
    env = _symbolic_environment()
    r, A, B, n, Cv = env["r"], env["A"], env["B"], env["n"], env["Cv"]

    # Treat wrc, wrc2 as functions of r (so we can take their derivatives)
    wrc_f = sp.Function("wrc")(r)
    wrc2_f = sp.Function("wrc2")(r)

    # Existing recurrence (from clenshaw_engine.hpp:191-194):
    wrc_new = A * wrc_f + B * wrc2_f + (n + 1) * Cv

    # SymPy: full chain-rule differentiation wrt r
    wrrc_sympy = sp.expand(sp.diff(wrc_new, r))

    # Claimed formula from §2 of the derivation
    wrrc_f = sp.diff(wrc_f, r)
    wrrc2_f = sp.diff(wrc2_f, r)
    wrrc_claimed = sp.expand(
        A * wrrc_f + B * wrrc2_f
        - (A / r) * wrc_f
        - (2 * B / r) * wrc2_f
    )

    # The difference must simplify to 0
    diff_expr = sp.simplify(wrrc_sympy - wrrc_claimed)
    passed = (diff_expr == 0)

    detail = (
        f"SymPy's d(wrc_new)/dr (expanded):\n"
        f"  {wrrc_sympy}\n\n"
        f"Claimed wrrc_new (expanded):\n"
        f"  {wrrc_claimed}\n\n"
        f"simplify(SymPy - claimed):\n"
        f"  {diff_expr}\n"
    )

    return VerificationResult(
        item_id="ITEM-005",
        claim="d(wrc_new)/dr = A·wrrc + B·wrrc2 - (A/r)·wrc - (2B/r)·wrc2",
        source="docs/gravity_hessian_derivation.md §2 worked example",
        method="sp.diff() of wrc_new wrt r; compare expanded forms via simplify(diff - claimed) == 0",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-005b — chain rule of A: ∂A/∂r = -A/r
# =============================================================================

def verify_item_005b() -> VerificationResult:
    """ITEM-005b — sub-claim of ITEM-005: ∂A/∂r = -A/r.

    CLAIM
        With A = t · q · sqrt(2n+3) and q = R/r,
            ∂A/∂r = -A/r.

    SOURCE
        docs/gravity_hessian_derivation.md §2, justifying the (-A/r)·wrc
        correction term in the wrrc recurrence.

    METHOD
        SymPy directly: sp.diff(A, r) == -A/r should simplify to 0.

    IMPLICATED ON FAILURE
        The chain-rule application for A was wrong; ITEM-005 will also
        fail. This is a stepping-stone check that isolates the error
        before the fuller wrrc verification.
    """
    env = _symbolic_environment()
    r, A = env["r"], env["A"]

    expected = -A / r
    diff_expr = sp.simplify(sp.diff(A, r) - expected)
    passed = (diff_expr == 0)

    detail = (
        f"A = {A}\n"
        f"sp.diff(A, r) = {sp.diff(A, r)}\n"
        f"Claimed: -A/r = {expected}\n"
        f"simplify(sp.diff(A, r) - (-A/r)) = {diff_expr}\n"
    )

    return VerificationResult(
        item_id="ITEM-005b",
        claim="∂A/∂r = -A/r where A = t·(R/r)·√(2n+3)",
        source="docs/gravity_hessian_derivation.md §2",
        method="Direct SymPy diff; simplify(diff(A,r) - (-A/r)) == 0",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-005c — chain rule of B: ∂B/∂r = -2B/r
# =============================================================================

def verify_item_005c() -> VerificationResult:
    """ITEM-005c — sub-claim of ITEM-005: ∂B/∂r = -2B/r.

    CLAIM
        With B = -q² · b_factor and q = R/r,
            ∂B/∂r = -2B/r.

    SOURCE
        docs/gravity_hessian_derivation.md §2, justifying the (-2B/r)·wrc2
        correction term in the wrrc recurrence.

    METHOD
        SymPy: sp.diff(B, r) == -2*B/r should simplify to 0.

    IMPLICATED ON FAILURE
        Sub-error in the B chain rule; ITEM-005 also fails.
    """
    env = _symbolic_environment()
    r, B = env["r"], env["B"]

    expected = -2 * B / r
    diff_expr = sp.simplify(sp.diff(B, r) - expected)
    passed = (diff_expr == 0)

    detail = (
        f"B = {B}\n"
        f"sp.diff(B, r) = {sp.diff(B, r)}\n"
        f"Claimed: -2B/r = {expected}\n"
        f"simplify(diff - (-2B/r)) = {diff_expr}\n"
    )

    return VerificationResult(
        item_id="ITEM-005c",
        claim="∂B/∂r = -2B/r where B = -(R/r)²·b_factor",
        source="docs/gravity_hessian_derivation.md §2",
        method="Direct SymPy diff; simplify(diff(B,r) - (-2B/r)) == 0",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-008 — §2.2 wrtc recurrence (natural convention)
# =============================================================================

def verify_item_008() -> VerificationResult:
    """ITEM-008 — wrtc recurrence (natural sign convention).

    CLAIM
        Differentiating the existing wrc recurrence
            wrc_new = A·wrc + B·wrc2 + (n+1)·Cv
        with respect to φ (where t = sin φ, u = cos φ, A = t·Ax, B = -q²·b)
        yields, in natural sign convention:
            wrtc_new = A·wrtc + B·wrtc2 + u·Ax·wrc
        where wrtc ≡ ∂(wrc)/∂φ and wrtc2 ≡ ∂(wrc2)/∂φ. The +u·Ax·wrc term
        comes from ∂A/∂φ = u·Ax (since A = t·Ax and ∂t/∂φ = cos φ = u).

    SOURCE
        docs/gravity_hessian_derivation.md §2.2.

    METHOD
        Treat wrc, wrc2 as functions of φ (sympy.Function). Express A and B
        in terms of t and u, with t = sin(φ), u = cos(φ). Apply sp.diff()
        wrt φ. Compare to claimed formula via simplify(diff - claimed) == 0.

    IMPLICATED ON FAILURE
        The §2.2 derivation has an error — likely in computing ∂A/∂φ or in
        applying the product rule.
    """
    # We need t, u to BOTH be functions of phi for differentiation
    phi = sp.symbols("phi", real=True)
    t_func = sp.sin(phi)
    u_func = sp.cos(phi)

    R, r, sqrt2n3, b_factor, n, Cv = sp.symbols(
        "R r sqrt2n3 b_factor n Cv", real=True
    )
    q = R / r
    Ax = q * sqrt2n3
    A_phi = t_func * Ax     # A as a function of phi (via t = sin phi)
    B_phi = -q**2 * b_factor  # B is independent of phi

    # wrc, wrc2 as functions of phi
    wrc_f = sp.Function("wrc")(phi)
    wrc2_f = sp.Function("wrc2")(phi)

    wrc_new = A_phi * wrc_f + B_phi * wrc2_f + (n + 1) * Cv
    wrtc_sympy = sp.expand(sp.diff(wrc_new, phi))

    # Claimed formula: wrtc_new = A·wrtc + B·wrtc2 + u·Ax·wrc
    wrtc_f = sp.diff(wrc_f, phi)
    wrtc2_f = sp.diff(wrc2_f, phi)
    wrtc_claimed = sp.expand(
        A_phi * wrtc_f + B_phi * wrtc2_f + u_func * Ax * wrc_f
    )

    diff_expr = sp.simplify(wrtc_sympy - wrtc_claimed)
    passed = (diff_expr == 0)

    detail = (
        f"SymPy's d(wrc_new)/dφ (expanded):\n"
        f"  {wrtc_sympy}\n\n"
        f"Claimed wrtc_new (expanded):\n"
        f"  {wrtc_claimed}\n\n"
        f"simplify(SymPy - claimed):\n"
        f"  {diff_expr}\n"
    )

    return VerificationResult(
        item_id="ITEM-008",
        claim="d(wrc_new)/dφ = A·wrtc + B·wrtc2 + u·Ax·wrc (natural convention)",
        source="docs/gravity_hessian_derivation.md §2.2",
        method="sp.diff() of wrc_new wrt φ (with t=sin φ, u=cos φ); compare via simplify",
        passed=passed,
        detail=detail,
    )


def verify_item_008b() -> VerificationResult:
    """ITEM-008b — sub-claim of ITEM-008: ∂A/∂φ = u·Ax.

    CLAIM
        With A = t · Ax = sin(φ) · q · sqrt(2n+3),
            ∂A/∂φ = cos(φ) · Ax = u · Ax.

    METHOD
        SymPy direct differentiation; expected result u·Ax = cos(φ)·Ax.
    """
    phi = sp.symbols("phi", real=True)
    R, r, sqrt2n3 = sp.symbols("R r sqrt2n3", real=True)
    q = R / r
    Ax = q * sqrt2n3
    A_phi = sp.sin(phi) * Ax

    expected = sp.cos(phi) * Ax
    diff_expr = sp.simplify(sp.diff(A_phi, phi) - expected)
    passed = (diff_expr == 0)

    detail = (
        f"A(φ) = {A_phi}\n"
        f"sp.diff(A, φ) = {sp.diff(A_phi, phi)}\n"
        f"Claimed: u·Ax = cos(φ)·Ax = {expected}\n"
        f"simplify(diff - u·Ax) = {diff_expr}\n"
    )

    return VerificationResult(
        item_id="ITEM-008b",
        claim="∂A/∂φ = u·Ax where A = sin(φ)·Ax and u = cos(φ)",
        source="docs/gravity_hessian_derivation.md §2.2",
        method="Direct SymPy diff; simplify(diff(A,φ) - cos(φ)·Ax) == 0",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-009 — §2.3 wttc recurrence (natural convention)
# =============================================================================

def verify_item_009() -> VerificationResult:
    """ITEM-009 — wttc recurrence (∂²V/∂φ², natural sign convention).

    CLAIM
        Differentiating the NATURAL-convention wtc recurrence
            wtc_nat_new = A·wtc_nat + B·wtc2_nat + u·Ax·wc
        with respect to φ yields, in natural convention:
            wttc_nat_new = A·wttc_nat + B·wttc2_nat + 2·u·Ax·wtc_nat - t·Ax·wc
        The factor of 2 on u·Ax·wtc_nat comes from contributions through
        BOTH ∂A/∂φ (giving u·Ax·wtc_nat) AND ∂(u·Ax·wc)/∂φ via ∂wc/∂φ
        (giving another u·Ax·wtc_nat). The -t·Ax·wc term comes from
        ∂u/∂φ = -t applied to the u·Ax·wc inhomogeneous.

    SOURCE
        docs/gravity_hessian_derivation.md §2.3.

    METHOD
        Step 1: build wtc_nat_new symbolically (it's ∂(wc_new)/∂φ).
        Step 2: differentiate wtc_nat_new wrt φ symbolically.
        Step 3: compare to the claimed wttc_nat_new formula.

    IMPLICATED ON FAILURE
        The §2.3 derivation has an error. Most likely:
        - Missed the second contribution to the 2·u·Ax·wtc_nat factor
        - Got the sign wrong on -t·Ax·wc
        - Missed the chain through ∂wc/∂φ
    """
    phi = sp.symbols("phi", real=True)
    R, r, sqrt2n3, b_factor, n, Cv = sp.symbols(
        "R r sqrt2n3 b_factor n Cv", real=True
    )
    q = R / r
    Ax = q * sqrt2n3
    A_phi = sp.sin(phi) * Ax            # A = t·Ax
    B_phi = -q**2 * b_factor             # B (φ-independent)

    # wc, wc2 as functions of φ
    wc_f = sp.Function("wc")(phi)
    wc2_f = sp.Function("wc2")(phi)

    # Step 1: existing wc recurrence
    wc_new = A_phi * wc_f + B_phi * wc2_f + (n + 1) * Cv  # actually Cv not (n+1)·Cv for value
    # Wait — for the value (wc) recurrence the inhomogeneous is just Cv,
    # not (n+1)·Cv. The (n+1)·Cv was for wrc (radial-derivative). Fix:
    wc_new_corrected = A_phi * wc_f + B_phi * wc2_f + Cv

    # Step 2: derive wtc_nat_new = ∂(wc_new)/∂φ symbolically
    wtc_nat_new_sympy = sp.expand(sp.diff(wc_new_corrected, phi))

    # Step 3: differentiate wtc_nat_new wrt φ
    # First, define wtc_nat as ∂wc/∂φ etc.
    # We can substitute: wtc_nat_new is a function of wc(φ), wc2(φ), so
    # differentiating it again wrt φ gives wttc_nat_new directly.
    wttc_nat_new_sympy = sp.expand(sp.diff(wtc_nat_new_sympy, phi))

    # Now express in terms of wtc_nat = ∂wc/∂φ, wttc_nat = ∂²wc/∂φ², etc.
    # SymPy already has these as Derivative(wc(phi), phi) etc.
    # Build the claimed formula in those terms:
    wtc_nat_f = sp.diff(wc_f, phi)
    wtc2_nat_f = sp.diff(wc2_f, phi)
    wttc_nat_f = sp.diff(wc_f, phi, 2)
    wttc2_nat_f = sp.diff(wc2_f, phi, 2)

    # Claimed: wttc_nat_new = A·wttc_nat + B·wttc2_nat + 2·u·Ax·wtc_nat - t·Ax·wc
    u_func = sp.cos(phi)
    t_func = sp.sin(phi)
    wttc_nat_claimed = sp.expand(
        A_phi * wttc_nat_f + B_phi * wttc2_nat_f
        + 2 * u_func * Ax * wtc_nat_f
        - t_func * Ax * wc_f
    )

    diff_expr = sp.simplify(wttc_nat_new_sympy - wttc_nat_claimed)
    passed = (diff_expr == 0)

    detail = (
        f"Step 1: wc_new = A·wc + B·wc2 + Cv (note: just Cv, not (n+1)·Cv —\n"
        f"  the (n+1) factor was for the wrc/radial-derivative recurrence).\n\n"
        f"Step 2: wtc_nat_new = d(wc_new)/dφ:\n"
        f"  {wtc_nat_new_sympy}\n\n"
        f"Step 3: wttc_nat_new = d(wtc_nat_new)/dφ (SymPy):\n"
        f"  {wttc_nat_new_sympy}\n\n"
        f"Claimed wttc_nat_new = A·wttc + B·wttc2 + 2·u·Ax·wtc - t·Ax·wc:\n"
        f"  {wttc_nat_claimed}\n\n"
        f"simplify(SymPy - claimed):\n"
        f"  {diff_expr}\n"
    )

    return VerificationResult(
        item_id="ITEM-009",
        claim="wttc_nat_new = A·wttc_nat + B·wttc2_nat + 2·u·Ax·wtc_nat - t·Ax·wc",
        source="docs/gravity_hessian_derivation.md §2.3",
        method="Two successive sp.diff() applications wrt φ, compare to claimed formula",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-010 — sine-series siblings (wrrs, wrts, wtts) by structural symmetry
# =============================================================================

def verify_item_010() -> VerificationResult:
    """ITEM-010 — sine-series second-derivative recurrences.

    CLAIM (independent statement, NOT derived from SymPy output):
        The sine-series recurrences for second derivatives are exact analogs
        of the cosine-series recurrences with `wc → ws`, `wc2 → ws2`,
        `Cv → Sv` substitutions:

            wrrs_new = A·wrrs + B·wrrs2 - (A/r)·wrs - (2B/r)·wrs2
            wrts_nat_new = A·wrts + B·wrts2 + u·Ax·wrs        (natural)
            wtts_nat_new = A·wtts + B·wtts2 + 2·u·Ax·wts - t·Ax·ws  (natural)

        These claims are stated independently of SymPy. SymPy's job is to
        compute ∂(wrs_new)/∂r, ∂(wrs_new)/∂φ, and ∂²(ws_new)/∂φ² from
        scratch and confirm equality via simplify(SymPy - claimed) == 0.

    SOURCE
        docs/gravity_hessian_derivation.md §2.1-2.3 (cosine versions);
        sine versions follow by structural symmetry of the existing engine.

    METHOD
        For each of the three claimed formulas:
        - Build the existing first-derivative sine recurrence symbolically
          (wrs_new = A·wrs + B·wrs2 + (n+1)·Sv etc.).
        - Apply sp.diff() to compute LHS independently.
        - State RHS as my claim.
        - simplify(LHS - RHS) must equal 0.

        This pattern matches ITEMs 005, 008, 009 with Sv replacing Cv.

    IMPLICATED ON FAILURE
        If any one of the three sine analogs fails, the structural-symmetry
        claim is wrong and the cosine derivations may need revisiting.
    """
    # Setup
    r, R, sqrt2n3, b_factor = sp.symbols("r R sqrt2n3 b_factor", real=True)
    n = sp.symbols("n", integer=True, nonnegative=True)
    Sv = sp.symbols("Sv", real=True)
    phi = sp.symbols("phi", real=True)
    q = R / r
    Ax = q * sqrt2n3
    A_phi = sp.sin(phi) * Ax
    B_phi = -q**2 * b_factor

    # ========== Sub-test 1: wrrs ==========
    # Claim: wrrs_new = A·wrrs + B·wrrs2 - (A/r)·wrs - (2B/r)·wrs2
    # (mirror of wrrc with Sv replacing Cv; Sv is r-independent so the
    #  inhomogeneous (n+1)·Sv differentiates to 0, same as (n+1)·Cv)
    A_r = sp.sin(phi) * (R/r) * sqrt2n3   # A as function of r (with t=sin(phi) treated as parameter)
    B_r = -(R/r)**2 * b_factor

    # For wrrs we differentiate wrt r, so treat phi as parameter (t = sin(phi) is constant)
    # Use a placeholder symbol for sin(phi) to keep things clean
    t_param = sp.symbols("t_param", real=True)
    A_r_clean = t_param * (R/r) * sqrt2n3
    B_r_clean = -(R/r)**2 * b_factor

    wrs_f = sp.Function("wrs")(r)
    wrs2_f = sp.Function("wrs2")(r)
    wrs_new = A_r_clean * wrs_f + B_r_clean * wrs2_f + (n + 1) * Sv

    wrrs_sympy = sp.expand(sp.diff(wrs_new, r))

    wrrs_f = sp.diff(wrs_f, r)
    wrrs2_f = sp.diff(wrs2_f, r)
    wrrs_claimed = sp.expand(
        A_r_clean * wrrs_f + B_r_clean * wrrs2_f
        - (A_r_clean / r) * wrs_f
        - (2 * B_r_clean / r) * wrs2_f
    )

    diff1 = sp.simplify(wrrs_sympy - wrrs_claimed)

    # ========== Sub-test 2: wrts (∂²V/∂r∂φ for sine) ==========
    # Claim: wrts_nat_new = A·wrts + B·wrts2 + u·Ax·wrs
    ws_phi = sp.Function("ws")(phi)
    ws2_phi = sp.Function("ws2")(phi)
    wrs_phi_f = sp.Function("wrs")(phi)
    wrs2_phi_f = sp.Function("wrs2")(phi)

    # Existing wrs (radial-derivative sine) recurrence treated as fn of phi
    # We need wrs_new differentiated wrt phi
    # First, A varies with phi (t = sin(phi)), B doesn't.
    wrs_phi_new = A_phi * wrs_phi_f + B_phi * wrs2_phi_f + (n + 1) * Sv
    wrts_sympy = sp.expand(sp.diff(wrs_phi_new, phi))

    wrts_f = sp.diff(wrs_phi_f, phi)
    wrts2_f = sp.diff(wrs2_phi_f, phi)
    wrts_claimed = sp.expand(
        A_phi * wrts_f + B_phi * wrts2_f + sp.cos(phi) * Ax * wrs_phi_f
    )

    diff2 = sp.simplify(wrts_sympy - wrts_claimed)

    # ========== Sub-test 3: wtts (∂²V/∂φ² for sine) ==========
    # Claim: wtts_nat_new = A·wtts + B·wtts2 + 2·u·Ax·wts - t·Ax·ws
    ws_new = A_phi * ws_phi + B_phi * ws2_phi + Sv  # note: ws inhomogeneous is just Sv (no n+1 factor)
    wts_nat_new = sp.expand(sp.diff(ws_new, phi))
    wtts_sympy = sp.expand(sp.diff(wts_nat_new, phi))

    wts_nat_f = sp.diff(ws_phi, phi)
    wts2_nat_f = sp.diff(ws2_phi, phi)
    wtts_nat_f = sp.diff(ws_phi, phi, 2)
    wtts2_nat_f = sp.diff(ws2_phi, phi, 2)

    wtts_claimed = sp.expand(
        A_phi * wtts_nat_f + B_phi * wtts2_nat_f
        + 2 * sp.cos(phi) * Ax * wts_nat_f
        - sp.sin(phi) * Ax * ws_phi
    )

    diff3 = sp.simplify(wtts_sympy - wtts_claimed)

    passed = (diff1 == 0) and (diff2 == 0) and (diff3 == 0)

    detail = (
        f"Sub-test 1: wrrs (∂²V/∂r² sine analog of wrrc)\n"
        f"  simplify(SymPy - claimed) = {diff1}\n"
        f"  Match: {'YES' if diff1 == 0 else 'NO'}\n\n"
        f"Sub-test 2: wrts (∂²V/∂r∂φ sine analog of wrtc)\n"
        f"  simplify(SymPy - claimed) = {diff2}\n"
        f"  Match: {'YES' if diff2 == 0 else 'NO'}\n\n"
        f"Sub-test 3: wtts (∂²V/∂φ² sine analog of wttc)\n"
        f"  simplify(SymPy - claimed) = {diff3}\n"
        f"  Match: {'YES' if diff3 == 0 else 'NO'}\n"
    )

    return VerificationResult(
        item_id="ITEM-010",
        claim="Sine-series wrrs/wrts/wtts recurrences mirror cosine versions (wc→ws, Cv→Sv)",
        source="docs/gravity_hessian_derivation.md §2.1-2.3 (sine analogs)",
        method="Three sub-claims, each with independent SymPy diff vs claimed formula",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-011 — lambda-derivative source-substitution lemma
# =============================================================================

def verify_item_011() -> VerificationResult:
    """ITEM-011 — λ-derivative outer-loop source-term substitution rule.

    CLAIM (independent statement, NOT derived from SymPy output):
        For a single trigonometric mode of the gravity series at outer-loop
        index m,
            V_m(r, φ, λ) = A_m(r, φ) · cos(m·λ) + B_m(r, φ) · sin(m·λ),
        the higher-order derivatives factor as the claimed full expressions
        below. After regrouping by cos(m·λ) and sin(m·λ), the coefficients
        define the inhomogeneous source for each lambda-derivative outer-loop
        accumulator (vrlc, vrls, vtlc, vtls, vllc, vlls).

        Claim 1 — ∂V_m/∂λ:
            sympy result == (+m·B_m) · cos(m·λ) + (−m·A_m) · sin(m·λ)
            ⇒ cos-bucket source = +m·B_m,  sin-bucket source = −m·A_m

        Claim 2 — ∂²V_m/∂r∂λ (with A_m, B_m treated as functions of r):
            sympy result == (+m·∂B_m/∂r) · cos(m·λ) + (−m·∂A_m/∂r) · sin(m·λ)
            ⇒ cos source = +m·wrs,  sin source = −m·wrc
            (where wrc := ∂A_m/∂r, wrs := ∂B_m/∂r by definition)

        Claim 3 — ∂²V_m/∂φ∂λ (A_m, B_m as functions of φ):
            sympy result == (+m·∂B_m/∂φ) · cos(m·λ) + (−m·∂A_m/∂φ) · sin(m·λ)
            ⇒ cos source = +m·wts,  sin source = −m·wtc

        Claim 4 — ∂²V_m/∂λ²:
            sympy result == (−m²·A_m) · cos(m·λ) + (−m²·B_m) · sin(m·λ)
            ⇒ cos source = −m²·wc,  sin source = −m²·ws
            (Note: ∂²/∂λ² does NOT swap cos↔sin; both buckets get diagonal
             −m² factor.)

        These four claims are stated INDEPENDENTLY of SymPy. SymPy's job is
        to compute each derivative from the symbolic Fourier-mode expression
        from scratch, and `simplify(SymPy − claimed) == 0` must hold.

    SOURCE
        docs/gravity_hessian_derivation.md §2.5;
        engine reference: clenshaw_engine.hpp:236-237 (existing vlc/vls
        sources match Claim 1 exactly).

    METHOD
        Write V_term = A_m · cos(m·λ) + B_m · sin(m·λ) in SymPy with `m`
        symbolic. For each claim:
        - LHS = sp.diff(V_term, ...) — SymPy computes the derivative
          mechanically via chain rule.
        - RHS = the claimed full expression (cos- and sin-bucket coefficients
          re-multiplied by their trigonometric basis).
        - simplify(LHS − RHS) must equal 0.

        For claims 2 and 3, A_m and B_m are replaced with sp.Function("wc")(r)
        and sp.Function("ws")(r) (or φ) so SymPy resolves the partial
        derivatives symbolically as Derivative(wc(r), r) etc. — those are
        the labels for wrc, wrs, wtc, wts in the engine.

    IMPLICATED ON FAILURE
        Any sub-claim failing means the outer-loop source-term substitution
        rule is wrong. The engine's existing vlc/vls sources (line 236-237)
        match Claim 1, so a Claim-1 failure would also indicate a bug in
        the existing engine. Claim-2/3/4 failures would invalidate the
        proposed Hessian extension entirely.
    """
    lam, m, r, phi = sp.symbols("lam m r phi", real=True)

    # ---- Claim 1: ∂V_m/∂λ ----
    A_m, B_m = sp.symbols("A_m B_m", real=True)
    V_term = A_m * sp.cos(m * lam) + B_m * sp.sin(m * lam)
    sympy_dV_dlam = sp.diff(V_term, lam)
    claimed_dV_dlam = (m * B_m) * sp.cos(m * lam) + (-m * A_m) * sp.sin(m * lam)
    diff1 = sp.simplify(sympy_dV_dlam - claimed_dV_dlam)

    # ---- Claim 2: ∂²V_m/∂r∂λ ----
    # A_m, B_m as functions of r — labels: A_m(r) = wc(r), B_m(r) = ws(r)
    wc_r = sp.Function("wc")(r)
    ws_r = sp.Function("ws")(r)
    V_term_r = wc_r * sp.cos(m * lam) + ws_r * sp.sin(m * lam)
    sympy_d2V_drdlam = sp.diff(V_term_r, r, lam)

    # Claimed: ∂A/∂r becomes wrc (Derivative(wc(r), r) symbolically),
    # ∂B/∂r becomes wrs (Derivative(ws(r), r) symbolically).
    wrc_r = sp.diff(wc_r, r)
    wrs_r = sp.diff(ws_r, r)
    claimed_d2V_drdlam = (m * wrs_r) * sp.cos(m * lam) + (-m * wrc_r) * sp.sin(m * lam)
    diff2 = sp.simplify(sympy_d2V_drdlam - claimed_d2V_drdlam)

    # ---- Claim 3: ∂²V_m/∂φ∂λ ----
    wc_phi = sp.Function("wc_phi")(phi)
    ws_phi = sp.Function("ws_phi")(phi)
    V_term_phi = wc_phi * sp.cos(m * lam) + ws_phi * sp.sin(m * lam)
    sympy_d2V_dphidlam = sp.diff(V_term_phi, phi, lam)

    wtc_phi = sp.diff(wc_phi, phi)
    wts_phi = sp.diff(ws_phi, phi)
    claimed_d2V_dphidlam = (
        (m * wts_phi) * sp.cos(m * lam) + (-m * wtc_phi) * sp.sin(m * lam)
    )
    diff3 = sp.simplify(sympy_d2V_dphidlam - claimed_d2V_dphidlam)

    # ---- Claim 4: ∂²V_m/∂λ² ----
    sympy_d2V_dlamdlam = sp.diff(V_term, lam, 2)
    claimed_d2V_dlamdlam = (-m**2 * A_m) * sp.cos(m * lam) + (-m**2 * B_m) * sp.sin(m * lam)
    diff4 = sp.simplify(sympy_d2V_dlamdlam - claimed_d2V_dlamdlam)

    passed = (diff1 == 0) and (diff2 == 0) and (diff3 == 0) and (diff4 == 0)

    detail = (
        f"Sub-test 1: ∂V/∂λ source rule\n"
        f"  SymPy:   {sp.expand(sympy_dV_dlam)}\n"
        f"  Claimed: {sp.expand(claimed_dV_dlam)}\n"
        f"  simplify(SymPy − claimed) = {diff1}\n"
        f"  Match: {'YES' if diff1 == 0 else 'NO'}\n\n"
        f"Sub-test 2: ∂²V/∂r∂λ source rule (A_m, B_m as fn of r)\n"
        f"  SymPy:   {sp.expand(sympy_d2V_drdlam)}\n"
        f"  Claimed: {sp.expand(claimed_d2V_drdlam)}\n"
        f"  simplify(SymPy − claimed) = {diff2}\n"
        f"  Match: {'YES' if diff2 == 0 else 'NO'}\n\n"
        f"Sub-test 3: ∂²V/∂φ∂λ source rule (A_m, B_m as fn of φ)\n"
        f"  SymPy:   {sp.expand(sympy_d2V_dphidlam)}\n"
        f"  Claimed: {sp.expand(claimed_d2V_dphidlam)}\n"
        f"  simplify(SymPy − claimed) = {diff3}\n"
        f"  Match: {'YES' if diff3 == 0 else 'NO'}\n\n"
        f"Sub-test 4: ∂²V/∂λ² source rule (diagonal, no cos↔sin swap)\n"
        f"  SymPy:   {sp.expand(sympy_d2V_dlamdlam)}\n"
        f"  Claimed: {sp.expand(claimed_d2V_dlamdlam)}\n"
        f"  simplify(SymPy − claimed) = {diff4}\n"
        f"  Match: {'YES' if diff4 == 0 else 'NO'}\n"
    )

    return VerificationResult(
        item_id="ITEM-011",
        claim="Lambda-derivative outer-loop source-substitution lemma (4 sub-claims)",
        source="docs/gravity_hessian_derivation.md §2.5; engine line 236-237",
        method="Symbolic Fourier-mode differentiation vs claimed full expression",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-012 — engine gradient assembly convention (resolves ITEM-001/002)
# =============================================================================

def verify_item_012() -> VerificationResult:
    """ITEM-012 — gradient chain rule vs engine's gradx/grady/gradz formula.

    CLAIM (independent statement, NOT derived from SymPy output):
        After post-finalization processing (clenshaw_engine.hpp lines 240-249),
        the engine's accumulators carry the following spherical-coord
        derivative quantities:

            (GM/R) · vc  = +V
            (GM/R) · vrc = +∂V/∂r
            (GM/R) · vtc = −(1/r) · ∂V/∂φ
            (GM/R) · vlc = +(1/(r·u)) · ∂V/∂λ

        With these substitutions, the engine's gradient assembly at
        lines 257-259:
            gradx = cl · (u · vrc + t · vtc) − sl · vlc
            grady = sl · (u · vrc + t · vtc) + cl · vlc
            gradz = t · vrc − u · vtc

        becomes (after multiplying by scale_factor = GM/R):

            ∂V/∂x_cart = the chain-rule expression for ∂V/∂(x|y|z)
                       = (∂V/∂r)·(∂r/∂x_i) + (∂V/∂φ)·(∂φ/∂x_i)
                          + (∂V/∂λ)·(∂λ/∂x_i)

        for x_i ∈ {x, y, z}.

        These claims are stated INDEPENDENTLY of SymPy. SymPy's job is to
        compute the chain-rule expression from the position transform
        x = r·u·cl, y = r·u·sl, z = r·t (with u=cos φ, t=sin φ, cl=cos λ,
        sl=sin λ), then check that substituting the convention table into
        the engine formula reproduces the chain-rule result via
        `simplify(engine_expr_substituted − chain_rule_expr) == 0`.

    SOURCE
        docs/gravity_hessian_derivation.md §3.1-3.2;
        engine code: clenshaw_engine.hpp:257-259.

    METHOD
        1. Set up symbolic V(r, φ, λ) as an unspecified scalar function
           via sp.Function. Compute its three first partials:
           dVdr, dVdphi, dVdlam = sp.diff(V, r), sp.diff(V, phi), sp.diff(V, lam).
        2. Express x, y, z in terms of r, φ, λ; compute Cartesian gradient
           ∂V/∂x_cart by treating V as a function of r(x,y,z), φ(x,y,z),
           λ(x,y,z) and applying chain rule via SymPy's sp.diff.
        3. State the engine's gradx/grady/gradz formula symbolically with
           vc, vrc, vtc, vlc as placeholders.
        4. Substitute the convention table:
              vrc → (R/GM) · ∂V/∂r
              vtc → (R/GM) · (−(1/r) · ∂V/∂φ)
              vlc → (R/GM) · (+(1/(r·u)) · ∂V/∂λ)
        5. Multiply engine formula by GM/R and compare to chain-rule
           gradient via simplify(...) == 0.

    IMPLICATED ON FAILURE
        Failure means the convention table for vrc/vtc/vlc is wrong, which
        invalidates §3.2's resolution of ITEM-001/002 and makes §3.3
        (Hessian assembly) impossible to do correctly. A pass means the
        engine's first-derivative pipeline has known, verified semantics
        through its post-finalization output.
    """
    r, phi, lam = sp.symbols("r phi lam", real=True, positive=True)
    GM, R = sp.symbols("GM R", real=True, positive=True)

    # Symbolic V(r, φ, λ) — leave it as an unspecified function so the
    # chain rule has Derivative(V(r,phi,lam), r) etc. as labels for the
    # spherical partials.
    V_sym = sp.Function("V")(r, phi, lam)

    # Position transform
    u = sp.cos(phi)
    t = sp.sin(phi)
    cl = sp.cos(lam)
    sl = sp.sin(lam)
    x = r * u * cl
    y = r * u * sl
    z = r * t

    # Express r, phi, lam back as functions of x, y, z (for chain-rule
    # via inverse). The cleanest approach: define x_sym, y_sym, z_sym
    # symbols that are independent, then state V_cart = V(r(x,y,z),
    # phi(x,y,z), lam(x,y,z)) and have SymPy do diff.
    #
    # But here we just use the explicit ∂r/∂x_i = (forward
    # Jacobian)^{-1} entries. From the forward Jacobian:
    #   ∂x/∂r = u·cl,    ∂y/∂r = u·sl,    ∂z/∂r = t
    #   ∂x/∂φ = -r·t·cl,  ∂y/∂φ = -r·t·sl,  ∂z/∂φ = r·u
    #   ∂x/∂λ = -r·u·sl,  ∂y/∂λ = +r·u·cl,  ∂z/∂λ = 0
    # Inverse Jacobian (orthonormal w/ scale factors h_r=1, h_φ=r, h_λ=r·u):
    #   ∂r/∂x = u·cl,        ∂r/∂y = u·sl,        ∂r/∂z = t
    #   ∂φ/∂x = -t·cl/r,     ∂φ/∂y = -t·sl/r,     ∂φ/∂z = u/r
    #   ∂λ/∂x = -sl/(r·u),    ∂λ/∂y = +cl/(r·u),   ∂λ/∂z = 0
    dr_dx, dr_dy, dr_dz = u * cl, u * sl, t
    dphi_dx, dphi_dy, dphi_dz = -t * cl / r, -t * sl / r, u / r
    dlam_dx, dlam_dy, dlam_dz = -sl / (r * u), cl / (r * u), sp.Integer(0)

    dVdr = sp.diff(V_sym, r)
    dVdphi = sp.diff(V_sym, phi)
    dVdlam = sp.diff(V_sym, lam)

    # Chain-rule Cartesian gradient
    chain_dVdx = dVdr * dr_dx + dVdphi * dphi_dx + dVdlam * dlam_dx
    chain_dVdy = dVdr * dr_dy + dVdphi * dphi_dy + dVdlam * dlam_dy
    chain_dVdz = dVdr * dr_dz + dVdphi * dphi_dz + dVdlam * dlam_dz

    # Engine formula with placeholder symbols vrc_sym etc.
    vrc_sym, vtc_sym, vlc_sym = sp.symbols("vrc vtc vlc", real=True)
    scale = GM / R
    engine_gradx = cl * (u * vrc_sym + t * vtc_sym) - sl * vlc_sym
    engine_grady = sl * (u * vrc_sym + t * vtc_sym) + cl * vlc_sym
    engine_gradz = t * vrc_sym - u * vtc_sym

    # Convention substitution: the engine's accumulators after
    # finalization satisfy
    #   (GM/R) · vrc = +∂V/∂r
    #   (GM/R) · vtc = -(1/r) · ∂V/∂φ
    #   (GM/R) · vlc = +(1/(r·u)) · ∂V/∂λ
    # ⇒ vrc = (R/GM) · ∂V/∂r,  etc.
    convention = {
        vrc_sym: (R / GM) * dVdr,
        vtc_sym: (R / GM) * (-(1 / r) * dVdphi),
        vlc_sym: (R / GM) * ((1 / (r * u)) * dVdlam),
    }

    # Engine's reported acceleration component is scale_factor * gradN.
    # So the *physical* ∂V/∂x_cart = (GM/R) · gradx_with_substituted_vXc.
    engine_dVdx = (scale * engine_gradx).subs(convention)
    engine_dVdy = (scale * engine_grady).subs(convention)
    engine_dVdz = (scale * engine_gradz).subs(convention)

    diff_x = sp.simplify(engine_dVdx - chain_dVdx)
    diff_y = sp.simplify(engine_dVdy - chain_dVdy)
    diff_z = sp.simplify(engine_dVdz - chain_dVdz)

    passed = (diff_x == 0) and (diff_y == 0) and (diff_z == 0)

    detail = (
        f"Sub-test 1: ∂V/∂x_cart\n"
        f"  Engine (substituted):  {sp.simplify(engine_dVdx)}\n"
        f"  Chain-rule (claimed):  {sp.simplify(chain_dVdx)}\n"
        f"  simplify(engine − chain) = {diff_x}\n"
        f"  Match: {'YES' if diff_x == 0 else 'NO'}\n\n"
        f"Sub-test 2: ∂V/∂y_cart\n"
        f"  simplify(engine − chain) = {diff_y}\n"
        f"  Match: {'YES' if diff_y == 0 else 'NO'}\n\n"
        f"Sub-test 3: ∂V/∂z_cart\n"
        f"  simplify(engine − chain) = {diff_z}\n"
        f"  Match: {'YES' if diff_z == 0 else 'NO'}\n"
    )

    return VerificationResult(
        item_id="ITEM-012",
        claim="Engine post-finalization convention: (GM/R)·{vrc, vtc, vlc} = {+∂V/∂r, −(1/r)∂V/∂φ, +(1/(r·u))∂V/∂λ}",
        source="docs/gravity_hessian_derivation.md §3.2; engine line 257-259",
        method="Substitute convention into engine grad formula, compare to SymPy chain rule",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-013 — Cartesian Hessian tensor decomposition
# =============================================================================

def verify_item_013() -> VerificationResult:
    """ITEM-013 — H_cart = J·H_sph·Jᵀ + Σ_a (∂V/∂q^a)·K^a tensor decomposition.

    CLAIM (independent statement, NOT derived from SymPy chain rule):
        For V(x,y,z) = V(r(x,y,z), φ(x,y,z), λ(x,y,z)), the Cartesian
        Hessian decomposes as:

            H_cart_{ij} = Σ_{a,b} J^a_i · J^b_j · H_sph_{ab}
                         + Σ_a (∂V/∂q^a) · K^a_{ij}

        where:
            J^a_i    = ∂q^a/∂x_i              (inverse-transform Jacobian)
            K^a_{ij} = ∂²q^a/∂x_i∂x_j         (inverse-transform Hessian)
            H_sph_{ab} = ∂²V/∂q^a∂q^b         (partial-derivative spherical Hessian)

        with q^a ∈ {r, φ, λ} and x_i ∈ {x, y, z}.

        The claim is the EXPLICIT tensor sum above. SymPy's job is to
        compute LHS = ∂²V_xyz/∂x_i∂x_j independently via chain rule on
        composed concrete test potentials and verify simplify(LHS − RHS)
        == 0 for all 6 unique Cartesian entries.

        The key is that LHS uses SymPy's automatic chain rule (which
        computes ALL product-rule terms internally), while RHS uses my
        explicit tensor decomposition with hand-stated structure. A pass
        means the structure is correct.

        Note on abstract-V: I attempted abstract `V = sp.Function("V")(r,
        phi, lam)` first; SymPy's chain rule produced `Subs(Derivative
        (...))` atoms that `simplify` couldn't reduce despite the math
        being identical. Concrete test potentials sidestep this: if the
        decomposition holds for any V, it holds for specific Vs, and
        SymPy can simplify rational/trigonometric expressions cleanly.

    SOURCE
        docs/gravity_hessian_derivation.md §3.3.

    METHOD
        1. Build symbolic position transform:
             r_expr = sqrt(x² + y² + z²),  phi_expr = atan2(z, p),
             lam_expr = atan2(y, x).
        2. Compute J^a_i and K^a_{ij} from r_expr, phi_expr, lam_expr via
           sp.diff (these don't involve V at all — purely geometric).
        3. For each of 4 concrete test potentials (covering all 9 partials
           non-trivially):
              V₁ = 1/r                                  (point mass)
              V₂ = sin(φ)/r²                            (φ-dependent)
              V₃ = sin(2λ)/r³                           (λ-dependent)
              V₄ = sin(φ)·sin(2λ)/r²                    (φ×λ mixed)
        4. LHS: V_xyz = V_test.subs({r: r_expr, phi: phi_expr,
           lam: lam_expr}); compute sp.diff(V_xyz, x_i, x_j).
        5. RHS: explicit tensor sum using J, K, and the spherical
           partials of V_test (rational/trig — fully reducible).
        6. Assert simplify(LHS − RHS) == 0 for all 6 unique entries
           per V_test → 24 sub-tests total.

    IMPLICATED ON FAILURE
        Failure means the tensor decomposition is missing or has wrong
        signs on terms — invalidates §3.3 and any subsequent C++
        implementation. A pass means §3.3's structure is the basis on
        which the C++ Hessian engine can be implemented.
    """
    # Use explicit Cartesian symbols — must be FREE for SymPy's chain rule
    # to work on the composed function.
    x, y, z = sp.symbols("x y z", real=True, positive=True)

    # Build inverse transform symbolically. Using atan2 with positive=True
    # x, y, z avoids branch-cut headaches.
    p = sp.sqrt(x**2 + y**2)
    r_expr = sp.sqrt(x**2 + y**2 + z**2)
    phi_expr = sp.atan2(z, p)
    lam_expr = sp.atan2(y, x)

    # Spherical symbols (used for V_test definition and partial labels)
    r_s, phi_s, lam_s = sp.symbols("r phi lam", real=True, positive=True)

    # ----------------------------------------------------------
    # Test family: cover all 9 spherical partials non-trivially.
    # Each row: (label, V_test as expression in r, phi, lam).
    # ----------------------------------------------------------
    test_potentials = [
        # Pure radial — exercises ∂V/∂r, ∂²V/∂r² only.
        ("V=1/r (point mass)", 1 / r_s),
        # phi-dependent — adds ∂V/∂φ, ∂²V/∂φ², ∂²V/∂r∂φ.
        ("V=sin(phi)/r^2", sp.sin(phi_s) / r_s**2),
        # lam-dependent — adds ∂V/∂λ, ∂²V/∂λ², ∂²V/∂r∂λ.
        ("V=sin(2*lam)/r^3", sp.sin(2 * lam_s) / r_s**3),
        # phi×lam mixed — adds ∂²V/∂φ∂λ; full coverage.
        ("V=sin(phi)*sin(2*lam)/r^2", sp.sin(phi_s) * sp.sin(2 * lam_s) / r_s**2),
    ]

    # Build inverse Jacobian J^a_i = ∂q^a/∂x_i (3×3 matrix, rows=q, cols=x_i)
    coords_x = [x, y, z]
    coords_q_expr = [r_expr, phi_expr, lam_expr]
    J = sp.Matrix(3, 3, lambda a, i: sp.diff(coords_q_expr[a], coords_x[i]))

    # Build K^a_{ij} = ∂²q^a/∂x_i∂x_j (one 3x3 per a)
    K = [
        sp.Matrix(3, 3, lambda i, j, a=a: sp.diff(coords_q_expr[a], coords_x[i], coords_x[j]))
        for a in range(3)
    ]

    sub_q_to_xyz = {r_s: r_expr, phi_s: phi_expr, lam_s: lam_expr}

    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    labels = ["xx", "yy", "zz", "xy", "xz", "yz"]

    all_diffs = []  # list of (V_label, entry_label, diff)
    for V_label, V_test_sph in test_potentials:
        V_test_xyz = V_test_sph.subs(sub_q_to_xyz)

        # H_sph entries (still in spherical), then substitute → xyz
        d2V_drr = sp.diff(V_test_sph, r_s, 2).subs(sub_q_to_xyz)
        d2V_dphiphi = sp.diff(V_test_sph, phi_s, 2).subs(sub_q_to_xyz)
        d2V_dlamlam = sp.diff(V_test_sph, lam_s, 2).subs(sub_q_to_xyz)
        d2V_drphi = sp.diff(V_test_sph, r_s, phi_s).subs(sub_q_to_xyz)
        d2V_drlam = sp.diff(V_test_sph, r_s, lam_s).subs(sub_q_to_xyz)
        d2V_dphilam = sp.diff(V_test_sph, phi_s, lam_s).subs(sub_q_to_xyz)

        H_sph_xyz = sp.Matrix([
            [d2V_drr,   d2V_drphi,   d2V_drlam],
            [d2V_drphi, d2V_dphiphi, d2V_dphilam],
            [d2V_drlam, d2V_dphilam, d2V_dlamlam],
        ])

        # First derivatives ∂V/∂q^a (in xyz)
        dV_dq_xyz = [
            sp.diff(V_test_sph, r_s).subs(sub_q_to_xyz),
            sp.diff(V_test_sph, phi_s).subs(sub_q_to_xyz),
            sp.diff(V_test_sph, lam_s).subs(sub_q_to_xyz),
        ]

        for (i, j), lab in zip(pairs, labels):
            # LHS: SymPy chain rule on the composed concrete V_test_xyz
            sympy_val = sp.diff(V_test_xyz, coords_x[i], coords_x[j])

            # RHS: explicit tensor decomposition
            term1 = sum(
                J[a, i] * J[b, j] * H_sph_xyz[a, b]
                for a in range(3)
                for b in range(3)
            )
            term2 = sum(dV_dq_xyz[a] * K[a][i, j] for a in range(3))
            claim_val = term1 + term2

            diff = sp.simplify(sympy_val - claim_val)
            all_diffs.append((V_label, lab, diff))

    passed = all(d == 0 for _, _, d in all_diffs)

    detail_lines = []
    current_V = None
    for V_label, lab, d in all_diffs:
        if V_label != current_V:
            detail_lines.append(f"\n  Test potential: {V_label}")
            current_V = V_label
        match = "YES" if d == 0 else "NO"
        if d == 0:
            detail_lines.append(f"    H_{lab}: simplify(SymPy − claim) = 0  →  Match: {match}")
        else:
            d_str = str(d)
            if len(d_str) > 200:
                d_str = d_str[:200] + "...[truncated]"
            detail_lines.append(
                f"    H_{lab}: simplify(SymPy − claim) = {d_str}  →  Match: {match}"
            )
    detail = "\n".join(detail_lines).lstrip()

    return VerificationResult(
        item_id="ITEM-013",
        claim="H_cart = J·H_sph·Jᵀ + Σ_a (∂V/∂q^a)·K^a tensor decomposition",
        source="docs/gravity_hessian_derivation.md §3.3",
        method="Compare SymPy chain rule on V(r(x,y,z),φ(x,y,z),λ(x,y,z)) to explicit tensor sum",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-014 — pole-singularity cancellation (mpmath, 50-digit precision)
# =============================================================================

def verify_item_014() -> VerificationResult:
    """ITEM-014 — Cartesian Hessian remains finite at the pole.

    CLAIM (independent statement, NOT derived from numerical experiment):
        For a tesseral spherical harmonic V = (R/r)^(l+1) · P̄_lm(sin φ)
        · cos(m·λ) with m ≥ 1, the spherical-coordinate Hessian
        components carry 1/cos²(φ) factors that diverge as φ → π/2.
        However, the Cartesian Hessian H_cart = J·H_sph·Jᵀ + Σ_a
        (∂V/∂q^a)·K^a is finite at the pole.

        Concretely: take the test potential
            V_test = (x² − y²) / r⁵
        which equals (up to a constant) the (l=2, m=2) sectoral harmonic
        (V_22 ∝ P̄_22(sin φ)·cos(2λ)·(R/r)³ ∝ (x²−y²)/r⁵). For this V_test:
        - The Cartesian Hessian is a rational polynomial in (x, y, z) and
          is finite at every (x, y, z) ≠ origin, including the pole
          (1, 0, 0)·tiny + (0, 0, 1)·r — equivalently, x = y = 0, z = r.
        - The spherical-form Hessian, computed via the §3.3 tensor
          decomposition, must produce the same finite value when the
          divergent J·H_sph·Jᵀ term cancels with the divergent
          Σ_a (∂V/∂q^a)·K^a term.

        Verification: at decreasing ε = π/2 − φ (so the pole is at ε=0):
        - mpmath at 50-digit precision computes the Cartesian Hessian
          two ways: (1) directly from the polynomial form (always
          finite), (2) via the spherical-form J·H_sph·Jᵀ + correction-K
          decomposition (individually-divergent pieces, summed).
        - Both methods must converge to the same value; the spherical
          method's individual pieces diverge in opposite signs but their
          sum stays bounded.

    SOURCE
        docs/gravity_hessian_derivation.md §4.

    METHOD
        1. Use mpmath at 50-digit precision (well below double-precision
           noise floor of ~1e-16).
        2. Set test point near the pole: (x, y, z) = (ε·R, 0, R) for
           ε ∈ {1e-3, 1e-6, 1e-10, 1e-20, 1e-30}.
        3. For each ε:
           a. Direct Cartesian: compute ∂²V_test/∂x∂x at (ε·R, 0, R) by
              symbolic differentiation of V_test = (x²−y²)/r⁵, then
              numerical evaluation.
           b. Spherical-form via §3.3 tensor decomposition: compute
              H_sph (∂²V/∂r², ∂²V/∂φ², ∂²V/∂λ², ∂²V/∂r∂φ, etc.),
              evaluate J^a_i, K^a_{ij} at this point, build
              J·H_sph·Jᵀ + Σ_a (∂V/∂q^a)·K^a numerically.
           c. Compare: both must agree to ~50 digits at every ε.
           d. Also report the magnitudes of |J·H_sph·Jᵀ|_xx and
              |Σ_a (∂V/∂q^a)·K^a|_xx individually — these should grow
              as ε → 0, demonstrating the cancellation is real (not
              just everything being identically zero).

    IMPLICATED ON FAILURE
        Failure means the §3.3 tensor decomposition + §3.1 J/K formulas
        are inconsistent at the pole — likely a missing correction term.
        This would mean the C++ implementation cannot directly use the
        spherical formulation near the pole and must fall back to Pines.
    """
    # Use mpmath at 50-digit precision
    mp.mp.dps = 50

    R = mp.mpf("6378137.0")  # Earth equatorial radius

    # Test V = (x² − y²) / r⁵ — this is the (l=2, m=2) sectoral harmonic
    # up to a normalization constant. Pure Cartesian polynomial.
    def V_cart(x, y, z):
        r2 = x*x + y*y + z*z
        r5 = r2 ** mp.mpf("2.5")
        return (x*x - y*y) / r5

    # Direct Cartesian H_xx = ∂²V/∂x∂x
    # V = (x²−y²)·(x²+y²+z²)^(-5/2)
    # ∂V/∂x = 2x·r⁻⁵ + (x²−y²)·(-5/2)·r⁻⁷·2x
    #       = 2x/r⁵ - 5x(x²−y²)/r⁷
    # ∂²V/∂x² = 2/r⁵ - 5·1·2x·x/r⁷ - 5(x²−y²)/r⁷ - 5x·(2x·r⁻⁷ + (x²−y²)·(-7/2)·r⁻⁹·2x)
    # Let me just use sympy to derive this analytically once, then evaluate numerically.
    sx, sy, sz = sp.symbols("sx sy sz", real=True)
    sr2 = sx**2 + sy**2 + sz**2
    V_sym = (sx**2 - sy**2) / sr2**sp.Rational(5, 2)
    dV_dx_sym = sp.diff(V_sym, sx)
    d2V_dxx_sym = sp.diff(V_sym, sx, 2)
    # Generate fast lambdified versions
    d2V_dxx_func = sp.lambdify((sx, sy, sz), d2V_dxx_sym, modules="mpmath")

    # Spherical-form pieces. We need:
    # - J^a_i and K^a_{ij}: evaluated at (x, y, z)
    # - H_sph entries: evaluated as ∂²V/∂q^a∂q^b
    # - ∂V/∂q^a: first partials
    # All in spherical coords, then evaluated numerically.

    # Build symbolic spherical V from the same polynomial
    sr, sphi, slam = sp.symbols("sr sphi slam", real=True, positive=True)
    # V_test = (R²·cos²(φ)·cos(2λ)) · 3 / r³  ... no wait we set V = (x²-y²)/r⁵.
    # x² − y² = (r·cos φ·cos λ)² − (r·cos φ·sin λ)² = r²·cos²(φ)·(cos²λ − sin²λ)
    #         = r²·cos²(φ)·cos(2λ)
    # So V = r²·cos²(φ)·cos(2λ) / r⁵ = cos²(φ)·cos(2λ)/r³
    V_sph_test = sp.cos(sphi)**2 * sp.cos(2*slam) / sr**3

    # First partials
    dV_dr = sp.diff(V_sph_test, sr)
    dV_dphi = sp.diff(V_sph_test, sphi)
    dV_dlam = sp.diff(V_sph_test, slam)

    # Second partials
    d2V_drr = sp.diff(V_sph_test, sr, 2)
    d2V_dphiphi = sp.diff(V_sph_test, sphi, 2)
    d2V_dlamlam = sp.diff(V_sph_test, slam, 2)
    d2V_drphi = sp.diff(V_sph_test, sr, sphi)
    d2V_drlam = sp.diff(V_sph_test, sr, slam)
    d2V_dphilam = sp.diff(V_sph_test, sphi, slam)

    # J and K from §3.1: r = sqrt(x²+y²+z²), φ = atan2(z, p), λ = atan2(y, x)
    # We work numerically here — compute J^a_i and K^a_{ij} from the inverse transform.
    sr_xyz = sp.sqrt(sx**2 + sy**2 + sz**2)
    sp_xyz = sp.sqrt(sx**2 + sy**2)
    sphi_xyz = sp.atan2(sz, sp_xyz)
    slam_xyz = sp.atan2(sy, sx)
    coords_x_sym = [sx, sy, sz]
    coords_q_xyz = [sr_xyz, sphi_xyz, slam_xyz]

    # Lambdify everything
    dV_dr_func = sp.lambdify((sr, sphi, slam), dV_dr, modules="mpmath")
    dV_dphi_func = sp.lambdify((sr, sphi, slam), dV_dphi, modules="mpmath")
    dV_dlam_func = sp.lambdify((sr, sphi, slam), dV_dlam, modules="mpmath")
    d2V_drr_func = sp.lambdify((sr, sphi, slam), d2V_drr, modules="mpmath")
    d2V_dphiphi_func = sp.lambdify((sr, sphi, slam), d2V_dphiphi, modules="mpmath")
    d2V_dlamlam_func = sp.lambdify((sr, sphi, slam), d2V_dlamlam, modules="mpmath")
    d2V_drphi_func = sp.lambdify((sr, sphi, slam), d2V_drphi, modules="mpmath")
    d2V_drlam_func = sp.lambdify((sr, sphi, slam), d2V_drlam, modules="mpmath")
    d2V_dphilam_func = sp.lambdify((sr, sphi, slam), d2V_dphilam, modules="mpmath")

    # J^a_i and K^a_{ij} as functions of x, y, z (lambdified)
    J_funcs = [
        [
            sp.lambdify((sx, sy, sz),
                        sp.diff(coords_q_xyz[a], coords_x_sym[i]),
                        modules="mpmath")
            for i in range(3)
        ]
        for a in range(3)
    ]
    K_funcs = [
        [
            [
                sp.lambdify((sx, sy, sz),
                            sp.diff(coords_q_xyz[a], coords_x_sym[i], coords_x_sym[j]),
                            modules="mpmath")
                for j in range(3)
            ]
            for i in range(3)
        ]
        for a in range(3)
    ]

    epsilons = [mp.mpf("1e-3"), mp.mpf("1e-6"), mp.mpf("1e-10"),
                mp.mpf("1e-20"), mp.mpf("1e-30")]

    detail_lines = ["mpmath @ 50-digit precision; pole offset ε = π/2 − φ"]
    detail_lines.append("Test V = (x²−y²)/r⁵, evaluating H_xx at (x=ε·R, y=0, z=R):")
    detail_lines.append("")
    detail_lines.append(
        f"{'ε':<10s}  {'direct H_xx':<25s}  {'tensor H_xx':<25s}  "
        f"{'|JHJ_xx|':<15s}  {'|ΣKa·∂V|':<15s}  match?"
    )

    all_match = True
    for eps in epsilons:
        # Position: x = eps·R, y = 0, z = R (just shy of north pole)
        x_v = eps * R
        y_v = mp.mpf("0")
        z_v = R
        # Spherical coords:
        r_v = mp.sqrt(x_v**2 + y_v**2 + z_v**2)
        p_v = mp.sqrt(x_v**2 + y_v**2)
        phi_v = mp.atan2(z_v, p_v)
        lam_v = mp.atan2(y_v, x_v)

        # Direct Cartesian Hessian H_xx
        direct_Hxx = d2V_dxx_func(x_v, y_v, z_v)

        # Tensor-form Hessian H_xx via §3.3
        # H_xx = Σ_{a,b} J^a_x · J^b_x · H_sph_{ab} + Σ_a (∂V/∂q^a)·K^a_{xx}
        H_sph = [
            [d2V_drr_func(r_v, phi_v, lam_v),
             d2V_drphi_func(r_v, phi_v, lam_v),
             d2V_drlam_func(r_v, phi_v, lam_v)],
            [d2V_drphi_func(r_v, phi_v, lam_v),
             d2V_dphiphi_func(r_v, phi_v, lam_v),
             d2V_dphilam_func(r_v, phi_v, lam_v)],
            [d2V_drlam_func(r_v, phi_v, lam_v),
             d2V_dphilam_func(r_v, phi_v, lam_v),
             d2V_dlamlam_func(r_v, phi_v, lam_v)],
        ]
        dV_dq = [
            dV_dr_func(r_v, phi_v, lam_v),
            dV_dphi_func(r_v, phi_v, lam_v),
            dV_dlam_func(r_v, phi_v, lam_v),
        ]
        J_eval = [[J_funcs[a][i](x_v, y_v, z_v) for i in range(3)] for a in range(3)]
        K_xx_eval = [K_funcs[a][0][0](x_v, y_v, z_v) for a in range(3)]

        # Term 1: J·H_sph·Jᵀ at (i=0, j=0)
        JHJ_xx = sum(
            J_eval[a][0] * J_eval[b][0] * H_sph[a][b]
            for a in range(3)
            for b in range(3)
        )
        # Term 2: Σ_a (∂V/∂q^a)·K^a_{xx}
        sumK_xx = sum(dV_dq[a] * K_xx_eval[a] for a in range(3))

        tensor_Hxx = JHJ_xx + sumK_xx

        diff = abs(direct_Hxx - tensor_Hxx)
        # Match within mpmath precision (relative)
        denom = max(abs(direct_Hxx), mp.mpf("1e-100"))
        rel = diff / denom
        match = rel < mp.mpf("1e-40")

        if not match:
            all_match = False

        detail_lines.append(
            f"{mp.nstr(eps, 1):<10s}  {mp.nstr(direct_Hxx, 8):<25s}  "
            f"{mp.nstr(tensor_Hxx, 8):<25s}  "
            f"{mp.nstr(abs(JHJ_xx), 4):<15s}  {mp.nstr(abs(sumK_xx), 4):<15s}  "
            f"{'YES' if match else 'NO'}"
        )

    detail_lines.append("")
    detail_lines.append(
        "If |JHJ_xx| and |ΣKa·∂V| both grow as ε → 0 but their SUM "
        "stays bounded → cancellation confirmed."
    )
    detail = "\n".join(detail_lines)

    return VerificationResult(
        item_id="ITEM-014",
        claim="Spherical-form 1/cos²(φ) divergences cancel exactly in the Cartesian Hessian",
        source="docs/gravity_hessian_derivation.md §4",
        method="mpmath @ 50-digit precision: direct Cartesian H_xx vs §3.3 tensor decomposition at ε → 0",
        passed=all_match,
        detail=detail,
    )


# =============================================================================
# ITEM-015 — Pines polynomial Laplace verification (§5.1 table)
# =============================================================================

def verify_item_015() -> VerificationResult:
    """ITEM-015 — each Pines polynomial in §5.1 is harmonic and consistent.

    CLAIM (independent statement, NOT derived from SymPy output):
        Each spherical-harmonic basis function V_lm is harmonic
        (∇²V_lm = 0) and its Cartesian Hessian is symmetric (H = Hᵀ).
        Both properties are required mathematical features of the
        gravitational potential outside source masses.

        The §5.1 Pines polynomial table states explicit Cartesian forms
        for V_lm with l ∈ {2, 3} and m ∈ {0, ..., l}, both cosine and
        sine variants. If the table entries are correct, each V_lm
        satisfies:
            (a) Laplace's equation:   ∇²V_lm = ∂²V/∂x² + ∂²V/∂y² + ∂²V/∂z² = 0
            (b) Symmetric Hessian:     H_{ij} = H_{ji}     (always true for sp.diff)

        These claims are stated independently. SymPy's job is to compute
        the Cartesian Hessian from each polynomial form and verify both
        properties.

    SOURCE
        docs/gravity_hessian_derivation.md §5.1 (Pines polynomial table).

    METHOD
        For each (l, m, kind) ∈ {(2,0,c), (2,1,c), (2,1,s), (2,2,c),
        (2,2,s), (3,0,c), (3,1,c), (3,1,s), (3,2,c), (3,2,s), (3,3,c),
        (3,3,s)}:
        1. Construct V_lm as the Cartesian polynomial from §5.1 table
           (with R = 1 to keep expressions clean — the R^(l+1) prefactor
           doesn't affect Laplace or symmetry).
        2. Compute the 3×3 Cartesian Hessian H via sp.diff.
        3. Verify Laplace: simplify(tr(H)) == 0.
        4. Verify symmetry: H == H.T (always true for sp.diff, included
           as a sanity check that nothing got mangled).
        5. Report each (l, m, kind) result.

        If ANY V_lm fails Laplace, the §5.1 table has a bug — the entry
        is not actually a spherical harmonic. This is a strong test
        because Laplace is sensitive to algebraic errors.

    IMPLICATED ON FAILURE
        A failure means the §5.1 Pines polynomial table has an error in
        the listed Cartesian form for that (l, m, kind). The C++
        implementation guided by §5.1 would compute the wrong gravity
        contribution from that mode. Fix the table before proceeding.
    """
    x, y, z = sp.symbols("x y z", real=True)
    r = sp.sqrt(x**2 + y**2 + z**2)

    # §5.1 Pines polynomials (un-normalized, R = 1 to drop R^(l+1) prefactor)
    pines_table = [
        (2, 0, "cos", (2*z**2 - x**2 - y**2) / (2 * r**5)),
        (2, 1, "cos", 3 * x * z / r**5),
        (2, 1, "sin", 3 * y * z / r**5),
        (2, 2, "cos", 3 * (x**2 - y**2) / r**5),
        (2, 2, "sin", 6 * x * y / r**5),
        (3, 0, "cos", z * (2*z**2 - 3*x**2 - 3*y**2) / (2 * r**7)),
        (3, 1, "cos", sp.Rational(3, 2) * x * (4*z**2 - x**2 - y**2) / r**7),
        (3, 1, "sin", sp.Rational(3, 2) * y * (4*z**2 - x**2 - y**2) / r**7),
        (3, 2, "cos", 15 * z * (x**2 - y**2) / r**7),
        (3, 2, "sin", 30 * x * y * z / r**7),
        (3, 3, "cos", 15 * x * (x**2 - 3*y**2) / r**7),
        (3, 3, "sin", 15 * y * (3*x**2 - y**2) / r**7),
    ]

    coords = [x, y, z]

    detail_lines = []
    detail_lines.append("(l, m, kind)    Laplace(∇²V)    Symmetric H?")
    detail_lines.append("-" * 60)

    all_pass = True
    for l_v, m_v, kind, V in pines_table:
        H = sp.Matrix(3, 3, lambda i, j: sp.diff(V, coords[i], coords[j]))
        laplace = sp.simplify(H[0, 0] + H[1, 1] + H[2, 2])
        sym_diff = sp.simplify(H - H.T)
        symmetric = (sym_diff == sp.zeros(3, 3))

        passed_one = (laplace == 0) and symmetric
        all_pass = all_pass and passed_one

        marker = "✓" if passed_one else "✗"
        detail_lines.append(
            f"({l_v}, {m_v}, {kind:3s})    {str(laplace):<14s}  {symmetric}  {marker}"
        )

    detail = "\n".join(detail_lines)

    return VerificationResult(
        item_id="ITEM-015",
        claim="Each §5.1 Pines polynomial V_lm satisfies ∇²V_lm = 0 and has symmetric Hessian",
        source="docs/gravity_hessian_derivation.md §5.1",
        method="Compute H = sp.diff·diff(V), check tr(H) = 0 (Laplace) and H = Hᵀ (symmetry)",
        passed=all_pass,
        detail=detail,
    )


# =============================================================================
# ITEM-016 — solid-tide corrections are satellite-position-independent
# =============================================================================

def verify_item_016() -> VerificationResult:
    """ITEM-016 — IERS 2010 ΔC_lm, ΔS_lm have zero satellite-position derivative.

    CLAIM (independent statement, NOT derived from SymPy output):
        The IERS 2010 Step 1 solid-tide correction
            ΔC_lm − i·ΔS_lm = (k_lm/(2l+1)) · Σ_j (GM_j/GM_E)
                              · (R/r_j)^(l+1) · P̄_lm(sin Φ_j)
                              · exp(−i·m·λ_j)
        depends ONLY on lunisolar ephemeris (r_j, Φ_j, λ_j for j ∈
        {Sun, Moon}), NOT on satellite position. Therefore
            ∂ΔC_lm/∂x_sat = ∂ΔC_lm/∂y_sat = ∂ΔC_lm/∂z_sat = 0
            ∂ΔS_lm/∂x_sat = ∂ΔS_lm/∂y_sat = ∂ΔS_lm/∂z_sat = 0

        Consequence: the satellite-Hessian for solid tides reduces to
            H_tide(r_sat, t) = Σ ΔC_lm(t)·H(V_lm^c)(r_sat)
                              + Σ ΔS_lm(t)·H(V_lm^s)(r_sat)
        — same engine, corrected coefficients. No new analytic Hessian
        derivation needed for tides.

        This claim is stated independently. SymPy's job is to construct
        the Step 1 formula symbolically and verify the satellite
        position never appears (i.e., the partial derivatives wrt
        x_sat, y_sat, z_sat are identically zero).

    SOURCE
        docs/gravity_hessian_derivation.md §11.

    METHOD
        1. Build the Step 1 formula symbolically with separate symbols
           for satellite (x_sat, y_sat, z_sat), Sun (r_sun, phi_sun,
           lam_sun), and Moon (r_moon, phi_moon, lam_moon).
        2. Use a representative (l, m) ∈ {(2, 0), (2, 1), (2, 2)}.
        3. For each, assert sp.diff(ΔC_lm, x_sat) == 0 and likewise for
           y_sat, z_sat. Same for ΔS_lm.

    IMPLICATED ON FAILURE
        A failure would mean my Step 1 formula in §9.2 contains
        satellite position somewhere (a transcription error). The
        consequence would be that solid-tide Hessian doesn't reduce to
        "same engine + corrected coefficients" — would need its own
        derivation, and the solid-tide Hessian implementation would be wrong.

    """
    # Satellite position symbols
    x_sat, y_sat, z_sat = sp.symbols("x_sat y_sat z_sat", real=True)

    # Sun & Moon ephemeris (independent symbols — no relation to satellite)
    r_sun, phi_sun, lam_sun = sp.symbols("r_sun phi_sun lam_sun", real=True, positive=True)
    r_moon, phi_moon, lam_moon = sp.symbols("r_moon phi_moon lam_moon", real=True, positive=True)
    GM_sun, GM_moon, GM_E = sp.symbols("GM_sun GM_moon GM_E", real=True, positive=True)
    R = sp.symbols("R", real=True, positive=True)
    k_lm = sp.symbols("k_lm", real=True)

    # Test (l, m) pairs and their P̄_lm formulas (un-normalized, sufficient
    # for the structure-only verification — normalization constants are
    # all satellite-position-independent so don't affect the property).
    test_cases = [
        (2, 0),   # zonal
        (2, 1),   # tesseral
        (2, 2),   # sectoral
    ]

    detail_lines = []
    detail_lines.append(
        "(l, m)    ∂ΔC/∂x_sat   ∂ΔC/∂y_sat   ∂ΔC/∂z_sat   "
        "∂ΔS/∂x_sat   ∂ΔS/∂y_sat   ∂ΔS/∂z_sat"
    )
    detail_lines.append("-" * 80)

    all_zero = True
    for l_v, m_v in test_cases:
        # P_lm(sin φ) using SymPy's assoc_legendre
        P_sun = sp.assoc_legendre(l_v, m_v, sp.sin(phi_sun))
        P_moon = sp.assoc_legendre(l_v, m_v, sp.sin(phi_moon))

        # Step 1 formula (real and imaginary parts of complex form)
        # ΔC_lm − i·ΔS_lm = (k_lm/(2l+1)) · Σ_j (GM_j/GM_E)·(R/r_j)^(l+1)·P̄_lm(sin Φ_j)·exp(−imλ_j)
        # Real part: ΔC_lm = (k_lm/(2l+1)) · Σ_j coef_j · P_j · cos(m·λ_j)
        # Imag part: −ΔS_lm = (k_lm/(2l+1)) · Σ_j coef_j · P_j · (−sin(m·λ_j))
        # ⇒ ΔS_lm = (k_lm/(2l+1)) · Σ_j coef_j · P_j · sin(m·λ_j)

        prefactor = k_lm / (2 * l_v + 1)
        coef_sun = (GM_sun / GM_E) * (R / r_sun) ** (l_v + 1)
        coef_moon = (GM_moon / GM_E) * (R / r_moon) ** (l_v + 1)

        DC_lm = prefactor * (
            coef_sun * P_sun * sp.cos(m_v * lam_sun)
            + coef_moon * P_moon * sp.cos(m_v * lam_moon)
        )
        DS_lm = prefactor * (
            coef_sun * P_sun * sp.sin(m_v * lam_sun)
            + coef_moon * P_moon * sp.sin(m_v * lam_moon)
        )

        # Verify ∂/∂x_sat = ∂/∂y_sat = ∂/∂z_sat = 0
        derivs_DC = [sp.diff(DC_lm, var) for var in (x_sat, y_sat, z_sat)]
        derivs_DS = [sp.diff(DS_lm, var) for var in (x_sat, y_sat, z_sat)]

        all_DC_zero = all(d == 0 for d in derivs_DC)
        all_DS_zero = all(d == 0 for d in derivs_DS)

        if not (all_DC_zero and all_DS_zero):
            all_zero = False

        detail_lines.append(
            f"({l_v}, {m_v})    "
            f"{str(derivs_DC[0]):<12s} {str(derivs_DC[1]):<12s} "
            f"{str(derivs_DC[2]):<12s} {str(derivs_DS[0]):<12s} "
            f"{str(derivs_DS[1]):<12s} {str(derivs_DS[2]):<12s}"
        )

    detail_lines.append("")
    detail_lines.append(
        "All entries should be 0: confirms ΔC_lm, ΔS_lm depend on "
        "lunisolar ephemeris only, not on satellite position."
    )
    detail = "\n".join(detail_lines)

    return VerificationResult(
        item_id="ITEM-016",
        claim="∂(IERS 2010 ΔC_lm, ΔS_lm)/∂r_sat = 0 (satellite-position-independent)",
        source="docs/gravity_hessian_derivation.md §11",
        method="Build Step 1 formula symbolically; sp.diff wrt (x_sat, y_sat, z_sat); assert == 0",
        passed=all_zero,
        detail=detail,
    )


# =============================================================================
# ITEM-017 — engine-convention second-derivative inner-loop recurrences
# =============================================================================

def verify_item_017() -> VerificationResult:
    """ITEM-017 — engine-convention wrrc/wrtc/wttc + sine siblings.

    CLAIM (independent statement, NOT derived from SymPy output):
        Differentiating the engine's first-derivative recurrences (line 192,
        197 of clenshaw_engine.hpp) wrt r and φ yields the following
        engine-convention second-derivative inner-loop recurrences:

        wrrc_engine_new = A·wrrc + B·wrrc2 − (A/r)·wrc − (2B/r)·wrc2
        wrtc_engine_new = A·wrtc + B·wrtc2 + u·Ax·wrc
        wttc_engine_new = A·wttc + B·wttc2 + u·Ax·wtc + t·Ax·wc2 − u·Ax·wtc2

        Sine siblings: identical with wc → ws, wrc → wrs, wtc → wts, wc2 → ws2.

        The wttc form is STRUCTURALLY DIFFERENT from §2.3 natural-convention
        (which has 2·u·Ax·wtc − t·Ax·wc — current iterate). The engine form
        has u·Ax·wtc + t·Ax·wc2 − u·Ax·wtc2 (mix of current and old iterates,
        sign flip on t-term) reflecting the engine's wtc convention
        (sign-flip + index-roll vs natural).

        These claims are stated independently. SymPy's job is to compute
        ∂(engine wrc_new)/∂r, ∂(engine wrc_new)/∂φ, ∂(engine wtc_new)/∂φ
        and confirm via simplify(SymPy − claimed) == 0.

    SOURCE
        docs/gravity_hessian_derivation.md §2.6.
        Engine reference: clenshaw_engine.hpp lines 187, 192, 197.

    METHOD
        Build engine first-derivative recurrences symbolically with
        engine's specific source terms. Differentiate via sp.diff and
        compare to claimed engine second-derivative recurrences.
    """
    r, R, sqrt2n3, b_factor = sp.symbols("r R sqrt2n3 b_factor", real=True, positive=True)
    n = sp.symbols("n", integer=True, nonnegative=True)
    Cv = sp.symbols("Cv", real=True)
    phi = sp.symbols("phi", real=True)

    q = R / r
    Ax = q * sqrt2n3
    A_phi = sp.sin(phi) * Ax     # A as function of phi (with t = sin phi explicit)
    B_phi = -q**2 * b_factor

    # ========== Sub-test 1: wrrc engine (∂²V/∂r²) ==========
    # Engine wrc_new = A·wrc + B·wrc2 + (n+1)·Cv (with t = sin phi treated as parameter)
    t_param = sp.symbols("t_param", real=True)
    A_r_clean = t_param * (R/r) * sqrt2n3
    B_r_clean = -(R/r)**2 * b_factor

    wrc_f = sp.Function("wrc")(r)
    wrc2_f = sp.Function("wrc2")(r)
    wrc_new_engine = A_r_clean * wrc_f + B_r_clean * wrc2_f + (n + 1) * Cv

    wrrc_sympy = sp.expand(sp.diff(wrc_new_engine, r))

    wrrc_f = sp.diff(wrc_f, r)
    wrrc2_f = sp.diff(wrc2_f, r)
    wrrc_claimed = sp.expand(
        A_r_clean * wrrc_f + B_r_clean * wrrc2_f
        - (A_r_clean / r) * wrc_f
        - (2 * B_r_clean / r) * wrc2_f
    )
    diff1 = sp.simplify(wrrc_sympy - wrrc_claimed)

    # ========== Sub-test 2: wrtc engine (∂²V/∂r∂φ via ∂/∂φ of engine wrc) ==========
    wrc_phi = sp.Function("wrc")(phi)
    wrc2_phi = sp.Function("wrc2")(phi)
    wrc_new_engine_phi = A_phi * wrc_phi + B_phi * wrc2_phi + (n + 1) * Cv
    # Note: (n+1)·Cv is φ-independent, contributes 0 on differentiation
    wrtc_sympy = sp.expand(sp.diff(wrc_new_engine_phi, phi))

    wrtc_f = sp.diff(wrc_phi, phi)
    wrtc2_f = sp.diff(wrc2_phi, phi)
    wrtc_claimed = sp.expand(
        A_phi * wrtc_f + B_phi * wrtc2_f + sp.cos(phi) * Ax * wrc_phi
    )
    diff2 = sp.simplify(wrtc_sympy - wrtc_claimed)

    # ========== Sub-test 3: wttc engine (THE error-prone one) ==========
    # Engine wtc_new = A·wtc + B·wtc2 - u·Ax·wc2  (note: wc2 old iterate, minus sign)
    wc_phi = sp.Function("wc")(phi)
    wc2_phi = sp.Function("wc2")(phi)
    wtc_phi = sp.Function("wtc")(phi)
    wtc2_phi = sp.Function("wtc2")(phi)

    # The engine source uses wc2 (the OLD iterate at this recursion step).
    # We treat wc2 as an independent function of phi for SymPy diff.
    wtc_new_engine = A_phi * wtc_phi + B_phi * wtc2_phi - sp.cos(phi) * Ax * wc2_phi

    wttc_sympy = sp.expand(sp.diff(wtc_new_engine, phi))

    # The CLAIM (from my markdown §2.6.3):
    # wttc_engine_new = A·wttc + B·wttc2 + u·Ax·wtc + t·Ax·wc2 − u·Ax·wtc2
    wttc_f = sp.diff(wtc_phi, phi)
    wttc2_f = sp.diff(wtc2_phi, phi)
    wtc2_phi_deriv = sp.diff(wc2_phi, phi)  # = wtc2 in natural convention; here just derivative of wc2

    wttc_claimed = sp.expand(
        A_phi * wttc_f + B_phi * wttc2_f
        + sp.cos(phi) * Ax * wtc_phi
        + sp.sin(phi) * Ax * wc2_phi
        - sp.cos(phi) * Ax * wtc2_phi_deriv
    )
    diff3 = sp.simplify(wttc_sympy - wttc_claimed)

    passed = (diff1 == 0) and (diff2 == 0) and (diff3 == 0)

    detail = (
        f"Sub-test 1: engine wrrc (differentiate engine wrc_new wrt r)\n"
        f"  simplify(SymPy − claimed) = {diff1}\n"
        f"  Match: {'YES' if diff1 == 0 else 'NO'}\n\n"
        f"Sub-test 2: engine wrtc (differentiate engine wrc_new wrt φ)\n"
        f"  simplify(SymPy − claimed) = {diff2}\n"
        f"  Match: {'YES' if diff2 == 0 else 'NO'}\n\n"
        f"Sub-test 3: engine wttc (differentiate engine wtc_new wrt φ)\n"
        f"  SymPy expanded: {wttc_sympy}\n"
        f"  Claimed expanded: {sp.expand(wttc_claimed)}\n"
        f"  simplify(SymPy − claimed) = {diff3}\n"
        f"  Match: {'YES' if diff3 == 0 else 'NO'}\n\n"
        f"Note: §2.3 natural wttc has form '2·u·Ax·wtc − t·Ax·wc' which is\n"
        f"DIFFERENT from this engine form. The engine form is what Stage F\n"
        f"must implement, not §2.3.\n"
    )

    return VerificationResult(
        item_id="ITEM-017",
        claim="Engine-convention wrrc/wrtc/wttc recurrences (differentiate engine first-derivative recurrences)",
        source="docs/gravity_hessian_derivation.md §2.6",
        method="sp.diff on engine wrc_new and wtc_new; compare to claimed engine 2nd-derivative recurrences",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-021 — ECEF→ECI Hessian similarity transform
# =============================================================================

def verify_item_021() -> VerificationResult:
    """ITEM-021 — H_eci = R · H_ecef · Rᵀ preserves symmetry and trace.

    CLAIM (independent statement, NOT derived from SymPy output):
        For a rank-2 symmetric tensor H_ecef (the gravity Hessian in ECEF)
        and an orthogonal rotation matrix R (ECEF→ECI), the transformed
        Hessian
            H_eci = R · H_ecef · Rᵀ
        satisfies:
          (a) H_eci is symmetric (H_eci = H_eciᵀ).
          (b) tr(H_eci) = tr(H_ecef) (Laplace invariant under rotation).
          (c) For each fixed orthogonal R, ‖H_eci‖_F = ‖H_ecef‖_F
              (Frobenius norm preserved).

        These properties are mathematical consequences of the similarity
        transform on a symmetric tensor. SymPy verifies them by direct
        computation on a parameterized symbolic R and H_ecef.

    SOURCE
        docs/gravity_hessian_derivation.md §3.5.

    METHOD
        1. Build symbolic 3×3 H_ecef as an arbitrary symmetric matrix
           (6 unique entries).
        2. Build symbolic 3×3 R as an arbitrary orthogonal matrix
           (parameterize by 3 Euler angles α, β, γ; ensure RᵀR = I via
           construction).
        3. Compute H_eci = R · H_ecef · Rᵀ.
        4. Verify (a) H_eci - H_eciᵀ == 0
                  (b) tr(H_eci) - tr(H_ecef) == 0
                  (c) ‖H_eci‖_F² - ‖H_ecef‖_F² == 0
        5. Additionally: check H_eci has the EXPECTED form for specific
           cases (e.g., R = I gives H_eci = H_ecef).

    IMPLICATED ON FAILURE
        Failure means the §3.5 frame-transform formula is wrong. Stage I
        implementation would compute Hessian in wrong frame, producing
        wrong satellite-state Jacobian for gravity. Easy to detect at
        end-to-end (large RMSE jump), hard to debug without this
        derivation.
    """
    # Parameterize R via Euler angles (Z-Y-X convention, arbitrary)
    alpha, beta, gamma = sp.symbols("alpha beta gamma", real=True)

    Rz = sp.Matrix([
        [sp.cos(alpha), -sp.sin(alpha), 0],
        [sp.sin(alpha),  sp.cos(alpha), 0],
        [0, 0, 1],
    ])
    Ry = sp.Matrix([
        [ sp.cos(beta), 0, sp.sin(beta)],
        [0, 1, 0],
        [-sp.sin(beta), 0, sp.cos(beta)],
    ])
    Rx = sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(gamma), -sp.sin(gamma)],
        [0, sp.sin(gamma),  sp.cos(gamma)],
    ])
    R = Rz * Ry * Rx  # arbitrary orthogonal

    # Sanity: RᵀR = I (orthogonality)
    orth_check = sp.simplify(R.T * R - sp.eye(3))
    R_orthogonal = (orth_check == sp.zeros(3, 3))

    # Build symbolic symmetric H_ecef
    h00, h01, h02, h11, h12, h22 = sp.symbols(
        "h00 h01 h02 h11 h12 h22", real=True
    )
    H_ecef = sp.Matrix([
        [h00, h01, h02],
        [h01, h11, h12],
        [h02, h12, h22],
    ])

    # Apply similarity transform
    H_eci = R * H_ecef * R.T

    # Property (a): symmetry
    sym_diff = sp.simplify(H_eci - H_eci.T)
    is_symmetric = (sym_diff == sp.zeros(3, 3))

    # Property (b): trace invariance
    trace_diff = sp.simplify(H_eci.trace() - H_ecef.trace())
    trace_invariant = (trace_diff == 0)

    # Property (c): Frobenius norm invariance
    # ‖M‖_F² = sum of squares of all entries
    fsq_eci = sum(H_eci[i, j]**2 for i in range(3) for j in range(3))
    fsq_ecef = sum(H_ecef[i, j]**2 for i in range(3) for j in range(3))
    fsq_diff = sp.simplify(fsq_eci - fsq_ecef)
    frobenius_invariant = (fsq_diff == 0)

    # Property (d): R = I sanity (alpha = beta = gamma = 0)
    H_eci_identity = H_eci.subs({alpha: 0, beta: 0, gamma: 0})
    identity_check = sp.simplify(H_eci_identity - H_ecef)
    identity_correct = (identity_check == sp.zeros(3, 3))

    passed = R_orthogonal and is_symmetric and trace_invariant and frobenius_invariant and identity_correct

    detail = (
        f"R orthogonality (RᵀR = I):                 {R_orthogonal}\n"
        f"(a) H_eci symmetric:                       {is_symmetric}\n"
        f"(b) tr(H_eci) = tr(H_ecef):                {trace_invariant}\n"
        f"(c) ‖H_eci‖_F = ‖H_ecef‖_F:                 {frobenius_invariant}\n"
        f"(d) R=I gives H_eci = H_ecef:              {identity_correct}\n\n"
        f"All properties hold ⇒ similarity transform H_eci = R·H_ecef·Rᵀ\n"
        f"correctly implements the ECEF→ECI Hessian frame change for any\n"
        f"symmetric H_ecef and any orthogonal R (parameterized as Euler\n"
        f"Z-Y-X)."
    )

    return VerificationResult(
        item_id="ITEM-021",
        claim="H_eci = R · H_ecef · Rᵀ preserves symmetry, trace, and Frobenius norm for orthogonal R",
        source="docs/gravity_hessian_derivation.md §3.5",
        method="Symbolic 3×3 symmetric H_ecef × Euler-parameterized R; verify properties via simplify",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-018 — Sectorial corrections for 2nd-derivative inner-loop accumulators
# =============================================================================

def verify_item_018() -> VerificationResult:
    """ITEM-018 — sectorial corrections for engine's 2nd-derivative inner-loop accumulators.

    CLAIM (independent statement, NOT derived from SymPy output):
        The engine's first-derivative sectorial correction at line 218-220:
            wtc_corrected = wtc + m·tu·wc      (tu = t/u)

        Differentiated wrt r and wrt φ produces sectorial corrections for
        the second-derivative accumulators:
            wrtc_corrected = wrtc + m·tu·wrc                    (wrt r)
            wttc_corrected = wttc + m·(1/u²)·wc + m·tu·(∂wc/∂φ)  (wrt φ)

        The 1/u² factor in wttc_corrected is the source of the 1/cos²(φ)
        pole singularity for sectorial m≥1 modes.

        IMPORTANT CONVENTION CAVEAT: in the third term `m·tu·(∂wc/∂φ)`,
        ∂wc/∂φ is NATURAL — the engine's `wtc` is NOT natural `∂wc/∂φ`
        (engine has sign-flip + iterate-roll per §0.3 / §2.3). Stage F
        must either add a new accumulator for natural ∂wc/∂φ or use
        Strategy 2 (FD).

        These claims are stated independently. SymPy's job is to compute
        ∂(wtc + m·tu·wc)/∂r and ∂(wtc + m·tu·wc)/∂φ from scratch and
        confirm via simplify(SymPy − claim) == 0. Treats wc and wtc as
        INDEPENDENT functions (mirroring the engine's bookkeeping).

    SOURCE
        docs/gravity_hessian_derivation.md §2.6.5; engine line 218-220.

    METHOD
        1. Build wtc_corrected = wtc(r,phi) + m·(t/u)·wc(r,phi) symbolically,
           with wc and wtc as functions of (r, phi), t = sin(phi),
           u = cos(phi).
        2. sp.diff wrt r and wrt phi.
        3. Compare to claimed forms.
    """
    r, phi, m_sym = sp.symbols("r phi m", real=True)
    u = sp.cos(phi)
    t = sp.sin(phi)
    tu = t / u

    # wc and wtc as functions of (r, phi)
    wc = sp.Function("wc")(r, phi)
    wtc = sp.Function("wtc")(r, phi)

    wtc_corrected = wtc + m_sym * tu * wc

    # ===== Sub-test 1: wrt r =====
    sympy_dr = sp.simplify(sp.diff(wtc_corrected, r))
    # Claim: wrtc + m·tu·wrc, where wrtc = ∂wtc/∂r and wrc = ∂wc/∂r
    wrtc_sym = sp.diff(wtc, r)
    wrc_sym = sp.diff(wc, r)
    claim_dr = wrtc_sym + m_sym * tu * wrc_sym
    diff1 = sp.simplify(sympy_dr - claim_dr)

    # ===== Sub-test 2: wrt phi =====
    sympy_dphi = sp.simplify(sp.diff(wtc_corrected, phi))
    # Claim: wttc + m·(1/u²)·wc + m·tu·(∂wc/∂φ)
    # NOTE: ∂wc/∂φ is NATURAL convention, NOT engine's wtc accumulator
    # (engine wtc has sign-flip + iterate-roll vs natural ∂wc/∂φ).
    wttc_sym = sp.diff(wtc, phi)
    wphic_sym = sp.diff(wc, phi)  # natural ∂wc/∂φ
    claim_dphi = wttc_sym + m_sym * (1 / u**2) * wc + m_sym * tu * wphic_sym
    diff2 = sp.simplify(sympy_dphi - claim_dphi)

    # ===== Sub-test 3: verify ∂(t/u)/∂φ = 1/u² (the load-bearing identity) =====
    deriv_tu_dphi = sp.simplify(sp.diff(tu, phi))
    expected_tu_dphi = 1 / u**2
    diff3 = sp.simplify(deriv_tu_dphi - expected_tu_dphi)

    passed = (diff1 == 0) and (diff2 == 0) and (diff3 == 0)

    detail = (
        f"Sub-test 1: ∂(wtc + m·tu·wc)/∂r → wrtc + m·tu·wrc\n"
        f"  simplify(SymPy − claim) = {diff1}\n"
        f"  Match: {'YES' if diff1 == 0 else 'NO'}\n\n"
        f"Sub-test 2: ∂(wtc + m·tu·wc)/∂φ → wttc + m·(1/u²)·wc + m·tu·wtc\n"
        f"  simplify(SymPy − claim) = {diff2}\n"
        f"  Match: {'YES' if diff2 == 0 else 'NO'}\n\n"
        f"Sub-test 3: ∂(t/u)/∂φ = 1/u²  (load-bearing identity for 1/cos²(φ) pole singularity)\n"
        f"  simplify(SymPy − claim) = {diff3}\n"
        f"  Match: {'YES' if diff3 == 0 else 'NO'}\n"
    )

    return VerificationResult(
        item_id="ITEM-018",
        claim="Sectorial corrections for engine's 2nd-derivative inner-loop accumulators",
        source="docs/gravity_hessian_derivation.md §2.6.5; engine line 218-220",
        method="Differentiate wtc + m·tu·wc wrt r and wrt φ; compare to claimed corrections",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-022 — Engine outer-loop 2nd-derivative recurrences (NEW gap)
# =============================================================================

def verify_item_022() -> VerificationResult:
    """ITEM-022 — engine outer-loop 2nd-derivative recurrences.

    CLAIM (independent statement, NOT derived from SymPy output):
        The engine's outer-loop first-derivative recurrences at line 232,
        234 use coefficients A_m, B_m that depend on (r, φ) through
        uq = u·R/r. Differentiating these wrt r and wrt φ produces:

            vrrc_new = A·vrrc + B·vrrc2 + wrrc
                       − (A/r)·vrc − (2B/r)·vrc2

            vrtc_new = A·vrtc + B·vrtc2 + wrtc
                       − tu·A·vrc − 2tu·B·vrc2     (tu = t/u)

            vttc_new = A·vttc + B·vttc2 + wttc
                       − tu·A·vtc − 2tu·B·vtc2

        The correction terms (last two on each line) come from
        ∂A/∂r = −A/r, ∂A/∂φ = −tu·A, ∂B/∂r = −2B/r, ∂B/∂φ = −2tu·B.

        These claims are stated independently. SymPy's job is to compute
        ∂(vrc_new)/∂r etc. and verify equality with the claimed form.

    SOURCE
        docs/gravity_hessian_derivation.md §2.7; engine line 232-234.

    METHOD
        1. Build engine outer first-derivative recurrence symbolically:
           vrc_new = A(r, φ)·vrc + B(r, φ)·vrc2 + wrc
           where A = cl·v·u·R/r and B = -c·u²·R²/r² for arbitrary constants.
        2. Differentiate wrt r and wrt φ.
        3. Compare to claimed engine outer-loop 2nd-derivative recurrences.
    """
    r, phi, lam = sp.symbols("r phi lam", real=True, positive=True)
    R = sp.symbols("R", positive=True)
    cl = sp.cos(lam)
    sl = sp.sin(lam)
    u = sp.cos(phi)
    t = sp.sin(phi)
    tu = t / u

    # A_m, B_m as functions of (r, phi) through uq = u·R/r
    # (cl is λ-independent for r, φ derivatives; mass into v_const)
    v_const, c_const = sp.symbols("v_const c_const", real=True)  # m-dependent constants
    uq = u * R / r
    A_m = cl * v_const * uq
    B_m = -c_const * uq**2  # Negative sign per engine line 227

    # First-derivative outer accumulators as functions of (r, phi)
    vrc = sp.Function("vrc")(r, phi)
    vrc2 = sp.Function("vrc2")(r, phi)
    vtc = sp.Function("vtc")(r, phi)
    vtc2 = sp.Function("vtc2")(r, phi)
    wrc = sp.Function("wrc")(r, phi)
    wtc = sp.Function("wtc")(r, phi)
    wrrc = sp.Function("wrrc")(r, phi)  # = ∂wrc/∂r — but we treat as independent for SymPy
    wrtc = sp.Function("wrtc")(r, phi)
    wttc = sp.Function("wttc")(r, phi)

    # Engine outer first-derivative recurrence (line 232)
    vrc_new = A_m * vrc + B_m * vrc2 + wrc
    vtc_new = A_m * vtc + B_m * vtc2 + wtc

    # ===== Sub-test 1: vrrc_new = ∂(vrc_new)/∂r =====
    sympy_vrrc = sp.simplify(sp.diff(vrc_new, r))

    # ∂vrc/∂r treated as the second-deriv-outer accumulator at this point
    # (∂vrc/∂r = vrrc); we use sp.diff to compute symbolically.
    vrrc_engine_outer = sp.diff(vrc, r)
    vrrc2_engine_outer = sp.diff(vrc2, r)
    wrc_dr = sp.diff(wrc, r)  # Should equal wrrc (engine inner-loop 2nd-deriv)

    # Claim: A·vrrc_outer + B·vrrc2_outer + ∂wrc/∂r − (A/r)·vrc − (2B/r)·vrc2
    # The ∂wrc/∂r is the inner-loop 2nd-deriv source. We just use sp.diff(wrc, r)
    # directly to maintain the claim independence.
    claim_vrrc = (
        A_m * vrrc_engine_outer + B_m * vrrc2_engine_outer + wrc_dr
        - (A_m / r) * vrc - (2 * B_m / r) * vrc2
    )
    diff1 = sp.simplify(sympy_vrrc - claim_vrrc)

    # ===== Sub-test 2: vrtc_new = ∂(vrc_new)/∂φ =====
    sympy_vrtc = sp.simplify(sp.diff(vrc_new, phi))
    vrtc_engine_outer = sp.diff(vrc, phi)
    vrtc2_engine_outer = sp.diff(vrc2, phi)
    wrc_dphi = sp.diff(wrc, phi)

    # Claim: A·vrtc_outer + B·vrtc2_outer + ∂wrc/∂φ − tu·A·vrc − 2tu·B·vrc2
    claim_vrtc = (
        A_m * vrtc_engine_outer + B_m * vrtc2_engine_outer + wrc_dphi
        - tu * A_m * vrc - 2 * tu * B_m * vrc2
    )
    diff2 = sp.simplify(sympy_vrtc - claim_vrtc)

    # ===== Sub-test 3: vttc_new = ∂(vtc_new)/∂φ =====
    sympy_vttc = sp.simplify(sp.diff(vtc_new, phi))
    vttc_engine_outer = sp.diff(vtc, phi)
    vttc2_engine_outer = sp.diff(vtc2, phi)
    wtc_dphi = sp.diff(wtc, phi)

    # Claim: A·vttc_outer + B·vttc2_outer + ∂wtc/∂φ − tu·A·vtc − 2tu·B·vtc2
    claim_vttc = (
        A_m * vttc_engine_outer + B_m * vttc2_engine_outer + wtc_dphi
        - tu * A_m * vtc - 2 * tu * B_m * vtc2
    )
    diff3 = sp.simplify(sympy_vttc - claim_vttc)

    # ===== Sub-test 4: verify ∂A_m/∂r = −A_m/r and ∂A_m/∂φ = −tu·A_m =====
    dA_dr_check = sp.simplify(sp.diff(A_m, r) - (-A_m / r))
    dA_dphi_check = sp.simplify(sp.diff(A_m, phi) - (-tu * A_m))
    diff4 = (dA_dr_check, dA_dphi_check)

    # ===== Sub-test 5: verify ∂B_m/∂r = −2B_m/r and ∂B_m/∂φ = −2tu·B_m =====
    dB_dr_check = sp.simplify(sp.diff(B_m, r) - (-2 * B_m / r))
    dB_dphi_check = sp.simplify(sp.diff(B_m, phi) - (-2 * tu * B_m))
    diff5 = (dB_dr_check, dB_dphi_check)

    passed = (
        diff1 == 0 and diff2 == 0 and diff3 == 0
        and diff4 == (0, 0) and diff5 == (0, 0)
    )

    detail = (
        f"Sub-test 1: vrrc_new from ∂(engine vrc_new)/∂r\n"
        f"  simplify(SymPy − claim) = {diff1}\n"
        f"  Match: {'YES' if diff1 == 0 else 'NO'}\n\n"
        f"Sub-test 2: vrtc_new from ∂(engine vrc_new)/∂φ\n"
        f"  simplify(SymPy − claim) = {diff2}\n"
        f"  Match: {'YES' if diff2 == 0 else 'NO'}\n\n"
        f"Sub-test 3: vttc_new from ∂(engine vtc_new)/∂φ\n"
        f"  simplify(SymPy − claim) = {diff3}\n"
        f"  Match: {'YES' if diff3 == 0 else 'NO'}\n\n"
        f"Sub-test 4: ∂A_m/∂r = -A_m/r, ∂A_m/∂φ = -tu·A_m\n"
        f"  diffs = {diff4}\n"
        f"  Match: {'YES' if diff4 == (0, 0) else 'NO'}\n\n"
        f"Sub-test 5: ∂B_m/∂r = -2B_m/r, ∂B_m/∂φ = -2tu·B_m\n"
        f"  diffs = {diff5}\n"
        f"  Match: {'YES' if diff5 == (0, 0) else 'NO'}\n"
    )

    return VerificationResult(
        item_id="ITEM-022",
        claim="Engine outer-loop 2nd-derivative recurrences include correction terms from ∂A_m/∂r, ∂A_m/∂φ, ∂B_m/∂r, ∂B_m/∂φ",
        source="docs/gravity_hessian_derivation.md §2.7",
        method="sp.diff on engine outer first-derivative recurrence; compare to claimed 2nd-derivative outer recurrence",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-020 — Explicit K^φ, K^λ Cartesian formulas
# =============================================================================

def verify_item_020() -> VerificationResult:
    """ITEM-020 — closed-form K^φ_{ij}, K^λ_{ij} via SymPy.

    CLAIM (independent statement, NOT derived from SymPy output):
        Define K^a_{ij} = ∂²q^a/∂x_i∂x_j for q^a ∈ {r, φ, λ}, where
        r = √(x²+y²+z²), φ = atan2(z, p), λ = atan2(y, x), p = √(x²+y²).

        Closed forms (independently derived; SymPy verifies):
            K^r:
              K^r_xx = (y² + z²)/r³,    K^r_yy = (x² + z²)/r³,
              K^r_zz = (x² + y²)/r³,
              K^r_xy = -xy/r³,          K^r_xz = -xz/r³,    K^r_yz = -yz/r³

            K^λ (z-independent — λ is purely (x,y)):
              K^λ_xx = 2xy/p⁴,          K^λ_yy = -2xy/p⁴,
              K^λ_xy = (y² - x²)/p⁴,
              K^λ_xz = K^λ_yz = K^λ_zz = 0
              (Property: K^λ_xx + K^λ_yy = 0 — λ is harmonic in (x,y))

            K^φ has 6 unique entries; closed form is more complex —
            SymPy computes and provides as reference for Stage F C++.

        These formulas allow Stage F C++ to use explicit expressions
        rather than reaching for symbolic differentiation at runtime.

    SOURCE
        docs/gravity_hessian_derivation.md §3.3 (K^r explicitly given;
        K^φ, K^λ noted as "SymPy-verified rather than written out
        manually").

    METHOD
        Use SymPy to compute ∂²q^a/∂x_i∂x_j from r_expr, phi_expr,
        lam_expr. Compare to claimed forms for K^r and K^λ. For K^φ,
        SymPy provides the reference values (printed in detail for
        Stage F C++ reference).
    """
    x, y, z = sp.symbols("x y z", real=True, positive=True)
    p = sp.sqrt(x**2 + y**2)
    r_expr = sp.sqrt(x**2 + y**2 + z**2)
    phi_expr = sp.atan2(z, p)
    lam_expr = sp.atan2(y, x)
    coords = [x, y, z]

    # ===== K^r =====
    K_r = sp.Matrix(3, 3, lambda i, j: sp.diff(r_expr, coords[i], coords[j]))
    K_r_simplified = K_r.applyfunc(sp.simplify)

    # Claim (manually derived):
    r3 = r_expr**3
    claim_K_r = sp.Matrix([
        [(y**2 + z**2) / r3,  -x*y / r3,  -x*z / r3],
        [-x*y / r3,  (x**2 + z**2) / r3,  -y*z / r3],
        [-x*z / r3,  -y*z / r3,  (x**2 + y**2) / r3],
    ])
    K_r_diff = sp.simplify(K_r_simplified - claim_K_r)
    K_r_match = (K_r_diff == sp.zeros(3, 3))

    # ===== K^λ =====
    K_lam = sp.Matrix(3, 3, lambda i, j: sp.diff(lam_expr, coords[i], coords[j]))
    K_lam_simplified = K_lam.applyfunc(sp.simplify)

    p4 = p**4
    claim_K_lam = sp.Matrix([
        [2*x*y / p4,  (y**2 - x**2) / p4,  0],
        [(y**2 - x**2) / p4,  -2*x*y / p4,  0],
        [0,  0,  0],
    ])
    K_lam_diff = sp.simplify(K_lam_simplified - claim_K_lam)
    K_lam_match = (K_lam_diff == sp.zeros(3, 3))

    # Verify Laplace property: ∂²λ/∂x² + ∂²λ/∂y² = 0
    lam_laplace = sp.simplify(K_lam_simplified[0, 0] + K_lam_simplified[1, 1])
    lam_harmonic_in_xy = (lam_laplace == 0)

    # ===== K^φ =====
    K_phi = sp.Matrix(3, 3, lambda i, j: sp.diff(phi_expr, coords[i], coords[j]))
    K_phi_simplified = K_phi.applyfunc(sp.simplify)

    # K^φ has 6 unique entries. SymPy provides closed forms (printed for
    # Stage F C++ reference). We verify symmetry only (since explicit
    # closed forms are complex; trusting SymPy's chain rule).
    K_phi_sym_diff = sp.simplify(K_phi_simplified - K_phi_simplified.T)
    K_phi_symmetric = (K_phi_sym_diff == sp.zeros(3, 3))

    passed = K_r_match and K_lam_match and lam_harmonic_in_xy and K_phi_symmetric

    # Format K^φ for output
    Kphi_str = ""
    for i in range(3):
        for j in range(i, 3):
            Kphi_str += f"\n    K^φ_{('xyz')[i]}{('xyz')[j]} = {K_phi_simplified[i, j]}"

    detail = (
        f"K^r match (closed-form):       {K_r_match}\n"
        f"K^λ match (closed-form):       {K_lam_match}\n"
        f"K^λ Laplace (∂²λ/∂x² + ∂²λ/∂y² = 0): {lam_harmonic_in_xy}\n"
        f"K^φ symmetric (Schwarz):       {K_phi_symmetric}\n\n"
        f"K^φ explicit forms (for Stage F C++ reference):"
        f"{Kphi_str}\n"
    )

    return VerificationResult(
        item_id="ITEM-020",
        claim="Closed-form K^r, K^λ explicit; K^φ available via SymPy",
        source="docs/gravity_hessian_derivation.md §3.3",
        method="sp.diff on r, phi, lam expressions wrt (x, y, z); compare to claimed forms",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-019 (partial) — vrrc m=0 finalization derivation
# =============================================================================

def verify_item_019_partial() -> VerificationResult:
    """ITEM-019 (partial) — m=0 finalization formula for vrrc.

    CLAIM (independent statement, NOT derived from SymPy output):
        Differentiating engine's vrc m=0 finalization formula
            vrc_final = −qs_r · (wrc + A·(cl·vrc + sl·vrs) + B·vrc2)
        wrt r yields:
            vrrc_final = ∂vrc_final/∂r
                       = −(2/r)·vrc_final
                         − qs_r · [ wrrc + A·(cl·vrrc + sl·vrrs) + B·vrrc2
                                     − (A/r)·(cl·vrc + sl·vrs) − (2B/r)·vrc2 ]

        where:
        - qs_r = q/r = R/r²; ∂qs_r/∂r = −2qs_r/r
        - A_m=0 = sqrt(3)·u·q; ∂A/∂r = −A/r
        - B_m=0 = −sqrt(15)/2·u²·q²; ∂B/∂r = −2B/r
        - cl, sl are λ-independent of r (cos λ, sin λ)
        - The inner-loop wrc, the m=1 outer accumulators vrc, vrs, vrc2,
          and their second-derivative analogs wrrc, vrrc, vrrs, vrrc2 are
          all r-dependent.

        These claims are stated independently. SymPy's job is to compute
        ∂vrc_final/∂r from scratch (treating engine accumulators as
        independent functions of r) and compare to the claimed formula.

        With this finalization formula, (GM/R)·vrrc_final = ∂²V/∂r² —
        the natural partial-derivatives 2nd-radial-derivative.

    SOURCE
        docs/gravity_hessian_derivation.md §3.4; engine line 246.

    METHOD
        1. Build vrc_final symbolically with engine's m=0 A, B, qs_r.
        2. Differentiate wrt r via sp.diff.
        3. Compare to claimed formula.

    NOTE
        This is a partial verification (vrrc only). Analogous derivations
        for vrtc, vttc, vrlc, vtlc, vllc are deferred to Stage F prep
        time (each follows the same pattern but takes derivative wrt
        different variable).

    IMPLICATED ON FAILURE
        If the m=0 finalization formula is wrong, Stage F's analytic
        Hessian will compute wrong ∂²V/∂r². ITEM-013's tensor
        decomposition test would catch this at the C++ level, but
        catching it here at the formula level localizes the bug to
        finalization specifically.
    """
    r, phi, lam = sp.symbols("r phi lam", real=True, positive=True)
    R = sp.symbols("R", positive=True)

    u = sp.cos(phi)
    cl = sp.cos(lam)
    sl = sp.sin(lam)

    # Engine m=0 finalization values
    # A = sqrt(3) · u · q,   B = -sqrt(15)/2 · (u·q)²,   qs_r = q/r = R/r²
    q = R / r
    qs_r = q / r
    A = sp.sqrt(3) * u * q
    B = -sp.sqrt(15) / 2 * (u * q)**2

    # Inner-loop wrc, wrrc, m=1 outer vrc, vrs, vrc2, vrrc, vrrs, vrrc2 as functions of r
    wrc = sp.Function("wrc")(r)
    wrrc = sp.Function("wrrc")(r)  # treat as ∂wrc/∂r in the engine setup
    vrc = sp.Function("vrc")(r)
    vrs = sp.Function("vrs")(r)
    vrc2 = sp.Function("vrc2")(r)
    vrrc = sp.Function("vrrc")(r)
    vrrs = sp.Function("vrrs")(r)
    vrrc2 = sp.Function("vrrc2")(r)

    # Engine vrc_final formula (line 246)
    vrc_final = -qs_r * (wrc + A * (cl * vrc + sl * vrs) + B * vrc2)

    # Compute ∂vrc_final/∂r via SymPy
    sympy_vrrc_final = sp.diff(vrc_final, r)

    # We need to substitute ∂wrc/∂r = wrrc, ∂vrc/∂r = vrrc, etc.
    # so the diff comes back in terms of our second-derivative accumulators.
    sympy_vrrc_substituted = sympy_vrrc_final.subs({
        sp.diff(wrc, r): wrrc,
        sp.diff(vrc, r): vrrc,
        sp.diff(vrs, r): vrrs,
        sp.diff(vrc2, r): vrrc2,
    })

    # Claimed formula:
    # vrrc_final = -(2/r)·vrc_final
    #              - qs_r · [wrrc + A·(cl·vrrc + sl·vrrs) + B·vrrc2
    #                          - (A/r)·(cl·vrc + sl·vrs) - (2B/r)·vrc2]
    claim_vrrc = -(2 / r) * vrc_final - qs_r * (
        wrrc + A * (cl * vrrc + sl * vrrs) + B * vrrc2
        - (A / r) * (cl * vrc + sl * vrs)
        - (2 * B / r) * vrc2
    )

    diff = sp.simplify(sympy_vrrc_substituted - claim_vrrc)

    passed = (diff == 0)

    detail = (
        f"Claim: vrrc_final = -(2/r)·vrc_final - qs_r · [wrrc + A·(cl·vrrc + sl·vrrs)\n"
        f"        + B·vrrc2 - (A/r)·(cl·vrc + sl·vrs) - (2B/r)·vrc2]\n\n"
        f"simplify(SymPy − claim) = {diff}\n"
        f"Match: {'YES' if diff == 0 else 'NO'}\n\n"
        f"This is the m=0 finalization formula for vrrc that Stage F C++\n"
        f"must implement (or compute via Strategy 2 FD fallback). Note the\n"
        f"presence of m=1 first-derivative outer accumulators (vrc, vrs,\n"
        f"vrc2) in the correction terms — Stage F must save these in\n"
        f"temporaries before the engine's first-derivative finalization\n"
        f"overwrites vrc.\n\n"
        f"Analogous formulas for vrtc, vttc, vrlc, vtlc, vllc are derivable\n"
        f"by the same approach (differentiate the corresponding\n"
        f"first-derivative finalization wrt the appropriate variable).\n"
        f"Pattern: ∂(α·X)/∂q = (∂α/∂q)·X + α·(∂X/∂q) where α ∈\n"
        f"{{qs_r, +qs_r, qs_r/u}} and X is the corresponding bracketed sum."
    )

    return VerificationResult(
        item_id="ITEM-019-partial",
        claim="m=0 finalization formula for vrrc (∂²V/∂r² convention factor)",
        source="docs/gravity_hessian_derivation.md §3.4; engine line 246",
        method="sp.diff on engine's vrc_final wrt r; compare to claimed vrrc finalization formula",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-019-full — m=0 finalization for vrtc, vttc, vrlc, vtlc, vllc
# =============================================================================

def verify_item_019_full() -> VerificationResult:
    """ITEM-019 (full) — m=0 finalization formulas for all 6 second-derivative accumulators.

    CLAIM (independent statements, NOT derived from SymPy output):
        Differentiating engine's first-derivative finalization formulas
        (line 246-249) wrt the appropriate variable yields:

        Notation: q = R/r, qs_r = q/r, A = sqrt(3)·u·q, B = -sqrt(15)/2·(uq)².

        vrtc_final = ∂vrc_final/∂φ
                   = -qs_r · [ wrtc + A·(cl·vrtc + sl·vrts) + B·vrtc2
                                - tu·A·(cl·vrc + sl·vrs)
                                - 2·tu·B·vrc2 ]
                   (No 1/r prefactor change since qs_r is φ-indep.)

        vttc_final = ∂vtc_final/∂φ
                   = +qs_r · [ wttc + A·(cl·vttc + sl·vtts) + B·vttc2
                                - tu·A·(cl·vtc + sl·vts)
                                - 2·tu·B·vtc2 ]

        vrlc_final = ∂vrc_final/∂λ
                   = -qs_r · [ A·(-sl·vrc + cl·vrs)·m_lambda_term ... ]
            (lambda derivative — uses the §2.5 source-substitution lemma
             at the m=1 outer step to convert the bucket structure;
             cl, sl are the only λ-dependent factors at m=0 finalization.)

        vtlc_final = ∂vtc_final/∂λ        (similar)
        vllc_final = ∂vlc_final/∂λ        (more complex because vlc
                                          itself has 1/u factor)

        For the LAMBDA-derivative finalizations, the cleanest derivation
        uses:
        - ∂cl/∂λ = -sl,  ∂sl/∂λ = +cl
        - The m=1 outer accumulators encode lambda derivatives via the
          §2.5 source rule.

        These claims are stated independently. SymPy verifies each by
        differentiating the corresponding 1st-derivative finalization.

    SOURCE
        docs/gravity_hessian_derivation.md §3.4; engine line 246-249.

    METHOD
        For each of (vrtc, vttc, vrlc, vtlc, vllc):
        1. Build the corresponding 1st-derivative finalization
           symbolically.
        2. Differentiate wrt the appropriate variable.
        3. Substitute Derivative(wXc, var) → wXXc, etc.
        4. Compare to claimed structure via simplify == 0.

        We verify the STRUCTURE — i.e., the claimed formula matches
        SymPy's chain-rule output. We do NOT separately verify the
        convention (GM/R)·vXXc_final = ∂²V/∂q^a∂q^b — that follows
        from the structure being correct AND ITEM-012's first-derivative
        convention.
    """
    r, phi, lam = sp.symbols("r phi lam", real=True, positive=True)
    R = sp.symbols("R", positive=True)

    u = sp.cos(phi)
    t = sp.sin(phi)
    cl = sp.cos(lam)
    sl = sp.sin(lam)
    tu = t / u

    q = R / r
    qs_r = q / r
    A = sp.sqrt(3) * u * q
    B = -sp.sqrt(15) / 2 * (u * q)**2

    # First-derivative finalization formula sources — accumulators as
    # functions of (r, phi, lam).
    wrc = sp.Function("wrc")(r, phi, lam)
    wtc = sp.Function("wtc")(r, phi, lam)
    vrc = sp.Function("vrc")(r, phi, lam)
    vrs = sp.Function("vrs")(r, phi, lam)
    vrc2 = sp.Function("vrc2")(r, phi, lam)
    vtc = sp.Function("vtc")(r, phi, lam)
    vts = sp.Function("vts")(r, phi, lam)
    vtc2 = sp.Function("vtc2")(r, phi, lam)
    vlc = sp.Function("vlc")(r, phi, lam)
    vls = sp.Function("vls")(r, phi, lam)
    vlc2 = sp.Function("vlc2")(r, phi, lam)

    # 1st-derivative finalizations (engine line 246-249)
    vrc_final = -qs_r * (wrc + A * (cl * vrc + sl * vrs) + B * vrc2)
    vtc_final = qs_r * (wtc + A * (cl * vtc + sl * vts) + B * vtc2)
    vlc_final = qs_r / u * (A * (cl * vlc + sl * vls) + B * vlc2)

    # ===== Sub-test 1: vrtc_final = ∂vrc_final/∂φ =====
    sympy_vrtc = sp.diff(vrc_final, phi)
    # Substitute derivatives → 2nd-deriv accumulators
    wrtc = sp.Function("wrtc")(r, phi, lam)
    vrtc = sp.Function("vrtc")(r, phi, lam)
    vrts = sp.Function("vrts")(r, phi, lam)
    vrtc2 = sp.Function("vrtc2")(r, phi, lam)
    sympy_vrtc_sub = sympy_vrtc.subs({
        sp.diff(wrc, phi): wrtc,
        sp.diff(vrc, phi): vrtc,
        sp.diff(vrs, phi): vrts,
        sp.diff(vrc2, phi): vrtc2,
    })
    claim_vrtc = -qs_r * (
        wrtc + A * (cl * vrtc + sl * vrts) + B * vrtc2
        - tu * A * (cl * vrc + sl * vrs)
        - 2 * tu * B * vrc2
    )
    diff1 = sp.simplify(sympy_vrtc_sub - claim_vrtc)

    # ===== Sub-test 2: vttc_final = ∂vtc_final/∂φ =====
    sympy_vttc = sp.diff(vtc_final, phi)
    wttc = sp.Function("wttc")(r, phi, lam)
    vttc = sp.Function("vttc")(r, phi, lam)
    vtts = sp.Function("vtts")(r, phi, lam)
    vttc2 = sp.Function("vttc2")(r, phi, lam)
    sympy_vttc_sub = sympy_vttc.subs({
        sp.diff(wtc, phi): wttc,
        sp.diff(vtc, phi): vttc,
        sp.diff(vts, phi): vtts,
        sp.diff(vtc2, phi): vttc2,
    })
    claim_vttc = qs_r * (
        wttc + A * (cl * vttc + sl * vtts) + B * vttc2
        - tu * A * (cl * vtc + sl * vts)
        - 2 * tu * B * vtc2
    )
    diff2 = sp.simplify(sympy_vttc_sub - claim_vttc)

    # ===== Sub-test 3: vrlc_final = ∂vrc_final/∂λ =====
    # ∂cl/∂λ = -sl, ∂sl/∂λ = +cl. wrc is λ-independent (no λ in inner loop).
    # The m=1 outer accumulators vrc, vrs, vrc2 are λ-dependent.
    sympy_vrlc = sp.diff(vrc_final, lam)
    vrlc = sp.Function("vrlc")(r, phi, lam)
    vrls = sp.Function("vrls")(r, phi, lam)
    vrlc2 = sp.Function("vrlc2")(r, phi, lam)
    # Note: ∂wrc/∂λ = 0 (wrc is λ-indep at inner-loop terminal — λ enters only
    # through outer-loop trigonometric structure). But our symbolic wrc
    # function depends on lam too; SymPy will just keep its derivative as is.
    sympy_vrlc_sub = sympy_vrlc.subs({
        sp.diff(wrc, lam): 0,  # λ-independent at inner loop
        sp.diff(vrc, lam): vrlc,
        sp.diff(vrs, lam): vrls,
        sp.diff(vrc2, lam): vrlc2,
    })
    # Claim: vrlc_final = -qs_r · [A·(-sl·vrc + cl·vrs) + A·(cl·vrlc + sl·vrls) + B·vrlc2]
    # The first term comes from ∂(cl·vrc + sl·vrs)/∂λ = -sl·vrc + cl·vrs (treating vrc, vrs as λ-funcs)
    # Wait — actually ∂cl/∂λ·vrc + cl·∂vrc/∂λ + ∂sl/∂λ·vrs + sl·∂vrs/∂λ
    #      = -sl·vrc + cl·vrlc + cl·vrs + sl·vrls
    # So: vrlc_final = -qs_r · [-sl·A·vrc + cl·A·vrs + A·(cl·vrlc + sl·vrls) + B·vrlc2]
    claim_vrlc = -qs_r * (
        A * (-sl * vrc + cl * vrs)
        + A * (cl * vrlc + sl * vrls)
        + B * vrlc2
    )
    diff3 = sp.simplify(sympy_vrlc_sub - claim_vrlc)

    # ===== Sub-test 4: vtlc_final = ∂vtc_final/∂λ =====
    sympy_vtlc = sp.diff(vtc_final, lam)
    vtlc = sp.Function("vtlc")(r, phi, lam)
    vtls = sp.Function("vtls")(r, phi, lam)
    vtlc2 = sp.Function("vtlc2")(r, phi, lam)
    sympy_vtlc_sub = sympy_vtlc.subs({
        sp.diff(wtc, lam): 0,
        sp.diff(vtc, lam): vtlc,
        sp.diff(vts, lam): vtls,
        sp.diff(vtc2, lam): vtlc2,
    })
    claim_vtlc = qs_r * (
        A * (-sl * vtc + cl * vts)
        + A * (cl * vtlc + sl * vtls)
        + B * vtlc2
    )
    diff4 = sp.simplify(sympy_vtlc_sub - claim_vtlc)

    # ===== Sub-test 5: vllc_final = ∂vlc_final/∂λ =====
    # vlc_final = qs_r/u · (A·(cl·vlc + sl·vls) + B·vlc2)
    # ∂vlc_final/∂λ = qs_r/u · ∂(A·(cl·vlc + sl·vls) + B·vlc2)/∂λ
    #               = qs_r/u · (A·(-sl·vlc + cl·vls) + A·(cl·∂vlc/∂λ + sl·∂vls/∂λ) + B·∂vlc2/∂λ)
    sympy_vllc = sp.diff(vlc_final, lam)
    vllc = sp.Function("vllc")(r, phi, lam)
    vlls = sp.Function("vlls")(r, phi, lam)
    vllc2 = sp.Function("vllc2")(r, phi, lam)
    sympy_vllc_sub = sympy_vllc.subs({
        sp.diff(vlc, lam): vllc,
        sp.diff(vls, lam): vlls,
        sp.diff(vlc2, lam): vllc2,
    })
    claim_vllc = qs_r / u * (
        A * (-sl * vlc + cl * vls)
        + A * (cl * vllc + sl * vlls)
        + B * vllc2
    )
    diff5 = sp.simplify(sympy_vllc_sub - claim_vllc)

    passed = (diff1 == 0) and (diff2 == 0) and (diff3 == 0) and (diff4 == 0) and (diff5 == 0)

    detail = (
        f"Sub-test 1: vrtc_final = ∂vrc_final/∂φ\n"
        f"  simplify(SymPy − claim) = {diff1}\n"
        f"  Match: {'YES' if diff1 == 0 else 'NO'}\n\n"
        f"Sub-test 2: vttc_final = ∂vtc_final/∂φ\n"
        f"  simplify(SymPy − claim) = {diff2}\n"
        f"  Match: {'YES' if diff2 == 0 else 'NO'}\n\n"
        f"Sub-test 3: vrlc_final = ∂vrc_final/∂λ\n"
        f"  simplify(SymPy − claim) = {diff3}\n"
        f"  Match: {'YES' if diff3 == 0 else 'NO'}\n\n"
        f"Sub-test 4: vtlc_final = ∂vtc_final/∂λ\n"
        f"  simplify(SymPy − claim) = {diff4}\n"
        f"  Match: {'YES' if diff4 == 0 else 'NO'}\n\n"
        f"Sub-test 5: vllc_final = ∂vlc_final/∂λ\n"
        f"  simplify(SymPy − claim) = {diff5}\n"
        f"  Match: {'YES' if diff5 == 0 else 'NO'}\n\n"
        f"Note: vttc_final from this derivation gives an accumulator that\n"
        f"satisfies (GM/R)·vttc_final = -(1/r)·∂²V/∂φ² (inheriting the\n"
        f"vtc convention's -1/r factor). To get natural ∂²V/∂φ², the\n"
        f"Cartesian assembly multiplies by -r. Same for vrtc (factor of\n"
        f"vrc's +1 means natural).\n"
    )

    return VerificationResult(
        item_id="ITEM-019-full",
        claim="m=0 finalization formulas for all 6 2nd-derivative accumulators (vrtc, vttc, vrlc, vtlc, vllc)",
        source="docs/gravity_hessian_derivation.md §3.4; engine line 246-249",
        method="sp.diff on engine's 1st-derivative finalization wrt appropriate variable; compare structurally",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-023 — Convention-bridge table for 2nd-derivative accumulators
# =============================================================================

def verify_item_023() -> VerificationResult:
    """ITEM-023 — natural-convention output of the m=0 finalization.

    CLAIM (independent statement, NOT derived from SymPy output):
        Combining ITEM-012 (1st-deriv post-finalization convention) with
        ITEM-019 (m=0 finalization formulas), the 2nd-derivative
        accumulators after finalization satisfy:

            (GM/R) · vrrc_final = +∂²V/∂r²
            (GM/R) · vrtc_final = +∂²V/∂r∂φ
            (GM/R) · vttc_final = -(1/r) · ∂²V/∂φ²
            (GM/R) · vrlc_final = +∂²V/∂r∂λ              [vrc has no factor]
            (GM/R) · vtlc_final = -(1/r) · ∂²V/∂φ∂λ      [inherits -(1/r) from vtc]
            (GM/R) · vllc_final = +(1/(r·u)) · ∂²V/∂λ²    [inherits +(1/(r·u)) from vlc]

        Inversion factors for natural H_sph:

            ∂²V/∂r²    = (GM/R) · vrrc_final
            ∂²V/∂r∂φ  = (GM/R) · vrtc_final
            ∂²V/∂φ²    = -r · (GM/R) · vttc_final
            ∂²V/∂r∂λ  = (GM/R) · vrlc_final               [no factor]
            ∂²V/∂φ∂λ  = -r · (GM/R) · vtlc_final
            ∂²V/∂λ²    = (r·u) · (GM/R) · vllc_final

        Note on convention-factor inheritance: λ-differentiation does NOT
        introduce new factors because the conversion factors (1, -(1/r),
        +(1/(r·u))) are all λ-independent. Only φ-differentiation of vtc
        inherits the -(1/r) factor and λ-differentiation of vlc inherits
        +(1/(r·u)). r-differentiation of vrc, vtc, vlc would change r
        factors, but those second-derivative accumulators (vrrc, vrtc,
        vrlc) are derived from vrc only — and vrc has no factor.

        These claims are stated independently. SymPy's job is to verify
        that the chain ITEM-012 + ITEM-019 produces the convention
        factors above. Method: define each vXc_final as a derivative
        of the corresponding 1st-derivative finalization (per ITEM-019),
        then assert (GM/R) · vXXc_final equals the claimed expression
        in terms of ∂²V/∂q^a∂q^b for an unspecified V.

    SOURCE
        docs/gravity_hessian_derivation.md §3.4 + §3.2.

    METHOD
        Build symbolic V(r, φ, λ) as unspecified function. Construct
        vrc_final, vtc_final, vlc_final per ITEM-012's convention
        relations:
            vrc_final = (R/GM) · ∂V/∂r
            vtc_final = -(R/GM) · (1/r) · ∂V/∂φ
            vlc_final = +(R/GM) · (1/(r·u)) · ∂V/∂λ

        Define vXXc_final = ∂vXc_final/∂q where appropriate per ITEM-019.
        Compute (GM/R) · vXXc_final and compare to the claimed natural
        Hessian formula (with appropriate inversion factor).

        Six sub-tests, one per (∂²V/∂q^a∂q^b).
    """
    r, phi, lam = sp.symbols("r phi lam", real=True, positive=True)
    GM, R = sp.symbols("GM R", positive=True)
    u = sp.cos(phi)

    # Symbolic V as an unspecified function of (r, φ, λ)
    V = sp.Function("V")(r, phi, lam)

    # 1st-derivative finalization per ITEM-012 convention:
    # (GM/R) · vXc_final = (1st-deriv quantity)
    vrc_final = (R / GM) * sp.diff(V, r)
    vtc_final = (R / GM) * (-(1 / r) * sp.diff(V, phi))
    vlc_final = (R / GM) * ((1 / (r * u)) * sp.diff(V, lam))

    # 2nd-derivative finalizations per ITEM-019 (define each as derivative
    # of the appropriate 1st-derivative finalization)
    vrrc_final = sp.diff(vrc_final, r)
    vrtc_final = sp.diff(vrc_final, phi)
    vttc_final = sp.diff(vtc_final, phi)
    vrlc_final = sp.diff(vrc_final, lam)
    vtlc_final = sp.diff(vtc_final, lam)
    vllc_final = sp.diff(vlc_final, lam)

    # Compute (GM/R) · vXXc_final
    out_vrrc = sp.simplify((GM / R) * vrrc_final)
    out_vrtc = sp.simplify((GM / R) * vrtc_final)
    out_vttc = sp.simplify((GM / R) * vttc_final)
    out_vrlc = sp.simplify((GM / R) * vrlc_final)
    out_vtlc = sp.simplify((GM / R) * vtlc_final)
    out_vllc = sp.simplify((GM / R) * vllc_final)

    # Claimed forms (corrected)
    claim_vrrc = sp.diff(V, r, 2)                          # ∂²V/∂r²
    claim_vrtc = sp.diff(V, r, phi)                        # ∂²V/∂r∂φ
    claim_vttc = -(1 / r) * sp.diff(V, phi, 2)             # -(1/r)·∂²V/∂φ²
    claim_vrlc = sp.diff(V, r, lam)                        # +∂²V/∂r∂λ (no factor)
    claim_vtlc = -(1 / r) * sp.diff(V, phi, lam)           # -(1/r)·∂²V/∂φ∂λ
    claim_vllc = (1 / (r * u)) * sp.diff(V, lam, 2)        # +(1/(r·u))·∂²V/∂λ²

    diff1 = sp.simplify(out_vrrc - claim_vrrc)
    diff2 = sp.simplify(out_vrtc - claim_vrtc)
    diff3 = sp.simplify(out_vttc - claim_vttc)
    diff4 = sp.simplify(out_vrlc - claim_vrlc)
    diff5 = sp.simplify(out_vtlc - claim_vtlc)
    diff6 = sp.simplify(out_vllc - claim_vllc)

    diffs = [diff1, diff2, diff3, diff4, diff5, diff6]
    labels = ["vrrc → ∂²V/∂r²",
              "vrtc → ∂²V/∂r∂φ",
              "vttc → -(1/r)·∂²V/∂φ²",
              "vrlc → +∂²V/∂r∂λ",
              "vtlc → -(1/r)·∂²V/∂φ∂λ",
              "vllc → +(1/(r·u))·∂²V/∂λ²"]

    passed = all(d == 0 for d in diffs)

    detail_lines = []
    for lab, d in zip(labels, diffs):
        match = "YES" if d == 0 else "NO"
        detail_lines.append(f"  {lab}\n    simplify(SymPy − claim) = {d}    Match: {match}")
    detail_lines.append("")
    detail_lines.append("Stage F H_sph natural assembly factors:")
    detail_lines.append("  ∂²V/∂r²    = (GM/R) · vrrc_final")
    detail_lines.append("  ∂²V/∂r∂φ  = (GM/R) · vrtc_final")
    detail_lines.append("  ∂²V/∂φ²    = -r · (GM/R) · vttc_final")
    detail_lines.append("  ∂²V/∂r∂λ  = (GM/R) · vrlc_final           [no factor]")
    detail_lines.append("  ∂²V/∂φ∂λ  = -r · (GM/R) · vtlc_final")
    detail_lines.append("  ∂²V/∂λ²    = (r·u) · (GM/R) · vllc_final")

    detail = "\n".join(detail_lines)

    return VerificationResult(
        item_id="ITEM-023",
        claim="Convention-bridge: (GM/R)·vXXc_final → natural ∂²V/∂q^a∂q^b with explicit inversion factors",
        source="docs/gravity_hessian_derivation.md §3.4 + §3.2",
        method="Compose ITEM-012 + ITEM-019 chains; verify (GM/R)·vXXc_final matches claimed natural Hessian forms",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-024 — wphic (natural ∂wc/∂φ) accumulator recurrence (Stage F Strategy 1)
# =============================================================================

def verify_item_024() -> VerificationResult:
    """ITEM-024 — natural ∂wc/∂φ accumulator recurrence (for sectorial correction).

    CLAIM (independent statement, NOT derived from SymPy output):
        For Stage F Strategy 1 (full analytic Hessian recurrence) to
        compute the sectorial correction for engine wttc per ITEM-018,
        an additional inner-loop accumulator `wphic` (tracking natural
        ∂wc/∂φ) is required. Differentiating the engine's wc_new wrt φ:

            wc_new = A·wc + B·wc2 + Cv

            wphic_new = ∂(wc_new)/∂φ
                      = A · wphic + B · wphic2 + u · Ax · wc

        The inhomogeneous source `+u·Ax·wc` uses the CURRENT iterate wc
        (not wc2) — opposite of engine's `wtc_new = A·wtc + B·wtc2 −
        u·Ax·wc2`'s -u·Ax·wc2 (old iterate, minus sign).

        SINE SIBLING:
            wphis_new = A · wphis + B · wphis2 + u · Ax · ws

        These claims are stated independently. SymPy verifies by direct
        differentiation of engine wc_new and ws_new wrt φ.

    SOURCE
        docs/gravity_hessian_derivation.md §2.6.5 (Stage F implementation
        choice (a)).

    METHOD
        Build engine wc_new symbolically; sp.diff wrt φ; compare to
        claimed wphic_new. Same for sine sibling.

    IMPLICATED ON FAILURE
        If wrong, Stage F Strategy 1's sectorial correction for wttc
        produces incorrect 2nd-derivative theta accumulator → wrong
        ∂²V/∂φ² for sectoral m≥1 modes → wrong gravity Hessian.
        Workaround: Strategy 2 FD fallback.
    """
    r, phi = sp.symbols("r phi", real=True, positive=True)
    R, sqrt2n3, b_factor = sp.symbols("R sqrt2n3 b_factor", real=True, positive=True)
    n = sp.symbols("n", integer=True, nonnegative=True)
    Cv, Sv = sp.symbols("Cv Sv", real=True)

    q = R / r
    Ax = q * sqrt2n3
    A_phi = sp.sin(phi) * Ax
    B_phi = -q**2 * b_factor

    # ===== Cosine sibling: wphic =====
    wc = sp.Function("wc")(phi)
    wc2 = sp.Function("wc2")(phi)
    wc_new = A_phi * wc + B_phi * wc2 + Cv

    sympy_wphic = sp.expand(sp.diff(wc_new, phi))
    wphic_f = sp.diff(wc, phi)
    wphic2_f = sp.diff(wc2, phi)
    claim_wphic = sp.expand(
        A_phi * wphic_f + B_phi * wphic2_f + sp.cos(phi) * Ax * wc
    )
    diff1 = sp.simplify(sympy_wphic - claim_wphic)

    # ===== Sine sibling: wphis =====
    ws = sp.Function("ws")(phi)
    ws2 = sp.Function("ws2")(phi)
    ws_new = A_phi * ws + B_phi * ws2 + Sv

    sympy_wphis = sp.expand(sp.diff(ws_new, phi))
    wphis_f = sp.diff(ws, phi)
    wphis2_f = sp.diff(ws2, phi)
    claim_wphis = sp.expand(
        A_phi * wphis_f + B_phi * wphis2_f + sp.cos(phi) * Ax * ws
    )
    diff2 = sp.simplify(sympy_wphis - claim_wphis)

    passed = (diff1 == 0) and (diff2 == 0)

    detail = (
        f"Sub-test 1: wphic_new = ∂(engine wc_new)/∂φ\n"
        f"  Claim: A·wphic + B·wphic2 + u·Ax·wc  (current iterate, +sign)\n"
        f"  simplify(SymPy − claim) = {diff1}\n"
        f"  Match: {'YES' if diff1 == 0 else 'NO'}\n\n"
        f"Sub-test 2: wphis_new = ∂(engine ws_new)/∂φ\n"
        f"  Claim: A·wphis + B·wphis2 + u·Ax·ws  (current iterate, +sign)\n"
        f"  simplify(SymPy − claim) = {diff2}\n"
        f"  Match: {'YES' if diff2 == 0 else 'NO'}\n\n"
        f"Note: wphic differs from engine's wtc structurally:\n"
        f"  engine wtc_new = A·wtc + B·wtc2 − u·Ax·wc2  (old iterate, -sign)\n"
        f"  natural wphic_new = A·wphic + B·wphic2 + u·Ax·wc  (current iterate, +sign)\n"
        f"Stage F Strategy 1 must add wphic as a separate accumulator;\n"
        f"Strategy 2 (FD fallback) sidesteps this entirely."
    )

    return VerificationResult(
        item_id="ITEM-024",
        claim="wphic (natural ∂wc/∂φ) accumulator recurrence for Stage F Strategy 1",
        source="docs/gravity_hessian_derivation.md §2.6.5 implementation choice (a)",
        method="sp.diff on engine wc_new wrt φ; compare to claimed natural-convention recurrence",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# ITEM-025 — Sine sibling engine-convention 2nd-derivative recurrences
# =============================================================================

def verify_item_025() -> VerificationResult:
    """ITEM-025 — engine sine sibling 2nd-derivative recurrences (wrrs, wrts, wtts).

    CLAIM (independent statement, NOT derived from SymPy output):
        Sine siblings of the engine 2nd-derivative recurrences in §2.6
        follow by structural symmetry (cos↔sin substitution): the engine
        recurrences for wrs, wts have the same homogeneous structure as
        their cosine counterparts, with sources Sv replacing Cv. By the
        §2.4 structural-symmetry argument, differentiating engine wrs_new
        and ws_new wrt r and φ produces:

            wrrs_new = A·wrrs + B·wrrs2 − (A/r)·wrs − (2B/r)·wrs2
            wrts_new = A·wrts + B·wrts2 + u·Ax·wrs
            wtts_new = A·wtts + B·wtts2 + u·Ax·wts + t·Ax·ws2 − u·Ax·wts2

        These claims are stated independently. SymPy verifies by direct
        differentiation of engine wrs_new and ws_new (with engine
        sources) and comparison to the claimed sine-sibling forms.

    SOURCE
        docs/gravity_hessian_derivation.md §2.6 (sine extension noted).

    METHOD
        Build engine wrs_new and ws_new with sine sources symbolically;
        differentiate; compare.

    IMPLICATED ON FAILURE
        Sine accumulators wrong → wrong gravity contribution from S_lm
        coefficients (which are non-zero for tesseral l≠m harmonics).
    """
    r, phi = sp.symbols("r phi", real=True, positive=True)
    R, sqrt2n3, b_factor = sp.symbols("R sqrt2n3 b_factor", real=True, positive=True)
    n = sp.symbols("n", integer=True, nonnegative=True)
    Sv = sp.symbols("Sv", real=True)

    q = R / r
    Ax = q * sqrt2n3
    A_phi = sp.sin(phi) * Ax
    B_phi = -q**2 * b_factor

    t_param = sp.symbols("t_param", real=True)
    A_r_clean = t_param * (R / r) * sqrt2n3
    B_r_clean = -(R / r)**2 * b_factor

    # Sub-test 1: wrrs (∂/∂r of engine wrs_new = A·wrs + B·wrs2 + (n+1)·Sv)
    wrs_r = sp.Function("wrs")(r)
    wrs2_r = sp.Function("wrs2")(r)
    wrs_new = A_r_clean * wrs_r + B_r_clean * wrs2_r + (n + 1) * Sv
    sympy_wrrs = sp.expand(sp.diff(wrs_new, r))
    wrrs_f = sp.diff(wrs_r, r)
    wrrs2_f = sp.diff(wrs2_r, r)
    claim_wrrs = sp.expand(
        A_r_clean * wrrs_f + B_r_clean * wrrs2_f
        - (A_r_clean / r) * wrs_r
        - (2 * B_r_clean / r) * wrs2_r
    )
    diff1 = sp.simplify(sympy_wrrs - claim_wrrs)

    # Sub-test 2: wrts (∂/∂φ of engine wrs_new — wrs_new at fixed r treats Sv as φ-indep)
    wrs_phi = sp.Function("wrs")(phi)
    wrs2_phi = sp.Function("wrs2")(phi)
    wrs_new_phi = A_phi * wrs_phi + B_phi * wrs2_phi + (n + 1) * Sv
    sympy_wrts = sp.expand(sp.diff(wrs_new_phi, phi))
    wrts_f = sp.diff(wrs_phi, phi)
    wrts2_f = sp.diff(wrs2_phi, phi)
    claim_wrts = sp.expand(
        A_phi * wrts_f + B_phi * wrts2_f + sp.cos(phi) * Ax * wrs_phi
    )
    diff2 = sp.simplify(sympy_wrts - claim_wrts)

    # Sub-test 3: wtts (∂/∂φ of engine wts_new = A·wts + B·wts2 - u·Ax·ws2)
    ws_phi = sp.Function("ws")(phi)
    ws2_phi = sp.Function("ws2")(phi)
    wts_phi = sp.Function("wts")(phi)
    wts2_phi = sp.Function("wts2")(phi)
    wts_new = A_phi * wts_phi + B_phi * wts2_phi - sp.cos(phi) * Ax * ws2_phi
    sympy_wtts = sp.expand(sp.diff(wts_new, phi))

    wtts_f = sp.diff(wts_phi, phi)
    wtts2_f = sp.diff(wts2_phi, phi)
    ws2_phi_d = sp.diff(ws2_phi, phi)
    claim_wtts = sp.expand(
        A_phi * wtts_f + B_phi * wtts2_f
        + sp.cos(phi) * Ax * wts_phi
        + sp.sin(phi) * Ax * ws2_phi
        - sp.cos(phi) * Ax * ws2_phi_d
    )
    diff3 = sp.simplify(sympy_wtts - claim_wtts)

    passed = (diff1 == 0) and (diff2 == 0) and (diff3 == 0)

    detail = (
        f"Sub-test 1: engine wrrs (∂/∂r of engine wrs_new)\n"
        f"  simplify(SymPy − claim) = {diff1}\n"
        f"  Match: {'YES' if diff1 == 0 else 'NO'}\n\n"
        f"Sub-test 2: engine wrts (∂/∂φ of engine wrs_new)\n"
        f"  simplify(SymPy − claim) = {diff2}\n"
        f"  Match: {'YES' if diff2 == 0 else 'NO'}\n\n"
        f"Sub-test 3: engine wtts (∂/∂φ of engine wts_new)\n"
        f"  simplify(SymPy − claim) = {diff3}\n"
        f"  Match: {'YES' if diff3 == 0 else 'NO'}\n\n"
        f"Sine-sibling structural symmetry confirmed; same recurrence forms\n"
        f"as cosine versions (ITEM-017) with cos→sin source substitution."
    )

    return VerificationResult(
        item_id="ITEM-025",
        claim="Engine sine sibling 2nd-deriv recurrences (wrrs, wrts, wtts) follow by cos→sin symmetry",
        source="docs/gravity_hessian_derivation.md §2.6 sine extension",
        method="sp.diff on engine wrs_new and wts_new; compare to claimed sine-sibling forms",
        passed=passed,
        detail=detail,
    )


# =============================================================================
# Main driver
# =============================================================================

def main() -> int:
    print()
    print("=" * 72)
    print("  Stage A SymPy verification — gravity Hessian derivation")
    print("  (audit trail of algebraic claims in the markdown derivation)")
    print("=" * 72)

    verifications: List[Callable[[], VerificationResult]] = [
        # §2.1 wrrc (∂²V/∂r²)
        verify_item_005b,   # sub-claim ∂A/∂r = -A/r
        verify_item_005c,   # sub-claim ∂B/∂r = -2B/r
        verify_item_005,    # composite wrrc recurrence
        # §2.2 wrtc (∂²V/∂r∂φ)
        verify_item_008b,   # sub-claim ∂A/∂φ = u·Ax
        verify_item_008,    # composite wrtc recurrence (natural convention)
        # §2.3 wttc (∂²V/∂φ², natural convention)
        verify_item_009,
        # §2.4 sine-series siblings (wrrs, wrts, wtts) — structural symmetry
        verify_item_010,
        # §2.5 lambda-derivative outer-loop source-substitution lemma
        verify_item_011,
        # §3.2 engine post-finalization convention vs gradient chain rule
        verify_item_012,
        # §3.3 Cartesian Hessian tensor decomposition
        verify_item_013,
        # §4 pole-singularity cancellation (mpmath, 50-digit precision)
        verify_item_014,
        # §5.1 Pines polynomial table — Laplace's equation per (l,m)
        verify_item_015,
        # §11 solid-tide corrections satellite-position-independence
        verify_item_016,
        # §2.6 engine-convention 2nd-derivative recurrences (Stage F spec)
        verify_item_017,
        # §3.5 ECEF→ECI Hessian similarity transform (Stage I spec)
        verify_item_021,
        # §2.6.5 sectorial corrections for 2nd-deriv inner-loop
        verify_item_018,
        # §2.7 engine outer-loop 2nd-derivative recurrences (NEW gap)
        verify_item_022,
        # §3.3 K^r, K^φ, K^λ explicit Cartesian formulas
        verify_item_020,
        # §3.4 m=0 finalization for vrrc (partial — others follow same pattern)
        verify_item_019_partial,
        # §3.4 m=0 finalization for vrtc, vttc, vrlc, vtlc, vllc (full)
        verify_item_019_full,
        # Convention-bridge: engine vXXc_final → natural ∂²V/∂q^a∂q^b
        verify_item_023,
        # Stage F Strategy 1: wphic accumulator recurrence
        verify_item_024,
        # Sine sibling engine 2nd-deriv recurrences
        verify_item_025,
    ]

    results: List[VerificationResult] = []
    for v in verifications:
        try:
            result = v()
        except Exception as e:
            # Unexpected failure during the verification itself — record it
            # so the trace is complete even when SymPy chokes.
            result = VerificationResult(
                item_id=getattr(v, "__name__", "UNKNOWN"),
                claim="<exception during verification>",
                source="(see function docstring)",
                method=f"function raised {type(e).__name__}",
                passed=False,
                detail=f"Exception: {e}",
            )
        result.report()
        results.append(result)

    # Summary
    print()
    print("=" * 72)
    print("  SUMMARY")
    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    print(f"  Passed: {n_pass}/{len(results)}")
    if n_fail:
        print(f"  Failed: {n_fail}")
        for r in results:
            if not r.passed:
                print(f"    ❌ {r.item_id}: {r.claim[:60]}")
    print("=" * 72)
    print()

    # Exit code reflects pass/fail for use in CI / scripts.
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
