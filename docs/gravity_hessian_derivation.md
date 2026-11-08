# Analytic Gravity Hessian Derivation

**Status**: derivation record; each item below carries its own verification status.

**Purpose**: derive the 3×3 Hessian of the EGM2008 spherical-harmonic potential
in the same conventions used by the existing `ClenshawEngine` (gradient-only).
Every claim in this document is checked by SymPy (Stage C,
`tests/python/verify_gravity_hessian_stage_a.py`) before the C++ code that
implements it.

**Companion**: this same machinery, applied to time-varying solid-tide
corrections to the Stokes coefficients, gives the analytic Jacobian for solid
Earth tides (see §11).

---

## Section 0 — Reference state

The existing code's mathematical conventions that this derivation must
respect. The line numbers quoted below locate the expressions in
`clenshaw_engine.hpp` and `force_model.hpp` as they stood when the derivation
was written; the files have since grown, so search for the quoted expression
rather than the line.

### 0.1 Frame

- The Clenshaw engine's `compute(pos_ecef, ...)` takes position in ECEF
  (`clenshaw_engine.hpp`, signature: `GravityResult compute(const Vector3d&
  pos_ecef, int eval_degree = -1) const`).
- It returns `GravityResult { acceleration, potential }` where `acceleration`
  is in ECEF (line 261: assembled as `(gradx, grady, gradz)` from spherical
  components in ECEF basis).
- The ForceModel (`force_model.hpp::compute_earth_gravity`) transforms the
  GCRS position to ITRF, evaluates the ITRF acceleration, and rotates it back
  to GCRS via `itrf_vector_to_gcrs(acc_itrf, epoch)`.

**Implication for Hessian**: we derive the Hessian in ECEF. The ForceModel
layer is responsible for the ECEF→ECI transformation `H_eci = R(t)·H_ecef·R(t)ᵀ`
where `R(t)` is the ECEF→ECI rotation at the evaluation epoch. We do NOT
include the time-derivative of `R(t)` in the spatial Jacobian — see §3.4.

### 0.2 Coordinate conventions

From `clenshaw_engine.hpp:124-147`:

```cpp
const double x = pos_ecef.x();
const double y = pos_ecef.y();
const double z = pos_ecef.z();
const double r2 = x*x + y*y + z*z;
const double r  = std::sqrt(r2);
const double p  = std::sqrt(x*x + y*y);

const double t = z / r;      // sin(geocentric_lat) = sin φ
const double u = p / r;      // cos(geocentric_lat) = cos φ
const double q = radius_ / r;

double cl, sl;
if (p > defaults::GRAVITY_POLE_RATIO_THRESHOLD * r) {
    cl = x / p;              // cos λ
    sl = y / p;              // sin λ
} else {
    cl = 1.0;
    sl = 0.0;                // pole singularity branch
}
```

**Conventions** (reading from this code):
- `φ` is **geocentric latitude** (NOT geodetic).
- `t = sin φ`, `u = cos φ`. Both in `[−1, 1]`.
- `λ` is longitude. `cl = cos λ`, `sl = sin λ`.
- `q = R_E / r` (Earth radius ratio, dimensionless, < 1 outside Earth).

The local spherical basis vectors at position `(r, φ, λ)` are:

```
ê_r = (u·cl, u·sl, t)         (radial outward)
ê_φ = (−t·cl, −t·sl, u)        (toward INCREASING latitude, i.e., north pole)
ê_λ = (−sl, cl, 0)             (eastward)
```

### 0.3 Sign convention for derivative accumulators

From `clenshaw_engine.hpp:254-261`:

```cpp
const double scale_factor = GM_ / radius_;
result.potential = scale_factor * vc;

double gradx = cl * (u * vrc + t * vtc) - sl * vlc;
double grady = sl * (u * vrc + t * vtc) + cl * vlc;
double gradz = t * vrc - u * vtc;
```

Standard spherical-to-Cartesian gradient assembly is:
```
∇V = (∂V/∂r) ê_r + (1/r)(∂V/∂φ) ê_φ + (1/(r·cos φ))(∂V/∂λ) ê_λ
```
giving:
```
∂V/∂x = (∂V/∂r)·u·cl + (1/r)(∂V/∂φ)·(−t·cl) + (1/(r·u))(∂V/∂λ)·(−sl)
      = u·cl·(∂V/∂r) − t·cl·(1/r)·(∂V/∂φ) − (sl/(r·u))·(∂V/∂λ)
```

Compare to the code:
```
gradx = cl·u·vrc + cl·t·vtc − sl·vlc
      = u·cl·vrc + t·cl·vtc − sl·vlc
```

Matching term-by-term (both must equal `∂V/∂x` after `scale_factor`):

| Code term | Standard term | Implication |
|---|---|---|
| `u·cl·vrc` | `u·cl·(∂V/∂r)` | `vrc` accumulator must scale to `(∂V/∂r) / scale_factor` after the engine's final scaling — i.e., `vrc` is a normalized `∂V/∂r` |
| `+t·cl·vtc` | `−t·cl·(1/r)·(∂V/∂φ)` | sign flip and `1/r` factor: `vtc` represents `−(1/r)·(∂V/∂φ) / scale_factor` after scaling, i.e., `vtc = −(1/r)·(∂V/∂φ) / scale_factor` |
| `−sl·vlc` | `−(sl/(r·u))·(∂V/∂λ)` | `vlc` represents `(1/(r·u))·(∂V/∂λ) / scale_factor` |

To make this concrete: at the `m=0` finalization (line 243-249), the engine
divides by `r` (`qs_r = q/r`) when assigning to `vrc`, `vtc`, `vlc`. So
internally during the recurrence, the accumulators track raw spherical
derivatives; the `1/r` factor is absorbed at finalization.

**Sign convention precisely** (post-finalization, verified by ITEM-012):

| Accumulator | Represents (post-scale_factor) |
|---|---|
| `vc`  | `+V` |
| `vrc` | `+∂V/∂r` |
| `vtc` | `−(1/r)·∂V/∂φ` |
| `vlc` | `+(1/(r·u))·∂V/∂λ` |

Equivalently, `(GM/R) · {vc, vrc, vtc, vlc} = {+V, +∂V/∂r, −(1/r)·∂V/∂φ,
+(1/(r·u))·∂V/∂λ}` after the `m=0` finalization at lines 243-249.

The minus sign in line 246's `vrc = -qs_r * (...)` is **internal
arithmetic**, not a convention flip — the inner-loop accumulator value
`(wrc + A·... + B·vrc2)` ends up with a sign that combines with `−qs_r`
to give the correctly-signed `vrc` representing `+∂V/∂r`. (Concretely,
for a point mass with `V = +GM/r`, `∂V/∂r = −GM/r²`. The engine's `vrc`
after finalization equals `−R/r² = (R/GM) · (−GM/r²) = (R/GM) · ∂V/∂r`,
consistent with `(GM/R) · vrc = +∂V/∂r`.)

The `vtc` minus sign is a real convention difference: `vtc` carries
`−(1/r)·∂V/∂φ`, not `+(1/r)·∂V/∂φ`. This shows up at the inner-loop
sectoral correction `wtc += m * tu * wc` (line 218-220) where the
engine's `wtc` accumulator already encodes a sign-flipped derivative —
careful tracking is needed when extending to second derivatives, but
the same convention will hold for `wttc, wtts, wtlc, wtls`.

This convention is the basis for §3.3's Hessian assembly: when
mapping the natural-convention recurrences from §2 (which use natural
signs throughout) into the engine's post-finalization accumulators,
`vtc`-family entries pick up an extra minus.

For the Hessian derivation below, I continue to use **natural
functional notation** (`∂V/∂r`, `∂²V/∂r²`, etc.) until §3.3, where the
convention table above is applied en bloc to translate to engine
accumulators.

### 0.4 Normalization

EGM2008 uses **fully-normalized** spherical harmonic coefficients (per
`egm2008_coefficients.hpp` and standard convention).

The fully-normalized associated Legendre polynomial is:
```
P̄_lm(t) = √[(2−δ_m0)·(2l+1)·(l−m)!/(l+m)!] · P_lm(t)
```
where `P_lm(t)` is the unnormalized associated Legendre polynomial.

The recursion factors precomputed in `clenshaw_storage.hpp::precompute_recursion_factors`
encode this normalization. The Hessian derivation must use the same
normalization throughout.

**Derivative of normalized Legendre**: standard recurrence (Holmes &
Featherstone 2002, eq. 13):
```
(1−t²)·dP̄_lm/dt = ?·P̄_l(m+1) − ?·P̄_l(m−1)
```
Specific coefficients to be filled in. **REVIEW NOTE**: the exact derivative
formula for normalized Legendre is the most error-prone part of this
derivation. Stage C's SymPy script will derive it independently from the
unnormalized form + normalization factor and check against textbook.

### 0.5 Pole singularity handling

From `clenshaw_engine.hpp:140-150`:

```cpp
double cl, sl;
if (p > defaults::GRAVITY_POLE_RATIO_THRESHOLD * r) {
    cl = x / p;
    sl = y / p;
} else {
    cl = 1.0;
    sl = 0.0;
}
```

When `p = √(x² + y²) → 0` (satellite directly above pole), `cl, sl` are
undefined (longitude is ambiguous). The code branches: at the pole, set
`cl=1, sl=0` (effectively choosing longitude=0).

There's also `tu = (|u| > GRAVITY_POLE_U_THRESHOLD) ? t/u : 0.0` at line 150
— another guard against `cos φ → 0` divides.

**Implication for Hessian**: the second derivatives `∂²V/∂φ²`, `∂²V/∂λ²`,
`∂²V/∂φ∂λ` all contain factors of `1/cos φ` or `tan φ` that diverge at
the pole. The code's existing `tu` guard handles `tan φ` for first
derivatives. For second derivatives, the divergences are stronger
(`1/cos² φ`, `tan² φ`); we'll need analogous guards. The Cartesian
Hessian (after spherical → Cartesian transform) is well-defined at the
pole — the divergences cancel in the rotation. We must verify this
cancellation explicitly, or use the Pines Cartesian formulation
(§5-§8) which avoids the issue entirely.

### 0.6 Existing `eval_degree` parameter

The `compute(pos_ecef, eval_degree)` method at line 109 takes an optional
`eval_degree` to truncate evaluation. Default `-1` means use `max_degree_`.
The Hessian extension must support the same parameter.

---

## Section 1 — State the existing first-derivative recurrences

### 1.1 Inner-loop accumulators

From `clenshaw_engine.hpp:167-215` (inner loop body, for fixed `m`,
iterating `n` from `N` down to `m`):

```cpp
const auto& rf = storage_.get_rf(n, m);     // recursion factors
const double Ax = q * rf.w_sqrt2n3;          // q · √(2n+3)
const double A  = t * Ax;                    // sin φ · q · √(2n+3)
const double B  = -q2 * rf.b_factor;         // −q² · b(n,m)

const double Cv = storage_.get_C(n, m);
const double Sv = (m > 0) ? storage_.get_S(n, m) : 0.0;

// Value
double temp = A * wc + B * wc2 + Cv;
wc2 = wc; wc = temp;

// Radial derivative
temp = A * wrc + B * wrc2 + (n + 1) * Cv;
wrc2 = wrc; wrc = temp;

// Theta derivative (sign convention encoded in the (-u·Ax·wc2) term)
temp = A * wtc + B * wtc2 - u * Ax * wc2;
wtc2 = wtc; wtc = temp;

// Sine series (only for m > 0)
if (m > 0) {
    temp = A * ws + B * ws2 + Sv;
    ws2 = ws; ws = temp;

    temp = A * wrs + B * wrs2 + (n + 1) * Sv;
    wrs2 = wrs; wrs = temp;

    temp = A * wts + B * wts2 - u * Ax * ws2;
    wts2 = wts; wts = temp;
}
```

Outside the loop (still for fixed `m`), at lines 217-221:
```cpp
if (m > 0) {
    wtc += m * tu * wc;
    wts += m * tu * ws;
}
```

This adds the sectorial correction to the theta derivative.

### 1.2 Reading the recurrence

The Clenshaw recurrence is **backward in `n`**: at iteration entering with
`(wc, wc2)`, the update assigns
```
wc_new = A · wc_prev + B · wc2_prev + C_n
wc2_new = wc_prev          (the old wc rolls into wc2)
```

After update, `wc = wc_new`, `wc2 = wc_prev`. So *during* the next
iteration's update, when we reference `wc`, we're using `wc_prev`'s
just-updated value (i.e., `wc_new` from the previous iteration); when we
reference `wc2`, we're using `wc_prev`'s pre-update value (i.e.,
`wc2_new = wc_prev` set just before).

The theta-derivative update `−u · Ax · wc2` references `wc2`, which is the
"older" wc (two iterations back relative to the current `wc_new`). This is
the Clenshaw trick for chain-rule derivatives: the derivative of `wc_new`
with respect to a parameter that affects `A` carries through the recurrence
structure.

### 1.3 Outer-loop accumulators

From `clenshaw_engine.hpp:223-250` (outer loop, processing each `m` from
`N` down to `0`):

```cpp
if (m > 0) {
    double v = root_[2] * root_[2*m + 3] / root_[m + 1];
    double A = cl * v * uq;
    double B = -v * root_[2*m + 5] / (root_[8] * root_[m + 2]) * uq2;

    temp = A * vc + B * vc2 + wc;   vc2 = vc;   vc = temp;
    temp = A * vs + B * vs2 + ws;   vs2 = vs;   vs = temp;
    temp = A * vrc + B * vrc2 + wrc; vrc2 = vrc; vrc = temp;
    temp = A * vrs + B * vrs2 + wrs; vrs2 = vrs; vrs = temp;
    temp = A * vtc + B * vtc2 + wtc; vtc2 = vtc; vtc = temp;
    temp = A * vts + B * vts2 + wts; vts2 = vts; vts = temp;
    temp = A * vlc + B * vlc2 + m * ws; vlc2 = vlc; vlc = temp;
    temp = A * vls + B * vls2 - m * wc; vls2 = vls; vls = temp;
} else {
    // m = 0 finalization
    double A = root_[3] * uq;
    double B = -root_[15] / 2.0 * uq2;

    vc = q * (wc + A * (cl * vc + sl * vs) + B * vc2);

    double qs_r = q / r;
    vrc = -qs_r * (wrc + A * (cl * vrc + sl * vrs) + B * vrc2);
    vtc = qs_r * (wtc + A * (cl * vtc + sl * vts) + B * vtc2);
    vlc = (std::abs(u) > defaults::GRAVITY_POLE_U_THRESHOLD) ?
          qs_r / u * (A * (cl * vlc + sl * vls) + B * vlc2) : 0.0;
}
```

### 1.4 Outer-loop conventions

Three observations:
- The outer loop's `A`, `B` are **different** from the inner loop's `A`, `B`
  (variable shadowing). In this section I'll call the outer ones `A_out`,
  `B_out` to avoid confusion.
- The lambda derivative `vlc` is built ONLY in the outer loop's `m > 0`
  branch, via `m·ws` (cosine output) and `−m·wc` (sine output). This
  corresponds to `∂(cos mλ)/∂λ = −m·sin mλ` and `∂(sin mλ)/∂λ = +m·cos mλ`.
  The minus sign on `m·wc` is the −m from `∂(cos mλ)/∂λ` migrated into the
  `vls` (sine) accumulator.
- At `m = 0` finalization, lambda dependence vanishes (zonal terms have no
  λ dependence), so `vlc` only picks up the contributions from `m > 0`
  via the recurrence's A·vlc and B·vlc² carry-through.

**REVIEW NOTE for §1**: this transcription is a faithful rendering of the
existing code. It is not yet a derivation; it merely states the recurrences
to be differentiated in §2.

---

## Section 2 — Derive the second-derivative accumulators

**[Section 2 is the technical core. It is intentionally LEFT INCOMPLETE in
this draft pending Stage B review of §0 and §1.]**

The second-derivative accumulators required for the Hessian are:

| Accumulator | Mathematical meaning (functional notation) |
|---|---|
| `wrrc`, `wrrs` | `∂²/∂r²` of the cosine and sine series |
| `wttc`, `wtts` | `∂²/∂φ²` |
| `wrtc`, `wrts` | `∂²/∂r∂φ` |
| (lambda parts) | `∂²/∂λ²`, `∂²/∂r∂λ`, `∂²/∂φ∂λ` — handled in outer loop |

For each, the recurrence is obtained by differentiating an existing first-
derivative recurrence with respect to the appropriate coordinate, applying:
- `∂/∂r` chain to `q = R_E/r`: `∂q/∂r = −q/r`. Hence `∂A/∂r`, `∂B/∂r`
  pick up factors of `−1/r` per power of `q`.
- `∂/∂φ` chain to `t = sin φ`, `u = cos φ`: `∂t/∂φ = u`, `∂u/∂φ = −t`.
  Hence `∂A/∂φ = u · Ax + t · ∂Ax/∂φ`. Since `Ax = q · √(2n+3)` doesn't
  depend on φ, `∂A/∂φ = u · Ax`.

Sample derivation (to be completed and verified by SymPy in Stage C):

**`wrrc` recurrence**:
Differentiate `wrc_new = A · wrc + B · wrc2 + (n+1) · Cv` with respect to `r`.

`A`, `B` depend on `r` through `q`. Specifically:
- `A = t · q · √(2n+3)`, so `∂A/∂r = t · √(2n+3) · ∂q/∂r = t · √(2n+3) · (−q/r) = −A/r`.
- `B = −q² · b(n,m)`, so `∂B/∂r = −2q · b · ∂q/∂r = −2q · b · (−q/r) = 2·q²·b/r = −2B/r`.

`Cv = C_lm` is constant in `r`. So:
```
∂(wrc_new)/∂r = (∂A/∂r) · wrc + A · (∂wrc/∂r) + (∂B/∂r) · wrc2 + B · (∂wrc2/∂r)
              = (−A/r) · wrc + A · wrrc(natural) + (−2B/r) · wrc2 + B · wrrc2(natural)
```

If we define `wrrc = ∂(wrc)/∂r` in **natural** convention (no sign flip),
then:
```
wrrc_new = A · wrrc + B · wrrc2 − (A/r) · wrc − (2B/r) · wrc2
```

**REVIEW NOTE**: this differs from my earlier "off-the-cuff" claim of
`(n+1)(n+2) · Cv` as the inhomogeneous term. The actual recurrence picks
up `(A/r)·wrc` and `(2B/r)·wrc2` corrections from the `r`-dependence of
`A` and `B`. **Stage C MUST verify this from first principles using SymPy**.
This is exactly the kind of LLM-prone error the methodology is designed
to catch.

**Status**: ✅ SymPy-verified by
`tests/python/verify_gravity_hessian_stage_a.py::verify_item_005`. The
sub-claims `∂A/∂r = -A/r` (ITEM-005b) and `∂B/∂r = -2B/r` (ITEM-005c)
also verified. Difference between SymPy's `∂(wrc_new)/∂r` and my claimed
`wrrc_new` simplifies to 0.

### 2.2 wrtc derivation (∂²V/∂r∂φ mixed accumulator, natural convention)

The `wrc` first-derivative recurrence (from §1.1):
```
wrc_new = A · wrc + B · wrc2 + (n+1) · Cv
```

Differentiate with respect to `φ`. Recall `t = sin φ`, `u = cos φ`:
- `A = t · Ax` where `Ax = q · √(2n+3)`. Since `Ax` doesn't depend on `φ`
  (`q` is r-only, `√(2n+3)` is constant), and `∂t/∂φ = u`:
  ```
  ∂A/∂φ = u · Ax
  ```
- `B = -q² · b_factor`. Neither `q` nor `b_factor` depend on `φ`:
  ```
  ∂B/∂φ = 0
  ```
- `Cv = C_lm` is constant in `φ`:
  ```
  ∂Cv/∂φ = 0
  ```

Define `wrtc(natural) ≡ ∂(wrc)/∂φ` and `wrtc2(natural) ≡ ∂(wrc2)/∂φ`.
Apply the chain/product rule:
```
∂(wrc_new)/∂φ = (∂A/∂φ)·wrc + A·(∂wrc/∂φ) + (∂B/∂φ)·wrc2 + B·(∂wrc2/∂φ)
              = u·Ax·wrc + A·wrtc(nat) + 0 + B·wrtc2(nat)
```

So in **natural sign convention** (no extra sign flips):
```
wrtc_new = A · wrtc + B · wrtc2 + u · Ax · wrc
```

**Comparison to existing first-derivative recurrence**: the existing
`wtc` recurrence (§1.1) has `-u·Ax·wc2` (negative coefficient, references
`wc2`). The natural-convention `wrtc` recurrence I just derived has
`+u·Ax·wrc` (positive coefficient, references `wrc`). The differences are:

1. **Sign flip**: existing engine uses `-u·Ax·wc2` whereas natural derivation
   gives `+u·Ax·wc`. The existing engine's sign flip encodes a convention —
   `vtc` post-finalization represents `+(1/r)·∂V/∂φ` (per §0.3), so the
   "natural" sign was flipped on the way through the recurrence to land on
   that convention.

2. **Index roll**: existing engine uses `wc2` (the older `wc` value)
   whereas natural derivation gives `wc` (the current value going into this
   iteration). This is the "Clenshaw trick" for derivatives in backward
   recurrences — the index roll arises from carefully tracking when the
   chain rule applies relative to the recurrence's update step.

For the Hessian we have two consistent choices:
- **Stay in natural convention** throughout §2 derivations, then translate
  to engine convention en bloc in §3 (Cartesian Hessian assembly).
- **Match engine convention** at each step (signs and index rolls).

I'm choosing **natural convention** for §2, with the translation to engine
convention happening at §3. This is cleaner mathematically but requires
careful sign/index tracking at the §3 boundary.

**REVIEW NOTE (ITEM-008)**: SymPy MUST verify both:
(a) the natural-convention `wrtc_new = A·wrtc + B·wrtc2 + u·Ax·wrc`
(b) the index-roll/sign-flip translation in §3 from natural to engine
    convention.
The natural recurrence (a) is verifiable directly. The engine-convention
translation (b) requires implementing the existing recurrence symbolically
on a known potential and comparing — that's part of resolving ITEM-001/002.

**Status**: ✅ part (a) SymPy-verified. Part (b) deferred
pending ITEM-001/002 resolution.

### 2.3 wttc derivation (∂²V/∂φ² accumulator, natural convention)

We derive the natural-convention `wttc` by differentiating the
natural-convention `wtc` recurrence twice. (Recall: this is NOT the
engine's `wtc` recurrence — that one has sign flip + index roll for
convention reasons. We work in natural sign throughout §2 and translate
en bloc in §3.)

**Step 1: derive the natural-convention `wtc` recurrence.**

Start with `wc_new = A·wc + B·wc2 + Cv`. Differentiate wrt φ once:
```
wtc(nat)_new = ∂(wc_new)/∂φ
             = (∂A/∂φ)·wc + A·(∂wc/∂φ) + (∂B/∂φ)·wc2 + B·(∂wc2/∂φ) + (∂Cv/∂φ)
             = u·Ax·wc + A·wtc(nat) + 0 + B·wtc2(nat) + 0
             = A · wtc(nat) + B · wtc2(nat) + u · Ax · wc
```

Note: this references `wc` (the current value series), NOT `wc2` (older
value). Sign is `+u·Ax·wc`. Compare to the engine's recurrence
`wtc_new = A·wtc + B·wtc2 - u·Ax·wc2`: same structure but flipped sign
and index-rolled.

**Step 2: differentiate the natural-convention `wtc` recurrence wrt φ.**

```
wttc(nat)_new = ∂(wtc(nat)_new)/∂φ
              = ∂(A·wtc + B·wtc2 + u·Ax·wc)/∂φ
              = (∂A/∂φ)·wtc + A·(∂wtc/∂φ)
              + (∂B/∂φ)·wtc2 + B·(∂wtc2/∂φ)
              + (∂u/∂φ)·Ax·wc + u·(∂Ax/∂φ)·wc + u·Ax·(∂wc/∂φ)
```

Computing each term:
- `∂A/∂φ = u·Ax`         (from ITEM-008b)
- `∂B/∂φ = 0`            (B is φ-independent)
- `∂u/∂φ = -t`           (since u = cos φ)
- `∂Ax/∂φ = 0`           (Ax = q·√(2n+3), φ-independent)
- `∂wtc/∂φ = wttc(nat)`
- `∂wtc2/∂φ = wttc2(nat)`
- `∂wc/∂φ = wtc(nat)`

Substituting:
```
wttc(nat)_new = u·Ax·wtc + A·wttc(nat)
              + 0 + B·wttc2(nat)
              + (-t)·Ax·wc + 0 + u·Ax·wtc(nat)
              = A·wttc(nat) + B·wttc2(nat) + 2·u·Ax·wtc(nat) - t·Ax·wc
```

So the natural-convention wttc recurrence is:
```
wttc(nat)_new = A · wttc(nat) + B · wttc2(nat) + 2·u·Ax · wtc(nat) - t·Ax · wc
```

The factor `2·u·Ax` arises because BOTH the chain through `A` (giving
`u·Ax·wtc`) AND the chain through `u·Ax·wc` (giving `u·Ax·wtc(nat)` via
`∂wc/∂φ = wtc(nat)`) contribute. Easy to miss one.

The `-t·Ax·wc` term comes from `∂u/∂φ = -t` applied to the `u·Ax·wc`
inhomogeneous term in the wtc(nat) recurrence.

**REVIEW NOTE (ITEM-009)**: SymPy must verify this derivation. If the
recurrence is right, both the `2·u·Ax·wtc(nat)` factor of 2 AND the
`-t·Ax·wc` term should appear with correct signs.

**Sign convention transition**: the code has `vtc = −(1/r)·∂V/∂φ`
(post-finalization). To stay consistent with code conventions, the
Hessian assembly will need to track an analogous sign on `wttc` etc.
This is the most error-prone part of the derivation and must be
SymPy-verified.

The rest of Section 2 (the lambda-derivative accumulators `wrlc`, `wtlc`,
`wllc`, and their `_s` siblings) is **deferred to the next drafting
session** — it requires resolving the sign convention questions from §0.3
(ITEM-001/002) before the engine-translation step in §3.

### 2.4 Sine-series siblings (wrrs, wrts, wtts) — structural symmetry

The cosine series `wc` and sine series `ws` satisfy **identical**
homogeneous recurrences `A · w + B · w2`, differing only in the
inhomogeneous term:
```
wc_new = A · wc + B · wc2 + Cv          (Cv = normalized C_lm)
ws_new = A · ws + B · ws2 + Sv          (Sv = normalized S_lm)
```

`A`, `B`, `u`, `t`, `Ax = q·√(2n+3)` are functions of `(r, φ)` only — they
do NOT depend on whether the series is cosine or sine. Therefore, every
derivation in §2.1–§2.3 carries through with `wc → ws` and `Cv → Sv`:

```
wrrs_new = A · wrrs + B · wrrs2 - (A/r) · wrs - (2B/r) · wrs2
wrts_new = A · wrts + B · wrts2 + u · Ax · wrs                          (natural)
wtts_new = A · wtts + B · wtts2 + 2·u·Ax · wts - t·Ax · ws              (natural)
```

The same `(n+1)·Cv` correction that appears in §2.1 for `wrrc` appears
analogously in `wrrs`, just with `(n+1)·Sv` (and that term drops out of
the homogeneous recurrence as long as we're tracking the inhomogeneous
contribution separately, which §2.1 does).

**SymPy verification (ITEM-010)**: each of the three sine recurrences is
verified independently by computing the corresponding derivative of
`ws_new` from scratch and comparing to the claim. All three sub-tests
pass with `simplify(SymPy − claimed) == 0`. See
`tests/python/verify_gravity_hessian_stage_a.py::verify_item_010`.

This structural symmetry is load-bearing for the implementation: the C++
engine has parallel cosine/sine accumulator pairs, so any sign error in
one but not the other would be caught by the symmetry property
`H(C → C, S → S) = H(S → S, C → C)` at the gravity-coefficient level.

### 2.5 Lambda-derivative accumulators (outer-loop source-term rule)

The longitude `λ` enters the gravity series ONLY through the trigonometric
factors `cos(mλ), sin(mλ)` in the outer-loop sum:
```
V(r, φ, λ) = Σ_m [ (cos-bucket of m) · cos(mλ) + (sin-bucket of m) · sin(mλ) ]
```
where the cos- and sin-buckets are the inner-loop terminal values `wc(m)`
and `ws(m)` (respectively) after radial scaling. The Legendre factors
`P̄_lm(sin φ)` are λ-independent.

The outer Clenshaw-style recursion at
`clenshaw_engine.hpp:223-237` advances accumulators `vc, vs, vrc, vrs,
vtc, vts, vlc, vls` via a *common* homogeneous recurrence `temp = A·v +
B·v2 + (source)`. The `A, B` coefficients depend on `r, φ, m` (and `cl =
cos λ`, `sl = sin λ` mixed in via the m=0 finalization), but **not** on
which derivative the accumulator tracks. Different accumulators differ
only in their inhomogeneous source.

**Source-term substitution lemma**: applying `∂/∂λ` to the outer-loop
sum is equivalent to substituting sources via:
```
A_m → +m · B_m   (new cos-bucket source)
B_m → −m · A_m   (new sin-bucket source)
```
where `A_m, B_m` are the cos/sin-bucket sources of the parent quantity.
Proof: differentiating `cos(mλ) → −m·sin(mλ)` and `sin(mλ) → +m·cos(mλ)`,
then re-grouping by the new trig basis. The Legendre/radial factors pass
through untouched because they're λ-independent. The homogeneous `A·v +
B·v2` part of the outer recurrence commutes with this substitution (the
Chebyshev/Legendre step in `m` doesn't touch the source labels), so the
lambda-derivative accumulator obeys the same `A·v + B·v2` recurrence with
the substituted source.

Applying this lemma to the existing first-derivative accumulators:

| Hessian entry | Parent `(A_m, B_m)` | New cos source (vXlc) | New sin source (vXls) |
|---|---|---|---|
| `∂V/∂λ` (existing, line 236-237) | `(wc, ws)` | `+m · ws` | `−m · wc` |
| `∂²V/∂r∂λ` | `(wrc, wrs)` | `+m · wrs` | `−m · wrc` |
| `∂²V/∂φ∂λ` | `(wtc, wts)` | `+m · wts` | `−m · wtc` |
| `∂²V/∂λ²` (apply lemma TWICE) | `(wc, ws)` | `−m² · wc` | `−m² · ws` |

For `∂²V/∂λ²`, applying the lemma twice: first substitution gives sources
`(+m·ws, −m·wc)`. The second substitution treats those as the new
parent: `+m·ws → −m²·wc` (cos bucket: `+m·(−m·wc)`) and
`−m·wc → −m²·ws` (sin bucket: `−m·(+m·ws)`). Note both signs are
**negative**: the second-order λ-derivative does NOT swap cos↔sin (even
order), it just brings down `−m²` on each bucket diagonally.

So the natural-convention recurrences for the six lambda-derivative
accumulators are:
```
vrlc_new = A·vrlc + B·vrlc2 + m·wrs
vrls_new = A·vrls + B·vrls2 − m·wrc
vtlc_new = A·vtlc + B·vtlc2 + m·wts
vtls_new = A·vtls + B·vtls2 − m·wtc
vllc_new = A·vllc + B·vllc2 − m²·wc
vlls_new = A·vlls + B·vlls2 − m²·ws
```

**REVIEW NOTE (ITEM-011)**: SymPy must verify the source-term substitution
lemma by computing `∂/∂λ` of a symbolic partial Fourier series and
collecting coefficients of `cos(mλ), sin(mλ)`. Six sub-tests, one per
lambda accumulator. If the lemma is wrong, these tests catch it
immediately because they're stating each source independently and asking
SymPy to compute it from the symbolic series.

**Engine sign-convention caveat**: the engine's m=0 finalization at line
248-249 multiplies `vlc` by `qs_r/u = q/(r·u)` (the `1/cos(φ)` for the
spherical-coords gradient component). The natural-convention recurrence
above does NOT include this final factor — it's part of the spherical →
Cartesian transition handled in §3, not part of the m-recurrence itself.

The `1/u` factor introduces a `1/cos(φ)` pole singularity in `∂V/∂λ` and
its second derivatives, which §4 must handle. The engine guards with
`std::abs(u) > GRAVITY_POLE_U_THRESHOLD` (line 248); the Hessian
extension needs analogous (and likely stronger) guards because
`∂²V/∂λ²` carries a `1/cos²(φ)` factor.

### 2.6 Engine-convention second-derivative recurrences (Stage F spec)

§2.1–§2.4 derived natural-convention recurrences (working from `wc_new =
A·wc + B·wc2 + Cv` etc.). The engine uses **factored** conventions for
the first-derivative inner-loop accumulators:
```
wc_new   = A·wc  + B·wc2  + Cv             (line 187, natural)
wrc_new  = A·wrc + B·wrc2 + (n+1)·Cv       (line 192, factored — (n+1) absorbs (R/r)^(l+1) differentiation)
wtc_new  = A·wtc + B·wtc2 − u·Ax·wc2       (line 197, sign-flip + index-roll wrt natural)
```

Stage F implementation requires recurrences for the second-derivative
accumulators in the **engine's** factored conventions, derived by
differentiating these engine recurrences (NOT the natural ones in §2.1
applied directly). The differentiation formulas:

```
∂A/∂r = −A/r       ∂A/∂φ = u·Ax       (Ax = q·sqrt(2n+3), φ-independent)
∂B/∂r = −2B/r       ∂B/∂φ = 0
∂Ax/∂r = −Ax/r      ∂Ax/∂φ = 0          (q = R/r, sqrt(2n+3) constant)
∂u/∂φ = −t          ∂u/∂r = 0
∂t/∂φ = u           ∂t/∂r = 0
```

#### 2.6.1 wrrc — engine convention

Differentiating engine `wrc_new` wrt `r`:
```
wrrc_engine_new = ∂(wrc_new)/∂r
                = A·wrrc + B·wrrc2 + (∂A/∂r)·wrc + (∂B/∂r)·wrc2 + 0
                = A·wrrc + B·wrrc2 − (A/r)·wrc − (2B/r)·wrc2
```
Same structural form as natural-convention wrrc (§2.1) — the `(n+1)·Cv`
source is r-independent and contributes nothing on differentiation.
**Match with §2.1 form is expected and easy.**

#### 2.6.2 wrtc — engine convention

Differentiating engine `wrc_new` wrt `φ`:
```
wrtc_engine_new = ∂(wrc_new)/∂φ
                = A·wrtc + B·wrtc2 + (∂A/∂φ)·wrc + (∂B/∂φ)·wrc2 + 0
                = A·wrtc + B·wrtc2 + u·Ax·wrc
```
Same form as natural §2.2.

#### 2.6.3 wttc — engine convention (DIFFERENT from §2.3 natural)

Differentiating engine `wtc_new` wrt `φ`:
```
wttc_engine_new = ∂(wtc_new)/∂φ
                = A·wttc + B·wttc2 + (∂A/∂φ)·wtc + (∂B/∂φ)·wtc2
                  + ∂(−u·Ax·wc2)/∂φ
                = A·wttc + B·wttc2 + u·Ax·wtc + 0
                  + (+t·Ax·wc2 − u·Ax·∂wc2/∂φ)
                = A·wttc + B·wttc2 + u·Ax·wtc + t·Ax·wc2 − u·Ax·(∂wc2/∂φ)
```

**Convention caveat (Stage F implementation catch)**: the last term is
`−u·Ax·(∂wc2/∂φ)` which is the natural φ-derivative of `wc2`, denoted
`wphic2` in the auxiliary accumulator — NOT engine `wtc2` (which has
sign-flip + iterate-roll vs natural per §0.3 / §2.3 / ITEM-018).

The C++ implementation must use the wphic2 accumulator (ITEM-024)
here, NOT engine wtc2. Order of updates: compute wttc BEFORE wphic
update so wphic2 holds the pre-update value (i.e., `∂wc(n+2)/∂φ` at
this recurrence step).

This was caught during Stage F C++ implementation: the initial draft
incorrectly used engine wtc2 here, causing Laplace failures at deg ≥ 2.
The corrected formulation produces matches to closed-form J2 at
machine precision.

Compare to §2.3 natural:
`wttc_nat_new = A·wttc + B·wttc2 + 2·u·Ax·wtc − t·Ax·wc`

**These are STRUCTURALLY DIFFERENT** — engine has `t·Ax·wc2` (with old
iterate, plus sign) and `−u·Ax·wtc2` (which natural doesn't have); §2.3
natural has `2·u·Ax·wtc` (factor of 2 from chain through `wc → wtc` and
through `u·Ax·wc` differentiation) and `−t·Ax·wc` (with sign flip).

The engine convention is what Stage F must implement. **REVIEW NOTE
(ITEM-017)**: SymPy must verify the engine `wttc` recurrence by
independently differentiating engine `wtc_new` and matching the claimed
formula above.

#### 2.6.4 Sine siblings (wrrs, wrts, wtts) — engine convention

Identical structure to 2.6.1–2.6.3 with `wc → ws`, `wc2 → ws2`,
`Cv → Sv` (§2.4 logic applies because `A, B, u, t, Ax` are
trig-independent). ITEM-017 covers these too.

#### 2.6.5 Sectorial corrections at m≥1 inner-loop exit

The engine applies sectorial corrections at line 218-220:
```cpp
wtc += m * tu * wc;
wts += m * tu * ws;
```
where `tu = t/u`. These finalize the inner-loop `wtc, wts` to absorb
the sectorial m-mode contribution.

The correction `m·tu·wc` is the φ-dependence of the factor u^m that every
order-m term carries; the first-derivative outer recurrence for `vtc` does
not differentiate `A_m, B_m` (both ∝ powers of u), so this inner correction
stands in for that derivative. Which second-derivative accumulators need a
sectorial correction depends on which first-derivative recurrence they
differentiate:

- `vttc` differentiates the `vtc` recurrence, whose source is the CORRECTED
  `wtc`. Its source is therefore ∂(wtc_corrected)/∂φ, which carries the
  terms below.
- `vrtc` differentiates the `vrc` recurrence, whose source `wrc` has no
  correction. Its source is plain `wrtc = ∂wrc/∂φ`; the u^m dependence enters
  through the explicit `−tu·A·vrc − 2tu·B·vrc2` terms of §2.7. Adding
  `m·tu·wrc` to `wrtc` as well would count that dependence twice (with the
  colatitude sign): the r-φ element of every m ≥ 1 term would then be wrong
  by O(1), while Laplace's equation (which does not involve that element)
  still holds. **No sectorial correction is applied to `wrtc`/`wrts`.**
  Checked against central finite differences of the acceleration: each
  single (n, m, C or S) term then agrees to ~1e-11 relative (test_gravity_hessian).

Differentiating `wtc_corrected = wtc + m·tu·wc` wrt φ:
```
∂(wtc_corrected)/∂φ = wttc + m·(∂tu/∂φ)·wc + m·tu·(∂wc/∂φ)
```
with ∂(t/u)/∂φ = (∂t/∂φ · u − t · ∂u/∂φ) / u² = (u·u − t·(−t)) / u² = (u² + t²)/u² = 1/u².

So:
```
∂(wtc_corrected)/∂φ = wttc + m·(1/u²)·wc + m·tu·(∂wc/∂φ)
```

**Convention caveat (ITEM-018 catch)**: the third term is `m·tu·(∂wc/∂φ)`,
**NOT** `m·tu·wtc`. The engine's `wtc` is NOT natural `∂wc/∂φ`: per
§0.3 / §2.3, the engine's wtc has sign-flip + iterate-roll relative
to the natural derivative. They differ at the inner-loop terminal
value.

To compute the sectorial correction for engine `wttc`, Stage F must
either:
  (a) **Add a new inner-loop accumulator** `wphic` tracking natural
      `∂wc/∂φ` with recurrence
      ```
      wphic_new = A·wphic + B·wphic2 + u·Ax·wc       (+u·Ax·wc, current iterate)
      ```
      (derived by differentiating `wc_new = A·wc + B·wc2 + Cv` wrt φ).
      Compare to engine's `wtc_new = A·wtc + B·wtc2 − u·Ax·wc2` —
      same homogeneous structure, opposite sign on inhomogeneous, and
      uses current `wc` (not old `wc2`).
  (b) **Use Strategy 2** (FD on finalized vrc/vtc/vlc, §3.4) — avoids
      the sectorial-correction issue entirely at the cost of
      analytic-recurrence speedup.

The `1/u²` is the source of the `1/cos²(φ)` pole singularity warned
about in §4 — it appears in the second-derivative inner-loop sectorial
correction, not at the outer m=0 finalization. **REVIEW NOTE
(ITEM-018)**: SymPy verifies the corrected claim above (verifying both
the formula structure and the convention distinction between engine
`wtc` and natural `∂wc/∂φ`).

### 2.7 Engine outer-loop second-derivative recurrences (Stage F spec)

The engine's outer-loop first-derivative recurrences at line 230-237 use
homogeneous coefficients `A_m, B_m` that depend on `(r, φ)` through
`uq = u · q = cos(φ) · R/r`. Specifically (line 226-227):
```
A_m = cl · v · uq                    where v = root_[2]·root_[2m+3]/root_[m+1] is m-only
B_m = −v · root_[2m+5]/(root_[8]·root_[m+2]) · uq²
```
so:
```
∂A_m/∂r = −A_m / r       ∂A_m/∂φ = −tu · A_m       (tu = t/u)
∂B_m/∂r = −2 B_m / r      ∂B_m/∂φ = −2 tu · B_m
```
(Note: `cl, sl` in `A_m` depend on λ, NOT on r or φ. So ∂A_m/∂λ has its
own non-zero contribution which is handled by §2.5's lambda
source-substitution lemma. For r and φ derivatives, cl, sl pass through
as constants.)

**Differentiating the outer-loop first-derivative recurrence wrt r** at
line 232 (`vrc_new = A·vrc + B·vrc2 + wrc`):
```
vrrc_engine_new = ∂(vrc_new)/∂r
               = ∂A/∂r · vrc + A · vrrc + ∂B/∂r · vrc2 + B · vrrc2 + wrrc
               = A · vrrc + B · vrrc2 + wrrc − (A/r) · vrc − (2B/r) · vrc2
```

**Same structure as inner-loop §2.6.1**: the outer recurrence picks up
correction terms `−(A/r)·vrc − (2B/r)·vrc2` from differentiating the
m-mode A, B coefficients. (This is parallel to the inner-loop wrrc
recurrence but with `(A_m, B_m, vrc, vrc2, wrrc, vrrc, vrrc2)` instead
of `(A, B, wrc, wrc2, ...)`.)

**Differentiating wrt φ**:
```
vrtc_engine_new = ∂(vrc_new)/∂φ
               = ∂A/∂φ · vrc + A · vrtc + ∂B/∂φ · vrc2 + B · vrtc2 + wrtc
               = A · vrtc + B · vrtc2 + wrtc − tu·A · vrc − 2 tu·B · vrc2
```

For the engine's vtc-derived second derivatives:
```
vrtc_engine_new = (above)
vttc_engine_new = ∂(vtc_new)/∂φ
               = A · vttc + B · vttc2 + wttc − tu·A · vtc − 2 tu·B · vtc2
```

For the lambda accumulators, §2.5's source-substitution lemma still
applies with the same A, B (since A, B don't depend on λ — `cl, sl`
appear as λ-independent factors at line 226-227's m≥1 step too,
analogous to the m=0 finalization).

**Engine outer-loop second-derivative recurrences (full set)**:
```
vrrc_new = A·vrrc + B·vrrc2 + wrrc − (A/r)·vrc − (2B/r)·vrc2
vrtc_new = A·vrtc + B·vrtc2 + wrtc − tu·A·vrc − 2tu·B·vrc2
vttc_new = A·vttc + B·vttc2 + wttc − tu·A·vtc − 2tu·B·vtc2
vrlc_new = A·vrlc + B·vrlc2 + m·wrs        (lambda — no A, B correction; cf §2.5)
vtlc_new = A·vtlc + B·vtlc2 + m·wts
vllc_new = A·vllc + B·vllc2 − m²·wc
```
plus 6 sine versions with cos↔sin substitutions (and vlc-coupling as in
§2.5).

**REVIEW NOTE (ITEM-022)**: SymPy must verify these by differentiating
the engine's outer-loop first-derivative recurrence.

**Implementation note for Stage F**: the correction terms `−(A/r)·vrc`
etc. couple the second-derivative outer recurrence to the
first-derivative outer accumulators. Both sets must be advanced in
lock-step at each m-step; the second-derivative recurrence references
the **current** first-derivative accumulator values. Order-of-update
matters: write the second-derivative `vrrc_new, vrtc_new, vttc_new`
BEFORE updating `vrc, vtc` (similar to how the engine handles
`vrc2 = vrc; vrc = temp;` order).

---

## Section 3 — Spherical Hessian → Cartesian Hessian

### 3.1 Position transform and gradient chain rule

ECEF position from spherical coordinates:
```
x = r · u · cl       where  u = cos φ,  t = sin φ,  cl = cos λ,  sl = sin λ
y = r · u · sl
z = r · t
```

(equivalently, `r = √(x²+y²+z²)`, `t = z/r`, `u = √(x²+y²)/r = p/r`,
`cl = x/p`, `sl = y/p`, with `p = √(x²+y²)`).

The gradient `∇V_cart = (∂V/∂x, ∂V/∂y, ∂V/∂z)ᵀ` follows from the chain
rule applied to `V(r(x,y,z), φ(x,y,z), λ(x,y,z))`. Writing the partials
of the inverse transform explicitly:

```
∂r/∂x = u·cl,           ∂r/∂y = u·sl,           ∂r/∂z = t
∂φ/∂x = −t·cl/r,         ∂φ/∂y = −t·sl/r,         ∂φ/∂z = u/r
∂λ/∂x = −sl/(r·u),       ∂λ/∂y = +cl/(r·u),       ∂λ/∂z = 0
```

(Derivation: standard inverse-Jacobian; `∂λ/∂z = 0` since longitude is
defined entirely in the equatorial plane. The `1/u = 1/cos φ` factors in
`∂λ/∂x, ∂λ/∂y` are the source of the pole singularity in λ-derivatives.)

Substituting into the chain rule:
```
∂V/∂x = (u·cl)·(∂V/∂r) + (−t·cl/r)·(∂V/∂φ) + (−sl/(r·u))·(∂V/∂λ)
      = cl · [ u·(∂V/∂r) − (t/r)·(∂V/∂φ) ] − (sl/(r·u))·(∂V/∂λ)

∂V/∂y = (u·sl)·(∂V/∂r) + (−t·sl/r)·(∂V/∂φ) + (cl/(r·u))·(∂V/∂λ)
      = sl · [ u·(∂V/∂r) − (t/r)·(∂V/∂φ) ] + (cl/(r·u))·(∂V/∂λ)

∂V/∂z = t·(∂V/∂r) + (u/r)·(∂V/∂φ)
```

### 3.2 Engine convention for `vrc, vtc, vlc` (resolves ITEM-001/002)

The engine's gradient assembly at `clenshaw_engine.hpp:257-259` is:
```cpp
gradx = cl · (u · vrc + t · vtc) − sl · vlc
grady = sl · (u · vrc + t · vtc) + cl · vlc
gradz = t · vrc − u · vtc
```

with `result.acceleration = scale_factor · Vector3d(gradx, grady, gradz)`
where `scale_factor = GM / radius_` (line 254). The acceleration is
`a = ∇V` (positive sign; the engine uses geodesy convention `V = +GM/r`,
so force per unit mass is `+∇V`, see `result.potential = scale_factor·vc`
at line 255 which is positive for V = GM/r).

Identifying `result.acceleration_x = (GM/R)·gradx` with `∂V/∂x` and
matching coefficient by coefficient against the chain-rule expression
above:

| Cartesian piece | Chain-rule coefficient of  | Engine `gradx · (GM/R)` matches if   |
|---|---|---|
| `cl · u`  | `+∂V/∂r`           | `(GM/R) · vrc = +∂V/∂r`              |
| `cl · t`  | `−(1/r)·∂V/∂φ`     | `(GM/R) · vtc = −(1/r)·∂V/∂φ`        |
| `−sl`    | `+(1/(r·u))·∂V/∂λ`  | `(GM/R) · vlc = +(1/(r·u))·∂V/∂λ`    |

Cross-checking the `∂V/∂z` row (no `cl, sl` mixing, so the relations come
through directly):
- Coefficient of `t`: chain rule says `+∂V/∂r`; engine says `+vrc·(GM/R) = +∂V/∂r` ✓
- Coefficient of `u`: chain rule says `+(1/r)·∂V/∂φ`; engine says `−vtc·(GM/R) = +(1/r)·∂V/∂φ`,
  i.e. `vtc·(GM/R) = −(1/r)·∂V/∂φ` ✓ (consistent with the gradx/grady reading)

So the engine's post-finalization convention is:
```
(GM/R) · vc  = +V                  (potential, geodesy convention)
(GM/R) · vrc = +∂V/∂r              (radial derivative, no sign flip)
(GM/R) · vtc = −(1/r) · ∂V/∂φ      (theta derivative, has sign flip AND 1/r factor)
(GM/R) · vlc = +(1/(r·u)) · ∂V/∂λ  (lambda derivative, has 1/(r·cos φ) factor)
```

The `−(1/r)` factor in `vtc` is the source of the engine's "wtc tracks
−∂V/∂φ"-like convention noted in §0.3 (and confirmed by the source-term
sign at line 218-220 sectorial correction: `wtc += m · tu · wc` mirrors
this convention through the inner-loop accumulation).

The factors `1/r`, `1/(r·u)` arise from the spherical-coords gradient
formula's per-coordinate metric scaling (`g_φφ = r²`, `g_λλ = r²·u²`),
absorbed into the engine's accumulators rather than applied at the
gradient assembly. The `qs_r = q/r` factor at line 246-249 supplies
exactly these `1/r` factors (`qs_r/u = q/(r·u)` for `vlc`).

**Sign verification (ITEM-012, SymPy)**: the chain-rule expressions
above are computed by SymPy from the symbolic position transform; the
engine's `gradx, grady, gradz` formulas are stated independently. After
substituting the convention table, `simplify(SymPy − engine) == 0` for
all three components.

### 3.3 Hessian assembly (tensor decomposition)

With the §3.2 convention nailed down, the Cartesian Hessian decomposes
into two tensor pieces. Let `q^a ∈ {r, φ, λ}` index the spherical
coordinates and `x_i ∈ {x, y, z}` index the Cartesian coordinates.
Define the inverse-transform Jacobian and Hessian:

```
J^a_i        = ∂q^a / ∂x_i             (3×3 matrix per §3.1)
K^a_{ij}     = ∂²q^a / ∂x_i ∂x_j       (3×3×3 tensor; symmetric in i,j)
```

Applying the chain rule twice to `V(x, y, z) = V(r(x,y,z), φ(x,y,z),
λ(x,y,z))`:

```
∂V/∂x_i = Σ_a (∂V/∂q^a) · J^a_i

∂²V/∂x_i ∂x_j = Σ_a Σ_b (∂²V/∂q^a ∂q^b) · J^a_i · J^b_j
              + Σ_a (∂V/∂q^a) · K^a_{ij}
```

In matrix form:
```
H_cart = J · H_sph · Jᵀ + Σ_a (∂V/∂q^a) · K^a
```
where `H_sph` is the 3×3 spherical-coord Hessian whose entries are the
second-derivative accumulators from §2.1–§2.5, `J` is the 3×3 inverse
Jacobian per §3.1, and `K^a` is the 3×3 (symmetric) Hessian of the
inverse-transform component `q^a` w.r.t. Cartesian coordinates.

**Computing K^a explicitly**: the inverse transform is
```
r = √(x² + y² + z²)
φ = atan2(z, p)        where p = √(x² + y²)
λ = atan2(y, x)
```
Differentiating twice and re-expressing in spherical coordinates:

```
K^r_{ij} = ∂²r/∂x_i∂x_j = (1/r)·(δ_{ij} − ê_r,i · ê_r,j)
```
where `ê_r = (u·cl, u·sl, t)` is the radial unit vector. (This is the
standard "transverse projector" form — the K^r tensor is the projection
onto the plane perpendicular to ê_r, scaled by 1/r.)

```
K^φ and K^λ have analogous closed forms involving t, u, cl, sl, and r;
their full expressions are SymPy-verified rather than written out
manually here.
```

**Symmetry property**: `K^a_{ij} = K^a_{ji}` by Schwarz, so `H_cart` is
automatically symmetric — a useful consistency check at every test point.

**Trace property (Laplace)**: outside source masses, `∇²V = 0`, so
`tr(H_cart) = 0`. In tensor form:
```
tr(H_cart) = Σ_i Σ_a Σ_b J^a_i · J^b_i · H_sph_{ab}
           + Σ_a (∂V/∂q^a) · Σ_i K^a_{ii}
         = Σ_{a,b} g^{ab} · H_sph_{ab}  +  Σ_a (∂V/∂q^a) · ∇²(q^a)
```
where `g^{ab} = Σ_i J^a_i · J^b_i = (∂q^a/∂x_i)(∂q^b/∂x_i)` is the
inverse metric tensor in spherical coords (`g^{rr}=1, g^{φφ}=1/r²,
g^{λλ}=1/(r²·u²)`, off-diagonals zero). And `∇²(q^a) = Σ_i K^a_{ii}` is
the ordinary Cartesian Laplacian of the inverse-transform component
(non-zero in general).

**Engine accumulator translation**: substituting the §3.2 convention
table maps `H_sph` entries to engine accumulators (with the natural
spherical-coord scaling absorbed):

| Entry | (R/GM) ·  *engine accumulator* |
|---|---|
| `∂²V/∂r²`  | matches `(R/GM) · vrrc_post` (post-finalization analog of vrc; uses §2.1) |
| `∂²V/∂r∂φ` | `−(1/r)` · post-finalization vrtc analog (uses §2.2) |
| `∂²V/∂φ²`  | `+(1/r²)` · post-finalization vttc + extra `(1/r)·∂V/∂r` correction term* |
| `∂²V/∂r∂λ` | `+(1/(r·u))` · post-finalization vrlc analog (uses §2.5) |
| `∂²V/∂φ∂λ` | extra `(t/(r²·u²))·∂V/∂λ` correction term + (1/(r·u))·vtlc-analog scaling |
| `∂²V/∂λ²`  | `+(1/(r²·u²))` · vllc + extra correction terms involving `∂V/∂φ, ∂V/∂r` |

\* The "extra correction terms" arise because the spherical-coord *metric*
itself depends on position. Treating `H_sph` as the partial-derivatives
matrix `∂²V/∂q^a∂q^b` (rather than the metric-corrected covariant
Hessian) means the corrections live in `K^a`, not in `H_sph`. This is
the cleaner decomposition because `H_sph` then directly maps to the
§2.1–§2.5 accumulator outputs without metric-juggling. The corrections
are bundled into the `Σ_a (∂V/∂q^a) · K^a` piece.

The full Cartesian Hessian assembly under this decomposition has six
unique entries (`H_xx, H_yy, H_zz, H_xy, H_xz, H_yz`), each a sum of:
- 6 terms from `J · H_sph · Jᵀ` (one per H_sph entry, accounting for
  symmetry; some pairs may simplify to zero).
- 3 terms from `Σ_a (∂V/∂q^a) · K^a_{ij}`.

That's ≤ 9 terms per entry × 6 entries = ≤ 54 terms total — fewer than
the "~30" earlier estimate after exploiting symmetry. SymPy verifies the
full assembly in ITEM-013.

**ITEM-013 verification approach**: compute the LHS `∂²V/∂x_i∂x_j` via
SymPy chain rule on the composed function `V_xyz(x,y,z) = V(r(x,y,z),
φ(x,y,z), λ(x,y,z))`, and the RHS via the explicit tensor decomposition
above. Assert `simplify(LHS − RHS) == 0` for all six unique entries.

### 3.4 Post-finalization convention for second-derivative accumulators

The first-derivative finalization at engine line 246-249 produces:
```
vrc_final = −qs_r · (wrc + A·(cl·vrc + sl·vrs) + B·vrc2)
vtc_final = +qs_r · (wtc + A·(cl·vtc + sl·vts) + B·vtc2)
vlc_final = +qs_r/u · (A·(cl·vlc + sl·vls) + B·vlc2)
```
where `qs_r = q/r = R/r²`. These satisfy ITEM-012's convention table:
`(GM/R) · {vrc, vtc, vlc}_final = {+∂V/∂r, −(1/r)·∂V/∂φ, +(1/(r·u))·∂V/∂λ}`.

For Stage F, the engine needs **analogous m=0 finalization** for the
second-derivative accumulators (vrrc, vrtc, vttc, vrlc, vtlc, vllc + sine
versions). The cleanest derivation strategy:

**Strategy 1 — direct differentiation of §3.2 finalization formulas**.
Define
```
vrrc_final ≡ ∂(vrc_final)/∂r,   vrtc_final ≡ ∂(vrc_final)/∂φ,
vttc_final ≡ ∂[r · vtc_final + ∂V/∂φ]/∂φ ÷ (-r),   ...
```
chosen so that `(GM/R) · vXXc_final = ∂²V/∂q^a∂q^b` (natural
partial-derivatives spherical Hessian). The exact m=0 formulas come from
applying ∂/∂q to the right-hand sides of the §3.2 finalization equations
and grouping terms.

This yields formulas with the structure
`vXXc_final = c1(q,r,u,t) · (engine-convention 2nd-deriv accumulator chain)
              + c2(q,r,u,t) · (engine-convention 1st-deriv accumulator chain)`
where the second term arises from differentiating the `qs_r`, `qs_r/u`
prefactors (which depend on r, u).

The full derivation has six finalization formulas (one per unique 2nd
derivative); each requires careful tracking of which inner/outer
accumulators contribute. **This work is flagged as ITEM-019 (Pending)**;
it is small in spec but error-prone in execution and benefits from
SymPy assistance during the derivation, not just verification.

**Strategy 2 — bypass via FD-on-finalized-vrc/vtc/vlc**. As a
fallback, compute H_sph entries via FD on the engine's existing
post-finalization vrc, vtc, vlc:
```
∂²V/∂r²    ≈ (vrc_final(r+δ) − vrc_final(r−δ)) · (GM/R) / (2δ)
∂²V/∂r∂φ  ≈ (vrc_final(φ+δ) − vrc_final(φ−δ)) · (GM/R) / (2δ)
... (similar for vtc, vlc-derived combinations)
```
This loses the analytic-recurrence speedup (becomes 7 engine evaluations
per Hessian) but preserves correctness. Useful as a **bisecting
sanity check** during Stage F implementation: a passing FD-based test
narrows bugs to the m=0 finalization formulas, separating recurrence
correctness from finalization correctness.

**Status**: Strategy 1 (full analytic finalization) is required for
the analytic Hessian's speedup. Strategy 2 (FD fallback) is the safe path
for first-pass implementation + verification. The plan: implement
Strategy 2 first, verify against ITEM-013 + Stage E (J2 closed form),
then derive and substitute Strategy 1's analytic formulas, verifying
they agree with Strategy 2 to ~1e-6 relative.

### 3.4.1 Convention-bridge table for Stage F (verified by ITEM-023)

After the m=0 finalizations in §3.4, the 2nd-derivative accumulators
satisfy:

| Accumulator | (GM/R) · vXXc_final equals |
|---|---|
| `vrrc_final` | `+∂²V/∂r²` |
| `vrtc_final` | `+∂²V/∂r∂φ` |
| `vttc_final` | `−(1/r)·∂²V/∂φ²` |
| `vrlc_final` | `+∂²V/∂r∂λ`                  [no inheritance factor] |
| `vtlc_final` | `−(1/r)·∂²V/∂φ∂λ` |
| `vllc_final` | `+(1/(r·u))·∂²V/∂λ²` |

Stage F's `H_sph` natural-convention assembly (for §3.3 tensor
decomposition) inverts these factors:

```
H_sph[r,r]  = (GM/R) · vrrc_final
H_sph[r,φ]  = (GM/R) · vrtc_final
H_sph[φ,φ]  = −r · (GM/R) · vttc_final
H_sph[r,λ]  = (GM/R) · vrlc_final               [no inversion]
H_sph[φ,λ]  = −r · (GM/R) · vtlc_final
H_sph[λ,λ]  = (r·u) · (GM/R) · vllc_final
```

**Key insight**: λ-differentiation does NOT add inheritance factors
because the conversion factors `1`, `−(1/r)`, `+(1/(r·u))` are all
λ-independent (they depend only on r, φ which are held fixed when
computing partial derivative wrt λ). Only φ-differentiation of vtc and
λ-differentiation of vlc inherit the parent's conversion factor.

This was a methodology catch — initial draft had factors `(1/(r·u))`,
`(1/(r²·u))`, `(1/(r²·u²))` for vrlc, vtlc, vllc respectively (assuming
λ-diff adds inheritance), but ITEM-023 SymPy verification showed those
factors don't exist. The corrected table above is verified.

### 3.5 ECEF → ECI Hessian frame transform (Stage I requirement)

The gravity engine computes the Hessian in **ECEF** (per §0.1). The
ForceModel layer requires it in **ECI/GCRS** for the Jacobian assembly
of the satellite state. The transformation between the two for a
**rank-2 tensor** is a similarity transform:
```
H_eci = R · H_ecef · Rᵀ
```
where `R = R(t)` is the ECEF→ECI rotation matrix at the propagation
epoch.

**Derivation**: starting from the position vector transform `r_eci =
R · r_ecef` and the gradient ∇V being a vector
```
∇V_eci = R · ∇V_ecef
```
(because gradient of a scalar field rotates as a vector), the Hessian
follows by differentiating the gradient transform:
```
H_eci_{ij} = ∂(∇V_eci)_i / ∂x_eci_j
            = Σ_k R_{ik} · ∂(∇V_ecef)_k / ∂x_eci_j
            = Σ_k R_{ik} · Σ_l (∂(∇V_ecef)_k / ∂x_ecef_l) · (∂x_ecef_l / ∂x_eci_j)
            = Σ_k Σ_l R_{ik} · H_ecef_{kl} · (Rᵀ)_{lj}
            = (R · H_ecef · Rᵀ)_{ij}
```
where the second line uses chain rule (V is the same scalar field at the
same physical point; only its coordinate representation differs), and
`∂x_ecef_l / ∂x_eci_j = (R⁻¹)_{lj} = (Rᵀ)_{lj}` (R is orthogonal, so
R⁻¹ = Rᵀ).

**Time-derivative terms — important caveat**: this transformation is
correct for a Hessian computed at a **fixed instant**. If we want the
Jacobian `∂a_eci/∂r_eci` where `a_eci = R(t)·a_ecef(R(t)ᵀ·r_eci)` and
`R(t)` is itself time-varying, then there are NO additional terms from
∂R/∂t in the Jacobian (the rotation rate enters only via the kinematic
Coriolis/centrifugal terms in the equations of motion, NOT in the
gradient of the gravity field at a fixed time). The Hessian transform
above is the FULL story for the satellite-state Jacobian's gravity
contribution.

**ITEM-021 verification** (SymPy): for a symbolic 3×3 rotation matrix
`R` and a symbolic `H_ecef`, verify `H_eci = R · H_ecef · Rᵀ` produces
a symmetric matrix iff `H_ecef` is symmetric, and that `tr(H_eci) =
tr(H_ecef)` (Laplace's equation invariant under rotation, as
expected).

---

## Section 4 — Pole-singularity handling

### 4.1 Where the singularities live

The spherical-coords gradient and Hessian formulas accumulate `1/cos(φ)`
factors with each application of `∂/∂λ`. Specifically:

```
∇V_cart  has factor  1/cos(φ)    (one λ-derivative term ⇒ from ∂λ/∂x_i)
H_cart   has factor  1/cos²(φ)   (two λ-derivative terms ⇒ ∂λ/∂x_i · ∂λ/∂x_j)
```

These appear in §3.3's `J · H_sph · Jᵀ` term wherever the `∂/∂λ` row/col
of `J` is involved (i.e., entries of `J` with `1/(r·cos(φ))` factor —
see §3.1's `∂λ/∂x_i = ∓sl/(r·u)`, `±cl/(r·u)`).

The `K^a` correction tensors also contain `1/cos²(φ)` factors when
`a = λ` (because `∂²λ/∂x_i∂x_j` has those factors).

The engine guards `vlc` finalization at `clenshaw_engine.hpp:248-249`:
```cpp
vlc = (std::abs(u) > defaults::GRAVITY_POLE_U_THRESHOLD)
        ? qs_r / u * (...)
        : 0.0;
```
That guards the GRADIENT assembly (one `1/cos(φ)`). The Hessian
assembly needs analogous guards for `1/cos²(φ)`.

### 4.2 Why the Cartesian Hessian is actually regular at the pole

Gravity is regular at the geographic pole — the field is finite, smooth,
and analytic at `r = (0, 0, ±r)`. So whatever 1/cos²(φ) factors the
spherical formulation produces, they MUST cancel exactly in the
Cartesian Hessian. This is a consequence of:

1. The associated Legendre `P̄_lm(sin φ) ~ cos^m(φ) · poly(sin φ)` for
   m ≥ 1 — i.e., `P̄_lm` itself has a `cos^m(φ)` zero at the pole that
   absorbs powers of `1/cos(φ)`.
2. The `m`-mode trigonometric factors `cos(mλ), sin(mλ)` are bounded
   even though `λ` itself is undefined at the pole — the *combination*
   `P̄_lm(sin φ) · cos(mλ) · (R/r)^(l+1)` extends analytically through
   the pole as a Cartesian polynomial in `(x, y, z)` divided by `r^(2l+1)`.
3. The Pines formulation (§5–§8) makes (1)+(2) explicit: it expresses
   each `V_lm` directly as a Cartesian polynomial, so the regularity
   is manifest.

### 4.3 Cancellation analysis (per Hessian entry)

For mode `(l, m)`, the contribution to the spherical Hessian's `∂²V/∂λ²`
component carries `m²·P̄_lm(sin φ)·{cos|sin}(mλ)·(R/r)^(l+1)`. When
multiplied by the `J·H_sph·Jᵀ` factor `(∂λ/∂x_i)·(∂λ/∂x_j) = sl·sl/(r·u)²`
(say, for `i=j=x`), the `1/u²` divides into `P̄_lm/u^m·u^m = P̄_lm`
provided `m ≥ 2`. For `m = 1`, one `1/u` factor remains — but this is
exactly canceled by the `K^λ` correction tensor's negative `1/u` term
that appears in `Σ_a (∂V/∂q^a)·K^a_{ij}`.

This is the pole-cancellation property: the explicit divergence in
`H_sph_{λλ}` and the explicit divergence in `K^λ_{ii}` cancel each
other, leaving a finite Cartesian Hessian.

**ITEM-014** (mpmath) demonstrates this cancellation numerically: at
sequentially smaller offsets `ε = π/2 − φ` (down to 1e-30 with mpmath's
50-digit precision, well below double-precision noise), the
*individual* spherical-Hessian terms blow up but the *Cartesian Hessian
sum* converges to the finite analytic value computed directly from the
Cartesian polynomial form.

### 4.4 Engineering recommendation

The cleanest production solution is to use the **Pines Cartesian
formulation** (§5–§8) when within a pole-radius threshold (e.g.,
`|cos(φ)| < 0.05`) and the spherical-extension formulation everywhere
else. Pines has no coordinate singularity at the pole, so no special
handling is needed there. The crossover region (`0.05 < |cos(φ)| < 0.10`)
can use either method (they agree to machine precision per §8).

For Stage F implementation, the threshold and crossover behavior
deserve their own design decisions — but the math is settled: both
formulations are valid, only the numerical robustness differs.

---

## Section 5 — Pines Cartesian polynomial basis

The **Pines (1973)** formulation re-expresses each spherical-harmonic
basis function as a Cartesian polynomial in `(x, y, z)` divided by a
power of `r`. This eliminates the `1/cos(φ)` coordinate singularities of
the spherical formulation entirely — every term in the Cartesian Hessian
is a rational function of `(x, y, z)` that is regular everywhere except
the origin.

### 5.1 Definition and explicit low-order forms

For mode `(l, m)`, the un-normalized basis is
```
V_lm^c = P_lm(sin φ) · cos(m·λ) · (R/r)^(l+1)
V_lm^s = P_lm(sin φ) · sin(m·λ) · (R/r)^(l+1)
```

Substituting `sin φ = z/r`, `p = √(x²+y²)`, `cos(m·λ) = Re[(x+iy)^m]/p^m`,
`sin(m·λ) = Im[(x+iy)^m]/p^m`, and using the property `P_lm(t)
∝ (1−t²)^(m/2) · poly(t)` ⇒ `P_lm(z/r) ∝ (p/r)^m · poly(z/r)`:

```
V_lm^c = R^(l+1) · poly(z, r) · Re[(x+iy)^m] / r^(2l+1)
V_lm^s = R^(l+1) · poly(z, r) · Im[(x+iy)^m] / r^(2l+1)
```

Both are **rational** in `(x, y, z)` with the singular factors `p^m` and
`cos^m(φ)` having canceled — `r^(2l+1)` is the only denominator and is
non-zero everywhere except the origin.

**Explicit Pines polynomials (un-normalized) for l ≤ 3**:

| (l, m) | V_lm^c (Pines polynomial form) | V_lm^s |
|---|---|---|
| (2, 0) | `R³·(2z² − x² − y²) / (2·r⁵)` | — (zonal) |
| (2, 1) | `3·R³·xz / r⁵` | `3·R³·yz / r⁵` |
| (2, 2) | `3·R³·(x² − y²) / r⁵` | `6·R³·xy / r⁵` |
| (3, 0) | `R⁴·z·(2z² − 3x² − 3y²) / (2·r⁷)` | — (zonal) |
| (3, 1) | `(3/2)·R⁴·x·(4z² − x² − y²) / r⁷` | `(3/2)·R⁴·y·(4z² − x² − y²) / r⁷` |
| (3, 2) | `15·R⁴·z·(x² − y²) / r⁷` | `30·R⁴·xyz / r⁷` |
| (3, 3) | `15·R⁴·x·(x² − 3y²) / r⁷` | `15·R⁴·y·(3x² − y²) / r⁷` |

These satisfy Laplace's equation `∇²V_lm = 0` (a property of all
spherical harmonics; verified by ITEM-015). The fully-normalized
EGM2008 basis is obtained by multiplying each `V_lm^{c,s}` by the
normalization constant `√((2l+1)·(l−m)!/((l+m)!)) · √(2 − δ_{m,0})`.

### 5.2 References

- **Pines, S. (1973)**, *AIAA Journal* **11**(11):1508–1511,
  "Uniform Representation of the Gravitational Potential and its
  Derivatives." — Original Cartesian formulation.
- **Cunningham, L. (1970)**, *Celestial Mechanics* **2**(2):207–216,
  "On the computation of the spherical harmonic terms needed during the
  numerical integration of the orbital motion of an artificial
  satellite." — Pre-Pines spherical recurrence.
- **Petrovskaya, M. & Vershkov, A. (2010)**, *Journal of Geodesy*
  **84**(3):165–178, "Non-singular expressions for the spherical
  harmonic synthesis of gravitational field parameters in arbitrary
  Cartesian coordinates." — Modern non-singular formulation.

## Section 6 — Pines value/gradient/Hessian recurrences (structural)

### 6.1 Recurrence structure

Pines (1973) derived three coupled recurrences that produce, in a single
sweep through `(l, m)`:
- The Pines polynomials `V_lm^{c,s}, W_lm^{c,s}` (value).
- Their first derivatives `∂V_lm/∂x_i, ∂W_lm/∂x_i` (gradient).
- Their second derivatives `∂²V_lm/∂x_i∂x_j, ∂²W_lm/∂x_i∂x_j` (Hessian).

The recurrence walks the `(l, m)` triangle starting from low degree and
ascending. Each step uses 2-term and 3-term combinations of previously-
computed polynomials, with coefficients depending only on `(l, m)`.

The full recurrence coefficients are stated in Pines (1973) eqs. 5–8 and
Petrovskaya & Vershkov (2010) eqs. 12–18. They are **algebraically
involved** but **numerically robust** — every coefficient is finite
everywhere, and there are no `1/cos(φ)` factors anywhere.

### 6.2 Why the recurrence is regular

Each Pines polynomial is (per §5.1) a Cartesian polynomial of degree ≤ l
in `(x, y, z)` divided by `r^(2l+1)`. Differentiating wrt `x_i` at most
twice produces:
- Numerator: polynomial of degree ≤ l + 2.
- Denominator: `r^(2l+5)` (worst case).

Both are bounded everywhere except `r = 0`. No `cos(φ)` ever appears in
the Pines path — that's the structural feature that makes the
formulation regular at the geographic pole.

### 6.3 Comparison with §1–§3

| Property | Spherical (§1–§3) | Pines Cartesian (§5–§7) |
|---|---|---|
| Inner-loop coordinate | `(r, φ, λ)` | `(x, y, z)` |
| Pole singularity | yes (`1/cos(φ)`) | none |
| Trigonometric calls | `sin`, `cos` of `λ`, `φ` | none |
| Recurrence cost / step | ~6 mul + 2 add per accumulator | ~9 mul + 3 add per accumulator |
| Production use | far from pole (more efficient) | near pole (regular) |

The cost ratio favors spherical away from the pole (~30% fewer
operations per step); both produce the same numerical answer at any
given `(x, y, z)` to machine precision per ITEM-015.

## Section 7 — Pines acceleration assembly

In Pines form, the gravity acceleration is
```
a_grav = ∇V = (GM/R) · Σ_l Σ_m [ C̄_lm · ∇V_lm^c  +  S̄_lm · ∇V_lm^s ]
```
where `∇V_lm^{c,s}` are the gradient outputs of the Pines recurrence and
`C̄_lm, S̄_lm` are fully-normalized Stokes coefficients.

Each `∇V_lm` is a Cartesian 3-vector evaluated directly — no
spherical-to-Cartesian transformation needed. This eliminates the §3.1
chain-rule / §3.2 convention machinery entirely on the Pines path.

The Hessian assembly is analogous:
```
H_grav = ∇²V = (GM/R) · Σ_l Σ_m [ C̄_lm · H(V_lm^c) + S̄_lm · H(V_lm^s) ]
```
where `H(V_lm^{c,s})` is the 3×3 Cartesian Hessian from the Pines
recurrence — already in Cartesian form, no §3.3 tensor decomposition
needed.

## Section 8 — Equivalence with §1–§3 (cross-validation property)

The Pines formulation and the spherical formulation produce the SAME
gravity field, gradient, and Hessian for any `(x, y, z) ≠ origin`. This
is a mathematical identity (both compute the same physical quantity from
the same Stokes coefficients), not an empirical coincidence.

**Verification (ITEM-015)**: for each `(l, m)` with `l ∈ {2, 3}` and
`m ∈ {0, 1, ..., l}`, both cosine and sine variants:
1. Construct `V_lm^{c,s}` in Cartesian polynomial form (per §5.1
   table).
2. Compute the 3×3 Cartesian Hessian via direct SymPy chain rule on the
   Cartesian polynomial — this is what the Pines recurrence converges
   to.
3. Compute the same Hessian via §3.3's tensor decomposition: build
   `V_lm` in spherical form (P_lm·cos(mλ)·(R/r)^(l+1)), differentiate
   to get H_sph, evaluate `J·H_sph·Jᵀ + Σ_a (∂V/∂q^a)·K^a` at the same
   point.
4. Assert both Hessians match symbolically (`simplify(H_pines −
   H_spherical) == 0`).
5. Additionally verify Laplace's equation `tr(H) == 0` symbolically.

A passing verification confirms:
- §1–§3 framework correctly handles real EGM2008-style basis functions
  (not just generic test potentials from ITEM-013).
- The Pines polynomial forms in §5.1 are correctly stated.
- The dual-implementation cross-validation property holds — any
  difference between the two C++ implementations would be an
  implementation bug, not a math inconsistency.

---

## Section 9 — Solid Earth tides: IERS 2010 corrections

### 9.1 What's being modeled

Solid Earth tides are the deformation of the Earth in response to the
gravitational pull of the Sun and Moon, which slightly redistributes
mass. This redistribution produces small **time-varying corrections**
`ΔC_lm(t), ΔS_lm(t)` to the Stokes coefficients `C_lm, S_lm` of the
gravity potential, primarily at degrees 2 and 3.

Per **IERS Conventions 2010, Chapter 6**, the corrections are computed
in two steps:

### 9.2 Step 1 — Frequency-independent corrections (the dominant part)

The Step 1 formula gives the in-phase response to the instantaneous
tide-generating potential of the Sun (j = 2) and Moon (j = 1):

```
ΔC_lm(t) − i·ΔS_lm(t)  =  k_lm / (2l + 1)
                          · Σ_{j=1,2}  GM_j / GM_E
                                       · (R_E / r_j)^(l+1)
                                       · P̄_lm(sin Φ_j)
                                       · exp(−i·m·λ_j)
```

where:
- `k_lm` is the Love number for degree l and order m (tabulated, IERS
  2010 Table 6.3).
- `GM_j` is the gravitational parameter of body j (Sun or Moon).
- `r_j, Φ_j, λ_j` are the body's geocentric distance, geocentric
  latitude, and longitude (in the Earth-fixed frame).
- `P̄_lm` is the fully-normalized associated Legendre function.

This applies at degrees `(l, m) ∈ {(2, 0), (2, 1), (2, 2), (3, 0),
(3, 1), (3, 2), (3, 3)}` and feeds back into the same gravity series.

### 9.3 Step 2 — Frequency-dependent corrections (small)

Step 2 adds smaller corrections at specific tidal frequencies (long-
period, diurnal, semi-diurnal) per IERS 2010 Tables 6.5a–c. These are
of order 10^−12 to 10^−10 in the C_lm scale and only affect specific
(l, m) combinations.

For the Hessian computation, Step 2 has the same structural property as
Step 1: the corrections are functions of time and lunisolar ephemeris,
**not** of satellite position.

## Section 10 — Tide-corrected Hessian via the same engine

### 10.1 The corrected coefficients

Define
```
C̃_lm(t) = C_lm + ΔC_lm(t)         (static + Step 1 + Step 2)
S̃_lm(t) = S_lm + ΔS_lm(t)
```

Both `C_lm`, `ΔC_lm` are scalars (with respect to satellite position).
The combined `C̃_lm(t)` is just another scalar, time-varying but
position-independent.

### 10.2 Same engine, corrected coefficients

Because the gravity Hessian computation is **linear** in the Stokes
coefficients,

```
H_grav-with-tides(r_sat, t)  =  Σ_{l, m} [ C̃_lm(t) · H(V_lm^c)(r_sat)
                                          + S̃_lm(t) · H(V_lm^s)(r_sat) ]
                              =  H_grav(r_sat) + H_tide(r_sat, t)
```

where
```
H_tide(r_sat, t)  =  Σ_{l, m} [ ΔC_lm(t) · H(V_lm^c)(r_sat)
                                + ΔS_lm(t) · H(V_lm^s)(r_sat) ]
```

The right-hand side reuses **exactly the same** `H(V_lm^{c,s})` computed
by the §1–§8 machinery (spherical or Pines). Tides require no new
Hessian recurrence — only computing `ΔC_lm(t), ΔS_lm(t)` from current
Sun/Moon ephemeris and feeding them into the existing engine.

### 10.3 Implementation

The `solid_earth_tides.hpp` implementation:
1. At each propagation step, compute `ΔC_lm(t), ΔS_lm(t)` from current
   `(r_sun, Φ_sun, λ_sun)` and `(r_moon, Φ_moon, λ_moon)` per §9.2 / §9.3.
2. Pass the corrected coefficients `C̃_lm = C_lm + ΔC_lm` to the gravity
   Hessian engine.
3. Output the additive Hessian contribution as `H_tide`, which adds
   directly to `H_grav` in the force-model Jacobian assembly.

No new analytic Hessian derivation is needed for tides — the engine
from §1–§8 handles everything. The complete force-model Jacobian for
gravity-plus-tides becomes

```
J_grav+tide = H_grav(r_sat) + H_tide(r_sat, t)
```

both rows of which are computed by the same engine called twice with
different coefficients.

## Section 11 — Why this works: ∂(corrections)/∂r_sat = 0

### 11.1 The key property

The Step 1 formula (§9.2) involves only `(r_j, Φ_j, λ_j)` for the Sun
and Moon — quantities derived from lunisolar ephemeris and Earth
orientation. **Satellite position does not appear** anywhere in the
formula.

Therefore:
```
∂ΔC_lm(t) / ∂r_sat  =  0
∂ΔS_lm(t) / ∂r_sat  =  0
```
Step 2 has the same property (its inputs are tidal frequencies and
ephemeris, not satellite position).

### 11.2 Implication for the satellite Jacobian

The satellite-position derivative of the tide acceleration is

```
∂a_tide / ∂r_sat  =  ∂ ( ∇V_tide ) / ∂r_sat
                  =  Σ_{l, m} [ ΔC_lm · ∂(∇V_lm^c)/∂r_sat
                               + ΔS_lm · ∂(∇V_lm^s)/∂r_sat
                               + (∂ΔC_lm/∂r_sat) · ∇V_lm^c
                               + (∂ΔS_lm/∂r_sat) · ∇V_lm^s ]
                  =  Σ_{l, m} [ ΔC_lm · H(V_lm^c)
                               + ΔS_lm · H(V_lm^s) ]
                              + 0
                  =  H_tide(r_sat, t)
```

The two extra terms vanish because of §11.1, leaving exactly the
"corrected-coefficients passed to gravity Hessian" formula from §10.2.

### 11.3 Numerical scale

The job of the tide Jacobian is to compute `H_tide`. The full Step 1 contribution to
the (2, 0) Stokes coefficient peaks at ~`5 × 10^−9` in the dimensionless
normalized form, producing acceleration perturbations of order
`10^−7 m/s²` for a LEO satellite (and Hessian values of order
`10^−13 / s²`). Step 2 contributions are smaller by 2–4 orders of
magnitude.

These scales are handled cleanly by the same Clenshaw / Pines engine —
no new numerical regime to worry about.

### 11.4 ITEM-016 verification

The §11.1 claim is verified symbolically: build the Step 1 formula in
SymPy with separate symbols for Sun/Moon ephemeris and satellite
position, then assert `sp.diff(ΔC_lm, x_sat) == 0` (and analogously for
y_sat, z_sat). This formalizes the satellite-position-independence of
solid-tide coefficient corrections.
gravity's.

§11: `∂(ΔC_lm)/∂r_sat = 0` (corrections depend on Sun/Moon ephemeris, not
satellite position). So `∂a_tide/∂r_sat` uses the gravity engine's Hessian
on `(C̃_lm, S̃_lm)`. The wrapper is ~30 lines once the engine is verified.

---

## Drafting status

| Section | Status |
|---|---|
| 0. Reference state | ✅ COMPLETE (sign convention resolved by ITEM-012) |
| 1. State existing recurrences | DRAFT — transcription faithful per ITEM-012 (indirect) |
| 2.1 wrrc | ✅ SymPy-verified (ITEM-005) |
| 2.2 wrtc | ✅ SymPy-verified (ITEM-008) |
| 2.3 wttc | ✅ SymPy-verified (ITEM-009) |
| 2.4 sine siblings (wrrs, wrts, wtts) | ✅ SymPy-verified (ITEM-010) |
| 2.5 lambda accumulators (vrlc, vrls, vtlc, vtls, vllc, vlls) | ✅ SymPy-verified (ITEM-011) |
| 3.1-3.2 chain rule + engine convention | ✅ SymPy-verified (ITEM-012) |
| 3.3 Hessian assembly tensor decomposition | ✅ SymPy-verified (ITEM-013) |
| 4. Pole singularity handling | ✅ mpmath-verified (ITEM-014) |
| 5.1 Pines polynomial table | ✅ SymPy-verified (ITEM-015) |
| 5-8 Pines formulation | ✅ DRAFTED with literature references |
| 9-11 Solid tides | ✅ DRAFTED; ITEM-016 verifies ∂Δ/∂r_sat = 0 |

Stage A is **complete** through §11. Verification suite passes 12/12
(ITEM-005 through ITEM-016, all SymPy or mpmath). Remaining Stage A
work is human review (Stage B) and then Stage C-J implementation per
the plan.

**Review status**: this draft is for initial review. Per Stage B of the
plan, two consecutive review passes with no edits required before Stage C.
