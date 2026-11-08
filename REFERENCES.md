# Ballistic Library - Reference Sources

This document lists the authoritative sources for all algorithms, coefficients, and models implemented in this library.

---

## Primary References

### Orbital Mechanics & Astrodynamics

1. **Vallado, D.A.** (2013). *Fundamentals of Astrodynamics and Applications*, 4th Edition.
   Microcosm Press.
   - Zonal harmonics: Section 8.7.1, Equations 8-25 through 8-30
   - Third-body perturbations: Section 8.6, Equation 8-36
   - Coordinate transformations: Chapter 3
   - Time systems: Chapter 3.5

2. **Montenbruck, O. & Gill, E.** (2000). *Satellite Orbits: Models, Methods, and Applications*.
   Springer-Verlag.
   - Spherical harmonics gravity: Chapter 3.2
   - Clenshaw summation: Chapter 3.2.4
   - Third-body perturbations: Chapter 3.3, Equation 3.34
   - Force model magnitudes: Table 3.1

3. **Curtis, H.D.** (2014). *Orbital Mechanics for Engineering Students*, 3rd Edition.
   Butterworth-Heinemann.
   - Two-body problem: Chapter 2
   - Orbit determination: Chapter 5

### Time and Reference Systems

4. **McCarthy, D.D. & Petit, G.** (2004). *IERS Conventions (2003)*.
   IERS Technical Note No. 32, Verlag des Bundesamts fur Kartographie und Geodasie.
   - Fundamental arguments: Table 5.1
   - Precession-nutation: Chapter 5
   - Earth Rotation Angle: Equation 5.15

5. **Petit, G. & Luzum, B.** (2010). *IERS Conventions (2010)*.
   IERS Technical Note No. 36, Verlag des Bundesamts fur Kartographie und Geodasie.
   - Updated precession-nutation: Chapter 5
   - Solid Earth tides: Chapter 6.2, Equation 6.6, Table 6.3
   - Time scales: Chapter 10

6. **Meeus, J.** (1998). *Astronomical Algorithms*, 2nd Edition.
   Willmann-Bell.
   - Julian Date: Chapter 7
   - Sidereal time: Chapter 12
   - TDB-TT approximation: Chapter 10

### Precession-Nutation Models

7. **Capitaine, N., Wallace, P.T., & Chapront, J.** (2003). "Expressions for IAU 2000
   precession quantities". *Astronomy & Astrophysics*, 412, 567-586.
   - IAU 2006/2000A nutation series coefficients (with J2 rate correction)

8. **Fukushima, T.** (2003). "A New Precession Formula".
   *The Astronomical Journal*, 126, 494-534.
   - Fukushima-Williams parameterization of precession

### Gravity Models

9. **Pavlis, N.K., Holmes, S.A., Kenyon, S.C., & Factor, J.K.** (2012). "The development
   and evaluation of the Earth Gravitational Model 2008 (EGM2008)".
   *Journal of Geophysical Research*, 117, B04406. DOI: 10.1029/2011JB008916.
   - EGM2008 spherical harmonic coefficients (degree 2190)
   - Coefficient source: NGA `EGM2008_to2190_TideFree`

10. **Holmes, S.A. & Featherstone, W.E.** (2002). "A unified approach to the Clenshaw
    summation and the recursive computation of very high degree and order normalized
    associated Legendre functions". *Journal of Geodesy*, 76(5), 279-299.
    DOI: 10.1007/s00190-002-0216-2.
    - Clenshaw summation algorithm for spherical harmonics
    - Numerically stable recursion for associated Legendre functions

11. **Lemoine, F.G., et al.** (1998). "The Development of the Joint NASA GSFC and
    NIMA Geopotential Model EGM96". NASA/TP-1998-206861.
    - EGM96 spherical harmonic coefficients (historical, superseded by EGM2008)

12. **NIMA** (2000). *Department of Defense World Geodetic System 1984*.
    NIMA TR8350.2, 3rd Edition.
    - WGS84 parameters
    - J2-J6 zonal harmonic coefficients

### Ephemeris

13. **Park, R.S., et al.** (2021). "The JPL Planetary and Lunar Ephemerides DE440 and
    DE441". *The Astronomical Journal*, 161(3), 105. DOI: 10.3847/1538-3881/abd414.
    - JPL DE440 planetary ephemeris (1550-2650)
    - GM values: `gm_de440.tpc` kernel

### Atmosphere Models

14. **Emmert, J.T., Drob, D.P., Picone, J.M., et al.** (2021). "NRLMSIS 2.0: A
    whole-atmosphere empirical model of temperature and neutral species densities".
    *Earth and Space Science*, 8(3), e2020EA001321. DOI: 10.1029/2020EA001321.
    - NRLMSIS 2.1 atmosphere model (full implementation)
    - Species: N2, O2, O, He, H, Ar, N, anomalous O

15. **COSPAR** (1986). *CIRA 1986: COSPAR International Reference Atmosphere*.
    Advances in Space Research, Vol. 10, No. 12.
    - Standard atmospheric profiles for validation

16. **US Standard Atmosphere** (1976). NOAA-S/T 76-1562.
    - 0-86 km model computed from its defining constants (`atmosphere/ussa76.hpp`)

16a. **Drob, D.P., et al.** (2015). "An update to the Horizontal Wind Model (HWM): The
    quiet time thermosphere". *Earth and Space Science*, 2, 301-319.
    DOI: 10.1002/2014EA000089.
    - HWM14 horizontal wind model (C++ translation of `hwm14.f90`)

16b. **Emmert, J.T., et al.** (2008). "DWM07 global climatological model of upper
    thermospheric storm-induced disturbance winds". *JGR*, 113, A11319.
    - Disturbance winds of HWM14

### Radiation Pressure

17. **Knocke, P.C., Ries, J.C., & Tapley, B.D.** (1988). "Earth radiation pressure
    effects on satellites". *AIAA/AAS Astrodynamics Conference*, AIAA-88-4292.
    - Earth albedo and IR radiation pressure model

### Relativistic Effects

18. **Soffel, M., et al.** (2003). "The IAU 2000 Resolutions for Astrometry, Celestial
    Mechanics, and Metrology in the Relativistic Framework". *The Astronomical Journal*,
    126, 2687-2706.
    - Schwarzschild and Lense-Thirring relativistic corrections

### Coordinate Systems

19. **Vermeille, H.** (2011). "An analytical method to transform geocentric into
    geodetic coordinates". *Journal of Geodesy*, 85, 105-117.
    - Closed-form ECEF-to-geodetic conversion

### Orbit Prediction from Element Sets

19a. **Hoots, F.R. & Roehrich, R.L.** (1980). "Models for Propagation of NORAD Element
    Sets". Spacetrack Report No. 3.
    - SGP4/SDP4 (clean-room implementation, `sgp4/sgp4.hpp`)

19b. **Vallado, D.A., Crawford, P., Hujsak, R., & Kelso, T.S.** (2006). "Revisiting
    Spacetrack Report #3". AIAA 2006-6753.
    - Corrections applied to SGP4/SDP4; SGP4-VER verification output

### Reentry Aerodynamics

19c. **Sentman, L.H.** (1961). Free-molecule flow theory; **Schaaf, S.A. & Chambre, P.L.**
    (1958). *Flow of Rarefied Gases*, Princeton.
    - Free-molecular drag coefficients (`aero/flow_regime.hpp`)

19d. **Henderson, C.B.** (1976). "Drag coefficients of spheres in continuum and rarefied
    flows". *AIAA Journal*, 14(6), 707.
    - Continuum sphere drag (`aero/flow_regime.hpp`)

19e. **Klinkrad, H.** (2006). *Space Debris: Models and Risk Analysis*, Springer, sec. 5.4.
    - Transitional bridging function of Koppenwallner & Legge

### Software References

20. **SOFA (Standards of Fundamental Astronomy)**. IAU SOFA Library.
    http://www.iausofa.org/
    - Reference implementation for IAU 2006/2000A transformations
    - Functions: `iauNut06a`, `iauPnm06a`, `iauEra00`, `iauGmst06`

21. **Karney, C.F.F.** (2011). GeographicLib library documentation.
    https://geographiclib.sourceforge.io/
    - Reference implementation for gravity field evaluation

22. **JPL Horizons System**. NASA Jet Propulsion Laboratory.
    https://ssd.jpl.nasa.gov/horizons/
    - Solar system ephemerides for validation

---

## Coefficient Sources by Module

### Time (`time/julian_date.hpp`, `time/epoch.hpp`)
- Julian Date algorithm: Meeus [6], Chapter 7
- Leap second table: IERS Bulletin C
- TDB-TT correction: Meeus [6], Chapter 10, Equation 10.2

### Coordinates (`coordinates/iau_transforms.hpp`)
- Fundamental arguments (l, l', F, D, Om): IERS TN 36 [5], Table 5.2a
- Planetary longitudes: SOFA `iauFaom03()` and related functions
- Precession angles (Fukushima-Williams): Fukushima [8], Table 1
- Nutation series: Capitaine et al. [7], Table 1 (1365 terms, nut06a)
- Earth Rotation Angle: IERS TN 36 [5], Equation 5.15
- CIO locator (s): IERS TN 36 [5], Equation 5.13

### Coordinates (`coordinates/earth_rotation.hpp`)
- Earth Rotation Angle: IERS TN 36 [5], Equation 5.15
- GMST polynomial: IERS TN 36 [5], Equation 5.32
- Mean obliquity: IAU 2006 value from IERS TN 36 [5]

### Coordinates (`coordinates/geodetic.hpp`)
- WGS84 ellipsoid parameters: NIMA TR8350.2 [12]
- Geodetic conversion: Vermeille [19]

### Gravity (`gravity/point_mass.hpp`)
- J2-J15 formulation: Vallado [1], Equations 8-25 through 8-30
- J2-J15 coefficient values: from the embedded EGM2008 C̄n0 (Pavlis et al. [9]), Jn = -C̄n0·sqrt(2n+1) (`core/constants.hpp`)

### Gravity (`gravity/egm_gravity.hpp`, `gravity/clenshaw_engine.hpp`)
- EGM2008 coefficients: Pavlis et al. [9], NGA `EGM2008_to2190_TideFree`
- Clenshaw summation: Holmes & Featherstone [10]
- GeographicLib cross-reference: Karney [21]

### Gravity (`gravity/third_body.hpp`)
- Third-body formulation: Vallado [1], Eq. 8-36; Montenbruck [2], Eq. 3.34
- GM values: JPL DE440 `gm_de440.tpc` (Park et al. [13])

### Gravity (`gravity/solid_earth_tides.hpp`)
- IERS 2010 conventions: Petit & Luzum [5], Chapter 6.2, Equation 6.6
- Love numbers: IERS TN 36 Table 6.3 (anelastic, degree 2-3)

### Math (`math/chebyshev.hpp`)
- Clenshaw recurrence: Clenshaw [2]; Numerical Recipes [1] §5.8
- Derivative recurrence with c'_0 halving: Numerical Recipes [1] §5.8, Eq. 5.8.11

### Math (`math/interpolation.hpp`)
- Lagrange interpolation: Numerical Recipes [1] §3.1; Montenbruck [2] §3.3.2

### Math (`math/quintic_hermite.hpp`)
- Quintic Hermite basis: Burden & Faires [1], Ch. 3; Hairer, Norsett & Wanner, Ch. II.6
- Dense output for ODE solvers: Hairer et al. "Solving ODEs I", 2nd Ed., Springer, 1993

### Ephemeris (`ephemeris/spk_ephemeris.hpp`, `src/ephemeris/de440_coefficients.cpp`)
- JPL DE440: Park et al. [13]
- Chebyshev interpolation of SPK segments

### Atmosphere (`atmosphere/internal/nrlmsis21_full.hpp`)
- NRLMSIS 2.1: Emmert et al. [14]
- Full parameterization with species-level density computation

### Atmosphere (`atmosphere/internal/hwm14.hpp`)
- HWM14: Drob et al. [16a]; DWM07: Emmert et al. [16b]

### Atmosphere (`atmosphere/ussa76.hpp`)
- US Standard Atmosphere 1976, 0-86 km: NOAA [16]

### Aero (`aero/flow_regime.hpp`)
- Sentman / Schaaf & Chambre [19c], Henderson [19d], Klinkrad [19e]

### SGP4 (`sgp4/`)
- Hoots & Roehrich [19a], Vallado et al. [19b]

### Forces (`forces/radiation_pressure.hpp`)
- Solar radiation pressure: cannonball model
- Earth albedo: Knocke et al. [17]

### Forces (`forces/relativistic.hpp`)
- Schwarzschild + Lense-Thirring: Soffel et al. [18]; IERS TN 36 [5]

---

## Validation Data Sources

- **ESA Sentinel EOF**: Precise orbit ephemerides for LEO validation (6 satellites)
- **IGS Precise Orbits**: International GNSS Service SP3 files (GPS, QZSS)
- **ILRS Combined Product**: SLR-derived orbits for LAGEOS and Etalon
- **GFZ SP3**: GRACE-FO and CHAMP precise orbits
- **ERFA/SOFA**: Reference IAU transformation test cases
- **JPL Horizons**: Reference ephemerides for Sun, Moon, planets

---

## Notes on Implementation Accuracy

### IAU 2006/2000A Precession-Nutation
The full IAU 2006/2000A nutation model (nut06a) has 1365 terms with the IAU 2006
J2 rate correction applied. This library implements:
- 678 luni-solar + 687 planetary terms (complete series)
- IAU 2006 J2 rate correction for consistency with IAU 2006 precession (pfw06)
- SOFA equivalent: iauNut06a + iauPfw06 = iauPnm06a

### NRLMSIS 2.1 Atmosphere
Full implementation of NRLMSIS 2.1 (Emmert et al. 2021) with all species,
diffusive equilibrium, and empirical F10.7/Ap parameterization. Reference validation:
200/200 test cases pass with 0.0% average density error against Fortran reference.

### EGM2008 Spherical Harmonics
Embedded coefficients up to degree/order 360 from the NGA `EGM2008_to2190_TideFree`
file. Default evaluation at degree 70 (sufficient for LEO). Clenshaw summation
algorithm from Holmes & Featherstone (2002) for numerical stability at high degrees.
16 spot-check coefficients verified against NGA source at machine precision (1e-15).
