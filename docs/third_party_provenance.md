# Third-Party Data Provenance and Regeneration

Companion to `THIRD_PARTY_NOTICES.md` (the terms live there; this file
records where every embedded artifact came from and how to regenerate it).
The `external/` directory holds local copies of the upstream inputs and the
official test vectors; it is git-ignored, because its contents are public
downloads, not repository material.

| Embedded artifact | Upstream source | Local input (`external/`, git-ignored) | Converter |
|---|---|---|---|
| `include/ballistic/atmosphere/internal/nrlmsis21_parm.hpp` | https://map.nrl.navy.mil/map/pub/nrl/NRLMSIS/NRLMSIS2.1/ (`nrlmsis2.1.zip`) | `msis21.parm` | `scripts/convert_msis21_parm.py` |
| `include/ballistic/atmosphere/internal/nrlmsis21_full.hpp` | same package (`msis_*.F90`) | `msis_calc.F90`, `msis_constants.F90`, `msis_dfn.F90`, `msis_gfn.F90`, `msis_init.F90`, `msis_tfn.F90`, `msis_utils.F90` | hand translation to C++ (verified against `msis2.1_test_ref_dp.txt`) |
| `include/ballistic/atmosphere/internal/hwm14_coefficients.hpp` | https://map.nrl.navy.mil/map/pub/nrl/HWM/HWM14/ (also the Supporting Information of doi:10.1002/2014EA000089) | `hwm14/hwm123114.bin`, `hwm14_repo/dwm07b104i.dat`, `hwm14_repo/gd2qd.dat` | `scripts/convert_hwm14_data.py` |
| `include/ballistic/atmosphere/internal/hwm14.hpp` | same package (`hwm14.f90`) | `hwm14_repo/src/…` | hand translation to C++ |
| `src/ephemeris/de440_coefficients.cpp` (declarations in `include/ballistic/ephemeris/de440_coefficients.hpp`) | https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/ (`de440.bsp`) | `de440.bsp` (or the platform data directory) | `scripts/extract_de440_subset.py` (1950–2100 subset; Sun, EMB, Moon, Jupiter barycenter) |
| `src/gravity/egm2008_coefficients.cpp` (declarations in `include/ballistic/gravity/egm2008_coefficients.hpp`) | https://earth-info.nga.mil/ `EGM2008_to2190_TideFree` | NGA coefficient file | `scripts/generate_egm2008_coefficients.py` (degrees 2–360 embedded; degrees 361–2190 loadable at run time via `config.egm2008_path`) |
| `include/ballistic/coordinates/nutation_coefficients.hpp` | IERS TN 32, Tables 5.3a/b (Mathews et al. 2002) | — | extracted from ERFA `eraNut00a()`, checked against SOFA `iauNut00a()` (see the file's header) |

## Generator output layout

Some generators write the layout the tree had when they were first run, and
the files were moved or split afterwards. A re-run must repeat those moves
by hand:

- `scripts/generate_egm2008_coefficients.py` and
  `scripts/extract_de440_subset.py` write a single header holding the data
  (`include/ballistic/gravity/egm2008_coefficients.hpp`,
  `include/ballistic/ephemeris/de440_coefficients.hpp`). The data now lives
  in `src/gravity/egm2008_coefficients.cpp` and
  `src/ephemeris/de440_coefficients.cpp`; the headers hold only the
  `extern const` declarations.
- `scripts/convert_msis21_parm.py` and `scripts/convert_hwm14_data.py`
  write into `include/ballistic/atmosphere/`; the files live in
  `include/ballistic/atmosphere/internal/`.
- The two converters do not write the license statement and the change
  notice that the NRL-derived headers carry (NRL-SOF-014-1 §4b/§4c for
  NRLMSIS; the attribution notice for HWM14). Copy those header comments
  into a regenerated file.

## Official test vectors kept in `external/`

- `msis2.1_test_in.txt` / `msis2.1_test_ref_dp.txt`: NRLMSIS 2.1
  double-precision reference outputs shipped with the official package; the
  C++ translation is validated against these (see the atmosphere validation
  tests).
- `hwm14_repo/`: a mirror of the official HWM14 package including
  `README.txt` and the check values (`checkhwm14.f90` outputs).

## Regeneration quick reference

```bash
# NRLMSIS 2.1 parameters (reads external/msis21.parm)
python scripts/convert_msis21_parm.py

# HWM14 coefficients
python scripts/convert_hwm14_data.py <path-to-hwm14-data-dir>

# DE440 subset
python scripts/extract_de440_subset.py path/to/de440.bsp \
    include/ballistic/ephemeris/de440_coefficients.hpp

# EGM2008 degrees 2-360 (reads data/EGM2008_Spherical_Harmonics/EGM2008_to2190_TideFree)
python scripts/generate_egm2008_coefficients.py
```

Each script's header is the single source of truth for its exact arguments
and output layout.
