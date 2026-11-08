# Sentinel precise orbit files (ESA EOF)

Not committed (the directory is git-ignored apart from this file): Copernicus
Sentinel precise orbit ephemerides (`POEORB` EOF/XML files) used by the
Sentinel validation tests and campaigns.

| Directory | Content | Used by |
|---|---|---|
| `sentinel1_2016/2014-10/` | Sentinel-1A POEORB files of October 2014 | `tests/validation/test_sentinel1_validation.cpp`, `tests/validation/test_egm2008_diagnostics.cpp`, `tests/validation/test_gcrs_validation.cpp` (first file of `sentinel1_2016/` in sorted order) |
| `sentinel1_2016/2016-1/` | Sentinel-1A POEORB files of January 2016 | `tests/python/test_validation_eof.py` |
| `sentinel1_2016/2016-8/` | Sentinel-1A POEORB files of August 2016 | `tests/validation/test_esa_pod_validation.cpp`, `tests/validation/test_multiday_propagation.cpp`, `tests/validation/test_trajectory_stepping.cpp` |
| `sentinel_jan2026/` | January 2026 POEORB files of S1A, S1C, S2B, S2C, S3A, S3B | `tests/validation/test_multiday_propagation.cpp` (joined Sentinel-1C days), `tests/python/test_recommended_config.py`, `scripts/validate_sentinel_srp_drag.py`, `scripts/validate_wind_effect.py` |

Source: the Copernicus Data Space Ecosystem (https://dataspace.copernicus.eu/),
which serves the POEORB products of every Sentinel mission; a free account is
required. Place the downloaded `*.EOF` files (or the `.zip`/`.TGZ` archives
they come in) in the directories above. The tests that need them skip with a
message naming this file when they are absent.

Attribution: "Contains modified Copernicus Sentinel data [2014, 2016, 2026]".
