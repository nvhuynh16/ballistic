# GOCE thermosphere density (ESA / TU Delft)

**Committed here**: `goce_denswind_ac082_v2_0_2013-10-15.txt` — a **verbatim
slice** (full `#` header block + all 8640 rows dated 2013-10-15) of the
official month file. It is a slice, not a re-derivation: every value is
byte-identical to the source, so provenance survives.

- Product: GOCE+ Theme 3 "Air density and wind retrieval using GOCE data",
  v2.0, accommodation coefficient 0.82, ESA HPF-funded, exported at TU Delft.
- Parent file: `goce_denswind_ac082_v2_0_2013-10.txt` (22 MB, sha256
  `71c6722ff871c93f…`), NOT committed.
- Slice command:
  ```
  grep '^#' goce_denswind_ac082_v2_0_2013-10.txt  > goce_denswind_ac082_v2_0_2013-10-15.txt
  awk '$1=="2013-10-15"' goce_denswind_ac082_v2_0_2013-10.txt >> goce_denswind_ac082_v2_0_2013-10-15.txt
  ```
- Full month + the deorbit files (`goce_denswind_ac082_v2_0_2013-10.txt`,
  `..._deorb-DFACS.txt`, `..._deorb-GPSonly.txt`): the ESA GOCE+ Theme 3 v2.0
  product as distributed by ESA (earth.esa.int, GOCE thermosphere data; the
  file header names the ESA EO helpdesk). No script in the repository fetches
  them; place them here by hand. None is committed (the `deorb-DFACS` file is
  211 MB).

**A different product**: `python scripts/download_goce_density.py` downloads
the TU Delft **v02** accelerometer densities `GO_DNS_ACC_2013_10_v02.zip` and
`GO_DNS_ACC_2013_11_v02.zip` (thermosphere.tudelft.nl, version_02) and
extracts them here (`GO_DNS_ACC_2013_10_v02.txt`, ...). They feed the
`accel_density` configuration of `scripts/validate_goce_reentry.py`; they are
not the ac082 v2.0 files above and are not committed.

**Consumers**: `scripts/goce_density_truth.py` (prefers the full month when
present, falls back to this slice) and gate T4
`tests/python/test_goce_density_truth.py` — the library's absolute-density
anchor (median NRLMSIS/measured ratio 0.937 at ~250 km).

