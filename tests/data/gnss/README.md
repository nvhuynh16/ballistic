# GNSS satellite metadata excerpt

`igs_satellite_metadata_excerpt.snx` is a verbatim excerpt (56 lines) of the public IGS
satellite metadata file, release 2441 (SINEX header dated 2026 day 295):
https://files.igs.org/pub/station/general/igs_satellite_metadata.snx (no login).
It keeps the SATELLITE/IDENTIFIER, PRN and MASS rows for the satellites of the committed
SP3 test files: GPS G01/G02 (SVNs G063/G061 on 2024 day 015, plus G049/G080 that held G01
later), QZSS J02/J03/J04 and BeiDou C01-C05. Read it with `scripts/_gnss_metadata.py`.
Refresh: `python scripts/download_igs_metadata.py --force --check-excerpt
tests/data/gnss/igs_satellite_metadata_excerpt.snx` (checks every row is still in the file).
