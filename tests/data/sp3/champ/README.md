# CHAMP Precise Orbit Data

This directory should contain CHAMP RSO (Rapid Science Orbit) SP3 files for drag validation testing.

## Data Source

**GFZ ISDC (Information System and Data Center)**

- FTP: `ftp://isdcftp.gfz.de/champ/`
- DOI: https://doi.org/10.5880/GFZ_ORBIT/RSO/L06_G_v01
- Documentation: https://isdc.gfz.de/champ-isdc/

## CHAMP Mission Overview

| Property | Value |
|----------|-------|
| Launch | July 15, 2000 |
| Re-entry | September 19, 2010 |
| Initial altitude | ~450 km |
| Final altitude | ~310 km |
| Inclination | 87.3 deg |
| SP3 satellite ID | L06 |
| Orbit accuracy | 1-2 cm (SLR validated) |

## Download Instructions

### Option 1: FTP (Anonymous)

```bash
# Connect to GFZ FTP
ftp isdcftp.gfz.de
# Login: anonymous
# Password: your-email@example.com

# Navigate to CHAMP RSO data
cd champ/ORBIT/CHAMP/RSO/

# Example: Download 2005 day 180 data
cd 2005/180/
get *.sp3
```

### Option 2: wget

```bash
# Download specific day (example: 2005, day 180)
wget -r -np -nH --cut-dirs=5 \
    ftp://isdcftp.gfz.de/champ/ORBIT/CHAMP/RSO/2005/180/
```

## Recommended Test Periods

| Period | Year | DOY | Altitude | F10.7 | Notes |
|--------|------|-----|----------|-------|-------|
| Early mission | 2001 | 100-110 | ~440 km | ~150 | Solar max, moderate drag |
| Mid mission | 2005 | 180-190 | ~380 km | ~90 | Solar min, low drag |
| Late mission | 2009 | 200-210 | ~320 km | ~70 | Solar min, higher drag (lower altitude) |

## File Format

CHAMP RSO files use standard SP3c format:
- Satellite ID: `L06` (the GFZ RSO files' header line `+    1   L06`)
- Position: km (converted to m by parser)
- Velocity: dm/s (converted to m/s by parser)
- Epoch interval: typically 30 seconds
- Two files per day: 22:00-12:00 and 10:00-24:00

## Usage in Tests

`CHAMP_2005_182.sp3` (GFZ RSO, 2005-07-01 10:00 to 07-02 00:00) is committed. `load_champ_sp3_file()` in `tests/validation/test_champ_drag_validation.cpp` takes it first, then any other uncompressed `.sp3` file here by sorted name:

```cpp
auto sp3 = load_champ_sp3_file();
auto records = sp3->get_satellite_records("L06");
```

## Solar Activity Data

For accurate atmosphere modeling, you may also need F10.7 and Ap indices:

- NOAA SWPC: `ftp://ftp.swpc.noaa.gov/pub/indices/old_indices/`
- CelesTrak: `https://celestrak.org/SpaceData/`

## References

1. GFZ CHAMP RSO Dataset: https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=43671489-b821-11ec-a8b4-7d2338b1cdf0
2. CHAMP Mission: https://www.gfz-potsdam.de/en/section/geomagnetism/infrastructure/gfz-satellite-mission-champ/
3. SP3 Format: https://files.igs.org/pub/data/format/sp3c.txt
