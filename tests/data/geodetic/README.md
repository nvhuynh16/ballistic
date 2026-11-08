# Geodetic Reference Data

This directory contains reference coordinates for geodetic function validation.

## Data Sources

### ITRF Station Coordinates
- URL: https://itrf.ign.fr/en/solutions
- High-precision geodetic control points
- Updated with each ITRF realization (ITRF2014, ITRF2020)

### NGS Benchmark Points
- URL: https://www.ngs.noaa.gov/
- US National Geodetic Survey control points

## Reference Points Used

The tests include embedded reference points computed using GeographicLib:

| Location | Latitude | Longitude | Altitude |
|----------|----------|-----------|----------|
| Equator-PM | 0.0 | 0.0 | 0 m |
| North Pole | 90.0 | 0.0 | 0 m |
| Mount Everest | 27.9881 | 86.9250 | 8848.86 m |
| Mariana Trench | 11.3493 | 142.1996 | -10994 m |
| ISS Altitude | 51.6 | 0.0 | 408,000 m |
| GPS Altitude | 0.0 | 0.0 | 20,200,000 m |
| GEO Altitude | 0.0 | 0.0 | 35,786,000 m |

## External Validation

For additional validation, compare against:

### GeographicLib
- URL: https://geographiclib.sourceforge.io/
- Gold standard for geodetic calculations
- C++ library available in `external/geographiclib/`

### PROJ
- URL: https://proj.org/
- Cartographic projections and coordinate transformations

## Expected Accuracy

- Round-trip conversion: < 1 micrometer
- Agreement with GeographicLib: < 1 mm
- Agreement with survey data: < 1 cm
