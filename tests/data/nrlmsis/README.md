# NRLMSIS 2.1 Reference Output

`msis2.1_test_ref_dp.txt` is the double-precision reference output distributed
with the official NRLMSIS 2.1 Fortran package
(https://map.nrl.navy.mil/map/pub/nrl/NRLMSIS/NRLMSIS2.1/), unmodified. It is
part of the MSIS® software and is governed by the NRL-SOF-014-1 license in
`docs/licenses/nrlmsis2.1_license.txt` (see `THIRD_PARTY_NOTICES.md`).

Each of its 200 rows holds one input set (day of year, UT seconds, altitude,
latitude, longitude, local time, F10.7 81-day mean, daily F10.7, Ap) and the
model output: number densities of He, O, N2, O2, Ar, H, N, anomalous O and NO
in cm⁻³, the total mass density in g/cm³ (all printed to four significant
figures; 0.9999E-37 marks a value the model does not compute at that altitude),
and the temperature in K to 0.01 K.

`tests/unit/test_nrlmsis21_reference.cpp` checks the C++ translation
against every value to half its last printed digit.
