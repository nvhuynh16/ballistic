"""Check that a built NRL models plugin loads and answers (wheels.yml).

Loads ballistic_nrl.<so|dylib|dll> with ctypes, without the library, and
checks its interface version and one NRLMSIS 2.1 and one HWM14 call: a
plugin that a fresh machine cannot load (a missing runtime library, a wrong
architecture) fails here, before it is attached to a release.

Usage: python scripts/check_nrl_plugin.py path/to/ballistic_nrl.so
"""

import argparse
import ctypes
import sys

ABI_VERSION = 1   # BALLISTIC_NRL_ABI_VERSION, include/ballistic/atmosphere/nrl_plugin_abi.h


class MsisInput(ctypes.Structure):
    _fields_ = [("day", ctypes.c_double), ("utsec", ctypes.c_double),
                ("alt_km", ctypes.c_double), ("lat_deg", ctypes.c_double),
                ("lon_deg", ctypes.c_double), ("f107a", ctypes.c_double),
                ("f107", ctypes.c_double), ("ap", ctypes.c_double * 7)]


class MsisOutput(ctypes.Structure):
    _fields_ = [("tn", ctypes.c_double), ("tex", ctypes.c_double),
                ("dn", ctypes.c_double * 11)]


class HwmInput(ctypes.Structure):
    _fields_ = [("iyd", ctypes.c_int), ("sec", ctypes.c_float), ("alt_km", ctypes.c_float),
                ("glat_deg", ctypes.c_float), ("glon_deg", ctypes.c_float),
                ("stl_h", ctypes.c_float), ("f107a", ctypes.c_float), ("f107", ctypes.c_float),
                ("ap", ctypes.c_float * 2)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plugin", help="path to ballistic_nrl.<so|dylib|dll>")
    args = parser.parse_args()
    lib = ctypes.CDLL(args.plugin)
    lib.ballistic_nrl_models.restype = ctypes.c_char_p
    lib.ballistic_nrl_last_error.restype = ctypes.c_char_p
    version = lib.ballistic_nrl_abi_version()
    if version != ABI_VERSION:
        print(f"interface version {version}, expected {ABI_VERSION}")
        return 1
    ap = (ctypes.c_double * 7)(*([4.0] * 7))
    out = MsisOutput()
    if lib.ballistic_nrl_msis(ctypes.byref(MsisInput(172.5, 29000.0, 400.0, 45.0, -70.0,
                                                     150.0, 150.0, ap)), ctypes.byref(out)):
        print("NRLMSIS 2.1 call failed:", lib.ballistic_nrl_last_error().decode())
        return 1
    wind = (ctypes.c_float * 2)()
    hin = HwmInput(2024172, 29000.0, 250.0, 45.0, -70.0, 3.4, 150.0, 150.0,
                   (ctypes.c_float * 2)(4.0, 4.0))
    if lib.ballistic_nrl_hwm14(ctypes.byref(hin), wind):
        print("HWM14 call failed:", lib.ballistic_nrl_last_error().decode())
        return 1
    rho = out.dn[1]
    # Density at 400 km is of order 1e-12 kg/m^3; winds are tens of m/s.
    if not (1e-14 < rho < 1e-10 and 300.0 < out.tn < 2500.0 and abs(wind[0]) < 500.0
            and abs(wind[1]) < 500.0):
        print(f"implausible output: rho {rho}, T {out.tn}, wind {wind[0]}, {wind[1]}")
        return 1
    print(f"{lib.ballistic_nrl_models().decode()}: rho(400 km) {rho:.3e} kg/m^3, "
          f"T {out.tn:.0f} K, wind(250 km) {wind[0]:.1f} / {wind[1]:.1f} m/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
