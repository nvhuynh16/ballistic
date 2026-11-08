#!/usr/bin/env python3
"""
Convert the NRLMSIS 2.1 binary parameter file to a C++ header.

NRLMSIS 2.1 is the U.S. Naval Research Laboratory's empirical model of the temperature,
composition and density of the atmosphere; the library uses it for drag. Its fitted
coefficients ship as a binary file. That file holds a 2D array of little-endian doubles,
[MAXNBF x NVERTPARM], stored column by column (Fortran order), where MAXNBF = 512 is the
number of horizontal/time basis functions and NVERTPARM = 131 is the number of vertical
parameters summed over temperature, surface pressure and the nine species (computed below).

License: NRLMSIS 2.1 and its parameters are distributed by NRL under the NRL-SOF-014-1
research license (research, academic and non-profit use only; no commercial use). The
generated header is a derivative of that file and carries the same terms; see
THIRD_PARTY_NOTICES.md and docs/licenses/nrlmsis2.1_license.txt.

Input:  external/msis21.parm (from the official NRLMSIS 2.1 package; see
        docs/third_party_provenance.md).
Output: include/ballistic/atmosphere/nrlmsis21_parm.hpp. This is a one-time
        generator: the header now lives at
        include/ballistic/atmosphere/internal/nrlmsis21_parm.hpp. Paths are fixed in
        main(); there are no command-line options. Importing the module prints the
        expected array size.

Usage:
    python scripts/convert_msis21_parm.py
"""

import struct
import numpy as np
import sys
from pathlib import Path

# MSIS 2.1 constants (from msis_constants.F90)
MAXNBF = 512  # Number of horizontal/temporal basis functions
ND = 27       # Number of temperature profile nodes
P_ORDER = 4   # Spline order
NL = ND - P_ORDER  # = 23, last temperature profile level index
NLS = 9       # Last parameter index for each species
ND_O1 = 13    # O1 spline nodes
NSPL_O1 = ND_O1 - 5  # = 8
ND_NO = 13    # NO spline nodes
NSPL_NO = ND_NO - 5  # = 8

# Calculate vertical parameter counts for each species
# TN: 0 to NL (24 levels)
# N2, O2, HE, H1, AR, N1, OA: 0 to NLS (10 levels each)
# O1: 0 to NLS+NSPL_O1 (18 levels)
# NO: 0 to NLS+NSPL_NO (18 levels)
# +1 for surface pressure

TN_LEVELS = NL + 1  # 24
N2_LEVELS = NLS + 1  # 10
O2_LEVELS = NLS + 1  # 10
O1_LEVELS = NLS + NSPL_O1 + 1  # 18
HE_LEVELS = NLS + 1  # 10
H1_LEVELS = NLS + 1  # 10
AR_LEVELS = NLS + 1  # 10
N1_LEVELS = NLS + 1  # 10
OA_LEVELS = NLS + 1  # 10
NO_LEVELS = NLS + NSPL_NO + 1  # 18

NVERTPARM = (TN_LEVELS + 1 +  # +1 for surface pressure
             N2_LEVELS + O2_LEVELS + O1_LEVELS +
             HE_LEVELS + H1_LEVELS + AR_LEVELS +
             N1_LEVELS + OA_LEVELS + NO_LEVELS)

print(f"NVERTPARM = {NVERTPARM}")
print(f"Expected file size: {MAXNBF * NVERTPARM * 8} bytes")


def read_msis_parm(filename):
    """Read the NRLMSIS binary parameter file.

    filename: path of msis21.parm. Returns (arr, nvertparm): arr is a numpy array of
    shape (MAXNBF, nvertparm), row = basis function, column = vertical parameter;
    nvertparm is inferred from the file size (number of doubles // MAXNBF).
    """
    with open(filename, 'rb') as f:
        data = f.read()

    print(f"File size: {len(data)} bytes")

    # File is stored as little-endian doubles, column-major (Fortran order)
    # parmin(0:MAXNBF-1, 0:NVERTPARM-1)
    # In Fortran, this is column-major, so data is organized as:
    # parmin(0,0), parmin(1,0), ..., parmin(511,0), parmin(0,1), ...

    n_doubles = len(data) // 8
    print(f"Number of doubles: {n_doubles}")

    actual_nvertparm = n_doubles // MAXNBF
    print(f"Actual NVERTPARM: {actual_nvertparm}")

    # Read all doubles
    fmt = '<' + 'd' * n_doubles  # little-endian doubles
    values = struct.unpack(fmt, data[:n_doubles * 8])

    # Reshape to 2D array (Fortran column-major)
    arr = np.array(values).reshape((actual_nvertparm, MAXNBF)).T
    print(f"Array shape: {arr.shape}")

    return arr, actual_nvertparm


def write_cpp_array(f, name, arr, start_col, num_cols):
    """Write columns start_col .. start_col+num_cols-1 of arr as a C++ array.

    f: open text file; name: species prefix (the array is called <name>_BETA);
    arr: (MAXNBF, NVERTPARM) parameter array. One C++ row per basis function.
    """
    f.write(f"\n// {name} parameters: {num_cols} vertical levels x {arr.shape[0]} basis functions\n")
    f.write(f"constexpr std::array<std::array<double, {num_cols}>, {arr.shape[0]}> {name}_BETA = {{{{\n")

    for i in range(arr.shape[0]):
        row = arr[i, start_col:start_col + num_cols]
        # Rows that are all zeros are written as literal 0.0 values
        if np.allclose(row, 0):
            f.write(f"  {{{{{', '.join(['0.0'] * num_cols)}}}}},\n")
        else:
            values = [f"{v:.15e}" if v != 0 else "0.0" for v in row]
            f.write(f"  {{{{{', '.join(values)}}}}},\n")

    f.write("}};\n")


def write_cpp_header(arr, nvertparm, output_path):
    """Write the C++ header with the parameters, one array per species.

    arr: (MAXNBF, NVERTPARM) parameter array; nvertparm: its column count (unused,
    the column offsets come from the *_LEVELS constants); output_path: header to write.
    """

    with open(output_path, 'w', encoding="utf-8") as f:
        f.write("""#pragma once
// =============================================================================
// NRLMSIS 2.1 Parameter Data (Auto-generated from msis21.parm)
// =============================================================================
// This file contains the fitted model coefficients for NRLMSIS 2.1.
// Generated by convert_msis21_parm.py
// =============================================================================

#include <array>

namespace ballistic {
namespace atmosphere {
namespace msis21 {
namespace parm {

// Array dimensions
constexpr int MAXNBF = 512;  // Number of basis functions
""")

        # Write array dimensions
        f.write(f"constexpr int TN_LEVELS = {TN_LEVELS};\n")
        f.write(f"constexpr int N2_LEVELS = {N2_LEVELS};\n")
        f.write(f"constexpr int O2_LEVELS = {O2_LEVELS};\n")
        f.write(f"constexpr int O1_LEVELS = {O1_LEVELS};\n")
        f.write(f"constexpr int HE_LEVELS = {HE_LEVELS};\n")
        f.write(f"constexpr int H1_LEVELS = {H1_LEVELS};\n")
        f.write(f"constexpr int AR_LEVELS = {AR_LEVELS};\n")
        f.write(f"constexpr int N1_LEVELS = {N1_LEVELS};\n")
        f.write(f"constexpr int OA_LEVELS = {OA_LEVELS};\n")
        f.write(f"constexpr int NO_LEVELS = {NO_LEVELS};\n")

        # Calculate offsets in the parameter array
        col = 0

        # TN (temperature)
        write_cpp_array(f, "TN", arr, col, TN_LEVELS)
        col += TN_LEVELS

        # Surface pressure (PR) - 1 level
        f.write("\n// Surface pressure parameter: 1 level\n")
        f.write(f"constexpr std::array<double, {arr.shape[0]}> PR_BETA = {{{{\n")
        pr_col = arr[:, col]
        for i in range(0, arr.shape[0], 8):
            chunk = pr_col[i:min(i+8, arr.shape[0])]
            values = [f"{v:.15e}" if v != 0 else "0.0" for v in chunk]
            f.write(f"  {', '.join(values)},\n")
        f.write("}};\n")
        col += 1

        # N2
        write_cpp_array(f, "N2", arr, col, N2_LEVELS)
        col += N2_LEVELS

        # O2
        write_cpp_array(f, "O2", arr, col, O2_LEVELS)
        col += O2_LEVELS

        # O1
        write_cpp_array(f, "O1", arr, col, O1_LEVELS)
        col += O1_LEVELS

        # HE
        write_cpp_array(f, "HE", arr, col, HE_LEVELS)
        col += HE_LEVELS

        # H1
        write_cpp_array(f, "H1", arr, col, H1_LEVELS)
        col += H1_LEVELS

        # AR
        write_cpp_array(f, "AR", arr, col, AR_LEVELS)
        col += AR_LEVELS

        # N1
        write_cpp_array(f, "N1", arr, col, N1_LEVELS)
        col += N1_LEVELS

        # OA (anomalous O)
        write_cpp_array(f, "OA", arr, col, OA_LEVELS)
        col += OA_LEVELS

        # NO
        write_cpp_array(f, "NO", arr, col, NO_LEVELS)
        col += NO_LEVELS

        f.write("""
}  // namespace parm
}  // namespace msis21
}  // namespace atmosphere
}  // namespace ballistic
""")

    print(f"Written {output_path}")
    print(f"Total columns used: {col}")


def main():
    """Read external/msis21.parm and write the header; exit 1 if the input is missing."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    parm_file = repo_root / "external" / "msis21.parm"
    output_file = repo_root / "include" / "ballistic" / "atmosphere" / "nrlmsis21_parm.hpp"

    if not parm_file.exists():
        print(f"Error: Parameter file not found: {parm_file}")
        sys.exit(1)

    print(f"Reading {parm_file}...")
    arr, nvertparm = read_msis_parm(parm_file)

    print(f"\nWriting {output_file}...")
    write_cpp_header(arr, nvertparm, output_file)

    # Print some statistics
    nonzero = np.count_nonzero(arr)
    total = arr.size
    print("\nStatistics:")
    print(f"  Total elements: {total}")
    print(f"  Non-zero elements: {nonzero} ({100*nonzero/total:.1f}%)")


if __name__ == "__main__":
    main()
