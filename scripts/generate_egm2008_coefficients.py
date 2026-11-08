#!/usr/bin/env python3
"""
Generate the embedded EGM2008 coefficient table (degrees 2-360).

EGM2008 (Earth Gravitational Model 2008) describes the Earth's gravity field as a sum of
spherical harmonics; each term of degree n and order m has two fully normalized
coefficients, C (cosine) and S (sine). The library embeds the terms up to degree 360.

Input:  data/EGM2008_Spherical_Harmonics/EGM2008_to2190_TideFree, the coefficient file
        published by the U.S. National Geospatial-Intelligence Agency (NGA); each row is
        "n m C S [sigmas]" with Fortran "D" exponents.
Output: include/ballistic/gravity/egm2008_coefficients.hpp (paths fixed in main(); no
        command-line options).

This is the one-time generator of the embedded table. Its output was later
moved into src/gravity/egm2008_coefficients.cpp (the header now holds only
the extern declarations), so a re-run needs that move repeated by hand.

Usage:
    python scripts/generate_egm2008_coefficients.py
"""

import sys
import re  # noqa: F401 (unused; the generator code is kept as it ran)
from pathlib import Path

def parse_egm2008_file(filepath: Path, max_degree: int) -> list:
    """Read the NGA EGM2008 coefficient file up to max_degree.

    filepath: the coefficient file; max_degree: highest degree n to keep (the file is
    sorted by degree, so reading stops at the first row above it).
    Returns a list of (n, m, C, S) tuples (ints n, m; floats C, S, fully normalized).
    Rows with fewer than four fields or that do not parse as numbers are skipped.
    """
    coefficients = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Replace Fortran D notation with E
            line = line.replace('D', 'E').replace('d', 'e')

            parts = line.split()
            if len(parts) < 4:
                continue

            try:
                n = int(parts[0])
                m = int(parts[1])
                C = float(parts[2])
                S = float(parts[3])

                if n > max_degree:
                    break  # File is sorted by degree, so we can stop

                coefficients.append((n, m, C, S))
            except (ValueError, IndexError):  # guard-exempt(EXCEPT-PASS): one-time generator; skips non-numeric header lines of the NGA file
                continue

    return coefficients

def generate_header(coefficients: list, max_degree: int) -> str:
    """Return the text of the C++ header that embeds the coefficients.

    coefficients: list of (n, m, C, S) from parse_egm2008_file; max_degree: the degree
    written into the header's MAX_EMBEDDED_DEGREE constant.
    """

    # Count coefficients
    num_coeffs = len(coefficients)

    header = f'''#pragma once

// =============================================================================
// EGM2008 Embedded Spherical Harmonic Coefficients
// =============================================================================
//
// This file contains embedded EGM2008 fully-normalized spherical harmonic
// coefficients up to degree {max_degree}.
//
// Source: NGA EGM2008_to2190_TideFree
// Reference: Pavlis, N.K., et al. (2012). "The development and evaluation of
//            the Earth Gravitational Model 2008 (EGM2008)."
//            Journal of Geophysical Research, 117, B04406.
//
// Total coefficients: {num_coeffs} (degrees 2-{max_degree})
// Generated automatically - do not edit manually.
//
// =============================================================================

#include <array>
#include <cstddef>

namespace ballistic {{
namespace gravity {{
namespace egm2008_data {{

// Maximum degree of embedded coefficients
constexpr int MAX_EMBEDDED_DEGREE = {max_degree};

// Number of coefficient pairs (n, m, Cnm, Snm)
constexpr std::size_t NUM_COEFFICIENTS = {num_coeffs};

// Coefficient structure: {{degree, order, Cnm, Snm}}
struct Coefficient {{
    int n;
    int m;
    double C;
    double S;
}};

// Embedded EGM2008 coefficients (fully normalized, tide-free)
// Format: {{degree, order, Cnm, Snm}}
inline constexpr std::array<Coefficient, NUM_COEFFICIENTS> COEFFICIENTS = {{{{
'''

    # Add coefficients
    for i, (n, m, C, S) in enumerate(coefficients):
        # Format with full precision
        C_str = f"{C: .15e}".replace('e', 'e')
        S_str = f"{S: .15e}".replace('e', 'e')

        comma = "," if i < num_coeffs - 1 else ""
        header += f"    {{{n:3d}, {m:3d}, {C_str}, {S_str}}}{comma}\n"

    header += '''}};

} // namespace egm2008_data
} // namespace gravity
} // namespace ballistic
'''

    return header

def main():
    """Read the NGA file, generate the header and write it; exit 1 if the input is missing."""
    # Configuration
    max_degree = 360  # Embed coefficients up to degree 360 (geodetic standard)

    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_file = project_root / "data" / "EGM2008_Spherical_Harmonics" / "EGM2008_to2190_TideFree"
    output_file = project_root / "include" / "ballistic" / "gravity" / "egm2008_coefficients.hpp"

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    print(f"Parsing EGM2008 coefficients up to degree {max_degree}...")
    coefficients = parse_egm2008_file(input_file, max_degree)
    print(f"Loaded {len(coefficients)} coefficients")

    print(f"Generating header file...")  # noqa: F541
    header_content = generate_header(coefficients, max_degree)

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header_content)

    print(f"Written to: {output_file}")
    print(f"File size: {len(header_content):,} bytes")

if __name__ == "__main__":
    main()
