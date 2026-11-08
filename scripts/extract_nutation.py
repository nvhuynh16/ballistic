#!/usr/bin/env python3
"""
Extract nutation coefficients from ERFA nut00a.c and generate a C++ header.

Nutation is the small periodic wobble of the Earth's spin axis caused by the Moon, the
Sun and the planets. The IAU 2000A model describes it as a series of 1365 sine/cosine
terms (678 luni-solar + 687 planetary), needed to convert between the inertial frame
(GCRS) and the Earth-fixed frame (ITRF). ERFA (Essential Routines for Fundamental
Astronomy, a BSD-licensed derivative of the IAU's SOFA library) lists the coefficients as C
arrays in nut00a.c; this script copies them into a C++ header.

Input:  ERFA's nut00a.c (IAU 2000A luni-solar and planetary terms).
Output: include/ballistic/coordinates/nutation_coefficients.hpp.

This is a one-time generator: the input and output paths are set at the top
of main() and must be edited before a re-run.

Usage:
    python scripts/extract_nutation.py

Reference: ERFA Library (derived from SOFA)
License: BSD-3-Clause
Source: https://github.com/liberfa/erfa
"""

import re
import sys

def parse_erfa_file(filepath):
    """Read ERFA's nut00a.c and return its two coefficient tables.

    filepath: path of nut00a.c. Returns (lunisolar_terms, planetary_terms), each a list
    of dicts keyed by the argument multipliers and the coefficients (units 0.1
    microarcseconds, as in ERFA); (None, None) with a printed error if either C array
    (xls or xpl) is not found.
    """

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract luni-solar terms (xls array)
    xls_match = re.search(r'xls\[\] = \{(.*?)\};', content, re.DOTALL)
    if not xls_match:
        print("ERROR: Could not find xls array")
        return None, None

    xls_data = xls_match.group(1)

    # Extract planetary terms (xpl array)
    xpl_match = re.search(r'xpl\[\] = \{(.*?)\};', content, re.DOTALL)
    if not xpl_match:
        print("ERROR: Could not find xpl array")
        return None, None

    xpl_data = xpl_match.group(1)

    # Parse luni-solar terms
    # Format: { nl, nlp, nf, nd, nom, sp, spt, cp, ce, cet, se }
    lunisolar_terms = []
    term_pattern = r'\{\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\}'

    for match in re.finditer(term_pattern, xls_data):
        nl, nlp, nf, nd, nom = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))
        sp, spt, cp, ce, cet, se = float(match.group(6)), float(match.group(7)), float(match.group(8)), float(match.group(9)), float(match.group(10)), float(match.group(11))

        # Rename to the header's field names: S, Sdot, C, Cdot, Cp, Sp
        # S=sp, Sdot=spt, C=ce, Cdot=cet, Cp=cp, Sp=se
        term = {
            'nl': nl, 'nlp': nlp, 'nF': nf, 'nD': nd, 'nOm': nom,
            'S': sp, 'Sdot': spt, 'C': ce, 'Cdot': cet, 'Cp': cp, 'Sp': se
        }
        lunisolar_terms.append(term)

    print(f"Parsed {len(lunisolar_terms)} luni-solar terms")

    # Parse planetary terms
    # Format: { nl, nf, nd, nom, nme, nve, nea, nma, nju, nsa, nur, nne, npa, sp, cp, se, ce }
    planetary_terms = []
    term_pattern_pl = r'\{\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\}'

    for match in re.finditer(term_pattern_pl, xpl_data):
        nl, nf, nd, nom = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        nme, nve, nea, nma = int(match.group(5)), int(match.group(6)), int(match.group(7)), int(match.group(8))
        nju, nsa, nur, nne, npa = int(match.group(9)), int(match.group(10)), int(match.group(11)), int(match.group(12)), int(match.group(13))
        sp, cp, se, ce = float(match.group(14)), float(match.group(15)), float(match.group(16)), float(match.group(17))

        term = {
            'nl': nl, 'nF': nf, 'nD': nd, 'nOm': nom,
            'nMe': nme, 'nVe': nve, 'nE': nea, 'nMa': nma,
            'nJ': nju, 'nSa': nsa, 'nU': nur, 'nN': nne, 'npa': npa,
            'S': sp, 'C': cp, 'Sp': se, 'Cp': ce
        }
        planetary_terms.append(term)

    print(f"Parsed {len(planetary_terms)} planetary terms")

    return lunisolar_terms, planetary_terms


def generate_cpp_header(lunisolar, planetary, output_path):
    """Write the C++ header with both nutation tables to output_path.

    lunisolar, planetary: the lists returned by parse_erfa_file.
    """

    header = '''#pragma once

// =============================================================================
// IAU 2000A Nutation Coefficients - Full 1365-term Series
// =============================================================================
//
// This file contains the complete IAU 2000A nutation series coefficients.
// Total: 678 luni-solar terms + 687 planetary terms = 1365 terms
//
// References:
// -----------
// [1] Mathews, P.M., Herring, T.A., & Buffett, B.A. (2002).
//     "Modeling of nutation and precession: New nutation series for
//     nonrigid Earth and insights into the Earth's interior".
//     Journal of Geophysical Research, 107(B4), 2068.
//     DOI: 10.1029/2001JB000390
//
// [2] McCarthy, D.D. & Petit, G. (2004). "IERS Conventions (2003)".
//     IERS Technical Note No. 32, Chapter 5, Tables 5.3a and 5.3b.
//
// [3] SOFA (Standards of Fundamental Astronomy) Library.
//     Function: iauNut00a() - source of coefficient verification.
//     http://www.iausofa.org/
//
// [4] ERFA (Essential Routines for Fundamental Astronomy) Library.
//     Coefficients extracted from eraNut00a() function.
//     https://github.com/liberfa/erfa
//     License: BSD-3-Clause
//
// Coefficient Format:
// -------------------
// Luni-Solar Terms (678 terms):
//   - Arguments: nl, nlp, nF, nD, nOm (multipliers for Delaunay arguments)
//   - Longitude: S + Sdot*t (sin coefficient), Cp (cos coefficient)
//   - Obliquity: C + Cdot*t (cos coefficient), Sp (sin coefficient)
//   - Units: 0.1 microarcseconds
//
// Planetary Terms (687 terms):
//   - Arguments: nl, nF, nD, nOm + 8 planetary longitudes + pa
//   - Longitude and obliquity coefficients
//   - Units: 0.1 microarcseconds
//
// =============================================================================

#include <array>
#include <cmath>

namespace ballistic {
namespace coords {
namespace nutation {

// Luni-solar nutation term structure
struct LuniSolarTerm {
    int nl, nlp, nF, nD, nOm;  // Delaunay argument multipliers
    double S, Sdot;             // Sin coefficients for dpsi (0.1 uas)
    double C, Cdot;             // Cos coefficients for deps (0.1 uas)
    double Cp;                  // Cos coefficient for dpsi (0.1 uas)
    double Sp;                  // Sin coefficient for deps (0.1 uas)
};

// Planetary nutation term structure
struct PlanetaryTerm {
    int nl, nF, nD, nOm;                    // Lunar arguments
    int nMe, nVe, nE, nMa, nJ, nSa, nU, nN; // Planetary longitude multipliers
    int npa;                                 // General precession multiplier
    double S, C;                             // Longitude coefficients (0.1 uas)
    double Sp, Cp;                           // Obliquity coefficients (0.1 uas)
};

// =============================================================================
// LUNI-SOLAR NUTATION SERIES (678 terms)
// Reference: IERS Conventions 2003, Table 5.3a
// Extracted from ERFA eraNut00a()
// =============================================================================

'''

    # Generate luni-solar array
    header += f'inline constexpr std::array<LuniSolarTerm, {len(lunisolar)}> LUNISOLAR_TERMS = {{{{\n'

    for i, term in enumerate(lunisolar):
        if i % 10 == 0:
            header += f'\n    // Terms {i+1}-{min(i+10, len(lunisolar))}\n'

        header += f'    {{{term["nl"]}, {term["nlp"]}, {term["nF"]}, {term["nD"]}, {term["nOm"]}, '
        header += f'{term["S"]:.1f}, {term["Sdot"]:.1f}, {term["C"]:.1f}, {term["Cdot"]:.1f}, {term["Cp"]:.1f}, {term["Sp"]:.1f}}}'

        if i < len(lunisolar) - 1:
            header += ','
        header += '\n'

    header += '}};\n\n'

    # Add count constant
    header += '// Number of luni-solar terms (all 678 terms included)\n'
    header += f'inline constexpr size_t LUNISOLAR_TERMS_COUNT = {len(lunisolar)};\n\n'

    # Generate planetary array
    header += '// =============================================================================\n'
    header += '// PLANETARY NUTATION SERIES (687 terms)\n'
    header += '// Reference: IERS Conventions 2003, Table 5.3b\n'
    header += '// Extracted from ERFA eraNut00a()\n'
    header += '// =============================================================================\n\n'

    header += f'inline constexpr std::array<PlanetaryTerm, {len(planetary)}> PLANETARY_TERMS = {{{{\n'

    for i, term in enumerate(planetary):
        if i % 10 == 0:
            header += f'\n    // Terms {i+1}-{min(i+10, len(planetary))}\n'

        header += f'    {{{term["nl"]}, {term["nF"]}, {term["nD"]}, {term["nOm"]}, '
        header += f'{term["nMe"]}, {term["nVe"]}, {term["nE"]}, {term["nMa"]}, '
        header += f'{term["nJ"]}, {term["nSa"]}, {term["nU"]}, {term["nN"]}, {term["npa"]}, '
        header += f'{term["S"]:.1f}, {term["C"]:.1f}, {term["Sp"]:.1f}, {term["Cp"]:.1f}}}'

        if i < len(planetary) - 1:
            header += ','
        header += '\n'

    header += '}};\n\n'

    # Add count constant
    header += '// Number of planetary terms (all 687 terms included)\n'
    header += f'inline constexpr size_t PLANETARY_TERMS_COUNT = {len(planetary)};\n\n'

    header += '''} // namespace nutation
} // namespace coords
} // namespace ballistic
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)

    print(f"Generated {output_path}")
    print(f"  - {len(lunisolar)} luni-solar terms")
    print(f"  - {len(planetary)} planetary terms")


def main():
    """Parse nut00a.c and write the header; return 0 on success, 1 on a parse failure."""
    input_path = r'F:\Ballistic\data\erfa_nut00a.c'
    output_path = r'F:\Ballistic\include\ballistic\coordinates\nutation_coefficients.hpp'

    lunisolar, planetary = parse_erfa_file(input_path)

    if lunisolar is None or planetary is None:
        print("Failed to parse ERFA file")
        return 1

    generate_cpp_header(lunisolar, planetary, output_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
