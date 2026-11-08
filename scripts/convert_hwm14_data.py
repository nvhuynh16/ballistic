#!/usr/bin/env python3
"""
Convert HWM14 binary data files to C++ embedded coefficient header.

HWM14 (Horizontal Wind Model 2014, U.S. Naval Research Laboratory) gives the wind in the
upper atmosphere, which the library uses in the drag force. It has two parts: the
quiet-time wind model (QWM) and the disturbance wind model (DWM07), which adds the winds
driven by geomagnetic storms and works in quasi-dipole coordinates (latitude/longitude
aligned with the Earth's magnetic field).

Reads the three HWM14 data files (written by Fortran programs: "stream" files are plain
bytes; "unformatted" files wrap each record in 4-byte length markers):
  - hwm123114.bin  (quiet-time wind coefficients, stream binary)
  - dwm07b104i.dat (disturbance wind coefficients, unformatted binary)
  - gd2qd.dat      (geodetic-to-quasi-dipole conversion, unformatted binary)

Outputs:
  include/ballistic/atmosphere/hwm14_coefficients.hpp (the library keeps it
  at include/ballistic/atmosphere/internal/hwm14_coefficients.hpp; move the
  regenerated file there)

Usage:
    python scripts/convert_hwm14_data.py <path-to-hwm14-data-dir>

<path-to-hwm14-data-dir> is the directory that holds the three files; the script exits
with status 1 if it is missing or a file is absent.

The HWM14 data files can be obtained from:
    https://github.com/jacobwilliams/HWM14
    https://map.nrl.navy.mil/map/pub/nrl/HWM/HWM14/

License: HWM14 was written by NRL employees (Drob and Emmert) as a work of the U.S.
Government, not subject to copyright in the United States (17 U.S.C. 105); NRL attached
no license. Cite Drob et al. (2015), Earth and Space Science 2, 301-319. See
THIRD_PARTY_NOTICES.md.
"""

import sys
import struct
from pathlib import Path


def read_fortran_stream_binary(filepath):
    """Read the quiet-time model file hwm123114.bin (Fortran access='stream' binary).

    The file is a sequence of little-endian 4-byte ints and 8-byte doubles: header
    sizes, the vertical spline nodes (vnode), per level the term order, term count and
    coefficients (mparm), and two 5-element arrays e1, e2. Returns a dict with those
    entries (lists of Python numbers).
    """
    data = open(filepath, 'rb').read()
    offset = 0

    def read_int4():
        nonlocal offset
        val = struct.unpack('<i', data[offset:offset+4])[0]
        offset += 4
        return val

    def read_real8():
        nonlocal offset
        val = struct.unpack('<d', data[offset:offset+8])[0]
        offset += 8
        return val

    def read_int4_array(n):
        nonlocal offset
        vals = struct.unpack(f'<{n}i', data[offset:offset+4*n])
        offset += 4 * n
        return list(vals)

    def read_real8_array(n):
        nonlocal offset
        vals = struct.unpack(f'<{n}d', data[offset:offset+8*n])
        offset += 8 * n
        return list(vals)

    # Read header: nbf, maxs, maxm, maxl, maxn, ncomp
    nbf = read_int4()
    maxs = read_int4()
    maxm = read_int4()
    maxl = read_int4()
    maxn = read_int4()
    ncomp = read_int4()
    print(f"QWM header: nbf={nbf}, maxs={maxs}, maxm={maxm}, maxl={maxl}, maxn={maxn}, ncomp={ncomp}")

    # Read nlev, p
    nlev = read_int4()
    p = read_int4()
    nnode = nlev + p
    print(f"QWM: nlev={nlev}, p={p}, nnode={nnode}")

    # Read vnode array (nnode+1 elements)
    vnode = read_real8_array(nnode + 1)
    print(f"QWM: vnode has {len(vnode)} elements, range [{vnode[0]}, {vnode[-1]}]")

    # Read level data
    nlevels = nlev - p + 1 - 2 + 1  # number of levels: 0 to nlev-p+1-2
    print(f"QWM: reading {nlevels} levels")

    order_data = []
    nb_data = []
    mparm_data = []

    for i in range(nlevels):
        order = read_int4_array(ncomp)
        nb = read_int4()
        mparm = read_real8_array(nbf)
        order_data.append(order)
        nb_data.append(nb)
        mparm_data.append(mparm)

    # Read e1, e2 (5 elements each)
    e1 = read_real8_array(5)
    e2 = read_real8_array(5)

    print(f"QWM: e1={e1}")
    print(f"QWM: e2={e2}")
    print(f"QWM: bytes read = {offset}, file size = {len(data)}")

    return {
        'nbf': nbf, 'maxs': maxs, 'maxm': maxm, 'maxl': maxl,
        'maxn': maxn, 'ncomp': ncomp, 'nlev': nlev, 'p': p,
        'vnode': vnode, 'order': order_data, 'nb': nb_data,
        'mparm': mparm_data, 'e1': e1, 'e2': e2
    }


def read_fortran_unformatted(filepath):
    """Read a Fortran unformatted (sequential) binary file into its records.

    Each record is stored as <int32 length><data><int32 length>. Returns the list of
    record payloads (bytes). Raises AssertionError if a record's two length markers
    differ.
    """
    data = open(filepath, 'rb').read()
    offset = 0
    records = []

    while offset < len(data):
        # Read 4-byte record length prefix
        if offset + 4 > len(data):
            break
        rec_len = struct.unpack('<i', data[offset:offset+4])[0]
        offset += 4
        # Read record data
        rec_data = data[offset:offset+rec_len]
        offset += rec_len
        # Read 4-byte record length suffix (should match)
        suffix = struct.unpack('<i', data[offset:offset+4])[0]
        offset += 4
        assert rec_len == suffix, f"Record length mismatch: {rec_len} vs {suffix}"
        records.append(rec_data)

    return records


def parse_dwm_data(filepath):
    """Read dwm07b104i.dat (DWM07 disturbance wind coefficients).

    Returns a dict with nterm, mmax, nmax (ints), termarr (3*nterm ints), coeff (nterm
    floats) and twidth (float).
    """
    records = read_fortran_unformatted(filepath)

    # Record 0: nterm, mmax, nmax (3 int4)
    nterm, mmax, nmax = struct.unpack('<3i', records[0])
    print(f"DWM: nterm={nterm}, mmax={mmax}, nmax={nmax}")

    # Record 1: termarr (3 x nterm int4)
    termarr = list(struct.unpack(f'<{3*nterm}i', records[1]))
    print(f"DWM: termarr has {len(termarr)} elements")

    # Record 2: coeff (nterm float32)
    coeff = list(struct.unpack(f'<{nterm}f', records[2]))
    print(f"DWM: coeff has {len(coeff)} elements")

    # Record 3: twidth (float32)
    twidth = struct.unpack('<f', records[3])[0]
    print(f"DWM: twidth={twidth}")

    return {
        'nterm': nterm, 'mmax': mmax, 'nmax': nmax,
        'termarr': termarr, 'coeff': coeff, 'twidth': twidth
    }


def parse_gd2qd_data(filepath):
    """Read gd2qd.dat (geodetic to quasi-dipole conversion coefficients).

    Returns a dict with nmax, mmax, nterm (ints), epoch and alt (floats, as stored in
    the file), and xcoeff, ycoeff, zcoeff (nterm doubles each).
    """
    records = read_fortran_unformatted(filepath)

    # Record 0: nmax, mmax, nterm, epoch, alt
    rec = records[0]
    nmax, mmax, nterm = struct.unpack('<3i', rec[:12])
    epoch, alt = struct.unpack('<2f', rec[12:20])
    print(f"GD2QD: nmax={nmax}, mmax={mmax}, nterm={nterm}, epoch={epoch}, alt={alt}")

    # Record 1: coeff (nterm x 3 float64)
    # Fortran coeff(0:nterm-1, 0:2) is column-major: first index varies fastest
    # File layout: all x-coefficients (nterm), then y (nterm), then z (nterm)
    coeff_flat = list(struct.unpack(f'<{3*nterm}d', records[1]))
    xcoeff = coeff_flat[0:nterm]
    ycoeff = coeff_flat[nterm:2*nterm]
    zcoeff = coeff_flat[2*nterm:3*nterm]

    print(f"GD2QD: coefficients loaded, {nterm} terms")

    return {
        'nmax': nmax, 'mmax': mmax, 'nterm': nterm,
        'epoch': epoch, 'alt': alt,
        'xcoeff': xcoeff, 'ycoeff': ycoeff, 'zcoeff': zcoeff
    }


def format_double_array(name, values, per_line=4):
    """Return C++ source for a static constexpr std::array<double> `name`,
    per_line values a line."""
    lines = [f"    static constexpr std::array<double, {len(values)}> {name} = {{{{"]
    for i in range(0, len(values), per_line):
        chunk = values[i:i+per_line]
        formatted = ", ".join(f"{v:.17e}" for v in chunk)
        if i + per_line < len(values):
            formatted += ","
        lines.append(f"        {formatted}")
    lines.append("    }};")
    return "\n".join(lines)


def format_float_array(name, values, per_line=6):
    """Return C++ source for a static constexpr std::array<float> `name`, per_line values a line."""
    lines = [f"    static constexpr std::array<float, {len(values)}> {name} = {{{{"]
    for i in range(0, len(values), per_line):
        chunk = values[i:i+per_line]
        formatted = ", ".join(f"{v:.9e}f" for v in chunk)
        if i + per_line < len(values):
            formatted += ","
        lines.append(f"        {formatted}")
    lines.append("    }};")
    return "\n".join(lines)


def format_int_array(name, values, per_line=12):
    """Return C++ source for a static constexpr std::array<int> `name`, per_line values a line."""
    lines = [f"    static constexpr std::array<int, {len(values)}> {name} = {{{{"]
    for i in range(0, len(values), per_line):
        chunk = values[i:i+per_line]
        formatted = ", ".join(str(v) for v in chunk)
        if i + per_line < len(values):
            formatted += ","
        lines.append(f"        {formatted}")
    lines.append("    }};")
    return "\n".join(lines)


def generate_header(qwm, dwm, gd2qd):
    """Return the text of the C++ header from the three parsed data sets (dicts)."""
    header = """#pragma once

// =============================================================================
// HWM14 Embedded Coefficient Data
// =============================================================================
//
// Auto-generated by scripts/convert_hwm14_data.py
// Source data files:
//   - hwm123114.bin  (quiet-time wind model)
//   - dwm07b104i.dat (disturbance wind model, DWM07)
//   - gd2qd.dat      (geodetic-to-quasi-dipole conversion)
//
// Original data: Douglas Drob, John Emmert (NRL)
// Reference: Drob et al. (2015), Earth and Space Science, 2, 301-319.
//
// DO NOT EDIT - regenerate with: python scripts/convert_hwm14_data.py
//
// =============================================================================

#include <array>
#include <cstdint>

namespace ballistic {
namespace atmosphere {
namespace hwm14 {

"""

    # QWM data
    nlevels = qwm['nlev'] - qwm['p'] + 1 - 2 + 1

    # Flatten order array (nlevels x ncomp)
    order_flat = []
    for level_order in qwm['order']:
        order_flat.extend(level_order)

    # Flatten nb array
    nb_flat = qwm['nb']

    # Flatten mparm array (nlevels x nbf)
    mparm_flat = []
    for level_mparm in qwm['mparm']:
        mparm_flat.extend(level_mparm)

    header += f"""// Quiet-time Wind Model (QWM) data
struct QWMData {{
    static constexpr int nbf = {qwm['nbf']};
    static constexpr int maxs = {qwm['maxs']};
    static constexpr int maxm = {qwm['maxm']};
    static constexpr int maxl = {qwm['maxl']};
    static constexpr int maxn = {qwm['maxn']};
    static constexpr int ncomp = {qwm['ncomp']};
    static constexpr int nlev = {qwm['nlev']};
    static constexpr int p = {qwm['p']};
    static constexpr int nlevels = {nlevels};

"""

    header += format_double_array("vnode", qwm['vnode']) + "\n\n"
    header += format_int_array("order", order_flat) + "\n\n"
    header += format_int_array("nb", nb_flat) + "\n\n"
    header += format_double_array("mparm", mparm_flat) + "\n\n"
    header += format_double_array("e1", qwm['e1']) + "\n\n"
    header += format_double_array("e2", qwm['e2']) + "\n"
    header += "};\n\n"
    header += "inline constexpr QWMData qwm_data{};\n\n"

    # DWM data
    header += f"""// Disturbance Wind Model (DWM07) data
struct DWMData {{
    static constexpr int nterm = {dwm['nterm']};
    static constexpr int mmax = {dwm['mmax']};
    static constexpr int nmax = {dwm['nmax']};
    static constexpr float twidth = {dwm['twidth']:.9e}f;

"""

    header += format_int_array("termarr", dwm['termarr']) + "\n\n"
    header += format_float_array("coeff", dwm['coeff']) + "\n"
    header += "};\n\n"
    header += "inline constexpr DWMData dwm_data{};\n\n"

    # GD2QD data
    header += f"""// Geodetic to Quasi-Dipole Coordinate Conversion data
struct GD2QDData {{
    static constexpr int nmax = {gd2qd['nmax']};
    static constexpr int mmax = {gd2qd['mmax']};
    static constexpr int nterm = {gd2qd['nterm']};
    static constexpr float epoch = {gd2qd['epoch']:.1f}f;
    static constexpr float alt = {gd2qd['alt']:.1f}f;

"""

    header += format_double_array("xcoeff", gd2qd['xcoeff']) + "\n\n"
    header += format_double_array("ycoeff", gd2qd['ycoeff']) + "\n\n"
    header += format_double_array("zcoeff", gd2qd['zcoeff']) + "\n"
    header += "};\n\n"
    header += "inline constexpr GD2QDData gd2qd_data{};\n\n"

    header += """} // namespace hwm14
} // namespace atmosphere
} // namespace ballistic
"""

    return header


def main():
    """Read the three files from the directory given on the command line and write the header."""
    if len(sys.argv) < 2:
        print("Usage: python convert_hwm14_data.py <path-to-hwm14-data-dir>")
        print()
        print("The data directory should contain:")
        print("  - hwm123114.bin")
        print("  - dwm07b104i.dat")
        print("  - gd2qd.dat")
        print()
        print("These files can be obtained from:")
        print("  https://github.com/jacobwilliams/HWM14")
        sys.exit(1)

    data_dir = Path(sys.argv[1])

    qwm_file = data_dir / "hwm123114.bin"
    dwm_file = data_dir / "dwm07b104i.dat"
    gd2qd_file = data_dir / "gd2qd.dat"

    for f in [qwm_file, dwm_file, gd2qd_file]:
        if not f.exists():
            print(f"Error: {f} not found")
            sys.exit(1)

    print("Reading QWM data...")
    qwm = read_fortran_stream_binary(qwm_file)

    print("\nReading DWM data...")
    dwm = parse_dwm_data(dwm_file)

    print("\nReading GD2QD data...")
    gd2qd = parse_gd2qd_data(gd2qd_file)

    print("\nGenerating C++ header...")
    header = generate_header(qwm, dwm, gd2qd)

    output_path = Path(__file__).parent.parent / "include" / "ballistic" / "atmosphere" / "hwm14_coefficients.hpp"
    output_path.write_text(header)
    print(f"\nWritten to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
