#!/usr/bin/env python3
"""
Extract DE440 Ephemeris Subset for Embedding

DE440 is JPL's planetary and lunar ephemeris: the positions of the Sun, Moon and
planets over time, which the library needs for third-body gravity and solar radiation
pressure. JPL distributes it as an SPK file (the SPICE toolkit's ephemeris format, built
on NAIF's Double precision Array File, DAF, container). Positions are stored as
Chebyshev polynomial coefficients, one record per fixed time interval.

This script reads such a file and extracts the coefficients for Sun, Moon, EMB
(Earth-Moon Barycenter), and Jupiter for the years 1950-2100: the records from the one
containing 1950-01-01 to the one after the record containing 2100-01-01. Times are
TDB (Barycentric Dynamical Time) seconds past J2000 (2000-01-01 12:00 TDB), which
SPICE calls ET.

Input: the SPK file (first argument). Output (second argument, default
include/ballistic/ephemeris/de440_coefficients.hpp relative to the current directory):
C++ header file with embedded coefficients. This is a one-time
generator: the library keeps the data in src/ephemeris/de440_coefficients.cpp
(the header holds only the extern declarations), so a re-run needs that move
repeated by hand.

Usage:
    python scripts/extract_de440_subset.py INPUT.bsp [OUTPUT.hpp]
    e.g. python scripts/extract_de440_subset.py data/de440.bsp \
             include/ballistic/ephemeris/de440_coefficients.hpp

References:
    [1] NAIF SPK Required Reading: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html
    [2] NAIF DAF Required Reading: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/daf.html
"""

import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
import datetime

# SPICE body codes
BODY_SSB = 0         # Solar System Barycenter
BODY_EMB = 3         # Earth-Moon Barycenter
BODY_JUPITER = 5     # Jupiter Barycenter
BODY_SUN = 10        # Sun
BODY_MOON = 301      # Moon

# Time constants
# J2000 epoch: 2000-01-01 12:00:00 TDB = JD 2451545.0
J2000_JD = 2451545.0
SECONDS_PER_DAY = 86400.0

# Date range for extraction, as Julian Dates
START_JD = 2433282.5  # 1950-01-01 00:00:00
END_JD = 2488069.5    # 2100-01-01 00:00:00

# Convert to ET (seconds past J2000)
START_ET = (START_JD - J2000_JD) * SECONDS_PER_DAY  # -1577880000.0
END_ET = (END_JD - J2000_JD) * SECONDS_PER_DAY      # 3155716800.0


@dataclass
class SPKSegment:
    """One SPK segment: the summary read from the file, the Type 2 (Chebyshev) layout
    read from its trailer, and its coefficients (a flat list of doubles, record after
    record; each record is [midpoint, half-interval, x coeffs, y coeffs, z coeffs])."""
    start_et: float      # Start ephemeris time (TDB seconds past J2000)
    end_et: float        # End ephemeris time
    target_body: int     # Target body ID
    center_body: int     # Center body ID
    ref_frame: int       # Reference frame ID (1 = J2000)
    data_type: int       # Data type (2 = Chebyshev)
    start_addr: int      # Start address in file (1-based, in doubles)
    end_addr: int        # End address in file

    # Type 2 metadata
    init_et: float = 0.0       # Initial epoch of first record
    interval_sec: float = 0.0  # Time interval per record (seconds)
    record_size: int = 0       # Size of each record (doubles)
    n_records: int = 0         # Number of records
    n_coeffs: int = 0          # Number of coefficients per component

    # Coefficient data
    coefficients: List[float] = None

    def __post_init__(self):
        if self.coefficients is None:
            self.coefficients = []


def read_daf_file_record(f) -> Tuple[int, int, int, int]:
    """Read the DAF file record (the first 1024 bytes) of an open binary file.

    Returns:
        (nd, ni, forward_ptr, backward_ptr): the number of doubles and of ints in each
        segment summary, and the record numbers of the first and last summary records.
    Raises ValueError if the file does not start with "DAF/SPK".
    """
    f.seek(0)
    data = f.read(1024)

    # Check signature
    if not data[:7].decode('ascii', errors='ignore').startswith('DAF/SPK'):
        raise ValueError("Not a valid DAF/SPK file")

    # ND, NI at offset 8 (integers)
    nd, ni = struct.unpack('<ii', data[8:16])

    # Forward/backward pointers at offset 76
    forward_ptr, backward_ptr = struct.unpack('<ii', data[76:84])

    return nd, ni, forward_ptr, backward_ptr


def read_segment_summaries(f, forward_ptr: int, nd: int, ni: int) -> List[SPKSegment]:
    """Read the segment summaries from the first summary record of the DAF file.

    Only the record at forward_ptr is read (enough for files whose summaries fit in
    one record). Returns a list of SPKSegment without coefficients.
    """
    segments = []

    # Summary records start at (forward_ptr - 1) * 1024 bytes
    offset = (forward_ptr - 1) * 1024
    f.seek(offset)

    data = f.read(1024)

    # First 16 bytes: next/prev summary record numbers (two doubles)
    next_ptr, prev_ptr = struct.unpack('<dd', data[:16])

    # Number of summaries (third double)
    n_summaries = int(struct.unpack('<d', data[16:24])[0])

    # Each summary is (nd * 8 + ni * 4) bytes = 2*8 + 6*4 = 40 bytes for SPK
    summary_size = nd * 8 + ni * 4
    ptr = 24

    for i in range(n_summaries):
        # Read doubles (start_et, end_et)
        start_et, end_et = struct.unpack('<dd', data[ptr:ptr+16])

        # Read integers (target, center, frame, type, start_addr, end_addr)
        target, center, frame, dtype, start_addr, end_addr = struct.unpack(
            '<iiiiii', data[ptr+16:ptr+40])

        seg = SPKSegment(
            start_et=start_et,
            end_et=end_et,
            target_body=target,
            center_body=center,
            ref_frame=frame,
            data_type=dtype,
            start_addr=start_addr,
            end_addr=end_addr
        )
        segments.append(seg)
        ptr += summary_size

    return segments


def read_segment_data(f, seg: SPKSegment) -> None:
    """Fill seg's Type 2 layout and coefficients from the file; other types are left untouched."""
    if seg.data_type != 2:
        return

    # Read metadata from end of segment
    # Last 4 doubles: init_et, interval_sec, record_size, n_records
    meta_offset = (seg.end_addr - 4) * 8
    f.seek(meta_offset)
    meta = struct.unpack('<4d', f.read(32))

    seg.init_et = meta[0]
    seg.interval_sec = meta[1]
    seg.record_size = int(meta[2])
    seg.n_records = int(meta[3])
    seg.n_coeffs = (seg.record_size - 2) // 3  # Each record: mid, rad, then 3*n_coeffs

    # Read all coefficient data
    data_offset = (seg.start_addr - 1) * 8
    data_size = seg.n_records * seg.record_size * 8

    f.seek(data_offset)
    raw_data = f.read(data_size)

    # Unpack as doubles
    n_doubles = seg.n_records * seg.record_size
    seg.coefficients = list(struct.unpack(f'<{n_doubles}d', raw_data))


def filter_records_by_time(seg: SPKSegment, start_et: float, end_et: float) -> Tuple[int, int, List[float]]:
    """Select the records of seg that overlap [start_et, end_et] (TDB seconds past J2000).

    Returns:
        (first_record_index, n_records, filtered_coefficients, new_init_et), where
        new_init_et is the start time of the first selected record. When nothing is
        selected it returns (0, 0, [], seg.init_et).
    """
    if not seg.coefficients:
        return 0, 0, [], seg.init_et

    # Find first and last record indices that overlap with time range
    first_rec = max(0, int((start_et - seg.init_et) / seg.interval_sec))

    # End time of last needed record
    last_rec = min(seg.n_records - 1,
                   int((end_et - seg.init_et) / seg.interval_sec) + 1)

    if first_rec > last_rec or first_rec >= seg.n_records:
        return 0, 0, [], seg.init_et

    n_recs = last_rec - first_rec + 1

    # Extract coefficients for selected records
    start_idx = first_rec * seg.record_size
    end_idx = (last_rec + 1) * seg.record_size
    filtered = seg.coefficients[start_idx:end_idx]

    # Update init_et to reflect first selected record
    new_init_et = seg.init_et + first_rec * seg.interval_sec

    return first_rec, n_recs, filtered, new_init_et


def segment_key(target: int, center: int) -> str:
    """Return the C++ name prefix for a (target, center) body pair, e.g. "SUN_SSB"."""
    names = {
        (10, 0): "SUN_SSB",
        (3, 0): "EMB_SSB",
        (301, 3): "MOON_EMB",
        (5, 0): "JUPITER_SSB"
    }
    return names.get((target, center), f"BODY{target}_{center}")


def generate_cpp_header(segments: Dict[Tuple[int, int], Tuple[SPKSegment, float, int, List[float]]],
                        output_path: Path,
                        emrat: float = 81.3005690699) -> None:
    """Write the C++ header with one SegmentInfo and one coefficient array per segment.

    segments: {(target, center): (segment, new_init_et, n_records, coefficients)};
    emrat: the Earth/Moon mass ratio written into the header.
    """

    lines = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("// =============================================================================")
    lines.append("// DE440 Embedded Chebyshev Coefficients (1950-2100 Subset)")
    lines.append("// =============================================================================")
    lines.append("//")
    lines.append("// This file contains embedded DE440 ephemeris coefficients for:")
    lines.append("// - Sun (body 10) relative to Solar System Barycenter (body 0)")
    lines.append("// - Earth-Moon Barycenter (body 3) relative to SSB (body 0)")
    lines.append("// - Moon (body 301) relative to EMB (body 3)")
    lines.append("// - Jupiter Barycenter (body 5) relative to SSB (body 0)")
    lines.append("//")
    lines.append("// Time range: 1950-01-01 to 2100-01-01 (150 years)")
    lines.append("// Source: JPL DE440 ephemeris (Park et al. 2021)")
    lines.append("//")
    lines.append(f"// Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("// Generator: scripts/extract_de440_subset.py")
    lines.append("//")
    lines.append("// =============================================================================")
    lines.append("")
    lines.append("#include <array>")
    lines.append("#include <cstddef>")
    lines.append("")
    lines.append("namespace ballistic {")
    lines.append("namespace ephemeris {")
    lines.append("namespace de440_embedded {")
    lines.append("")

    # Constants
    lines.append("// DE440 constants")
    lines.append(f"inline constexpr double EMRAT = {emrat:.10f};  // Earth/Moon mass ratio")
    lines.append(f"inline constexpr double START_ET = {START_ET:.1f};  // 1950-01-01 TDB seconds past J2000")
    lines.append(f"inline constexpr double END_ET = {END_ET:.1f};    // 2100-01-01 TDB seconds past J2000")
    lines.append("")

    # Segment info struct
    lines.append("// Segment metadata")
    lines.append("struct SegmentInfo {")
    lines.append("    double init_et;        // Initial epoch (TDB seconds past J2000)")
    lines.append("    double interval_sec;   // Time interval per record")
    lines.append("    int n_coeffs;          // Chebyshev coefficients per component")
    lines.append("    int n_records;         // Number of records")
    lines.append("    int record_size;       // Size of each record (doubles)")
    lines.append("};")
    lines.append("")

    total_coeffs = 0

    # Generate data for each segment
    for (target, center), (seg, new_init_et, n_recs, coeffs) in segments.items():
        key = segment_key(target, center)
        n_coeffs_total = len(coeffs)
        total_coeffs += n_coeffs_total

        lines.append(f"// {key}: body {target} relative to body {center}")
        lines.append(f"// Records: {n_recs}, Coeffs per component: {seg.n_coeffs}, Total doubles: {n_coeffs_total}")
        lines.append(f"inline constexpr SegmentInfo {key}_INFO = {{")
        lines.append(f"    {new_init_et:.1f},  // init_et")
        lines.append(f"    {seg.interval_sec:.1f},        // interval_sec")
        lines.append(f"    {seg.n_coeffs},              // n_coeffs")
        lines.append(f"    {n_recs},            // n_records")
        lines.append(f"    {seg.record_size}              // record_size")
        lines.append("};")
        lines.append("")

        # Generate coefficient array
        lines.append(f"inline constexpr std::array<double, {n_coeffs_total}> {key}_DATA = {{{{")

        # Format coefficients (4 per line for readability)
        for i in range(0, len(coeffs), 4):
            chunk = coeffs[i:i+4]
            formatted = ", ".join(f"{c:.17g}" for c in chunk)
            if i + 4 < len(coeffs):
                lines.append(f"    {formatted},")
            else:
                lines.append(f"    {formatted}")

        lines.append("}};")
        lines.append("")

    # Size summary
    total_bytes = total_coeffs * 8
    lines.append(f"// Total embedded data: {total_coeffs} doubles = {total_bytes:,} bytes ({total_bytes/1024/1024:.2f} MB)")
    lines.append("")

    lines.append("} // namespace de440_embedded")
    lines.append("} // namespace ephemeris")
    lines.append("} // namespace ballistic")
    lines.append("")

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Generated {output_path}")
    print(f"Total coefficients: {total_coeffs}")
    print(f"Total size: {total_bytes:,} bytes ({total_bytes/1024/1024:.2f} MB)")


def main():
    """Read the SPK file named on the command line and write the header."""
    if len(sys.argv) < 2:
        print("Usage: python extract_de440_subset.py <input.bsp> [output.hpp]")
        print("       python extract_de440_subset.py data/de440.bsp")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = Path("include/ballistic/ephemeris/de440_coefficients.hpp")

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Reading {input_path}...")

    with open(input_path, 'rb') as f:
        # Read file header
        nd, ni, forward_ptr, backward_ptr = read_daf_file_record(f)
        print(f"DAF format: ND={nd}, NI={ni}")

        # Read segment summaries
        segments_list = read_segment_summaries(f, forward_ptr, nd, ni)
        print(f"Found {len(segments_list)} segments")

        # Bodies we need
        needed_bodies = {
            (10, 0),   # Sun relative to SSB
            (3, 0),    # EMB relative to SSB
            (301, 3),  # Moon relative to EMB
            (5, 0),    # Jupiter relative to SSB
        }

        # Filter and read needed segments
        filtered_segments = {}

        for seg in segments_list:
            key = (seg.target_body, seg.center_body)
            if key not in needed_bodies:
                continue

            print(f"Processing {segment_key(*key)}: body {seg.target_body} -> {seg.center_body}")
            print(f"  Time range: {seg.start_et:.0f} to {seg.end_et:.0f} ET")

            # Read coefficient data
            read_segment_data(f, seg)
            print(f"  Records: {seg.n_records}, Coeffs/component: {seg.n_coeffs}, Record size: {seg.record_size}")

            # Filter to 1950-2100 range
            first_rec, n_recs, filtered_coeffs, new_init_et = filter_records_by_time(seg, START_ET, END_ET)
            print(f"  Filtered: records {first_rec} to {first_rec + n_recs - 1} ({n_recs} records)")
            print(f"  Coefficients: {len(filtered_coeffs)} doubles")

            if n_recs > 0:
                filtered_segments[key] = (seg, new_init_et, n_recs, filtered_coeffs)

        if len(filtered_segments) != len(needed_bodies):
            missing = needed_bodies - set(filtered_segments.keys())
            print(f"Warning: Missing segments: {missing}")

    # Generate C++ header
    print(f"\nGenerating {output_path}...")
    generate_cpp_header(filtered_segments, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
