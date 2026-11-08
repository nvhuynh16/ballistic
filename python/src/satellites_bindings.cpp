// The satellite registry (include/ballistic/satellites.hpp) for Python:
// ballistic.satellite_properties builds its dataclasses from these rows, so
// C++ and Python read the same values from one table.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ballistic/satellites.hpp>
#include <ballistic/gnss_blocks.hpp>

#include <string>

namespace nb = nanobind;
using namespace ballistic;

void bind_satellites(nb::module_& m) {
    m.def("_satellite_table", [] {
        nb::list out;
        for (const auto& s : satellites::all()) {
            nb::dict d;
            d["name"] = std::string(s.name);
            d["mass_kg"] = s.mass_kg;
            d["drag_area_m2"] = s.drag_area_m2;
            d["cd"] = s.cd;
            d["srp_area_m2"] = s.srp_area_m2;
            d["srp_reflectivity"] = s.srp_reflectivity;
            d["altitude_km"] = s.altitude_km;
            d["gravity_degree"] = s.gravity_degree;
            d["drag_kind"] = std::string(satellites::fit_kind_name(s.drag_kind));
            d["srp_kind"] = std::string(satellites::fit_kind_name(s.srp_kind));
            d["provenance"] = std::string(s.provenance);
            out.append(d);
        }
        return out;
    }, "Internal: the rows of the C++ satellite registry (mass, drag area, drag "
       "coefficient, radiation area, ... with their sources), one dict per satellite; "
       "ballistic.satellite_properties is built from them.");

    m.def("_srp_fit_table", [] {
        nb::dict out;
        for (const auto& f : satellites::kSrpFit)
            out[nb::str(std::string(f.name).c_str())] = nb::make_tuple(f.base_area_m2, f.k);
        return out;
    }, "Internal: radiation-pressure area corrections fitted to orbit data, as a dict "
       "name -> (base area [m^2], scale k); the effective area is base area times k.");

    // GNSS blocks (include/ballistic/gnss_blocks.hpp): ballistic.satellite_properties
    // builds GnssBlock from these rows; the prefix rule stays in C++ (find_block).
    auto block_dict = [](const gnss::Block& b) {
        nb::dict d;
        d["name"] = std::string(b.name);
        d["mass_kg_nominal"] = b.mass_kg_nominal;
        d["srp_base_area_m2"] = b.srp_base_area_m2;
        d["srp_k"] = b.srp_k;
        d["srp_area_m2"] = b.srp_area_m2();
        d["cr"] = b.cr;
        d["srp_kind"] = std::string(satellites::fit_kind_name(b.srp_kind));
        d["provenance"] = std::string(b.provenance);
        return d;
    };

    m.def("_gnss_block_table", [block_dict] {
        nb::list out;
        for (const auto& b : gnss::all()) out.append(block_dict(b));
        return out;
    }, "Internal: the rows of the C++ GNSS block table (radiation base area, fitted "
       "scale k, Cr, mass, with their sources), one dict per block, fits applied.");

    m.def("_gnss_srp_fit_table", [] {
        nb::dict out;
        for (const auto& f : gnss::kGnssSrpFit)
            out[nb::str(std::string(f.block).c_str())] = f.k;
        return out;
    }, "Internal: the fitted GNSS radiation-area scales, as a dict block -> k.");

    m.def("_gnss_find_block", [block_dict](const std::string& name) {
        return block_dict(gnss::get_block(name));
    }, nb::arg("igs_block_name"),
       "Internal: the GNSS block row for an IGS block name (exact, or the longest known "
       "name followed by '-', e.g. GPS-IIR-A -> GPS-IIR). Raises ConfigError otherwise.");
}
