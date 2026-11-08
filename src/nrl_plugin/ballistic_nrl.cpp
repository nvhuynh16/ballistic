// =============================================================================
// ballistic_nrl — the NRLMSIS 2.1 and HWM14 plugin
// =============================================================================
//
// The only compiled code that contains the NRL models. It is built as its own
// shared library (CMake target ballistic_nrl) and loaded by the library at run
// time (atmosphere/nrl_models.hpp), so the library and the Python wheel carry
// no NRL code. NRLMSIS 2.1 is licensed for research, academic and non-profit
// use only (NRL-SOF-014-1, docs/licenses/nrlmsis2.1_license.txt); HWM14 is
// distributed as a U.S. Government work (THIRD_PARTY_NOTICES.md).
//
// No exception may cross the C interface: each entry point turns a failure into
// a return code and a message (ballistic_nrl_last_error).
// =============================================================================

#define BALLISTIC_NRL_BUILDING_PLUGIN
#include "ballistic/atmosphere/nrl_plugin_abi.h"
#include "ballistic/atmosphere/internal/hwm14.hpp"
#include "ballistic/atmosphere/internal/nrlmsis21_full.hpp"

#include <exception>
#include <string>

namespace {

thread_local std::string g_last_error;

int fail(const char* what) {
    g_last_error = what;
    return 1;
}

}  // namespace

extern "C" {

int ballistic_nrl_abi_version(void) { return BALLISTIC_NRL_ABI_VERSION; }

const char* ballistic_nrl_models(void) { return "NRLMSIS 2.1, HWM14"; }

int ballistic_nrl_msis(const BallisticNrlMsisInput* in, BallisticNrlMsisOutput* out) {
    using ballistic::atmosphere::NRLMSIS21;
    if (in == nullptr || out == nullptr) return fail("ballistic_nrl_msis: null argument");
    try {
        // NRLMSIS21 is const and keeps no state: one instance serves every thread.
        static const NRLMSIS21 model;
        NRLMSIS21::Input input;
        input.day = in->day;
        input.utsec = in->utsec;
        input.alt = in->alt_km;
        input.lat = in->lat_deg;
        input.lon = in->lon_deg;
        input.f107a = in->f107a;
        input.f107 = in->f107;
        for (int i = 0; i < 7; ++i) input.ap[i] = in->ap[i];
        const NRLMSIS21::Output o = model.msiscalc(input);
        out->tn = o.tn;
        out->tex = o.tex;
        for (int i = 0; i < 11; ++i) out->dn[i] = o.dn[i];
    } catch (const std::exception& e) {
        return fail(e.what());
    } catch (...) {  // guard-exempt(CATCH-ALL): C interface; the failure is returned, not swallowed
        return fail("ballistic_nrl_msis: unknown error");
    }
    g_last_error.clear();
    return 0;
}

int ballistic_nrl_hwm14(const BallisticNrlHwmInput* in, float wind[2]) {
    using ballistic::atmosphere::HWM14;
    if (in == nullptr || wind == nullptr) return fail("ballistic_nrl_hwm14: null argument");
    try {
        // HWM14 caches intermediate terms in mutable members: one per thread.
        thread_local HWM14 model = [] { HWM14 h; h.init(); return h; }();
        model.hwm14_calc(in->iyd, in->sec, in->alt_km, in->glat_deg, in->glon_deg, in->stl_h,
                         in->f107a, in->f107, in->ap, wind);
    } catch (const std::exception& e) {
        return fail(e.what());
    } catch (...) {  // guard-exempt(CATCH-ALL): C interface; the failure is returned, not swallowed
        return fail("ballistic_nrl_hwm14: unknown error");
    }
    g_last_error.clear();
    return 0;
}

const char* ballistic_nrl_last_error(void) { return g_last_error.c_str(); }

}  // extern "C"
