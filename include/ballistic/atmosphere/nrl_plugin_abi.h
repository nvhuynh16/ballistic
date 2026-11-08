#ifndef BALLISTIC_ATMOSPHERE_NRL_PLUGIN_ABI_H
#define BALLISTIC_ATMOSPHERE_NRL_PLUGIN_ABI_H

/*
 * =============================================================================
 * C interface between the library and the NRL models plugin
 * =============================================================================
 *
 * NRLMSIS 2.1 (atmosphere) and HWM14 (wind) are not part of the library: they
 * are built as a separate shared library, ballistic_nrl, because NRLMSIS 2.1 is
 * licensed for research, academic and non-profit use only (NRL-SOF-014-1; see
 * THIRD_PARTY_NOTICES.md). The library loads it at run time (nrl_models.hpp)
 * and calls the models only through the functions below.
 *
 * The interface is plain C so that a plugin built by one compiler can be loaded
 * by a library built by another. The inputs are exactly the arguments of the
 * models' own entry points (NRLMSIS21::msiscalc and HWM14::hwm14_calc), so a
 * call through the plugin gives the same numbers as a direct call.
 *
 * Change BALLISTIC_NRL_ABI_VERSION whenever a struct or a signature changes;
 * the loader refuses a plugin built for another version.
 * =============================================================================
 */

#define BALLISTIC_NRL_ABI_VERSION 1

#if defined(_WIN32)
#  if defined(BALLISTIC_NRL_BUILDING_PLUGIN)
#    define BALLISTIC_NRL_EXPORT __declspec(dllexport)
#  else
#    define BALLISTIC_NRL_EXPORT __declspec(dllimport)
#  endif
#else
#  define BALLISTIC_NRL_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* NRLMSIS 2.1 input (NRLMSIS21::Input). */
typedef struct BallisticNrlMsisInput {
    double day;      /* day of year, fractional (1-366) */
    double utsec;    /* UT seconds of the day */
    double alt_km;   /* geodetic altitude [km] */
    double lat_deg;  /* geodetic latitude [deg] */
    double lon_deg;  /* geodetic longitude [deg] */
    double f107a;    /* 81-day centered mean F10.7 */
    double f107;     /* daily F10.7, previous day */
    double ap[7];    /* NRLMSIS Ap array */
} BallisticNrlMsisInput;

/* NRLMSIS 2.1 output (NRLMSIS21::Output). dn is 1-based as in the model:
 * dn[1] total mass density [kg/m^3], dn[2..10] number densities [m^-3] of
 * N2, O2, O, He, H, Ar, N, anomalous O, NO; dn[0] is unused. */
typedef struct BallisticNrlMsisOutput {
    double tn;       /* temperature at the altitude [K] */
    double tex;      /* exospheric temperature [K] */
    double dn[11];
} BallisticNrlMsisOutput;

/* HWM14 input (the arguments of HWM14::hwm14_calc). */
typedef struct BallisticNrlHwmInput {
    int iyd;         /* year * 1000 + day of year */
    float sec;       /* UT seconds of the day */
    float alt_km;    /* geodetic altitude [km] */
    float glat_deg;  /* geodetic latitude [deg] */
    float glon_deg;  /* geodetic longitude [deg] */
    float stl_h;     /* local solar time [h] */
    float f107a;     /* 81-day mean F10.7 (not used by HWM14) */
    float f107;      /* daily F10.7 (not used by HWM14) */
    float ap[2];     /* ap[1]: 3-hour Ap for the disturbance winds */
} BallisticNrlHwmInput;

/* BALLISTIC_NRL_ABI_VERSION of the plugin. */
BALLISTIC_NRL_EXPORT int ballistic_nrl_abi_version(void);

/* One line naming the models and their versions, e.g. "NRLMSIS 2.1, HWM14". */
BALLISTIC_NRL_EXPORT const char* ballistic_nrl_models(void);

/* NRLMSIS 2.1 at one point. Returns 0 on success; otherwise the reason is in
 * ballistic_nrl_last_error(). */
BALLISTIC_NRL_EXPORT int ballistic_nrl_msis(const BallisticNrlMsisInput* in,
                                            BallisticNrlMsisOutput* out);

/* HWM14 horizontal wind at one point: wind[0] meridional (north +), wind[1]
 * zonal (east +) [m/s]. Returns 0 on success. Each calling thread keeps its own
 * model instance (HWM14 caches intermediate terms). */
BALLISTIC_NRL_EXPORT int ballistic_nrl_hwm14(const BallisticNrlHwmInput* in, float wind[2]);

/* Why the last call on this thread failed (empty after a success). */
BALLISTIC_NRL_EXPORT const char* ballistic_nrl_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* BALLISTIC_ATMOSPHERE_NRL_PLUGIN_ABI_H */
