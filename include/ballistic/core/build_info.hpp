#pragma once

// =============================================================================
// Build information — how this copy of the library was compiled
// =============================================================================
//
// The library is mostly header-only, so the answer comes from the macros of
// the translation unit that calls build_info(). The ones that matter are set
// PUBLIC by the CMake target (OpenMP), so every
// consumer of the `ballistic` target sees the same values as the library's
// own compiled sources.
//
//   with_openmp   compiled with OpenMP (parallel STM and 4D table builders)
//   compiler      compiler name and version, e.g. "GCC 13.3.0"
//   cpp_standard  C++ standard year modulo 100 (23, 20, 17), from __cplusplus
//                 (GCC 13 reports C++23 as 202100, so ranges are used)
//   build_type    "Release" when NDEBUG is defined (also RelWithDebInfo and
//                 MinSizeRel), "Debug" otherwise
//
// The NRL atmosphere and wind models are not a build property: they are a
// plugin loaded at run time (atmosphere/nrl_models.hpp, nrl::status()).
// =============================================================================

#include <string_view>

namespace ballistic {

struct BuildInfo {
    bool with_openmp;
    std::string_view compiler;
    int cpp_standard;
    std::string_view build_type;
};

namespace build_info_detail {
#define BALLISTIC_BI_STR2(x) #x
#define BALLISTIC_BI_STR(x) BALLISTIC_BI_STR2(x)
#if defined(__clang__)
inline constexpr std::string_view kCompiler =
    "Clang " BALLISTIC_BI_STR(__clang_major__) "." BALLISTIC_BI_STR(__clang_minor__) "." BALLISTIC_BI_STR(__clang_patchlevel__);
#elif defined(__GNUC__)
inline constexpr std::string_view kCompiler =
    "GCC " BALLISTIC_BI_STR(__GNUC__) "." BALLISTIC_BI_STR(__GNUC_MINOR__) "." BALLISTIC_BI_STR(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
inline constexpr std::string_view kCompiler = "MSVC " BALLISTIC_BI_STR(_MSC_VER);
#else
inline constexpr std::string_view kCompiler = "unknown compiler";
#endif
#undef BALLISTIC_BI_STR
#undef BALLISTIC_BI_STR2

// MSVC keeps __cplusplus at 199711L unless /Zc:__cplusplus; _MSVC_LANG is right.
#if defined(_MSVC_LANG)
inline constexpr long kCplusplus = _MSVC_LANG;
#else
inline constexpr long kCplusplus = __cplusplus;
#endif
}  // namespace build_info_detail

/// How this copy of the library was compiled (see the header comment).
[[nodiscard]] constexpr BuildInfo build_info() noexcept {
    namespace m = build_info_detail;
    return BuildInfo{
#ifdef _OPENMP
        .with_openmp = true,
#else
        .with_openmp = false,
#endif
        .compiler = m::kCompiler,
        .cpp_standard = m::kCplusplus >= 202100L ? 23
                      : m::kCplusplus >= 202002L ? 20
                      : m::kCplusplus >= 201703L ? 17
                                                 : 14,
#ifdef NDEBUG
        .build_type = "Release",
#else
        .build_type = "Debug",
#endif
    };
}

}  // namespace ballistic
