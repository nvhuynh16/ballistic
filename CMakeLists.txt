cmake_minimum_required(VERSION 3.20)
project(Ballistic
    VERSION 0.1.3
    DESCRIPTION "High-fidelity orbital trajectory prediction library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BALLISTIC_BUILD_TESTS "Build unit tests" ON)
option(BALLISTIC_BUILD_EXAMPLES "Build examples" ON)
option(BALLISTIC_BUILD_PYTHON "Build Python bindings" OFF)
option(BALLISTIC_BUILD_TUTORIAL_EXAMPLES "Build docs/latex tutorial example programs" OFF)
# NRLMSIS 2.1 (non-commercial license, NRL-SOF-014-1) and HWM14 (U.S.
# Government work) are never compiled into the library: they are the separate
# shared library ballistic_nrl, loaded at run time for model-based drag
# (include/ballistic/atmosphere/nrl_models.hpp). The wheel build never builds
# it; wheels.yml builds it on its own for the GitHub release.
if(SKBUILD)
    set(_ballistic_nrl_default OFF)
else()
    set(_ballistic_nrl_default ON)
endif()
option(BALLISTIC_BUILD_NRL_PLUGIN "Build the NRLMSIS 2.1 / HWM14 plugin ballistic_nrl (non-commercial license)" ${_ballistic_nrl_default})

# Find Eigen3 (system install preferred; fall back to FetchContent)
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(Eigen3_FOUND)
    # A previously FetchContent'd Eigen re-resolves through the cached
    # Eigen3_DIR inside the build tree — label it honestly.
    if(Eigen3_DIR MATCHES "_deps")
        set(BALLISTIC_EIGEN_PROVENANCE "FetchContent ${Eigen3_VERSION} (cached: ${Eigen3_DIR})")
    else()
        set(BALLISTIC_EIGEN_PROVENANCE "system ${Eigen3_VERSION} (${Eigen3_DIR})")
    endif()
else()
    message(STATUS "System Eigen3 not found; fetching v3.4.0 via FetchContent")
    include(FetchContent)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(eigen)
    set(BALLISTIC_EIGEN_PROVENANCE "FetchContent 3.4.0 (gitlab tag)")
endif()
message(STATUS "Eigen3: ${BALLISTIC_EIGEN_PROVENANCE}")

# Record the toolchain so FP-sensitive baselines (byte-stability probes, tight
# accuracy gates) are diffable against the environment they were verified on.
cmake_host_system_information(RESULT BALLISTIC_CPU_DESC QUERY PROCESSOR_DESCRIPTION)
file(WRITE "${CMAKE_BINARY_DIR}/toolchain_info.txt"
    "compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\n"
    "eigen: ${BALLISTIC_EIGEN_PROVENANCE}\n"
    "build_type: ${CMAKE_BUILD_TYPE}\n"
    "system: ${CMAKE_SYSTEM}\n"
    "cpu: ${BALLISTIC_CPU_DESC}\n"
    "native_arch: ${BALLISTIC_NATIVE_ARCH}\n"
    "tutorial_examples: ${BALLISTIC_BUILD_TUTORIAL_EXAMPLES}\n"
    "nrl_plugin: ${BALLISTIC_BUILD_NRL_PLUGIN}\n"
    "cmake: ${CMAKE_VERSION} (${CMAKE_COMMAND})\n"
)
message(STATUS "Toolchain record: ${CMAKE_BINARY_DIR}/toolchain_info.txt "
    "(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")

# Main library (header-only with minimal compiled components)
add_library(ballistic
    src/time/epoch.cpp
    src/atmosphere/drag_environment_generators.cpp
    src/atmosphere/nrl_models.cpp
    # Embedded coefficient blobs: data lives in .cpp (compiled once) with only
    # `extern const` declarations in the headers, so TUs that include the
    # umbrella header no longer parse ~280K lines of constants they never use.
    src/ephemeris/de440_coefficients.cpp
    src/gravity/egm2008_coefficients.cpp
)

target_include_directories(ballistic
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(ballistic
    PUBLIC
        Eigen3::Eigen
)

target_compile_features(ballistic PUBLIC cxx_std_23)

# The NRL models plugin is opened with dlopen (LoadLibrary on Windows).
target_link_libraries(ballistic PUBLIC ${CMAKE_DL_LIBS})

# The NRL models plugin: a plain shared library with a C interface
# (nrl_plugin_abi.h), named ballistic_nrl.<so|dylib|dll> on every platform.
# It includes the model headers only; it does not link the library.
if(BALLISTIC_BUILD_NRL_PLUGIN)
    add_library(ballistic_nrl SHARED src/nrl_plugin/ballistic_nrl.cpp)
    target_include_directories(ballistic_nrl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(ballistic_nrl PRIVATE Eigen3::Eigen)
    target_compile_features(ballistic_nrl PRIVATE cxx_std_23)
    set_target_properties(ballistic_nrl PROPERTIES
        PREFIX ""
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nrl"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nrl")
    foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER "${_cfg}" _cfg_upper)
        set_target_properties(ballistic_nrl PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/nrl"
            RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/nrl")
    endforeach()
    if(MSVC)
        target_compile_options(ballistic_nrl PRIVATE /utf-8 /bigobj /permissive- /EHsc)
        target_compile_definitions(ballistic_nrl PRIVATE _USE_MATH_DEFINES NOMINMAX)
    endif()
    message(STATUS "Ballistic: building the NRL models plugin ballistic_nrl (NRLMSIS 2.1 is "
        "licensed for research, academic and non-profit use only; see THIRD_PARTY_NOTICES.md)")
endif()

# Required so the static library can link into the Python shared module.
set_target_properties(ballistic PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Compiler warnings
if(MSVC)
    target_compile_options(ballistic PRIVATE /W4)
else()
    target_compile_options(ballistic PRIVATE -Wall -Wextra -Wpedantic)
endif()

# MSVC: sources are UTF-8 (comments use ρ, °, ×); the embedded coefficient
# files exceed the default object-section limit; standard conformance and
# C++ exception semantics; M_PI and friends from <cmath>; no min/max macros
# from <windows.h>; no deprecation warnings for the standard C I/O functions.
# PUBLIC because every consumer compiles the same headers.
if(MSVC)
    target_compile_options(ballistic PUBLIC /utf-8 /bigobj /permissive- /EHsc)
    target_compile_definitions(ballistic PUBLIC
        _USE_MATH_DEFINES NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

# CPU-specific optimization (~1.5-2x speedup on Core Ultra 9 / AVX-512).
# Public so test executables and diagnostics also pick it up.
# Default OFF: portable codegen by default — native codegen leaks
# to consumers through the PUBLIC flag and CPU-locks all FP pins. Opt in
# locally with -DBALLISTIC_NATIVE_ARCH=ON.
option(BALLISTIC_NATIVE_ARCH "Build with -march=native" OFF)
if(BALLISTIC_NATIVE_ARCH AND NOT MSVC)
    target_compile_options(ballistic PUBLIC -march=native)
endif()

# OpenMP (for STM parallel A(t) computation)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ballistic PUBLIC OpenMP::OpenMP_CXX)
endif()

# Project-root macro for locating committed data without hardcoded paths.
# Needed by tests AND by the tutorial examples' _check_glide_parity target
# (it includes tests/test_helpers.hpp) — so define it whenever either is on,
# else BUILD_TUTORIAL_EXAMPLES=ON with BUILD_TESTS=OFF fails to configure.
if(BALLISTIC_BUILD_TESTS OR BALLISTIC_BUILD_TUTORIAL_EXAMPLES)
    add_compile_definitions(BALLISTIC_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
endif()

# Tests
if(BALLISTIC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(BALLISTIC_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Python bindings
if(BALLISTIC_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Tutorial example programs (docs/latex) — opt-in, not part of normal/CI builds
if(BALLISTIC_BUILD_TUTORIAL_EXAMPLES)
    add_subdirectory(docs/latex/examples/cpp)
endif()

# Installation. Skipped entirely under scikit-build-core (SKBUILD): the wheel
# must contain only the Python package + _core module, not the C++ SDK
# (91 headers + libballistic.a made the wheel 47.8 MB).
if(NOT SKBUILD)
include(GNUInstallDirs)
install(TARGETS ballistic
    EXPORT ballisticTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ballistic
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
# The plugin is not linked by anything; point InitConfig::nrl_models_path (or
# BALLISTIC_NRL_MODELS) at the installed file.
if(BALLISTIC_BUILD_NRL_PLUGIN)
    install(TARGETS ballistic_nrl
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
endif()
