# Python bindings for the Ballistic library using nanobind
# =============================================================================
#
# This builds the _core extension module that provides Python bindings for:
# - Epoch class (time handling)
# - TrajectoryConfig, BallisticModel, WeatherConditions (configuration)
# - Trajectory class (orbital propagation)
# - TrajectoryInterpolator class (high-accuracy interpolation)
# - TrajectoryBatch class (batch processing)
#
# Usage:
#   cmake -B build -DBALLISTIC_BUILD_PYTHON=ON
#   cmake --build build --config Release
#
# Or via uv:
#   uv pip install ./python
#
# =============================================================================

cmake_minimum_required(VERSION 3.20)

# Find Python
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# Fetch nanobind
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind
    GIT_TAG v2.4.0
)
FetchContent_MakeAvailable(nanobind)

# Define the extension module
nanobind_add_module(_core
    NB_STATIC
    src/bindings.cpp
    src/init_bindings.cpp
    src/diagnostics_bindings.cpp
    src/satellites_bindings.cpp
    src/constants_bindings.cpp
    src/atmosphere_bindings.cpp
    src/ea_bindings.cpp
    src/epoch_bindings.cpp
    src/types_bindings.cpp
    src/trajectory_bindings.cpp
    src/batch_bindings.cpp
    src/calibration_bindings.cpp
    src/multisensor_bindings.cpp
    src/validation_bindings.cpp
    src/sp3_bindings.cpp
    src/sgp4_bindings.cpp
)

# Link to the ballistic library
target_link_libraries(_core PRIVATE ballistic)

# Include path for trajectory_batch.hpp
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(SKBUILD)
    # Wheel build: install the module into the package directory of the wheel
    # (pyproject.toml sets wheel.install-dir = "ballistic", so "." is
    # <wheel>/ballistic). Nothing is written into the source tree, so a module
    # left there by a developer build can never end up in a wheel.
    install(TARGETS _core
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .)
else()
    # Developer build: put the module next to the Python sources so that
    # PYTHONPATH=./python imports it. Multi-config generators (Visual Studio,
    # Xcode) would add a per-configuration subdirectory, so the output
    # directory is set for every configuration explicitly.
    set_target_properties(_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ballistic"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ballistic"
    )
    foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER "${_cfg}" _cfg_upper)
        set_target_properties(_core PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_CURRENT_SOURCE_DIR}/ballistic"
            RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_CURRENT_SOURCE_DIR}/ballistic"
        )
    endforeach()
endif()
