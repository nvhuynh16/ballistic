# Tests CMakeLists.txt

# =========================================================================
# Helper functions to reduce boilerplate
# =========================================================================

# gzip support for the tests' SP3 reader (sp3_truth.hpp reads committed .gz
# truth files): system zlib when found, otherwise zlib fetched and built
# statically. Without it sp3_truth.hpp falls back to a `gzip -dc` pipe.
add_library(ballistic_test_gzip INTERFACE)
find_package(ZLIB QUIET)
if(ZLIB_FOUND)
    target_link_libraries(ballistic_test_gzip INTERFACE ZLIB::ZLIB)
    message(STATUS "Tests: gzip via system zlib ${ZLIB_VERSION_STRING}")
else()
    message(STATUS "Tests: system zlib not found; fetching zlib v1.3.1")
    include(FetchContent)
    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
    FetchContent_Declare(zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG v1.3.1
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(zlib)
    # Only the static library is needed; keep the rest out of `all`.
    foreach(_zt zlib example minigzip example64 minigzip64)
        if(TARGET ${_zt})
            set_target_properties(${_zt} PROPERTIES EXCLUDE_FROM_ALL ON)
        endif()
    endforeach()
    target_link_libraries(ballistic_test_gzip INTERFACE zlibstatic)
    # zlib's own CMakeLists sets its include paths per directory, not on the
    # target; zconf.h is generated in the binary directory.
    target_include_directories(ballistic_test_gzip INTERFACE
        ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
endif()
target_compile_definitions(ballistic_test_gzip INTERFACE BALLISTIC_HAVE_ZLIB)

# Add a test executable that is registered with CTest.
#
# Labels: the tier label (unit / integration / validation) derives from the
# source directory; extra labels may be appended as trailing arguments —
# notably `heavy` (long-running or assertion-light; excluded from the
# default gate).
# `nrl`: the test needs the NRLMSIS 2.1 / HWM14 plugin (model-based drag).
# With -DBALLISTIC_BUILD_NRL_PLUGIN=OFF these are not built (several include
# the model headers directly); tests that only have a few such cells compile
# them out with #ifndef BALLISTIC_TEST_NO_NRL_PLUGIN instead.
# Default gate: ctest -LE heavy. Without the plugin: ctest -LE 'heavy|nrl'.
# Every test and diagnostic links tests/test_env.cpp, which points
# BALLISTIC_DATA_DIR at tests/data/eop/ and BALLISTIC_NRL_MODELS at the plugin
# built here (each unless already set), so results do not depend on the
# user's data directory.
# (Source properties are per directory, so they are set by each caller.)
function(ballistic_test_env_setup target)
    set(env_source ${PROJECT_SOURCE_DIR}/tests/test_env.cpp)
    set(defs "BALLISTIC_TEST_DATA_DIR=\"${PROJECT_SOURCE_DIR}/tests/data/eop\"")
    if(TARGET ballistic_nrl)
        list(APPEND defs "BALLISTIC_TEST_NRL_PLUGIN=\"$<TARGET_FILE:ballistic_nrl>\"")
    endif()
    set_source_files_properties(${env_source} PROPERTIES COMPILE_DEFINITIONS "${defs}")
    target_sources(${target} PRIVATE ${env_source})
    if(TARGET ballistic_nrl)
        add_dependencies(${target} ballistic_nrl)
    else()
        target_compile_definitions(${target} PRIVATE BALLISTIC_TEST_NO_NRL_PLUGIN)
    endif()
endfunction()

function(ballistic_add_test test_name source_file)
    if(NOT TARGET ballistic_nrl AND "nrl" IN_LIST ARGN)
        return()
    endif()
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} PRIVATE ballistic ballistic_test_gzip)
    ballistic_test_env_setup(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
    if(source_file MATCHES "^unit/")
        set(_labels unit)
    elseif(source_file MATCHES "^integration/")
        set(_labels integration)
    elseif(source_file MATCHES "^validation/")
        set(_labels validation)
    else()
        set(_labels misc)
    endif()
    list(APPEND _labels ${ARGN})
    set_tests_properties(${test_name} PROPERTIES LABELS "${_labels}")
endfunction()

# =========================================================================
# Modular Unit Tests
# =========================================================================

# Core utilities
ballistic_add_test(test_constants unit/test_constants.cpp)
ballistic_add_test(test_defaults unit/test_defaults.cpp)
ballistic_add_test(test_build_info unit/test_build_info.cpp)
ballistic_add_test(test_nrl_models unit/test_nrl_models.cpp nrl)
if(TARGET test_nrl_models)   # it loads the built plugin itself, by path
    target_compile_definitions(test_nrl_models PRIVATE
        BALLISTIC_TEST_NRL_PLUGIN="$<TARGET_FILE:ballistic_nrl>")
endif()
ballistic_add_test(test_types unit/test_types.cpp)
ballistic_add_test(test_julian_date unit/test_julian_date.cpp)
ballistic_add_test(test_force_jacobians unit/test_force_jacobians.cpp)
ballistic_add_test(test_j2_hessian_closed_form unit/test_j2_hessian_closed_form.cpp)
ballistic_add_test(test_gravity_hessian unit/test_gravity_hessian.cpp)
ballistic_add_test(test_force_model_gravity_hessian unit/test_force_model_gravity_hessian.cpp)
ballistic_add_test(test_brent_minimize unit/test_brent_minimize.cpp)
ballistic_add_test(test_ea_cv_tiebreaker unit/test_ea_cv_tiebreaker.cpp)
ballistic_add_test(test_gm_rts_core unit/test_gm_rts_core.cpp)
ballistic_add_test(test_ea_method_advisor unit/test_ea_method_advisor.cpp)

# Time and coordinates
ballistic_add_test(test_epoch unit/test_epoch.cpp)
ballistic_add_test(test_epoch_label_drift unit/test_epoch_label_drift.cpp)
ballistic_add_test(test_staleness_warnings unit/test_staleness_warnings.cpp)
ballistic_add_test(test_geodetic unit/test_geodetic.cpp)
ballistic_add_test(test_ussa76 unit/test_ussa76.cpp)
ballistic_add_test(test_events unit/test_events.cpp)
ballistic_add_test(test_flow_regime_cd unit/test_flow_regime_cd.cpp)
ballistic_add_test(test_initial_state unit/test_initial_state.cpp)
ballistic_add_test(test_sgp4 unit/test_sgp4.cpp)

# Gravity and transforms
ballistic_add_test(test_iau_transforms unit/test_iau_transforms.cpp)
ballistic_add_test(test_point_mass unit/test_point_mass.cpp)

# Third body, atmosphere and drag
ballistic_add_test(test_third_body unit/test_third_body.cpp)
ballistic_add_test(test_embedded_ephemeris unit/test_embedded_ephemeris.cpp)
ballistic_add_test(test_space_weather_history unit/test_space_weather_history.cpp)
ballistic_add_test(test_make_drag_environment unit/test_make_drag_environment.cpp nrl)
ballistic_add_test(test_live_drag_env unit/test_live_drag_env.cpp nrl)
ballistic_add_test(test_diagnostics unit/test_diagnostics.cpp)
ballistic_add_test(test_log_density_storage unit/test_log_density_storage.cpp)
ballistic_add_test(test_drag_profiles unit/test_drag_profiles.cpp)
ballistic_add_test(test_drag_environment_lookup unit/test_drag_environment_lookup.cpp)
ballistic_add_test(test_drag_environment_acceleration unit/test_drag_environment_acceleration.cpp)
ballistic_add_test(test_trajectory_drag_env unit/test_trajectory_drag_env.cpp)
ballistic_add_test(test_from_models_snapshot unit/test_from_models_snapshot.cpp nrl)
ballistic_add_test(test_from_models_along unit/test_from_models_along.cpp nrl)
ballistic_add_test(test_refine_beta_from_ea unit/test_refine_beta_from_ea.cpp)
ballistic_add_test(test_estimator_drag_env unit/test_estimator_drag_env.cpp)
ballistic_add_test(test_empirical_acceleration unit/test_empirical_acceleration.cpp)
ballistic_add_test(test_cubic_spline unit/test_cubic_spline.cpp)
ballistic_add_test(test_hermite_table unit/test_hermite_table.cpp)

# Force model and integrators
ballistic_add_test(test_force_model unit/test_force_model.cpp)
ballistic_add_test(test_rk45 unit/test_rk45.cpp)
ballistic_add_test(test_symplectic unit/test_symplectic.cpp)

# Propagator and estimators
ballistic_add_test(test_propagator integration/test_propagator.cpp)
ballistic_add_test(test_from_models_4d integration/test_from_models_4d.cpp nrl)
ballistic_add_test(test_from_models_along_iterated integration/test_from_models_along_iterated.cpp nrl)
ballistic_add_test(test_coordinate_chains integration/test_coordinate_chains.cpp)
ballistic_add_test(test_bspline_estimator integration/test_bspline_estimator.cpp)
ballistic_add_test(test_bspline_vel_accel integration/test_bspline_vel_accel.cpp)
# highrate / matched assert only non-null results
# (no RMSE gates) — coverage value is smoke-only, wall cost is minutes.
# Labeled heavy pending real gates (the claims suite carries those).
ballistic_add_test(test_bspline_matched integration/test_bspline_matched.cpp heavy)
ballistic_add_test(test_bspline_highrate integration/test_bspline_highrate.cpp heavy)
# Explicit walls (measured: matched ~71 s, highrate ~524 s)
# so a hung run surfaces well before CTest's 1500 s fallback.
set_tests_properties(test_bspline_matched PROPERTIES TIMEOUT 600)
set_tests_properties(test_bspline_highrate PROPERTIES TIMEOUT 1200)
ballistic_add_test(test_beta_ea_glide integration/test_beta_ea_glide.cpp heavy)
# 12 sub-tests, most of them GN refinement variants (~120-300 s each).
# Default CTest timeout (1500 s nominal) is too tight.
set_tests_properties(test_beta_ea_glide PROPERTIES TIMEOUT 1800)
ballistic_add_test(test_leo_verification integration/test_leo_verification.cpp)

ballistic_add_test(test_glide_verification integration/test_glide_verification.cpp)
set_tests_properties(test_glide_verification PROPERTIES TIMEOUT 2400)  # ~641 s solo; headroom for CI runners
ballistic_add_test(test_ea_cv integration/test_ea_cv.cpp)
ballistic_add_test(test_ea_estimator integration/test_ea_estimator.cpp)
ballistic_add_test(test_ea_estimator_glide integration/test_ea_estimator_glide.cpp)
# The executable version of the README's EA accuracy table: every
# documented reconstruction claim has an asserting cell here.
ballistic_add_test(test_ea_accuracy_claims integration/test_ea_accuracy_claims.cpp claims)
set_tests_properties(test_ea_accuracy_claims PROPERTIES TIMEOUT 900 PROCESSORS 4)

ballistic_add_test(test_ea_smoother integration/test_ea_smoother.cpp)
ballistic_add_test(test_ea_problem_spec integration/test_ea_problem_spec.cpp)
set_tests_properties(test_ea_smoother PROPERTIES TIMEOUT 300)

ballistic_add_test(test_eof_parser unit/test_eof_parser.cpp)

# External validation
ballistic_add_test(test_orbit_validation validation/test_orbit_validation.cpp)
ballistic_add_test(test_sentinel1_validation validation/test_sentinel1_validation.cpp)
ballistic_add_test(test_esa_pod_validation validation/test_esa_pod_validation.cpp)
ballistic_add_test(test_champ_drag_validation validation/test_champ_drag_validation.cpp nrl)
# Note: test_champ_drag_validation gates its heaviest arcs behind the
# BALLISTIC_RUN_HEAVY_TESTS environment variable (off by default).
ballistic_add_test(test_gracefo_drag_validation validation/test_gracefo_drag_validation.cpp nrl)
ballistic_add_test(test_goce_reentry_validation validation/test_goce_reentry_validation.cpp nrl)
ballistic_add_test(test_gcrs_validation validation/test_gcrs_validation.cpp)
ballistic_add_test(test_vallado_transform validation/test_vallado_transform.cpp)
ballistic_add_test(test_egm2008_diagnostics validation/test_egm2008_diagnostics.cpp)
ballistic_add_test(test_gravity_reference validation/test_gravity_reference.cpp)

# Comprehensive SOFA/ERFA validation (special: needs extra include dir)
ballistic_add_test(test_sofa_comprehensive validation/test_sofa_comprehensive.cpp)
target_include_directories(test_sofa_comprehensive PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/validation)

# Feature Tests
ballistic_add_test(test_egm2008_loading unit/test_egm2008_loading.cpp)
ballistic_add_test(test_space_weather unit/test_space_weather.cpp)
ballistic_add_test(test_silent_substitutions unit/test_silent_substitutions.cpp)
ballistic_add_test(test_api_consistency unit/test_api_consistency.cpp)
ballistic_add_test(test_sp3 unit/test_sp3.cpp)
ballistic_add_test(test_jpl_integration unit/test_jpl_integration.cpp)
ballistic_add_test(test_trajectory unit/test_trajectory.cpp)
ballistic_add_test(test_force_breakdown unit/test_force_breakdown.cpp)
ballistic_add_test(test_quintic_hermite unit/test_quintic_hermite.cpp)
ballistic_add_test(test_trajectory_interpolator validation/test_trajectory_interpolator.cpp)
ballistic_add_test(test_radiation_pressure unit/test_radiation_pressure.cpp)
ballistic_add_test(test_relativistic unit/test_relativistic.cpp)
ballistic_add_test(test_solid_earth_tides unit/test_solid_earth_tides.cpp)
ballistic_add_test(test_ocean_tides unit/test_ocean_tides.cpp)
ballistic_add_test(test_chebyshev unit/test_chebyshev.cpp)
ballistic_add_test(test_math_audit unit/test_math_audit.cpp)
ballistic_add_test(test_propagator_audit unit/test_propagator_audit.cpp)
ballistic_add_test(test_itrf_transforms unit/test_itrf_transforms.cpp)
ballistic_add_test(test_earth_rotation unit/test_earth_rotation.cpp)
ballistic_add_test(test_interpolation unit/test_interpolation.cpp)
ballistic_add_test(test_eop_data unit/test_eop_data.cpp)
ballistic_add_test(test_leap_second_table unit/test_leap_second_table.cpp)
ballistic_add_test(test_util unit/test_util.cpp)
ballistic_add_test(test_beta_ea_roundtrip unit/test_beta_ea_roundtrip.cpp)
ballistic_add_test(test_glide_reentry unit/test_glide_reentry.cpp)

# Coverage Gap Tests (Multi-Regime External Validation)
ballistic_add_test(test_multiday_propagation validation/test_multiday_propagation.cpp)
ballistic_add_test(test_meo_gps_validation validation/test_meo_gps_validation.cpp)
ballistic_add_test(test_geo_validation validation/test_geo_validation.cpp)
ballistic_add_test(test_geodetic_external validation/test_geodetic_external.cpp)
ballistic_add_test(test_erfa_reference validation/test_erfa_reference.cpp)
ballistic_add_test(test_heo_validation validation/test_heo_validation.cpp)
ballistic_add_test(test_trajectory_stepping validation/test_trajectory_stepping.cpp)
ballistic_add_test(test_qzss_heo_validation validation/test_qzss_heo_validation.cpp)
ballistic_add_test(test_time_references validation/test_time_references.cpp)

# Diagnostics and benchmarks (research instruments, not part of the test gate) live in
# tests/diagnostics with their own CMakeLists; the directory is absent from the public tree.
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/diagnostics")
    add_subdirectory(diagnostics)
endif()

# Custom target to run all unit tests
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    COMMENT "Running all unit tests..."
)

# =========================================================================
# Strict-mode subset: CI runs `ctest -L unit -LE warns` with
# BALLISTIC_STRICT_WARNINGS=1, so a new unit test must not trigger a library
# warning unless it is listed here. These do on purpose: they exercise the
# warnings themselves (staleness, silent substitutions, EOP/leap-second edges,
# forecast weather) or propagate with default settings without the ocean-tide
# file, which C++ tests leave unloaded to stay hermetic (one ModelSubstitution
# note per process).
# =========================================================================
foreach(_warns IN ITEMS
        test_epoch test_staleness_warnings test_events test_sgp4
        test_iau_transforms test_embedded_ephemeris test_live_drag_env
        test_trajectory_drag_env test_estimator_drag_env test_space_weather
        test_symplectic test_silent_substitutions test_trajectory test_force_model
        test_beta_ea_roundtrip test_force_breakdown test_eof_parser test_eop_data)
    if(TEST ${_warns})   # nrl tests are absent without the plugin
        set_property(TEST ${_warns} APPEND PROPERTY LABELS warns)
    endif()
endforeach()

# =========================================================================
# Enforce test_utils.hpp usage in test directories
# =========================================================================
file(GLOB_RECURSE _UNIT_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp")
file(GLOB_RECURSE _VALIDATION_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/validation/*.cpp")
file(GLOB_RECURSE _INTEGRATION_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/*.cpp")

foreach(_test_file ${_UNIT_TESTS} ${_VALIDATION_TESTS} ${_INTEGRATION_TESTS})
    file(READ "${_test_file}" _contents)
    string(FIND "${_contents}" "test_utils.hpp" _found)
    if(_found EQUAL -1)
        message(FATAL_ERROR
            "Test file ${_test_file} does not include test_utils.hpp.\n"
            "All tests in unit/, validation/, and integration/ must use the shared test framework.\n"
            "If this is a diagnostic tool, move it to diagnostics/.")
    endif()
endforeach()

# Thread safety: concurrent first use of the lazily loaded gravity/ephemeris
# tables, and concurrent propagations/estimators with shared drag and weather.
ballistic_add_test(test_concurrent_first_touch unit/test_concurrent_first_touch.cpp)
ballistic_add_test(test_concurrency unit/test_concurrency.cpp nrl)
# Velocity cross-validation end to end (~10 min): the same source with its
# heavy cell compiled in, run alone.
ballistic_add_test(test_ea_cv_heavy integration/test_ea_cv.cpp heavy)
target_compile_definitions(test_ea_cv_heavy PRIVATE BALLISTIC_EA_CV_HEAVY=1)
set_tests_properties(test_ea_cv_heavy PROPERTIES TIMEOUT 2400 PROCESSORS 8
                     ENVIRONMENT "TEST_FILTER=cv_velocity_end_to_end")

# State transition matrix and covariance along a propagated Trajectory
# (state_transition.hpp): Phi against central differences, determinant,
# symplectic structure, composition, covariance growth, errors.
ballistic_add_test(test_state_transition unit/test_state_transition.cpp)

# GNSS block radiation table (gnss_blocks.hpp): every IGS block name resolves,
# the prefix rule, the fit table, positive areas.
ballistic_add_test(test_gnss_blocks unit/test_gnss_blocks.cpp)
