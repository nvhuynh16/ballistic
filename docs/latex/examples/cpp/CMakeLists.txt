# Tutorial example programs + the glide fixture generator.
# Opt-in via -DBALLISTIC_BUILD_TUTORIAL_EXAMPLES=ON (default OFF) so normal
# and CI builds are unaffected. Run them via docs/latex/regen.sh.

# Glide fixture generator (support tool; mirrors tests/test_helpers.hpp glide).
add_executable(gen_glide_fixture fixture/gen_glide_fixture.cpp)
target_link_libraries(gen_glide_fixture PRIVATE ballistic)

# Parity check: runs the real make_glide() and prints IC/eval_end for diffing
# against the generator (dev tool — needs the tests/ include for test_helpers).
add_executable(_check_glide_parity fixture/_check_glide_parity.cpp)
target_link_libraries(_check_glide_parity PRIVATE ballistic)
target_include_directories(_check_glide_parity PRIVATE ${CMAKE_SOURCE_DIR}/tests)

# Tutorial chapter example programs (compiled, run, listed verbatim in PDFs).
foreach(ex
    ex01_setup
    ex02_timescales
    ex03_sentinel_propagation
    ex06_ballistic_recovery
    ex07_multisensor
    ex09_pitfalls
    ex04_ea_clean_glide
    ex05_ea_noisy_glide
)
    add_executable(${ex} ${ex}.cpp)
    target_link_libraries(${ex} PRIVATE ballistic)
    # glide_common.hpp (this dir) + generated glide_fixture_data.hpp (../fixtures)
    target_include_directories(${ex} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../fixtures)
endforeach()
