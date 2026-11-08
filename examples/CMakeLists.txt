# Examples CMakeLists.txt

add_executable(orbit_propagation
    orbit_propagation.cpp
)

target_link_libraries(orbit_propagation
    PRIVATE
        ballistic
)

add_executable(sentinel_prediction
    sentinel_prediction.cpp
)

target_link_libraries(sentinel_prediction
    PRIVATE
        ballistic
)
