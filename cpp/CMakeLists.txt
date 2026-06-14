find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(pybind11)

set(ICECORE_NATIVE_SOURCES
    src/kalman_filter.cpp
    src/fft_utils.cpp
    src/signal_utils.cpp
    bindings/pybind_module.cpp
)

set(ICECORE_NATIVE_HEADERS
    include/icecore_ms/kalman_filter.hpp
    include/icecore_ms/fft_utils.hpp
    include/icecore_ms/signal_utils.hpp
)

pybind11_add_module(_icecore_native
    ${ICECORE_NATIVE_SOURCES}
    ${ICECORE_NATIVE_HEADERS}
)

target_include_directories(_icecore_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(_icecore_native PRIVATE /W4 /O2)
    target_compile_definitions(_icecore_native PRIVATE _USE_MATH_DEFINES)
else()
    target_compile_options(_icecore_native PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native)
endif()

find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    target_link_libraries(_icecore_native PRIVATE Eigen3::Eigen)
    target_compile_definitions(_icecore_native PRIVATE ICECORE_MS_USE_EIGEN)
    message(STATUS "Using Eigen3 for linear algebra")
else()
    message(STATUS "Eigen3 not found, using built-in linear algebra")
endif()

install(TARGETS _icecore_native
    LIBRARY DESTINATION ${Python_SITEARCH}/icecore_ms
    RUNTIME DESTINATION ${Python_SITEARCH}/icecore_ms
)
