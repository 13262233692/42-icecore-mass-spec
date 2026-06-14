cmake_minimum_required(VERSION 3.24)
project(icecore_ms
    VERSION 0.1.0
    DESCRIPTION "Ice Core Isotope Abundance Tomography Kernel"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

option(ICECORE_MS_BUILD_TESTS "Build C++ unit tests" OFF)

set(ICECORE_MS_CPP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp)

add_subdirectory(cpp)

if(ICECORE_MS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
