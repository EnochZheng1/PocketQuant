cmake_minimum_required(VERSION 3.13)
project(turboquant LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# TurboQuant: KV cache compression via Hadamard rotation + 1-bit quantization
# ---------------------------------------------------------------------------

set(TURBOQUANT_SOURCES
    src/hadamard.cpp
    src/hadamard_neon.cpp
    src/quantize.cpp
    src/qjl.cpp
    src/turbo_cache.cpp
)

add_library(turboquant STATIC ${TURBOQUANT_SOURCES})

target_include_directories(turboquant PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(turboquant PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Enable NEON on ARM64 (always available on arm64-v8a)
if(ANDROID_ABI STREQUAL "arm64-v8a" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_compile_definitions(turboquant PUBLIC TQ_HAVE_NEON=1)
endif()

# ---------------------------------------------------------------------------
# Tests (only when building standalone, not as part of the Android app)
# ---------------------------------------------------------------------------
option(TURBOQUANT_BUILD_TESTS "Build turboquant unit tests" OFF)

if(TURBOQUANT_BUILD_TESTS)
    enable_testing()

    foreach(test_name test_hadamard test_quantize test_qjl test_pipeline)
        add_executable(${test_name} tests/${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE turboquant)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()
