cmake_minimum_required(VERSION 3.13)

# MUST be named "appmodules" — React Native 0.84 loads libappmodules.so at startup
project(appmodules)

# Include the standard React Native new architecture setup
# This registers all codegen-generated TurboModules and Fabric components
include(${REACT_ANDROID_DIR}/cmake-utils/ReactNative-application.cmake)

# ---------------------------------------------------------------------------
# llama.cpp inference engine (custom addition)
# ---------------------------------------------------------------------------

# ABI-specific config
if(DEFINED ANDROID_ABI)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(GGML_CPU_KLEIDIAI ON)
        set(GGML_OPENMP ON)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    endif()
endif()

# Path to llama.cpp submodule (from android/app/src/main/jni/ -> project root)
set(LLAMA_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../engine/llama.cpp")
add_subdirectory(${LLAMA_SRC} build-llama)

# ---------------------------------------------------------------------------
# TurboQuant: KV cache compression library (Milestone B)
# ---------------------------------------------------------------------------
set(TURBOQUANT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../engine/turboquant")
add_subdirectory(${TURBOQUANT_SRC} build-turboquant)

# ---------------------------------------------------------------------------
# JNI shared library for our inference engine
# ---------------------------------------------------------------------------
add_library(remotellm SHARED
    ../cpp/llama_jni.cpp
)

target_include_directories(remotellm PRIVATE
    ${LLAMA_SRC}
    ${LLAMA_SRC}/common
    ${LLAMA_SRC}/include
    ${LLAMA_SRC}/ggml/include
    ${LLAMA_SRC}/ggml/src
    ${TURBOQUANT_SRC}/include
)

target_link_libraries(remotellm
    llama
    common
    turboquant
    android
    log
)
