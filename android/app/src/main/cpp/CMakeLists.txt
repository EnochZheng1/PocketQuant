cmake_minimum_required(VERSION 3.22.1)

project("remotellm" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# --------------------------------------------------------------------------
# ABI-specific configuration (matching llama.cpp Android example)
# --------------------------------------------------------------------------
if(DEFINED ANDROID_ABI)
    message(STATUS "Detected Android ABI: ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(GGML_CPU_KLEIDIAI ON)
        set(GGML_OPENMP ON)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    else()
        message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
    endif()
endif()

# Path to llama.cpp submodule
# From android/app/src/main/cpp/ -> 5 levels up -> engine/llama.cpp
set(LLAMA_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../engine/llama.cpp")
add_subdirectory(${LLAMA_SRC} build-llama)

# --------------------------------------------------------------------------
# JNI shared library
# --------------------------------------------------------------------------
add_library(${CMAKE_PROJECT_NAME} SHARED
    llama_jni.cpp
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${LLAMA_SRC}
    ${LLAMA_SRC}/common
    ${LLAMA_SRC}/include
    ${LLAMA_SRC}/ggml/include
    ${LLAMA_SRC}/ggml/src
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    llama
    common
    android
    log
)
