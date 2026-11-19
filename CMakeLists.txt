cmake_minimum_required(VERSION 3.24)

project(road_damage VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RD_ENABLE_CUDA "Build CUDA components" ON)
option(RD_ENABLE_TENSORRT "Build TensorRT inference components" ON)
option(RD_BUILD_TESTS "Build unit tests" ON)
option(RD_BUILD_BENCHMARKS "Build benchmarks" ON)
option(RD_BUILD_TOOLS "Build command-line tools" ON)
option(RD_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

if(RD_ENABLE_TENSORRT AND NOT RD_ENABLE_CUDA)
    message(FATAL_ERROR "RD_ENABLE_TENSORRT requires RD_ENABLE_CUDA=ON")
endif()

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
if(RD_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures" FORCE)
    endif()
    if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda-13.5/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.5/bin/nvcc" CACHE FILEPATH "CUDA compiler")
    endif()
    if(NOT DEFINED CUDAToolkit_ROOT AND EXISTS "/usr/local/cuda-13.5")
        set(CUDAToolkit_ROOT "/usr/local/cuda-13.5" CACHE PATH "CUDA toolkit root")
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    find_package(CUDAToolkit REQUIRED)

    # Ubuntu ships CUDA 12.4 headers in /usr/include while the build uses 13.5.
    # Put the 13.5 include directory first so <cuda_runtime_api.h> matches libcudart.
    include_directories(BEFORE "${CUDAToolkit_INCLUDE_DIRS}")
endif()

# ---------------------------------------------------------------------------
# Warnings helper
# ---------------------------------------------------------------------------
function(rd_set_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
    if(RD_WARNINGS_AS_ERRORS)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Werror>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
        )
    endif()
endfunction()

# ---------------------------------------------------------------------------
# rd_core: types, configuration, CPU reference code (no CUDA / TensorRT)
# ---------------------------------------------------------------------------
add_library(rd_core
    src/showinfo/version.cpp
    src/cpu/preprocess_cpu.cpp
)
target_include_directories(rd_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(rd_core PUBLIC cxx_std_17)
target_compile_definitions(rd_core PUBLIC RD_VERSION="${PROJECT_VERSION}")
rd_set_warnings(rd_core)

# ---------------------------------------------------------------------------
# rd_cuda: CUDA kernels and GPU runtime helpers
# ---------------------------------------------------------------------------
if(RD_ENABLE_CUDA)
    add_library(rd_cuda
        src/showinfo/cuda_info.cu
    )
    target_include_directories(rd_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(rd_cuda PUBLIC CUDA::cudart)
    target_compile_definitions(rd_cuda PUBLIC RD_WITH_CUDA=1)
    rd_set_warnings(rd_cuda)
endif()

# ---------------------------------------------------------------------------
# rd_trt: TensorRT integration
# ---------------------------------------------------------------------------
if(RD_ENABLE_TENSORRT)
    find_package(TensorRT REQUIRED)
    add_library(rd_trt
        src/showinfo/trt_info.cpp
        src/inference/tensorrt_backend.cpp
    )
    target_include_directories(rd_trt PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(rd_trt PUBLIC
        TensorRT::nvinfer
        TensorRT::nvonnxparser
        CUDA::cudart
    )
    target_compile_definitions(rd_trt PUBLIC RD_WITH_TENSORRT=1)
    rd_set_warnings(rd_trt)
endif()

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
add_executable(road_damage src/app/main.cpp)
target_link_libraries(road_damage PRIVATE rd_core)
if(RD_ENABLE_CUDA)
    target_link_libraries(road_damage PRIVATE rd_cuda)
endif()
if(RD_ENABLE_TENSORRT)
    target_link_libraries(road_damage PRIVATE rd_trt)
endif()
rd_set_warnings(road_damage)

# ---------------------------------------------------------------------------
# Tests / benchmarks
# ---------------------------------------------------------------------------
if(RD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(RD_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(RD_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

message(STATUS "road_damage ${PROJECT_VERSION}")
message(STATUS "  CUDA      : ${RD_ENABLE_CUDA}")
message(STATUS "  TensorRT  : ${RD_ENABLE_TENSORRT}")
message(STATUS "  Tests     : ${RD_BUILD_TESTS}")
message(STATUS "  Benchmarks: ${RD_BUILD_BENCHMARKS}")
message(STATUS "  Tools     : ${RD_BUILD_TOOLS}")
