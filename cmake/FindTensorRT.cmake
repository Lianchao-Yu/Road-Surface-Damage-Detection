# FindTensorRT.cmake
#
# Locates the TensorRT headers and libraries and defines:
#   TensorRT::nvinfer
#   TensorRT::nvonnxparser
#
# Hints (in order): TENSORRT_ROOT, the TENSORRT_ROOT environment variable,
# then the standard system locations. TENSORRT_ROOT may point either to the
# TensorRT prefix or to the extracted package root (e.g. ~/opt/tensorrt-11.4/root/usr).

set(_trt_hints "")
if(DEFINED TENSORRT_ROOT AND NOT "${TENSORRT_ROOT}" STREQUAL "")
    list(APPEND _trt_hints
        "${TENSORRT_ROOT}"
        "${TENSORRT_ROOT}/include"
        "${TENSORRT_ROOT}/include/x86_64-linux-gnu"
    )
endif()
if(DEFINED ENV{TENSORRT_ROOT} AND NOT "$ENV{TENSORRT_ROOT}" STREQUAL "")
    list(APPEND _trt_hints
        "$ENV{TENSORRT_ROOT}"
        "$ENV{TENSORRT_ROOT}/include"
        "$ENV{TENSORRT_ROOT}/include/x86_64-linux-gnu"
    )
endif()

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${_trt_hints}
    PATHS
        /usr/include
        /usr/local/include
        /usr/include/x86_64-linux-gnu
        /usr/local/include/x86_64-linux-gnu
    DOC "TensorRT include directory (NvInfer.h)"
)

set(_trt_lib_hints "")
if(DEFINED TENSORRT_ROOT AND NOT "${TENSORRT_ROOT}" STREQUAL "")
    list(APPEND _trt_lib_hints
        "${TENSORRT_ROOT}"
        "${TENSORRT_ROOT}/lib"
        "${TENSORRT_ROOT}/lib/x86_64-linux-gnu"
    )
endif()
if(DEFINED ENV{TENSORRT_ROOT} AND NOT "$ENV{TENSORRT_ROOT}" STREQUAL "")
    list(APPEND _trt_lib_hints
        "$ENV{TENSORRT_ROOT}"
        "$ENV{TENSORRT_ROOT}/lib"
        "$ENV{TENSORRT_ROOT}/lib/x86_64-linux-gnu"
    )
endif()

find_library(TensorRT_NVINFER_LIBRARY
    NAMES nvinfer
    HINTS ${_trt_lib_hints}
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/local/lib/x86_64-linux-gnu
    DOC "TensorRT nvinfer library"
)

find_library(TensorRT_NVONNXPARSER_LIBRARY
    NAMES nvonnxparser
    HINTS ${_trt_lib_hints}
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/local/lib/x86_64-linux-gnu
    DOC "TensorRT nvonnxparser library"
)

# Version from NvInferVersion.h (TensorRT 11 uses TRT_*_ENTERPRISE macros).
set(TensorRT_VERSION "")
if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    foreach(_part MAJOR MINOR PATCH)
        file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _line
            REGEX "^#define TRT_${_part}_ENTERPRISE [0-9]+")
        if(_line MATCHES "([0-9]+)")
            set(_trt_${_part} "${CMAKE_MATCH_1}")
        else()
            set(_trt_${_part} "0")
        endif()
    endforeach()
    set(TensorRT_VERSION "${_trt_MAJOR}.${_trt_MINOR}.${_trt_PATCH}")
    unset(_line)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS
        TensorRT_INCLUDE_DIR
        TensorRT_NVINFER_LIBRARY
        TensorRT_NVONNXPARSER_LIBRARY
    VERSION_VAR TensorRT_VERSION
)

if(TensorRT_FOUND)
    if(NOT TARGET TensorRT::nvinfer)
        add_library(TensorRT::nvinfer SHARED IMPORTED)
        set_target_properties(TensorRT::nvinfer PROPERTIES
            IMPORTED_LOCATION "${TensorRT_NVINFER_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    endif()
    if(NOT TARGET TensorRT::nvonnxparser)
        add_library(TensorRT::nvonnxparser SHARED IMPORTED)
        set_target_properties(TensorRT::nvonnxparser PROPERTIES
            IMPORTED_LOCATION "${TensorRT_NVONNXPARSER_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES TensorRT::nvinfer
        )
    endif()
endif()

mark_as_advanced(
    TensorRT_INCLUDE_DIR
    TensorRT_NVINFER_LIBRARY
    TensorRT_NVONNXPARSER_LIBRARY
)
