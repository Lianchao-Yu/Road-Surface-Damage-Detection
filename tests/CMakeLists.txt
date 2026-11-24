find_package(GTest REQUIRED)
find_package(OpenCV QUIET)

add_executable(rd_tests
    test_environment.cpp
    test_preprocess_cpu.cpp
    test_letterbox.cpp
)

if(OpenCV_FOUND)
    target_sources(rd_tests PRIVATE test_preprocess_vs_opencv.cpp)
    target_include_directories(rd_tests PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(rd_tests PRIVATE ${OpenCV_LIBS})
    if(TARGET rd_cv)
        target_sources(rd_tests PRIVATE test_preprocess_cv.cpp)
        target_link_libraries(rd_tests PRIVATE rd_cv)
    endif()
    message(STATUS "rd_tests: OpenCV cross-check enabled (${OpenCV_VERSION})")
else()
    message(STATUS "rd_tests: OpenCV not found, cross-check test skipped")
endif()

target_compile_definitions(rd_tests PRIVATE
    RD_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

target_link_libraries(rd_tests PRIVATE
    rd_core
    GTest::gtest
    GTest::gtest_main
)

if(TARGET rd_cuda)
    target_sources(rd_tests PRIVATE test_preprocess_gpu.cpp)
    target_link_libraries(rd_tests PRIVATE rd_cuda)
endif()

if(TARGET rd_trt)
    target_sources(rd_tests PRIVATE test_tensorrt_backend.cpp test_postprocess_cpu.cpp)
    target_link_libraries(rd_tests PRIVATE rd_trt)
endif()

if(TARGET rd_io)
    target_sources(rd_tests PRIVATE test_video_source.cpp)
    target_link_libraries(rd_tests PRIVATE rd_io)
endif()

rd_set_warnings(rd_tests)

include(GoogleTest)
gtest_discover_tests(rd_tests)
