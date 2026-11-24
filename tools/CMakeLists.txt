add_executable(preprocess_case
    preprocess_case.cpp
)

target_link_libraries(preprocess_case PRIVATE rd_core)
rd_set_warnings(preprocess_case)

if(TARGET rd_io)
    add_executable(video_probe
        video_probe.cpp
    )
    target_link_libraries(video_probe PRIVATE rd_io)
    rd_set_warnings(video_probe)
    message(STATUS "video_probe: enabled")
endif()

find_package(OpenCV QUIET)

if(OpenCV_FOUND AND TARGET rd_trt AND TARGET rd_cv)
    add_executable(yolo_infer
        yolo_infer.cpp
    )

    target_include_directories(yolo_infer PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(yolo_infer PRIVATE
        rd_core
        rd_cv
        rd_cuda
        rd_trt
        ${OpenCV_LIBS}
    )
    rd_set_warnings(yolo_infer)
    message(STATUS "yolo_infer: enabled")
else()
    message(STATUS "yolo_infer: skipped (OpenCV or TensorRT unavailable)")
endif()
