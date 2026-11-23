add_executable(preprocess_case
    preprocess_case.cpp
)

target_link_libraries(preprocess_case PRIVATE rd_core)
rd_set_warnings(preprocess_case)

find_package(OpenCV QUIET)

if(OpenCV_FOUND AND TARGET rd_trt)
    add_executable(yolo_infer
        yolo_infer.cpp
    )

    target_include_directories(yolo_infer PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(yolo_infer PRIVATE
        rd_core
        rd_trt
        ${OpenCV_LIBS}
    )
    rd_set_warnings(yolo_infer)
    message(STATUS "yolo_infer: enabled")
else()
    message(STATUS "yolo_infer: skipped (OpenCV or TensorRT unavailable)")
endif()
