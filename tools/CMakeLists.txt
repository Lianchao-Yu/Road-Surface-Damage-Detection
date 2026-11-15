add_executable(preprocess_case
    preprocess_case.cpp
)

target_link_libraries(preprocess_case PRIVATE rd_core)
rd_set_warnings(preprocess_case)
